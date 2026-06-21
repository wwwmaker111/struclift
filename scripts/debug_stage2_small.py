#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小样本 Stage2 诊断：默认只读 3 条 jsonl，组 1 个 batch，在同一 batch 上重复若干步
forward + backward + optimizer，打印 loss 各分量、梯度范数、alignment 权重 L2 是否变化。

用于判断「无梯度 / 梯度极小 / 单 batch 能否过拟合」而无需全量训练。

示例（Linux / 服务器）::

  python scripts/debug_stage2_small.py \\
    --jsonl /path/to/train.jsonl \\
    --init-from /path/to/best_stage1_snapshot.pt \\
    --device cuda

Windows 请把路径改成你的 jsonl / ckpt；仅用 CPU 时 ``--device cpu``。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import make_binskel_stage2_dataloader, move_binskel_batch_to_device
from struclift.models.struclift import StrucLift
from struclift.training.stage2_alignment import Stage2Trainer, _module_param_l2_norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("debug_stage2_small")


def _encode(trainer: Stage2Trainer, batch: dict) -> object:
    m = trainer.model
    return m.encode(
        bin_opcodes=batch["bin_opcodes"],
        bin_operand_types=batch["bin_operand_types"],
        bin_operand_values=batch["bin_operand_values"],
        bin_block_lengths=batch["bin_block_lengths"],
        bin_edge_index=batch["bin_edge_index"],
        bin_edge_type=batch["bin_edge_type"],
        bin_struct_features=batch["bin_struct_features"],
        bin_batch=batch.get("bin_batch"),
        src_token_ids=batch.get("src_token_ids"),
        src_stmt_lengths=batch.get("src_stmt_lengths"),
        src_edge_index=batch.get("src_edge_index"),
        src_edge_type=batch.get("src_edge_type"),
        src_ast_types=batch.get("src_ast_types"),
        src_compound_id=batch.get("src_compound_id"),
        src_batch=batch.get("src_batch"),
        bin_adj=batch.get("adj_bin"),
        src_adj=batch.get("adj_src_gt"),
        M_gt=batch.get("M_gt"),
        run_alignment=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage2 小样本梯度/过拟合诊断（默认 3 条）")
    ap.add_argument("--jsonl", type=Path, required=True, help="含 alignment 的 binskel jsonl")
    ap.add_argument("--init-from", type=Path, default=None, help="Stage1 等 checkpoint（可选）")
    ap.add_argument("--max-samples", type=int, default=3, help="最多有效条数（默认 3）")
    ap.add_argument("--batch-size", type=int, default=None, help="默认与 max-samples 相同")
    ap.add_argument("--steps", type=int, default=3, help="同一 batch 上优化步数")
    ap.add_argument("--epoch", type=int, default=1, help="模拟第几 epoch（影响 node_hn 开关与 warmup）")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--no-fp16", action="store_true", help="Stage2 用 fp32（默认配置已是 fp32）")
    ap.add_argument("--fp16", action="store_true", help="强制 Stage2 fp16+GradScaler（复现 nan 对照用）")
    ap.add_argument(
        "--lambda-src-count",
        type=float,
        default=None,
        dest="src_count_lambda",
        help="覆盖 s2_lambda_src_count；0 可关掉 n_s 回归项的加权（对照 L_src 是否导致 nan）",
    )
    ap.add_argument("--strict-jsonl", action="store_true", help="坏行即失败")
    args = ap.parse_args()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    bs = args.batch_size if args.batch_size is not None else max(1, int(args.max_samples))
    skip_invalid = not args.strict_jsonl

    loader = make_binskel_stage2_dataloader(
        args.jsonl.expanduser().resolve(),
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        max_samples=int(args.max_samples),
        skip_invalid=skip_invalid,
        pin_memory=device.type == "cuda",
        prefetch_factor=2,
    )
    n_ds = len(loader.dataset)
    if n_ds == 0:
        logger.error("数据集为空：请检查 jsonl 与 --max-samples")
        sys.exit(1)

    cfg = StrucLiftConfig()
    if args.fp16 and args.no_fp16:
        logger.error("--fp16 与 --no-fp16 不能同时使用")
        sys.exit(1)
    if args.no_fp16:
        cfg.training.s2_fp16 = False
    elif args.fp16:
        cfg.training.s2_fp16 = True
    if args.src_count_lambda is not None:
        cfg.training.s2_lambda_src_count = float(args.src_count_lambda)

    model = StrucLift(cfg)
    if args.init_from is not None:
        ckpt_path = args.init_from.expanduser().resolve()
        if not ckpt_path.is_file():
            logger.error("checkpoint 不存在: %s", ckpt_path)
            sys.exit(1)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        miss, unexp = model.load_state_dict(state, strict=False)
        logger.info("load_state_dict missing=%d unexpected=%d", len(miss), len(unexp))

    trainer = Stage2Trainer(
        model,
        cfg,
        device=device,
        curriculum_dataset=None,
        stage2_loader_kwargs={"batch_size": bs, "num_workers": 0, "skip_invalid": skip_invalid},
    )

    batch = next(iter(loader))
    batch = move_binskel_batch_to_device(batch, device)

    model.train()
    model.binary_encoder.eval()
    model.source_encoder.eval()

    hn_on = trainer._node_hn_active(int(args.epoch))
    p0 = _module_param_l2_norm(model.alignment_module)
    logger.info(
        "数据集条数=%d batch 内图数≈根据 collate；模拟 epoch=%d node_hn=%s； alignment L2(步前)=%.6f",
        n_ds,
        int(args.epoch),
        "on" if hn_on else "off",
        p0,
    )

    last_loss: float | None = None
    for step in range(int(args.steps)):
        trainer.optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=trainer.fp16 and device.type == "cuda"):
            enc_out = _encode(trainer, batch)
        with torch.amp.autocast("cuda", enabled=trainer.fp16 and device.type == "cuda"):
            loss, comp = trainer._loss_tensors_from_enc_out(
                enc_out, batch, hn_on=hn_on, epoch=int(args.epoch)
            )

        if not loss.requires_grad:
            logger.error("step=%d loss 无 grad_fn，反传无效", step)
            sys.exit(2)

        vals = {k: float(v.detach().float().item()) for k, v in comp.items()}
        total_f = float(loss.detach().float().item())

        if trainer.scaler is not None:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                model.alignment_module.parameters(),
                max_norm=float("inf"),
            )
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.alignment_module.parameters(),
                max_norm=float("inf"),
            )
            trainer.optimizer.step()

        n_none = sum(
            1
            for p in model.alignment_module.parameters()
            if p.grad is None
        )
        p_after = _module_param_l2_norm(model.alignment_module)

        logger.info(
            "step=%d total=%.6f |grad|_2=%.6e align_L2=%.6f n_params_grad_none=%d",
            step,
            total_f,
            float(gn),
            p_after,
            n_none,
        )
        logger.info(
            "  scot=%.4f edge=%.4f ca=%.4f region=%.4f src_count=%.4f node_hn=%.4f",
            vals["scot"],
            vals["edge"],
            vals["ca_align"],
            vals["region"],
            vals["src_count"],
            vals["node_hn"],
        )

        if not (total_f == total_f and total_f < 1e30):
            logger.error("step=%d loss 非有限", step)
            sys.exit(3)
        last_loss = total_f

    p1 = _module_param_l2_norm(model.alignment_module)
    delta = p1 - p0
    logger.info("alignment L2 变化: %.6e (若 ~0 则多步未移动参数)", delta)
    if last_loss is not None and args.steps >= 2:
        logger.info(
            "提示: 若 total 连降或 |grad| 非零，说明小样本上可优化；全量训练不动更偏 lr/fp16/任务难度。"
        )


if __name__ == "__main__":
    main()

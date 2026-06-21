#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 / Module A 快速自检（可用已有 best_stage1.pt）：

1. 取 **固定 1 个 batch**，做前向 + 反传，检查 loss / 梯度是否 NaN/Inf
2. 可选：在同一 batch 上重复优化 **多步**，看 total loss 是否明显下降（过拟合探针）

不能证明「无 bug」，但能快速发现「算子炸了、形状错了、梯度断了」等明显问题。

示例（WSL）::

  cd /mnt/e/structlift && source .venv/bin/activate
  python3 scripts/sanity_check_stage1.py \\
    --jsonl /mnt/e/structlift_datasets/binskel_zlib_o0.jsonl \\
    --init-from /mnt/e/structlift/checkpoints/stage1_zlib_5opt_mixed/best_stage1.pt \\
    --max-samples 8 --batch-size 2 --overfit-steps 80
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
import torch.nn as nn

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    describe_binskel_batch,
    make_binskel_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.training.stage1_pretrain import Stage1Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _grad_nan_params(model: nn.Module) -> list[str]:
    bad: list[str] = []
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(name)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage1 单 batch 形状/梯度/过拟合自检")
    ap.add_argument("--jsonl", type=Path, required=True, help="binskel jsonl（须含源码以测 graph_align）")
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="可选：载入 best_stage1.pt / final_stage1.pt（strict=False）",
    )
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=8, help="数据集最多条数，保证能组 batch")
    ap.add_argument("--overfit-steps", type=int, default=0, help=">0 时对同一 batch 重复训练这么多步")
    ap.add_argument(
        "--no-fp16",
        action="store_true",
        help="强制关闭 fp16（CPU 或查梯度时更稳）",
    )
    args = ap.parse_args()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    cfg = StrucLiftConfig()
    if args.no_fp16:
        cfg.training.fp16 = False

    loader = make_binskel_dataloader(
        args.jsonl.expanduser().resolve(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        max_samples=args.max_samples,
        require_source=True,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
    )
    if len(loader.dataset) == 0:
        logger.error("数据集无有效样本")
        sys.exit(1)

    batch = next(iter(loader))
    logger.info("固定 batch 形状:\n%s", describe_binskel_batch(batch))
    batch = move_binskel_batch_to_device(batch, device)

    model = StrucLift(cfg).to(device)
    if args.init_from is not None:
        ckpt = args.init_from.expanduser().resolve()
        if not ckpt.is_file():
            logger.error("checkpoint 不存在: %s", ckpt)
            sys.exit(1)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.info("未载入键(示例): %s", missing[:3])
        if unexpected:
            logger.info("checkpoint 多余键(示例): %s", unexpected[:3])

    trainer = Stage1Trainer(model, cfg, device=device)
    tc = cfg.training
    model.train()

    def forward_loss() -> tuple[torch.Tensor, dict[str, float]]:
        with torch.amp.autocast("cuda", enabled=tc.fp16 and device.type == "cuda"):
            enc_out = model.encode(
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
                run_alignment=False,
            )

            loss_pattern = trainer.pattern_loss_fn(
                enc_out.pattern_logits,
                batch["pattern_labels"],
                weights=batch.get("pattern_weights"),
            )
            h_norm = nn.functional.normalize(enc_out.binary_node_embs, dim=-1)
            loss_neighbor = trainer.neighbor_loss_fn(
                h_norm,
                batch["bin_edge_index"],
                batch.get("bin_batch"),
            )
            loss = loss_pattern + trainer.lambda_recon * loss_neighbor
            parts: dict[str, float] = {
                "pattern": float(loss_pattern.detach().item()),
                "neighbor": float(loss_neighbor.detach().item()),
                "graph_align": 0.0,
            }
            if (
                trainer.lambda_graph_align > 0
                and enc_out.source_graph_emb is not None
                and enc_out.source_graph_emb.numel() > 0
            ):
                loss_ga = trainer.graph_align_loss_fn(
                    enc_out.binary_graph_emb,
                    enc_out.source_graph_emb,
                )
                loss = loss + trainer.lambda_graph_align * loss_ga
                parts["graph_align"] = float(loss_ga.detach().item())
            parts["total"] = float(loss.detach().item())
        return loss, parts

    # --- 第一步：前向 + 一次反传，查 NaN ---
    loss0, parts0 = forward_loss()
    logger.info(
        "首步 loss: total=%.6f pattern=%.6f neighbor=%.6f graph_align=%.6f",
        parts0["total"],
        parts0["pattern"],
        parts0["neighbor"],
        parts0["graph_align"],
    )
    if not torch.isfinite(loss0).all():
        logger.error("loss 非有限值，终止")
        sys.exit(2)

    trainer.optimizer.zero_grad(set_to_none=True)
    if trainer.scaler is not None:
        trainer.scaler.scale(loss0).backward()
        trainer.scaler.unscale_(trainer.optimizer)
    else:
        loss0.backward()

    bad = _grad_nan_params(model)
    if bad:
        logger.error("以下参数梯度含 NaN/Inf（示例最多 10 个）: %s", bad[:10])
        sys.exit(3)
    logger.info("首步反传: 未发现 NaN/Inf 梯度（已检查全部有 grad 的参数）")

    if args.overfit_steps <= 0:
        logger.info("未指定 --overfit-steps，只做单步检查，结束。")
        return

    # --- 过拟合同一 batch：从当前权重继续多步 ---
    losses: list[float] = []
    for step in range(1, args.overfit_steps + 1):
        loss, parts = forward_loss()
        if not torch.isfinite(loss).all():
            logger.error("step %d loss 非有限", step)
            sys.exit(4)
        trainer.optimizer.zero_grad(set_to_none=True)
        if trainer.scaler is not None:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            loss.backward()
            trainer.optimizer.step()
        losses.append(parts["total"])
        if step in (1, 2, 5, 10) or step % max(1, args.overfit_steps // 4) == 0 or step == args.overfit_steps:
            logger.info(
                "overfit step %d/%d total=%.6f pattern=%.6f",
                step,
                args.overfit_steps,
                parts["total"],
                parts["pattern"],
            )

    first, last = losses[0], losses[-1]
    logger.info("过拟合: 首步 total=%.6f 末步 total=%.6f Δ=%.6f", first, last, first - last)
    if last > first * 0.995:
        logger.warning(
            "末步 loss 几乎未降（可能 lr 太小、或该 batch 已极优）。"
            "可加大 --overfit-steps 或暂时提高 config.training.s1_lr 再试。"
        )
    else:
        logger.info("同一 batch 上 loss 明显下降，说明反传与优化链路基本正常。")


if __name__ == "__main__":
    main()

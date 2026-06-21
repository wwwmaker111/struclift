#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 测时：256 条样本、指定 batch size、跑 N 个 epoch，统计
  - 每轮 epoch 墙钟
  - 每步细分占比：DataLoader(next)、host→GPU、encode、各项 loss 前向、bwd+optimizer

实现方式：与 train_epoch 等价的手动循环 + CUDA synchronize 分段，便于判断瓶颈。

用法（服务器；**勿用系统 python3**，须用 wqm_struc 解释器，见 docs/server_wqm_env.md）::

  cd /data/chaoni/WQM/model_code/structlift && \\
  CUDA_VISIBLE_DEVICES=0 /data/chaoni/miniconda3/envs/wqm_struc/bin/python3 scripts/profile_stage2_batchsize_256x3.py \\
    --jsonl-o0 /data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl \\
    --init-from /data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt \\
    --max-samples 256 --batch-sizes 8 16 32 --epochs 3 --num-workers 4 --prefetch-factor 4

或先: ``source /data/chaoni/miniconda3/etc/profile.d/conda.sh && conda activate wqm_struc`` 再 ``python3 ...``。

默认: max_samples=256, batch_size 扫 8/16/32, epochs=3, O0 单档。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import Stage2CurriculumDataset, build_stage2_train_loader
from struclift.models.struclift import StrucLift
from struclift.training.stage2_alignment import Stage2Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("profile_s2_bs")


@dataclass
class EpochTiming:
    wall_s: float = 0.0
    n_batches: int = 0
    sum_data: float = 0.0
    sum_h2d: float = 0.0
    sum_enc: float = 0.0
    sum_loss: float = 0.0
    sum_bwd: float = 0.0
    per_batch: Dict[str, List[float]] = field(
        default_factory=lambda: {k: [] for k in ("data", "h2d", "enc", "loss", "bwd")}
    )

    def pct(self) -> Dict[str, float]:
        s = self.sum_data + self.sum_h2d + self.sum_enc + self.sum_loss + self.sum_bwd
        if s <= 1e-9:
            return {k: 0.0 for k in ("data", "h2d", "enc", "loss", "bwd")}
        return {
            "data": 100.0 * self.sum_data / s,
            "h2d": 100.0 * self.sum_h2d / s,
            "enc": 100.0 * self.sum_enc / s,
            "loss": 100.0 * self.sum_loss / s,
            "bwd": 100.0 * self.sum_bwd / s,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_one_epoch(
    trainer: Stage2Trainer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> EpochTiming:
    """与 Stage2Trainer.train_epoch 前向/反传一致，另计分段耗时。"""
    model = trainer.model
    model.train()
    model.binary_encoder.eval()
    model.source_encoder.eval()
    hn_on = trainer._node_hn_active(epoch)

    out = EpochTiming()
    t_epoch0 = time.perf_counter()
    it = iter(dataloader)

    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        data_dt = t1 - t0

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        _sync(device)
        t2 = time.perf_counter()
        h2d_dt = t2 - t1

        with torch.amp.autocast("cuda", enabled=trainer.fp16):
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
                bin_adj=batch.get("adj_bin"),
                src_adj=batch.get("adj_src_gt"),
                M_gt=batch.get("M_gt"),
                run_alignment=True,
            )
        _sync(device)
        t3 = time.perf_counter()
        enc_dt = t3 - t2

        with torch.amp.autocast("cuda", enabled=trainer.fp16):
            loss, _comp = trainer._loss_tensors_from_enc_out(
                enc_out, batch, hn_on=hn_on, epoch=epoch
            )
        _sync(device)
        t4 = time.perf_counter()
        loss_dt = t4 - t3

        trainer.optimizer.zero_grad()
        if trainer.scaler is not None:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.alignment_module.parameters(),
                max_norm=float("inf"),
            )
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.alignment_module.parameters(),
                max_norm=float("inf"),
            )
            trainer.optimizer.step()
        _sync(device)
        t5 = time.perf_counter()
        bwd_dt = t5 - t4

        out.n_batches += 1
        out.sum_data += data_dt
        out.sum_h2d += h2d_dt
        out.sum_enc += enc_dt
        out.sum_loss += loss_dt
        out.sum_bwd += bwd_dt
        for k, v in (
            ("data", data_dt),
            ("h2d", h2d_dt),
            ("enc", enc_dt),
            ("loss", loss_dt),
            ("bwd", bwd_dt),
        ):
            out.per_batch[k].append(v)

    out.wall_s = time.perf_counter() - t_epoch0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage2：256 样本 × 多 batch size × 多 epoch 分段测时"
    )
    ap.add_argument(
        "--jsonl-o0",
        type=Path,
        default=Path("/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl"),
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=Path("/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt"),
    )
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    o0 = args.jsonl_o0.expanduser().resolve()
    if not o0.is_file():
        logger.error("jsonl 不存在: %s", o0)
        sys.exit(1)
    init_ck = args.init_from.expanduser().resolve()
    if not init_ck.is_file():
        logger.error("init checkpoint 不存在: %s", init_ck)
        sys.exit(1)

    pin = device.type == "cuda"
    all_rows: List[str] = []

    for bs in args.batch_sizes:
        cfg = StrucLiftConfig()
        cfg.training.s2_batch_size = int(bs)

        paths = {"O0": o0}
        ds = Stage2CurriculumDataset(
            paths,
            max_samples=int(args.max_samples),
            skip_invalid=True,
        )
        ds.set_stage(0)
        n = len(ds)
        if n == 0:
            logger.error("数据集为空")
            sys.exit(1)

        loader = build_stage2_train_loader(
            ds,
            batch_size=cfg.training.s2_batch_size,
            shuffle=True,
            num_workers=int(args.num_workers),
            skip_invalid=True,
            pin_memory=pin,
            prefetch_factor=int(args.prefetch_factor),
        )
        n_batches = len(loader)
        logger.info(
            "==== batch_size=%d | 样本数=%d | 每 epoch batch 数=%d ====",
            bs,
            n,
            n_batches,
        )

        model = StrucLift(cfg).to(device)
        try:
            state = torch.load(init_ck, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(init_ck, map_location=device)
        st = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(st, strict=False)
        model.freeze_encoders()

        trainer = Stage2Trainer(
            model,
            cfg,
            device=device,
            curriculum_dataset=None,
            stage2_loader_kwargs={
                "batch_size": cfg.training.s2_batch_size,
                "num_workers": int(args.num_workers),
                "skip_invalid": True,
                "pin_memory": pin,
                "prefetch_factor": int(args.prefetch_factor),
            },
        )

        epoch_walls: List[float] = []
        mean_pct: Dict[str, float] = {k: 0.0 for k in ("data", "h2d", "enc", "loss", "bwd")}
        for ep in range(1, int(args.epochs) + 1):
            t = profile_one_epoch(trainer, loader, device, epoch=ep)
            epoch_walls.append(t.wall_s)
            p = t.pct()
            logger.info(
                "  epoch %d/%d  wall=%.3fs  batches=%d  |  "
                "%%: data=%.1f  h2d=%.1f  enc=%.1f  loss=%.1f  bwd=%.1f",
                ep,
                int(args.epochs),
                t.wall_s,
                t.n_batches,
                p["data"],
                p["h2d"],
                p["enc"],
                p["loss"],
                p["bwd"],
            )
            for k in mean_pct:
                mean_pct[k] += p[k]

        for k in mean_pct:
            mean_pct[k] /= max(int(args.epochs), 1)

        avg_w = sum(epoch_walls) / max(len(epoch_walls), 1)
        logger.info(
            "  >>> batch_size=%d  每 epoch 平均墙钟=%.3fs (3 轮: %s)  分段平均(%%)=%s",
            bs,
            avg_w,
            ", ".join(f"{x:.2f}s" for x in epoch_walls),
            " ".join(f"{k}={mean_pct[k]:.1f}%%" for k in ("data", "h2d", "enc", "loss", "bwd")),
        )
        all_rows.append(
            f"bs={bs}  avg_epoch={avg_w:.3f}s  " + " ".join(f"{k}={mean_pct[k]:.1f}%" for k in mean_pct)
        )

    logger.info("======== 汇总 (max_samples=%d, epochs=%d) ========", args.max_samples, args.epochs)
    for row in all_rows:
        logger.info("%s", row)
    print(
        "\n[说明] data=DataLoader 取 batch+collate 阻塞; h2d=tensor 拷到 GPU; "
        "enc=encode(含冻结 encoder); loss=各对齐损失前向; bwd=反传+clip+step。\n"
        "第 1 个 epoch 常因 cuDNN autotune 略慢，若需可只看 epoch 2–3 平均。"
    )


if __name__ == "__main__":
    main()

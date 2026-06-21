#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 小样本 step 测时：DataLoader 取数 + train_epoch 内首步 GPU 分段（需 struclift.training.stage2_alignment
中 STAGE2_PROFILE_STEP=1 支持）。

用法（服务器上；解释器见 docs/server_wqm_env.md）::

  cd /data/chaoni/WQM/model_code/structlift && \\
  STAGE2_PROFILE_STEP=1 CUDA_VISIBLE_DEVICES=0 \\
  /data/chaoni/miniconda3/envs/wqm_struc/bin/python3 scripts/profile_stage2_step_timing.py \\
    --init-from /data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt

默认使用与正式训练相同的 curriculum 五档路径（顺序 O0→O1→O2→O3→Os；可全部用 --jsonl-o* 覆盖）。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在 import Stage2Trainer 之前设置，首步才会打印 [Stage2][profile]
os.environ.setdefault("STAGE2_PROFILE_STEP", "1")

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
logger = logging.getLogger("profile_stage2")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage2 小样本 step 测时")
    ap.add_argument(
        "--jsonl-o0",
        type=Path,
        default=Path("/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl"),
    )
    ap.add_argument("--jsonl-o1", type=Path, default=None)
    ap.add_argument("--jsonl-o2", type=Path, default=None)
    ap.add_argument("--jsonl-o3", type=Path, default=None)
    ap.add_argument("--jsonl-os", type=Path, default=None)
    ap.add_argument(
        "--init-from",
        type=Path,
        default=Path("/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt"),
    )
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dl-batches",
        type=int,
        default=3,
        help="单独测 DataLoader 连续取几个 batch（仅 wall 时间，不计 GPU）",
    )
    args = ap.parse_args()

    paths: dict[str, Path] = {"O0": args.jsonl_o0.expanduser().resolve()}
    for tag, p in [
        ("O1", args.jsonl_o1),
        ("O2", args.jsonl_o2),
        ("O3", args.jsonl_o3),
        ("Os", args.jsonl_os),
    ]:
        if p is not None:
            paths[tag] = p.expanduser().resolve()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    cfg = StrucLiftConfig()
    cfg.training.s2_batch_size = int(args.batch_size)

    ds = Stage2CurriculumDataset(paths, max_samples=args.max_samples, skip_invalid=True)
    ds.set_stage(0)
    pin = device.type == "cuda"

    def _make_loader():
        return build_stage2_train_loader(
            ds,
            batch_size=cfg.training.s2_batch_size,
            shuffle=True,
            num_workers=int(args.num_workers),
            skip_invalid=True,
            pin_memory=pin,
            prefetch_factor=int(args.prefetch_factor),
        )

    # 1) 仅 DataLoader：连续取若干 batch（与训练无关的独立 iterator）
    logger.info("=== [1/2] 仅 DataLoader：取数 + collate（CPU/wall）===")
    _dl = _make_loader()
    it = iter(_dl)
    dl_times: list[float] = []
    for i in range(max(0, int(args.dl_batches))):
        t0 = time.perf_counter()
        _ = next(it)
        dl_times.append(time.perf_counter() - t0)
    if dl_times:
        logger.info(
            "DataLoader 每 batch wall: %s  avg=%.3fs",
            ", ".join(f"{x:.3f}s" for x in dl_times),
            sum(dl_times) / len(dl_times),
        )

    # 2) 整轮 train_epoch：首步会打 STAGE2_PROFILE_STEP 的 GPU 分段
    logger.info("=== [2/2] train_epoch 1 轮（小样本；首步 GPU 细分见下条 [Stage2][profile]）===")
    loader = _make_loader()
    if len(loader.dataset) == 0:
        logger.error("无有效样本")
        sys.exit(1)

    model = StrucLift(cfg).to(device)
    ck = args.init_from.expanduser().resolve()
    if not ck.is_file():
        logger.error("checkpoint 不存在: %s", ck)
        sys.exit(1)
    try:
        state = torch.load(ck, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ck, map_location=device)
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

    t0 = time.perf_counter()
    m = trainer.train_epoch(loader, epoch=1)
    wall = time.perf_counter() - t0
    logger.info(
        "train_epoch 1 轮结束 wall=%.2fs  loss=%.4f scot=%.4f edge=%.4f",
        wall,
        m["total"],
        m["scot"],
        m["edge"],
    )


if __name__ == "__main__":
    main()

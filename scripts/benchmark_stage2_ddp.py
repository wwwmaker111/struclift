#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 多卡 micro-benchmark（与 ``train_stage2_binskel.py`` 一致的训练语义）。

- 使用 ``torchrun --nproc_per_node=N``；``--batch-size`` 为**每卡** batch（与训练脚本相同）。
- **不使用** ``nn.parallel.DDP`` 包装模型（Stage2 会在一次 encode 里多次进入 alignment）；
  由 ``Stage2Trainer._sync_alignment_gradients`` 在手写 backward 后做 **all_reduce**。
- ``bwd`` 计时包含：``zero_grad`` → ``backward`` → ``_sync_alignment_gradients`` →
  ``clip_grad_norm_(alignment, inf)`` → ``optimizer.step()``（与 ``train_epoch`` 一致）。

默认数据集上限（与单卡 ``benchmark_stage2_cache.py`` 的「每卡样本量」对齐）::

    max_samples = (warmup + measure) * batch_size * world_size

这样每张卡上的 batch 数与单卡 benchmark 相同（例如 bs=64、warmup=1、measure=3 → 每 rank 4 个 batch）。
若希望 jsonl **总条数**与单卡脚本完全一致（不乘 world_size），见 ``--ddp-global-dataset``。

用法::

    CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \\
      scripts/benchmark_stage2_ddp.py \\
      --jsonl /path/to/train.jsonl \\
      --cache-dir /path/to/cache \\
      --init-from /path/to/stage1.pt \\
      --batch-size 64 --n-samples 256 --n-warmup-batches 1 \\
      --num-workers 4 --prefetch-factor 4

单进程冒烟（world_size=1，无 NCCL）::

    python scripts/benchmark_stage2_ddp.py --mode no-cache --jsonl ... --batch-size 64
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import make_binskel_stage2_dataloader
from struclift.models.struclift import StrucLift
from struclift.training.stage2_alignment import Stage2Trainer

try:
    from struclift.models.module_b import (
        report_alignment_profile,
        reset_alignment_profile,
        _PROFILE_ALIGNMENT as _ALIGN_PROFILE_ON,
    )
except ImportError:
    _ALIGN_PROFILE_ON = False

    def report_alignment_profile() -> Dict[str, float]:  # type: ignore[no-redef]
        return {}

    def reset_alignment_profile() -> None:  # type: ignore[no-redef]
        return None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_init_from(model: StrucLift, path: Path, device: torch.device, *, rank: int) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        logger.info("init_from 已载入：missing=%d unexpected=%d", len(missing), len(unexpected))


def benchmark_mode(
    *,
    mode_name: str,
    model: StrucLift,
    trainer: Stage2Trainer,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    n_warmup: int,
    n_measure: int,
) -> List[Dict[str, float]]:
    timings: List[Dict[str, float]] = []
    loader_iter = iter(loader)
    optimizer = trainer.optimizer

    measured = 0
    seen = 0
    while measured < n_measure:
        t0_data = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            if rank == 0:
                logger.warning("[%s] DataLoader 用尽，仅记录 %d 个 measure batch", mode_name, measured)
            break
        t_data = time.perf_counter() - t0_data

        t0 = time.perf_counter()
        batch_d: Dict[str, Any] = {
            k: (v.to(device, non_blocking=device.type == "cuda") if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        _sync(device)
        t_h2d = time.perf_counter() - t0

        if _ALIGN_PROFILE_ON:
            reset_alignment_profile()
        t0 = time.perf_counter()
        enc_out = trainer._forward_stage2_encoder_out(batch_d)
        _sync(device)
        t_enc = time.perf_counter() - t0
        align_breakdown = report_alignment_profile() if _ALIGN_PROFILE_ON else {}

        t0 = time.perf_counter()
        loss, _comp = trainer._loss_tensors_from_enc_out(
            enc_out, batch_d, hn_on=False, epoch=1, log_nonfinite_warnings=False,
        )
        _sync(device)
        t_loss = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer._sync_alignment_gradients()
        torch.nn.utils.clip_grad_norm_(
            trainer._unwrap_model().alignment_module.parameters(),
            max_norm=float("inf"),
        )
        optimizer.step()
        _sync(device)
        t_bwd = time.perf_counter() - t0

        align_str = ""
        if align_breakdown:
            align_str = "  align=[" + " ".join(
                f"{k}:{v:.2f}s" for k, v in align_breakdown.items()
            ) + "]"

        seen += 1
        if seen <= n_warmup:
            if rank == 0:
                logger.info(
                    "[%s][warmup %d/%d] data=%.3fs h2d=%.3fs enc=%.3fs loss=%.3fs bwd=%.3fs total=%.3fs%s",
                    mode_name, seen, n_warmup,
                    t_data, t_h2d, t_enc, t_loss, t_bwd,
                    t_data + t_h2d + t_enc + t_loss + t_bwd,
                    align_str,
                )
            continue

        rec = {
            "data": t_data, "h2d": t_h2d, "encode": t_enc,
            "loss": t_loss, "bwd": t_bwd,
            "total": t_data + t_h2d + t_enc + t_loss + t_bwd,
        }
        timings.append(rec)
        measured += 1
        if rank == 0:
            logger.info(
                "[%s] step %d/%d  data=%.3f  h2d=%.3f  enc=%.3f  loss=%.3f  bwd=%.3f  total=%.3f%s",
                mode_name, measured, n_measure,
                rec["data"], rec["h2d"], rec["encode"], rec["loss"], rec["bwd"], rec["total"],
                align_str,
            )
    return timings


def _summary(timings: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = ["data", "h2d", "encode", "loss", "bwd", "total"]
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = [t[k] for t in timings]
        out[k] = (
            float(statistics.mean(vals)) if vals else 0.0,
            float(statistics.median(vals)) if vals else 0.0,
        )
    return out


def _print_mean_pct(summary: Dict[str, Tuple[float, float]], title: str) -> None:
    keys = ["data", "h2d", "encode", "loss", "bwd"]
    tot = summary["total"][0]
    logger.info("")
    logger.info("======== %s | mean 占比（分母 mean total=%.4fs）========", title, tot)
    if tot < 1e-9:
        logger.info("(total≈0，跳过占比)")
        return
    row_s = " | ".join(f"{k:7s}={summary[k][0]:8.4f}s" for k in keys)
    row_p = " | ".join(
        f"{k:7s}={summary[k][0] / tot * 100.0:6.1f}%%" for k in keys
    )
    logger.info("%s", row_s)
    logger.info("%s", row_p)
    ssum = sum(summary[k][0] for k in keys)
    logger.info(
        "(五项之和=%.4fs, 与 total 差=%.4fs —— 计时缝与浮点误差)",
        ssum,
        abs(ssum - tot),
    )


def _print_compare(s_no: Dict[str, Tuple[float, float]], s_ca: Dict[str, Tuple[float, float]]) -> None:
    keys = ["data", "h2d", "encode", "loss", "bwd", "total"]
    logger.info("")
    logger.info("=" * 90)
    logger.info("%-10s %12s %12s %12s %12s %12s %12s",
                "mode", *keys)
    logger.info("-" * 90)
    logger.info("%-10s " + " ".join(["%12.4f"] * len(keys)),
                "no-cache", *(s_no[k][0] for k in keys))
    logger.info("%-10s " + " ".join(["%12.4f"] * len(keys)),
                "cache", *(s_ca[k][0] for k in keys))
    logger.info("-" * 90)
    deltas: List[str] = []
    for k in keys:
        m_no = s_no[k][0]
        m_ca = s_ca[k][0]
        diff = m_ca - m_no
        pct = (diff / m_no * 100.0) if m_no > 1e-9 else 0.0
        deltas.append(f"{k}: {diff:+.3f}s ({pct:+.0f}%)")
    logger.info("delta cache - no-cache (mean): %s", "  |  ".join(deltas))
    logger.info("=" * 90)
    _print_mean_pct(s_no, "no-cache")
    _print_mean_pct(s_ca, "cache")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage2 cache vs no-cache micro-benchmark（DDP / torchrun）")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--n-warmup-batches", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--shard-lru", type=int, default=4)
    ap.add_argument("--mode", choices=("both", "no-cache", "cache"), default="both")
    ap.add_argument("--compare-mean-total", type=float, default=None, metavar="SEC")
    ap.add_argument(
        "--ddp-global-dataset",
        action="store_true",
        help=(
            "默认：max_samples = (warmup+measure)*batch_size*world_size，使每 rank 的 batch 数与单卡脚本一致。"
            "若加本开关：max_samples = (warmup+measure)*batch_size（总 jsonl 上限与单卡相同，各 rank 分片）。"
        ),
    )
    args = ap.parse_args()

    if args.mode in ("both", "cache") and args.cache_dir is None:
        logger.error("--mode %s 时必须提供 --cache-dir", args.mode)
        sys.exit(1)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and not dist.is_initialized():
        _pg_timeout = _dt.timedelta(seconds=int(os.environ.get("PG_TIMEOUT_SECONDS", "14400")))
        dist.init_process_group(backend="nccl", timeout=_pg_timeout)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    try:
        if use_ddp:
            if not torch.cuda.is_available():
                logger.error("DDP 需要 CUDA"); sys.exit(1)
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            if not args.no_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        cfg = StrucLiftConfig()
        model = StrucLift(cfg).to(device)
        if args.init_from is not None:
            if not args.init_from.is_file():
                logger.error("--init-from 不存在: %s", args.init_from)
                sys.exit(1)
            _load_init_from(model, args.init_from.expanduser().resolve(), device, rank=rank)
        model.freeze_encoders()
        model.freeze_decoder_adapters()

        trainer = Stage2Trainer(
            model, cfg, device=device,
            rank=rank, world_size=world_size, use_ddp=use_ddp,
        )

        n_measure = max(1, (args.n_samples // args.batch_size) - args.n_warmup_batches)
        n_total_batches = args.n_warmup_batches + n_measure
        per_gpu_floor = n_total_batches * args.batch_size
        if args.ddp_global_dataset:
            max_samples_cap = per_gpu_floor
        else:
            max_samples_cap = per_gpu_floor * max(world_size, 1)

        if rank == 0:
            logger.info(
                "Benchmark plan: world_size=%d batch_size(per GPU)=%d warmup=%d measure=%d | "
                "max_samples(jsonl cap)=%d | ddp_global_dataset=%s",
                world_size, args.batch_size, args.n_warmup_batches, n_measure,
                max_samples_cap, args.ddp_global_dataset,
            )

        s_no: Dict[str, Tuple[float, float]] = {}
        s_ca: Dict[str, Tuple[float, float]] = {}

        if args.mode in ("both", "no-cache"):
            if rank == 0:
                logger.info("")
                logger.info("########## NO-CACHE (DDP) ##########")
            if use_ddp:
                dist.barrier()
            loader_no = make_binskel_stage2_dataloader(
                args.jsonl,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                max_samples=max_samples_cap,
                skip_invalid=True,
                pin_memory=device.type == "cuda",
                prefetch_factor=int(args.prefetch_factor),
                encoder_cache_dir=None,
                distributed=use_ddp,
                rank=rank,
                world_size=world_size,
            )
            sampler = getattr(loader_no, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(0)
            t_no = benchmark_mode(
                mode_name="no-cache",
                model=model, trainer=trainer, loader=loader_no, device=device,
                rank=rank, n_warmup=args.n_warmup_batches, n_measure=n_measure,
            )
            s_no = _summary(t_no)
            if use_ddp:
                dist.barrier()

        if args.mode in ("both", "cache"):
            assert args.cache_dir is not None
            if rank == 0:
                logger.info("")
                logger.info("########## WITH CACHE (DDP) ##########")
            if use_ddp:
                dist.barrier()
            loader_ca = make_binskel_stage2_dataloader(
                args.jsonl,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                max_samples=max_samples_cap,
                skip_invalid=True,
                pin_memory=device.type == "cuda",
                prefetch_factor=int(args.prefetch_factor),
                encoder_cache_dir=args.cache_dir,
                encoder_cache_shard_lru=int(args.shard_lru),
                distributed=use_ddp,
                rank=rank,
                world_size=world_size,
            )
            sampler_ca = getattr(loader_ca, "sampler", None)
            if sampler_ca is not None and hasattr(sampler_ca, "set_epoch"):
                sampler_ca.set_epoch(0)
            t_ca = benchmark_mode(
                mode_name="cache",
                model=model, trainer=trainer, loader=loader_ca, device=device,
                rank=rank, n_warmup=args.n_warmup_batches, n_measure=n_measure,
            )
            s_ca = _summary(t_ca)
            if use_ddp:
                dist.barrier()

        if rank == 0:
            if args.mode == "both":
                _print_compare(s_no, s_ca)
            elif args.mode == "no-cache":
                keys = ["data", "h2d", "encode", "loss", "bwd", "total"]
                logger.info("")
                logger.info("[no-cache only] mean / median (s): %s",
                            ", ".join(f"{k}={s_no[k][0]:.4f}/{s_no[k][1]:.4f}" for k in keys))
                _print_mean_pct(s_no, "no-cache (DDP)")
            else:
                keys = ["data", "h2d", "encode", "loss", "bwd", "total"]
                logger.info("")
                logger.info("[cache only] mean / median (s): %s",
                            ", ".join(f"{k}={s_ca[k][0]:.4f}/{s_ca[k][1]:.4f}" for k in keys))
                _print_mean_pct(s_ca, "cache (DDP)")

            if args.compare_mean_total is not None:
                base = float(args.compare_mean_total)
                ref_mean = (
                    s_no["total"][0] if args.mode in ("both", "no-cache") else s_ca["total"][0]
                )
                if base > 1e-9:
                    delta = ref_mean - base
                    pct = (ref_mean / base - 1.0) * 100.0
                    logger.info("")
                    logger.info(
                        "[vs baseline] mean_total=%.4fs  baseline=%.4fs  Δ=%+.4fs (%+.1f%% vs baseline)",
                        ref_mean, base, delta, pct,
                    )

            logger.info("")
            logger.info("说明（DDP）：")
            logger.info("  mean total ≈ 单训练 step 墙钟（rank0）；含 backward 后 all_reduce(grad) + clip + step")
            logger.info("  全局每 step 样本数 ≈ batch_size × world_size（本脚本与 train_stage2_binskel 一致）")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

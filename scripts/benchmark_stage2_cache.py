#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 cache vs no-cache 单卡 micro-benchmark。

跑同一份 jsonl 的前 ``--n-samples`` 条，分两次跑（无 cache / 有 cache），细分每个 step
到 data / h2d / encode / loss / bwd+opt 五项耗时。最后给两模式的并排对比，便于决定是
保留 cache（且需要换格式）还是回退到 no-cache。

用法（单卡）::

    # 仅测 no-cache 整步耗时（256 样本、bs=64、卡 1），并与优化前基线 22s/step 对比：
    CUDA_VISIBLE_DEVICES=1 \
    /path/to/env/bin/python3 scripts/benchmark_stage2_cache.py \
      --mode no-cache \
      --jsonl /path/to/train.jsonl \
      --init-from /path/to/stage1.pt \
      --batch-size 64 --n-samples 256 --num-workers 4 --prefetch-factor 4 \
      --compare-mean-total 22

    # no-cache + cache 并排对比（需提供 --cache-dir）：
    CUDA_VISIBLE_DEVICES=1 \
    /path/to/env/bin/python3 scripts/benchmark_stage2_cache.py \
      --jsonl /path/to/train.jsonl \
      --cache-dir /path/to/cache \
      --init-from /path/to/stage1.pt \
      --batch-size 64 --n-samples 256 --num-workers 4 --prefetch-factor 4

输出示例（节选）::

    [no-cache]  step  data    h2d   encode   loss    bwd     total
                  0   0.45   0.04   2.130   0.087   1.821   4.524
                  ...
                avg   0.32   0.03   2.054   0.092   1.815   4.314

    [cache]     step  data    h2d   encode   loss    bwd     total
                  0   3.21   0.05   0.412   0.090   1.832   5.564
                  ...
                avg   2.87   0.04   0.398   0.094   1.821   5.222

    [delta]     data: +2.55s/step (+800%)  encode: -1.66s/step (-81%)  total: +0.91s/step (+21%)
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import make_binskel_stage2_dataloader
from struclift.models.struclift import StrucLift
from struclift.training.stage2_alignment import Stage2Trainer

# Optional fine-grained alignment profiler. 设 STRUCLIFT_PROFILE_ALIGNMENT=1 以
# 在每 step 末尾打印 alignment 内部 (proj_sim/region_aligner/scot_region/...) 段耗时。
try:
    from struclift.models.module_b import (
        report_alignment_profile,
        reset_alignment_profile,
        _PROFILE_ALIGNMENT as _ALIGN_PROFILE_ON,
    )
except ImportError:  # 兼容旧版 module_b
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


def _load_init_from(model: StrucLift, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("init_from 已载入：missing=%d unexpected=%d", len(missing), len(unexpected))


def benchmark_mode(
    *,
    mode_name: str,
    model: StrucLift,
    trainer: Stage2Trainer,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int,
    n_measure: int,
) -> List[Dict[str, float]]:
    """对一个 DataLoader 跑 ``n_warmup + n_measure`` 个 batch，返回每 batch 的细分耗时。"""
    timings: List[Dict[str, float]] = []
    loader_iter = iter(loader)
    optimizer = trainer.optimizer

    measured = 0
    seen = 0
    while measured < n_measure:
        # ── data fetch（含 worker 收数 / collate / cache shard load）─────
        t0_data = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            logger.warning("[%s] DataLoader 用尽，仅记录 %d 个 measure batch", mode_name, measured)
            break
        # 不 sync——data 时间是 CPU 侧的 Python 等待
        t_data = time.perf_counter() - t0_data

        # ── H2D ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        batch_d: Dict[str, Any] = {
            k: (v.to(device, non_blocking=device.type == "cuda") if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        _sync(device)
        t_h2d = time.perf_counter() - t0

        # ── encode（含 alignment forward）────────────────────────────────
        if _ALIGN_PROFILE_ON:
            reset_alignment_profile()
        t0 = time.perf_counter()
        enc_out = trainer._forward_stage2_encoder_out(batch_d)
        _sync(device)
        t_enc = time.perf_counter() - t0
        align_breakdown = report_alignment_profile() if _ALIGN_PROFILE_ON else {}

        # ── loss compute（SCOT / edge / region / src_count / node_hn）──
        t0 = time.perf_counter()
        loss, _comp = trainer._loss_tensors_from_enc_out(
            enc_out, batch_d, hn_on=False, epoch=1, log_nonfinite_warnings=False,
        )
        _sync(device)
        t_loss = time.perf_counter() - t0

        # ── bwd + step ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
        logger.info(
            "[%s] step %d/%d  data=%.3f  h2d=%.3f  enc=%.3f  loss=%.3f  bwd=%.3f  total=%.3f%s",
            mode_name, measured, n_measure,
            rec["data"], rec["h2d"], rec["encode"], rec["loss"], rec["bwd"], rec["total"],
            align_str,
        )
    return timings


def _summary(timings: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """每项返回 (mean, median)。"""
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
    """打印各阶段 mean 秒数及占 mean total 的百分比（训练一步五项之和 ≈ total）。"""
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
                "cache",    *(s_ca[k][0] for k in keys))
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
    ap = argparse.ArgumentParser(description="Stage2 cache vs no-cache micro-benchmark")
    ap.add_argument("--jsonl", type=Path, required=True, help="训练 jsonl（含 alignment_matrix）")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="encoder cache 根目录；--mode cache/both 时需要",
    )
    ap.add_argument("--init-from", type=Path, default=None,
                    help="可选 stage1 ckpt（不传时 encoder 用随机初始化，对耗时无影响）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=256, help="每模式跑多少样本（含 warmup）")
    ap.add_argument("--n-warmup-batches", type=int, default=2, help="不计入统计的 warmup batch 数")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--shard-lru", type=int, default=4)
    ap.add_argument(
        "--mode",
        choices=("both", "no-cache", "cache"),
        default="both",
        help="both=先后跑 no-cache 与 cache；no-cache=仅测完整 encode（常用：与优化前 step 对比）；cache=仅 cache",
    )
    ap.add_argument(
        "--compare-mean-total",
        type=float,
        default=None,
        metavar="SEC",
        help="可选：打印当前 mean total 相对该基线（秒）的差值与比例，例如优化前约 22",
    )
    args = ap.parse_args()

    if args.mode in ("both", "cache"):
        if args.cache_dir is None:
            logger.error("--mode %s 时必须提供 --cache-dir", args.mode)
            sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
            logger.error("--init-from 不存在: %s", args.init_from); sys.exit(1)
        _load_init_from(model, args.init_from.expanduser().resolve(), device)
    # 与训练一致：冻结 encoder + decoder，仅 alignment 反传
    model.freeze_encoders()
    model.freeze_decoder_adapters()

    trainer = Stage2Trainer(
        model, cfg, device=device,
        rank=0, world_size=1, use_ddp=False,
    )

    n_measure = max(1, (args.n_samples // args.batch_size) - args.n_warmup_batches)
    n_total = args.n_warmup_batches + n_measure
    n_samples_needed = n_total * args.batch_size

    logger.info(
        "Benchmark plan: batch_size=%d  warmup=%d  measure=%d  (共需 %d 样本)",
        args.batch_size, args.n_warmup_batches, n_measure, n_samples_needed,
    )

    s_no: Dict[str, Tuple[float, float]] = {}
    s_ca: Dict[str, Tuple[float, float]] = {}

    # ── 模式 1：no-cache ───────────────────────────────────────────────
    if args.mode in ("both", "no-cache"):
        logger.info("")
        logger.info("########## NO-CACHE ##########")
        loader_no = make_binskel_stage2_dataloader(
            args.jsonl,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=n_samples_needed,
            skip_invalid=True,
            pin_memory=device.type == "cuda",
            prefetch_factor=int(args.prefetch_factor),
            encoder_cache_dir=None,
        )
        t_no = benchmark_mode(
            mode_name="no-cache",
            model=model, trainer=trainer, loader=loader_no, device=device,
            n_warmup=args.n_warmup_batches, n_measure=n_measure,
        )
        s_no = _summary(t_no)

    # ── 模式 2：cache ─────────────────────────────────────────────────
    if args.mode in ("both", "cache"):
        logger.info("")
        logger.info("########## WITH CACHE ##########")
        assert args.cache_dir is not None
        loader_ca = make_binskel_stage2_dataloader(
            args.jsonl,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=n_samples_needed,
            skip_invalid=True,
            pin_memory=device.type == "cuda",
            prefetch_factor=int(args.prefetch_factor),
            encoder_cache_dir=args.cache_dir,
            encoder_cache_shard_lru=int(args.shard_lru),
        )
        t_ca = benchmark_mode(
            mode_name="cache",
            model=model, trainer=trainer, loader=loader_ca, device=device,
            n_warmup=args.n_warmup_batches, n_measure=n_measure,
        )
        s_ca = _summary(t_ca)

    if args.mode == "both":
        _print_compare(s_no, s_ca)
    elif args.mode == "no-cache":
        keys = ["data", "h2d", "encode", "loss", "bwd", "total"]
        logger.info("")
        logger.info("[no-cache only] mean / median (s): %s",
                    ", ".join(f"{k}={s_no[k][0]:.4f}/{s_no[k][1]:.4f}" for k in keys))
        _print_mean_pct(s_no, "no-cache")
    else:
        keys = ["data", "h2d", "encode", "loss", "bwd", "total"]
        logger.info("")
        logger.info("[cache only] mean / median (s): %s",
                    ", ".join(f"{k}={s_ca[k][0]:.4f}/{s_ca[k][1]:.4f}" for k in keys))
        _print_mean_pct(s_ca, "cache")

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
    logger.info("说明：")
    logger.info("  data    : 等 DataLoader 给出 batch 的 CPU 时间（含 collate 与 cache shard IO）")
    logger.info("  h2d     : batch 张量拷贝到 GPU 的时间（含 cuda.synchronize）")
    logger.info("  encode  : 完整 model.encode（cache 模式下 binary/source encoder 跳过，但 alignment 仍跑）")
    logger.info("  loss    : SCOT/edge/CA/region/src_count/node_hn 计算（不含 bwd）")
    logger.info("  bwd     : loss.backward() + optimizer.step()")
    logger.info("  total   : 上述 5 项之和；可与之前的 epoch 整体时间反推 step 数对得上")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 训练性能诊断脚本（~90 秒出结果）。

用法（单卡、单进程即可，不需要 torchrun）::

  CUDA_VISIBLE_DEVICES=6 python scripts/diag_stage1_perf.py \
      --jsonl /data/chaoni/WQM/datasets/AB_train.jsonl \
      --batch-size 64 \
      --num-workers 8 \
      --cross-opt-pairs

输出 4 段数字，分别说明：
  [A] 磁盘读取 JSONL 单行耗时
  [B] 单样本 collate 耗时（含 region 划分）
  [C] 纯 DataLoader 吐 batch 速度（不过模型）
  [D] 完整训练 step 耗时分段（io / move / fwd / bwd / opt）

把输出完整贴回来，我根据这四组数字定瓶颈位置。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    BinSkelJsonlDataset,
    collate_binskel_batch,
    make_binskel_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.losses import (
    GraphBinarySourceContrastiveLoss,
    NeighborReconLoss,
    PatternClassificationLoss,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def bench_raw_jsonl_read(path: Path, n: int = 100) -> None:
    section(f"[A] 纯磁盘读+JSON 解析速度（采样 {n} 条）")
    offsets: List[int] = []
    with open(path, "rb") as f:
        cur = 0
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            offsets.append(cur)
            cur += len(line)
    if len(offsets) < n:
        print(f"  文件只有 {len(offsets)} 行，n 调小")
        n = len(offsets)
    t0 = time.time()
    total_bytes = 0
    total_keys = 0
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            line = f.readline()
            d = json.loads(line.decode("utf-8"))
            total_bytes += len(line)
            total_keys += len(d)
    elapsed = time.time() - t0
    print(f"  读 {n} 行共 {total_bytes/1024:.1f} KB，耗时 {elapsed*1000:.1f} ms")
    print(f"  平均每条 {elapsed/n*1000:.2f} ms （含 json.loads）")
    print(f"  吞吐 {total_bytes/1024/1024/elapsed:.1f} MB/s")
    print(f"  若 < 50 MB/s：磁盘/文件系统是瓶颈（HDD / 网络盘 / 远程挂载）")
    print(f"  若 > 200 MB/s：磁盘 OK，瓶颈在别处（NVMe 合理值）")


def bench_collate(dataset: BinSkelJsonlDataset, batch_size: int, n_batches: int = 20) -> None:
    section(f"[B] collate_binskel_batch 单 batch 耗时（{n_batches} 个 bs={batch_size}）")
    samples: List[Dict[str, Any]] = []
    idx = 0
    while len(samples) < batch_size * n_batches and idx < len(dataset):
        item = dataset._read_one(idx)
        idx += 1
        if item:
            samples.append(item)
    if len(samples) < batch_size * n_batches:
        n_batches = len(samples) // batch_size
        if n_batches == 0:
            print("  数据不够，跳过")
            return
    times_ms: List[float] = []
    for bi in range(n_batches):
        chunk = samples[bi * batch_size:(bi + 1) * batch_size]
        t0 = time.time()
        _ = collate_binskel_batch(
            chunk,
            inst_dropout_p=0.15,
            edge_dropout_p=0.10,
            add_func_uid=True,
        )
        times_ms.append((time.time() - t0) * 1000)
    times_ms.sort()
    mean = sum(times_ms) / len(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p90 = times_ms[int(len(times_ms) * 0.9)]
    print(f"  mean={mean:.1f}ms  p50={p50:.1f}ms  p90={p90:.1f}ms")
    print(f"  若 > 300ms：Python collate 是瓶颈（region 划分 / dropout / tensor 构造）")
    print(f"  若 < 100ms：collate 不是问题")


def bench_loader_pure(loader, n_steps: int = 30, warmup: int = 5) -> None:
    section(f"[C] 纯 DataLoader 吐 batch 速度（{n_steps} step，前 {warmup} 为 warmup）")
    it = iter(loader)
    # warmup
    for _ in range(warmup):
        try:
            _ = next(it)
        except StopIteration:
            break
    times_ms: List[float] = []
    t_prev = time.time()
    for _ in range(n_steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        t_now = time.time()
        times_ms.append((t_now - t_prev) * 1000)
        t_prev = t_now
    if not times_ms:
        print("  loader 吐不出 batch")
        return
    times_ms.sort()
    mean = sum(times_ms) / len(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p90 = times_ms[int(len(times_ms) * 0.9)]
    print(f"  mean={mean:.0f}ms/batch  p50={p50:.0f}ms  p90={p90:.0f}ms")
    print(f"  若 > 2000ms：IO + collate 合计太慢，workers 不够或文件读太慢")
    print(f"  若 < 500ms：DataLoader 不是瓶颈，真瓶颈在 GPU 端")


def bench_training_step(
    loader,
    device: torch.device,
    cfg: StrucLiftConfig,
    n_steps: int = 15,
    warmup: int = 5,
) -> None:
    section(f"[D] 完整训练 step 分段耗时（{n_steps} step，前 {warmup} 为 warmup）")
    model = StrucLift(cfg).to(device)
    model.freeze_alignment()
    model.freeze_decoder_adapters()
    model.train()
    params = (
        list(model.binary_encoder.parameters())
        + list(model.source_encoder.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    pattern_fn = PatternClassificationLoss(num_classes=cfg.vocab.num_patterns)
    neighbor_fn = NeighborReconLoss(tau=0.07)
    graph_fn = GraphBinarySourceContrastiveLoss(tau=0.07, mode="multi_positive")

    it = iter(loader)
    stats: Dict[str, List[float]] = {k: [] for k in ("io", "move", "fwd", "bwd", "opt")}

    t_last = time.time()
    for i in range(warmup + n_steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        t0 = time.time()
        batch = move_binskel_batch_to_device(batch, device, non_blocking=True)
        t1 = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(
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
            enc_out = out.encoder_out
            loss_p = pattern_fn(
                enc_out.pattern_logits, batch["pattern_labels"],
                weights=batch.get("pattern_weights"),
            )
            h_norm = torch.nn.functional.normalize(enc_out.binary_node_embs, dim=-1)
            loss_n = neighbor_fn(h_norm, batch["bin_edge_index"], batch.get("bin_batch"))
            loss = loss_p + 0.5 * loss_n
            if enc_out.source_graph_emb is not None and enc_out.source_graph_emb.numel() > 0:
                loss_g = graph_fn(
                    enc_out.binary_graph_emb, enc_out.source_graph_emb,
                    func_uid=batch.get("graph_func_uid"),
                )
                loss = loss + 0.1 * loss_g
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.time()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.time()

        if i >= warmup:
            stats["io"].append((t0 - t_last) * 1000)
            stats["move"].append((t1 - t0) * 1000)
            stats["fwd"].append((t2 - t1) * 1000)
            stats["bwd"].append((t3 - t2) * 1000)
            stats["opt"].append((t4 - t3) * 1000)
        t_last = t4

    def mean(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    total = sum(mean(v) for v in stats.values())
    print(f"  平均一 step 总耗时: {total:.0f} ms")
    for k in ("io", "move", "fwd", "bwd", "opt"):
        m = mean(stats[k])
        pct = 100.0 * m / max(total, 1e-6)
        print(f"    {k:4s} = {m:6.1f} ms  ({pct:5.1f}%)")
    print("  io 就是 DataLoader 等待时间；占比越高说明 GPU 越空转")
    print("  理想情况：io<30% + fwd+bwd 占 60%+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--cross-opt-pairs", action="store_true")
    ap.add_argument("--inst-dropout", type=float, default=0.15)
    ap.add_argument("--edge-dropout", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    path = args.jsonl.expanduser().resolve()
    print(f"[info] JSONL: {path}  size={path.stat().st_size/1024/1024:.1f} MB")

    dev_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_s)
    print(f"[info] device: {device}")

    bench_raw_jsonl_read(path, n=200)

    ds = BinSkelJsonlDataset(
        path, require_source=True, skip_invalid=True,
        cross_opt_pairs=args.cross_opt_pairs,
    )
    print(f"[info] dataset len = {len(ds)}")
    bench_collate(ds, batch_size=args.batch_size, n_batches=20)

    loader = make_binskel_dataloader(
        path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_samples=args.max_samples,
        require_source=True,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
        distributed=False,
        inst_dropout_p=args.inst_dropout,
        edge_dropout_p=args.edge_dropout,
        add_func_uid=True,
        cross_opt_pairs=args.cross_opt_pairs,
    )
    bench_loader_pure(loader, n_steps=30, warmup=5)

    # 重新构造 loader（上一步 iter 已用掉 warmup+n_steps batch）
    loader2 = make_binskel_dataloader(
        path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_samples=args.max_samples,
        require_source=True,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
        distributed=False,
        inst_dropout_p=args.inst_dropout,
        edge_dropout_p=args.edge_dropout,
        add_func_uid=True,
        cross_opt_pairs=args.cross_opt_pairs,
    )
    cfg = StrucLiftConfig()
    bench_training_step(loader2, device, cfg, n_steps=15, warmup=5)

    print("\n诊断完成。请把以上完整输出贴给我分析瓶颈。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage1 前向瓶颈细粒度诊断（~60 秒出结果）。

完整模拟 ``Stage1Trainer.train_epoch`` 的前向链路，把每一 step 拆成 7 段：

    io   : DataLoader 等待（GPU 空转的主嫌疑）
    move : batch → device 搬运
    enc  : BinaryEncoder + SourceEncoder 前向
    pat  : pattern_cls + neighbor_recon 两个小 loss
    grp  : graph_align（含 hard-neg lookup 分支，默认不挂 bank 所以接近纯前向）
    reg  : region_infonce（Python 循环重灾区）
    bwd  : backward + optimizer.step

用法（**在空闲卡上跑**，不动 6/7 训练）::

    CUDA_VISIBLE_DEVICES=0 python scripts/diag_stage1_fwd_breakdown.py \
        --jsonl /data/chaoni/WQM/datasets/AB_2048.jsonl \
        --batch-size 64 \
        --num-workers 16 \
        --prefetch-factor 4 \
        --cross-opt-pairs \
        --max-samples 4000

对比组：
    - 默认：开 region + cross-opt
    - ``--no-region``：排除 region_infonce，看它是否是主因
    - ``--no-cross-opt``：关掉 cross-opt，看配对的开销

输出里 **占比 > 40% 的阶段就是瓶颈**。把 3 次运行结果贴回来分析。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    make_binskel_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.training.stage1_pretrain import Stage1Trainer

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_fwd_breakdown(
    loader,
    trainer: Stage1Trainer,
    device: torch.device,
    n_steps: int,
    warmup: int,
    disable_region: bool,
) -> Dict[str, float]:
    """模拟 train_epoch 单 step 并对 7 个阶段计时（每段后都 ``cuda.synchronize``）。"""
    model = trainer.model
    model.train()
    optimizer = trainer.optimizer
    pattern_fn = trainer.pattern_loss_fn
    neighbor_fn = trainer.neighbor_loss_fn
    graph_fn = trainer.graph_align_loss_fn
    lam_recon = trainer.lambda_recon
    lam_graph = trainer.lambda_graph_align
    lam_region = 0.0 if disable_region else trainer.lambda_region

    it = iter(loader)
    stats: Dict[str, List[float]] = {
        k: [] for k in ("io", "move", "enc", "pat", "grp", "reg", "bwd")
    }

    t_last = time.time()
    for i in range(warmup + n_steps):
        try:
            batch = next(it)
        except StopIteration:
            break

        t0 = time.time()
        batch = move_binskel_batch_to_device(batch, device, non_blocking=True)
        _sync(device)
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
            _sync(device)
            t2 = time.time()
            enc_out = out.encoder_out

            loss_p = pattern_fn(
                enc_out.pattern_logits, batch["pattern_labels"],
                weights=batch.get("pattern_weights"),
            )
            h_norm = nn.functional.normalize(enc_out.binary_node_embs, dim=-1)
            loss_n = neighbor_fn(h_norm, batch["bin_edge_index"], batch.get("bin_batch"))
            loss = loss_p + lam_recon * loss_n
            _sync(device)
            t3 = time.time()

            if (
                lam_graph > 0
                and enc_out.source_graph_emb is not None
                and enc_out.source_graph_emb.numel() > 0
            ):
                loss_g = graph_fn(
                    enc_out.binary_graph_emb,
                    enc_out.source_graph_emb,
                    func_uid=batch.get("graph_func_uid"),
                )
                loss = loss + lam_graph * loss_g
            _sync(device)
            t4 = time.time()

            if (
                lam_region > 0
                and enc_out.source_node_embs is not None
                and batch.get("binary_regions_list") is not None
                and batch.get("source_regions_list") is not None
                and batch.get("region_pos_mask_list") is not None
                and batch.get("bin_batch") is not None
                and batch.get("src_batch") is not None
            ):
                loss_r = trainer._compute_region_infonce(
                    enc_out.binary_node_embs,
                    enc_out.source_node_embs,
                    batch["bin_batch"],
                    batch["src_batch"],
                    batch["binary_regions_list"],
                    batch["source_regions_list"],
                    batch["region_pos_mask_list"],
                )
                if loss_r is not None:
                    loss = loss + lam_region * loss_r
            _sync(device)
            t5 = time.time()

        loss.backward()
        optimizer.step()
        _sync(device)
        t6 = time.time()

        if i >= warmup:
            stats["io"].append((t0 - t_last) * 1000)
            stats["move"].append((t1 - t0) * 1000)
            stats["enc"].append((t2 - t1) * 1000)
            stats["pat"].append((t3 - t2) * 1000)
            stats["grp"].append((t4 - t3) * 1000)
            stats["reg"].append((t5 - t4) * 1000)
            stats["bwd"].append((t6 - t5) * 1000)
        t_last = t6

    def _mean(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    return {k: _mean(v) for k, v in stats.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="只取前 N 条（默认 4000，约 60 秒跑完诊断）")
    ap.add_argument("--n-steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cross-opt-pairs", action="store_true")
    ap.add_argument("--no-region", action="store_true",
                    help="禁用 region_infonce，用于对比确认 region 是不是主因")
    ap.add_argument("--inst-dropout", type=float, default=0.15)
    ap.add_argument("--edge-dropout", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    path = args.jsonl.expanduser().resolve()
    dev_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_s)
    print(f"[info] JSONL: {path}")
    print(f"[info] device={device}  batch_size={args.batch_size}  "
          f"workers={args.num_workers}  prefetch={args.prefetch_factor}  "
          f"cross_opt={args.cross_opt_pairs}  region={'OFF' if args.no_region else 'ON'}")

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

    cfg = StrucLiftConfig()
    model = StrucLift(cfg).to(device)
    model.freeze_alignment()
    model.freeze_decoder_adapters()
    trainer = Stage1Trainer(model, cfg, device=device)

    means = bench_fwd_breakdown(
        loader, trainer, device,
        n_steps=args.n_steps, warmup=args.warmup,
        disable_region=args.no_region,
    )

    total = sum(means.values())
    print("\n" + "=" * 72)
    print(f"单 step 总耗时（均值，n={args.n_steps}）：{total:.0f} ms")
    print("=" * 72)
    labels = {
        "io": "DataLoader 等待（io）",
        "move": "Host → Device 搬运",
        "enc": "Encoder 前向（binary + source）",
        "pat": "pattern + neighbor loss",
        "grp": "graph_align loss",
        "reg": "region_infonce（Python 循环）",
        "bwd": "backward + optimizer.step",
    }
    for k in ("io", "move", "enc", "pat", "grp", "reg", "bwd"):
        m = means[k]
        pct = 100.0 * m / max(total, 1e-6)
        marker = "  <-- 瓶颈" if pct > 40 else ""
        print(f"  {k:4s}  {m:7.1f} ms  {pct:5.1f}%   {labels[k]}{marker}")

    print("\n判读提示:")
    print("  - io > 40%  : GPU 在等数据 → 加 num-workers / prefetch-factor")
    print("  - enc > 40% : 真正 GPU-bound → 只能 torch.compile / flash-attention")
    print("  - reg > 30% : _compute_region_infonce 的 Python 循环慢 → 需改为批量化实现")
    print("  - bwd > 50% : 反向梯度同步慢（DDP）或模型过大")
    print("\n诊断完成。")


if __name__ == "__main__":
    main()

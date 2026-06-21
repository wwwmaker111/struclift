#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 encoder cache 离线预计算。

Stage2 训练中 Module A（``binary_encoder`` / ``source_encoder`` + ``SubgraphPatternClassifier``）
**冻结**，相同 jsonl + 相同 stage1 权重的 forward 输出每轮都一样。本脚本一次性把这些
张量算好写到磁盘，训练时由 :class:`struclift.data.binskel_jsonl.Stage2EncoderCache`
按 idx 读出，跳过 encoder forward —— stage4 单 epoch 量级可降至原来的 ~1/2。

输出布局（每个 jsonl 一组）::

    <cache_dir>/<jsonl_filename_stem>/
        manifest.json
        shard_00000000.pt
        shard_00000001.pt
        ...

每个 ``shard_xxx.pt`` = ``list[shard_size]``，元素为
``dict[str, Tensor]``：``bin_node_embs / pattern_logits / bin_graph_emb /
src_node_embs / src_graph_emb``，全部 CPU + ``torch.float32``。

manifest 含 stage1 ckpt 的 sha256 + jsonl 文件 sha256，训练侧加载时**仅校验文件存在**，
若 stage1 / jsonl 改动，请手动 rm 重生（或加 ``--force`` 重写）。

用法示例（生成 5 档训练 + 5 档验证 = 10 个 jsonl 的 cache，单卡）::

    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_stage2_encoder_cache.py \
        --init-from /data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt \
        --jsonl /data/chaoni/WQM/datasets/stage2_train_o0.jsonl \
                /data/chaoni/WQM/datasets/stage2_train_o1.jsonl \
                /data/chaoni/WQM/datasets/stage2_train_o2.jsonl \
                /data/chaoni/WQM/datasets/stage2_train_o3.jsonl \
                /data/chaoni/WQM/datasets/stage2_train_os.jsonl \
                /data/chaoni/WQM/datasets/stage2_val_o0.jsonl \
                /data/chaoni/WQM/datasets/stage2_val_o1.jsonl \
                /data/chaoni/WQM/datasets/stage2_val_o2.jsonl \
                /data/chaoni/WQM/datasets/stage2_val_o3.jsonl \
                /data/chaoni/WQM/datasets/stage2_val_os.jsonl \
        --cache-dir /data/chaoni/WQM/cache/stage2_encoder \
        --batch-size 16 --shard-size 4096
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    BinSkelJsonlDataset,
    STAGE2_CACHE_FIELDS,
    STAGE2_CACHE_MANIFEST_NAME,
    STAGE2_CACHE_VERSION,
    collate_binskel_stage2_batch,
)
from struclift.models.struclift import StrucLift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """流式 sha256；用于 manifest 校验 stage1 ckpt 与 jsonl 是否一致。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device, non_blocking=device.type == "cuda")
        if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _split_per_sample(
    enc_out: Any, batch: Dict[str, torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    """把 PyG-batched encoder 输出按样本拆回单图，确保 idx 与 jsonl 行对齐。"""
    bin_batch = batch["bin_batch"].cpu()
    src_batch = batch["src_batch"].cpu()
    B = int(bin_batch.max().item()) + 1 if bin_batch.numel() > 0 else 1

    bin_node = enc_out.binary_node_embs.detach().to(dtype=torch.float32, device="cpu")
    pat_lo = enc_out.pattern_logits.detach().to(dtype=torch.float32, device="cpu")
    bin_g = enc_out.binary_graph_emb.detach().to(dtype=torch.float32, device="cpu")
    src_node = enc_out.source_node_embs.detach().to(dtype=torch.float32, device="cpu")
    src_g = enc_out.source_graph_emb.detach().to(dtype=torch.float32, device="cpu")

    out: List[Dict[str, torch.Tensor]] = []
    for gi in range(B):
        b_mask = (bin_batch == gi)
        s_mask = (src_batch == gi)
        out.append({
            "bin_node_embs": bin_node[b_mask].contiguous(),
            "pattern_logits": pat_lo[b_mask].contiguous(),
            "bin_graph_emb": bin_g[gi : gi + 1].contiguous(),
            "src_node_embs": src_node[s_mask].contiguous(),
            "src_graph_emb": src_g[gi : gi + 1].contiguous(),
        })
    return out


def _process_one_jsonl(
    *,
    jsonl_path: Path,
    cache_subdir: Path,
    model: StrucLift,
    device: torch.device,
    batch_size: int,
    shard_size: int,
    num_workers: int,
    stage1_meta: Dict[str, Any],
    force: bool,
) -> None:
    manifest_path = cache_subdir / STAGE2_CACHE_MANIFEST_NAME
    if manifest_path.is_file() and not force:
        logger.info("[%s] manifest 已存在，跳过（用 --force 强制重写）: %s",
                    jsonl_path.name, manifest_path)
        return
    cache_subdir.mkdir(parents=True, exist_ok=True)

    ds = BinSkelJsonlDataset(
        jsonl_path,
        require_alignment=True,
        skip_invalid=True,
    )
    n_samples = len(ds)
    if n_samples == 0:
        logger.warning("[%s] 数据集无有效样本，跳过", jsonl_path.name)
        return
    n_shards = (n_samples + shard_size - 1) // shard_size

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_binskel_stage2_batch,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    logger.info("[%s] 开始预计算: n_samples=%d shard_size=%d n_shards=%d",
                jsonl_path.name, n_samples, shard_size, n_shards)

    pending: List[Dict[str, torch.Tensor]] = []
    next_shard_id = 0
    written = 0
    t0 = time.perf_counter()
    last_log_t = t0

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = _move_batch(batch, device)
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
                run_alignment=False,
            )
            per_sample = _split_per_sample(enc_out, batch)
            pending.extend(per_sample)

            while len(pending) >= shard_size:
                shard = pending[:shard_size]
                pending = pending[shard_size:]
                shard_path = cache_subdir / f"shard_{next_shard_id:08d}.pt"
                torch.save(shard, shard_path)
                next_shard_id += 1
                written += shard_size

            now = time.perf_counter()
            if now - last_log_t > 30.0:
                ratio = (written + len(pending)) / max(n_samples, 1)
                logger.info(
                    "[%s] progress: %d/%d (%.1f%%) shards_written=%d elapsed=%.0fs",
                    jsonl_path.name,
                    written + len(pending), n_samples, ratio * 100.0,
                    next_shard_id, now - t0,
                )
                last_log_t = now

    if pending:
        shard_path = cache_subdir / f"shard_{next_shard_id:08d}.pt"
        torch.save(pending, shard_path)
        next_shard_id += 1
        written += len(pending)

    if written != n_samples:
        raise RuntimeError(
            f"[{jsonl_path.name}] 写入样本数 {written} 与数据集 n_samples {n_samples} 不一致；"
            f"中途可能 OOM/数据损坏，请检查。",
        )

    manifest = {
        "version": STAGE2_CACHE_VERSION,
        "n_samples": int(n_samples),
        "shard_size": int(shard_size),
        "n_shards": int(next_shard_id),
        "fields": list(STAGE2_CACHE_FIELDS),
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": _file_sha256(jsonl_path),
        "stage1_ckpt_path": stage1_meta["path"],
        "stage1_ckpt_sha256": stage1_meta["sha256"],
        "created_at": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        "[%s] 完成: n_samples=%d n_shards=%d 耗时=%.0fs manifest=%s",
        jsonl_path.name, n_samples, next_shard_id, time.perf_counter() - t0,
        manifest_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage2 encoder cache 预计算（freeze stage1 后离线一次）",
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个 jsonl（含 alignment_matrix）；每个 jsonl 单独生成一个 cache 子目录",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        required=True,
        help="Stage1 checkpoint，用于跑 binary_encoder / source_encoder 的 forward",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="cache 根目录；每个 jsonl 在其下用 ``<jsonl_stem>`` 子目录存放 manifest+shard",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="预计算 batch 大小（受显存限制；越大吞吐越高，~16 通常稳）",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=4096,
        help="每个 shard 文件的样本数（越大磁盘块越大，越小内存峰值越低；4096 折中）",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker 数",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda / cpu（默认自动）",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="若目标 manifest 已存在则重写；否则跳过",
    )
    ap.add_argument(
        "--no-tf32",
        action="store_true",
        help="关闭 TF32（默认在 CUDA 上开启以加速 matmul/conv）",
    )
    args = ap.parse_args()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    if device.type == "cuda":
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    init_path = args.init_from.expanduser().resolve()
    if not init_path.is_file():
        logger.error("--init-from 不存在: %s", init_path)
        sys.exit(1)

    cfg = StrucLiftConfig()
    model = StrucLift(cfg).to(device)
    try:
        state = torch.load(init_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(init_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info("init_from 未载入键（前 5）: %s", missing[:5])
    if unexpected:
        logger.info("init_from 中未使用键（前 5）: %s", unexpected[:5])

    model.freeze_encoders()
    model.freeze_alignment()
    model.freeze_decoder_adapters()

    stage1_meta = {
        "path": str(init_path),
        "sha256": _file_sha256(init_path),
    }
    logger.info("Stage1 ckpt sha256=%s", stage1_meta["sha256"][:16])

    cache_root = args.cache_dir.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    jsonls: List[Path] = [p.expanduser().resolve() for p in args.jsonl]
    for p in jsonls:
        if not p.is_file():
            logger.error("jsonl 不存在: %s", p)
            sys.exit(1)

    for p in jsonls:
        cache_sub = cache_root / p.stem
        _process_one_jsonl(
            jsonl_path=p,
            cache_subdir=cache_sub,
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            shard_size=int(args.shard_size),
            num_workers=int(args.num_workers),
            stage1_meta=stage1_meta,
            force=bool(args.force),
        )

    logger.info("全部 jsonl 预计算完成: %d 个", len(jsonls))


if __name__ == "__main__":
    main()

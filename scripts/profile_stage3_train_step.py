#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3 **训练一步**细分耗时探针（对齐 ``Stage3Trainer.train_epoch`` 真实路径）。

对每个 batch 统计（秒）：
  01_data          DataLoader 取 batch（CPU，含 collate / 可选 encoder cache 读盘）
  02_h2d           ``move_binskel_batch_to_device``
  03_slot_weight   ``SlotWeightBuilder.build_slot_weights_*``
  04_sched_sampling  scheduled sampling 额外整图前向（仅 ss_p>0）
  05_encode_AB     ``StrucLift.encode(run_alignment=True)``（Module A+B）
  06_graph_kv      ``batched_graph_kv_from_nodes``（若 batch 无 graph_kv）
  07_decoder_LLM   ``decoder(..., labels=...)`` 含 slot-weighted CE
  08_compile_loss  ``L_compile``（``s3_lambda_compile>0`` 时）
  09_backward      ``loss.backward()``（含 GradScaler unscale 前向图释放）
  10_optim_step    ``clip_grad_norm_`` + ``optimizer.step`` (+ scaler.update)

可选：``export STRUCLIFT_PROFILE_ALIGNMENT=1`` 在 05 阶段结束后打印 Module B 内部分项。

示例（与正式训练同卡、同 batch、同 checkpointing）::

  cd /data/chaoni/WQM/model_code/structlift
  unset STRUCLIFT_STAGE3_LOG_STEP_MS
  export STRUCLIFT_ENCODER_BF16=1
  CUDA_VISIBLE_DEVICES=2 python3 scripts/profile_stage3_train_step.py \\
    --jsonl /data/chaoni/WQM/datasets/C_train_o0_deepseek_coder6.7b_p20.jsonl \\
    --init-from /data/chaoni/WQM/checkpoints/stage2_ddp2/best_stage2_curriculum_stage4.pt \\
    --llm /data/chaoni/WQM/models/deepseek-coder-6.7b-instruct \\
    --tokenizer-saved /data/chaoni/WQM/datasets/structlift_tokenizer_deepseek_6.7b_instruct_fill \\
    --batch-size 16 --llm-gradient-checkpointing \\
    --warmup-steps 2 --measure-steps 30 --num-workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    BinSkelJsonlDataset,
    STAGE2_CACHE_MANIFEST_NAME,
    Stage2EncoderCachedDataset,
    Stage2EncoderCache,
    make_binskel_stage3_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.module_c import SlotWeightBuilder
from struclift.models.struclift import StrucLift, batched_graph_kv_from_nodes
from struclift.training.stage3_sft import Stage3Trainer
from struclift.utils.state_dict_compat import load_state_dict_skip_shape_mismatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 与 train_epoch 分段键一致（便于对照日志）
SEG_KEYS = (
    "01_data",
    "02_h2d",
    "03_slot_weight",
    "04_sched_sampling",
    "05_encode_AB",
    "06_graph_kv",
    "07_decoder_LLM",
    "08_compile_loss",
    "09_backward",
    "10_optim_step",
)


def _resolve_llm_path(raw: str) -> str:
    p = Path(raw.strip()).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    return raw.strip()


def _load_autoconfig_module():
    path = SCRIPT_DIR / "train_stage3_binskel_autoconfig.py"
    spec = importlib.util.spec_from_file_location("_t3_autoconfig_prof", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cuda_sync(device: torch.device, *, device_map_auto: bool) -> None:
    if device.type != "cuda":
        return
    if device_map_auto:
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(torch.device("cuda", i))
    else:
        torch.cuda.synchronize(device)


def _precomputed_from_batch(batch: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
    return {
        "precomputed_bin_node_embs": batch.get("cache_bin_node_embs"),
        "precomputed_pattern_logits": batch.get("cache_pattern_logits"),
        "precomputed_bin_graph_emb": batch.get("cache_bin_graph_emb"),
        "precomputed_src_node_embs": batch.get("cache_src_node_embs"),
        "precomputed_src_graph_emb": batch.get("cache_src_graph_emb"),
    }


def _make_stage3_loader(
    paths: List[Path],
    *,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    skip_invalid: bool,
    pin_memory: bool,
    pad_token_id: int,
    max_sft_seq_len: Optional[int],
    fill_token_ids: torch.Tensor,
    encoder_cache_dir: Optional[Path],
    encoder_cache_shard_lru: int,
) -> DataLoader:
    """Stage3 loader；可选挂 ``Stage2EncoderCachedDataset``。"""
    from functools import partial
    from torch.utils.data import ConcatDataset, Dataset

    from struclift.data.binskel_jsonl import (
        collate_binskel_stage3_with_skip,
    )

    cache_root = (
        encoder_cache_dir.expanduser().resolve()
        if encoder_cache_dir is not None
        else None
    )

    datasets: List[Dataset] = []
    for p in paths:
        base = BinSkelJsonlDataset(
            p,
            max_samples=max_samples,
            require_sft=True,
            skip_invalid=skip_invalid,
            max_sft_seq_len=max_sft_seq_len,
        )
        if cache_root is not None:
            cache_sub = cache_root / p.stem
            manifest = cache_sub / STAGE2_CACHE_MANIFEST_NAME
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"encoder cache 不存在: {manifest}；"
                    f"请先 scripts/precompute_stage2_encoder_cache.py",
                )
            cache = Stage2EncoderCache(
                cache_sub, shard_lru_size=int(encoder_cache_shard_lru),
            )
            datasets.append(Stage2EncoderCachedDataset(base, cache))
        else:
            datasets.append(base)

    ds: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    collate_fn = partial(
        collate_binskel_stage3_with_skip,
        pad_token_id=pad_token_id,
        max_sft_seq_len=max_sft_seq_len,
        fill_token_ids=fill_token_ids,
        rebuild_slot_mask_when_missing=True,
    )
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return DataLoader(ds, **kw)


def _acc_add(accum: Dict[str, float], key: str, delta: float) -> None:
    accum[key] = accum.get(key, 0.0) + delta


def _timed_cuda(
    key: str,
    device: torch.device,
    device_map_auto: bool,
    fn: Callable[[], Any],
    accum: Dict[str, float],
) -> Any:
    _cuda_sync(device, device_map_auto=device_map_auto)
    t0 = time.perf_counter()
    out = fn()
    _cuda_sync(device, device_map_auto=device_map_auto)
    accum[key] = accum.get(key, 0.0) + (time.perf_counter() - t0)
    return out


def _profile_one_step(
    *,
    trainer: Stage3Trainer,
    model: StrucLift,
    batch_cpu: Dict[str, Any],
    device: torch.device,
    device_map_auto: bool,
    fill_token_ids: torch.Tensor,
    step_in_epoch: int,
    steps_per_epoch: int,
    force_ss_p: Optional[float],
    accum: Dict[str, float],
) -> None:
    """跑一个与 ``train_epoch`` 等价的训练 step 并累加各段 GPU 同步后耗时。"""
    cfg = trainer.config
    fp16 = trainer.fp16
    scaler = trainer.scaler
    optimizer = trainer.optimizer
    slot_builder = trainer.slot_weight_builder

    t0 = time.perf_counter()
    batch = move_binskel_batch_to_device(batch_cpu, device)
    _cuda_sync(device, device_map_auto=device_map_auto)
    _acc_add(accum, "02_h2d", time.perf_counter() - t0)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    t0 = time.perf_counter()
    slot_weight_mask = slot_builder.build_slot_weights_batched_for_supervised_labels(
        input_ids,
        labels,
        batch["slot_types"],
        cfg.module_c,
        fill_token_ids=fill_token_ids,
    )
    _cuda_sync(device, device_map_auto=device_map_auto)
    _acc_add(accum, "03_slot_weight", time.perf_counter() - t0)

    spe = max(1, int(steps_per_epoch))
    sie = max(0, min(int(step_in_epoch), spe - 1))
    if force_ss_p is not None:
        ss_p = float(force_ss_p)
    else:
        ss_p = trainer._scheduled_sampling_prob_for_step(sie, spe)

    if ss_p > 0.0 and model.decoder._llm is not None:
        def _ss() -> torch.Tensor:
            return trainer._apply_scheduled_sampling(
                input_ids, labels, batch, slot_weight_mask, ss_p,
            )

        input_ids = _timed_cuda(
            "04_sched_sampling", device, device_map_auto, _ss, accum,
        )

    precomp = _precomputed_from_batch(batch)
    bb = batch.get("bin_batch")

    def _encode() -> Any:
        with torch.amp.autocast("cuda", enabled=fp16):
            return model.encode(
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
                **precomp,
            )

    enc_out = _timed_cuda("05_encode_AB", device, device_map_auto, _encode, accum)

    try:
        from struclift.models.module_b import report_alignment_profile

        ab = report_alignment_profile()
        if ab:
            logger.info(
                "  [alignment 细分] %s",
                " | ".join(f"{k}={v:.3f}s" for k, v in sorted(ab.items())),
            )
    except ImportError:
        pass

    graph_kv = batch.get("graph_kv")

    def _pack_kv() -> torch.Tensor:
        nonlocal graph_kv, bb
        if graph_kv is not None:
            return graph_kv
        if bb is None:
            n_n = enc_out.binary_node_embs.size(0)
            bb = torch.zeros(
                n_n, dtype=torch.long, device=enc_out.binary_node_embs.device,
            )
        return batched_graph_kv_from_nodes(
            enc_out.binary_node_embs, bb, batch_size=int(input_ids.size(0)),
        )

    if graph_kv is None:
        graph_kv = _timed_cuda("06_graph_kv", device, device_map_auto, _pack_kv, accum)
    else:
        _acc_add(accum, "06_graph_kv", 0.0)

    slot_mask = batch.get("slot_mask")
    if slot_mask is None:
        slot_mask = torch.ones(
            input_ids.size(0),
            input_ids.size(1),
            graph_kv.size(1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    logits_holder: Dict[str, Any] = {}
    loss_holder: Dict[str, Any] = {}

    def _decoder() -> None:
        with torch.amp.autocast("cuda", enabled=fp16):
            logits, lm_loss = model.decoder(
                input_ids=input_ids,
                graph_kv=graph_kv,
                slot_mask=slot_mask,
                labels=labels,
                slot_weight_mask=slot_weight_mask,
            )
        logits_holder["logits"] = logits
        loss_holder["lm_loss"] = lm_loss

    _timed_cuda("07_decoder_LLM", device, device_map_auto, _decoder, accum)

    loss = loss_holder.get("lm_loss")
    if loss is None:
        return

    if trainer.lambda_compile > 0:
        out_stub = type("Out", (), {"logits": logits_holder["logits"]})()

        def _compile() -> Optional[torch.Tensor]:
            return trainer._compute_compile_loss(
                out_stub, input_ids, labels, slot_weight_mask,
            )

        compile_loss = _timed_cuda(
            "08_compile_loss", device, device_map_auto, _compile, accum,
        )
        if compile_loss is not None:
            loss = loss + trainer.lambda_compile * compile_loss

    def _bwd() -> None:
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    _timed_cuda("09_backward", device, device_map_auto, _bwd, accum)

    def _opt() -> None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

    _timed_cuda("10_optim_step", device, device_map_auto, _opt, accum)


def _summary_rows(
    records: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for k in SEG_KEYS:
        vals = [r[k] for r in records if k in r]
        if not vals:
            out[k] = (0.0, 0.0)
        else:
            out[k] = (float(statistics.mean(vals)), float(statistics.median(vals)))
    tot_vals = [sum(r.get(k, 0.0) for k in SEG_KEYS) for r in records]
    out["total_synced"] = (
        float(statistics.mean(tot_vals)) if tot_vals else 0.0,
        float(statistics.median(tot_vals)) if tot_vals else 0.0,
    )
    return out


def _print_report(
    summary: Dict[str, Tuple[float, float]],
    *,
    n_steps: int,
    batch_size: int,
    steps_per_epoch: int,
    meta: Dict[str, Any],
) -> None:
    mean_total = summary["total_synced"][0]
    logger.info("")
    logger.info("=" * 88)
    logger.info(
        "Stage3 训练步细分 | measure_steps=%d batch_size=%d | mean total(同步分段和)=%.4fs",
        n_steps,
        batch_size,
        mean_total,
    )
    logger.info(
        "配置: llm_gc=%s fp16=%s lambda_compile=%s encoder_cache=%s STRUCLIFT_ENCODER_BF16=%s",
        meta.get("llm_gc"),
        meta.get("fp16"),
        meta.get("lambda_compile"),
        meta.get("encoder_cache"),
        os.environ.get("STRUCLIFT_ENCODER_BF16", ""),
    )
    logger.info("-" * 88)
    logger.info("%-22s %10s %10s %10s", "阶段", "mean(s)", "median(s)", "占total%")
    logger.info("-" * 88)
    for k in SEG_KEYS:
        m, med = summary[k]
        pct = 100.0 * m / mean_total if mean_total > 1e-9 else 0.0
        logger.info("%-22s %10.4f %10.4f %9.1f%%", k, m, med, pct)
    logger.info("-" * 88)
    logger.info("%-22s %10.4f %10.4f", "total_synced(求和)", mean_total, summary["total_synced"][1])

  # epoch 粗估
    if steps_per_epoch > 0 and mean_total > 0:
        epoch_h = steps_per_epoch * mean_total / 3600.0
        logger.info(
            "粗估 train 一轮: steps/epoch≈%d → %.2f h（仅同步分段和，未含 01_data 里全部主机等待）",
            steps_per_epoch,
            epoch_h,
        )
    logger.info("=" * 88)

    # 提速建议（启发式）
    tips: List[str] = []
    if mean_total < 1e-9:
        return
    ranked = sorted(
        ((k, summary[k][0]) for k in SEG_KEYS),
        key=lambda x: -x[1],
    )
    top_key, top_s = ranked[0]
    top_pct = 100.0 * top_s / mean_total
    if top_key == "05_encode_AB" and top_pct > 18:
        tips.append(
            "Module A+B 占比高：预计算 encoder cache（--encoder-cache-dir）+ "
            "STRUCLIFT_ENCODER_BF16=1；Stage3 训练入口需接 cache_*（正式训练尚未接线）。",
        )
    if top_key == "07_decoder_LLM" and top_pct > 35:
        tips.append(
            "LLM+decoder 主导：双卡 --llm-device-map-auto、保持 gradient checkpointing、"
            "确认无 STRUCLIFT_STAGE3_LOG_STEP_MS。",
        )
    if summary["08_compile_loss"][0] / mean_total > 0.08:
        tips.append(
            "L_compile（含 gcc）占比明显：属提案损失一部分，不能删；"
            "可检查是否误用 reinforce 模式（更重）。",
        )
    if summary["01_data"][0] / mean_total > 0.12:
        tips.append("DataLoader 偏慢：增大 --num-workers、cache 放 NVMe、提高 shard_lru。")
    if summary["04_sched_sampling"][0] / mean_total > 0.05:
        tips.append("scheduled sampling 额外前向：训练后 40%% 步会变慢，属算法设定。")
    if tips:
        logger.info("--- 提速方向（不改损失/算法前提下）---")
        for i, t in enumerate(tips, 1):
            logger.info("  %d. %s", i, t)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage3 训练一步各阶段耗时与占比（对齐 Stage3Trainer）",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path, nargs="+")
    src.add_argument("--jsonl-list", type=Path, metavar="PATH.txt")
    ap.add_argument("--init-from", type=Path, required=True)
    ap.add_argument("--llm", type=str, required=True)
    ap.add_argument("--tokenizer-saved", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--measure-steps", type=int, default=20)
    ap.add_argument("--max-sft-seq-len", type=int, default=2048)
    ap.add_argument("--llm-gradient-checkpointing", action="store_true")
    ap.add_argument("--llm-device-map-auto", action="store_true")
    ap.add_argument("--llm-per-gpu-max-gib", type=float, default=None)
    ap.add_argument(
        "--encoder-cache-dir",
        type=Path,
        default=None,
        help="可选：Stage2 encoder cache 根目录（须已 precompute）",
    )
    ap.add_argument("--encoder-cache-shard-lru", type=int, default=4)
    ap.add_argument(
        "--force-ss-p",
        type=float,
        default=None,
        metavar="P",
        help="强制 scheduled sampling 概率（测 ss 开销；默认按 global_step 计算）",
    )
    ap.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="用于粗估一轮小时数；0=按数据集长度与 batch-size 推断",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="写出 JSON 汇总")
    ap.add_argument("--strict-jsonl", action="store_true")
    args = ap.parse_args()

    t3m = _load_autoconfig_module()

    if args.device:
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.llm_device_map_auto and device.type != "cuda":
        logger.error("--llm-device-map-auto 需要 CUDA")
        sys.exit(1)

    cfg = StrucLiftConfig()
    from transformers import AutoConfig, AutoTokenizer

    llm_id = _resolve_llm_path(args.llm)
    hf_cfg = AutoConfig.from_pretrained(llm_id, trust_remote_code=True)
    t3m._apply_config_dims(cfg, hf_cfg, None, None)
    cfg.training.s3_batch_size = args.batch_size
    if args.max_sft_seq_len <= 0:
        cfg.training.s3_max_sft_seq_len = None
    else:
        cfg.training.s3_max_sft_seq_len = int(args.max_sft_seq_len)
    if args.llm_gradient_checkpointing:
        cfg.training.llm_gradient_checkpointing = True

    tok_path = (
        str(args.tokenizer_saved.expanduser().resolve())
        if args.tokenizer_saved is not None
        else llm_id
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    mft = int(cfg.module_c.max_fill_tokens)
    needed = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(mft)]
    missing = [t for t in needed if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_tokens(missing)
    fill_token_ids_cpu = torch.tensor(
        [tokenizer.convert_tokens_to_ids(t) for t in needed], dtype=torch.long,
    )
    target_vs = len(tokenizer)
    pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)

    if args.jsonl is not None:
        paths = [p.expanduser().resolve() for p in args.jsonl]
    else:
        lines = [
            ln.strip()
            for ln in args.jsonl_list.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        paths = [Path(ln).expanduser().resolve() for ln in lines]

    if args.encoder_cache_dir is not None:
        loader = _make_stage3_loader(
            paths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=not args.strict_jsonl,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=cfg.training.s3_max_sft_seq_len,
            fill_token_ids=fill_token_ids_cpu,
            encoder_cache_dir=args.encoder_cache_dir,
            encoder_cache_shard_lru=args.encoder_cache_shard_lru,
        )
    else:
        loader = make_binskel_stage3_dataloader(
            paths,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=not args.strict_jsonl,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=cfg.training.s3_max_sft_seq_len,
            fill_token_ids=fill_token_ids_cpu,
        )

    n_dataset = len(loader.dataset)
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0 and n_dataset > 0:
        steps_per_epoch = (n_dataset + args.batch_size - 1) // args.batch_size

    try:
        state = torch.load(args.init_from, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.init_from, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model = StrucLift(cfg)
    load_state_dict_skip_shape_mismatch(model, state, log=logger)

    mmap = bool(args.llm_device_map_auto)
    batch_anchor = torch.device("cuda", 0) if mmap else device

    if mmap:
        t3m._move_struclift_encoders_alignment_to_cuda0(model)
        llm = t3m._load_hf_llm_autoconfig(
            cfg,
            llm_id,
            device,
            target_vocab_size=target_vs,
            device_map_auto=True,
            llm_max_memory_per_gpu_gib=args.llm_per_gpu_max_gib,
        )
    else:
        model = model.to(device)
        llm = t3m._load_hf_llm_autoconfig(
            cfg, llm_id, device, target_vocab_size=target_vs,
        )

    model.decoder.inject_into_llm(llm)
    if mmap:
        t3m._align_decoder_adapters_with_llm_shards(model.decoder)
    if args.llm_gradient_checkpointing:
        t3m._maybe_enable_llm_gradient_checkpoint(model, cfg)

    trainer = Stage3Trainer(
        model,
        cfg,
        device=batch_anchor,
        fill_token_ids=fill_token_ids_cpu.to(batch_anchor),
        tokenizer=tokenizer,
        decoder_device_map_auto=mmap,
    )
    fill_ids_dev = fill_token_ids_cpu.to(batch_anchor)

    total_train_steps = max(steps_per_epoch * int(cfg.training.s3_epochs), 1)
    trainer.total_train_steps = total_train_steps

    accum: Dict[str, float] = defaultdict(float)
    per_step_records: List[Dict[str, float]] = []
    loader_iter = iter(loader)
    steps_per_epoch = max(1, len(loader))
    global_step = 0
    measured = 0
    warmup = max(0, args.warmup_steps)
    need = warmup + args.measure_steps

    logger.info(
        "开始 profile: warmup=%d measure=%d dataset=%d batch=%d steps/epoch≈%d",
        warmup, args.measure_steps, n_dataset, args.batch_size, steps_per_epoch,
    )
    if os.environ.get("STRUCLIFT_PROFILE_ALIGNMENT", "").strip() == "1":
        logger.info("STRUCLIFT_PROFILE_ALIGNMENT=1 → 每步 05 后打印 Module B 细分")

    while measured < args.measure_steps:
        t_data = time.perf_counter()
        try:
            batch_cpu = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_cpu = next(loader_iter)
        t_data_elapsed = time.perf_counter() - t_data

        step_accum: Dict[str, float] = {}
        step_accum["01_data"] = t_data_elapsed

        is_warmup = global_step < warmup
        tag = "warmup" if is_warmup else "measure"

        _profile_one_step(
            trainer=trainer,
            model=model,
            batch_cpu=batch_cpu,
            device=batch_anchor,
            device_map_auto=mmap,
            fill_token_ids=fill_ids_dev,
            step_in_epoch=global_step % steps_per_epoch,
            steps_per_epoch=steps_per_epoch,
            force_ss_p=args.force_ss_p,
            accum=step_accum,
        )

        step_total = sum(step_accum.get(k, 0.0) for k in SEG_KEYS)
        logger.info(
            "[%s step %d] total=%.3fs | 05=%.3fs 07=%.3fs 09=%.3fs | data=%.3fs",
            tag,
            global_step,
            step_total,
            step_accum.get("05_encode_AB", 0.0),
            step_accum.get("07_decoder_LLM", 0.0),
            step_accum.get("09_backward", 0.0),
            step_accum.get("01_data", 0.0),
        )

        if not is_warmup:
            for k, v in step_accum.items():
                _acc_add(accum, k, v)
            per_step_records.append(dict(step_accum))
            measured += 1

        global_step += 1

    summary = _summary_rows(per_step_records)
    meta = {
        "llm_gc": args.llm_gradient_checkpointing,
        "fp16": trainer.fp16,
        "lambda_compile": cfg.training.s3_lambda_compile,
        "encoder_cache": str(args.encoder_cache_dir) if args.encoder_cache_dir else None,
        "device_map_auto": mmap,
        "batch_size": args.batch_size,
        "measure_steps": args.measure_steps,
        "steps_per_epoch": steps_per_epoch,
    }
    _print_report(
        summary,
        n_steps=len(per_step_records),
        batch_size=args.batch_size,
        steps_per_epoch=steps_per_epoch,
        meta=meta,
    )

    if args.out_json is not None:
        payload = {
            "meta": meta,
            "segments": SEG_KEYS,
            "summary_mean": {k: summary[k][0] for k in SEG_KEYS},
            "summary_median": {k: summary[k][1] for k in SEG_KEYS},
            "total_synced_mean": summary["total_synced"][0],
            "per_step": per_step_records,
        }
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("已写入 %s", args.out_json)


if __name__ == "__main__":
    main()

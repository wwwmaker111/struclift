#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 训练：从 binskel ``.jsonl`` 读入，使用 :mod:`struclift.data.binskel_jsonl` 组 batch。

示例::

  python scripts/train_stage1_binskel.py --jsonl E:/structlift_datasets/binskel_busybox_o0.jsonl
  python scripts/train_stage1_binskel.py --jsonl a.jsonl b.jsonl --epochs 10 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    describe_binskel_batch,
    make_binskel_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.training.stage1_pretrain import Stage1Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage1 预训练（binskel jsonl）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        help="一个或多个 binskel .jsonl（多文件拼接为同一数据集）",
    )
    src.add_argument(
        "--jsonl-list",
        type=Path,
        metavar="PATH.txt",
        help="文本文件：每行一个 jsonl 路径（# 开头为注释）；用于 8:1:1 划分后的 train_jsonl_paths.txt 等",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints/stage1"),
        help="checkpoint 目录",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda / cpu（默认自动）",
    )
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 config.training.s1_batch_size")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config.training.s1_epochs")
    ap.add_argument("--lambda-graph", type=float, default=None, help="覆盖 config.training.s1_lambda_graph_align")
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="覆盖 config.training.s1_early_stop_patience（0=关闭早停）",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="覆盖 config.training.s1_early_stop_min_delta",
    )
    ap.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=None,
        help="覆盖 config.training.s1_early_stop_min_epochs",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="最多使用多少条有效样本（调试）")
    ap.add_argument(
        "--no-source",
        action="store_true",
        help="不要求源码侧（仅训二进制；不启用图级 bin–src 对比）",
    )
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader 进程数（建议 4~16）")
    ap.add_argument("--prefetch-factor", type=int, default=4, help="每个 worker 预取 batch 数")
    ap.add_argument("--ddp", action="store_true", help="启用 DistributedDataParallel（配合 torchrun）")
    ap.add_argument("--no-tf32", action="store_true", help="关闭 TF32（默认开启以提速）")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只加载一个 batch、打印张量形状并做一次前向（不训练、不写 checkpoint）",
    )
    ap.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="索引阶段遇坏行即失败（默认跳过坏行）",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="可选：从已有 checkpoint（如 latest_stage1.pt / best_stage1.pt）恢复编码器权重后继续训练",
    )
    ap.add_argument(
        "--inst-dropout",
        type=float,
        default=None,
        help="§9.2 指令 dropout 概率（0 关闭；建议 0.15）——每个 batch 随机将该比例指令置 PAD",
    )
    ap.add_argument(
        "--edge-dropout",
        type=float,
        default=None,
        help="§9.2 CFG 边 dropout 概率（0 关闭；建议 0.10）——每个 batch 随机删该比例 CFG 边",
    )
    ap.add_argument(
        "--no-pseudo-neg-mask",
        action="store_true",
        help="关闭 §9.2 伪负过滤（默认开启：batch 内同函数不同编译不作为负样本）",
    )
    ap.add_argument(
        "--cross-opt-pairs",
        action="store_true",
        help=(
            "启用 §9.2 跨优化显式正对：Dataset.__getitem__ 返回 [anchor, companion]，"
            "同函数不同编译作为正样本（multi-positive InfoNCE）。开启后 effective "
            "batch ≈ 2 × --batch-size；建议把 --batch-size 设为目标的一半。"
        ),
    )
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(args.ddp or world_size > 1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_s)

    cfg = StrucLiftConfig()
    if args.batch_size is not None:
        cfg.training.s1_batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.s1_epochs = args.epochs
    if args.lambda_graph is not None:
        cfg.training.s1_lambda_graph_align = args.lambda_graph
    if args.early_stop_patience is not None:
        cfg.training.s1_early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg.training.s1_early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_min_epochs is not None:
        cfg.training.s1_early_stop_min_epochs = args.early_stop_min_epochs
    if args.inst_dropout is not None:
        cfg.training.s1_inst_dropout = float(args.inst_dropout)
    if args.edge_dropout is not None:
        cfg.training.s1_edge_dropout = float(args.edge_dropout)
    if args.no_pseudo_neg_mask:
        cfg.training.s1_pseudo_neg_mask = False
    if args.cross_opt_pairs:
        cfg.training.s1_cross_opt_pairs = True

    if device.type == "cuda":
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    require_source = not args.no_source
    skip_invalid = not args.strict_jsonl

    if args.jsonl_list is not None:
        list_path = args.jsonl_list.expanduser().resolve()
        raw = list_path.read_text(encoding="utf-8")
        paths: list[Path] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line).expanduser().resolve())
        if not paths:
            logger.error("--jsonl-list 无有效路径: %s", list_path)
            sys.exit(1)
        jsonl_arg: object = paths[0] if len(paths) == 1 else paths
    else:
        assert args.jsonl is not None
        jsonl_arg = (
            args.jsonl[0].expanduser().resolve()
            if len(args.jsonl) == 1
            else [p.expanduser().resolve() for p in args.jsonl]
        )
    loader = make_binskel_dataloader(
        jsonl_arg,
        batch_size=cfg.training.s1_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_samples=args.max_samples,
        require_source=require_source,
        skip_invalid=skip_invalid,
        pin_memory=device.type == "cuda",
        distributed=use_ddp,
        rank=rank,
        world_size=world_size,
        inst_dropout_p=cfg.training.s1_inst_dropout,
        edge_dropout_p=cfg.training.s1_edge_dropout,
        # 开启跨优化正对时 func_uid 必须存在（multi-positive InfoNCE 依赖它）
        add_func_uid=cfg.training.s1_pseudo_neg_mask or cfg.training.s1_cross_opt_pairs,
        cross_opt_pairs=cfg.training.s1_cross_opt_pairs,
    )
    if len(loader.dataset) == 0:
        logger.error("无有效样本：检查 jsonl 或去掉 --no-source / 放宽数据")
        sys.exit(1)

    if args.dry_run:
        batch = next(iter(loader))
        if rank == 0:
            logger.info("dry-run batch:\n%s", describe_binskel_batch(batch))
        model = StrucLift(cfg).to(device)
        batch = move_binskel_batch_to_device(batch, device, non_blocking=True)
        with torch.no_grad():
            enc = model.encode(
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
        if rank == 0:
            logger.info(
                "dry-run encode OK: pattern_logits=%s binary_graph_emb=%s source_graph_emb=%s",
                tuple(enc.pattern_logits.shape),
                None if enc.binary_graph_emb is None else tuple(enc.binary_graph_emb.shape),
                None if enc.source_graph_emb is None else tuple(enc.source_graph_emb.shape),
            )
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()
        return

    model = StrucLift(cfg).to(device)
    # Stage1 仅训练 Module A（binary/source encoders）；冻结 B/C 避免 DDP unused-params 报错。
    model.freeze_alignment()
    model.freeze_decoder_adapters()
    if args.resume is not None:
        ckpt_path = args.resume.expanduser().resolve()
        if not ckpt_path.is_file():
            logger.error("--resume 路径不存在: %s", ckpt_path)
            sys.exit(1)
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            logger.info(
                "已从 %s 恢复权重（missing=%d unexpected=%d）",
                ckpt_path,
                len(missing),
                len(unexpected),
            )
    if use_ddp:
        # 同 scripts/train_stage1_binskel.py：必须 find_unused_parameters=True，
        # 否则 graph_align/region 有条件分支会导致 rank 间参数使用不一致，
        # DDP Reducer 桶死锁 → NCCL watchdog timeout。
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    trainer = Stage1Trainer(
        model,
        cfg,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        logger.info(
            "开始 Stage1: 样本数=%d batch_size(per-rank)=%d epochs=%d require_source=%s device=%s world_size=%d",
            len(loader.dataset),
            cfg.training.s1_batch_size,
            cfg.training.s1_epochs,
            require_source,
            device,
            world_size,
        )
        logger.info(
            "§9.2 增强: inst_dropout=%g edge_dropout=%g pseudo_neg_mask=%s cross_opt_pairs=%s",
            cfg.training.s1_inst_dropout,
            cfg.training.s1_edge_dropout,
            cfg.training.s1_pseudo_neg_mask,
            cfg.training.s1_cross_opt_pairs,
        )
        logger.info(
            "早停配置: patience=%d min_delta=%g min_epochs=%d",
            cfg.training.s1_early_stop_patience,
            cfg.training.s1_early_stop_min_delta,
            cfg.training.s1_early_stop_min_epochs,
        )
    trainer.train(loader, save_dir=str(args.save_dir.expanduser().resolve()))
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

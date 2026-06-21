#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 训练：从 binskel ``.jsonl`` 读入，使用 :mod:`struclift.data.binskel_jsonl` 组 batch。

示例::

  python scripts/train_stage1_binskel.py --jsonl E:/structlift_datasets/binskel_busybox_o0.jsonl
  python scripts/train_stage1_binskel.py --jsonl a.jsonl b.jsonl --epochs 10 --batch-size 8

  # 带验证集：``best``/早停按 **val total loss**；从旧 checkpoint 续训并改用验证集时请加
  # ``--reset-selection-baseline``（此前 ``best_loss`` 多为训集，不能与 val 直接比）::

  python scripts/train_stage1_binskel.py --jsonl train.jsonl --val-jsonl val.jsonl --resume checkpoints/stage1/latest_stage1.pt --reset-selection-baseline
"""

from __future__ import annotations

import argparse
import datetime as _dt
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
        "--val-jsonl",
        type=Path,
        nargs="+",
        default=None,
        help="可选：验证集 binskel jsonl（一个或多个文件，与训练集应无重叠样本）",
    )
    ap.add_argument(
        "--val-jsonl-list",
        type=Path,
        default=None,
        metavar="PATH.txt",
        help="可选：验证集路径列表（每行一个 jsonl，同 --jsonl-list）",
    )
    ap.add_argument(
        "--reset-selection-baseline",
        action="store_true",
        help=(
            "与 --val-jsonl* 联用：续训时把早停/ best 的基线重置为 inf（"
            "此前 checkpoint 的 best_loss 是按训练集记的，不能与验证集 total 直接比）"
        ),
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
    ap.add_argument(
        "--hard-neg-refresh-every",
        type=int,
        default=None,
        help=(
            "§9.2 硬负样本挖掘刷新周期（epoch），0=关闭；建议 3-5。启用后每 N epoch "
            "对全数据集做一次 forward 建 memory bank，为每个 anchor 预计算 Top-K 最难负样本。"
            "只支持单 JSONL 文件（不支持 --jsonl a.jsonl b.jsonl）。"
        ),
    )
    ap.add_argument(
        "--hard-neg-topk",
        type=int,
        default=None,
        help="§9.2 硬负 Top-K（默认 32；建议 16-64）",
    )
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(args.ddp or world_size > 1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and not dist.is_initialized():
        # NOTE: 默认 ProcessGroup watchdog timeout = 10 min，bank refresh 在 resume
        # 首轮（cudnn autotune 未热身）可能需要 20–40 min 才能完成，两个 rank 进度
        # 差几分钟 NCCL 默认就会崩。这里显式放到 4 小时，覆盖 bank refresh + 冷启动
        # + 数据 IO 所有慢路径。注意：NCCL_TIMEOUT 环境变量对 PyTorch DDP 不生效，
        # 必须在 init_process_group 里传 timeout=。
        _pg_timeout = _dt.timedelta(
            seconds=int(os.environ.get("PG_TIMEOUT_SECONDS", "14400")),
        )
        dist.init_process_group(backend="nccl", timeout=_pg_timeout)
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
    if args.hard_neg_refresh_every is not None:
        cfg.training.s1_hard_neg_refresh_every = int(args.hard_neg_refresh_every)
    if args.hard_neg_topk is not None:
        cfg.training.s1_hard_neg_topk = int(args.hard_neg_topk)

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

    if args.val_jsonl_list is not None and args.val_jsonl is not None:
        logger.error("不要同时使用 --val-jsonl 与 --val-jsonl-list")
        sys.exit(1)
    val_jsonl_arg: object | None = None
    if args.val_jsonl_list is not None:
        vlist_path = args.val_jsonl_list.expanduser().resolve()
        vraw = vlist_path.read_text(encoding="utf-8")
        vpaths: list[Path] = []
        for line in vraw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vpaths.append(Path(line).expanduser().resolve())
        if not vpaths:
            logger.error("--val-jsonl-list 无有效路径: %s", vlist_path)
            sys.exit(1)
        val_jsonl_arg = vpaths[0] if len(vpaths) == 1 else vpaths
    elif args.val_jsonl is not None:
        val_jsonl_arg = (
            args.val_jsonl[0].expanduser().resolve()
            if len(args.val_jsonl) == 1
            else [p.expanduser().resolve() for p in args.val_jsonl]
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

    val_loader = None
    if val_jsonl_arg is not None:
        val_loader = make_binskel_dataloader(
            val_jsonl_arg,
            batch_size=cfg.training.s1_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            max_samples=None,
            require_source=require_source,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            distributed=use_ddp,
            rank=rank,
            world_size=world_size,
            inst_dropout_p=0.0,
            edge_dropout_p=0.0,
            add_func_uid=cfg.training.s1_pseudo_neg_mask
            or cfg.training.s1_cross_opt_pairs,
            cross_opt_pairs=False,
        )
        if len(val_loader.dataset) == 0:
            logger.error("验证集无有效样本：检查 val jsonl 或 --no-source")
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
    # NOTE: 真正的 resume 推迟到 trainer 创建之后（load_full_checkpoint 需要 optimizer
    # 已经构造完成；旧版直接 load 权重会丢失优化器动量和早停计数）
    if args.resume is not None:
        resume_path_preview = args.resume.expanduser().resolve()
        if not resume_path_preview.is_file():
            logger.error("--resume 路径不存在: %s", resume_path_preview)
            sys.exit(1)
    if use_ddp:
        # NOTE: 必须 find_unused_parameters=True。train_epoch 里 loss_graph_align
        # 在 source_graph_emb.numel()==0 时会跳过；_compute_region_infonce 在
        # pos_mask.sum()<=0 或样本过滤后为空时会返回 None。这些条件分支会导致
        # 不同 rank 同一 step 参与 backward 的参数集合不同，DDP Reducer 桶在
        # 默认 find_unused_parameters=False 下会死等不来的 all-reduce，触发
        # NCCL watchdog timeout（表现为 bank refresh 后 10 分钟内崩）。
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

    # 续训信息：如果 --resume，trainer 会加载模型 + 优化器 + 早停计数（新格式），
    # 或仅加载权重（legacy 格式）。
    resume_info = {
        "start_epoch": 1,
        "best_loss": float("inf"),
        "stalled": 0,
    }
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        info = trainer.load_full_checkpoint(str(resume_path))
        resume_info["start_epoch"] = int(info["start_epoch"])
        resume_info["best_loss"] = float(info["best_loss"])
        resume_info["stalled"] = int(info["stalled"])
        if val_loader is not None and args.reset_selection_baseline:
            resume_info["best_loss"] = float("inf")
            resume_info["stalled"] = 0
            if rank == 0:
                logger.info(
                    "[Stage1] 已按 --reset-selection-baseline 将早停/ best 基线重置（改用验证集 total）",
                )
        if rank == 0:
            logger.info(
                "已从 %s 恢复（format=%s start_epoch=%d best_loss=%.6f stalled=%d "
                "missing=%d unexpected=%d）",
                resume_path,
                "legacy" if info["is_legacy"] else "full",
                resume_info["start_epoch"],
                resume_info["best_loss"],
                resume_info["stalled"],
                info["missing"],
                info["unexpected"],
            )

    # §9.2 硬负挖掘：若启用，挂载 bank + refresh loader 工厂
    if cfg.training.s1_hard_neg_refresh_every > 0:
        if isinstance(jsonl_arg, list):
            if rank == 0:
                logger.error(
                    "--hard-neg-refresh-every 仅支持单 JSONL；当前为多文件拼接（不支持）",
                )
            if use_ddp and dist.is_initialized():
                dist.destroy_process_group()
            sys.exit(1)
        hn_bs = max(cfg.training.s1_batch_size, 64)  # refresh 走 larger batch 加速
        single_jsonl = jsonl_arg

        def _refresh_factory() -> "torch.utils.data.DataLoader":
            return make_binskel_dataloader(
                single_jsonl,
                batch_size=hn_bs,
                shuffle=False,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                max_samples=args.max_samples,
                require_source=require_source,
                skip_invalid=skip_invalid,
                pin_memory=device.type == "cuda",
                distributed=use_ddp,
                rank=rank,
                world_size=world_size,
                inst_dropout_p=0.0,         # refresh 阶段不做扰动
                edge_dropout_p=0.0,
                add_func_uid=True,          # bank 需要 func_uid 做同函数 mask
                cross_opt_pairs=False,      # refresh 不组对
            )

        trainer.attach_hard_negative_bank(
            num_samples=len(loader.dataset),
            d_model=cfg.module_a.d_model,
            refresh_loader_factory=_refresh_factory,
        )

    if args.reset_selection_baseline and val_loader is None and rank == 0:
        logger.warning(
            "[Stage1] 已忽略 --reset-selection-baseline（未提供 --val-jsonl / --val-jsonl-list）",
        )
    if rank == 0:
        logger.info(
            "开始 Stage1: 训练样本数=%d val样本数=%s batch_size(per-rank)=%d epochs=%d "
            "require_source=%s device=%s world_size=%d",
            len(loader.dataset),
            (len(val_loader.dataset) if val_loader is not None else None),
            cfg.training.s1_batch_size,
            cfg.training.s1_epochs,
            require_source,
            device,
            world_size,
        )
        logger.info(
            "§9.2 增强: inst_dropout=%g edge_dropout=%g pseudo_neg_mask=%s cross_opt_pairs=%s "
            "hard_neg_refresh_every=%d hard_neg_topk=%d",
            cfg.training.s1_inst_dropout,
            cfg.training.s1_edge_dropout,
            cfg.training.s1_pseudo_neg_mask,
            cfg.training.s1_cross_opt_pairs,
            cfg.training.s1_hard_neg_refresh_every,
            cfg.training.s1_hard_neg_topk,
        )
        logger.info(
            "早停配置: patience=%d min_delta=%g min_epochs=%d",
            cfg.training.s1_early_stop_patience,
            cfg.training.s1_early_stop_min_delta,
            cfg.training.s1_early_stop_min_epochs,
        )
    trainer.train(
        loader,
        val_loader=val_loader,
        save_dir=str(args.save_dir.expanduser().resolve()),
        start_epoch=resume_info["start_epoch"],
        best_loss=resume_info["best_loss"],
        stalled=resume_info["stalled"],
    )
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

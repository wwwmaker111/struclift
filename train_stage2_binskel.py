#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 训练：binskel jsonl 须含 ``alignment_matrix``（及 ``alignment_confidence`` 可选）。

与提案 §6.6 一致（另可选 λ₅ L_src_count，§6.5.3）；Module A 冻结。

§9.3 可选：
  - ``--curriculum`` + ``--jsonl-o0`` … ``--jsonl-o3`` (+ 可选 ``--jsonl-os``)：
    按档 **O0→O1→O2→O3→Os** 递进；若不提供 ``--jsonl-os`` 则自动跳过 Os 档；
  - 节点级硬负：默认开启，``--node-hard-neg-interval``（默认 5，0=每 epoch）。

示例::

  python scripts/train_stage2_binskel.py \\
    --jsonl E:/structlift_datasets/binskel_busybox_o0.jsonl \\
    --init-from checkpoints/stage1/best_stage1.pt

  python scripts/train_stage2_binskel.py --curriculum \\
    --jsonl-o0 data/proj_o0.jsonl --jsonl-o1 data/proj_o1.jsonl \\
    --jsonl-o2 data/proj_o2.jsonl --jsonl-o3 data/proj_o3.jsonl \\
    --jsonl-os data/proj_os.jsonl \\
    --curriculum-epochs-per-stage 8 \\
    --init-from checkpoints/stage1/best_stage1.pt

  带验证集（每轮打 train/val，以 val total 选 ``best_stage2.pt``）::

    python scripts/train_stage2_binskel.py --jsonl train.jsonl --val-jsonl val.jsonl \\
      --init-from checkpoints/stage1/best_stage1.pt

  课程 + 分档验证（各 stage 用对应并集 val，O0|O0+O1|…+Os 顺序传 5 个）::

    python scripts/train_stage2_binskel.py --curriculum --jsonl-o0 ... --jsonl-os ... \\
      --val-jsonl-per-curriculum-stage B_val_o0.jsonl B_val_o01.jsonl B_val_o012.jsonl \\
      B_val_o0123.jsonl B_val_o0123s.jsonl --init-from ...

  多卡（``torchrun`` 调 ``WORLD_SIZE``；``--batch-size`` 为**每卡**；alignment 梯度在训练器内 ``all_reduce``，**未使用** ``DistributedDataParallel``，因 encode 多图会多次调 alignment）::

    CUDA_VISIBLE_DEVICES=0,7 torchrun --standalone --nproc_per_node=2 scripts/train_stage2_binskel.py \\
      --curriculum --jsonl-o0 data/o0.jsonl ... --init-from ... --save-dir ... \\
      --batch-size 64
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
from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    Stage2CurriculumDataset,
    build_stage2_train_loader,
    describe_binskel_batch,
    make_binskel_stage2_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.training.stage2_alignment import Stage2Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage2 对齐训练（含 M_gt 的 binskel jsonl）")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        help="一个或多个含 alignment_matrix 的 binskel .jsonl（与 --curriculum 二选一）",
    )
    src.add_argument(
        "--jsonl-list",
        type=Path,
        metavar="PATH.txt",
        help="文本文件：每行一个 jsonl 路径（# 开头为注释）",
    )
    ap.add_argument(
        "--curriculum",
        action="store_true",
        help="§9.3：启用 O0→O1→O2→O3→Os 课程学习（须配合 --jsonl-o0 等）",
    )
    ap.add_argument(
        "--jsonl-o0",
        type=Path,
        default=None,
        help="Curriculum：O0 档 binskel jsonl（必须）",
    )
    ap.add_argument("--jsonl-o1", type=Path, default=None, help="Curriculum：O1 档（可选）")
    ap.add_argument("--jsonl-o2", type=Path, default=None, help="Curriculum：O2 档（可选）")
    ap.add_argument("--jsonl-o3", type=Path, default=None, help="Curriculum：O3 档（可选）")
    ap.add_argument(
        "--jsonl-os",
        type=Path,
        default=None,
        help="Curriculum：Os 档（-Os，可选；**最后**并入，序为 O0→O1→O2→O3→Os）",
    )
    ap.add_argument(
        "--curriculum-epochs-per-stage",
        type=int,
        default=None,
        help="每多少 epoch 并入下一优化档（默认读 config s2_curriculum_epochs_per_stage）",
    )
    ap.add_argument(
        "--no-node-hard-neg",
        action="store_true",
        help="关闭 §9.3 节点级硬负对比损失",
    )
    ap.add_argument(
        "--node-hard-neg-interval",
        type=int,
        default=None,
        help="每多少个 epoch 启用一次节点硬负（0=每 epoch；默认 5）",
    )
    ap.add_argument(
        "--lambda-node-hard-neg",
        type=float,
        default=None,
        help="节点硬负损失权重（覆盖 config s2_lambda_node_hard_neg）",
    )
    ap.add_argument(
        "--node-hard-neg-warmup-epochs",
        type=int,
        default=None,
        help="从首次启用 node_hn 的 epoch 起，历元内线性地将 node_hn λ 乘子 0→1 的长度（0=关；默认可读 config s2_node_hard_neg_warmup_epochs）",
    )
    ap.add_argument(
        "--s2-region-tau",
        type=float,
        default=None,
        help="Region InfoNCE 温度 τ（覆盖 config，默认与 Stage1 s1_infonce_tau 或 s2_region_infonce_tau 一致）",
    )
    ap.add_argument(
        "--s2-node-hn-tau",
        type=float,
        default=None,
        help="节点硬负 InfoNCE 温度 τ（覆盖 config，默认同上）",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Stage1 等 checkpoint（.pt），载入与模型匹配的权重；未匹配键跳过",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "从 Stage2 checkpoint（latest_stage2.pt / best_stage2.pt）无缝续训："
            "模型权重 + 优化器动量 + epoch + best_loss 全部保留。"
            "与 --init-from 互斥；若同时提供则优先使用 --resume。"
        ),
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints/stage2"),
        help="checkpoint 目录",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda / cpu（默认自动）",
    )
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 config.training.s2_batch_size")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config.training.s2_epochs")
    ap.add_argument(
        "--no-fp16",
        action="store_true",
        help="Stage2 使用 fp32（覆盖 s2_fp16=False；与默认一致，显式写上便于脚本）",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="Stage2 开启 fp16+GradScaler（默认关；旧行为/省显存时用，易 nan 梯度）",
    )
    ap.add_argument(
        "--lambda-src-count",
        type=float,
        default=None,
        dest="src_count_lambda",
        help="覆盖 s2_lambda_src_count（0=关掉 n_s 回归项加权；对照实验用）",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="每文件最多有效样本数（调试）")
    ap.add_argument(
        "--val-jsonl",
        type=Path,
        nargs="+",
        default=None,
        help="可选：验证集 jsonl（可多个，须含 alignment_matrix）。提供时每轮计算 val 并以 val 的 total 选 best",
    )
    ap.add_argument(
        "--val-jsonl-per-curriculum-stage",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "与 --curriculum 同用：N 个验证 jsonl，顺序=课程 stage0..(N-1)（O0|O0+O1|…|+Os），"
            "与训练各档一一对应；与 --val-jsonl 二选一。阶段切换时会对 best_loss 重排"
        ),
    )
    ap.add_argument(
        "--val-max-samples",
        type=int,
        default=None,
        help="验证集每文件最大样本数（默认同训练集不额外截断）",
    )
    ap.add_argument(
        "--reset-best-baseline",
        action="store_true",
        help="续训时忽略 checkpoint 中的 best_loss，从 +inf 重排（如刚启用 --val-jsonl 时避免与旧「训集 best」混用）",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader 预取批数（num_workers>0 时生效；SSD 上 4~8 常见，CPU/内存紧可降至 2）",
    )
    ap.add_argument(
        "--no-tf32",
        action="store_true",
        help="关闭 TF32（默认在 CUDA 上开启以加速 matmul/conv，与 Stage1 一致）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只跑一个 batch 前向并打印形状，不训练",
    )
    ap.add_argument(
        "--strict-jsonl",
        action="store_true",
        help="索引阶段遇坏行即失败（默认跳过）",
    )
    args = ap.parse_args()

    if not args.curriculum and args.jsonl is None and args.jsonl_list is None:
        ap.error("请指定 --jsonl、--jsonl-list 或 --curriculum + --jsonl-o0")
    if args.curriculum and args.jsonl_o0 is None:
        ap.error("--curriculum 需要至少指定 --jsonl-o0")
    if args.curriculum and (args.jsonl is not None or args.jsonl_list is not None):
        ap.error("--curriculum 与 --jsonl/--jsonl-list 请勿同时使用")
    if args.val_jsonl is not None and args.val_jsonl_per_curriculum_stage is not None:
        ap.error("--val-jsonl 与 --val-jsonl-per-curriculum-stage 不能同时指定")
    if args.val_jsonl_per_curriculum_stage is not None and not args.curriculum:
        ap.error("--val-jsonl-per-curriculum-stage 需要 --curriculum")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and not dist.is_initialized():
        _pg_timeout = _dt.timedelta(
            seconds=int(os.environ.get("PG_TIMEOUT_SECONDS", "14400")),
        )
        dist.init_process_group(backend="nccl", timeout=_pg_timeout)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp:
        if not torch.cuda.is_available():
            ap.error("DDP 需要 CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_s)

    if device.type == "cuda":
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    cfg = StrucLiftConfig()
    if args.fp16 and args.no_fp16:
        ap.error("--fp16 与 --no-fp16 不能同时使用")
    if args.no_fp16:
        cfg.training.s2_fp16 = False
    elif args.fp16:
        cfg.training.s2_fp16 = True
    if args.src_count_lambda is not None:
        cfg.training.s2_lambda_src_count = float(args.src_count_lambda)
    if args.batch_size is not None:
        cfg.training.s2_batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.s2_epochs = args.epochs
    if args.no_node_hard_neg:
        cfg.training.s2_node_hard_neg_enabled = False
    if args.node_hard_neg_interval is not None:
        cfg.training.s2_node_hard_neg_interval = int(args.node_hard_neg_interval)
    if args.lambda_node_hard_neg is not None:
        cfg.training.s2_lambda_node_hard_neg = float(args.lambda_node_hard_neg)
    if args.node_hard_neg_warmup_epochs is not None:
        cfg.training.s2_node_hard_neg_warmup_epochs = int(args.node_hard_neg_warmup_epochs)
    if args.s2_region_tau is not None:
        cfg.training.s2_region_infonce_tau = float(args.s2_region_tau)
    if args.s2_node_hn_tau is not None:
        cfg.training.s2_node_hn_infonce_tau = float(args.s2_node_hn_tau)
    if args.curriculum_epochs_per_stage is not None:
        cfg.training.s2_curriculum_epochs_per_stage = int(args.curriculum_epochs_per_stage)

    skip_invalid = not args.strict_jsonl

    curriculum_ds: Stage2CurriculumDataset | None = None
    jsonl_arg: object | None = None

    if args.curriculum:
        paths_by_opt: dict[str, Path] = {"O0": args.jsonl_o0.expanduser().resolve()}
        if args.jsonl_o1 is not None:
            paths_by_opt["O1"] = args.jsonl_o1.expanduser().resolve()
        if args.jsonl_o2 is not None:
            paths_by_opt["O2"] = args.jsonl_o2.expanduser().resolve()
        if args.jsonl_o3 is not None:
            paths_by_opt["O3"] = args.jsonl_o3.expanduser().resolve()
        if args.jsonl_os is not None:
            paths_by_opt["Os"] = args.jsonl_os.expanduser().resolve()
        curriculum_ds = Stage2CurriculumDataset(
            paths_by_opt,
            max_samples=args.max_samples,
            skip_invalid=skip_invalid,
        )
        curriculum_ds.set_stage(0)
        if not use_ddp or rank == 0:
            logger.info(
                "[Stage2] Curriculum 训练档**固定顺序** O0→O1→O2→O3→Os（与代码 OPT_ORDER 一致），"
                "本 run 已载入: %s",
                " → ".join(curriculum_ds.opt_tags_in_order()),
            )
        loader = build_stage2_train_loader(
            curriculum_ds,
            batch_size=cfg.training.s2_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            prefetch_factor=int(args.prefetch_factor),
            distributed=use_ddp,
            rank=rank,
            world_size=world_size,
        )
    elif args.jsonl_list is not None:
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
        jsonl_arg = paths[0] if len(paths) == 1 else paths
        loader = make_binskel_stage2_dataloader(
            jsonl_arg,
            batch_size=cfg.training.s2_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            distributed=use_ddp,
            rank=rank,
            world_size=world_size,
        )
    else:
        assert args.jsonl is not None
        jsonl_arg = (
            args.jsonl[0].expanduser().resolve()
            if len(args.jsonl) == 1
            else [p.expanduser().resolve() for p in args.jsonl]
        )
        loader = make_binskel_stage2_dataloader(
            jsonl_arg,
            batch_size=cfg.training.s2_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            prefetch_factor=int(args.prefetch_factor),
            distributed=use_ddp,
            rank=rank,
            world_size=world_size,
        )

    if len(loader.dataset) == 0:
        logger.error("无有效样本：jsonl 须含 alignment_matrix 与完整源码字段")
        sys.exit(1)

    val_loader = None
    val_loaders_by_curriculum_stage: list | None = None
    if args.val_jsonl_per_curriculum_stage is not None:
        assert curriculum_ds is not None
        n_st = curriculum_ds.max_stage() + 1
        paths = [p.expanduser().resolve() for p in args.val_jsonl_per_curriculum_stage]
        if len(paths) != n_st:
            logger.error(
                "[Stage2] --val-jsonl-per-curriculum-stage 须 %d 个文件（与课程档数一致），实际 %d",
                n_st,
                len(paths),
            )
            sys.exit(1)
        if not use_ddp or rank == 0:
            val_loaders_by_curriculum_stage = []
            for i, p in enumerate(paths):
                if not p.is_file():
                    logger.error("验证集文件不存在: %s", p)
                    sys.exit(1)
                vl = make_binskel_stage2_dataloader(
                    p,
                    batch_size=cfg.training.s2_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    max_samples=args.val_max_samples,
                    skip_invalid=skip_invalid,
                    pin_memory=device.type == "cuda",
                    prefetch_factor=int(args.prefetch_factor),
                )
                n_val = len(vl.dataset)
                if n_val == 0:
                    logger.warning(
                        "[Stage2] 验证 curriculum stage %d 无有效样本，该阶段将无 val", i
                    )
                val_loaders_by_curriculum_stage.append(vl)
                logger.info(
                    "[Stage2] 验证集 curriculum stage %d: %s 样本数=%d batch 数=%d 选优=val total",
                    i,
                    p.name,
                    n_val,
                    len(vl) if n_val else 0,
                )
    elif args.val_jsonl is not None and (not use_ddp or rank == 0):
        v_resolved = [p.expanduser().resolve() for p in args.val_jsonl]
        v_in: object = v_resolved[0] if len(v_resolved) == 1 else v_resolved
        val_loader = make_binskel_stage2_dataloader(
            v_in,
            batch_size=cfg.training.s2_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.val_max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            prefetch_factor=int(args.prefetch_factor),
        )
        n_val = len(val_loader.dataset)
        if n_val == 0:
            logger.warning("[Stage2] 验证集无有效样本，已忽略 --val-jsonl")
            val_loader = None
        else:
            logger.info(
                "[Stage2] 验证集: 样本数=%d  batch 数=%d 选优=val total",
                n_val,
                len(val_loader),
            )

    model = StrucLift(cfg).to(device)
    # --init-from 仅在 **非 resume** 情况下生效；resume 时会走 trainer 的完整加载
    if args.init_from is not None and args.resume is None:
        ckpt_path = args.init_from.expanduser().resolve()
        if not ckpt_path.is_file():
            logger.error("checkpoint 不存在: %s", ckpt_path)
            sys.exit(1)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.info("load_state_dict 未载入键（示例最多 5 个）: %s", missing[:5])
        if unexpected:
            logger.info("checkpoint 中未使用键（示例最多 5 个）: %s", unexpected[:5])
    elif args.resume is not None and args.init_from is not None:
        logger.warning(
            "同时提供 --resume 与 --init-from；--resume 优先（将忽略 --init-from）",
        )

    # 不能用 nn.parallel.DDP：StrucLift.encode 在**一次 forward** 里 for 多图，多次调
    # alignment_module，DDP 会对同一参数 mark-ready 多次 → RuntimeError。多卡时由
    # Stage2Trainer._sync_alignment_gradients 在 backward 后对 alignment 做 all_reduce 平均。
    if use_ddp:
        model.freeze_encoders()
        model.freeze_decoder_adapters()

    loader_kw = dict(
        batch_size=cfg.training.s2_batch_size,
        num_workers=args.num_workers,
        skip_invalid=skip_invalid,
        pin_memory=device.type == "cuda",
        prefetch_factor=int(args.prefetch_factor),
        distributed=use_ddp,
        rank=rank,
        world_size=world_size,
    )

    if args.dry_run:
        if use_ddp and rank != 0 and dist.is_initialized():
            dist.destroy_process_group()
            return
        batch = next(iter(loader))
        logger.info("dry-run batch:\n%s", describe_binskel_batch(batch))
        batch = move_binskel_batch_to_device(batch, device)
        core = model
        model.train()
        core.binary_encoder.eval()
        core.source_encoder.eval()
        with torch.amp.autocast(
            "cuda",
            enabled=bool(getattr(cfg.training, "s2_fp16", False)) and device.type == "cuda",
        ):
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
                bin_adj=batch.get("adj_bin"),
                src_adj=batch.get("adj_src_gt"),
                M_gt=batch.get("M_gt"),
                run_alignment=True,
            )
            assert out.encoder_out is not None
            enc_out = out.encoder_out
        al = enc_out.alignment
        assert al is not None
        logger.info(
            "dry-run OK: %d 图, M 形状示例 %s",
            len(al.alignment_matrix),
            tuple(al.alignment_matrix[0].shape) if al.alignment_matrix else (),
        )
        sk0 = enc_out.skeleton_list[0] if enc_out.skeleton_list else None
        if sk0 is not None:
            preview = sk0.skeleton_text[:800] + ("…" if len(sk0.skeleton_text) > 800 else "")
            logger.info("§6.5 骨架预览(图0, 前800字符):\n%s", preview)
        return

    trainer = Stage2Trainer(
        model,
        cfg,
        device=device,
        curriculum_dataset=curriculum_ds,
        curriculum_epochs_per_stage=cfg.training.s2_curriculum_epochs_per_stage,
        stage2_loader_kwargs=loader_kw,
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )

    resume_info = {"start_epoch": 1, "best_loss": float("inf")}
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            logger.error("--resume 路径不存在: %s", resume_path)
            sys.exit(1)
        info = trainer.load_full_checkpoint(str(resume_path))
        resume_info["start_epoch"] = int(info["start_epoch"])
        resume_info["best_loss"] = float(info["best_loss"])
        logger.info(
            "已从 %s 恢复（format=%s start_epoch=%d best_loss=%.6f missing=%d unexpected=%d）",
            resume_path,
            "legacy" if info["is_legacy"] else "full",
            resume_info["start_epoch"],
            resume_info["best_loss"],
            info["missing"],
            info["unexpected"],
        )
    if args.reset_best_baseline:
        resume_info["best_loss"] = float("inf")
        logger.info("[Stage2] --reset-best-baseline：已将 best 基线置为 +inf")

    _r_t = cfg.training.s2_region_infonce_tau
    _nh_t = cfg.training.s2_node_hn_infonce_tau
    _tau_s1 = cfg.training.s1_infonce_tau
    _show_reg = _r_t if _r_t is not None else _tau_s1
    _show_nh = _nh_t if _nh_t is not None else _tau_s1
    logger.info(
        "开始 Stage2: 样本数=%d batch_size(per-GPU)=%d epochs=%d curriculum=%s node_hn=%s "
        "interval=%d warmup_epochs=%d τ_region=%.4f τ_node_hn=%.4f s2_fp16=%s device=%s "
        "ddp=%s world_size=%d rank=%d",
        len(loader.dataset),
        cfg.training.s2_batch_size,
        cfg.training.s2_epochs,
        bool(curriculum_ds),
        cfg.training.s2_node_hard_neg_enabled,
        cfg.training.s2_node_hard_neg_interval,
        cfg.training.s2_node_hard_neg_warmup_epochs,
        float(_show_reg),
        float(_show_nh),
        cfg.training.s2_fp16,
        device,
        use_ddp,
        world_size,
        rank,
    )
    if device.type == "cuda":
        logger.info(
            "训练加速: cudnn.benchmark=%s TF32(matmul)=%s | DataLoader num_workers=%d prefetch_factor=%d",
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
            args.num_workers,
            int(args.prefetch_factor) if args.num_workers > 0 else 0,
        )
    trainer.train(
        loader,
        val_loader=val_loader,
        val_loaders_by_curriculum_stage=val_loaders_by_curriculum_stage,
        save_dir=str(args.save_dir.expanduser().resolve()),
        start_epoch=resume_info["start_epoch"],
        best_loss=resume_info["best_loss"],
    )
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

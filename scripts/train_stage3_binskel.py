#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3 监督微调（提案 §7 / §9）：jsonl 在 Stage2 字段基础上增加：

- ``sft_input_ids`` / ``sft_labels``：与 HF 词表一致的整型序列（``labels`` 中非监督位为 ``-100``）
- ``slot_types``：与序列中 FILL token 顺序一致的类型名列表
  （``FILL_COND`` / ``FILL_STMT`` / ``FILL_EXPR`` / ``FILL_TYPE`` / ``FILL_PARAM``）
- ``sft_fill_positions`` / ``sft_slot_bb_indices``：由 ``augment_binskel_sft.py`` 写入，用于 §7.3.2 ``slot_mask``

需安装 ``transformers``；可选 ``peft`` 做 LoRA。

示例::

  pip install transformers accelerate

  python scripts/train_stage3_binskel.py \\
    --jsonl path/to/sft_binskel.jsonl \\
    --init-from checkpoints/stage2/best_stage2.pt \\
    --llm deepseek-ai/deepseek-coder-6.7b-base
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    describe_binskel_batch,
    make_binskel_stage3_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.training.stage3_sft import Stage3Trainer
from struclift.utils.state_dict_compat import load_state_dict_skip_shape_mismatch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_hf_llm(
    cfg: StrucLiftConfig,
    llm_name_or_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
    llm_dtype: str = "auto",
):
    from transformers import AutoModelForCausalLM

    if llm_dtype == "auto":
        if device.type != "cuda":
            dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif llm_dtype == "bf16":
        dtype = torch.bfloat16
    elif llm_dtype == "fp16":
        dtype = torch.float16
    elif llm_dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported --llm-dtype: {llm_dtype}")
    logger.info("加载 LLM dtype=%s", dtype)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if target_vocab_size is not None and target_vocab_size > 0:
        cur = llm.get_input_embeddings().num_embeddings
        if target_vocab_size != cur:
            try:
                llm.resize_token_embeddings(
                    target_vocab_size, mean_resizing=False,
                )
            except TypeError:
                llm.resize_token_embeddings(target_vocab_size)
            logger.info(
                "resize_token_embeddings: %d -> %d（与 augment_binskel_sft 保存的 tokenizer 对齐）",
                cur,
                target_vocab_size,
            )
    try:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=cfg.module_c.lora_r,
            lora_alpha=cfg.module_c.lora_alpha,
            lora_dropout=cfg.module_c.lora_dropout,
            target_modules=list(cfg.module_c.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lc)
        logger.info("已启用 PEFT LoRA (r=%d)", cfg.module_c.lora_r)
    except ImportError:
        logger.warning("未安装 peft：冻结 LLM 基座，仅训练 CrossAttentionAdapter")
        for p in llm.parameters():
            p.requires_grad = False

    llm = llm.to(device)
    return llm


def _sync_module_c_shape_from_hf_config(cfg: StrucLiftConfig, llm_name_or_path: str) -> None:
    """Keep Module-C adapters aligned with the concrete HF model."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return

    hf_cfg = AutoConfig.from_pretrained(llm_name_or_path, trust_remote_code=True)
    n_layers = getattr(hf_cfg, "num_hidden_layers", None)
    hidden = getattr(hf_cfg, "hidden_size", None)
    if isinstance(n_layers, int) and n_layers > 0:
        if cfg.module_c.num_llm_layers != n_layers:
            logger.info(
                "ModuleC num_llm_layers: %d -> %d（来自 HF config）",
                cfg.module_c.num_llm_layers,
                n_layers,
            )
        cfg.module_c.num_llm_layers = n_layers
    if isinstance(hidden, int) and hidden > 0:
        if cfg.module_c.d_llm != hidden:
            logger.info(
                "ModuleC d_llm: %d -> %d（来自 HF config）",
                cfg.module_c.d_llm,
                hidden,
            )
        cfg.module_c.d_llm = hidden


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage3 SFT（Module C + 注入 LLM）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        help="含 sft_input_ids / sft_labels / slot_types 的 binskel .jsonl",
    )
    src.add_argument(
        "--jsonl-list",
        type=Path,
        metavar="PATH.txt",
        help="每行一个 jsonl 路径（# 开头为注释）",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Stage2（或含 A+B）checkpoint .pt",
    )
    ap.add_argument(
        "--llm",
        type=str,
        default=None,
        help="HF 模型名或本地目录（默认 config.module_c.llm_name）",
    )
    ap.add_argument(
        "--llm-dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="LLM load dtype. auto uses bf16 only when CUDA reports support, otherwise fp16.",
    )
    ap.add_argument(
        "--tokenizer-saved",
        type=Path,
        default=None,
        help="augment_binskel_sft.py --save-tokenizer 输出目录；将 resize LLM 词表并用于 pad_token_id",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=Path("checkpoints/stage3"),
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="跑一个 batch；若提供 --llm 则做一次完整前向",
    )
    ap.add_argument("--val-jsonl", type=Path, nargs="+", default=None,
                    help="验证集 jsonl（可选）")
    ap.add_argument("--strict-jsonl", action="store_true")
    args = ap.parse_args()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    cfg = StrucLiftConfig()
    if args.batch_size is not None:
        cfg.training.s3_batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.s3_epochs = args.epochs

    skip_invalid = not args.strict_jsonl
    llm_id = args.llm or cfg.module_c.llm_name
    _sync_module_c_shape_from_hf_config(cfg, llm_id)

    from transformers import AutoTokenizer

    tok_path_str = (
        str(args.tokenizer_saved.expanduser().resolve())
        if args.tokenizer_saved is not None
        else llm_id
    )
    tokenizer_stage3 = AutoTokenizer.from_pretrained(
        tok_path_str, trust_remote_code=True,
    )
    mft = int(cfg.module_c.max_fill_tokens)
    needed_fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(mft)]
    vocab_map = tokenizer_stage3.get_vocab()
    missing_fill = [t for t in needed_fill_tokens if t not in vocab_map]
    if missing_fill:
        tokenizer_stage3.add_tokens(missing_fill)
    fill_token_ids_cpu = torch.tensor(
        [tokenizer_stage3.convert_tokens_to_ids(t) for t in needed_fill_tokens],
        dtype=torch.long,
    )
    target_vs: int | None = len(tokenizer_stage3)
    pad_id = tokenizer_stage3.pad_token_id
    if pad_id is None:
        pad_id = tokenizer_stage3.eos_token_id
    if pad_id is None:
        pad_id = 0
    pad_id = int(pad_id)
    if args.tokenizer_saved is not None:
        logger.info(
            "使用 --tokenizer-saved: len=%d pad_token_id=%d",
            target_vs or 0,
            pad_id,
        )

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

    try:
        loader = make_binskel_stage3_dataloader(
            jsonl_arg,
            batch_size=cfg.training.s3_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=getattr(cfg.training, "s3_max_sft_seq_len", None),
            fill_token_ids=fill_token_ids_cpu,
            rebuild_slot_mask_when_missing=True,
        )
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
    if len(loader.dataset) == 0:
        logger.error("无有效样本：检查 jsonl 是否含 alignment_matrix 与 SFT 字段")
        sys.exit(1)

    model = StrucLift(cfg)
    if args.init_from is not None:
        ckpt_path = args.init_from.expanduser().resolve()
        if not ckpt_path.is_file():
            logger.error("checkpoint 不存在: %s", ckpt_path)
            sys.exit(1)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
            logger.info("已从 Stage2 封装 ckpt 取出 state[\"model\"] 载入 StrucLift")
        missing, unexpected = load_state_dict_skip_shape_mismatch(
            model, state, log=logger,
        )
        if missing:
            logger.info("load_state_dict 未载入键（示例最多 5 个）: %s", missing[:5])
        if unexpected:
            logger.info("checkpoint 中未使用键（示例最多 5 个）: %s", unexpected[:5])

    if args.dry_run:
        batch = next(iter(loader))
        logger.info("dry-run batch:\n%s", describe_binskel_batch(batch))
        if args.llm is None:
            logger.info("未指定 --llm，跳过 decoder 前向（仅 collate 检查）")
            return
        try:
            llm = _load_hf_llm(
                cfg,
                llm_id,
                device,
                target_vocab_size=target_vs,
                llm_dtype=args.llm_dtype,
            )
        except ImportError as e:
            logger.error("%s", e)
            sys.exit(1)
        model.decoder.inject_into_llm(llm)
        batch = move_binskel_batch_to_device(batch, device)
        trainer = Stage3Trainer(
            model,
            cfg,
            device=device,
            fill_token_ids=fill_token_ids_cpu.to(device),
            tokenizer=tokenizer_stage3,
        )
        model.train()
        model.binary_encoder.eval()
        model.source_encoder.eval()
        model.alignment_module.eval()
        with torch.amp.autocast(
            "cuda", enabled=cfg.training.fp16 and device.type == "cuda",
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
                input_ids=batch["input_ids"],
                graph_kv=batch.get("graph_kv"),
                slot_mask=batch.get("slot_mask"),
                labels=batch["labels"],
                slot_weight_mask=batch.get("slot_weight_mask")
                if batch.get("slot_weight_mask") is not None
                else trainer.slot_weight_builder.build_slot_weights_batched_for_supervised_labels(
                    batch["input_ids"],
                    batch["labels"],
                    batch["slot_types"],
                    cfg.module_c,
                    fill_token_ids=trainer.fill_token_ids,
                ),
            )
        logger.info(
            "dry-run OK: lm_loss=%s logits=%s",
            out.lm_loss,
            None if out.logits is None else tuple(out.logits.shape),
        )
        return

    try:
        llm = _load_hf_llm(
            cfg,
            llm_id,
            device,
            target_vocab_size=target_vs,
            llm_dtype=args.llm_dtype,
        )
    except ImportError as e:
        logger.error("需要 transformers: pip install transformers。%s", e)
        sys.exit(1)
    model.decoder.inject_into_llm(llm)

    trainer = Stage3Trainer(
        model,
        cfg,
        device=device,
        fill_token_ids=fill_token_ids_cpu.to(device),
        tokenizer=tokenizer_stage3,
    )
    logger.info(
        "开始 Stage3: 样本数=%d batch_size=%d epochs=%d device=%s llm=%s",
        len(loader.dataset),
        cfg.training.s3_batch_size,
        cfg.training.s3_epochs,
        device,
        llm_id,
    )
    val_loader = None
    if args.val_jsonl is not None:
        val_paths = (
            args.val_jsonl[0].expanduser().resolve()
            if len(args.val_jsonl) == 1
            else [p.expanduser().resolve() for p in args.val_jsonl]
        )
        val_loader = make_binskel_stage3_dataloader(
            val_paths,
            batch_size=cfg.training.s3_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=getattr(cfg.training, "s3_max_sft_seq_len", None),
            fill_token_ids=fill_token_ids_cpu,
            rebuild_slot_mask_when_missing=True,
        )
        logger.info("验证集样本数=%d", len(val_loader.dataset))
    trainer.train(loader, val_loader=val_loader, save_dir=str(args.save_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()

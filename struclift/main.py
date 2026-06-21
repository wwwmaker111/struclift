#!/usr/bin/env python3
"""
StrucLift — Main Entry Point.

Usage
-----
    python -m struclift.main --stage 1 --jsonl data/*.jsonl
    python -m struclift.main --stage 2 --jsonl data/*.jsonl --resume checkpoints/stage1/best_stage1.pt
    python -m struclift.main --stage 3 --jsonl data/*.jsonl --resume checkpoints/stage2/best_stage2.pt --llm deepseek-ai/deepseek-coder-6.7b-base
    python -m struclift.main --stage 4 --jsonl data/*.jsonl --resume checkpoints/stage3/best_stage3.pt --llm deepseek-ai/deepseek-coder-6.7b-base

Stages
------
  1: Contrastive pre-training  (Module A)
  2: Alignment training        (Module B, freeze A)
  3: Supervised fine-tuning    (Module C, freeze A+B)
  4: RL fine-tuning            (Module C RL, freeze A+B)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from struclift.config import StrucLiftConfig
from struclift.models.struclift import StrucLift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("struclift")


# ═══════════════════════════ helpers ══════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | None) -> StrucLiftConfig:
    """Load configuration from a JSON file, or return default config."""
    if path is None or not os.path.exists(path):
        logger.info("Using default StrucLiftConfig.")
        return StrucLiftConfig()

    with open(path, "r") as f:
        raw = json.load(f)

    config = StrucLiftConfig()
    for section_name, section_data in raw.items():
        if hasattr(config, section_name) and isinstance(section_data, dict):
            section = getattr(config, section_name)
            for key, value in section_data.items():
                if hasattr(section, key):
                    setattr(section, key, value)
    return config


def build_model(config: StrucLiftConfig, device: torch.device) -> StrucLift:
    """Instantiate the full StrucLift model."""
    model = StrucLift(config)
    model = model.to(device)
    param_info = model.count_parameters()
    logger.info(
        f"Model built. Total params: {param_info['total']:,}  "
        f"Trainable: {param_info['trainable']:,}"
    )
    return model


def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def _resolve_jsonl_paths(args: argparse.Namespace) -> List[Path]:
    """Collect jsonl paths from --jsonl and --jsonl-list arguments."""
    paths: List[Path] = []
    if args.jsonl:
        paths.extend(Path(p) for p in args.jsonl)
    if args.jsonl_list:
        with open(args.jsonl_list, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(Path(line))
    if not paths:
        raise ValueError(
            "No data specified.  Use --jsonl <path> ... or --jsonl-list <file>."
        )
    for p in paths:
        if not p.expanduser().resolve().exists():
            raise FileNotFoundError(f"JSONL file not found: {p}")
    return paths


def _resolve_val_paths(args: argparse.Namespace) -> Optional[List[Path]]:
    """Collect optional validation jsonl paths."""
    if not args.val_jsonl:
        return None
    paths = [Path(p) for p in args.val_jsonl]
    for p in paths:
        if not p.expanduser().resolve().exists():
            raise FileNotFoundError(f"Val JSONL not found: {p}")
    return paths


def _load_hf_llm(config: StrucLiftConfig, llm_name: str, device: torch.device,
                  tokenizer_dir: Optional[str] = None):
    """Load HuggingFace LLM + optional LoRA, return (llm, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = tokenizer_dir or llm_name
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # 与 augment_binskel_sft 一致：先注册 <|STRUCLIFT_FILL_xxxx|>，再对齐词表。
    # 否则 SlotWeightBuilder 用 id>=32000 会把 DeepSeek 词表高区普通 token 误判为 FILL。
    mft = int(getattr(config.module_c, "max_fill_tokens", 512))
    extra = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(mft)]
    n_added = tokenizer.add_tokens(extra)
    if n_added:
        logger.info(
            "Registered %d StrucLift FILL placeholder tokens (max_fill_tokens=%d)",
            n_added,
            mft,
        )

    want_8bit = (
        device.type == "cuda"
        and getattr(config.training, "llm_load_in_8bit", False)
    )
    use_8bit = False
    llm: Optional[torch.nn.Module] = None
    if want_8bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            gpu = device.index if device.index is not None else 0
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map={"": gpu},
            )
            use_8bit = True
            logger.info("LLM loaded in 8-bit (bitsandbytes); lower VRAM than bf16 base.")
        except ImportError:
            logger.warning(
                "llm_load_in_8bit=True 但无法使用 BitsAndBytesConfig（需 transformers/bitsandbytes），改用 bf16。"
            )
        except Exception as e:
            logger.warning(
                "8-bit 加载失败（需 GPU 与 bitsandbytes；错误: %s），回退 bf16。",
                e,
            )

    if llm is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, trust_remote_code=True, torch_dtype=dtype,
        )

    target_vocab = len(tokenizer)
    cur_vocab = llm.get_input_embeddings().num_embeddings
    if target_vocab != cur_vocab:
        try:
            llm.resize_token_embeddings(target_vocab, mean_resizing=False)
        except TypeError:
            llm.resize_token_embeddings(target_vocab)
        logger.info("resize_token_embeddings: %d -> %d", cur_vocab, target_vocab)

    try:
        from peft import LoraConfig, get_peft_model
        if use_8bit:
            try:
                from peft import prepare_model_for_kbit_training
            except ImportError:
                logger.warning("prepare_model_for_kbit_training 不可用，跳过 kbit 准备。")
            else:
                llm = prepare_model_for_kbit_training(llm)
        lc = LoraConfig(
            r=config.module_c.lora_r,
            lora_alpha=config.module_c.lora_alpha,
            lora_dropout=config.module_c.lora_dropout,
            target_modules=list(config.module_c.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lc)
        logger.info("PEFT LoRA enabled (r=%d)", config.module_c.lora_r)
    except ImportError:
        logger.warning(
            "peft not installed: freezing LLM base, only adapters are trainable."
        )
        for p in llm.parameters():
            p.requires_grad = False

    if not use_8bit:
        llm = llm.to(device)
    return llm, tokenizer


# ═══════════════════════════ stage runners ════════════════════════════════════

def run_stage1(
    config: StrucLiftConfig,
    model: StrucLift,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Stage 1: Contrastive pre-training (Module A)."""
    from struclift.training.stage1_pretrain import Stage1Trainer
    from struclift.data.binskel_jsonl import make_binskel_dataloader

    paths = _resolve_jsonl_paths(args)

    trainer = Stage1Trainer(model, config, device)

    logger.info("[Stage1] Contrastive pre-training.")
    logger.info(
        "  epochs=%d  lr=%s  batch_size=%d  jsonl_files=%d",
        config.training.s1_epochs, config.training.s1_lr,
        config.training.s1_batch_size, len(paths),
    )
    esp = int(getattr(config.training, "s1_early_stop_patience", 0) or 0)
    if esp > 0:
        logger.info(
            "  early_stop: patience=%d  min_delta=%s  min_epochs=%d",
            esp,
            getattr(config.training, "s1_early_stop_min_delta", 1e-4),
            int(getattr(config.training, "s1_early_stop_min_epochs", 5)),
        )

    train_loader = make_binskel_dataloader(
        paths,
        batch_size=config.training.s1_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        require_source=True,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    val_paths = _resolve_val_paths(args)
    if val_paths:
        val_loader = make_binskel_dataloader(
            val_paths,
            batch_size=config.training.s1_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            require_source=True,
            skip_invalid=True,
            pin_memory=device.type == "cuda",
        )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=os.path.join(args.output_dir, "stage1"),
    )


def run_stage2(
    config: StrucLiftConfig,
    model: StrucLift,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Stage 2: Alignment training (Module B, freeze A)."""
    from struclift.training.stage2_alignment import Stage2Trainer
    from struclift.data.binskel_jsonl import make_binskel_stage2_dataloader

    paths = _resolve_jsonl_paths(args)

    trainer = Stage2Trainer(model, config, device)

    logger.info("[Stage2] Alignment training.")
    logger.info(
        "  epochs=%d  lr=%s  batch_size=%d  jsonl_files=%d",
        config.training.s2_epochs, config.training.s2_lr,
        config.training.s2_batch_size, len(paths),
    )

    train_loader = make_binskel_stage2_dataloader(
        paths,
        batch_size=config.training.s2_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    val_paths = _resolve_val_paths(args)
    if val_paths:
        val_loader = make_binskel_stage2_dataloader(
            val_paths,
            batch_size=config.training.s2_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            skip_invalid=True,
            pin_memory=device.type == "cuda",
        )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=os.path.join(args.output_dir, "stage2"),
    )


def run_stage3(
    config: StrucLiftConfig,
    model: StrucLift,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Stage 3: Supervised fine-tuning (Module C, freeze A+B)."""
    from struclift.training.stage3_sft import Stage3Trainer
    from struclift.data.binskel_jsonl import make_binskel_stage3_dataloader

    paths = _resolve_jsonl_paths(args)
    llm_name = args.llm or config.module_c.llm_name
    llm, tokenizer = _load_hf_llm(config, llm_name, device, args.tokenizer_dir)
    model.decoder.inject_into_llm(llm)
    logger.info("LLM injected: %s", llm_name)

    if getattr(config.training, "llm_gradient_checkpointing", False):
        inner = model.decoder._llm
        if inner is not None and hasattr(inner, "gradient_checkpointing_enable"):
            inner.gradient_checkpointing_enable()
            logger.info("LLM gradient checkpointing enabled (lower VRAM)")

    mft = int(getattr(config.module_c, "max_fill_tokens", 512))
    fill_token_ids = torch.tensor(
        [
            tokenizer.convert_tokens_to_ids(f"<|STRUCLIFT_FILL_{i:04d}|>")
            for i in range(mft)
        ],
        dtype=torch.long,
        device=device,
    )
    fu = torch.unique(fill_token_ids)
    logger.info(
        "Stage3 FILL token ids: %d unique (max_fill_tokens=%d), id range [%d, %d]",
        int(fu.numel()),
        mft,
        int(fill_token_ids.min()),
        int(fill_token_ids.max()),
    )
    if fu.numel() < min(64, mft // 2):
        logger.warning(
            "FILL token id 几乎无区分度（大量重复 id）；请确认 _load_hf_llm 已 "
            "tokenizer.add_tokens(<|STRUCLIFT_FILL_xxxx|>) 或使用 augment 保存的 tokenizer 目录。",
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    trainer = Stage3Trainer(
        model,
        config,
        device,
        fill_token_ids=fill_token_ids,
        tokenizer=tokenizer,
    )

    logger.info("[Stage3] Supervised fine-tuning.")
    logger.info(
        "  epochs=%d  lr=%s  batch_size=%d  jsonl_files=%d  llm=%s",
        config.training.s3_epochs, config.training.s3_lr,
        config.training.s3_batch_size, len(paths), llm_name,
    )

    msl = getattr(config.training, "s3_max_sft_seq_len", None)
    if msl is not None:
        logger.info("Stage3 max_sft_seq_len=%s（超长样本将跳过）", msl)

    train_loader = make_binskel_stage3_dataloader(
        paths,
        batch_size=config.training.s3_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
        pad_token_id=pad_id,
        max_sft_seq_len=msl,
    )
    val_loader = None
    val_paths = _resolve_val_paths(args)
    if val_paths:
        val_loader = make_binskel_stage3_dataloader(
            val_paths,
            batch_size=config.training.s3_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            skip_invalid=True,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=msl,
        )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=os.path.join(args.output_dir, "stage3"),
    )


def run_stage4(
    config: StrucLiftConfig,
    model: StrucLift,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Stage 4: RL fine-tuning (GRPO, freeze A+B)."""
    from struclift.training.stage4_rl import Stage4Trainer
    from struclift.data.binskel_jsonl import make_binskel_stage4_dataloader

    paths = _resolve_jsonl_paths(args)
    llm_name = args.llm or config.module_c.llm_name
    llm, tokenizer = _load_hf_llm(config, llm_name, device, args.tokenizer_dir)
    model.decoder.inject_into_llm(llm)
    logger.info("LLM injected: %s", llm_name)

    # GRPO 对每个候选单独前向（completion 长度不同）；HF 的 gradient checkpointing
    # 在 backward 时重算前向，要求与首次前向图一致，多长度前向会触发 CheckpointError。
    inner = model.decoder._llm
    if inner is not None and hasattr(inner, "gradient_checkpointing_disable"):
        inner.gradient_checkpointing_disable()
    if getattr(config.training, "llm_gradient_checkpointing", False):
        logger.info(
            "Stage4: 已忽略 llm_gradient_checkpointing（GRPO 与 checkpoint 不兼容；Stage3 仍可用）"
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    trainer = Stage4Trainer(
        model, config, device,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )

    logger.info("[Stage4] RL fine-tuning (GRPO).")
    logger.info(
        "  epochs=%d  lr=%s  G=%d  jsonl_files=%d  llm=%s",
        config.module_d.rl_epochs, config.module_d.rl_lr,
        config.module_d.num_candidates, len(paths), llm_name,
    )

    msl = getattr(config.training, "s3_max_sft_seq_len", None)
    if msl is not None:
        logger.info("Stage4 max_sft_seq_len=%s（与 Stage3 同配置，超长样本将跳过）", msl)

    train_loader = make_binskel_stage4_dataloader(
        paths,
        batch_size=1,
        shuffle=True,
        num_workers=config.training.num_workers,
        skip_invalid=True,
        pin_memory=device.type == "cuda",
        pad_token_id=pad_id,
        max_sft_seq_len=msl,
    )
    val_loader = None
    val_paths = _resolve_val_paths(args)
    if val_paths:
        val_loader = make_binskel_stage4_dataloader(
            val_paths,
            batch_size=1,
            shuffle=False,
            num_workers=config.training.num_workers,
            skip_invalid=True,
            pin_memory=device.type == "cuda",
            pad_token_id=pad_id,
            max_sft_seq_len=msl,
        )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=os.path.join(args.output_dir, "stage4"),
    )


# ═══════════════════════════ CLI ══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StrucLift — Structure-Aware Binary Decompilation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2, 3, 4],
        help="Training stage (1=pretrain, 2=alignment, 3=SFT, 4=RL)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (uses defaults if omitted)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints",
        help="Directory for saving checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config.training.seed)",
    )

    # ── data ────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("data")
    data_grp.add_argument(
        "--jsonl", type=str, nargs="+", default=None,
        help="One or more binskel .jsonl file paths",
    )
    data_grp.add_argument(
        "--jsonl-list", type=str, default=None, metavar="PATH.txt",
        help="Text file listing one jsonl path per line (# comments allowed)",
    )
    data_grp.add_argument(
        "--val-jsonl", type=str, nargs="+", default=None,
        help="Optional validation jsonl files",
    )

    # ── LLM (Stage 3/4) ────────────────────────────────────────────────
    llm_grp = parser.add_argument_group("LLM (Stage 3 & 4)")
    llm_grp.add_argument(
        "--llm", type=str, default=None,
        help="HuggingFace model name or local path for the LLM "
             "(default: config.module_c.llm_name)",
    )
    llm_grp.add_argument(
        "--tokenizer-dir", type=str, default=None,
        help="Tokenizer directory (if different from --llm, "
             "e.g. after augment_binskel_sft added special tokens)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── config ───────────────────────────────────────────────────────────
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.training.seed
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # ── device ───────────────────────────────────────────────────────────
    device = get_device()

    # ── model ────────────────────────────────────────────────────────────
    model = build_model(config, device)

    # ── resume from checkpoint ───────────────────────────────────────────
    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded (strict=False).")

    # ── dispatch to stage ────────────────────────────────────────────────
    stage_runners = {
        1: run_stage1,
        2: run_stage2,
        3: run_stage3,
        4: run_stage4,
    }
    runner = stage_runners[args.stage]
    runner(config, model, device, args)


if __name__ == "__main__":
    main()

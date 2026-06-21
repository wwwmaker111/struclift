#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage4 结构一致性 GRPO（提案 §8–§9）：与 Stage3 相同 jsonl 格式；可选字段：

- ``bin_pattern_dist``：长度须与词表 pattern 类数一致（默认 10）
- ``bin_cst_serialised``：控制结构树前序序列（字符串列表）
- ``bin_nesting_depth``：整型嵌套深度

须已安装 ``transformers``；本机需 ``gcc`` 在 PATH 内以计算编译奖励（见 ``module_d.compile_reward``）。

默认 ``batch_size=1``（每个样本内采样 G 个候选做组内标准化）。

示例::

  python scripts/train_stage4_binskel.py \\
    --jsonl path/to/sft_binskel.jsonl \\
    --init-from checkpoints/stage3/best_stage3.pt \\
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
    make_binskel_stage4_dataloader,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.training.stage4_rl import Stage4Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_hf_llm(
    cfg: StrucLiftConfig,
    llm_name_or_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
):
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
                "resize_token_embeddings: %d -> %d（与 Stage3 / augment tokenizer 对齐）",
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage4 GRPO（Module C RL）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path, nargs="+")
    src.add_argument(
        "--jsonl-list",
        type=Path,
        metavar="PATH.txt",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        required=True,
        help="Stage3 SFT checkpoint（含注入用 LLM 权重）",
    )
    ap.add_argument("--llm", type=str, default=None)
    ap.add_argument(
        "--tokenizer-saved",
        type=Path,
        default=None,
        help="augment_binskel_sft.py --save-tokenizer 目录；与 Stage3 一致用于 FILL id 与词表大小",
    )
    ap.add_argument("--save-dir", type=Path, default=Path("checkpoints/stage4"))
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="建议保持 1（每样本独立 GRPO）",
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict-jsonl", action="store_true")
    args = ap.parse_args()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    cfg = StrucLiftConfig()
    skip_invalid = not args.strict_jsonl
    llm_id = args.llm or cfg.module_c.llm_name

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("需要 transformers: pip install transformers")
        sys.exit(1)
    tok_path_str = (
        str(args.tokenizer_saved.expanduser().resolve())
        if args.tokenizer_saved is not None
        else llm_id
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path_str, trust_remote_code=True)
    mft = int(cfg.module_c.max_fill_tokens)
    needed_fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(mft)]
    vocab_map = tokenizer.get_vocab()
    missing_fill = [t for t in needed_fill_tokens if t not in vocab_map]
    if missing_fill:
        tokenizer.add_tokens(missing_fill)
    fill_token_ids_cpu = torch.tensor(
        [tokenizer.convert_tokens_to_ids(t) for t in needed_fill_tokens],
        dtype=torch.long,
    )
    pad_tok = tokenizer.pad_token_id
    if pad_tok is None:
        pad_tok = tokenizer.eos_token_id
    if pad_tok is None:
        pad_tok = 0
    pad_id = int(pad_tok)
    target_vs = len(tokenizer)
    if args.tokenizer_saved is not None:
        logger.info(
            "使用 --tokenizer-saved: len=%d pad_token_id=%d",
            target_vs,
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
            logger.error("--jsonl-list 无有效路径")
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
        loader = make_binskel_stage4_dataloader(
            jsonl_arg,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            skip_invalid=skip_invalid,
            pin_memory=device.type == "cuda",
            pad_token_id=int(pad_id),
            num_patterns=cfg.vocab.num_patterns,
            fill_token_ids=fill_token_ids_cpu,
            rebuild_slot_mask_when_missing=True,
        )
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
    if len(loader.dataset) == 0:
        logger.error("无有效样本")
        sys.exit(1)

    model = StrucLift(cfg)
    ckpt_path = args.init_from.expanduser().resolve()
    if not ckpt_path.is_file():
        logger.error("checkpoint 不存在: %s", ckpt_path)
        sys.exit(1)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    llm = _load_hf_llm(cfg, llm_id, device, target_vocab_size=target_vs)
    model.decoder.inject_into_llm(llm)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info("load_state_dict 未载入键（示例最多 5 个）: %s", missing[:5])
    if unexpected:
        logger.info("checkpoint 中未使用键（示例最多 5 个）: %s", unexpected[:5])

    trainer = Stage4Trainer(
        model,
        cfg,
        device=device,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    if args.dry_run:
        batch = next(iter(loader))
        logger.info("dry-run batch:\n%s", describe_binskel_batch(batch))
        batch = move_binskel_batch_to_device(batch, device)
        trainer.model.eval()
        with torch.no_grad():
            graph_kv, sm = trainer._ensure_graph_kv_slot_mask(batch, batch["input_ids"])
        logger.info(
            "dry-run: graph_kv=%s slot_mask=%s",
            tuple(graph_kv.shape),
            tuple(sm.shape),
        )
        return

    logger.info(
        "开始 Stage4: 样本数=%d batch_size=%d rl_epochs=%d device=%s",
        len(loader.dataset),
        args.batch_size,
        cfg.module_d.rl_epochs,
        device,
    )
    trainer.train(loader, save_dir=str(args.save_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()

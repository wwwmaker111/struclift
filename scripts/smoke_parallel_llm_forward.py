#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 GPU 冒烟：``device_map=\"auto\"`` 将 HF CausalLM 切到多块卡，跑一次最小前向。

用途
----
在改造 Stage3 多卡之前，验证：
  - ``accelerate`` + ``device_map=\"auto\"`` 能否在你机器上拆分权重；
  - 多块 GPU 空闲显存之和是否够用（仍需与模型量级匹配）。

**注意**：本脚本只测「HF 底座」并行加载；不等于当前 ``Stage3Trainer`` 已自动多卡，
仅用于环境/驱动的并行冒烟。

示例::

  # 两台逻辑 GPU（请先 export CUDA_VISIBLE_DEVICES=物理卡编号）
  CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_parallel_llm_forward.py \\
    --model /path/to/Qwen3.6_27B \\
    --per-gpu-max-gib 38

  # 不写 --per-gpu-max-gib 时尽量自动切分（大模型一般由 accelerate 摊开）

  # 极小模型 + 每张卡只允许很少显存 → 有时会强迫层跨过两张卡（仅验证管线）
  CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_parallel_llm_forward.py --tiny \\
    --per-gpu-max-gib 1

依赖
----
  pip install transformers accelerate torch
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _require_accelerate() -> None:
    try:
        import accelerate  # noqa: F401
    except ImportError as e:
        logger.error(
            "需要 accelerate（device_map=auto）。请: pip install accelerate。%s",
            e,
        )
        sys.exit(1)


def _max_memory(per_gpu_gib: float | None) -> dict[int, str] | None:
    """若指定每张卡上限，可强迫大模型跨过更多 GPU（冒烟 / 压榨切分时用）。"""
    if per_gpu_gib is None or per_gpu_gib <= 0:
        return None
    import torch

    mm: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        mm[i] = f"{float(per_gpu_gib):g}GiB"
    return mm


def main() -> int:
    _require_accelerate()

    ap = argparse.ArgumentParser(
        description="HF CausalLM 多卡 device_map auto 冒烟前向",
    )
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument(
        "--model",
        type=str,
        help="本地 HF 快照目录或 hub id（如 Qwen），与 --tiny 二选一",
    )
    mx.add_argument(
        "--tiny",
        action="store_true",
        help="使用 huggingface 极小 Llama smoke 模型（只验证多卡管线）",
    )
    ap.add_argument(
        "--per-gpu-max-gib",
        type=float,
        default=None,
        metavar="G",
        help=(
            "传给 max_memory：每张可见 GPU 至多使用 G GiB。"
            "可略低于实际空闲显存以强迫层跨卡；小模型 + 多卡测试时可设 1~2。"
        ),
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="占位 input_ids 长度（默认短序列省激活）",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        logger.error("本脚本为多卡冒烟，请先在有 CUDA 的环境运行。")
        return 1

    n = torch.cuda.device_count()
    try:
        gname = torch.cuda.get_device_name(0)
    except Exception:
        gname = "?"
    logger.info(
        "可见 GPU 数量=%d，device0=%s，CUDA_VISIBLE_DEVICES=%r",
        n,
        gname,
        __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    tiny_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model_id = tiny_id if args.tiny else args.model

    max_mem = _max_memory(args.per_gpu_max_gib)

    logger.info("加载模型: %s ...", model_id)
    fm_kw: dict = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if max_mem is not None:
        fm_kw["max_memory"] = max_mem

    model = AutoModelForCausalLM.from_pretrained(model_id, **fm_kw)
    model.eval()

    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict):
        uniq = sorted({str(v) for v in dm.values()})
        logger.info("hf_device_map 涉及设备: %s（共 %d 个条目）", uniq, len(dm))
    else:
        logger.warning("模型无 hf_device_map 属性（可能整模单设备）")

    for i in range(n):
        torch.cuda.reset_peak_memory_stats(i)

    emb = model.get_input_embeddings().weight.device
    sl = max(4, min(int(args.seq_len), 8192))
    input_ids = torch.ones(1, sl, dtype=torch.long, device=emb)
    attn = torch.ones_like(input_ids)

    logger.info(
        "前向: seq_len=%d device(input)=%s dtype=%s",
        sl,
        emb,
        dtype,
    )
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
    except RuntimeError as e:
        logger.error("前向失败: %s", e)
        return 1

    if not hasattr(out, "logits"):
        logger.error("输出无 logits，与 CausalLM 预期不符。")
        return 1

    for i in range(n):
        peak = torch.cuda.max_memory_allocated(i) / (1024**3)
        logger.info("GPU %d peak_memory_allocated ~= %.2f GiB", i, peak)

    logger.info(
        "OK logits.shape=%s",
        tuple(out.logits.shape),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

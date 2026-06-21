#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-V4-Flash（或同目录结构的本地 HF 快照）最小加载 + generate 冒烟。

用法::

  # 默认使用常见 ModelScope/HF 镜像路径；可改环境变量或参数
  export MODEL_DIR=/data/chaoni/WQM/models/DeepSeek-V4-Flash/deepseek-ai/DeepSeek-V4-Flash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/smoke_deepseek_v4_flash.py

  python3 scripts/smoke_deepseek_v4_flash.py \\
    --model /data/chaoni/WQM/models/DeepSeek-V4-Flash/DeepSeek-V4-Flash \\
    --max-new-tokens 16

成功时打印续写文本并以退出码 0 结束；失败打印异常并以 1 结束。

若报错 ``does not recognize ... architecture deepseek_v4``：当前 ``transformers``
版本过旧（DeepSeek-V4 需含 PR #45643 之后的版本）。可先::

  pip install -U transformers accelerate

仍不行则从源码安装::

  pip install -U git+https://github.com/huggingface/transformers.git

若报错 ``module 'torch' has no attribute 'float8_e8m0fnu'``：本地快照为 FP8 权重，
解析该 dtype 需要 **较新 PyTorch（通常 >=2.6）**。在已激活的 conda 环境中::

  pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

（与机器驱动匹配的 CUDA wheel 任选 cu121/cu124 等即可。）

（需与模型卡页要求的版本一致时再对齐）
"""

from __future__ import annotations

import argparse
import os
import sys


def _hint_deepseek_v4_transformers() -> None:
    print(
        "[smoke] DeepSeek-V4 需要新版 transformers（识别 model_type=deepseek_v4）。\n"
        "[smoke] 尝试: pip install -U transformers accelerate\n"
        "[smoke] 仍失败: pip install -U git+https://github.com/huggingface/transformers.git",
        file=sys.stderr,
    )


def main() -> int:
    default_dir = os.environ.get(
        "MODEL_DIR",
        "/data/chaoni/WQM/models/DeepSeek-V4-Flash/deepseek-ai/DeepSeek-V4-Flash",
    )
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Flash local snapshot smoke test")
    ap.add_argument(
        "--model",
        type=str,
        default=default_dir,
        help="含 config.json 与权重的目录（不要用 .../inference 子目录除非含完整权重）",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", type=str, default="Hello")
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model)
    if not os.path.isdir(model_path):
        print(f"[smoke] ERROR: 目录不存在: {model_path}", file=sys.stderr)
        return 1
    cfg_json = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_json):
        print(
            f"[smoke] ERROR: 未找到 config.json，请指到快照根目录: {cfg_json}",
            file=sys.stderr,
        )
        return 1

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[smoke] python_executable={sys.executable}")
    print(f"[smoke] MODEL={model_path}")
    print(f"[smoke] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    print(f"[smoke] transformers={transformers.__version__}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except ValueError as e:
        err = str(e).lower()
        if "deepseek_v4" in err or "does not recognize" in err:
            print(f"[smoke] FAILED: {e}", file=sys.stderr)
            _hint_deepseek_v4_transformers()
            return 1
        raise
    inputs = tok(args.prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    model.eval()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print("[smoke] ---- output ----")
    print(text)
    print("[smoke] ---- end ----")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        if "float8_e8m0fnu" in str(e):
            print(
                "[smoke] FP8 权重需要 PyTorch 提供 float8 dtype（常见原因：torch 过旧）。\n"
                "[smoke] 尝试: pip install -U torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu121",
                file=sys.stderr,
            )
        raise SystemExit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无源码推理 + 评测一条龙：ELF → 推理 jsonl → Stage3 填槽 → ``pred.c``。

阶段 1 — ``build_binskel_infer_jsonl``（本仓库）  
  从 ``.o/.elf`` 反汇编，写出仅含 ``bin_*`` 的 jsonl（**不需要 .c**）。

阶段 2 — ``eval_stage3_binskel.py``  
  加载 Stage2/3 ckpt + DeepSeek 基座；``--alignment bin_only``、
  ``--skeleton-patterns module_a``（默认，部署向）；输出 ``pred.c``。

可选 ``--gt-jsonl``：带源码监督的 val jsonl，仅用于对比 ``gt.c`` / 准确率（推理不用）。

示例（10 个函数）::

  CUDA_VISIBLE_DEVICES=7 python3 scripts/infer_binskel_pipeline.py \\
    --elf /path/to/zlib_o0.o \\
    --func-names-file /tmp/ten_funcs.txt \\
    --init-from /data/chaoni/WQM/checkpoints/stage3_o0_deepseek67_p20_2gpu_b4/best_stage3.pt \\
    --llm /data/chaoni/WQM/models/deepseek-coder-6.7b-instruct \\
    --tokenizer-saved /data/chaoni/WQM/datasets/structlift_tokenizer_deepseek_6.7b_instruct_fill \\
    --out-dir /tmp/infer_10funcs \\
    --gt-jsonl /data/chaoni/WQM/datasets/C_val_o0_deepseek_coder6.7b_p20.jsonl

若已有推理 jsonl，可跳过阶段 1::

  python3 scripts/infer_binskel_pipeline.py \\
    --infer-jsonl /tmp/foo_infer.jsonl \\
    --init-from ... --llm ... --tokenizer-saved ... --out-dir /tmp/out
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_build_infer():
    path = SCRIPT_DIR / "build_binskel_infer_jsonl.py"
    spec = importlib.util.spec_from_file_location("_build_binskel_infer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="无源码：ELF/jsonl → Stage3 推理评测")
    ap.add_argument("--elf", type=Path, default=None, help="输入 ELF（与 --infer-jsonl 二选一）")
    ap.add_argument("--infer-jsonl", type=Path, default=None, help="已有推理 jsonl，跳过构建")
    ap.add_argument("--work-dir", type=Path, default=None, help="中间 jsonl 目录（默认 --out-dir）")
    ap.add_argument("--func-names", type=str, default=None)
    ap.add_argument("--func-names-file", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--opt", default="O0")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--init-from", type=Path, required=True)
    ap.add_argument("--llm", type=str, required=True)
    ap.add_argument("--tokenizer-saved", type=Path, required=True)
    ap.add_argument("--gt-jsonl", type=Path, default=None, help="可选：仅用于 gt 对比")
    ap.add_argument("--max-samples", type=int, default=0, help="0=jsonl 全部")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm-device-map-auto", action="store_true")
    ap.add_argument(
        "--skeleton-patterns",
        choices=("jsonl_gt", "module_a"),
        default="module_a",
    )
    ap.add_argument(
        "--alignment",
        choices=("bin_only", "gt_src"),
        default="bin_only",
    )
    ap.add_argument("--max-sft-seq-len", type=int, default=2048)
    ap.add_argument("--skip-build", action="store_true", help="等同提供 --infer-jsonl")
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = (args.work_dir or out_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    infer_jsonl: Path
    if args.infer_jsonl is not None:
        infer_jsonl = args.infer_jsonl.expanduser().resolve()
        if not infer_jsonl.is_file():
            logger.error("--infer-jsonl 不存在: %s", infer_jsonl)
            sys.exit(1)
    elif args.elf is not None and not args.skip_build:
        build_mod = _load_build_infer()
        filt = build_mod._parse_func_names_arg(args.func_names)
        if args.func_names_file is not None:
            ff = build_mod._parse_func_names_file(args.func_names_file.expanduser().resolve())
            filt = ff if filt is None else (filt & ff)
        stem = args.elf.expanduser().resolve().name.replace(".", "_")
        infer_jsonl = work / f"{stem}_infer.jsonl"
        n = build_mod.build_infer_jsonl(
            args.elf.expanduser().resolve(),
            infer_jsonl,
            func_names=filt,
            limit=args.limit,
            opt_level=args.opt,
        )
        logger.info("阶段1 完成: %d 条 → %s", n, infer_jsonl)
    else:
        logger.error("请提供 --elf 或 --infer-jsonl")
        sys.exit(1)

    eval_script = SCRIPT_DIR / "eval_stage3_binskel.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--init-from",
        str(args.init_from.expanduser().resolve()),
        "--jsonl",
        str(infer_jsonl),
        "--llm",
        args.llm,
        "--tokenizer-saved",
        str(args.tokenizer_saved.expanduser().resolve()),
        "--out-dir",
        str(out_dir),
        "--profile",
        "deploy",
        "--decode",
        "autoregressive",
        "--eval-mode",
        "bin_skeleton",
        "--alignment",
        args.alignment,
        "--skeleton-patterns",
        args.skeleton_patterns,
        "--device",
        args.device,
        "--max-sft-seq-len",
        str(args.max_sft_seq_len),
    ]
    cmd.extend(["--max-samples", str(args.max_samples if args.max_samples > 0 else 10**9)])
    if args.llm_device_map_auto:
        cmd.append("--llm-device-map-auto")
    if args.gt_jsonl is not None:
        cmd.extend(["--gt-jsonl", str(args.gt_jsonl.expanduser().resolve())])

    logger.info("阶段2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    logger.info("完成。结果: %s/summary.json", out_dir)


if __name__ == "__main__":
    main()

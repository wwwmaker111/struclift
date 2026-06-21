#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 9×5=45 个标准 binskel jsonl 逐一运行 augment_binskel_sft.py，产出可训练 Stage1–4 的「完整」jsonl。

- Stage1/2：binskel 原有字段已足够；本步在原文件基础上 **追加** sft_input_ids / sft_labels / slot_types。
- Stage3：需要上述三字段（本脚本产出）。
- Stage4：建议 augment 时加 --emit-rl-fields（本脚本默认开启）。

输入/输出命名与 split_binskel_45_datasets_811.py 一致，便于同一目录结构再跑 8:1:1 划分。

用法::

  cd /path/to/structlift
  python scripts/augment_all_45_to_full_sft.py \\
      --binskel-dir /mnt/e/structlift_datasets \\
      --out-dir /mnt/e/structlift_datasets/full_sft_45 \\
      --roots-json scripts/full_dataset_45_roots.json \\
      --tokenizer deepseek-ai/deepseek-coder-6.7b-base

仅干跑（不执行子进程）::

  python scripts/augment_all_45_to_full_sft.py --binskel-dir ... --out-dir ... --roots-json ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPTS = ("o0", "o1", "o2", "o3", "os")
PROJECTS_SIMPLE = (
    "zlib",
    "curl",
    "sqlite",
    "busybox",
    "openssl",
    "libxml2",
)
PROJECTS_MULTIBIN = ("openssh", "ffmpeg", "coreutils")


def expected_pairs() -> List[Tuple[str, Path]]:
    """(逻辑名, 文件名) 共 45 个。"""
    out: List[Tuple[str, Path]] = []
    for proj in PROJECTS_SIMPLE:
        for opt in OPTS:
            name = f"binskel_{proj}_{opt}.jsonl"
            out.append((f"{proj}_{opt}", Path(name)))
    for proj in PROJECTS_MULTIBIN:
        for opt in OPTS:
            name = f"binskel_{proj}_{opt}_multibin.jsonl"
            out.append((f"{proj}_{opt}_multibin", Path(name)))
    assert len(out) == 45
    return out


def project_from_logical(logical: str) -> str:
    """zlib_o0 → zlib；openssh_o0_multibin → openssh。"""
    if logical.endswith("_multibin"):
        base = logical[: -len("_multibin")].strip("_")  # e.g. openssh_o0
        return base.rsplit("_", 1)[0]
    return logical.rsplit("_", 1)[0]


def load_roots(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    roots: Dict[str, str] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        roots[k] = v.strip()
    return roots


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    augment_py = root / "scripts" / "augment_binskel_sft.py"

    ap = argparse.ArgumentParser(
        description="45 个 binskel jsonl → 追加 SFT+RL 字段，供 Stage3/4 使用",
    )
    ap.add_argument("--binskel-dir", type=Path, required=True, help="含 binskel_*.jsonl 的目录")
    ap.add_argument("--out-dir", type=Path, required=True, help="写出完整 jsonl 的目录（与输入同名）")
    ap.add_argument(
        "--roots-json",
        type=Path,
        required=True,
        help="9 个 key（zlib,curl,…）→ 源码根路径 的 JSON",
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/deepseek-coder-6.7b-base",
    )
    ap.add_argument(
        "--no-emit-rl-fields",
        action="store_true",
        help="不写入 Stage4 可选结构字段（默认会写）",
    )
    ap.add_argument(
        "--keep-full-labels",
        action="store_true",
        help="保留整段 labels（默认仅监督 FILL 槽位真值，并屏蔽 prompt）",
    )
    ap.add_argument(
        "--save-tokenizer-dir",
        type=Path,
        default=None,
        help="将第一次成功 augment 时的 tokenizer 存到此目录（Stage3 应用 --tokenizer-dir 指向这里）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    ap.add_argument("--max-lines", type=int, default=None, help="每条 jsonl 仅处理前 N 行（调试）")
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="传给 augment_binskel_sft（0=不传参，与 rebuild_loss_datasets_full 未设 SFT_MAX_SEQ_LEN 时一致）",
    )
    args = ap.parse_args()

    if not augment_py.is_file():
        logger.error("找不到 %s", augment_py)
        sys.exit(1)

    binskel_dir = args.binskel_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    roots = load_roots(args.roots_json.expanduser().resolve())

    missing_proj = []
    for p in list(PROJECTS_SIMPLE) + list(PROJECTS_MULTIBIN):
        if p not in roots:
            missing_proj.append(p)
    if missing_proj:
        logger.error("roots-json 缺少项目: %s", ", ".join(missing_proj))
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = expected_pairs()

    planned: List[Tuple[Path, Path, Path]] = []
    for logical, fname in pairs:
        proj = project_from_logical(logical)
        src_root = Path(roots[proj]).expanduser().resolve()
        inp = binskel_dir / fname
        outp = out_dir / fname
        planned.append((inp, outp, src_root))

    if args.dry_run:
        for idx, (inp, outp, sr) in enumerate(planned):
            logical = pairs[idx][0]
            ex = "存在" if inp.is_file() else "缺失"
            logger.info("[%s] %s → %s (source-root=%s) 输入%s", logical, inp, outp, sr, ex)
        logger.info("dry-run: 共 %d 个文件", len(planned))
        return

    saved_tok = False
    ok_n = skip_n = fail_n = 0
    for idx, (inp, outp, src_root) in enumerate(planned):
        logical = f"{pairs[idx][0]}"
        if not inp.is_file():
            logger.warning("跳过（无输入）: %s", inp)
            skip_n += 1
            continue
        if not src_root.is_dir():
            logger.error("源码根不存在，无法 augment: %s (项目 %s)", src_root, project_from_logical(logical))
            fail_n += 1
            continue

        cmd = [
            sys.executable,
            str(augment_py),
            "--input",
            str(inp),
            "--output",
            str(outp),
            "--tokenizer",
            args.tokenizer,
            "--source-root",
            str(src_root),
        ]
        if not args.no_emit_rl_fields:
            cmd.append("--emit-rl-fields")
        if not args.keep_full_labels:
            cmd.extend(["--mask-prompt-labels", "--mask-non-fill-labels"])
        if args.max_lines is not None:
            cmd.extend(["--max-lines", str(args.max_lines)])
        if args.max_seq_len and args.max_seq_len > 0:
            cmd.extend(["--max-seq-len", str(args.max_seq_len)])

        save_tok_dir = args.save_tokenizer_dir
        tok_this_run = False
        if save_tok_dir is not None and not saved_tok:
            sd = save_tok_dir.expanduser().resolve()
            sd.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--save-tokenizer", str(sd)])
            tok_this_run = True

        logger.info("(%d/45) %s", idx + 1, logical)
        try:
            subprocess.run(cmd, cwd=str(root), check=True)
            ok_n += 1
            if tok_this_run:
                saved_tok = True
        except subprocess.CalledProcessError as e:
            logger.error("失败: %s returncode=%s", logical, e.returncode)
            fail_n += 1

    logger.info(
        "完成: 成功 %d, 跳过(无输入) %d, 失败 %d → 输出目录 %s",
        ok_n,
        skip_n,
        fail_n,
        out_dir,
    )
    if args.save_tokenizer_dir and saved_tok:
        logger.info("Tokenizer 已保存（首次成功时）: %s", args.save_tokenizer_dir.expanduser().resolve())


if __name__ == "__main__":
    main()

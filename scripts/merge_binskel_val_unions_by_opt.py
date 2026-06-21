#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 Stage2 课程顺序（O0 → O1 → O2 → O3 → **Os 最后**；即 0123s）用「并集」合并
单档 binskel .jsonl，得到 5 个验证集文件::

  B_val_o0.jsonl         — O0
  B_val_o01.jsonl        — O0 ∪ O1
  B_val_o012.jsonl       — O0 ∪ O1 ∪ O2
  B_val_o0123.jsonl      — O0 ∪ O1 ∪ O2 ∪ O3
  B_val_o0123s.jsonl     — 五档全集（含 Os）

合并规则：按顺序逐文件**拼接行**（jsonl 一行一样本），不排序、不删重。

用法::

  python scripts/merge_binskel_val_unions_by_opt.py \\
    --o0 "E:/ABCD_datasets/AB/binskel_AB_busybox_o0.jsonl" \\
    --o1 "E:/ABCD_datasets/AB/binskel_AB_busybox_o1.jsonl" \\
    --o2 "E:/ABCD_datasets/AB/binskel_AB_busybox_o2.jsonl" \\
    --o3 "E:/ABCD_datasets/AB/binskel_AB_busybox_o3.jsonl" \\
    --os "E:/ABCD_datasets/AB/binskel_AB_busybox_os.jsonl" \\
    --out-dir "E:/ABCD_datasets/AB"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 输出 basename（上传服务器时即用这些名）
OUT_NAMES: list[tuple[str, list[str]]] = [
    ("B_val_o0.jsonl", ["O0"]),
    ("B_val_o01.jsonl", ["O0", "O1"]),
    ("B_val_o012.jsonl", ["O0", "O1", "O2"]),
    ("B_val_o0123.jsonl", ["O0", "O1", "O2", "O3"]),
    ("B_val_o0123s.jsonl", ["O0", "O1", "O2", "O3", "Os"]),
]


def _append_jsonl(in_path: Path, out_fp) -> int:
    n = 0
    with in_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out_fp.write(s + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按课程序 0123s 并单档 jsonl 为 5 个 B_val_o*.jsonl"
    )
    ap.add_argument("--o0", type=Path, required=True)
    ap.add_argument("--o1", type=Path, required=True)
    ap.add_argument("--o2", type=Path, required=True)
    ap.add_argument("--o3", type=Path, required=True)
    ap.add_argument("--os", type=Path, required=True, dest="os_")
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="输出目录",
    )
    args = ap.parse_args()

    key_path: dict[str, Path] = {
        "O0": args.o0.expanduser().resolve(),
        "O1": args.o1.expanduser().resolve(),
        "O2": args.o2.expanduser().resolve(),
        "O3": args.o3.expanduser().resolve(),
        "Os": args.os_.expanduser().resolve(),
    }
    for tag, p in key_path.items():
        if not p.is_file():
            print(f"ERROR: 文件不存在 ({tag}): {p}", file=sys.stderr)
            sys.exit(1)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for out_name, tags in OUT_NAMES:
        out_path = out_dir / out_name
        tot = 0
        with out_path.open("w", encoding="utf-8", newline="\n") as w:
            for t in tags:
                tot += _append_jsonl(key_path[t], w)
        parts = [key_path[t].name for t in tags]
        print(f"OK  {out_path}  行数={tot}  来源={parts}")

    print(f"\n输出目录: {out_dir}  命名: B_val_o0 / B_val_o01 / B_val_o012 / B_val_o0123 / B_val_o0123s")


if __name__ == "__main__":
    main()

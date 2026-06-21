#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 binskel jsonl：行数(样本)、文件大小。可选：--validate 对前 N 条跑 validate_alignment 逻辑。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path, help="binskel_*.jsonl")
    ap.add_argument("--markdown", action="store_true", help="输出 Markdown 表")
    args = ap.parse_args()

    rows = []
    for p in args.paths:
        p = p.expanduser()
        if not p.is_file():
            rows.append((p.name, "MISSING", 0, ""))
            continue
        n = 0
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    n += 1
        sz = p.stat().st_size
        rows.append((p.name, "OK", n, f"{sz / (1024 * 1024):.1f} MB"))

    if args.markdown:
        print("| 数据集 | 状态 | 样本数 | 大小 |")
        print("|--------|------|--------|------|")
        for name, st, n, sz in rows:
            print(f"| {name} | {st} | {n} | {sz} |")
    else:
        print(f"{'数据集':<40} {'状态':<8} {'样本':>8} {'大小':>12}")
        for name, st, n, sz in rows:
            print(f"{name:<40} {st:<8} {n:>8} {sz:>12}")
        tot = sum(r[2] for r in rows if r[1] == "OK")
        print("-" * 72)
        print(f"{'合计(已存在文件)':<40} {'':8} {tot:>8}")


if __name__ == "__main__":
    main()

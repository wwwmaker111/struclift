#!/usr/bin/env python3
"""汇总 full_bb_semantic_audit 输出的 jsonl。"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    args = ap.parse_args()
    by_f: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tot: dict[str, int] = defaultdict(int)
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        f = r["func_name"]
        t = r["semantic_tier"]
        by_f[f][t] += 1
        tot[t] += 1
    print("=== 全局 ===")
    for k in sorted(tot, key=lambda x: (-tot[x], x)):
        print(f"  {k}: {tot[k]}")
    print()
    print("=== 按函数: HI_CONFLICT / COARSE / UNALIGNED / NO_ASM / RULES_CLEAN ===")
    for f in sorted(by_f.keys()):
        d = by_f[f]
        print(
            f"{f:30s}  HI={d['HI_CONFLICT']:2d}  "
            f"COARSE={d['COARSE_ALIGN']:3d}  NA={d['UNALIGNED']:2d}  "
            f"no_asm={d['ALIGNED_NO_ASM']:2d}  clean={d['RULES_CLEAN']:3d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

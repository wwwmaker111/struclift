#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按长度上限过滤 binskel JSONL，剔除极端长样本。

用法::

  python scripts/filter_jsonl_by_length.py \
      --in  /data/chaoni/WQM/datasets/AB_train.jsonl \
      --out /data/chaoni/WQM/datasets/AB_train_p99.jsonl \
      --max-instrs 560 \
      --max-bbs 119 \
      --max-tokens 2856 \
      --max-stmts 62 \
      --max-edges 175

只需跑一次，之后训练 ``--jsonl`` 指向 ``--out`` 即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--max-instrs", type=int, default=None, help="bin_opcodes 长度上限")
    ap.add_argument("--max-bbs", type=int, default=None, help="bin_block_lengths 长度上限")
    ap.add_argument("--max-tokens", type=int, default=None, help="src_token_ids 长度上限")
    ap.add_argument("--max-stmts", type=int, default=None, help="src_stmt_lengths 长度上限")
    ap.add_argument("--max-edges", type=int, default=None, help="bin_edge_index 边数上限")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.is_file():
        print(f"[error] 输入不存在: {src}", file=sys.stderr)
        sys.exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    reasons: dict = {}

    def _check(d: dict) -> str:
        if args.max_instrs is not None and len(d.get("bin_opcodes", [])) > args.max_instrs:
            return "instrs"
        if args.max_bbs is not None and len(d.get("bin_block_lengths", [])) > args.max_bbs:
            return "bbs"
        if args.max_tokens is not None and len(d.get("src_token_ids", [])) > args.max_tokens:
            return "tokens"
        if args.max_stmts is not None and len(d.get("src_stmt_lengths", [])) > args.max_stmts:
            return "stmts"
        if args.max_edges is not None:
            ei = d.get("bin_edge_index") or [[]]
            n_e = len(ei[0]) if ei and ei[0] else 0
            if n_e > args.max_edges:
                return "edges"
        return ""

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for i, raw in enumerate(fin):
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                dropped += 1
                reasons["parse"] = reasons.get("parse", 0) + 1
                continue
            reason = _check(d)
            if reason:
                dropped += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            fout.write(raw if raw.endswith(b"\n") else raw + b"\n")
            kept += 1
            if (i + 1) % 100000 == 0:
                print(f"  processed {i+1} lines  kept={kept} dropped={dropped}", flush=True)

    total = kept + dropped
    print(f"\n完成：{src.name} -> {dst.name}")
    print(f"  原始样本: {total}")
    print(f"  保留    : {kept} ({100.0*kept/max(total,1):.2f}%)")
    print(f"  剔除    : {dropped} ({100.0*dropped/max(total,1):.2f}%)")
    if reasons:
        print(f"  剔除原因:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {k:10s}: {v}")
    print(f"\n输出路径: {dst}")


if __name__ == "__main__":
    main()

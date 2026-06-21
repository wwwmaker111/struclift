#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 O0–O3 四份 binskel jsonl 中，选取「同名+同 source_file」的函数，对比各档字段是否一致。

用法（FFmpeg 示例）:
  python3 scripts/sample_compare_binskel_opts.py \\
      --dir ~/struclift_wsl/datasets/ffmpeg_binskel \\
      --prefix binskel_ffmpeg

  python3 scripts/sample_compare_binskel_opts.py \\
      --o0 a/o0.jsonl --o1 a/o1.jsonl --o2 a/o2.jsonl --o3 a/o3.jsonl

默认抽 5 个函数；对比「对齐/统计相关字段」是否与 O0 逐档相同（忽略 opt_level、binary 路径字面量）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 与 validate_alignment / diff 指纹思路一致：这些变了才算「各档真不同」
COMPARE_KEYS = (
    "bin_block_lengths",
    "bin_edge_index",
    "bin_edge_type",
    "bin_opcodes",
    "block_src_lines",
    "block_src_confidence",
    "alignment_matrix",
    "alignment_confidence",
    "bin_struct_features",
    "src_token_ids",
    "src_stmt_lengths",
    "bin_operand_types",
    "bin_operand_values",
)


def _key(rec: dict) -> Tuple[str, str]:
    return (str(rec.get("func_name") or ""), str(rec.get("source_file") or ""))


def _norm(d: dict) -> str:
    sub = {k: d.get(k) for k in COMPARE_KEYS}
    return json.dumps(sub, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_index(path: Path) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[_key(d)] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="四档 binskel 抽相同函数对比字段是否一致")
    ap.add_argument("--dir", type=Path, default=None, help="含 o0..o3 jsonl 的目录")
    ap.add_argument(
        "--prefix",
        default="binskel_ffmpeg",
        help="文件名前缀，默认 binskel_ffmpeg → binskel_ffmpeg_o0.jsonl",
    )
    ap.add_argument("--o0", "--O0", dest="p0", type=Path, default=None)
    ap.add_argument("--o1", "--O1", dest="p1", type=Path, default=None)
    ap.add_argument("--o2", "--O2", dest="p2", type=Path, default=None)
    ap.add_argument("--o3", "--O3", dest="p3", type=Path, default=None)
    ap.add_argument("-n", "--num", type=int, default=5, help="抽取函数个数")
    ap.add_argument(
        "--show-binary",
        action="store_true",
        help="打印各档 binary 字段（路径通常每档不同，仅作检查）",
    )
    args = ap.parse_args()

    if args.dir:
        d = args.dir.expanduser().resolve()
        paths = [d / f"{args.prefix}_o{k}.jsonl" for k in range(4)]
    elif args.p0 and args.p1 and args.p2 and args.p3:
        paths = [args.p0.expanduser().resolve(), args.p1.expanduser().resolve(), args.p2.expanduser().resolve(), args.p3.expanduser().resolve()]
    else:
        print("请指定 --dir + --prefix，或同时指定 --o0 --o1 --o2 --o3", file=sys.stderr)
        sys.exit(1)

    for p in paths:
        if not p.is_file():
            print(f"错误: 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    tags = ["O0", "O1", "O2", "O3"]
    indices = [load_index(p) for p in paths]
    common = set(indices[0].keys())
    for idx in indices[1:]:
        common &= set(idx.keys())
    common_list = sorted(common)
    print(f"各档样本数: {[len(i) for i in indices]}")
    print(f"四档键交集 (func_name+source_file): {len(common)}")
    if not common:
        print("无交集，无法对比。")
        sys.exit(1)

    pick = common_list[: args.num]
    if len(pick) < args.num:
        print(f"注意: 交集只有 {len(pick)} 个，少于请求的 {args.num} 个\n")

    print("\n对比字段: " + ", ".join(COMPARE_KEYS))
    print("(忽略 func_name/source_file/opt_level/binary 字面量；--show-binary 可显示 binary)\n")

    for k in pick:
        fn, sf = k
        print("=" * 72)
        print(f"func_name={fn!r}")
        print(f"source_file={sf!r}")
        recs = [indices[i][k] for i in range(4)]
        if args.show_binary:
            for t, r in zip(tags, recs):
                print(f"  [{t}] binary={r.get('binary')!r}")
        norms = [_norm(r) for r in recs]
        all_same = len(set(norms)) == 1
        if all_same:
            print("  结果: 四档 **COMPARE_KEYS 子集完全相同**（与 O0 一致）")
        else:
            print("  结果: 四档 **存在差异**（至少有一档与 O0 不同）")
            ref = norms[0]
            for t, n in zip(tags, norms):
                mark = "==" if n == ref else "!="
                print(f"    vs O0  [{t}] {mark}")
        print()


if __name__ == "__main__":
    main()

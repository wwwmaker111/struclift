#!/usr/bin/env python3
"""从完整 binskel jsonl 抽取 N 条，只保留推理所需的二进制侧字段。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BIN_KEYS = (
    "binary",
    "func_name",
    "opt_level",
    "arch",
    "compiler",
    "bin_opcodes",
    "bin_operand_types",
    "bin_operand_values",
    "bin_block_lengths",
    "bin_edge_index",
    "bin_edge_type",
    "bin_struct_features",
    "pattern_labels",
)


def strip_bin_only(rec: Dict[str, Any], *, opt_override: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"infer_only": True}
    for k in BIN_KEYS:
        if k in rec:
            out[k] = rec[k]
    if opt_override:
        out["opt_level"] = opt_override
    if "arch" not in out:
        out["arch"] = rec.get("arch") or rec.get("architecture") or "x86_64"
    if "compiler" not in out:
        out["compiler"] = rec.get("compiler") or "unknown (stripped)"
    return out


def extract_gt_for_infer_jsonl(
    full_jsonl: Path,
    infer_jsonl: Path,
    output: Path,
) -> int:
    """按 infer jsonl 里的 ``func_name``，从完整 jsonl 抽出带源码的原始行（小体积 gt）。"""
    names: List[str] = []
    with infer_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = rec.get("func_name")
            if isinstance(fn, str) and fn:
                names.append(fn)
    if not names:
        return 0
    want = set(names)
    found: Dict[str, Dict[str, Any]] = {}
    with full_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if len(found) >= len(want):
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = rec.get("func_name")
            if isinstance(fn, str) and fn in want and fn not in found:
                found[fn] = rec
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for fn in names:
            rec = found.get(fn)
            if rec is None:
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    missing = [fn for fn in names if fn not in found]
    if missing:
        print(f"警告: 未在完整 jsonl 找到 {len(missing)} 个函数: {missing[:5]}", file=sys.stderr)
    print(f"写出 {len(names) - len(missing)} 条 gt → {output}", file=sys.stderr)
    return len(names) - len(missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--extract-gt-from",
        type=Path,
        default=None,
        help="完整 binskel jsonl；与 --infer-jsonl 同用，输出带源码 gt（忽略其它模式）",
    )
    ap.add_argument(
        "--infer-jsonl",
        type=Path,
        default=None,
        help="与 --extract-gt-from 搭配：按其中 func_name 抽 gt",
    )
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--func-names", type=str, default=None, help="逗号分隔，优先这些函数")
    args = ap.parse_args()

    if args.extract_gt_from is not None:
        if args.infer_jsonl is None:
            ap.error("--extract-gt-from 需要同时指定 --infer-jsonl")
        n = extract_gt_for_infer_jsonl(
            args.extract_gt_from.expanduser().resolve(),
            args.infer_jsonl.expanduser().resolve(),
            args.output.expanduser().resolve(),
        )
        if n == 0:
            sys.exit(1)
        return

    want = None
    if args.func_names:
        want = {x.strip() for x in args.func_names.split(",") if x.strip()}

    picked: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if want and rec.get("func_name") in want:
                picked.append(strip_bin_only(rec))
            elif not want:
                pool.append(rec)
            if want and len(picked) >= args.n:
                break

    if not want:
        rng = random.Random(args.seed)
        if args.shuffle:
            rng.shuffle(pool)
        for rec in pool[: args.n]:
            picked.append(strip_bin_only(rec))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for rec in picked:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"写出 {len(picked)} 条 → {args.output}", file=sys.stderr)
    for i, r in enumerate(picked):
        print(f"  [{i}] {r.get('func_name')} n_bb={len(r.get('bin_block_lengths', []))}", file=sys.stderr)
    if len(picked) < args.n:
        sys.exit(1)


if __name__ == "__main__":
    main()

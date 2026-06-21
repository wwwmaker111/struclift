#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge real FunctionIR/SSA enrichment rows back into Module-B candidates.

``enrich_module_b_v3_real_function_ir100.py`` rebuilds real binary-derived
FunctionIR on the source/enriched JSONL line that a Module-B row references via
``input_line``.  Module-C cannot consume that source row directly because the
verified skeleton, slot grounding, and candidate metadata live in the Module-B
fullcandidates JSONL.  This bridge copies the real IR fields back onto the
corresponding Module-B rows while preserving all Module-B data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


IR_FIELDS = (
    "function_ir",
    "lifted_instructions",
    "ssa_values",
    "def_use_graph",
    "instruction_provenance",
    "branch_metadata",
    "function_ir_enrichment",
    "module_b_v3_binary_path_recovery",
)


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def build_real_ir_index(path: Path, limit: int = 0) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for line_no, row in iter_jsonl(path, limit=limit):
        keys = []
        for raw in (
            row.get("input_line"),
            row.get("module_b_teacher_input_line"),
            (row.get("function_ir_enrichment") or {}).get("input_line")
            if isinstance(row.get("function_ir_enrichment"), dict)
            else None,
        ):
            key = safe_int(raw, 0)
            if key > 0:
                keys.append(key)
        # Older enrichment files preserved source JSONL line numbering by
        # writing every non-target row.  Newer target-only files do not, so
        # output line_no is only a last-resort compatibility key.
        keys.append(line_no)
        for key in keys:
            current = out.get(key)
            if current is None or row_has_real_ir(row) or not row_has_real_ir(current):
                out[key] = row
    return out


def row_has_real_ir(row: Dict[str, Any]) -> bool:
    enrichment = row.get("function_ir_enrichment") if isinstance(row.get("function_ir_enrichment"), dict) else {}
    function_ir = row.get("function_ir") if isinstance(row.get("function_ir"), dict) else {}
    return bool(
        enrichment.get("real_lifted_available")
        or enrichment.get("real_ssa_available")
        or enrichment.get("real_def_use_available")
        or str(enrichment.get("mode") or "").startswith("binary")
        or function_ir.get("lifted_instructions_available")
        or function_ir.get("ssa_available")
        or function_ir.get("def_use_available")
    )


def merge(args: argparse.Namespace) -> Counter:
    module_b_path = Path(args.module_b_jsonl).expanduser().resolve()
    real_ir_path = Path(args.real_ir_jsonl).expanduser().resolve()
    out_path = Path(args.out_jsonl).expanduser().resolve()
    report_path = Path(args.out_txt).expanduser().resolve() if args.out_txt else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    real_index = build_real_ir_index(real_ir_path, limit=0)
    stats: Counter = Counter()
    mode_dist: Counter = Counter()
    missing_reason: Counter = Counter()

    with module_b_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for row_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            if int(args.limit) > 0 and stats["rows_seen"] >= int(args.limit):
                break
            row = json.loads(line)
            stats["rows_seen"] += 1
            input_line = safe_int(row.get("input_line"), row_no)
            real_row = real_index.get(input_line)
            out = dict(row)
            if real_row is None:
                stats["missing_real_ir_row"] += 1
                missing_reason["no_real_ir_row_for_input_line"] += 1
            else:
                copied = 0
                for field in IR_FIELDS:
                    value = real_row.get(field)
                    if value not in (None, [], {}):
                        out[field] = value
                        copied += 1
                if copied:
                    stats["rows_merged"] += 1
                    enrichment = out.get("function_ir_enrichment") if isinstance(out.get("function_ir_enrichment"), dict) else {}
                    mode = str(enrichment.get("mode") or "unknown")
                    mode_dist[mode] += 1
                    if row_has_real_ir(out):
                        stats["rows_with_real_ir"] += 1
                    else:
                        stats["rows_with_proxy_or_partial_ir"] += 1
                else:
                    stats["real_ir_row_without_mergeable_fields"] += 1
                    missing_reason["real_ir_row_without_mergeable_fields"] += 1
            out.setdefault("module_c_real_function_ir_merge", {})
            out["module_c_real_function_ir_merge"] = {
                "schema": "ModuleC.RealFunctionIRMerge.v0",
                "source_real_ir_jsonl": str(real_ir_path),
                "input_line": int(input_line),
                "merged": bool(real_row is not None),
                "has_real_ir": bool(row_has_real_ir(out)),
            }
            dst.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "=== Module-B real FunctionIR merge ===",
        f"module_b_jsonl = {module_b_path}",
        f"real_ir_jsonl = {real_ir_path}",
        f"out_jsonl = {out_path}",
        "",
        f"rows_seen = {stats['rows_seen']}",
        f"rows_merged = {pct(int(stats['rows_merged']), int(stats['rows_seen']))}",
        f"rows_with_real_ir = {pct(int(stats['rows_with_real_ir']), int(stats['rows_seen']))}",
        f"rows_with_proxy_or_partial_ir = {pct(int(stats['rows_with_proxy_or_partial_ir']), int(stats['rows_seen']))}",
        f"missing_real_ir_row = {stats['missing_real_ir_row']}",
        f"real_ir_row_without_mergeable_fields = {stats['real_ir_row_without_mergeable_fields']}",
        f"mode_dist = {mode_dist.most_common()}",
        f"missing_reason_dist = {missing_reason.most_common()}",
    ]
    text = "\n".join(lines) + "\n"
    if report_path:
        report_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module-b-jsonl", required=True)
    ap.add_argument("--real-ir-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    merge(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export source functions and AB skeletons for manual control-flow comparison.

Selects functions whose active source side has at least one high-level control
statement (if/loop/switch), then writes each complete real source function and
the complete AB-generated skeleton side by side.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402


PATTERN_NAMES = {
    0: "sequential",
    1: "if_header",
    2: "if_then_body",
    3: "if_else_body",
    4: "loop_header",
    5: "loop_body",
    6: "loop_exit",
    7: "switch_header",
    8: "switch_case_body",
    9: "function_prologue_epilogue",
}


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _source_function_with_numbers(record: Dict[str, Any]) -> Tuple[str, Optional[Tuple[int, int]]]:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt:
        return "<source file unreadable>", None
    bounds = repack._function_body_line_range(record, lt)
    if bounds is None:
        return "<function bounds unavailable>", None
    lo, hi = bounds
    out: List[str] = []
    for ln in range(lo, hi + 1):
        if ln in lt:
            out.append(f"{ln:5d}: {lt[ln]}")
    return ("\n".join(out) if out else "<empty source range>"), bounds


def _pattern_label_summary(record: Dict[str, Any]) -> str:
    raw = record.get("pattern_labels")
    if not isinstance(raw, list):
        return "<missing pattern_labels>"
    counts: Dict[int, int] = {}
    for x in raw:
        try:
            lab = int(x)
        except Exception:
            continue
        counts[lab] = counts.get(lab, 0) + 1
    parts = [
        f"{PATTERN_NAMES.get(lab, f'unknown_{lab}')}={counts[lab]}"
        for lab in sorted(counts)
    ]
    return ", ".join(parts) if parts else "<empty pattern_labels>"


def _pattern_label_sequence(record: Dict[str, Any], max_items: int = 160) -> str:
    raw = record.get("pattern_labels")
    if not isinstance(raw, list):
        return "<missing pattern_labels>"
    vals: List[str] = []
    for i, x in enumerate(raw[:max_items]):
        try:
            lab = int(x)
        except Exception:
            vals.append(f"BB_{i}:unknown({x})")
            continue
        vals.append(f"BB_{i}:{PATTERN_NAMES.get(lab, f'unknown_{lab}')}")
    if len(raw) > max_items:
        vals.append("...")
    return "\n".join(vals)


def _write_example(
    wf: Any,
    idx: int,
    line_no: int,
    row: Dict[str, Any],
    audit: Dict[str, Any],
    source_text: str,
    bounds: Tuple[int, int],
) -> None:
    skeleton = effect._skeleton_from_record(row)
    wf.write("\n" + "=" * 120 + "\n")
    wf.write(
        f"example = {idx}  input_line = {line_no}  func = {row.get('func_name')}  "
        f"verdict = {audit.get('verdict')}\n"
    )
    wf.write(f"source_file = {row.get('source_file')}\n")
    wf.write(f"source_bounds = {bounds[0]}-{bounds[1]}\n")
    wf.write(
        f"bb = {audit.get('n_bb')}  src_rows = {audit.get('source_statement_rows')}  "
        f"src_ref = {audit.get('source_outline_source')}  "
        f"parse = {audit.get('source_parse_status')}\n"
    )
    wf.write(
        f"ctrl(pred/source) = {audit.get('pred_control_count')}/"
        f"{audit.get('source_control_count')}  "
        f"gaps(if,loop,switch)=({audit.get('if_gap')},{audit.get('loop_gap')},"
        f"{audit.get('switch_gap')})  "
        f"sim(type,depth)={float(audit.get('type_seq_sim') or 0.0):.3f}/"
        f"{float(audit.get('type_depth_seq_sim') or 0.0):.3f}\n"
    )
    wf.write(
        f"bb_cover = {audit.get('bb_covered')}/{audit.get('n_bb')}  "
        f"reasons = {audit.get('reasons')}\n"
    )
    wf.write(f"10class_pattern_label_counts = {_pattern_label_summary(row)}\n")

    wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
    wf.write("\n".join(audit.get("source_outline") or ["<none>"]) + "\n")
    wf.write("\n-- AB SKELETON CONTROL OUTLINE --\n")
    wf.write("\n".join(audit.get("pred_outline") or ["<none>"]) + "\n")

    wf.write("\n-- 10CLASS BB LABEL SEQUENCE --\n")
    wf.write(_pattern_label_sequence(row) + "\n")

    wf.write("\n-- REAL SOURCE FUNCTION --\n")
    wf.write(source_text.rstrip() + "\n")

    wf.write("\n-- AB GENERATED FULL SKELETON --\n")
    wf.write((skeleton or "<missing skeleton>").rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", "--input", dest="jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--num-examples", type=int, default=50)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum input lines to scan. 0 means scan the whole jsonl.",
    )
    ap.add_argument(
        "--source-control-filter",
        choices=["raw", "bb-start-line", "bb-range"],
        default="bb-start-line",
    )
    ap.add_argument(
        "--min-source-controls",
        type=int,
        default=1,
        help="Require at least this many active source control statements.",
    )
    ap.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Do not deduplicate repeated source_file/function/bounds samples.",
    )
    ap.add_argument(
        "--random-sample",
        action="store_true",
        help="Reservoir-sample eligible functions instead of taking the first N.",
    )
    ap.add_argument("--seed", type=int, default=527)
    ap.add_argument(
        "--include-fallback-source",
        action="store_true",
        help=(
            "Include samples whose source control outline came from active-line "
            "fallback. Full source bounds are still required."
        ),
    )
    args = ap.parse_args()

    chosen: List[Tuple[int, Dict[str, Any], Dict[str, Any], str, Tuple[int, int]]] = []
    seen = set()
    scanned = 0
    eligible = 0
    skipped: collections.Counter[str] = collections.Counter()
    rng = random.Random(int(args.seed))

    for line_no, row in _iter_jsonl(args.jsonl.expanduser().resolve(), int(args.limit)):
        scanned += 1
        audit = effect._audit_row(
            line_no,
            row,
            strict_filled_terminals=False,
            source_control_filter=str(args.source_control_filter),
        )
        if int(audit.get("source_control_count") or 0) < int(args.min_source_controls):
            skipped["no_source_control"] += 1
            continue
        if not args.include_fallback_source and audit.get("source_outline_source") != "statement_rows":
            skipped["fallback_source_outline"] += 1
            continue
        source_text, bounds = _source_function_with_numbers(row)
        if bounds is None:
            skipped["source_bounds_unavailable"] += 1
            continue
        key = (str(row.get("source_file") or ""), str(row.get("func_name") or ""), bounds)
        if not args.allow_duplicates and key in seen:
            skipped["duplicate_function"] += 1
            continue
        seen.add(key)
        item = (line_no, row, audit, source_text, bounds)
        eligible += 1
        if args.random_sample:
            if len(chosen) < int(args.num_examples):
                chosen.append(item)
            else:
                j = rng.randrange(eligible)
                if j < int(args.num_examples):
                    chosen[j] = item
        else:
            chosen.append(item)
        if not args.random_sample and len(chosen) >= int(args.num_examples):
            break

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== STAGE2AB CONTROL-FLOW SOURCE VS FULL SKELETON EXPORT =====\n")
        wf.write(f"input = {args.jsonl.expanduser().resolve()}\n")
        wf.write(f"scanned = {scanned}\n")
        wf.write(f"eligible = {eligible}\n")
        wf.write(f"exported = {len(chosen)}\n")
        wf.write(f"requested = {int(args.num_examples)}\n")
        wf.write(f"source_control_filter = {args.source_control_filter}\n")
        wf.write(f"min_source_controls = {int(args.min_source_controls)}\n")
        wf.write(f"allow_duplicates = {bool(args.allow_duplicates)}\n")
        wf.write(f"include_fallback_source = {bool(args.include_fallback_source)}\n")
        wf.write(f"random_sample = {bool(args.random_sample)}\n")
        wf.write(f"seed = {int(args.seed)}\n")
        wf.write(f"skipped = {dict(skipped)}\n")
        for idx, (line_no, row, audit, source_text, bounds) in enumerate(chosen):
            _write_example(wf, idx, line_no, row, audit, source_text, bounds)

    print(f"saved_to = {out}")
    print(f"scanned = {scanned}")
    print(f"eligible = {eligible}")
    print(f"exported = {len(chosen)}")
    print(f"skipped = {dict(skipped)}")


if __name__ == "__main__":
    main()

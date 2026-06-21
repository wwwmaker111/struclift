#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export representative Stage2AB skeleton examples.

This script reuses ``audit_stage2ab_skeleton_effect`` and writes side-by-side
source function text versus AB skeleton.  By default it exports the verdicts
that correspond to the suspicious bucket, but ``--verdict`` can also be used to
sample OK buckets such as STRAIGHTLINE_OK or SURFACE_MATCH.

It is intended for manual comparison, not for training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402


SUSPICIOUS_VERDICTS = {
    "CONTROL_COUNT_MISMATCH",
    "LOOP_SWITCH_MISMATCH",
    "CONTROL_ORDER_MISMATCH",
}

ALL_VERDICTS = {
    "STRAIGHTLINE_OK",
    "SURFACE_MATCH",
    "SURFACE_DIFF_PROBABLY_EQUIV",
    "NEEDS_PATH_AUDIT",
    "CONTROL_COUNT_MISMATCH",
    "LOOP_SWITCH_MISMATCH",
    "CONTROL_ORDER_MISMATCH",
    "UNJUDGEABLE",
}

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


def _source_function_with_numbers(record: Dict[str, Any]) -> str:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt:
        return "<source file unreadable>"
    bounds = repack._function_body_line_range(record, lt)
    if bounds is None:
        return "<function bounds unavailable>"
    lo, hi = bounds
    out: List[str] = []
    for ln in range(lo, hi + 1):
        if ln in lt:
            out.append(f"{ln:5d}: {lt[ln]}")
    return "\n".join(out) if out else "<empty source range>"


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


def _risk_key(item: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[int, int, int, int]:
    audit, _row = item
    loop_switch_gap = abs(int(audit.get("loop_gap") or 0)) + abs(
        int(audit.get("switch_gap") or 0)
    )
    if_gap = abs(int(audit.get("if_gap") or 0))
    ctrl_gap = abs(
        int(audit.get("pred_control_count") or 0)
        - int(audit.get("source_control_count") or 0)
    )
    n_bb = int(audit.get("n_bb") or 0)
    return (loop_switch_gap, if_gap, ctrl_gap, n_bb)


def _write_example(wf: Any, idx: int, audit: Dict[str, Any], row: Dict[str, Any]) -> None:
    skeleton = str(audit.get("skeleton_excerpt") or "")
    if not skeleton:
        skeleton = effect._skeleton_from_record(row)

    wf.write("\n" + "=" * 120 + "\n")
    wf.write(
        f"example = {idx}  line = {audit.get('line')}  "
        f"func = {audit.get('func_name')}  verdict = {audit.get('verdict')}\n"
    )
    wf.write(f"source_file = {audit.get('source_file')}\n")
    wf.write(
        f"bb = {audit.get('n_bb')}  src_rows = {audit.get('source_statement_rows')}  "
        f"src_ref = {audit.get('source_outline_source')}  "
        f"parse = {audit.get('source_parse_status')}  "
        f"ctrl(pred/source) = {audit.get('pred_control_count')}/{audit.get('source_control_count')}\n"
    )
    wf.write(f"10class_pattern_labels = {_pattern_label_summary(row)}\n")
    wf.write(
        f"gaps(if,loop,switch) = "
        f"({audit.get('if_gap')},{audit.get('loop_gap')},{audit.get('switch_gap')})  "
        f"sim(type,depth) = {float(audit.get('type_seq_sim') or 0.0):.3f}/"
        f"{float(audit.get('type_depth_seq_sim') or 0.0):.3f}\n"
    )
    wf.write(
        f"bb_cover = {audit.get('bb_covered')}/{audit.get('n_bb')}  "
        f"reasons = {audit.get('reasons')}\n"
    )

    wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
    wf.write("\n".join(audit.get("source_outline") or ["<none>"]) + "\n")
    wf.write("\n-- AB SKELETON CONTROL OUTLINE --\n")
    wf.write("\n".join(audit.get("pred_outline") or ["<none>"]) + "\n")

    wf.write("\n-- REAL SOURCE FUNCTION --\n")
    wf.write(_source_function_with_numbers(row) + "\n")

    wf.write("\n-- AB RECOVERED SKELETON --\n")
    wf.write(skeleton.strip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", "--input", dest="jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--num-examples", type=int, default=5)
    ap.add_argument(
        "--func",
        action="append",
        help=(
            "Export specific function name(s). Can be passed multiple times. "
            "When set, verdict filtering is ignored unless --also-filter-verdict is passed."
        ),
    )
    ap.add_argument(
        "--also-filter-verdict",
        action="store_true",
        help="When --func is set, also apply --verdict filtering.",
    )
    ap.add_argument(
        "--verdict",
        action="append",
        choices=sorted(ALL_VERDICTS),
        help=(
            "Restrict to a verdict. Can be passed multiple times. Defaults to "
            "the suspicious verdicts."
        ),
    )
    ap.add_argument(
        "--source-control-filter",
        choices=["raw", "bb-start-line", "bb-range"],
        default="bb-start-line",
        help=(
            "Filter source controls before comparing with skeleton.  Keep this "
            "aligned with audit_stage2ab_skeleton_effect.py."
        ),
    )
    args = ap.parse_args()

    selected_verdicts = set(args.verdict or SUSPICIOUS_VERDICTS)
    selected_funcs = set(str(x) for x in (args.func or []))
    candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for line_no, row in _iter_jsonl(args.jsonl.expanduser().resolve(), int(args.limit)):
        audit = effect._audit_row(
            line_no,
            row,
            strict_filled_terminals=False,
            source_control_filter=str(args.source_control_filter),
        )
        func_name = str(row.get("func_name") or "")
        if selected_funcs:
            if func_name not in selected_funcs:
                continue
            if args.also_filter_verdict and audit.get("verdict") not in selected_verdicts:
                continue
            candidates.append((audit, row))
        elif audit.get("verdict") in selected_verdicts:
            candidates.append((audit, row))

    candidates.sort(key=_risk_key, reverse=True)
    chosen = candidates[: max(0, int(args.num_examples))]

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== REPRESENTATIVE SUSPICIOUS STAGE2AB SKELETONS =====\n")
        wf.write(f"input = {args.jsonl.expanduser().resolve()}\n")
        wf.write(f"limit = {int(args.limit)}\n")
        wf.write(f"selected_verdicts = {sorted(selected_verdicts)}\n")
        wf.write(f"candidate_count = {len(candidates)}\n")
        wf.write(f"exported = {len(chosen)}\n")
        for idx, (audit, row) in enumerate(chosen):
            _write_example(wf, idx, audit, row)

    print(f"saved_to = {out}")
    print(f"candidate_count = {len(candidates)}")
    print(f"exported = {len(chosen)}")


if __name__ == "__main__":
    main()

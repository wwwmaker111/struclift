#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Stage2AB skeleton structure without evaluating Module C fills.

This script deliberately ignores generated code quality.  It asks whether the
AB-emitted skeleton has enough source-compatible structure to host GT slots:

* control count/type rough match
* FILL_COND / FILL_STMT / FILL_EXPR capacity
* source declarations that have no obvious BB line
* BBs whose source lines contain multiple source statements/controls
* M columns shared by multiple BBs

It is a diagnostic, not a final semantic metric.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Counter, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import repack_modulec_v9_to_v10 as repack  # noqa: E402


FILL_RE = re.compile(r"<FILL_([A-Z_]+):([^>]+)>")
CONTROL_RE = re.compile(r"\b(if|while|for|switch)\s*\(")

CONTROL_TYPES = {
    "if_statement",
    "while_statement",
    "for_statement",
    "do_statement",
    "switch_statement",
}
LEAF_STMT_TYPES = {
    "declaration",
    "expression_statement",
    "return_statement",
    "break_statement",
    "continue_statement",
    "goto_statement",
}


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _extract_skeleton_text(row: Dict[str, Any]) -> str:
    skel = str(row.get("sft_skeleton_text_pretty") or row.get("skeleton_text") or "")
    if skel:
        return skel
    prompt = str(row.get("sft_prompt_text_pretty") or "")
    m = re.search(r"<\|skeleton\|>\s*(.*?)\s*<\|/skeleton\|>", prompt, re.S)
    return m.group(1) if m else ""


def _slot_counts(skeleton_text: str) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    for typ, _sid in FILL_RE.findall(skeleton_text or ""):
        slot_type = typ if typ.startswith("FILL_") else f"FILL_{typ}"
        c[f"slot_{slot_type}"] += 1
        if slot_type not in {"FILL_TYPE", "FILL_PARAM"}:
            c["semantic_slots"] += 1
        if slot_type in {"FILL_STMT", "FILL_EXPR"}:
            c["stmt_capacity"] += 1
    return c


def _source_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = repack._source_all_statement_rows(row)
    if not rows:
        rows = repack._source_statement_table(row)
    return rows


def _source_counts(row: Dict[str, Any]) -> Tuple[collections.Counter[str], List[Dict[str, Any]]]:
    rows = _source_rows(row)
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        typ = str(r.get("type") or "")
        if typ in CONTROL_TYPES:
            c["source_controls"] += 1
            c[f"source_control_{typ}"] += 1
        if typ in LEAF_STMT_TYPES:
            c["source_leaf_stmts"] += 1
            c[f"source_leaf_{typ}"] += 1
        if typ == "declaration":
            c["source_decls"] += 1
    c["source_rows"] = len(rows)
    c["source_semantic_nodes"] = c["source_controls"] + c["source_leaf_stmts"]
    return c, rows


def _line_range_for_row(r: Dict[str, Any]) -> Set[int]:
    lo = int(r.get("start_line") or 0)
    hi = int(r.get("end_line") or lo)
    if lo <= 0 or hi <= 0:
        return set()
    return set(range(min(lo, hi), max(lo, hi) + 1))


def _bb_source_line_sets(row: Dict[str, Any]) -> List[Set[int]]:
    out: List[Set[int]] = []
    block_src_lines = row.get("block_src_lines") or []
    if not isinstance(block_src_lines, list):
        return out
    for group in block_src_lines:
        lines: Set[int] = set()
        if isinstance(group, list):
            for x in group:
                try:
                    lines.add(int(x))
                except Exception:
                    pass
        out.append(lines)
    return out


def _bb_source_granularity(row: Dict[str, Any], source_rows: List[Dict[str, Any]]) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    bb_lines = _bb_source_line_sets(row)
    if not bb_lines or not source_rows:
        return c
    all_bb_lines = set().union(*bb_lines) if bb_lines else set()

    for r in source_rows:
        if str(r.get("type") or "") == "declaration":
            if _line_range_for_row(r).isdisjoint(all_bb_lines):
                c["decl_rows_without_bb_line"] += 1

    relevant = [
        r for r in source_rows
        if str(r.get("type") or "") in CONTROL_TYPES or str(r.get("type") or "") in LEAF_STMT_TYPES
    ]
    for lines in bb_lines:
        if not lines:
            continue
        matched = [
            r for r in relevant
            if not _line_range_for_row(r).isdisjoint(lines)
        ]
        if len(matched) > 1:
            c["bb_lines_multi_source_rows"] += 1
        if len(matched) == 0:
            c["bb_lines_no_source_rows"] += 1
    return c


def _alignment_granularity(row: Dict[str, Any], threshold_ratio: float = 0.20) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    M = row.get("alignment_matrix")
    if not isinstance(M, list) or not M or not isinstance(M[0], list):
        return c
    n_b = len(M)
    n_s = len(M[0])
    col_hits = [0 for _ in range(n_s)]
    for i in range(n_b):
        vals: List[float] = []
        for x in M[i]:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(0.0)
        if not vals:
            continue
        mx = max(vals)
        if mx <= 1e-12:
            c["m_zero_rows"] += 1
            continue
        chosen = [j for j, v in enumerate(vals) if v > 1e-12 and v >= mx * threshold_ratio]
        if len(chosen) > 1:
            c["m_row_multi_source_cols"] += 1
        for j in chosen:
            if 0 <= j < n_s:
                col_hits[j] += 1
    c["m_cols_multi_bb"] = sum(1 for x in col_hits if x > 1)
    return c


def _skeleton_outline(text: str, max_lines: int = 80) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if CONTROL_RE.search(stripped) or FILL_RE.search(stripped):
            out.append(line)
        if len(out) >= max_lines:
            out.append("  ...")
            break
    return out


def _source_outline(rows: List[Dict[str, Any]], max_lines: int = 80) -> List[str]:
    if not rows:
        return []
    min_depth = min(int(r.get("depth") or 0) for r in rows)
    out: List[str] = []
    for r in sorted(
        rows,
        key=lambda x: (
            int(x.get("start_line") or 0),
            int(x.get("start_col") or 0),
            int(x.get("end_line") or 0),
            -int(x.get("depth") or 0),
        ),
    ):
        typ = str(r.get("type") or "")
        if typ not in CONTROL_TYPES and typ not in LEAF_STMT_TYPES:
            continue
        depth = max(0, int(r.get("depth") or 0) - min_depth)
        out.append(
            f"{'  ' * min(depth, 8)}{typ} "
            f"L{int(r.get('start_line') or 0)}:{int(r.get('start_col') or 0)}"
        )
        if len(out) >= max_lines:
            out.append("  ...")
            break
    return out


def _audit_row(line_no: int, row: Dict[str, Any]) -> Dict[str, Any]:
    skel = _extract_skeleton_text(row)
    slot_c = _slot_counts(skel)
    source_c, rows = _source_counts(row)
    bb_c = _bb_source_granularity(row, rows)
    m_c = _alignment_granularity(row)

    skeleton_controls = len(CONTROL_RE.findall(skel))
    fill_cond = int(slot_c.get("slot_FILL_COND", 0))
    stmt_capacity = int(slot_c.get("stmt_capacity", 0))
    semantic_slots = int(slot_c.get("semantic_slots", 0))
    source_controls = int(source_c.get("source_controls", 0))
    source_leaf = int(source_c.get("source_leaf_stmts", 0))
    source_semantic = int(source_c.get("source_semantic_nodes", 0))

    issues: List[str] = []
    if not skel:
        issues.append("missing_skeleton")
    if int(source_c.get("source_rows", 0)) == 0:
        issues.append("missing_source_ast")
    if skeleton_controls != source_controls:
        issues.append("control_count_mismatch")
    if fill_cond != source_controls:
        issues.append("cond_slot_count_mismatch")
    if stmt_capacity < source_leaf:
        issues.append("missing_stmt_capacity")
    elif stmt_capacity > source_leaf + 2:
        issues.append("extra_stmt_capacity")
    if abs(semantic_slots - source_semantic) > 2:
        issues.append("semantic_slot_count_gap_gt2")
    if int(bb_c.get("decl_rows_without_bb_line", 0)) > 0:
        issues.append("source_decl_without_bb_line")
    if int(bb_c.get("bb_lines_multi_source_rows", 0)) > 0:
        issues.append("bb_line_multi_source_rows")
    if int(m_c.get("m_row_multi_source_cols", 0)) > 0:
        issues.append("m_row_multi_source_cols")
    if int(m_c.get("m_cols_multi_bb", 0)) > 0:
        issues.append("m_cols_multi_bb")

    return {
        "line": line_no,
        "func_name": row.get("func_name"),
        "source_file": row.get("source_file"),
        "n_bb": len(row.get("bin_block_lengths") or row.get("block_src_lines") or []),
        "skeleton_controls": skeleton_controls,
        "fill_cond": fill_cond,
        "stmt_capacity": stmt_capacity,
        "semantic_slots": semantic_slots,
        "source_controls": source_controls,
        "source_leaf_stmts": source_leaf,
        "source_semantic_nodes": source_semantic,
        "control_gap": skeleton_controls - source_controls,
        "cond_gap": fill_cond - source_controls,
        "stmt_gap": stmt_capacity - source_leaf,
        "semantic_gap": semantic_slots - source_semantic,
        "slot_counts": dict(slot_c),
        "source_counts": dict(source_c),
        "bb_granularity": dict(bb_c),
        "m_granularity": dict(m_c),
        "issues": issues,
        "skeleton_outline": _skeleton_outline(skel),
        "source_outline": _source_outline(rows),
        "skeleton": skel,
    }


def _median(xs: Sequence[int]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def _write_report(audits: List[Dict[str, Any]], out: Path, max_examples: int) -> None:
    issue_counts: collections.Counter[str] = collections.Counter()
    for a in audits:
        issue_counts.update(a.get("issues") or [])

    def vals(key: str) -> List[int]:
        out_vals: List[int] = []
        for a in audits:
            v = a.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                out_vals.append(int(v))
        return out_vals

    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== STAGE2AB SKELETON STRUCTURE AUDIT =====\n")
        wf.write(f"total = {len(audits)}\n")
        for key in ["control_gap", "cond_gap", "stmt_gap", "semantic_gap"]:
            xs = vals(key)
            if xs:
                wf.write(
                    f"{key} min/median/max = {min(xs)} / {_median(xs)} / {max(xs)}\n"
                )

        wf.write("\n===== ISSUE COUNTS =====\n")
        for k, v in issue_counts.most_common():
            wf.write(f"{v:4d}  {k}\n")

        wf.write("\n===== PER FUNCTION =====\n")
        for i, a in enumerate(audits):
            issues = ",".join(a.get("issues") or []) or "-"
            wf.write(
                f"{i:4d} line={a['line']} func={a.get('func_name')} "
                f"bb={a['n_bb']} skel_ctrl={a['skeleton_controls']} src_ctrl={a['source_controls']} "
                f"stmt_cap={a['stmt_capacity']} src_stmt={a['source_leaf_stmts']} "
                f"semantic_gap={a['semantic_gap']} issues={issues}\n"
            )

        wf.write("\n===== EXAMPLES =====\n")
        ranked = sorted(
            audits,
            key=lambda a: (
                len(a.get("issues") or []),
                abs(int(a.get("semantic_gap") or 0)),
                abs(int(a.get("control_gap") or 0)),
            ),
            reverse=True,
        )
        for idx, a in enumerate(ranked[:max_examples]):
            wf.write("\n" + "=" * 100 + "\n")
            wf.write(
                f"example = {idx} line={a['line']} func={a.get('func_name')} "
                f"bb={a['n_bb']} issues={a.get('issues')}\n"
            )
            wf.write(
                f"skel_ctrl/source_ctrl = {a['skeleton_controls']}/{a['source_controls']}  "
                f"stmt_capacity/source_leaf = {a['stmt_capacity']}/{a['source_leaf_stmts']}  "
                f"semantic_slots/source_semantic = {a['semantic_slots']}/{a['source_semantic_nodes']}\n"
            )
            wf.write(f"bb_granularity = {a.get('bb_granularity')}\n")
            wf.write(f"m_granularity = {a.get('m_granularity')}\n")
            wf.write("\n-- SOURCE OUTLINE --\n")
            wf.write("\n".join(a.get("source_outline") or ["<missing>"]) + "\n")
            wf.write("\n-- SKELETON OUTLINE --\n")
            wf.write("\n".join(a.get("skeleton_outline") or ["<missing>"]) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", "--input", dest="jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--examples-jsonl", type=Path, default=None)
    args = ap.parse_args()

    rows = list(_iter_jsonl(args.jsonl.expanduser().resolve(), int(args.limit)))
    audits = [_audit_row(i, row) for i, row in rows]

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(audits, out, int(args.max_examples))

    if args.examples_jsonl is not None:
        p = args.examples_jsonl.expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for a in audits:
                f.write(json.dumps(a, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"saved_to = {out}")
    if args.examples_jsonl is not None:
        print(f"examples_jsonl = {args.examples_jsonl.expanduser().resolve()}")


if __name__ == "__main__":
    main()

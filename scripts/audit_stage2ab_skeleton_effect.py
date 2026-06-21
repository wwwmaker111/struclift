#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit 100 Stage2AB skeletons with an equivalence-aware control-flow lens.

This is a diagnostic script, not a proof of C semantic equivalence.  It avoids
the main pitfall of the older control-outline audit: a skeleton can be useful
even when it is not source-AST-exact.  For example,

    if (a) return x; return y;

and

    if (a) { return x; } else { return y; }

have different AST nesting but the same control behavior.  This script keeps
that distinction visible by reporting verdict buckets:

* SURFACE_MATCH: source and skeleton control outline match exactly.
* SURFACE_DIFF_PROBABLY_EQUIV: same control type sequence/counts, but different
  nesting; often early-return vs nested-if/else.
* NEEDS_PATH_AUDIT: close enough that path-level/manual audit is needed.
* CONTROL_COUNT_MISMATCH / LOOP_SWITCH_MISMATCH / CONTROL_ORDER_MISMATCH:
  increasingly suspicious control differences.
* STRAIGHTLINE_OK: both source and skeleton have no if/loop/switch.
* UNJUDGEABLE: source AST rows or skeleton are missing.

The source side is still extracted from source statement rows, so source parse
failures are separated instead of being counted as AB failures.
"""

from __future__ import annotations

import argparse
import collections
import json
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

try:
    import torch  # type: ignore
    from struclift.utils.skeleton import build_skeleton_for_binary_cfg  # type: ignore
except Exception:  # pragma: no cover - remote env normally has torch
    torch = None
    build_skeleton_for_binary_cfg = None


FILL_RE = re.compile(r"<FILL_([A-Z_]+):([^>]+)>")
COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.S | re.M)
CONTROL_RE = re.compile(r"\b(if|while|for|switch)\s*\(")
DO_RE = re.compile(r"\bdo\b")
TERMINAL_RE = re.compile(r"\b(return|break|continue|goto)\b")
CASE_RE = re.compile(r"\b(case\b[^:]*:|default\s*:)")
BB_MARK_RE = re.compile(r"BB_(\d+)")

SOURCE_CONTROL_TYPES = {
    "if_statement": "if",
    "while_statement": "loop",
    "for_statement": "loop",
    "do_statement": "loop",
    "switch_statement": "switch",
}

SOURCE_TERMINAL_TYPES = {
    "return_statement": "return",
    "break_statement": "break",
    "continue_statement": "continue",
    "goto_statement": "goto",
}


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _n_bb(row: Dict[str, Any]) -> int:
    for key in ("bin_block_lengths", "block_src_lines", "pattern_labels"):
        val = row.get(key)
        if isinstance(val, list):
            return len(val)
    return 0


def _skeleton_from_record(row: Dict[str, Any]) -> str:
    skel = str(row.get("sft_skeleton_text_pretty") or row.get("skeleton_text") or "")
    if skel:
        return skel

    prompt = str(row.get("sft_prompt_text_pretty") or "")
    m = re.search(r"<\|skeleton\|>\s*(.*?)\s*<\|/skeleton\|>", prompt, re.S)
    if m:
        return m.group(1)

    # Raw binskel fallback.  Prefer Stage2AB jsonl with materialized skeletons
    # when available; raw pattern_labels may be GT/rule labels, not predictions.
    if build_skeleton_for_binary_cfg is None or torch is None:
        return ""
    pl = row.get("pattern_labels")
    ei = row.get("bin_edge_index")
    if not isinstance(pl, list) or not isinstance(ei, list) or len(ei) != 2:
        return ""
    try:
        n = len(pl)
        pl_t = torch.tensor([int(x) for x in pl], dtype=torch.long)
        ei_t = torch.tensor(
            [[int(x) for x in ei[0]], [int(x) for x in ei[1]]],
            dtype=torch.long,
        )
        skel_obj = build_skeleton_for_binary_cfg(pl_t, ei_t, n)
        return str(getattr(skel_obj, "skeleton_text", "") or "")
    except Exception:
        return ""


def _render_with_targets(skeleton: str, targets: Sequence[Any]) -> str:
    i = 0

    def repl(_m: re.Match[str]) -> str:
        nonlocal i
        val = str(targets[i]) if i < len(targets) else ""
        i += 1
        return val

    return FILL_RE.sub(repl, skeleton or "")


def _filled_text_from_record(row: Dict[str, Any], skeleton: str) -> str:
    out = str(row.get("sft_output_text_pretty") or "")
    if out:
        return out
    targets = row.get("sft_teacher_targets") or []
    if isinstance(targets, list) and targets:
        return _render_with_targets(skeleton, targets)
    return ""


def _strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", str(text or ""))


def _canon_control(keyword: str) -> str:
    return "loop" if keyword in {"while", "for", "do"} else keyword


def _control_outline_from_text(text: str) -> List[Tuple[str, int]]:
    """Approximate preorder control outline from C-like text."""
    cleaned = _strip_comments(text)
    depth = 0
    out: List[Tuple[str, int]] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        leading_closes = 0
        for ch in line:
            if ch == "}":
                leading_closes += 1
            elif ch.isspace():
                continue
            else:
                break
        cur_depth = max(0, depth - leading_closes)
        for m in CONTROL_RE.finditer(line):
            out.append((_canon_control(m.group(1)), cur_depth))
        # `do { ... } while (...)` exposes both `do` and trailing while.  Count
        # a leading do as one loop and ignore the trailing while only loosely;
        # this path is mostly for skeleton text, where do-while is rare.
        if DO_RE.search(line) and not re.search(r"\bwhile\s*\(", line):
            out.append(("loop", cur_depth))
        depth = max(0, depth + line.count("{") - line.count("}"))
    if out:
        min_depth = min(d for _t, d in out)
        out = [(t, max(0, d - min_depth)) for t, d in out]
    return out


def _source_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = repack._source_all_statement_rows(row)
    if not rows:
        rows = repack._source_statement_table(row)
    return list(rows or [])


def _source_parse_status(row: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> str:
    if rows:
        return "statement_rows_ok"
    source_file = Path(str(row.get("source_file") or ""))
    if not source_file:
        return "missing_source_file_field"
    if not source_file.is_file():
        return "source_file_unreadable"
    func_name = str(row.get("func_name") or "").strip()
    if not func_name:
        return "missing_func_name"
    lt = repack._line_to_text(source_file)
    if not lt:
        return "source_file_empty_or_unreadable"
    try:
        bounds = repack._function_body_line_range(row, lt)
    except Exception:
        bounds = None
    if bounds is None:
        return "function_bounds_unavailable"
    return "statement_rows_empty"


def _active_source_lines(row: Dict[str, Any]) -> Set[int]:
    out: Set[int] = set()
    block_src_lines = row.get("block_src_lines") or []
    if not isinstance(block_src_lines, list):
        return out
    for group in block_src_lines:
        if not isinstance(group, list):
            continue
        for x in group:
            try:
                out.add(int(x))
            except Exception:
                continue
    return out


def _row_passes_source_filter(
    row: Dict[str, Any],
    active_lines: Set[int],
    source_control_filter: str,
) -> bool:
    if source_control_filter == "raw" or not active_lines:
        return True
    try:
        start = int(row.get("start_line") or 0)
        end = int(row.get("end_line") or start)
    except Exception:
        return True
    if source_control_filter == "bb-start-line":
        return start in active_lines
    if source_control_filter == "bb-range":
        lo, hi = min(start, end), max(start, end)
        return any(lo <= ln <= hi for ln in active_lines)
    return True


def _source_control_outline(
    rows: Sequence[Dict[str, Any]],
    *,
    active_lines: Optional[Set[int]] = None,
    source_control_filter: str = "bb-start-line",
) -> List[Tuple[str, int, int]]:
    active = set(active_lines or set())
    controls = [
        r
        for r in rows
        if str(r.get("type") or "") in SOURCE_CONTROL_TYPES
        and _row_passes_source_filter(r, active, source_control_filter)
    ]
    controls.sort(
        key=lambda r: (
            int(r.get("start_line") or 0),
            int(r.get("start_col") or 0),
            int(r.get("end_line") or 0),
            int(r.get("depth") or 0),
        )
    )
    if not controls:
        return []
    min_depth = min(int(r.get("depth") or 0) for r in controls)
    out: List[Tuple[str, int, int]] = []
    for r in controls:
        typ = SOURCE_CONTROL_TYPES[str(r.get("type") or "")]
        depth = max(0, int(r.get("depth") or 0) - min_depth)
        line = int(r.get("start_line") or 0)
        out.append((typ, depth, line))
    return out


def _active_line_control_outline(
    row: Dict[str, Any],
    active_lines: Set[int],
    *,
    source_control_filter: str,
) -> List[Tuple[str, int, int]]:
    """Approximate active-source control rows when source AST extraction failed.

    This fallback is deliberately conservative and line-based.  It exists only
    to avoid marking samples UNJUDGEABLE when the source file is present but the
    statement-row extractor cannot recover function bounds.  Depth is estimated
    from braces on active lines, so callers should treat these verdicts as
    weaker than tree-sitter/fallback statement-row verdicts.
    """
    if source_control_filter == "raw" or not active_lines:
        return []
    lt = repack._line_to_text(row.get("source_file"))
    if not lt:
        return []
    out: List[Tuple[str, int, int]] = []
    depth = 0
    for ln in sorted(active_lines):
        raw = str(lt.get(int(ln), ""))
        line = _strip_comments(raw).strip()
        if not line or line.startswith("#"):
            continue
        leading_close = len(line) - len(line.lstrip("}"))
        cur_depth = max(0, depth - leading_close)
        for m in CONTROL_RE.finditer(line):
            out.append((_canon_control(m.group(1)), cur_depth, int(ln)))
        if DO_RE.search(line) and not re.search(r"\bwhile\s*\(", line):
            out.append(("loop", cur_depth, int(ln)))
        depth = max(0, depth + line.count("{") - line.count("}"))
    if out:
        min_depth = min(d for _t, d, _ln in out)
        out = [(t, max(0, d - min_depth), ln) for t, d, ln in out]
    return out


def _terminal_counts_from_rows(rows: Sequence[Dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = collections.Counter()
    for r in rows:
        typ = SOURCE_TERMINAL_TYPES.get(str(r.get("type") or ""))
        if typ:
            c[typ] += 1
    return c


def _terminal_counts_from_text(text: str) -> Counter[str]:
    cleaned = _strip_comments(text)
    return collections.Counter(m.group(1) for m in TERMINAL_RE.finditer(cleaned))


def _case_count_from_rows(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for r in rows if str(r.get("type") or "") == "case_statement")


def _case_count_from_text(text: str) -> int:
    return len(CASE_RE.findall(_strip_comments(text)))


def _counts(seq: Sequence[Tuple[str, int]]) -> Counter[str]:
    return collections.Counter(t for t, _d in seq)


def _levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _sim(a: Sequence[Any], b: Sequence[Any]) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - (_levenshtein(a, b) / denom)


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _pct(n: int, d: int) -> str:
    return "0.0%" if d <= 0 else f"{100.0 * n / d:.1f}%"


def _slot_counts(skeleton: str) -> Dict[str, int]:
    c: Counter[str] = collections.Counter()
    for m in FILL_RE.finditer(skeleton or ""):
        c[f"FILL_{m.group(1)}"] += 1
    c["TOTAL"] = sum(v for k, v in c.items() if k != "TOTAL")
    return dict(c)


def _bb_marker_stats(skeleton: str, n_bb: int) -> Dict[str, Any]:
    markers = [int(m.group(1)) for m in BB_MARK_RE.finditer(skeleton or "")]
    uniq = sorted(set(x for x in markers if x >= 0))
    duplicated = sorted(x for x, n in collections.Counter(markers).items() if n > 1)
    in_range = [x for x in uniq if 0 <= x < n_bb]
    missing = [i for i in range(n_bb) if i not in set(in_range)]
    return {
        "bb_markers": len(markers),
        "bb_covered": len(in_range),
        "bb_missing": len(missing),
        "bb_duplicated": duplicated[:20],
        "bb_missing_list": missing[:30],
    }


def _outline_lines(seq: Sequence[Tuple[str, int]], limit: int = 80) -> List[str]:
    lines: List[str] = []
    for typ, depth in seq[:limit]:
        lines.append(f"{'  ' * min(depth, 10)}{typ}")
    if len(seq) > limit:
        lines.append("  ...")
    return lines


def _source_outline_lines(seq: Sequence[Tuple[str, int, int]], limit: int = 80) -> List[str]:
    lines: List[str] = []
    for typ, depth, line in seq[:limit]:
        suffix = f" L{line}" if line else ""
        lines.append(f"{'  ' * min(depth, 10)}{typ}{suffix}")
    if len(seq) > limit:
        lines.append("  ...")
    return lines


def _classify(
    *,
    source_rows: int,
    skeleton: str,
    src_seq: Sequence[Tuple[str, int]],
    pred_seq: Sequence[Tuple[str, int]],
    src_counts: Counter[str],
    pred_counts: Counter[str],
    src_term: Counter[str],
    filled_term: Counter[str],
    filled_available: bool,
    strict_filled_terminals: bool,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if source_rows <= 0:
        reasons.append("source_ast_missing")
    if not skeleton:
        reasons.append("skeleton_missing")
    if reasons:
        return "UNJUDGEABLE", reasons

    src_types = [t for t, _d in src_seq]
    pred_types = [t for t, _d in pred_seq]
    type_seq_sim = _sim(src_types, pred_types)
    depth_seq_sim = _sim(src_seq, pred_seq)

    gaps = {
        typ: int(pred_counts.get(typ, 0)) - int(src_counts.get(typ, 0))
        for typ in ("if", "loop", "switch")
    }
    for typ, gap in gaps.items():
        if gap < 0:
            reasons.append(f"missing_{typ}:{-gap}")
        elif gap > 0:
            reasons.append(f"extra_{typ}:{gap}")

    term_mismatch = bool(filled_available and dict(src_term) != dict(filled_term))
    if term_mismatch:
        reasons.append("filled_terminal_count_mismatch")

    if not src_seq and not pred_seq:
        if strict_filled_terminals and term_mismatch:
            return "NEEDS_PATH_AUDIT", reasons
        return "STRAIGHTLINE_OK", reasons

    if src_seq == pred_seq:
        if strict_filled_terminals and term_mismatch:
            return "NEEDS_PATH_AUDIT", reasons
        return "SURFACE_MATCH", reasons

    if gaps["loop"] != 0 or gaps["switch"] != 0:
        return "LOOP_SWITCH_MISMATCH", reasons

    if gaps["if"] == 0 and pred_types == src_types:
        reasons.append("same_control_sequence_different_depth")
        if strict_filled_terminals and term_mismatch:
            return "NEEDS_PATH_AUDIT", reasons
        return "SURFACE_DIFF_PROBABLY_EQUIV", reasons

    if any(gaps.values()):
        if abs(gaps["if"]) <= 2 and type_seq_sim >= 0.75 and depth_seq_sim >= 0.25:
            reasons.append("small_if_count_gap")
            return "NEEDS_PATH_AUDIT", reasons
        return "CONTROL_COUNT_MISMATCH", reasons

    if type_seq_sim >= 0.80:
        reasons.append("same_counts_nearby_order")
        return "NEEDS_PATH_AUDIT", reasons

    return "CONTROL_ORDER_MISMATCH", reasons


def _audit_row(
    line_no: int,
    row: Dict[str, Any],
    *,
    strict_filled_terminals: bool,
    source_control_filter: str = "bb-start-line",
) -> Dict[str, Any]:
    n_bb = _n_bb(row)
    skeleton = _skeleton_from_record(row)
    filled = _filled_text_from_record(row, skeleton)
    rows = _source_rows(row)
    parse_status = _source_parse_status(row, rows)
    active_lines = _active_source_lines(row)

    src_outline3 = _source_control_outline(
        rows,
        active_lines=active_lines,
        source_control_filter=source_control_filter,
    )
    source_outline_source = "statement_rows"
    source_rows_for_judgment = len(rows)
    if not rows and active_lines:
        fallback_outline = _active_line_control_outline(
            row,
            active_lines,
            source_control_filter=source_control_filter,
        )
        if fallback_outline or source_control_filter != "raw":
            src_outline3 = fallback_outline
            source_outline_source = "active_line_fallback"
            source_rows_for_judgment = len(active_lines)
    src_seq = [(t, d) for t, d, _line in src_outline3]
    pred_seq = _control_outline_from_text(skeleton)
    src_counts = _counts(src_seq)
    pred_counts = _counts(pred_seq)
    src_types = [t for t, _d in src_seq]
    pred_types = [t for t, _d in pred_seq]

    src_term = _terminal_counts_from_rows(rows)
    filled_term = _terminal_counts_from_text(filled) if filled else collections.Counter()
    verdict, reasons = _classify(
        source_rows=source_rows_for_judgment,
        skeleton=skeleton,
        src_seq=src_seq,
        pred_seq=pred_seq,
        src_counts=src_counts,
        pred_counts=pred_counts,
        src_term=src_term,
        filled_term=filled_term,
        filled_available=bool(filled),
        strict_filled_terminals=bool(strict_filled_terminals),
    )
    if source_outline_source == "active_line_fallback":
        reasons.append("source_rows_missing_used_active_line_fallback")
    if parse_status != "statement_rows_ok":
        reasons.append(f"source_parse:{parse_status}")

    bb_stats = _bb_marker_stats(skeleton, n_bb)
    case_source = _case_count_from_rows(rows)
    case_filled = _case_count_from_text(filled) if filled else 0
    slot_counts = _slot_counts(skeleton)

    return {
        "line": line_no,
        "func_name": row.get("func_name"),
        "source_file": row.get("source_file"),
        "n_bb": n_bb,
        "source_statement_rows": len(rows),
        "source_rows_for_judgment": source_rows_for_judgment,
        "source_parse_status": parse_status,
        "source_outline_source": source_outline_source,
        "source_control_filter": source_control_filter,
        "active_source_lines": len(active_lines),
        "skeleton_present": bool(skeleton),
        "filled_available": bool(filled),
        "verdict": verdict,
        "reasons": reasons,
        "source_control_count": len(src_seq),
        "pred_control_count": len(pred_seq),
        "source_counts": dict(src_counts),
        "pred_counts": dict(pred_counts),
        "if_gap": int(pred_counts.get("if", 0)) - int(src_counts.get("if", 0)),
        "loop_gap": int(pred_counts.get("loop", 0)) - int(src_counts.get("loop", 0)),
        "switch_gap": int(pred_counts.get("switch", 0)) - int(src_counts.get("switch", 0)),
        "source_terminal_counts": dict(src_term),
        "filled_terminal_counts": dict(filled_term),
        "source_case_count": case_source,
        "filled_case_count": case_filled,
        "type_seq_sim": _sim(src_types, pred_types),
        "type_depth_seq_sim": _sim(src_seq, pred_seq),
        "max_source_depth": max((d for _t, d in src_seq), default=0),
        "max_pred_depth": max((d for _t, d in pred_seq), default=0),
        "slot_counts": slot_counts,
        **bb_stats,
        "source_outline": _source_outline_lines(src_outline3),
        "pred_outline": _outline_lines(pred_seq),
        "skeleton_excerpt": "\n".join((skeleton or "").splitlines()[:80]),
        "filled_excerpt": "\n".join((filled or "").splitlines()[:80]),
    }


def _nums(audits: Sequence[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for a in audits:
        v = a.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    return vals


def _write_min_med_max(wf: Any, audits: Sequence[Dict[str, Any]], key: str) -> None:
    xs = _nums(audits, key)
    if not xs:
        return
    wf.write(
        f"{key} min/median/max = {min(xs):.4g} / {_median(xs):.4g} / {max(xs):.4g}\n"
    )


def _write_report(audits: List[Dict[str, Any]], out: Path, max_examples: int) -> None:
    verdict_counts = collections.Counter(str(a.get("verdict")) for a in audits)
    reason_counts: Counter[str] = collections.Counter()
    for a in audits:
        reason_counts.update(str(r) for r in a.get("reasons") or [])

    judgeable = [a for a in audits if a.get("verdict") != "UNJUDGEABLE"]
    source_has_control = [
        a for a in judgeable if int(a.get("source_control_count") or 0) > 0
    ]
    useful_or_ok = [
        a
        for a in judgeable
        if a.get("verdict")
        in {"SURFACE_MATCH", "SURFACE_DIFF_PROBABLY_EQUIV", "STRAIGHTLINE_OK"}
    ]
    suspicious = [
        a
        for a in judgeable
        if a.get("verdict")
        in {
            "CONTROL_COUNT_MISMATCH",
            "LOOP_SWITCH_MISMATCH",
            "CONTROL_ORDER_MISMATCH",
        }
    ]

    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== STAGE2AB SKELETON EFFECT AUDIT =====\n")
        wf.write(f"total = {len(audits)}\n")
        if audits:
            wf.write(f"source_control_filter = {audits[0].get('source_control_filter')}\n")
        source_source_counts = collections.Counter(
            str(a.get("source_outline_source") or "") for a in audits
        )
        parse_status_counts = collections.Counter(
            str(a.get("source_parse_status") or "") for a in audits
        )
        wf.write(f"source_outline_sources = {dict(source_source_counts)}\n")
        wf.write(f"source_parse_status = {dict(parse_status_counts)}\n")
        wf.write(f"judgeable = {len(judgeable)} / {len(audits)} ({_pct(len(judgeable), len(audits))})\n")
        wf.write(
            f"source_has_control = {len(source_has_control)} / {len(judgeable)} "
            f"({_pct(len(source_has_control), len(judgeable))})\n"
        )
        wf.write(
            f"useful_or_ok = {len(useful_or_ok)} / {len(judgeable)} "
            f"({_pct(len(useful_or_ok), len(judgeable))})\n"
        )
        wf.write(
            f"suspicious = {len(suspicious)} / {len(judgeable)} "
            f"({_pct(len(suspicious), len(judgeable))})\n"
        )
        for key in [
            "source_control_count",
            "pred_control_count",
            "if_gap",
            "loop_gap",
            "switch_gap",
            "type_seq_sim",
            "type_depth_seq_sim",
            "bb_covered",
            "bb_missing",
        ]:
            _write_min_med_max(wf, audits, key)

        wf.write("\n===== VERDICT COUNTS =====\n")
        for k, v in verdict_counts.most_common():
            wf.write(f"{v:4d}  {k}\n")

        wf.write("\n===== REASON COUNTS =====\n")
        for k, v in reason_counts.most_common():
            wf.write(f"{v:4d}  {k}\n")

        wf.write("\n===== HOW TO READ =====\n")
        wf.write("SURFACE_MATCH: source AST outline and skeleton outline match.\n")
        wf.write(
            "SURFACE_DIFF_PROBABLY_EQUIV: same if/loop/switch sequence/counts, "
            "different nesting; often early-return vs nested if/else.\n"
        )
        wf.write("NEEDS_PATH_AUDIT: close but not enough to call equivalent automatically.\n")
        wf.write("LOOP_SWITCH_MISMATCH: loop/switch count differs, usually high risk.\n")
        wf.write("CONTROL_COUNT_MISMATCH: if/loop/switch count differs.\n")
        wf.write("CONTROL_ORDER_MISMATCH: counts may match but control order differs strongly.\n")
        wf.write("UNJUDGEABLE: source rows or skeleton are missing.\n")

        wf.write("\n===== PER FUNCTION =====\n")
        for i, a in enumerate(audits):
            wf.write(
                f"{i:4d} line={a['line']} func={a.get('func_name')} bb={a['n_bb']} "
                f"src_rows={a['source_statement_rows']} "
                f"src_ref={a.get('source_outline_source')} "
                f"ctrl(pred/source)={a['pred_control_count']}/{a['source_control_count']} "
                f"gaps(if,loop,switch)=({a['if_gap']},{a['loop_gap']},{a['switch_gap']}) "
                f"sim(type,depth)={a['type_seq_sim']:.3f}/{a['type_depth_seq_sim']:.3f} "
                f"bb_cover={a['bb_covered']}/{a['n_bb']} "
                f"verdict={a['verdict']} "
                f"reasons={','.join(a.get('reasons') or []) or '-'}\n"
            )

        wf.write("\n===== EXAMPLES BY VERDICT =====\n")
        verdict_order = [
            "SURFACE_DIFF_PROBABLY_EQUIV",
            "NEEDS_PATH_AUDIT",
            "LOOP_SWITCH_MISMATCH",
            "CONTROL_COUNT_MISMATCH",
            "CONTROL_ORDER_MISMATCH",
            "UNJUDGEABLE",
            "SURFACE_MATCH",
            "STRAIGHTLINE_OK",
        ]
        written = 0
        for verdict in verdict_order:
            group = [a for a in audits if a.get("verdict") == verdict]
            group.sort(
                key=lambda a: (
                    -abs(int(a.get("loop_gap") or 0)) - abs(int(a.get("switch_gap") or 0)),
                    -abs(int(a.get("if_gap") or 0)),
                    float(a.get("type_depth_seq_sim") or 0.0),
                )
            )
            for a in group[: max(1, max_examples // max(1, len(verdict_order)))]:
                if written >= max_examples:
                    break
                written += 1
                wf.write("\n" + "=" * 100 + "\n")
                wf.write(
                    f"example={written - 1} verdict={a['verdict']} line={a['line']} "
                    f"func={a.get('func_name')} bb={a['n_bb']}\n"
                )
                wf.write(
                    f"ctrl(pred/source)={a['pred_control_count']}/{a['source_control_count']} "
                    f"pred_counts={a['pred_counts']} source_counts={a['source_counts']} "
                    f"term(filled/source)={a['filled_terminal_counts']}/{a['source_terminal_counts']} "
                    f"case(filled/source)={a['filled_case_count']}/{a['source_case_count']} "
                    f"sim={a['type_seq_sim']:.3f}/{a['type_depth_seq_sim']:.3f}\n"
                )
                wf.write(f"reasons={a.get('reasons')}\n")
                wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
                wf.write("\n".join(a.get("source_outline") or ["<none>"]) + "\n")
                wf.write("\n-- SKELETON CONTROL OUTLINE --\n")
                wf.write("\n".join(a.get("pred_outline") or ["<none>"]) + "\n")
                wf.write("\n-- SKELETON EXCERPT --\n")
                wf.write(str(a.get("skeleton_excerpt") or "<none>") + "\n")
            if written >= max_examples:
                break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", "--input", dest="jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--examples-jsonl", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-examples", type=int, default=24)
    ap.add_argument(
        "--strict-filled-terminals",
        action="store_true",
        help=(
            "If set, filled return/break/continue/goto count mismatch downgrades "
            "otherwise equivalent-looking skeletons to NEEDS_PATH_AUDIT.  Leave "
            "off for pure skeleton-shape audit."
        ),
    )
    ap.add_argument(
        "--source-control-filter",
        choices=["raw", "bb-start-line", "bb-range"],
        default="bb-start-line",
        help=(
            "How to filter source control rows before comparing with the AB skeleton. "
            "raw counts all source rows, including preprocessor-inactive code. "
            "bb-start-line keeps controls whose header line appears in block_src_lines. "
            "bb-range keeps controls whose source span intersects block_src_lines."
        ),
    )
    args = ap.parse_args()

    audits = [
        _audit_row(
            line_no,
            row,
            strict_filled_terminals=bool(args.strict_filled_terminals),
            source_control_filter=str(args.source_control_filter),
        )
        for line_no, row in _iter_jsonl(args.jsonl.expanduser().resolve(), int(args.limit))
    ]

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

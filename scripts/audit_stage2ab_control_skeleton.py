#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit AB control skeleton only.

This diagnostic ignores Module C, fill targets, declarations, and statement
capacity.  It only asks whether the control-flow skeleton recovered from
BB pattern labels + CFG resembles the source control skeleton:

* counts of if / loop / switch
* preorder control sequence
* approximate nesting depth sequence

The metric is intentionally coarse: C source and binary CFG can express the
same semantics with different but valid surface control forms.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
except Exception:  # pragma: no cover
    torch = None
    build_skeleton_for_binary_cfg = None


CONTROL_RE = re.compile(r"\b(if|while|for|switch)\s*\(")
COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.S | re.M)

SOURCE_CONTROL_TYPES = {
    "if_statement": "if",
    "while_statement": "loop",
    "for_statement": "loop",
    "do_statement": "loop",
    "switch_statement": "switch",
}


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _skeleton_from_record(row: Dict[str, Any]) -> str:
    skel = str(row.get("sft_skeleton_text_pretty") or row.get("skeleton_text") or "")
    if skel:
        return skel
    prompt = str(row.get("sft_prompt_text_pretty") or "")
    m = re.search(r"<\|skeleton\|>\s*(.*?)\s*<\|/skeleton\|>", prompt, re.S)
    if m:
        return m.group(1)

    # Fallback for raw binskel records.  This uses whatever pattern_labels are
    # present in the JSONL; depending on the file those may be GT labels rather
    # than Module-A predictions, so prefer Stage2AB output when available.
    if build_skeleton_for_binary_cfg is None or torch is None:
        return ""
    pl = row.get("pattern_labels")
    ei = row.get("bin_edge_index")
    if not isinstance(pl, list) or not pl or not isinstance(ei, list) or len(ei) != 2:
        return ""
    try:
        n_bb = len(pl)
        pl_t = torch.tensor([int(x) for x in pl], dtype=torch.long)
        ei_t = torch.tensor(
            [[int(x) for x in ei[0]], [int(x) for x in ei[1]]],
            dtype=torch.long,
        )
        skel_obj = build_skeleton_for_binary_cfg(pl_t, ei_t, n_bb)
        return str(getattr(skel_obj, "skeleton_text", "") or "")
    except Exception:
        return ""


def _canon_control(keyword: str) -> str:
    return "loop" if keyword in {"while", "for"} else keyword


def _control_outline_from_skeleton(text: str) -> List[Tuple[str, int]]:
    """Extract approximate preorder (control_type, brace_depth)."""
    cleaned = COMMENT_RE.sub("", str(text or ""))
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
        depth = max(0, depth + line.count("{") - line.count("}"))
    if out:
        min_depth = min(d for _t, d in out)
        out = [(t, max(0, d - min_depth)) for t, d in out]
    return out


def _source_control_outline(row: Dict[str, Any]) -> Tuple[List[Tuple[str, int]], int]:
    rows = repack._source_all_statement_rows(row)
    if not rows:
        rows = repack._source_statement_table(row)
    controls: List[Dict[str, Any]] = [
        r for r in rows if str(r.get("type") or "") in SOURCE_CONTROL_TYPES
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
        return [], len(rows)
    min_depth = min(int(r.get("depth") or 0) for r in controls)
    out: List[Tuple[str, int]] = []
    for r in controls:
        typ = SOURCE_CONTROL_TYPES[str(r.get("type") or "")]
        out.append((typ, max(0, int(r.get("depth") or 0) - min_depth)))
    return out, len(rows)


def _counts(seq: Sequence[Tuple[str, int]]) -> collections.Counter[str]:
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


def _outline_lines(seq: Sequence[Tuple[str, int]], limit: int = 120) -> List[str]:
    lines: List[str] = []
    for typ, depth in seq[:limit]:
        lines.append(f"{'  ' * min(depth, 10)}{typ}")
    if len(seq) > limit:
        lines.append("  ...")
    return lines


def _audit_row(line_no: int, row: Dict[str, Any]) -> Dict[str, Any]:
    skel = _skeleton_from_record(row)
    pred = _control_outline_from_skeleton(skel)
    src, source_statement_rows = _source_control_outline(row)
    pc = _counts(pred)
    sc = _counts(src)
    pred_types = [t for t, _d in pred]
    src_types = [t for t, _d in src]
    pred_depth = list(pred)
    src_depth = list(src)
    max_pred_depth = max((d for _t, d in pred), default=0)
    max_src_depth = max((d for _t, d in src), default=0)

    issues: List[str] = []
    if not skel:
        issues.append("missing_skeleton")
    if source_statement_rows <= 0:
        issues.append("missing_source_ast")
    for typ in ("if", "loop", "switch"):
        gap = int(pc.get(typ, 0)) - int(sc.get(typ, 0))
        if gap < 0:
            issues.append(f"missing_{typ}:{-gap}")
        elif gap > 0:
            issues.append(f"extra_{typ}:{gap}")
    if pred_types != src_types:
        issues.append("control_type_sequence_mismatch")
    if pred_depth != src_depth:
        issues.append("control_depth_sequence_mismatch")

    return {
        "line": line_no,
        "func_name": row.get("func_name"),
        "n_bb": len(row.get("bin_block_lengths") or row.get("block_src_lines") or []),
        "pred_count": len(pred),
        "source_count": len(src),
        "source_statement_rows": source_statement_rows,
        "count_gap": len(pred) - len(src),
        "pred_counts": dict(pc),
        "source_counts": dict(sc),
        "if_gap": int(pc.get("if", 0)) - int(sc.get("if", 0)),
        "loop_gap": int(pc.get("loop", 0)) - int(sc.get("loop", 0)),
        "switch_gap": int(pc.get("switch", 0)) - int(sc.get("switch", 0)),
        "max_depth_pred": max_pred_depth,
        "max_depth_source": max_src_depth,
        "depth_gap": max_pred_depth - max_src_depth,
        "type_seq_sim": _sim(pred_types, src_types),
        "type_depth_seq_sim": _sim(pred_depth, src_depth),
        "issues": issues,
        "pred_outline": _outline_lines(pred),
        "source_outline": _outline_lines(src),
    }


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _write_report(audits: List[Dict[str, Any]], out: Path, max_examples: int) -> None:
    issue_counts: collections.Counter[str] = collections.Counter()
    for a in audits:
        issue_counts.update(str(x) for x in a.get("issues") or [])

    source_ast_ok = [a for a in audits if int(a.get("source_statement_rows") or 0) > 0]
    source_ast_missing = [a for a in audits if int(a.get("source_statement_rows") or 0) <= 0]
    source_has_control = [a for a in source_ast_ok if int(a.get("source_count") or 0) > 0]
    source_no_control = [a for a in source_ast_ok if int(a.get("source_count") or 0) == 0]
    source_no_control_pred0 = [
        a for a in source_no_control if int(a.get("pred_count") or 0) == 0
    ]
    source_no_control_pred_extra = [
        a for a in source_no_control if int(a.get("pred_count") or 0) > 0
    ]

    def _exact_counts(a: Dict[str, Any]) -> bool:
        return (
            int(a.get("if_gap") or 0) == 0
            and int(a.get("loop_gap") or 0) == 0
            and int(a.get("switch_gap") or 0) == 0
        )

    def _exact_type_seq(a: Dict[str, Any]) -> bool:
        return "control_type_sequence_mismatch" not in (a.get("issues") or [])

    def _exact_depth_seq(a: Dict[str, Any]) -> bool:
        return "control_depth_sequence_mismatch" not in (a.get("issues") or [])

    def _pct(n: int, d: int) -> str:
        return "0.0%" if d <= 0 else f"{(100.0 * n / d):.1f}%"

    def _bucket_line(name: str, items: Sequence[Dict[str, Any]]) -> str:
        total = len(items)
        if total <= 0:
            return f"{name}: 0\n"
        counts_ok = sum(1 for a in items if _exact_counts(a))
        type_ok = sum(1 for a in items if _exact_type_seq(a))
        depth_ok = sum(1 for a in items if _exact_depth_seq(a))
        return (
            f"{name}: {total}  "
            f"counts_ok={counts_ok}/{total}({_pct(counts_ok, total)})  "
            f"type_seq_ok={type_ok}/{total}({_pct(type_ok, total)})  "
            f"depth_seq_ok={depth_ok}/{total}({_pct(depth_ok, total)})\n"
        )

    def nums(key: str) -> List[float]:
        vals: List[float] = []
        for a in audits:
            v = a.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        return vals

    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== AB CONTROL SKELETON AUDIT =====\n")
        wf.write(f"total = {len(audits)}\n")
        for key in [
            "pred_count",
            "source_count",
            "count_gap",
            "if_gap",
            "loop_gap",
            "switch_gap",
            "depth_gap",
            "type_seq_sim",
            "type_depth_seq_sim",
        ]:
            xs = nums(key)
            if xs:
                wf.write(
                    f"{key} min/median/max = "
                    f"{min(xs):.4g} / {_median(xs):.4g} / {max(xs):.4g}\n"
                )

        exact_counts = 0
        exact_type_seq = 0
        exact_depth_seq = 0
        for a in audits:
            if _exact_counts(a):
                exact_counts += 1
            if _exact_type_seq(a):
                exact_type_seq += 1
            if _exact_depth_seq(a):
                exact_depth_seq += 1
        wf.write(f"exact_control_type_counts = {exact_counts}\n")
        wf.write(f"exact_control_type_sequence = {exact_type_seq}\n")
        wf.write(f"exact_control_depth_sequence = {exact_depth_seq}\n")

        wf.write("\n===== BUCKETS =====\n")
        wf.write(f"source_ast_ok = {len(source_ast_ok)}\n")
        wf.write(f"source_ast_missing = {len(source_ast_missing)}\n")
        wf.write(f"source_no_control = {len(source_no_control)}\n")
        wf.write(f"source_no_control_pred0 = {len(source_no_control_pred0)}\n")
        wf.write(f"source_no_control_pred_extra = {len(source_no_control_pred_extra)}\n")
        wf.write(f"source_has_control = {len(source_has_control)}\n")
        wf.write(_bucket_line("all_with_source_ast", source_ast_ok))
        wf.write(_bucket_line("source_has_control_only", source_has_control))

        wf.write("\n===== ISSUE COUNTS =====\n")
        for k, v in issue_counts.most_common():
            wf.write(f"{v:4d}  {k}\n")

        wf.write("\n===== PER FUNCTION =====\n")
        for i, a in enumerate(audits):
            issues = ",".join(a.get("issues") or []) or "-"
            wf.write(
                f"{i:4d} line={a['line']} func={a.get('func_name')} bb={a['n_bb']} "
                f"src_rows={a.get('source_statement_rows')} "
                f"pred/source={a['pred_count']}/{a['source_count']} "
                f"gaps(if,loop,switch)=({a['if_gap']},{a['loop_gap']},{a['switch_gap']}) "
                f"depth={a['max_depth_pred']}/{a['max_depth_source']} "
                f"sim={a['type_seq_sim']:.3f}/{a['type_depth_seq_sim']:.3f} "
                f"issues={issues}\n"
            )

        wf.write("\n===== EXAMPLES =====\n")
        ranked = sorted(
            audits,
            key=lambda a: (
                len(a.get("issues") or []),
                abs(int(a.get("count_gap") or 0)),
                1.0 - float(a.get("type_depth_seq_sim") or 0.0),
            ),
            reverse=True,
        )
        for i, a in enumerate(ranked[:max_examples]):
            wf.write("\n" + "=" * 100 + "\n")
            wf.write(
                f"example={i} line={a['line']} func={a.get('func_name')} bb={a['n_bb']}\n"
            )
            wf.write(
                f"pred/source={a['pred_count']}/{a['source_count']} "
                f"pred_counts={a['pred_counts']} source_counts={a['source_counts']} "
                f"depth={a['max_depth_pred']}/{a['max_depth_source']} "
                f"sim={a['type_seq_sim']:.3f}/{a['type_depth_seq_sim']:.3f}\n"
            )
            wf.write(f"issues={a.get('issues')}\n")
            wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
            wf.write("\n".join(a.get("source_outline") or ["<none>"]) + "\n")
            wf.write("\n-- PRED CONTROL OUTLINE --\n")
            wf.write("\n".join(a.get("pred_outline") or ["<none>"]) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", "--input", dest="jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--examples-jsonl", type=Path, default=None)
    args = ap.parse_args()

    audits = [
        _audit_row(line_no, row)
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

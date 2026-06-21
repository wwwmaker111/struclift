#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit kept V10 rendered GT against the real source function."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import repack_modulec_v9_to_v10 as repack  # noqa: E402


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rendered_body(text: str) -> str:
    s = str(text or "")
    open_brace = s.find("{")
    if open_brace < 0:
        return s
    close_brace = repack._balanced_brace_end(s, open_brace)
    if close_brace < 0:
        return s[open_brace:]
    return s[open_brace:close_brace + 1]


def _norm_body(text: str) -> str:
    return repack._normalise_source_text(_rendered_body(text))


def _counter_delta(a: Counter[str], b: Counter[str]) -> Counter[str]:
    out: Counter[str] = Counter()
    for key, cnt in a.items():
        extra = int(cnt) - int(b.get(key, 0))
        if extra > 0:
            out[key] = extra
    return out


def _short_counter(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return ""
    parts: List[str] = []
    for key, cnt in counter.most_common(limit):
        item = key
        if len(item) > 140:
            item = item[:137] + "..."
        parts.append(f"{cnt}x {item}")
    more = len(counter) - limit
    if more > 0:
        parts.append(f"... +{more} more")
    return " | ".join(parts)


def _real_source_function_context(record: Dict[str, Any]) -> str:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt:
        return ""
    bounds = repack._function_body_line_range(record, lt)
    if bounds is None:
        return ""
    lo, hi = bounds
    return "\n".join(f"{ln:5d}: {lt.get(ln, '')}" for ln in range(lo, hi + 1))


def _audit_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rendered = str(row.get("sft_output_text_pretty") or "")
    source = repack._source_function_text(row)
    rendered_norm = _norm_body(rendered)
    source_norm = _norm_body(source)

    rendered_cond = Counter(repack._control_conditions_in_text(rendered))
    source_cond = Counter(repack._control_conditions_in_text(source))
    extra_cond = _counter_delta(rendered_cond, source_cond)
    missing_cond = _counter_delta(source_cond, rendered_cond)

    rendered_stmt = Counter(repack._normalised_semicolon_statement_counts(rendered))
    source_stmt = Counter(repack._normalised_semicolon_statement_counts(source))
    extra_stmt = _counter_delta(rendered_stmt, source_stmt)
    missing_stmt = _counter_delta(source_stmt, rendered_stmt)

    exact_body_match = bool(source_norm and rendered_norm == source_norm)

    warnings = [str(x) for x in (row.get("sft_v10_quality_warnings") or [])]
    if exact_body_match:
        # Exact source match wins over heuristic QA warnings. Several local-name
        # checks are intentionally conservative and can false-positive on
        # already exact rendered GT.
        order_mismatch = False
        use_before_decl = False
        undeclared: List[str] = []
    else:
        order_mismatch = repack._rendered_statement_order_mismatch(rendered, row)
        use_before_decl = repack._rendered_use_before_declaration(rendered, row)
        undeclared = repack._rendered_uses_obvious_undeclared_locals(rendered)

    reasons: List[str] = []
    if not source_norm:
        reasons.append("missing_source_function")
    if extra_cond:
        reasons.append("extra_condition")
    if missing_cond:
        reasons.append("missing_condition")
    if extra_stmt:
        reasons.append("extra_statement")
    if missing_stmt:
        reasons.append("missing_statement")
    if order_mismatch:
        reasons.append("statement_order_mismatch")
    if use_before_decl:
        reasons.append("use_before_declaration")
    if undeclared:
        reasons.append("obvious_undeclared_locals:" + ",".join(undeclared))
    if not exact_body_match and not reasons:
        reasons.append("format_or_control_shape_diff_only")

    oracle_ok = bool(source_norm) and (exact_body_match or not reasons)

    return {
        "func_name": row.get("func_name"),
        "bucket": row.get("sft_v10_quality_bucket"),
        "warnings": warnings,
        "exact_body_match": exact_body_match,
        "oracle_ok": oracle_ok,
        "reasons": reasons,
        "extra_cond": extra_cond,
        "missing_cond": missing_cond,
        "extra_stmt": extra_stmt,
        "missing_stmt": missing_stmt,
        "rendered": rendered,
        "source_context": _real_source_function_context(row),
        "slots": len(row.get("slot_types") or []),
    }


def _write_report(rows: List[Dict[str, Any]], out_path: Path, max_examples: int) -> None:
    audits = [_audit_row(row) for row in rows]
    reason_counts: Counter[str] = Counter()
    for item in audits:
        for reason in item["reasons"]:
            reason_counts[str(reason)] += 1

    bucket_counts = Counter(str(row.get("sft_v10_quality_bucket") or "missing") for row in rows)
    warning_counts: Counter[str] = Counter()
    for row in rows:
        for warning in row.get("sft_v10_quality_warnings") or []:
            warning_counts[str(warning)] += 1

    exact = sum(1 for x in audits if x["exact_body_match"])
    oracle_ok = sum(1 for x in audits if x["oracle_ok"])
    exact_and_oracle = sum(1 for x in audits if x["exact_body_match"] and x["oracle_ok"])

    with out_path.open("w", encoding="utf-8") as wf:
        wf.write("===== KEPT VS SOURCE SUMMARY =====\n")
        wf.write(f"total = {len(rows)}\n")
        wf.write(f"exact_body_match = {exact}\n")
        wf.write(f"oracle_ok = {oracle_ok}\n")
        wf.write(f"exact_and_oracle_ok = {exact_and_oracle}\n")
        wf.write(f"not_oracle_ok = {len(rows) - oracle_ok}\n")

        wf.write("\n===== QUALITY BUCKETS =====\n")
        for bucket, cnt in bucket_counts.most_common():
            wf.write(f"{cnt:4d}  {bucket}\n")

        wf.write("\n===== WARNING COUNTS =====\n")
        for warning, cnt in warning_counts.most_common():
            wf.write(f"{cnt:4d}  {warning}\n")

        wf.write("\n===== ORACLE MISMATCH REASONS =====\n")
        for reason, cnt in reason_counts.most_common():
            wf.write(f"{cnt:4d}  {reason}\n")

        wf.write("\n===== PER-FUNCTION VERDICTS =====\n")
        for idx, item in enumerate(audits):
            verdict = "OK" if item["oracle_ok"] else "BAD"
            exact_flag = "exact" if item["exact_body_match"] else "not_exact"
            reasons = ",".join(item["reasons"]) if item["reasons"] else "-"
            wf.write(
                f"{idx:4d}  {verdict:3s}  {exact_flag:9s}  "
                f"func={item['func_name']} bucket={item['bucket']} slots={item['slots']} reasons={reasons}\n"
            )

        wf.write("\n===== BAD EXAMPLES =====\n")
        shown = 0
        for idx, item in enumerate(audits):
            if item["oracle_ok"] and item["exact_body_match"]:
                continue
            if shown >= max_examples:
                break
            shown += 1
            wf.write("\n" + "=" * 100 + "\n")
            wf.write(
                f"kept_index = {idx} func = {item['func_name']} "
                f"bucket = {item['bucket']} slots = {item['slots']}\n"
            )
            wf.write(f"exact_body_match = {item['exact_body_match']} oracle_ok = {item['oracle_ok']}\n")
            wf.write(f"warnings = {item['warnings']}\n")
            wf.write(f"reasons = {item['reasons']}\n")
            if item["extra_cond"]:
                wf.write("extra_conditions = " + _short_counter(item["extra_cond"]) + "\n")
            if item["missing_cond"]:
                wf.write("missing_conditions = " + _short_counter(item["missing_cond"]) + "\n")
            if item["extra_stmt"]:
                wf.write("extra_statements = " + _short_counter(item["extra_stmt"]) + "\n")
            if item["missing_stmt"]:
                wf.write("missing_statements = " + _short_counter(item["missing_stmt"]) + "\n")
            wf.write("\nRENDERED GT:\n")
            wf.write(str(item["rendered"]) + "\n")
            wf.write("\nREAL SOURCE FUNCTION:\n")
            wf.write(str(item["source_context"]) + "\n")


def _write_split_jsonl(
    rows: List[Dict[str, Any]],
    ok_path: Path | None,
    bad_path: Path | None,
) -> None:
    if ok_path is None and bad_path is None:
        return
    audits = [_audit_row(row) for row in rows]
    ok_f = None
    bad_f = None
    try:
        if ok_path is not None:
            ok_path.parent.mkdir(parents=True, exist_ok=True)
            ok_f = ok_path.open("w", encoding="utf-8")
        if bad_path is not None:
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_f = bad_path.open("w", encoding="utf-8")
        for row, audit in zip(rows, audits):
            out = dict(row)
            out["sft_v10_source_exact_body_match"] = bool(audit["exact_body_match"])
            out["sft_v10_source_oracle_ok"] = bool(audit["oracle_ok"])
            out["sft_v10_source_oracle_reasons"] = list(audit["reasons"])
            wf = ok_f if audit["oracle_ok"] else bad_f
            if wf is not None:
                wf.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        if ok_f is not None:
            ok_f.close()
        if bad_f is not None:
            bad_f.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kept-jsonl", "--input", dest="kept_jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=30)
    ap.add_argument("--ok-jsonl", type=Path, default=None)
    ap.add_argument("--bad-jsonl", type=Path, default=None)
    args = ap.parse_args()

    rows = _load_jsonl(args.kept_jsonl.expanduser().resolve())
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(rows, out_path, int(args.max_examples))
    ok_path = args.ok_jsonl.expanduser().resolve() if args.ok_jsonl is not None else None
    bad_path = args.bad_jsonl.expanduser().resolve() if args.bad_jsonl is not None else None
    _write_split_jsonl(rows, ok_path, bad_path)
    print(f"saved_to = {out_path}")
    if ok_path is not None:
        print(f"ok_jsonl = {ok_path}")
    if bad_path is not None:
        print(f"bad_jsonl = {bad_path}")


if __name__ == "__main__":
    main()

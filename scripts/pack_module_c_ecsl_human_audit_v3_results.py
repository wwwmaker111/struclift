#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack human audit verdicts into Module-C ECSL strong-evidence JSONL.

The output is intentionally named as strong evidence rather than automatic V3:
human review can be used for hard SFT, but it is not differential/trace/symbolic
execution proof.  ``build_module_c_ecsl_strong_sft.py`` accepts these rows when
``human_audit_pass`` and ``hard_ce_allowed`` are true.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCHEMA = "ModuleC.ECSL.HumanAuditedStrongEvidence.v0"
TIER_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
TIER_WEIGHT = {"C3": 0.5, "C4": 0.8, "C5": 1.0}


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def strict_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "accept", "accepted", "approved"}
    return False


def human_pass(row: Dict[str, Any]) -> bool:
    verdict = str(
        row.get("reviewer_verdict")
        or row.get("human_verdict")
        or row.get("manual_verdict")
        or ""
    ).strip().lower()
    return bool(
        strict_true(row.get("human_audit_pass"))
        or strict_true(row.get("manual_semantic_pass"))
        or strict_true(row.get("human_semantic_pass"))
        or verdict in {"pass", "passed", "accept", "accepted", "approved", "yes", "true"}
    )


def compact_counter(counter: collections.Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def row_key(row: Dict[str, Any]) -> Tuple[Any, Any, str]:
    return (row.get("example"), row.get("input_line"), str(row.get("func_name") or ""))


def function_row_template(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "function_key": row.get("function_key"),
        "example": row.get("example"),
        "input_line": row.get("input_line"),
        "func_name": row.get("func_name"),
        "passed": True,
        "v3_pass": False,
        "differential_pass": False,
        "trace_match": False,
        "symbolic_match": False,
        "human_audit_pass": True,
        "manual_semantic_pass": True,
        "confidence_level": "C3",
        "tier": "C3",
        "hard_ce_allowed": True,
        "oracle_type": "human_semantic_audit",
        "evidence_kind": "human_audit",
        "reason": "human_audited_slot_semantic_pass",
        "label_policy": "slot",
        "oracle_note": "Human semantic audit evidence for hard SFT; not automatic differential/trace/symbolic V3.",
        "slot_v3_results": [],
    }


def pack_slot(row: Dict[str, Any], *, min_tier: str) -> Tuple[bool, Dict[str, Any], str]:
    if not human_pass(row):
        return False, {}, "human_audit_not_passed"
    text = str(row.get("target_text") or row.get("candidate_text") or row.get("text") or "").strip()
    if not text:
        return False, {}, "empty_target_text"
    level = str(row.get("confidence_level") or row.get("tier") or min_tier or "C3")
    if TIER_ORDER.get(level, 0) < TIER_ORDER.get(min_tier, 3):
        level = str(min_tier)
    hard_ok = strict_true(row.get("hard_ce_allowed")) or ("hard_ce_allowed" not in row)
    # Exported task templates default hard_ce_allowed=false so a reviewer must
    # opt in.  For edited rows that omitted the field, human pass is enough.
    if "hard_ce_allowed" in row and not hard_ok:
        return False, {}, "hard_ce_allowed_false"
    slot = {
        "slot_index": row.get("slot_index"),
        "slot_id": row.get("slot_id"),
        "slot_type": row.get("slot_type"),
        "candidate_id": row.get("candidate_id"),
        "target_text": text,
        "passed": True,
        "v3_pass": False,
        "human_audit_pass": True,
        "manual_semantic_pass": True,
        "differential_pass": False,
        "trace_match": False,
        "symbolic_match": False,
        "confidence_level": level,
        "tier": level,
        "confidence_weight": TIER_WEIGHT.get(level, 0.5),
        "hard_ce_allowed": True,
        "oracle_type": "human_semantic_audit",
        "evidence_kind": "human_audit",
        "reviewer": row.get("reviewer"),
        "reviewer_verdict": row.get("reviewer_verdict") or "pass",
        "review_reason": row.get("review_reason") or row.get("reason") or "",
        "reason": row.get("review_reason") or row.get("reason") or "human_audited_slot_semantic_pass",
        "validation": {
            "V3": {
                "passed": True,
                "status": "human_audit_pass",
                "reason": "human semantic audit pass; not automatic symbolic/trace/differential V3",
                "human_audit_pass": True,
                "differential_pass": False,
                "trace_match": False,
                "symbolic_match": False,
            }
        },
    }
    return True, slot, "packed"


def build(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_func: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    stats: Dict[str, Any] = {
        "rows_seen": 0,
        "rows_packed": 0,
        "functions_packed": 0,
        "skip_reason_dist": collections.Counter(),
        "slot_type_dist": collections.Counter(),
        "tier_dist": collections.Counter(),
    }
    for _line_no, row in iter_jsonl(Path(args.verdicts_jsonl).expanduser().resolve(), limit=int(args.limit)):
        stats["rows_seen"] += 1
        ok, slot, reason = pack_slot(row, min_tier=str(args.min_tier))
        if not ok:
            stats["skip_reason_dist"][reason] += 1
            continue
        key = row_key(row)
        func = by_func.setdefault(key, function_row_template(row))
        func["slot_v3_results"].append(slot)
        stats["rows_packed"] += 1
        stats["slot_type_dist"][str(slot.get("slot_type") or "UNKNOWN")] += 1
        stats["tier_dist"][str(slot.get("confidence_level") or "C3")] += 1
    rows = list(by_func.values())
    for row in rows:
        slots = row.get("slot_v3_results") or []
        best = "C3"
        for slot in slots:
            level = str(slot.get("confidence_level") or "C3")
            if TIER_ORDER.get(level, 0) > TIER_ORDER.get(best, 0):
                best = level
        row["confidence_level"] = best
        row["tier"] = best
        row["human_audit_pass_slot_count"] = len(slots)
        row["reason"] = f"human_audited_slot_semantic_pass_slots={len(slots)}"
    stats["functions_packed"] = len(rows)
    return rows, stats


def write_report(path: Path, rows: List[Dict[str, Any]], stats: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "=== Module C ECSL Human Audit Strong Evidence Pack ===",
        f"verdicts_jsonl = {args.verdicts_jsonl}",
        f"rows_seen = {stats['rows_seen']}",
        f"rows_packed = {stats['rows_packed']}",
        f"functions_packed = {stats['functions_packed']}",
        f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}",
        f"tier_dist = {compact_counter(stats['tier_dist'])}",
        f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}",
        "",
        "Interpretation:",
        "- Output rows can be supplied as --v3-results-jsonl to the strong SFT builder.",
        "- Evidence kind is human_audit. It is hard SFT eligible but not automatic differential/trace/symbolic V3.",
        "",
        "-- first packed functions --",
    ]
    for row in rows[:30]:
        lines.append(
            json.dumps(
                {
                    "func_name": row.get("func_name"),
                    "input_line": row.get("input_line"),
                    "human_audit_pass_slot_count": row.get("human_audit_pass_slot_count"),
                    "confidence_level": row.get("confidence_level"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if not rows:
        lines.append("<none>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdicts-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-tier", default="C3", choices=["C3", "C4", "C5"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = build(args)
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    for path in (out_jsonl, out_txt):
        path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(out_txt, rows, stats, args)
    print(f"rows_seen = {stats['rows_seen']}")
    print(f"rows_packed = {stats['rows_packed']}")
    print(f"functions_packed = {stats['functions_packed']}")
    print(f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}")
    print(f"saved_strong_evidence_jsonl = {out_jsonl}")
    print(f"saved_report = {out_txt}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Module-C ECSL coverage with slot-level, not function-level, metrics."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def q(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def canonical_slot_id(value: Any) -> str:
    text = str(value or "").strip()
    m = re.search(r"slot_(\d+)", text)
    if m:
        return f"slot_{int(m.group(1)):03d}"
    return text


def function_key(row: Dict[str, Any]) -> str:
    return str(row.get("function_key") or f"{row.get('input_line')}:{row.get('func_name')}")


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def compact(counter: collections.Counter, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def candidate_text(cand: Dict[str, Any]) -> str:
    for item in cand.get("c_renderings") or []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            return str(item.get("text") or "").strip()
    return ""


def strict_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return False


def load_contract_slots(path: Optional[str], limit: int) -> Tuple[Dict[Tuple[str, str], str], collections.Counter]:
    out: Dict[Tuple[str, str], str] = {}
    type_dist: collections.Counter = collections.Counter()
    if not path:
        return out, type_dist
    for _line_no, row in iter_jsonl(Path(path).expanduser().resolve(), limit=limit):
        fk = function_key(row)
        for slot in row.get("semantic_slots") or []:
            if not isinstance(slot, dict):
                continue
            sid = canonical_slot_id(slot.get("slot_id"))
            typ = str(slot.get("slot_type") or "UNKNOWN")
            out[(fk, sid)] = typ
            type_dist[typ] += 1
    return out, type_dist


def load_candidate_slots(path: Optional[str], limit: int) -> Dict[str, Any]:
    stats = {
        "slots": set(),
        "concrete_slots": set(),
        "v2_pass_slots": set(),
        "v2_partial_slots": set(),
        "v3_candidate_slots": set(),
        "concrete_by_type": collections.Counter(),
        "v2_pass_by_type": collections.Counter(),
        "v2_partial_by_type": collections.Counter(),
    }
    if not path:
        return stats
    for _line_no, row in iter_jsonl(Path(path).expanduser().resolve(), limit=limit):
        fk = function_key(row)
        for slot_set in row.get("slot_candidate_sets") or []:
            if not isinstance(slot_set, dict):
                continue
            slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
            sid = canonical_slot_id(slot.get("slot_id"))
            typ = str(slot.get("slot_type") or "UNKNOWN")
            key = (fk, sid)
            stats["slots"].add(key)
            has_concrete = False
            has_v2_pass = False
            has_v2_partial = False
            has_v3 = False
            for cand in slot_set.get("candidates") or []:
                if not isinstance(cand, dict):
                    continue
                if q(cand, "keep", "status") == "KEEP_EXTERNAL":
                    continue
                if candidate_text(cand):
                    has_concrete = True
                status = str(q(cand, "validation", "V2", "status") or "")
                if status == "pass":
                    has_v2_pass = True
                elif status == "partial_needs_v3":
                    has_v2_partial = True
                if strict_true(q(cand, "validation", "V3", "passed")):
                    has_v3 = True
            if has_concrete:
                stats["concrete_slots"].add(key)
                stats["concrete_by_type"][typ] += 1
            if has_v2_pass:
                stats["v2_pass_slots"].add(key)
                stats["v2_pass_by_type"][typ] += 1
            if has_v2_partial:
                stats["v2_partial_slots"].add(key)
                stats["v2_partial_by_type"][typ] += 1
            if has_v3:
                stats["v3_candidate_slots"].add(key)
    return stats


def load_v3_slots(path: Optional[str], limit: int) -> Dict[str, Any]:
    stats = {
        "tasks_seen": 0,
        "passed_rows": 0,
        "task_slots": set(),
        "passed_slots": set(),
        "strict_passed_slots": set(),
        "contract_passed_slots": set(),
        "passed_by_type": collections.Counter(),
        "strict_passed_by_type": collections.Counter(),
        "contract_passed_by_type": collections.Counter(),
        "proof_kind_dist": collections.Counter(),
        "reason_dist": collections.Counter(),
    }
    if not path:
        return stats
    for _line_no, row in iter_jsonl(Path(path).expanduser().resolve(), limit=limit):
        stats["tasks_seen"] += 1
        fk = function_key(row)
        sid = canonical_slot_id(row.get("slot_id"))
        typ = str(row.get("slot_type") or "UNKNOWN")
        key = (fk, sid)
        stats["task_slots"].add(key)
        proof = str(q(row, "evidence", "proof_kind") or row.get("proof_kind") or "unknown")
        reason = str(row.get("reason") or "")
        pass_like_reason = reason.endswith("_pass") or "_pass_" in reason or "pass" in proof
        if strict_true(row.get("v3_pass")) or strict_true(row.get("passed")) or pass_like_reason:
            stats["passed_rows"] += 1
            stats["passed_slots"].add(key)
            stats["passed_by_type"][typ] += 1
            stats["proof_kind_dist"][proof] += 1
            if proof in {
                "control_contract_symbolic",
                "contract_backed_symbolic",
                "contract_backed_symbolic_flagdef_verified",
                "contract_empty_effect_symbolic",
                "contract_return_constant_symbolic",
            }:
                stats["contract_passed_slots"].add(key)
                stats["contract_passed_by_type"][typ] += 1
            else:
                stats["strict_passed_slots"].add(key)
                stats["strict_passed_by_type"][typ] += 1
        if reason:
            stats["reason_dist"][reason[:160]] += 1
    return stats


def load_strong_slots(path: Optional[str], limit: int) -> Dict[str, Any]:
    stats = {
        "functions": 0,
        "hard_functions": 0,
        "supervised_slots": 0,
        "dist": collections.Counter(),
    }
    if not path:
        return stats
    for _line_no, row in iter_jsonl(Path(path).expanduser().resolve(), limit=limit):
        stats["functions"] += 1
        hard = bool(row.get("hard_sft_eligible")) or str(row.get("confidence_level") or "").startswith("C3")
        if hard:
            stats["hard_functions"] += 1
        n = int(row.get("supervised_slot_count") or 0)
        stats["supervised_slots"] += n
        stats["dist"][n] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contracts-jsonl")
    ap.add_argument("--candidates-jsonl")
    ap.add_argument("--v3-results-jsonl")
    ap.add_argument("--strong-audit-jsonl")
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    contract_slots, contract_type_dist = load_contract_slots(args.contracts_jsonl, int(args.limit))
    cand = load_candidate_slots(args.candidates_jsonl, int(args.limit))
    v3 = load_v3_slots(args.v3_results_jsonl, int(args.limit))
    strong = load_strong_slots(args.strong_audit_jsonl, int(args.limit))
    total = len(contract_slots) if contract_slots else (len(cand["slots"]) if cand["slots"] else len(v3["task_slots"]))

    lines = [
        "=== Module C ECSL Slot Coverage Audit ===",
        f"contracts_jsonl = {args.contracts_jsonl or '<none>'}",
        f"candidates_jsonl = {args.candidates_jsonl or '<none>'}",
        f"v3_results_jsonl = {args.v3_results_jsonl or '<none>'}",
        f"strong_audit_jsonl = {args.strong_audit_jsonl or '<none>'}",
        "",
        f"contract_slots_total = {total}",
        f"contract_slot_type_dist = {compact(contract_type_dist)}",
        f"candidate_slots_seen = {len(cand['slots'])}",
        f"candidate_concrete_slots = {pct(len(cand['concrete_slots']), total)}",
        f"candidate_v2_pass_slots = {pct(len(cand['v2_pass_slots']), total)}",
        f"candidate_v2_partial_slots = {pct(len(cand['v2_partial_slots']), total)}",
        f"candidate_v3_embedded_slots = {pct(len(cand['v3_candidate_slots']), total)}",
        f"candidate_concrete_by_type = {compact(cand['concrete_by_type'])}",
        f"candidate_v2_pass_by_type = {compact(cand['v2_pass_by_type'])}",
        f"candidate_v2_partial_by_type = {compact(cand['v2_partial_by_type'])}",
        "",
        f"v3_tasks_seen = {v3['tasks_seen']}",
        f"v3_pass_rows = {v3['passed_rows']}",
        f"v3_task_unique_slots = {pct(len(v3['task_slots']), total)}",
        f"v3_pass_unique_slots = {pct(len(v3['passed_slots']), total)}",
        f"v3_strict_non_contract_unique_slots = {pct(len(v3['strict_passed_slots']), total)}",
        f"v3_contract_backed_unique_slots = {pct(len(v3['contract_passed_slots']), total)}",
        f"v3_pass_by_type = {compact(v3['passed_by_type'])}",
        f"v3_strict_pass_by_type = {compact(v3['strict_passed_by_type'])}",
        f"v3_contract_pass_by_type = {compact(v3['contract_passed_by_type'])}",
        f"v3_proof_kind_dist = {compact(v3['proof_kind_dist'])}",
        f"v3_reason_dist = {compact(v3['reason_dist'])}",
        "",
        f"strong_audit_functions = {strong['functions']}",
        f"strong_hard_functions = {strong['hard_functions']}",
        f"strong_supervised_slots = {strong['supervised_slots']}",
        f"strong_supervised_slot_dist = {compact(strong['dist'])}",
        "",
        "Interpretation:",
        "- function-level pass is not slot coverage; use v3_pass_unique_slots and strong_supervised_slots for coverage.",
        "- candidate_concrete_slots shows whether SCTS produced text for a slot.",
        "- candidate_v2_pass/partial shows whether candidate text reached the validator pool.",
        "- v3_pass_unique_slots is the hard-label coverage bottleneck before strong SFT.",
    ]
    out = Path(args.out_txt).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:24]))
    print(f"saved_report = {out}")


if __name__ == "__main__":
    main()

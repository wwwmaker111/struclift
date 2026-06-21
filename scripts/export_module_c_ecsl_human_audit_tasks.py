#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Module-C ECSL human semantic audit tasks.

This script does not create strong labels.  It prepares per-slot candidate rows
for a human reviewer to decide whether a slot fill is semantically correct for
the fixed Module-B skeleton.

Workflow:

1. Export tasks from ``ecsl_slot_candidates.jsonl``.
2. Copy/edit the JSONL rows as verdicts, setting ``human_audit_pass=true`` for
   approved slot fills and optionally editing ``target_text``.
3. Pack verdicts with ``pack_module_c_ecsl_human_audit_v3_results.py``.
4. Feed the packed JSONL to ``build_module_c_ecsl_strong_sft.py`` or the
   strong500 runner as ``--v3-results-jsonl``.

Human approval is treated as strong semantic evidence for hard SFT, but it is
kept distinct from automatic differential/trace/symbolic V3 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCHEMA = "ModuleC.ECSL.HumanAuditTask.v0"


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


def strict_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return False


def stable_id(*parts: Any, n: int = 16) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def compact_counter(counter: Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def candidate_text(cand: Dict[str, Any]) -> str:
    for item in cand.get("c_renderings") or []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            return str(item.get("text") or "").strip()
    return str(cand.get("target_text") or cand.get("text") or "").strip()


def candidate_filter_pass(cand: Dict[str, Any], mode: str) -> Tuple[bool, str]:
    if q(cand, "keep", "status") == "KEEP_EXTERNAL":
        return False, "keep_external_not_generated"
    if not candidate_text(cand):
        return False, "empty_candidate_text"
    v0 = strict_true(q(cand, "validation", "V0", "passed"))
    v1 = strict_true(q(cand, "validation", "V1", "passed"))
    v2 = strict_true(q(cand, "validation", "V2", "passed"))
    if mode == "all_generated":
        return True, "ok"
    if mode == "v0":
        return (v0, "ok" if v0 else "v0_failed")
    if mode == "v0v1":
        return (v0 and v1, "ok" if (v0 and v1) else ("v0_failed" if not v0 else "v1_failed"))
    return (v0 and v1 and v2, "ok" if (v0 and v1 and v2) else ("v0_failed" if not v0 else ("v1_failed" if not v1 else "v2_failed")))


def score_key(cand: Dict[str, Any]) -> Tuple[int, float, float, str]:
    tier_order = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
    tier = str(q(cand, "confidence", "confidence_level") or "C0")
    weight = 0.0
    try:
        weight = float(q(cand, "confidence", "confidence_weight") or 0.0)
    except Exception:
        pass
    score = 0.0
    try:
        score = float(q(cand, "matching", "base_alignment_score") or 0.0)
    except Exception:
        pass
    return tier_order.get(tier, 0), weight, score, str(cand.get("candidate_id") or "")


def slot_contract(slot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "slot_id": slot.get("slot_id"),
        "slot_type": slot.get("slot_type"),
        "semantic_slot_type": slot.get("semantic_slot_type"),
        "role": slot.get("role"),
        "anchor": slot.get("anchor"),
        "region": slot.get("region"),
        "attention_mask": slot.get("attention_mask"),
        "control_contract": slot.get("control_contract"),
        "data_contract": slot.get("data_contract"),
        "effect_contract": slot.get("effect_contract"),
        "grammar_contract": slot.get("grammar_contract"),
        "observable_semantics": slot.get("observable_semantics"),
    }


def build_task(record: Dict[str, Any], slot_set: Dict[str, Any], cand: Dict[str, Any], *, rank: int) -> Dict[str, Any]:
    slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
    slot_id = str(slot.get("slot_id") or cand.get("slot_id") or "")
    cand_id = str(cand.get("candidate_id") or "")
    text = candidate_text(cand)
    task_id = "humanaudit_" + stable_id(record.get("function_key"), record.get("input_line"), record.get("func_name"), slot_id, cand_id, text)
    return {
        "schema": SCHEMA,
        "task_id": task_id,
        "function_key": record.get("function_key"),
        "example": record.get("example"),
        "input_line": record.get("input_line"),
        "func_name": record.get("func_name"),
        "slot_id": slot_id,
        "slot_type": slot.get("slot_type"),
        "candidate_id": cand_id,
        "candidate_rank_for_slot": int(rank),
        "candidate_text": text,
        "target_text": text,
        "human_audit_pass": False,
        "reviewer_verdict": "unreviewed",
        "reviewer": "",
        "review_reason": "",
        "confidence_level": "C3",
        "hard_ce_allowed": False,
        "audit_instruction": "Set human_audit_pass=true and hard_ce_allowed=true only if this slot fill is semantically correct for the fixed skeleton; edit target_text if needed.",
        "skeleton": record.get("verified_skeleton_text") or record.get("verified_skeleton_preview"),
        "slot_contract": slot_contract(slot),
        "candidate_validation": cand.get("validation"),
        "candidate_confidence": cand.get("confidence"),
        "candidate_matching": cand.get("matching"),
        "candidate_provenance": cand.get("provenance"),
        "candidate_renderings": cand.get("c_renderings"),
        "semantic_evidence_kind": "human_audit_pending",
        "notes": [
            "Human pass is strong human semantic evidence for hard SFT.",
            "It is not automatic differential/trace/symbolic V3.",
            "Prefer candidates that already pass V0/V1/V2; use relaxed exports only for manual override audits.",
        ],
    }


def md_escape(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", "<br>")


def write_md(path: Path, tasks: Sequence[Dict[str, Any]], stats: Dict[str, Any], args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# Module C ECSL Human Audit Tasks")
    lines.append("")
    lines.append(f"- candidates_jsonl: `{args.candidates_jsonl}`")
    lines.append(f"- candidate_filter: `{args.candidate_filter}`")
    lines.append(f"- tasks_written: `{len(tasks)}`")
    lines.append(f"- functions_seen: `{stats['functions_seen']}`")
    lines.append(f"- slots_seen: `{stats['slots_seen']}`")
    lines.append("")
    lines.append("Reviewer rule: only mark a row pass when the candidate text is semantically correct for that slot in the fixed Module-B skeleton. Edit `target_text` before passing if the shown candidate is close but not exact.")
    lines.append("")
    lines.append("| # | func | slot | type | tier | v2 | candidate | text |")
    lines.append("|---:|---|---|---|---|---|---|---|")
    for i, task in enumerate(tasks[: int(args.md_max_rows)], 1):
        v2 = q(task, "candidate_validation", "V2", "status") or q(task, "candidate_validation", "V2", "reason")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    md_escape(task.get("func_name")),
                    md_escape(task.get("slot_id")),
                    md_escape(task.get("slot_type")),
                    md_escape(q(task, "candidate_confidence", "confidence_level")),
                    md_escape(v2),
                    md_escape(task.get("candidate_id")),
                    md_escape(str(task.get("candidate_text") or "")[:240]),
                ]
            )
            + " |"
        )
    if len(tasks) > int(args.md_max_rows):
        lines.append("")
        lines.append(f"... truncated markdown preview; full task JSONL has {len(tasks)} rows.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "functions_seen": 0,
        "slots_seen": 0,
        "candidates_seen": 0,
        "tasks_written": 0,
        "slot_type_dist": Counter(),
        "candidate_tier_dist": Counter(),
        "skip_reason_dist": Counter(),
    }
    tasks: List[Dict[str, Any]] = []
    for _line_no, record in iter_jsonl(Path(args.candidates_jsonl).expanduser().resolve(), limit=int(args.limit)):
        stats["functions_seen"] += 1
        per_function = 0
        for slot_set in record.get("slot_candidate_sets") or []:
            if not isinstance(slot_set, dict):
                continue
            slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
            stats["slots_seen"] += 1
            stats["slot_type_dist"][str(slot.get("slot_type") or "UNKNOWN")] += 1
            rows = [cand for cand in (slot_set.get("candidates") or []) if isinstance(cand, dict)]
            rows.sort(key=score_key, reverse=True)
            per_slot = 0
            for cand in rows:
                stats["candidates_seen"] += 1
                stats["candidate_tier_dist"][str(q(cand, "confidence", "confidence_level") or "C0")] += 1
                ok, reason = candidate_filter_pass(cand, str(args.candidate_filter))
                if not ok:
                    stats["skip_reason_dist"][reason] += 1
                    continue
                tasks.append(build_task(record, slot_set, cand, rank=per_slot + 1))
                stats["tasks_written"] += 1
                per_slot += 1
                per_function += 1
                if int(args.max_candidates_per_slot) > 0 and per_slot >= int(args.max_candidates_per_slot):
                    break
            if int(args.max_tasks_per_function) > 0 and per_function >= int(args.max_tasks_per_function):
                break
    return tasks, stats


def write_report(path: Path, stats: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "=== Module C ECSL Human Audit Task Export ===",
        f"candidates_jsonl = {args.candidates_jsonl}",
        f"candidate_filter = {args.candidate_filter}",
        f"functions_seen = {stats['functions_seen']}",
        f"slots_seen = {stats['slots_seen']}",
        f"candidates_seen = {stats['candidates_seen']}",
        f"tasks_written = {stats['tasks_written']}",
        f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}",
        f"candidate_tier_dist = {compact_counter(stats['candidate_tier_dist'])}",
        f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}",
        "",
        "Interpretation:",
        "- This file is for manual review; it is not strong evidence until reviewer verdicts are edited and packed.",
        "- Passing human audit is accepted by the strong SFT builder as human semantic evidence, not automatic V3.",
        "- If V2 coverage is too low, re-export with --candidate-filter v0v1 and inspect those candidates manually.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--candidate-filter", choices=["v2", "v0v1", "v0", "all_generated"], default="v2")
    ap.add_argument("--max-candidates-per-slot", type=int, default=3)
    ap.add_argument("--max-tasks-per-function", type=int, default=0)
    ap.add_argument("--md-max-rows", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tasks, stats = build(args)
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    for path in (out_jsonl, out_md, out_txt):
        path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    write_md(out_md, tasks, stats, args)
    write_report(out_txt, stats, args)
    print(f"functions_seen = {stats['functions_seen']}")
    print(f"tasks_written = {stats['tasks_written']}")
    print(f"candidate_tier_dist = {compact_counter(stats['candidate_tier_dist'])}")
    print(f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}")
    print(f"saved_tasks_jsonl = {out_jsonl}")
    print(f"saved_tasks_md = {out_md}")
    print(f"saved_report = {out_txt}")


if __name__ == "__main__":
    main()

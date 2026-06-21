#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export strict Module-C ECSL V3 validation tasks.

This script does not validate semantics by itself.  It packages V0/V1/V2-passing
slot candidates into a JSONL task file for a real symbolic, trace, or
differential validator.  The validator's output JSONL is then consumed by
build_module_c_ecsl_candidates.py / build_module_c_ecsl_strong_sft.py as
external V3 evidence.

Source-teacher projections and compile-smoke results must not be written here as
passing V3 results.  A real V3 result needs differential_pass, trace_match, or
symbolic_match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCHEMA = "ModuleC.ECSL.V3ValidationTask.v0"


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


def stable_id(*parts: Any, n: int = 16) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def candidate_text(cand: Dict[str, Any]) -> str:
    for item in cand.get("c_renderings") or []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            return str(item.get("text") or "").strip()
    return ""


def is_source_projection_candidate(cand: Dict[str, Any]) -> bool:
    prov = cand.get("provenance") if isinstance(cand.get("provenance"), dict) else {}
    sources = {str(x) for x in (prov.get("source") or [])}
    if "source_teacher_projection" in sources:
        return True
    if str(prov.get("oracle_type") or "") == "training_source_teacher_slot_projection":
        return True
    for item in cand.get("c_renderings") or []:
        if isinstance(item, dict) and str(item.get("style") or "") == "source_teacher_projection":
            return True
    return False


def candidate_is_taskable(
    cand: Dict[str, Any],
    *,
    require_v2: bool,
    allow_partial_v2: bool,
    allow_source_projection: bool,
) -> Tuple[bool, str]:
    if q(cand, "keep", "status") == "KEEP_EXTERNAL":
        return False, "keep_external_not_generated"
    if not candidate_text(cand):
        return False, "empty_candidate_text"
    if is_source_projection_candidate(cand) and not allow_source_projection:
        return False, "source_projection_candidate_disabled"
    if not q(cand, "validation", "V0", "passed"):
        return False, "v0_failed"
    if not q(cand, "validation", "V1", "passed"):
        return False, "v1_failed"
    v2_status = str(q(cand, "validation", "V2", "status") or "")
    if require_v2:
        if not q(cand, "validation", "V2", "passed"):
            if allow_partial_v2 and v2_status == "partial_needs_v3":
                pass
            else:
                return False, "v2_failed_or_partial"
    if q(cand, "validation", "V3", "passed"):
        return False, "already_has_v3_pass"
    return True, "taskable"


def compact_counter(counter: Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def build_task(record: Dict[str, Any], slot_set: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
    slot_id = str(slot.get("slot_id") or "")
    cand_id = str(cand.get("candidate_id") or "")
    text = candidate_text(cand)
    task_id = "v3task_" + stable_id(record.get("function_key"), record.get("input_line"), record.get("func_name"), slot_id, cand_id, text)
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
        "candidate_text": text,
        "expected_result_schema": {
            "required_identity": ["function_key or input_line+func_name", "slot_id", "candidate_id"],
            "required_semantic_pass_field": "one of differential_pass, trace_match, symbolic_match must be true",
            "required_confidence_level": "C3/C4/C5",
            "forbidden_shortcut": "source_teacher_projection or compile_smoke alone",
        },
        "skeleton": record.get("verified_skeleton_text") or record.get("verified_skeleton_preview"),
        "module_b": record.get("module_b"),
        "slot_contract": {
            "control_contract": slot.get("control_contract"),
            "data_contract": slot.get("data_contract"),
            "effect_contract": slot.get("effect_contract"),
            "grammar_contract": slot.get("grammar_contract"),
            "region": slot.get("region"),
            "observable_semantics": slot.get("observable_semantics"),
        },
        "candidate_validation": cand.get("validation"),
        "candidate_confidence": cand.get("confidence"),
        "candidate_provenance": cand.get("provenance"),
        "candidate_is_source_projection": is_source_projection_candidate(cand),
        "notes": [
            "Validator must compare candidate observable behavior against binary/source semantic slice.",
            "Passing output must report differential_pass, trace_match, or symbolic_match; source projection provenance or compile smoke alone is insufficient.",
        ],
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    out_path = Path(args.out_jsonl).expanduser().resolve()
    report_path = Path(args.out_txt).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "functions_seen": 0,
        "slots_seen": 0,
        "candidates_seen": 0,
        "tasks_written": 0,
        "skip_reason_dist": Counter(),
        "slot_type_dist": Counter(),
        "candidate_tier_dist": Counter(),
    }

    with out_path.open("w", encoding="utf-8") as out:
        for _line_no, record in iter_jsonl(Path(args.candidates_jsonl).expanduser().resolve(), limit=int(args.limit)):
            stats["functions_seen"] += 1
            per_function = 0
            for slot_set in record.get("slot_candidate_sets") or []:
                if not isinstance(slot_set, dict):
                    continue
                slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
                stats["slots_seen"] += 1
                stats["slot_type_dist"][str(slot.get("slot_type") or "UNKNOWN")] += 1
                for cand in slot_set.get("candidates") or []:
                    if not isinstance(cand, dict):
                        continue
                    stats["candidates_seen"] += 1
                    stats["candidate_tier_dist"][str(q(cand, "confidence", "confidence_level") or "C0")] += 1
                    ok, reason = candidate_is_taskable(
                        cand,
                        require_v2=bool(args.require_v2),
                        allow_partial_v2=bool(args.allow_partial_v2_tasks),
                        allow_source_projection=bool(args.allow_source_projection_tasks),
                    )
                    if not ok:
                        stats["skip_reason_dist"][reason] += 1
                        continue
                    out.write(json.dumps(build_task(record, slot_set, cand), ensure_ascii=False, sort_keys=True) + "\n")
                    stats["tasks_written"] += 1
                    per_function += 1
                    if int(args.max_tasks_per_function) > 0 and per_function >= int(args.max_tasks_per_function):
                        break
                if int(args.max_tasks_per_function) > 0 and per_function >= int(args.max_tasks_per_function):
                    break

    lines = [
        "=== Module C ECSL V3 Validation Task Export ===",
        f"candidates_jsonl = {args.candidates_jsonl}",
        f"out_jsonl = {out_path}",
        "",
        f"functions_seen = {stats['functions_seen']}",
        f"slots_seen = {stats['slots_seen']}",
        f"candidates_seen = {stats['candidates_seen']}",
        f"tasks_written = {stats['tasks_written']}",
        f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}",
        f"candidate_tier_dist = {compact_counter(stats['candidate_tier_dist'])}",
        f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}",
        "",
        "Interpretation:",
        "- This is a task file, not V3 evidence.",
        "- A downstream validator must produce JSONL rows with differential_pass, trace_match, or symbolic_match plus confidence_level C3+.",
        "- Do not train hard SFT from this task file directly.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tasks-per-function", type=int, default=0)
    ap.add_argument("--require-v2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--allow-partial-v2-tasks",
        action="store_true",
        help="Compatibility mode: export partial_needs_v3 candidates. Default exports only clean V2 pass candidates.",
    )
    ap.add_argument(
        "--allow-source-projection-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export source projection candidates as concrete texts for an independent V3 backend. The backend still must prove differential/trace/symbolic equivalence.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stats = build(args)
    print(f"functions_seen = {stats['functions_seen']}")
    print(f"tasks_written = {stats['tasks_written']}")
    print(f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}")
    print(f"saved_tasks_jsonl = {args.out_jsonl}")
    print(f"saved_report = {args.out_txt}")


if __name__ == "__main__":
    main()

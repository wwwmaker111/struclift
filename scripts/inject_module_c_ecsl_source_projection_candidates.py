#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inject source-teacher projection fills as concrete ECSL candidates.

This is a candidate-synthesis bridge, not a V3 proof.  The injected candidates
are useful for audit/bootstrap and for downstream validators that need concrete
text to check.  They become C2 only when the local V0/V1/V2 gates cleanly pass,
and always keep ``hard_ce_allowed=false`` until an independent
differential/trace/symbolic or human-audit result promotes them.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIER_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}


def load_candidates_module() -> Any:
    path = ROOT / "scripts" / "build_module_c_ecsl_candidates.py"
    spec = importlib.util.spec_from_file_location("_ecsl_candidates_for_injection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def compact_counter(counter: collections.Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def canonical_slot_id(value: Any) -> str:
    s = str(value or "").strip()
    m = re.search(r"slot_(\d+)", s)
    if m:
        return f"slot_{int(m.group(1)):03d}"
    return s


def stable_id(*parts: Any, n: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def function_keys(row: Dict[str, Any]) -> List[Tuple[Any, Any, str]]:
    func = str(row.get("func_name") or "")
    return [
        (row.get("function_key"), None, ""),
        (row.get("example"), row.get("input_line"), func),
        (None, row.get("input_line"), func),
    ]


def load_projection_index(path: Path) -> Dict[Tuple[Any, Any, str], Dict[str, Any]]:
    out: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(path):
        for key in function_keys(row):
            out[key] = row
    return out


def lookup_projection(index: Dict[Tuple[Any, Any, str], Dict[str, Any]], row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in function_keys(row):
        if key in index:
            return index[key]
    return None


def candidate_text(cand: Dict[str, Any]) -> str:
    for item in cand.get("c_renderings") or []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            return str(item.get("text") or "").strip()
    return ""


def source_slot_map(row: Dict[str, Any], *, min_source_tier: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in row.get("slot_v3_results") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("confidence_level") or item.get("tier") or "C0")
        target = str(item.get("target_text") or "").strip()
        passed = strict_true(item.get("source_teacher_passed")) or strict_true(item.get("passed"))
        if not (passed and target and TIER_ORDER.get(level, 0) >= TIER_ORDER.get(min_source_tier, 2)):
            continue
        sid = canonical_slot_id(item.get("slot_id"))
        if sid:
            out[sid] = item
    return out


def injected_candidate(
    record: Dict[str, Any],
    slot: Dict[str, Any],
    source_slot: Dict[str, Any],
    validation: Dict[str, Any],
    confidence: Dict[str, Any],
) -> Dict[str, Any]:
    slot_id = canonical_slot_id(slot.get("slot_id") or source_slot.get("slot_id"))
    text = str(source_slot.get("target_text") or "").strip()
    cid = "srcproj_" + stable_id(record.get("function_key"), record.get("input_line"), record.get("func_name"), slot_id, text, n=14)
    return {
        "schema": "SlotCandidate.v0",
        "candidate_id": cid,
        "slot_id": slot_id,
        "semantic_object": {
            "kind": "source_teacher_projection_fill",
            "summary": "concrete source-teacher projection candidate for downstream validation",
        },
        "typed_canonical_ir": {
            "expr_tree": None,
            "stmt_region": None,
            "transformer": {"live_in_to_live_out": None, "status": "source_projection_candidate_not_v3"},
            "type": None,
        },
        "c_renderings": [
            {
                "text": text,
                "style": "source_teacher_projection",
                "requires": ["independent_v3_differential_trace_or_symbolic_validation"],
            }
        ],
        "provenance": {
            "source": ["source_teacher_projection"],
            "oracle_type": "training_source_teacher_slot_projection",
            "source_slot_index": source_slot.get("slot_index"),
            "original_slot_index": source_slot.get("original_slot_index"),
            "status": "concrete_candidate_not_v3",
        },
        "validation": validation,
        "confidence": confidence,
    }


def inject(args: argparse.Namespace) -> Dict[str, Any]:
    cand_mod = load_candidates_module()
    projection_index = load_projection_index(Path(args.source_projection_jsonl).expanduser().resolve())
    out_path = Path(args.out_candidates_jsonl).expanduser().resolve()
    report_path = Path(args.out_txt).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "functions_seen": 0,
        "functions_with_projection": 0,
        "slots_seen": 0,
        "source_slots_available": 0,
        "candidates_injected": 0,
        "injected_confidence_dist": collections.Counter(),
        "injected_v2_status_dist": collections.Counter(),
        "skip_reason_dist": collections.Counter(),
        "slot_type_dist": collections.Counter(),
    }
    validator_args = argparse.Namespace(
        enable_v2_validator=True,
        v2_require_full_dataflow=bool(args.v2_require_full_dataflow),
        external_v3_jsonl=None,
        v3_command=None,
        enable_v3_compile_smoke=False,
        v3_cc="cc",
        v3_compile_keep_dir=None,
        allow_compile_smoke_as_v3=False,
        v0_cc=str(args.v0_cc),
    )
    with Path(args.candidates_jsonl).expanduser().resolve().open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(src, 1):
            if int(args.limit) > 0 and line_no > int(args.limit):
                break
            if not line.strip():
                continue
            record = json.loads(line)
            stats["functions_seen"] += 1
            projection = lookup_projection(projection_index, record)
            source_map = source_slot_map(projection or {}, min_source_tier=str(args.min_source_tier)) if projection else {}
            if projection:
                stats["functions_with_projection"] += 1
            if source_map:
                stats["source_slots_available"] += len(source_map)
            for slot_set in record.get("slot_candidate_sets") or []:
                if not isinstance(slot_set, dict):
                    continue
                slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
                slot_id = canonical_slot_id(slot.get("slot_id"))
                stats["slots_seen"] += 1
                stats["slot_type_dist"][str(slot.get("slot_type") or "UNKNOWN").upper()] += 1
                source_slot = source_map.get(slot_id)
                if not source_slot:
                    stats["skip_reason_dist"]["no_source_projection_for_slot"] += 1
                    continue
                text = str(source_slot.get("target_text") or "").strip()
                existing = [candidate_text(c) for c in (slot_set.get("candidates") or []) if isinstance(c, dict)]
                if text in existing and not bool(args.allow_duplicate_text):
                    stats["skip_reason_dist"]["candidate_text_already_exists"] += 1
                    continue
                raw_cand = {
                    "schema": "SlotCandidate.v0",
                    "candidate_id": "srcproj_" + stable_id(record.get("function_key"), record.get("input_line"), record.get("func_name"), slot_id, text, n=14),
                    "slot_id": slot_id,
                    "c_renderings": [{"text": text, "style": "source_teacher_projection"}],
                    "provenance": {"source": ["source_teacher_projection"], "oracle_type": "training_source_teacher_slot_projection"},
                }
                validation = cand_mod.validate_candidate(record, slot, raw_cand, validator_args, {})
                confidence = cand_mod.score_confidence(slot, raw_cand, validation)
                if not bool(args.allow_partial_v2_as_c2) and q(validation, "V2", "status") == "partial_needs_v3":
                    confidence = {
                        "confidence_level": "C1",
                        "confidence_weight": 0.0,
                        "hard_ce_allowed": False,
                        "reasons": ["source_projection_partial_v2_needs_independent_v3"],
                    }
                cand = injected_candidate(record, slot, source_slot, validation, confidence)
                slot_set.setdefault("candidates", []).insert(0, cand)
                slot_set["candidate_count"] = len(slot_set.get("candidates") or [])
                tier_dist = collections.Counter(str(q(c, "confidence", "confidence_level") or "C0") for c in slot_set.get("candidates") or [])
                slot_set["tier_dist"] = dict(tier_dist)
                stats["candidates_injected"] += 1
                stats["injected_confidence_dist"][str(q(cand, "confidence", "confidence_level") or "C0")] += 1
                stats["injected_v2_status_dist"][str(q(cand, "validation", "V2", "status") or "unknown")] += 1
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    lines = [
        "=== Module C ECSL Source Projection Candidate Injection ===",
        f"candidates_jsonl = {args.candidates_jsonl}",
        f"source_projection_jsonl = {args.source_projection_jsonl}",
        f"out_candidates_jsonl = {out_path}",
        "",
        f"functions_seen = {stats['functions_seen']}",
        f"functions_with_projection = {stats['functions_with_projection']}",
        f"slots_seen = {stats['slots_seen']}",
        f"source_slots_available = {stats['source_slots_available']}",
        f"candidates_injected = {stats['candidates_injected']}",
        f"injected_confidence_dist = {compact_counter(stats['injected_confidence_dist'])}",
        f"injected_v2_status_dist = {compact_counter(stats['injected_v2_status_dist'])}",
        f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}",
        f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}",
        "",
        "Interpretation:",
        "- Injected source-projection rows are concrete candidates, not automatic V3.",
        "- Only injected candidates with clean V0/V1/V2 pass become C2; partial candidates stay C1 until independent V3.",
        "- They need independent differential/trace/symbolic or human audit evidence before hard SFT.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--source-projection-jsonl", required=True)
    ap.add_argument("--out-candidates-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-source-tier", default="C2", choices=["C2", "C3", "C4", "C5"])
    ap.add_argument("--allow-duplicate-text", action="store_true")
    ap.add_argument("--v2-require-full-dataflow", action="store_true")
    ap.add_argument("--v0-cc", default="cc")
    ap.add_argument(
        "--allow-partial-v2-as-c2",
        action="store_true",
        help="Compatibility mode only: keep partial_needs_v3 source projection candidates at C2. Default demotes them to C1.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stats = inject(args)
    print(f"functions_seen = {stats['functions_seen']}")
    print(f"functions_with_projection = {stats['functions_with_projection']}")
    print(f"source_slots_available = {stats['source_slots_available']}")
    print(f"candidates_injected = {stats['candidates_injected']}")
    print(f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}")
    print(f"saved_candidates_jsonl = {args.out_candidates_jsonl}")
    print(f"saved_report = {args.out_txt}")


if __name__ == "__main__":
    main()

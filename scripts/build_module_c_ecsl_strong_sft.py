#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Module-C ECSL strong-label Stage3 SFT records.

This builder is intentionally conservative:

* V0/V1 come from local slot grammar, strict source teacher cleanup, and fixed
  Module-B skeleton preservation.
* V2 is read from the slot-candidate table as concrete V0/V1/V2-compatible
  slot fills.  Source/B skeleton constructability remains diagnostic evidence,
  but it must not veto skeleton-compatible candidates just because source AST
  projection is incomplete.  V2 is still not a differential execution proof.
* V3 is accepted only from an explicit external validation result JSONL.  The
  default path does not promote C2 into hard CE labels.

Therefore the default ``--min-hard-tier C3`` may produce zero SFT rows until a
real V3 verifier is connected.  That is a feature, not a bug: weak labels must
not be silently treated as semantic strong labels.
"""

from __future__ import annotations

import argparse
import collections
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
FILL_PLACEHOLDER_RE = re.compile(r"<FILL_[A-Z_]+:[^>]+>")
DIAGNOSTIC_V3_ORACLE_TYPES = {
    "v3_differential_lite_candidate_reference",
    "training_source_teacher_slot_projection",
    "training_source_teacher_strict_render",
    "source_teacher_strict_projection",
}
UNVERIFIED_CONTRACT_PROOF_KINDS = {
    "contract_backed_symbolic",
    "contract_backed_symbolic_needs_audit",
}
VERIFIED_CONTRACT_PROOF_KINDS = {
    "contract_backed_symbolic_flagdef_verified",
    "control_contract_symbolic",
    "contract_empty_effect_symbolic",
    "contract_return_constant_symbolic",
}


def load_overfit_helpers() -> Any:
    path = ROOT / "scripts" / "build_module_c_ecsl_overfit_sft.py"
    spec = importlib.util.spec_from_file_location("_ecsl_overfit_helpers", path)
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def human_audit_passed(item: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(item, dict):
        return False
    verdict = str(
        item.get("reviewer_verdict")
        or item.get("human_verdict")
        or item.get("manual_verdict")
        or ""
    ).strip().lower()
    return bool(
        strict_true(item.get("human_audit_pass"))
        or strict_true(item.get("manual_semantic_pass"))
        or strict_true(item.get("human_semantic_pass"))
        or verdict in {"pass", "passed", "accept", "accepted", "approved", "yes", "true"}
    )


def v3_oracle_type(item: Optional[Dict[str, Any]]) -> str:
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("oracle_type")
        or q(item, "evidence", "oracle_type")
        or q(item, "validation", "V3", "oracle_type")
        or q(item, "validation", "V3", "evidence", "oracle_type")
        or ""
    )


def v3_proof_kind(item: Optional[Dict[str, Any]]) -> str:
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("proof_kind")
        or q(item, "evidence", "proof_kind")
        or q(item, "validation", "V3", "proof_kind")
        or q(item, "validation", "V3", "evidence", "proof_kind")
        or ""
    )


def automatic_v3_passed(item: Optional[Dict[str, Any]], *, allow_diagnostic_v3: bool = False) -> bool:
    if not isinstance(item, dict):
        return False
    oracle_type = v3_oracle_type(item)
    if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES and not allow_diagnostic_v3:
        return False
    proof_kind = v3_proof_kind(item)
    if proof_kind in UNVERIFIED_CONTRACT_PROOF_KINDS and not allow_diagnostic_v3:
        return False
    return bool(
        strict_true(item.get("differential_pass"))
        or strict_true(item.get("trace_match"))
        or strict_true(item.get("symbolic_match"))
        or strict_true(q(item, "V3", "differential_pass"))
        or strict_true(q(item, "V3", "trace_match"))
        or strict_true(q(item, "V3", "symbolic_match"))
        or strict_true(q(item, "validation", "V3", "differential_pass"))
        or strict_true(q(item, "validation", "V3", "trace_match"))
        or strict_true(q(item, "validation", "V3", "symbolic_match"))
    )


def strong_semantic_evidence_kind(item: Optional[Dict[str, Any]], *, allow_diagnostic_v3: bool = False) -> Optional[str]:
    if automatic_v3_passed(item, allow_diagnostic_v3=allow_diagnostic_v3):
        proof_kind = v3_proof_kind(item)
        if proof_kind in VERIFIED_CONTRACT_PROOF_KINDS:
            return proof_kind
        return "automatic_v3"
    if human_audit_passed(item):
        return "human_audit"
    return None


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def load_external_v3(path: Optional[str]) -> Dict[Tuple[Any, Any, str], Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    out: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    grouped: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(p):
        func = str(row.get("func_name") or "")
        keys = [
            (row.get("example"), row.get("input_line"), func),
            (None, row.get("input_line"), func),
            (row.get("function_key"), None, ""),
        ]
        is_slot_row = bool(row.get("slot_id")) and bool(
            row.get("candidate_id") or row.get("candidate_text") or row.get("target_text")
        )
        if is_slot_row:
            semantic = bool(
                automatic_v3_passed(row, allow_diagnostic_v3=True)
                or human_audit_passed(row)
            )
            level = str(row.get("confidence_level") or row.get("tier") or ("C3" if semantic else "C0"))
            slot_item = {
                "slot_id": row.get("slot_id"),
                "slot_index": row.get("slot_index"),
                "slot_type": row.get("slot_type"),
                "candidate_id": row.get("candidate_id"),
                "target_text": row.get("target_text") or row.get("candidate_text"),
                "candidate_text": row.get("candidate_text") or row.get("target_text"),
                "confidence_level": level,
                "tier": level,
                "confidence_weight": row.get("confidence_weight"),
                "oracle_type": row.get("oracle_type") or "external_v3_task_result",
                "reason": row.get("reason") or row.get("status"),
                "differential_pass": row.get("differential_pass"),
                "trace_match": row.get("trace_match"),
                "symbolic_match": row.get("symbolic_match"),
                "hard_ce_allowed": row.get("hard_ce_allowed", semantic),
                "evidence": row.get("evidence") if isinstance(row.get("evidence"), dict) else {},
            }
            for key in keys:
                bucket = grouped.setdefault(
                    key,
                    {
                        "schema": "ModuleC.ECSL.ExternalV3GroupedByFunction.v0",
                        "function_key": row.get("function_key"),
                        "example": row.get("example"),
                        "input_line": row.get("input_line"),
                        "func_name": func,
                        "label_policy": "slot",
                        "oracle_type": "external_v3_task_results",
                        "confidence_level": "C3",
                        "slot_v3_results": [],
                    },
                )
                bucket["slot_v3_results"].append(slot_item)
            continue
        for key in keys:
            out[key] = row
    for key, row in grouped.items():
        # Prefer explicit function-level rows if present; otherwise expose the
        # grouped task-level validator rows as slot-level V3 evidence.
        out.setdefault(key, row)
    return out


def external_v3_row(index: Dict[Tuple[Any, Any, str], Dict[str, Any]], cand_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    func = str(cand_record.get("func_name") or "")
    keys = [
        (cand_record.get("example"), cand_record.get("input_line"), func),
        (None, cand_record.get("input_line"), func),
        (cand_record.get("function_key"), None, ""),
    ]
    for key in keys:
        if key in index:
            return index[key]
    return None


def row_has_slot_level_v3(row: Optional[Dict[str, Any]]) -> bool:
    return bool(isinstance(row, dict) and isinstance(row.get("slot_v3_results"), list))


def canonical_slot_key(value: Any, helper: Any) -> str:
    try:
        return helper.canonical_slot_id(value)
    except Exception:
        s = str(value or "").strip()
        m = re.search(r"slot_(\d+)", s)
        if m:
            return f"slot_{int(m.group(1)):03d}"
        return s


def external_slot_v3_patch(
    rec: Dict[str, Any],
    row: Dict[str, Any],
    helper: Any,
    *,
    min_tier: str,
    allow_diagnostic_v3: bool = False,
) -> Dict[str, Any]:
    """Return prompt-order targets/flags from per-slot V3 evidence.

    The ECSL document allows unmatched B slots to stay out of hard SFT.  This
    patch therefore keeps only C3+ slot targets and masks every other slot.
    """
    slot_ids = [str(x) for x in (rec.get("sft_prompt_slot_ids") or [])]
    original_targets = list(rec.get("sft_teacher_targets_prompt_order") or rec.get("sft_teacher_targets") or [])
    by_id: Dict[str, Dict[str, Any]] = {}
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in row.get("slot_v3_results") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("confidence_level") or item.get("tier") or "C0")
        evidence_kind = strong_semantic_evidence_kind(item, allow_diagnostic_v3=allow_diagnostic_v3)
        passed = evidence_kind is not None
        hard_ok = strict_true(item.get("hard_ce_allowed")) if "hard_ce_allowed" in item else passed
        if not (passed and hard_ok and TIER_ORDER.get(level, 0) >= TIER_ORDER.get(str(min_tier), 3)):
            continue
        sid = canonical_slot_key(item.get("slot_id"), helper)
        if sid:
            by_id[sid] = item
        idx = safe_int(item.get("slot_index"), -1)
        if idx >= 0:
            by_index[idx] = item

    targets: List[str] = []
    flags: List[bool] = []
    applied: List[Dict[str, Any]] = []
    for i, sid_raw in enumerate(slot_ids):
        sid = canonical_slot_key(sid_raw, helper)
        item = by_id.get(sid) or by_index.get(i)
        text = ""
        if item is not None:
            text = str(item.get("target_text") or "")
        if not text and i < len(original_targets) and item is not None:
            text = str(original_targets[i] or "")
        keep = bool(item is not None and text.strip())
        targets.append(text if keep else "")
        flags.append(keep)
        if keep:
            applied.append(
                {
                    "slot_index": i,
                    "slot_id": sid,
                    "slot_type": item.get("slot_type"),
                    "confidence_level": item.get("confidence_level") or item.get("tier") or "C3",
                    "confidence_weight": item.get("confidence_weight"),
                    "source": item.get("oracle_type") or row.get("oracle_type"),
                    "evidence_kind": evidence_kind,
                    "reviewer": item.get("reviewer") or row.get("reviewer"),
                    "reason": item.get("reason") or row.get("reason"),
                }
            )
    return {"targets": targets, "flags": flags, "applied": applied}


def external_slot_v3_summary(
    row: Optional[Dict[str, Any]],
    helper: Any,
    *,
    min_tier: str,
    required_slot_count: int,
    allow_diagnostic_v3: bool = False,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    if not row_has_slot_level_v3(row):
        return False, "not_available", "no_external_slot_v3_result", {}
    assert row is not None
    passed_rows = []
    best_level = "C0"
    fail_reasons: List[str] = []
    evidence_kind_dist: collections.Counter = collections.Counter()
    for item in row.get("slot_v3_results") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("confidence_level") or item.get("tier") or "C0")
        if TIER_ORDER.get(level, 0) > TIER_ORDER.get(best_level, 0):
            best_level = level
        evidence_kind = strong_semantic_evidence_kind(item, allow_diagnostic_v3=allow_diagnostic_v3)
        passed = evidence_kind is not None
        hard_ok = strict_true(item.get("hard_ce_allowed")) if "hard_ce_allowed" in item else passed
        if passed and hard_ok and TIER_ORDER.get(level, 0) >= TIER_ORDER.get(str(min_tier), 3):
            passed_rows.append(item)
            evidence_kind_dist[str(evidence_kind)] += 1
        elif len(fail_reasons) < 8:
            oracle_type = v3_oracle_type(item)
            if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES and not allow_diagnostic_v3:
                fail_reasons.append(f"diagnostic_v3_oracle_rejected={oracle_type}")
            else:
                fail_reasons.append(str(item.get("reason") or "slot_v3_failed"))
    total = len(row.get("slot_v3_results") or [])
    needed = max(1, int(required_slot_count))
    evidence = {
        "slot_count": total,
        "v3_pass_slot_count": len(passed_rows),
        "required_slot_count": needed,
        "label_policy": row.get("label_policy"),
        "oracle_type": row.get("oracle_type"),
        "evidence_kind_dist": dict(evidence_kind_dist),
        "sample_fail_reasons": fail_reasons,
    }
    if len(passed_rows) >= needed:
        return True, str(row.get("confidence_level") or "C3"), f"external_slot_v3_pass_slots={len(passed_rows)}/{total}", evidence
    return False, best_level, f"external_slot_v3_insufficient_pass_slots={len(passed_rows)}/{total}", evidence


def candidate_validation_has_semantic_v3(v3: Any, *, allow_diagnostic_v3: bool = False) -> bool:
    if not isinstance(v3, dict):
        return False
    if automatic_v3_passed(v3, allow_diagnostic_v3=allow_diagnostic_v3):
        return True
    evidence = v3.get("evidence") if isinstance(v3.get("evidence"), dict) else {}
    for key in ("external_v3", "parsed"):
        item = evidence.get(key)
        oracle_type = v3_oracle_type(item) if isinstance(item, dict) else ""
        if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES and not allow_diagnostic_v3:
            continue
        if isinstance(item, dict) and (
            strict_true(item.get("differential_pass"))
            or strict_true(item.get("trace_match"))
            or strict_true(item.get("symbolic_match"))
        ):
            return True
    command = evidence.get("v3_command")
    if isinstance(command, dict):
        parsed = q(command, "evidence", "parsed")
        oracle_type = v3_oracle_type(parsed) if isinstance(parsed, dict) else ""
        if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES and not allow_diagnostic_v3:
            return False
        if isinstance(parsed, dict) and (
            strict_true(parsed.get("differential_pass"))
            or strict_true(parsed.get("trace_match"))
            or strict_true(parsed.get("symbolic_match"))
        ):
            return True
    return False


def candidate_record_v3_summary(
    cand_record: Dict[str, Any],
    *,
    required_slot_count: int = 1,
    allow_diagnostic_v3: bool = False,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """Return whether the candidate table itself contains C3+ slot V3 evidence."""
    pass_count = 0
    total_slots = 0
    candidate_ids: List[str] = []
    failing_reasons: List[str] = []
    best_level = "C0"
    for slot_set in cand_record.get("slot_candidate_sets") or []:
        if not isinstance(slot_set, dict):
            continue
        total_slots += 1
        slot_pass = False
        for cand in slot_set.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            level = str(q(cand, "confidence", "confidence_level") or "C0")
            if TIER_ORDER.get(level, 0) > TIER_ORDER.get(best_level, 0):
                best_level = level
            v3 = q(cand, "validation", "V3")
            v3_ok = bool(q(cand, "validation", "V3", "passed")) and candidate_validation_has_semantic_v3(
                v3,
                allow_diagnostic_v3=allow_diagnostic_v3,
            )
            hard_ok = bool(q(cand, "confidence", "hard_ce_allowed"))
            if v3_ok and hard_ok and TIER_ORDER.get(level, 0) >= TIER_ORDER["C3"]:
                slot_pass = True
                candidate_ids.append(str(cand.get("candidate_id") or ""))
                break
            reason = q(cand, "validation", "V3", "reason")
            oracle_type = v3_oracle_type(v3) if isinstance(v3, dict) else ""
            if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES and not allow_diagnostic_v3 and len(failing_reasons) < 8:
                failing_reasons.append(f"diagnostic_v3_oracle_rejected={oracle_type}")
            elif reason and len(failing_reasons) < 8:
                failing_reasons.append(str(reason))
        if slot_pass:
            pass_count += 1
    evidence = {
        "slot_count": total_slots,
        "v3_pass_slot_count": pass_count,
        "required_slot_count": max(1, int(required_slot_count)),
        "candidate_ids": candidate_ids[:32],
        "best_candidate_level": best_level,
        "sample_fail_reasons": failing_reasons,
    }
    if total_slots > 0 and pass_count >= max(1, int(required_slot_count)):
        return True, "C3", f"candidate_record_v3_pass_slots={pass_count}/{total_slots}", evidence
    return False, best_level, "no_candidate_record_v3_pass", evidence


def candidate_text(cand: Dict[str, Any]) -> str:
    rows = cand.get("c_renderings")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("text") or "").strip():
                return str(row.get("text") or "").strip()
    return ""


def candidate_record_v2_summary(
    cand_record: Dict[str, Any],
    *,
    required_slot_count: int = 1,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Summarize slot-level C2 evidence already present in the candidate table.

    Strong-SFT records deliberately mask slot supervision until real V3 is
    available, so the rebuilt SFT object can have zero supervised slots.  That
    must not be treated as V2 failure.  Per the ECSL design, V2 is a candidate
    compatibility gate: at least one concrete V0/V1/V2-passing slot candidate
    is enough to mark the function as C2-ready for downstream V3 validation.
    """
    total_slots = 0
    generated_candidate_count = 0
    v0_v1_pass_candidate_count = 0
    v2_pass_slot_count = 0
    v2_pass_candidate_count = 0
    candidate_ids: List[str] = []
    failing_reasons: List[str] = []

    for slot_set in cand_record.get("slot_candidate_sets") or []:
        if not isinstance(slot_set, dict):
            continue
        total_slots += 1
        slot_has_v2 = False
        for cand in slot_set.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            if q(cand, "keep", "status") == "KEEP_EXTERNAL":
                continue
            text = candidate_text(cand)
            if not text:
                if len(failing_reasons) < 8:
                    failing_reasons.append("empty_candidate_text")
                continue
            generated_candidate_count += 1
            v0_ok = bool(q(cand, "validation", "V0", "passed"))
            v1_ok = bool(q(cand, "validation", "V1", "passed"))
            v2_ok = bool(q(cand, "validation", "V2", "passed"))
            v2_status = str(q(cand, "validation", "V2", "status") or "")
            if v0_ok and v1_ok:
                v0_v1_pass_candidate_count += 1
            if v0_ok and v1_ok and v2_ok and v2_status == "pass":
                v2_pass_candidate_count += 1
                slot_has_v2 = True
                if len(candidate_ids) < 32:
                    candidate_ids.append(str(cand.get("candidate_id") or ""))
            elif len(failing_reasons) < 8:
                reason = (
                    q(cand, "validation", "V2", "reason")
                    or q(cand, "validation", "V1", "reason")
                    or q(cand, "validation", "V0", "reason")
                    or "candidate_not_v2_compatible"
                )
                failing_reasons.append(str(reason))
        if slot_has_v2:
            v2_pass_slot_count += 1

    needed = max(1, int(required_slot_count))
    evidence = {
        "slot_count": total_slots,
        "generated_candidate_count": generated_candidate_count,
        "v0_v1_pass_candidate_count": v0_v1_pass_candidate_count,
        "v2_pass_candidate_count": v2_pass_candidate_count,
        "v2_pass_slot_count": v2_pass_slot_count,
        "required_slot_count": needed,
        "candidate_ids": candidate_ids,
        "sample_fail_reasons": failing_reasons,
    }
    if total_slots > 0 and v2_pass_slot_count >= needed:
        return True, f"candidate_record_v2_pass_slots={v2_pass_slot_count}/{total_slots}", evidence
    return False, f"candidate_record_v2_insufficient_pass_slots={v2_pass_slot_count}/{total_slots}", evidence


def apply_candidate_only_strong_labels(rec: Dict[str, Any], helper: Any, candidate_choices: Dict[str, Any]) -> Dict[str, Any]:
    """Keep supervision only for slots with C3+ candidate choices."""
    slot_ids = [str(x) for x in (rec.get("sft_prompt_slot_ids") or [])]
    targets: List[str] = []
    flags: List[bool] = []
    applied: List[Dict[str, Any]] = []
    for i, sid in enumerate(slot_ids):
        choice = candidate_choices.get(helper.canonical_slot_id(sid))
        if choice is None:
            targets.append("")
            flags.append(False)
            continue
        targets.append(str(choice.text))
        flags.append(True)
        applied.append(
            {
                "slot_index": i,
                "slot_id": sid,
                "text": str(choice.text),
                "candidate_id": str(choice.candidate_id),
                "slot_type": str(choice.slot_type),
                "confidence_level": str(choice.confidence_level),
                "source": str(choice.source),
                "reason": "candidate_level_v3_c3_strong_label",
            }
        )
    return {"targets": targets, "flags": flags, "applied": applied}


def v3_passed(row: Optional[Dict[str, Any]], *, allow_diagnostic_v3: bool = False) -> Tuple[bool, str, str]:
    if not row:
        return False, "not_available", "no_external_v3_result"
    oracle_type = v3_oracle_type(row)
    if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES and not allow_diagnostic_v3:
        level = str(row.get("confidence_level") or row.get("tier") or "C0")
        return False, level, f"diagnostic_v3_oracle_rejected={oracle_type}"
    semantic_pass = bool(
        strict_true(row.get("differential_pass"))
        or strict_true(row.get("trace_match"))
        or strict_true(row.get("symbolic_match"))
        or strict_true(q(row, "V3", "differential_pass"))
        or strict_true(q(row, "V3", "trace_match"))
        or strict_true(q(row, "V3", "symbolic_match"))
    )
    level = str(row.get("confidence_level") or row.get("tier") or ("C3" if semantic_pass else "C0"))
    reason = str(
        row.get("reason")
        or row.get("status")
        or ("external_v3_semantic_pass" if semantic_pass else "external_v3_missing_semantic_proof")
    )
    return semantic_pass and TIER_ORDER.get(level, 0) >= TIER_ORDER["C3"], level, reason


def count_supervised_slots(record: Dict[str, Any]) -> int:
    flags = list(
        record.get("sft_teacher_slot_supervise_prompt_order")
        or record.get("sft_teacher_slot_supervise")
        or []
    )
    slot_ids = record.get("sft_prompt_slot_ids") if isinstance(record.get("sft_prompt_slot_ids"), list) else []
    while len(flags) < len(slot_ids):
        flags.append(False)
    flags = flags[: len(slot_ids)] if slot_ids else flags
    return sum(1 for x in flags if bool(x))


def count_supervised_tokens(record: Dict[str, Any]) -> int:
    return sum(1 for x in record.get("sft_labels") or [] if safe_int(x, -100) != -100)


def b_gate_pass(cand_record: Dict[str, Any], mb: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failed: List[str] = []
    if cand_record.get("b_gate") and not bool(q(cand_record, "b_gate", "pass")):
        failed.append("ecsl_b_gate_failed")
    rep = q(mb, "validator", "selected_report") or q(mb, "verified_skeleton", "validator_report") or {}
    if isinstance(rep, dict):
        required = [
            "valid",
            "bb_coverage_ok",
            "edge_preservation_ok",
            "strict_edge_preservation_ok",
            "slot_grounding_ok",
            "parse_success",
        ]
        for key in required:
            if key in rep and not bool(rep.get(key)):
                failed.append(f"validator_{key}_false")
        if rep.get("fatal_errors"):
            failed.append("validator_fatal_errors")
    st = q(mb, "document_metric_contract", "structuredness") or q(mb, "teacher_diagnostics", "ranker_metrics") or {}
    if isinstance(st, dict):
        if "slot_locality_ready" in st and not bool(st.get("slot_locality_ready")):
            failed.append("slot_locality_not_ready")
    return not failed, failed


def v0_v1_pass(record: Dict[str, Any]) -> Tuple[bool, bool, List[str]]:
    reasons: List[str] = []
    if count_supervised_slots(record) <= 0:
        reasons.append("no_supervised_slots_yet")
    if count_supervised_tokens(record) <= 0:
        reasons.append("no_supervised_tokens_yet")
    if record.get("sft_teacher_strict_drop_reasons"):
        # strict drop reasons are allowed only when they do not affect rendered output;
        # process_record(strict_drop_record=True) would already reject fatal cases.
        pass
    rendered = str(record.get("sft_output_text_pretty") or "")
    if not rendered.strip():
        reasons.append("empty_rendered_teacher")
    if FILL_PLACEHOLDER_RE.search(rendered):
        reasons.append("remaining_fill_placeholder_after_render")
    return not any(r.startswith(("empty_", "remaining_")) for r in reasons), not any(
        r.startswith(("empty_", "remaining_")) for r in reasons
    ), reasons


def module_c_contract(mb: Dict[str, Any]) -> Dict[str, Any]:
    value = mb.get("module_c_contract")
    return value if isinstance(value, dict) else {}


def ranker_metrics(mb: Dict[str, Any]) -> Dict[str, Any]:
    value = q(mb, "teacher_diagnostics", "ranker_metrics")
    return value if isinstance(value, dict) else {}


def v2_static_constructability_pass(mb: Dict[str, Any], record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Return function-level V2 precheck; does not claim differential equivalence.

    This is intentionally a weak proxy.  Older runs used Module-B/source-teacher
    constructability fields as a hard V2 veto, which incorrectly punished valid
    skeleton-compatible slot labels when source AST projection was imperfect.
    Explicit unsafe-B facts still fail, but source-teacher coverage gaps are
    now warnings because ECSL is allowed to use candidate/V3 slot evidence
    instead of one-to-one source AST labels.
    """
    cc = module_c_contract(mb)
    rm = ranker_metrics(mb)
    reasons: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    if cc:
        checks = {
            "safe_for_modulec": bool(cc.get("safe_for_modulec")),
            "pred_slots_labelable": bool(cc.get("pred_slots_labelable")),
            "teacher_primary_coverage_ok": bool(cc.get("teacher_primary_coverage_ok")),
            "cond_count_match": bool(cc.get("cond_count_match")),
            "unsafe_nonempty_slot_count_zero": safe_int(cc.get("unsafe_nonempty_slot_count"), 999) == 0,
            "missing_teacher_primary_count_zero": safe_int(cc.get("missing_teacher_primary_count"), 999) == 0,
        }
        evidence["module_c_contract"] = checks
        for key, ok in checks.items():
            if not ok:
                if key in {"safe_for_modulec", "pred_slots_labelable", "unsafe_nonempty_slot_count_zero"}:
                    reasons.append(f"module_c_contract_{key}_failed")
                else:
                    warnings.append(f"needs_v3_or_slot_labels_module_c_contract_{key}_failed")
    elif rm:
        checks = {
            "modulec_safe_for_rank": bool(rm.get("modulec_safe_for_rank")),
            "modulec_teacher_render_structural_ready": bool(rm.get("modulec_teacher_render_structural_ready")),
            "modulec_teacher_primary_coverage_ok": bool(rm.get("modulec_teacher_primary_coverage_ok")),
            "modulec_cond_count_match": bool(rm.get("modulec_cond_count_match")),
            "modulec_unsafe_nonempty_slot_count_zero": safe_int(rm.get("modulec_unsafe_nonempty_slot_count"), 999) == 0,
            "modulec_missing_teacher_primary_count_zero": safe_int(rm.get("modulec_missing_teacher_primary_count"), 999) == 0,
        }
        evidence["ranker_metrics"] = checks
        for key, ok in checks.items():
            if not ok:
                if key in {"modulec_safe_for_rank", "modulec_teacher_render_structural_ready", "modulec_unsafe_nonempty_slot_count_zero"}:
                    reasons.append(f"ranker_metrics_{key}_failed")
                else:
                    warnings.append(f"needs_v3_or_slot_labels_ranker_metrics_{key}_failed")
    else:
        warnings.append("needs_v3_no_module_c_constructability_contract")

    if count_supervised_slots(record) <= 0:
        reasons.append("no_supervised_slots")
    evidence["warnings"] = warnings
    return not reasons, reasons, evidence


def assign_tier(v0: bool, v1: bool, v2: bool, v3: bool, v3_level: str) -> str:
    if v3 and v0 and v1:
        return v3_level if TIER_ORDER.get(v3_level, 0) >= TIER_ORDER["C3"] else "C3"
    if v2 and v0 and v1:
        return "C2"
    if v0 and v1:
        return "C1"
    return "C0"


def mark_c0_skip(stats: Dict[str, Any], reason: str) -> None:
    """Count a function that could not reach strict SFT/V0 grading."""
    stats["tier_dist"]["C0"] += 1
    stats["v0_status_dist"]["not_attempted"] += 1
    stats["v1_status_dist"]["not_attempted"] += 1
    stats["v2_status_dist"]["not_attempted"] += 1
    stats["v3_status_dist"]["not_available_or_fail"] += 1
    stats["skip_reason_dist"][str(reason).split(":")[0][:100]] += 1


def build(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
    helper = load_overfit_helpers()
    aug = helper.load_augment_module()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"需要 transformers: {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(int(args.max_fill_tokens))]
    tokenizer.add_tokens(fill_tokens)
    fill_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in fill_tokens]
    if args.save_tokenizer:
        save_tok = Path(args.save_tokenizer).expanduser().resolve()
        save_tok.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_tok))

    module_b_paths = [Path(p).expanduser().resolve() for p in args.module_b_jsonl]
    module_b_index = helper.load_module_b_index(module_b_paths, limit=int(args.max_module_b_rows))
    source_index: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    if args.source_jsonl:
        source_paths = [Path(p).expanduser().resolve() for p in args.source_jsonl]
        source_index = helper.load_module_b_index(source_paths, limit=int(args.max_source_rows))
    v3_index = load_external_v3(args.v3_results_jsonl)

    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    prefix_rewrites = [(str(a), str(b)) for a, b in (args.rewrite_source_prefix or [])]

    hard_records: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    skips: List[str] = []
    stats: Dict[str, Any] = {
        "rows_seen": 0,
        "hard_records": 0,
        "tier_dist": collections.Counter(),
        "v0_status_dist": collections.Counter(),
        "v1_status_dist": collections.Counter(),
        "v2_status_dist": collections.Counter(),
        "v3_status_dist": collections.Counter(),
        "skip_reason_dist": collections.Counter(),
        "slot_type_dist": collections.Counter(),
        "prompt_compact_mode_dist": collections.Counter(),
        "prompt_skeleton_compacted_dist": collections.Counter(),
        "sft_seq_len_dist": collections.Counter(),
        "hard_supervised_slot_count": 0,
        "hard_supervised_token_count": 0,
        "hard_record_supervised_slot_dist": collections.Counter(),
        "diagnostic_v3_rejected": collections.Counter(),
    }

    for _line_no, cand_record in iter_jsonl(Path(args.candidates_jsonl).expanduser().resolve(), limit=int(args.limit)):
        stats["rows_seen"] += 1
        key = helper.contract_function_key(cand_record)
        mb = helper.find_module_b_row(module_b_index, cand_record)
        if mb is None:
            skips.append(f"{key}: module_b_row_not_found")
            mark_c0_skip(stats, "module_b_row_not_found")
            audit_rows.append({"key": key, "confidence_level": "C0", "reasons": ["module_b_row_not_found"]})
            continue
        src = helper.find_module_b_row(source_index, cand_record) if source_index else None
        if source_index and src is None:
            src = helper.find_module_b_row(source_index, mb)
        ext_row = external_v3_row(v3_index, cand_record)
        has_slot_level_external_v3 = row_has_slot_level_v3(ext_row)
        try:
            skel = helper.external_skeleton_from_candidate_record(cand_record, mb)
        except Exception as e:
            reason = f"skeleton_error:{e}"
            skips.append(f"{key}: {reason}")
            mark_c0_skip(stats, "skeleton_error")
            audit_rows.append({"key": key, "confidence_level": "C0", "reasons": [reason]})
            continue

        work = helper.ensure_minimal_stage_fields(helper.merge_source_and_module_b(src, mb))
        try:
            rec, err = aug.process_record(
                work,
                tokenizer,
                fill_ids,
                source_root,
                emit_rl_fields=True,
                num_patterns=10,
                mask_prompt_labels=True,
                mask_non_fill_labels=True,
                mask_supervised_inputs=bool(args.mask_supervised_inputs),
                supervised_input_mask_prob=float(args.supervised_input_mask_prob),
                max_seq_len=int(args.max_seq_len),
                prefix_rewrites=prefix_rewrites,
                external_skeleton=skel,
                skeleton_source="module_c_ecsl_strong_verified_skeleton",
                strict_teacher=True,
                strict_teacher_drop_record=False,
                min_supervised_semantic_slots=0 if not has_slot_level_external_v3 else int(args.min_supervised_semantic_slots),
                asm_max_total_lines=int(args.asm_max_total_lines),
                asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                prompt_compact_mode=str(args.prompt_compact_mode),
                max_slot_context_chars=int(args.max_slot_context_chars),
            )
        except Exception as e:
            rec, err = None, f"process_record_exception:{e}"
        if rec is None:
            skips.append(f"{key}: {err}")
            mark_c0_skip(stats, str(err))
            audit_rows.append(
                {
                    "function_key": cand_record.get("function_key"),
                    "example": cand_record.get("example"),
                    "input_line": cand_record.get("input_line"),
                    "func_name": cand_record.get("func_name"),
                    "confidence_level": "C0",
                    "reasons": [err],
                }
            )
            continue

        if not has_slot_level_external_v3:
            # Audit V0/V1 on the fixed B skeleton even before V3 exists, but do
            # not let source-teacher projection create hard supervision.  Real
            # targets are patched in only after slot/function V3 evidence.
            empty_targets = ["" for _ in (rec.get("sft_prompt_slot_ids") or [])]
            empty_flags = [False for _ in empty_targets]
            try:
                helper.rebuild_sft_with_prompt_targets(
                    rec,
                    aug,
                    tokenizer,
                    fill_ids,
                    empty_targets,
                    empty_flags,
                )
            except Exception as e:
                rec = None
                err = f"empty_unverified_slot_rebuild_error:{e}"
            if rec is None:
                skips.append(f"{key}: {err}")
                mark_c0_skip(stats, str(err))
                audit_rows.append(
                    {
                        "function_key": cand_record.get("function_key"),
                        "example": cand_record.get("example"),
                        "input_line": cand_record.get("input_line"),
                        "func_name": cand_record.get("func_name"),
                        "confidence_level": "C0",
                        "reasons": [err],
                    }
                )
                continue

        bg_ok, bg_reasons = b_gate_pass(cand_record, mb)
        stats["prompt_compact_mode_dist"][str(rec.get("sft_prompt_compact_mode") or "unknown")] += 1
        stats["prompt_skeleton_compacted_dist"][str(bool(rec.get("sft_prompt_skeleton_compacted")))] += 1
        seq_len_bucket = (len(rec.get("sft_input_ids") or []) // 1024) * 1024
        stats["sft_seq_len_dist"][f"{seq_len_bucket}-{seq_len_bucket + 1023}"] += 1
        v0_ok, v1_ok, v01_reasons = v0_v1_pass(rec)
        if not bg_ok:
            v1_ok = False
        required_candidate_slots = max(1, int(args.min_supervised_semantic_slots))
        v2_ok, v2_reason, v2_evidence = candidate_record_v2_summary(
            cand_record,
            required_slot_count=required_candidate_slots,
        )
        function_v2_ok, function_v2_reasons, function_v2_evidence = v2_static_constructability_pass(mb, rec)
        v2_reasons: List[str] = [] if v2_ok else [v2_reason]
        v2_evidence = {
            **(v2_evidence or {}),
            "function_level_static_constructability": {
                "passed": bool(function_v2_ok),
                "reasons": function_v2_reasons,
                "evidence": function_v2_evidence,
                "note": "Function-level source/Module-B constructability is diagnostic here; slot-level candidate V2 drives C2 readiness.",
            },
        }
        candidate_choices = helper.select_candidates_for_overfit(
            cand_record,
            aug,
            min_tier=str(args.min_hard_tier),
            require_v0_v1=True,
        )
        cand_v3_ok, cand_v3_level, cand_v3_reason, cand_v3_evidence = candidate_record_v3_summary(
            cand_record,
            required_slot_count=required_candidate_slots,
            allow_diagnostic_v3=bool(args.allow_diagnostic_v3_as_hard_label),
        )
        ext_slot_v3_ok, ext_slot_v3_level, ext_slot_v3_reason, ext_slot_v3_evidence = external_slot_v3_summary(
            ext_row,
            helper,
            min_tier=str(args.min_hard_tier),
            required_slot_count=required_candidate_slots,
            allow_diagnostic_v3=bool(args.allow_diagnostic_v3_as_hard_label),
        )
        ext_v3_ok, ext_v3_level, ext_v3_reason = v3_passed(
            ext_row if not row_has_slot_level_v3(ext_row) else None,
            allow_diagnostic_v3=bool(args.allow_diagnostic_v3_as_hard_label),
        )
        if ext_slot_v3_ok:
            slot_patch = external_slot_v3_patch(
                rec,
                ext_row or {},
                helper,
                min_tier=str(args.min_hard_tier),
                allow_diagnostic_v3=bool(args.allow_diagnostic_v3_as_hard_label),
            )
            try:
                helper.rebuild_sft_with_prompt_targets(
                    rec,
                    aug,
                    tokenizer,
                    fill_ids,
                    slot_patch["targets"],
                    slot_patch["flags"],
                )
            except Exception as e:
                rec = None
                err = f"external_slot_v3_rebuild_error:{e}"
            if rec is None:
                skips.append(f"{key}: {err}")
                mark_c0_skip(stats, str(err))
                audit_rows.append(
                    {
                        "function_key": cand_record.get("function_key"),
                        "example": cand_record.get("example"),
                        "input_line": cand_record.get("input_line"),
                        "func_name": cand_record.get("func_name"),
                        "confidence_level": "C0",
                        "reasons": [err],
                    }
                )
                continue
            v3_ok, v3_level, v3_reason = ext_slot_v3_ok, ext_slot_v3_level, ext_slot_v3_reason
            v3_source = "external_slot_v3_jsonl"
            v0_ok, v1_ok, v01_reasons = v0_v1_pass(rec)
            # Slot-level ECSL follows the unmatched-mass rule: only verified
            # slots are supervised.  Function-level Module-C constructability
            # is retained as evidence, but it must not veto matched slots.
            function_v2_proxy_reasons = list(v2_reasons)
            v2_evidence = {
                **(v2_evidence or {}),
                "slot_level_unmatched_mass": True,
                "function_level_v2_proxy_passed_before_slot_mask": bool(v2_ok),
                "function_level_v2_proxy_reasons_before_slot_mask": function_v2_proxy_reasons,
                "external_slot_v3": ext_slot_v3_evidence,
            }
            v2_reasons = []
            v2_ok = True
            cand_v3_evidence = dict(cand_v3_evidence)
            cand_v3_evidence["external_slot_labels_applied"] = slot_patch["applied"]
            cand_v3_evidence["external_slot_v3"] = ext_slot_v3_evidence
        elif ext_v3_ok:
            v3_ok, v3_level, v3_reason = ext_v3_ok, ext_v3_level, ext_v3_reason
            v3_source = "external_function_v3_jsonl"
        elif cand_v3_ok:
            candidate_patch = apply_candidate_only_strong_labels(rec, helper, candidate_choices)
            try:
                helper.rebuild_sft_with_prompt_targets(
                    rec,
                    aug,
                    tokenizer,
                    fill_ids,
                    candidate_patch["targets"],
                    candidate_patch["flags"],
                )
            except Exception as e:
                rec = None
                err = f"candidate_v3_rebuild_error:{e}"
            if rec is None:
                skips.append(f"{key}: {err}")
                mark_c0_skip(stats, str(err))
                audit_rows.append(
                    {
                        "function_key": cand_record.get("function_key"),
                        "example": cand_record.get("example"),
                        "input_line": cand_record.get("input_line"),
                        "func_name": cand_record.get("func_name"),
                        "confidence_level": "C0",
                        "reasons": [err],
                    }
                )
                continue
            v3_ok, v3_level, v3_reason = cand_v3_ok, cand_v3_level, cand_v3_reason
            v3_source = "candidate_record_slot_v3"
            v0_ok, v1_ok, v01_reasons = v0_v1_pass(rec)
            v2_ok = True
            v2_reasons = []
            v2_evidence = {
                **(v2_evidence or {}),
                "candidate_slot_labels_applied": candidate_patch["applied"],
                "candidate_record_v3": cand_v3_evidence,
            }
            cand_v3_evidence["candidate_slot_labels_applied"] = candidate_patch["applied"]
        else:
            v3_ok = False
            v3_level = ext_v3_level if TIER_ORDER.get(ext_v3_level, 0) >= TIER_ORDER.get(cand_v3_level, 0) else cand_v3_level
            v3_reason = ext_v3_reason if ext_v3_reason != "no_external_v3_result" else cand_v3_reason
            v3_source = "none"
        for reason in (cand_v3_reason, ext_slot_v3_reason, ext_v3_reason):
            if "diagnostic_v3_oracle_rejected" in str(reason):
                stats["diagnostic_v3_rejected"][str(reason)[:160]] += 1
        tier = assign_tier(v0_ok, v1_ok, v2_ok, v3_ok, v3_level)

        reasons = []
        reasons.extend(bg_reasons)
        reasons.extend(v01_reasons)
        reasons.extend(v2_reasons if not v2_ok else [])
        if not v3_ok:
            reasons.append(f"v3_{v3_reason}")

        audit = {
            "function_key": cand_record.get("function_key"),
            "example": cand_record.get("example"),
            "input_line": cand_record.get("input_line"),
            "func_name": cand_record.get("func_name"),
            "confidence_level": tier,
            "hard_sft_eligible": TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(str(args.min_hard_tier), 3),
            "v0": {"passed": bool(v0_ok), "status": "pass" if v0_ok else "fail", "reasons": v01_reasons},
            "v1": {"passed": bool(v1_ok), "status": "pass" if v1_ok else "fail", "b_gate_reasons": bg_reasons},
            "v2": {
                "passed": bool(v2_ok),
                "status": "pass_static_constructability" if v2_ok else "fail_or_not_available",
                "reasons": v2_reasons,
                "evidence": v2_evidence,
                "note": "V2 here is static constructability/live-out proxy evidence, not V3 differential execution.",
            },
            "v3": {
                "passed": bool(v3_ok),
                "status": "pass" if v3_ok else "not_available_or_fail",
                "confidence_level": v3_level,
                "reason": v3_reason,
                "source": v3_source,
                "candidate_record_evidence": cand_v3_evidence,
                "external_slot_level": ext_slot_v3_level,
                "external_slot_reason": ext_slot_v3_reason,
                "external_slot_evidence": ext_slot_v3_evidence,
                "external_function_level": ext_v3_level,
                "external_function_reason": ext_v3_reason,
            },
            "supervised_slot_count": count_supervised_slots(rec),
            "supervised_token_count": count_supervised_tokens(rec),
            "sft_seq_len": len(rec.get("sft_input_ids") or []),
            "sft_prompt_compact_mode": rec.get("sft_prompt_compact_mode"),
            "sft_prompt_retry_trace": rec.get("sft_prompt_retry_trace"),
            "reasons": reasons,
        }
        audit_rows.append(audit)
        stats["tier_dist"][tier] += 1
        stats["v0_status_dist"]["pass" if v0_ok else "fail"] += 1
        stats["v1_status_dist"]["pass" if v1_ok else "fail"] += 1
        stats["v2_status_dist"]["pass" if v2_ok else "fail"] += 1
        stats["v3_status_dist"]["pass" if v3_ok else "not_available_or_fail"] += 1
        stats["slot_type_dist"].update(rec.get("slot_types") or [])

        if TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(str(args.min_hard_tier), 3):
            supervised_slot_count = count_supervised_slots(rec)
            supervised_token_count = count_supervised_tokens(rec)
            rec["module_c_ecsl_strong_label"] = {
                "schema": "ModuleC.ECSL.StrongSFT.v1",
                "confidence_level": tier,
                "hard_sft_eligible": True,
                "min_hard_tier": str(args.min_hard_tier),
                "label_granularity": "slot" if v3_source in {"external_slot_v3_jsonl", "candidate_record_slot_v3"} else "function",
                "supervised_slot_count": supervised_slot_count,
                "unmatched_slots_masked_from_hard_sft": v3_source in {"external_slot_v3_jsonl", "candidate_record_slot_v3"},
                "v0": audit["v0"],
                "v1": audit["v1"],
                "v2": audit["v2"],
                "v3": audit["v3"],
                "warning": "Only C3+ V3-validated slots have hard labels; unmatched slots are masked out.",
            }
            hard_records.append(rec)
            stats["hard_records"] += 1
            stats["hard_supervised_slot_count"] += supervised_slot_count
            stats["hard_supervised_token_count"] += supervised_token_count
            stats["hard_record_supervised_slot_dist"][supervised_slot_count] += 1

    return hard_records, audit_rows, skips, stats


def write_report(path: Path, *, args: argparse.Namespace, hard_records: Sequence[Dict[str, Any]], audit_rows: Sequence[Dict[str, Any]], skips: Sequence[str], stats: Dict[str, Any]) -> None:
    n = int(stats.get("rows_seen") or 0)
    lines: List[str] = []
    lines.append("=== Module C ECSL Strong Label SFT Build ===")
    lines.append(f"candidates_jsonl = {args.candidates_jsonl}")
    lines.append(f"module_b_jsonl = {list(args.module_b_jsonl)}")
    lines.append(f"source_jsonl = {list(args.source_jsonl or [])}")
    lines.append(f"v3_results_jsonl = {args.v3_results_jsonl or '<none>'}")
    lines.append("")
    lines.append(f"rows_seen = {n}")
    lines.append(f"audit_rows = {len(audit_rows)}")
    lines.append(f"hard_sft_records = {len(hard_records)}")
    lines.append(f"records_skipped_before_audit = {len(skips)}")
    lines.append(f"hard_sft_rate = {pct(len(hard_records), max(1, n))}")
    lines.append(f"hard_supervised_slot_count = {stats.get('hard_supervised_slot_count', 0)}")
    lines.append(f"hard_supervised_token_count = {stats.get('hard_supervised_token_count', 0)}")
    if hard_records:
        lines.append(f"hard_supervised_slots_per_record_mean = {float(stats.get('hard_supervised_slot_count', 0)) / max(1, len(hard_records)):.2f}")
    lines.append(f"hard_record_supervised_slot_dist = {stats.get('hard_record_supervised_slot_dist', collections.Counter()).most_common()}")
    lines.append("")
    lines.append(f"tier_dist = {stats.get('tier_dist', collections.Counter()).most_common()}")
    lines.append(f"v0_status_dist = {stats.get('v0_status_dist', collections.Counter()).most_common()}")
    lines.append(f"v1_status_dist = {stats.get('v1_status_dist', collections.Counter()).most_common()}")
    lines.append(f"v2_status_dist = {stats.get('v2_status_dist', collections.Counter()).most_common()}")
    lines.append(f"v3_status_dist = {stats.get('v3_status_dist', collections.Counter()).most_common()}")
    lines.append(f"slot_type_dist = {stats.get('slot_type_dist', collections.Counter()).most_common()}")
    lines.append(f"prompt_compact_mode_dist = {stats.get('prompt_compact_mode_dist', collections.Counter()).most_common()}")
    lines.append(f"prompt_skeleton_compacted_dist = {stats.get('prompt_skeleton_compacted_dist', collections.Counter()).most_common()}")
    lines.append(f"sft_seq_len_dist = {stats.get('sft_seq_len_dist', collections.Counter()).most_common()}")
    if stats.get("diagnostic_v3_rejected"):
        lines.append(f"diagnostic_v3_rejected_dist = {stats.get('diagnostic_v3_rejected', collections.Counter()).most_common()}")
    if skips:
        lines.append(f"skip_reason_dist = {stats.get('skip_reason_dist', collections.Counter()).most_common()}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- C3+ rows are hard SFT eligible.")
    lines.append("- For slot-level V3, only C3+ slots are supervised; unmatched B slots are masked out per ECSL unmatched-mass semantics.")
    lines.append("- C2 rows passed static constructability evidence but are not hard CE by default.")
    lines.append("- Diagnostic V3-like sources such as V3-lite/source projection are rejected by default; full trace/symbolic/differential or human-audit evidence is required.")
    lines.append("- If hard_sft_records is zero, do not train yet; inspect candidate-record V3 and/or connect external V3 validation results.")
    lines.append("")
    lines.append("-- first non-hard reasons --")
    shown = 0
    for row in audit_rows:
        if row.get("hard_sft_eligible"):
            continue
        lines.append(
            json.dumps(
                {
                    "example": row.get("example"),
                    "input_line": row.get("input_line"),
                    "func_name": row.get("func_name"),
                    "tier": row.get("confidence_level"),
                    "reasons": (row.get("reasons") or [])[:8],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        shown += 1
        if shown >= 30:
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--module-b-jsonl", nargs="+", required=True)
    ap.add_argument("--source-jsonl", nargs="+", default=None)
    ap.add_argument("--v3-results-jsonl", default=None, help="External V3/differential validation results. Required to produce C3+ hard labels.")
    ap.add_argument("--out-hard-jsonl", required=True)
    ap.add_argument("--out-audit-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--save-tokenizer", default=None)
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-module-b-rows", type=int, default=0)
    ap.add_argument("--max-source-rows", type=int, default=0)
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--min-hard-tier", default="C3", choices=["C3", "C4", "C5"])
    ap.add_argument("--min-supervised-semantic-slots", type=int, default=1)
    ap.add_argument("--mask-supervised-inputs", action="store_true")
    ap.add_argument("--supervised-input-mask-prob", type=float, default=1.0)
    ap.add_argument("--asm-max-total-lines", type=int, default=0)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=0)
    ap.add_argument("--prompt-compact-mode", default="ultra", choices=["auto", "full", "compact", "ultra"])
    ap.add_argument("--max-slot-context-chars", type=int, default=256)
    ap.add_argument("--rewrite-source-prefix", nargs=2, action="append", metavar=("OLD", "NEW"))
    ap.add_argument(
        "--allow-diagnostic-v3-as-hard-label",
        action="store_true",
        help=(
            "Experimental/debug only. Allow diagnostic oracle types such as V3-lite/source projection "
            "to become hard labels. Leave disabled for md-compliant strong semantic data."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    hard_records, audit_rows, skips, stats = build(args)

    hard_path = Path(args.out_hard_jsonl).expanduser().resolve()
    audit_path = Path(args.out_audit_jsonl).expanduser().resolve()
    txt_path = Path(args.out_txt).expanduser().resolve()
    for path in (hard_path, audit_path, txt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with hard_path.open("w", encoding="utf-8") as f:
        for rec in hard_records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    with audit_path.open("w", encoding="utf-8") as f:
        for rec in audit_rows:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(txt_path, args=args, hard_records=hard_records, audit_rows=audit_rows, skips=skips, stats=stats)

    print(f"rows_seen = {stats.get('rows_seen', 0)}")
    print(f"audit_rows = {len(audit_rows)}")
    print(f"hard_sft_records = {len(hard_records)}")
    print(f"records_skipped_before_audit = {len(skips)}")
    print(f"tier_dist = {stats.get('tier_dist', collections.Counter()).most_common()}")
    print(f"v0_status_dist = {stats.get('v0_status_dist', collections.Counter()).most_common()}")
    print(f"v1_status_dist = {stats.get('v1_status_dist', collections.Counter()).most_common()}")
    print(f"v2_status_dist = {stats.get('v2_status_dist', collections.Counter()).most_common()}")
    print(f"v3_status_dist = {stats.get('v3_status_dist', collections.Counter()).most_common()}")
    print(f"skip_reason_dist = {stats.get('skip_reason_dist', collections.Counter()).most_common()}")
    print(f"saved_hard_jsonl = {hard_path}")
    print(f"saved_audit_jsonl = {audit_path}")
    print(f"saved_report = {txt_path}")


if __name__ == "__main__":
    main()

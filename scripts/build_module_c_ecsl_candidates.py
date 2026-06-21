#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build ECSL candidate tables, validator reports, and training records.

Input is the contract JSONL produced by build_module_c_ecsl_slot_contracts.py.
This script implements the next practical ECSL layers from the design doc:

* SCTS candidate construction with provenance;
* lightweight semantic matching scores;
* V0 slot grammar validation;
* V1 deterministic skeleton preservation validation;
* V2 slot-local SSA/control/effect compatibility precheck;
* V3 external symbolic/trace/differential validation loop;
* confidence scoring into C0..C5 tiers;
* multi-reference SFT records and global decoding seed records.

V2 is a compatibility gate, not a semantic proof.  It rejects explicit
contradictions and unsafe placeholders, but evidence that is simply absent is
reported as a needs-v3 warning so trace/symbolic/differential V3 can decide.
The optional compile-smoke harness records compile evidence only; it is never
promoted to semantic V3/C3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CANDIDATE_SCHEMA = "ModuleC.ECSL.SlotCandidateSet.v0"
TRAIN_SCHEMA = "ModuleC.ECSL.MultiReferenceSFT.v0"
DECODE_SCHEMA = "ModuleC.ECSL.GlobalDecodeSeed.v0"

TIER_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
TIER_WEIGHT = {"C0": 0.0, "C1": 0.0, "C2": 0.2, "C3": 0.5, "C4": 0.8, "C5": 1.0}
PLACEHOLDER_RE = re.compile(r"<(?P<tag>[A-Z_]+):(?P<slot_ref>slot_(?P<sid>\d+))>")
CONTROL_KEYWORDS = ("return", "break", "continue", "goto")
C_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Bool",
    "bool",
    "true",
    "false",
    "NULL",
}
EXTERNAL_KEEP_SOURCES = {"source_cfg", "execution_trace", "llm_proposal", "self_training"}
DOC_SLOT_TYPES = {
    "COND",
    "LOOP_EXIT",
    "SWITCH_DISPATCH",
    "CASE_LABEL",
    "EXPR",
    "LVALUE",
    "RETURN",
    "CALL_ARG",
    "CALL_TARGET",
    "MEMORY_EFFECT",
    "PHI_OR_MERGE",
    "STMT",
    "BLOCK",
    "EXIT",
}
DIAGNOSTIC_V3_ORACLE_TYPES = {
    "v3_differential_lite_candidate_reference",
    "training_source_teacher_slot_projection",
    "training_source_teacher_strict_render",
    "source_teacher_strict_projection",
}


REWRITE_KINDS = {
    "early_return_if_else": {
        "summary": "early return <-> if-else candidate request",
        "requires": ["return_value_recovery", "exit_successor_validation", "live_out_validation"],
    },
    "while_loop_break": {
        "summary": "while <-> loop plus break candidate request",
        "requires": ["loop_header_latch_validation", "break_target_validation", "loop_carried_liveout_validation"],
    },
    "switch_if_chain": {
        "summary": "switch <-> if-chain candidate request",
        "requires": ["dispatch_expression_recovery", "case_value_type_validation", "fallthrough_validation"],
    },
    "branch_inversion": {
        "summary": "branch inversion / then-else swap candidate request",
        "requires": ["condition_side_effect_free", "successor_polarity_validation"],
    },
    "nested_if_conjunction": {
        "summary": "nested if <-> conjunctive condition candidate request",
        "requires": ["short_circuit_semantics_validation", "false_successor_validation"],
    },
    "continue_guarded_body": {
        "summary": "break/continue restructuring <-> guarded body candidate request",
        "requires": ["continue_target_validation", "body_effect_validation"],
    },
}

MATCHING_COMPONENT_NAMES = [
    "type_mismatch",
    "live_in_mismatch",
    "live_out_mismatch",
    "read_write_mismatch",
    "effect_mismatch",
    "cfg_successor_mismatch",
    "dominance_postdominance_mismatch",
    "pdg_dependency_mismatch",
    "semantic_summary_distance",
    "branch_polarity_mismatch",
    "rewrite_cost",
    "rewrite_compatibility_bonus",
    "verifier_confidence_bonus",
    "bb_overlap_bonus",
]

HARD_CONSTRAINT_NAMES = [
    "entry_exit_successor_consistency",
    "postdom_boundary_consistency",
    "scc_loop_membership_consistency",
    "live_out_compatibility",
    "call_memory_effect_compatibility",
    "branch_polarity_compatibility",
    "type_compatibility",
    "no_undefined_behavior_introduced",
    "skeleton_immutability",
    "slot_grammar_compatibility",
]

C_LOCAL_STUB_PREFIX = "\n".join(
    [
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "extern long long ECSL_ext(void);",
        "",
    ]
)


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def q(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_slot_type(value: Any) -> str:
    raw = str(value or "").strip().upper()
    aliases = {
        "FILL_COND": "COND",
        "COND_SLOT": "COND",
        "CONDITION": "COND",
        "SWITCH_DISPATCH": "COND",
        "FILL_STMT": "STMT",
        "STMT": "STMT",
        "STATEMENT": "STMT",
        "FILL_BLOCK": "BLOCK",
        "BLOCK_SLOT": "BLOCK",
        "FILL_CASE": "CASE_LABEL",
        "CASE_SLOT": "CASE_LABEL",
        "CASE": "CASE_LABEL",
        "FILL_GOTO": "EXIT",
        "GOTO_SLOT": "EXIT",
        "GOTO": "EXIT",
        "FILL_BREAK": "LOOP_EXIT",
        "BREAK_SLOT": "LOOP_EXIT",
        "BREAK": "LOOP_EXIT",
        "FILL_CONTINUE": "LOOP_EXIT",
        "CONTINUE_SLOT": "LOOP_EXIT",
        "CONTINUE": "LOOP_EXIT",
        "LOOP_CONTINUE": "LOOP_EXIT",
        "FILL_EARLY_EXIT": "RETURN",
        "EARLY_EXIT_SLOT": "RETURN",
        "EARLY_EXIT": "RETURN",
        "FILL_CLEANUP_EXIT": "EXIT",
        "CLEANUP_EXIT_SLOT": "EXIT",
        "CLEANUP_EXIT": "EXIT",
        "UNSTRUCTURED_REGION": "BLOCK",
    }
    return aliases.get(raw, raw)


def module_b_slot_kind(slot: Dict[str, Any]) -> str:
    has_explicit_kind = bool(slot.get("module_b_slot_kind") or q(slot, "provenance", "module_b_slot_kind"))
    raw = str((slot.get("module_b_slot_kind") or q(slot, "provenance", "module_b_slot_kind") or slot.get("slot_type") or "")).strip().upper()
    aliases = {
        "FILL_COND": "COND",
        "COND_SLOT": "COND",
        "FILL_STMT": "STMT",
        "STATEMENT": "STMT",
        "FILL_BLOCK": "BLOCK",
        "BLOCK_SLOT": "BLOCK",
        "FILL_CASE": "CASE",
        "CASE_SLOT": "CASE",
        "FILL_GOTO": "GOTO",
        "GOTO_SLOT": "GOTO",
        "FILL_BREAK": "BREAK",
        "BREAK_SLOT": "BREAK",
        "FILL_CONTINUE": "CONTINUE",
        "CONTINUE_SLOT": "CONTINUE",
        "FILL_EARLY_EXIT": "EARLY_EXIT",
        "EARLY_EXIT_SLOT": "EARLY_EXIT",
        "FILL_CLEANUP_EXIT": "CLEANUP_EXIT",
        "CLEANUP_EXIT_SLOT": "CLEANUP_EXIT",
    }
    if has_explicit_kind:
        return aliases.get(raw, raw)
    if raw in DOC_SLOT_TYPES:
        return "UNKNOWN_DOC_SLOT_WITHOUT_MODULE_B_KIND"
    return aliases.get(raw, raw)


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        return value
    return None


def condition_ssa_from_slot(slot: Dict[str, Any]) -> Any:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    observable = slot.get("observable_semantics") if isinstance(slot.get("observable_semantics"), dict) else {}
    anchor = slot.get("anchor") if isinstance(slot.get("anchor"), dict) else {}
    return first_nonempty(
        q(data, "ssa_grounding", "condition_ssa"),
        q(data, "condition_ssa"),
        q(data, "branch_condition_ssa"),
        q(data, "predicate_ssa"),
        q(observable, "condition_ssa"),
        q(observable, "branch_condition_ssa"),
        q(anchor, "condition_ssa"),
        slot.get("condition_ssa"),
        slot.get("branch_condition_ssa"),
    )


def symbols_from_contract_value(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        if value.strip():
            out.append(value.strip())
        return out
    if isinstance(value, (int, float)):
        out.append(str(value))
        return out
    if isinstance(value, dict):
        for key in (
            "ssa",
            "ssa_name",
            "name",
            "var",
            "variable",
            "symbol",
            "canonical",
            "canonical_symbol",
            "expr",
            "value",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                out.extend(symbols_from_contract_value(item))
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(symbols_from_contract_value(item))
    return out


def condition_symbol_aliases(slot: Dict[str, Any]) -> List[str]:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    aliases: List[str] = []
    condition_ssa = condition_ssa_from_slot(slot)
    aliases.extend(symbols_from_contract_value(condition_ssa))
    aliases.extend(symbols_from_contract_value(q(data, "ssa_grounding")))
    aliases.extend(symbols_from_contract_value(data.get("canonical_symbols")))
    aliases.extend(symbols_from_contract_value(data.get("type_hints")))
    aliases.extend(symbols_from_contract_value(data.get("live_in")))
    aliases.extend(symbols_from_contract_value(data.get("uses")))
    sanitized: List[str] = []
    seen = set()
    for item in aliases:
        base_variants = [item, sanitize_identifier(item, prefix="ssa")]
        item_text = str(item or "").strip()
        for sep in ("@", ".", "#"):
            if sep in item_text:
                head = item_text.split(sep, 1)[0].lstrip("%")
                if head:
                    base_variants.extend([head, sanitize_identifier(head, prefix="ssa")])
        if item_text.startswith("%"):
            base_variants.append(item_text.lstrip("%"))
        for value in base_variants:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                sanitized.append(text)
    return sanitized


def symbol_from_fact(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in (
            "ssa",
            "ssa_name",
            "name",
            "var",
            "variable",
            "symbol",
            "id",
            "expr",
            "value",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        for key in ("ref", "target", "source"):
            item = value.get(key)
            if isinstance(item, dict):
                got = symbol_from_fact(item)
                if got:
                    return got
    return None


def first_symbol_from_list(value: Any) -> Optional[str]:
    if not isinstance(value, list):
        return symbol_from_fact(value)
    for item in value:
        got = symbol_from_fact(item)
        if got:
            return got
    return None


def return_ssa_from_slot(slot: Dict[str, Any]) -> Any:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    control = slot.get("control_contract") if isinstance(slot.get("control_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    observable = slot.get("observable_semantics") if isinstance(slot.get("observable_semantics"), dict) else {}
    return first_nonempty(
        q(data, "ssa_grounding", "return_ssa"),
        q(data, "return_ssa"),
        q(data, "return_value_ssa"),
        q(control, "return_ssa"),
        q(control, "return_value_ssa"),
        q(effect, "return_ssa"),
        q(observable, "return_ssa"),
        q(observable, "return_value_ssa"),
        slot.get("return_ssa"),
        slot.get("return_value_ssa"),
    )


def return_constant_from_slot(slot: Dict[str, Any]) -> Any:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    control = slot.get("control_contract") if isinstance(slot.get("control_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    observable = slot.get("observable_semantics") if isinstance(slot.get("observable_semantics"), dict) else {}
    return first_nonempty(
        q(data, "ssa_grounding", "return_constant"),
        q(data, "ssa_grounding", "return_value_constant"),
        q(data, "return_constant"),
        q(data, "return_value_constant"),
        q(control, "return_constant"),
        q(control, "return_value_constant"),
        q(effect, "return_constant"),
        q(observable, "return_constant"),
        q(observable, "return_value_constant"),
        slot.get("return_constant"),
        slot.get("return_value_constant"),
    )


def simple_assignment_from_slot(slot: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    defs = data.get("defs") or data.get("defined") or data.get("writes") or effect.get("write_set")
    uses = data.get("uses") or data.get("used") or data.get("reads") or effect.get("read_set")
    lhs = first_symbol_from_list(defs)
    rhs = first_symbol_from_list(uses)
    if lhs and rhs:
        return sanitize_identifier(lhs, prefix="lhs"), sanitize_identifier(rhs, prefix="rhs")
    return None


def strict_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return False


def compact_counter(counter: Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def stable_id(*parts: Any, n: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def canonical_slot_key(value: Any) -> str:
    s = str(value or "").strip()
    m = re.search(r"slot_(\d+)", s)
    if m:
        return f"slot_{int(m.group(1)):03d}"
    if re.fullmatch(r"\d+", s):
        return f"slot_{int(s):03d}"
    return s


def unavailable_component(reason: str) -> Dict[str, Any]:
    return {"value": None, "status": "not_available", "reason": reason}


def static_component(value: float, note: str) -> Dict[str, Any]:
    return {"value": round(float(value), 6), "status": "static_o1", "note": note}


def semantic_matching_components(
    slot: Dict[str, Any],
    *,
    source: str,
    base_alignment_score: float,
    rewrite_kind: Optional[str],
    generated_text_available: bool,
) -> Dict[str, Any]:
    raw_slot_type = str(slot.get("slot_type") or "").upper()
    slot_type = normalize_slot_type(raw_slot_type)
    region_bbs = list(q(slot, "region", "region_bbs") or [])
    semantic_distance = max(0.0, min(1.0, 1.0 - float(base_alignment_score)))
    rewrite_cost = 0.12 if rewrite_kind else 0.0
    bb_bonus = min(0.10, 0.02 * len(region_bbs)) if region_bbs else 0.0
    unavailable_reason = "requires source/binary SSA, PDG, effect model, or execution traces"

    if not generated_text_available:
        return {name: unavailable_component("KEEP_EXTERNAL candidate has no generated text yet") for name in MATCHING_COMPONENT_NAMES}

    return {
        "type_mismatch": static_component(0.0, "slot grammar category is locally known; exact C type remains V0 parser/type work"),
        "live_in_mismatch": unavailable_component(unavailable_reason),
        "live_out_mismatch": unavailable_component(unavailable_reason),
        "read_write_mismatch": unavailable_component(unavailable_reason),
        "effect_mismatch": unavailable_component(unavailable_reason),
        "cfg_successor_mismatch": unavailable_component("V1 skeleton rendering checks placeholders; successor polarity requires CFG validator"),
        "dominance_postdominance_mismatch": unavailable_component("requires dominator/postdominator tree on unified IR"),
        "pdg_dependency_mismatch": unavailable_component("requires PDG extraction"),
        "semantic_summary_distance": static_component(semantic_distance, "local O1 semantic summary prior from slot type, region, and candidate source"),
        "branch_polarity_mismatch": unavailable_component("requires true/false successor polarity validation") if slot_type == "COND" else static_component(0.0, "not a condition slot"),
        "rewrite_cost": static_component(rewrite_cost, "explicit rewrite templates carry a small structural cost"),
        "rewrite_compatibility_bonus": static_component(0.05 if rewrite_kind else 0.0, "candidate was emitted by a rewrite generator compatible with this slot"),
        "verifier_confidence_bonus": unavailable_component("filled after V2/V3 verifier upgrades the candidate"),
        "bb_overlap_bonus": static_component(bb_bonus, "candidate references the Module-B region BB set"),
    }


def hard_constraint_report(slot: Dict[str, Any], *, generated_text_available: bool) -> Dict[str, Any]:
    if not generated_text_available:
        statuses = {
            name: {
                "status": "not_available",
                "passed": False,
                "reason": "KEEP_EXTERNAL candidate has no generated text to validate",
            }
            for name in HARD_CONSTRAINT_NAMES
        }
        return {
            "overall_static_precheck_pass": False,
            "all_constraints_proven": False,
            "items": statuses,
        }

    raw_slot_type = str(slot.get("slot_type") or "").upper()
    slot_type = normalize_slot_type(raw_slot_type)
    items = {
        "entry_exit_successor_consistency": {
            "status": "pending_v1_cfg_validator",
            "passed": None,
            "reason": "requires rendered skeleton CFG successor check",
        },
        "postdom_boundary_consistency": {
            "status": "KEEP_EXTERNAL",
            "passed": None,
            "reason": "postdominator boundary is not extracted in this offline builder",
        },
        "scc_loop_membership_consistency": {
            "status": "KEEP_EXTERNAL",
            "passed": None,
            "reason": "loop/SCC membership proof requires CFG analysis beyond slot text",
        },
        "live_out_compatibility": {
            "status": "KEEP_EXTERNAL",
            "passed": None,
            "reason": "requires SSA live-out extraction and candidate effect summary",
        },
        "call_memory_effect_compatibility": {
            "status": "KEEP_EXTERNAL",
            "passed": None,
            "reason": "requires memory/call effect model",
        },
        "branch_polarity_compatibility": {
            "status": "pending_v1_cfg_validator" if slot_type == "COND" else "not_applicable",
            "passed": None if slot_type == "COND" else True,
            "reason": "condition polarity must be checked against true/false successors" if slot_type == "COND" else "not a condition slot",
        },
        "type_compatibility": {
            "status": "pending_v0_parser_type_checker",
            "passed": None,
            "reason": "local grammar precheck is not a full C type checker",
        },
        "no_undefined_behavior_introduced": {
            "status": "KEEP_EXTERNAL",
            "passed": None,
            "reason": "requires semantic validator or compiler/sanitizer evidence",
        },
        "skeleton_immutability": {
            "status": "pending_v1_renderer",
            "passed": None,
            "reason": "validated after deterministic rendering into the fixed Module-B skeleton",
        },
        "slot_grammar_compatibility": {
            "status": "pending_v0_grammar_checker",
            "passed": None,
            "reason": "validated by grammar_check during candidate validation",
        },
    }
    return {
        "overall_static_precheck_pass": True,
        "all_constraints_proven": False,
        "items": items,
    }


def semantic_matching_report(
    slot: Dict[str, Any],
    *,
    source: str,
    base_alignment_score: float,
    rewrite_kind: Optional[str],
    generated_text_available: bool,
) -> Dict[str, Any]:
    components = semantic_matching_components(
        slot,
        source=source,
        base_alignment_score=base_alignment_score,
        rewrite_kind=rewrite_kind,
        generated_text_available=generated_text_available,
    )
    static_values = [
        safe_float(item.get("value"))
        for item in components.values()
        if isinstance(item, dict) and item.get("value") is not None
    ]
    static_cost = round(sum(static_values), 6) if static_values else None
    hard = hard_constraint_report(slot, generated_text_available=generated_text_available)
    return {
        "scheme": "ECSL.semantic_matching_uot_lite.v0",
        "base_alignment_score": round(float(base_alignment_score), 6),
        "cost": None if not generated_text_available else round(max(0.0, 1.0 - float(base_alignment_score)), 6),
        "static_cost_sum": static_cost,
        "cost_components": components,
        "hard_constraints": hard,
        "hard_constraints_pass": bool(hard.get("overall_static_precheck_pass")),
        "hard_constraints_proven": bool(hard.get("all_constraints_proven")),
        "hard_constraint_notes": [
            "O1 builds an auditable matching object; V2/V3 or external tools must prove semantic constraints.",
        ],
        "uot": {
            "status": "KEEP_EXTERNAL_uot_solver_not_connected",
            "unit": "semantic_unit_to_skeleton_slot",
            "unmatched_source_unit_policy": "may become weak candidate or be dropped",
            "unmatched_b_slot_policy": "prior only until generated/validated",
            "mass_regularization": "not_configured_in_offline_builder",
        },
        "source": source,
        "rewrite_kind": rewrite_kind,
    }


def sanitize_identifier(value: Any, *, prefix: str = "ssa") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    if text[0].isdigit():
        text = f"{prefix}_{text}"
    if not re.match(r"^[A-Za-z_]", text):
        text = f"{prefix}_{text}"
    return text


def balanced(text: str, pairs: Sequence[Tuple[str, str]]) -> bool:
    stack: List[str] = []
    opens = {a: b for a, b in pairs}
    closes = {b: a for a, b in pairs}
    for ch in text:
        if ch in opens:
            stack.append(ch)
        elif ch in closes:
            if not stack or stack[-1] != closes[ch]:
                return False
            stack.pop()
    return not stack


def contains_forbidden_control(text: str, forbidden: Sequence[str]) -> List[str]:
    found: List[str] = []
    for kw in forbidden:
        if re.search(rf"\b{re.escape(str(kw))}\b", text):
            found.append(str(kw))
    return found


def identifiers_in_text(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", str(text or "")):
        ident = match.group(0)
        if ident in C_KEYWORDS:
            continue
        if ident.startswith("label_") or ident in {"cleanup", "ECSL_ext"}:
            continue
        if ident not in seen:
            seen.add(ident)
            out.append(ident)
    return out


def function_like_identifiers(text: str) -> List[str]:
    funcs: List[str] = []
    seen = set()
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(text or "")):
        ident = match.group(1)
        if ident in C_KEYWORDS or ident in seen:
            continue
        seen.add(ident)
        funcs.append(ident)
    return funcs


def label_identifiers(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for match in re.finditer(r"\blabel_[A-Za-z0-9_]+\b", str(text or "")):
        label = match.group(0)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def local_stub_declarations(text: str) -> str:
    funcs = set(function_like_identifiers(text))
    builtin_locals = {"a", "b", "c", "x", "y", "z"}
    vars_ = [x for x in identifiers_in_text(text) if x not in funcs and x not in builtin_locals]
    lines = [f"extern long long {name}();" for name in sorted(funcs)]
    lines.extend(f"    long long {name} = 0;" for name in vars_)
    return "\n".join(lines)


def local_stub_source(text: str, category: str) -> str:
    fill = str(text or "").strip()
    decls = local_stub_declarations(fill)
    extern_decls = "\n".join(line for line in decls.splitlines() if line.startswith("extern "))
    local_decls = "\n".join(line for line in decls.splitlines() if not line.startswith("extern "))
    labels = label_identifiers(fill)
    built_in_labels = {"label_BB_0", "label_BB_1", "label_UNKNOWN"}
    label_defs = "\n".join(f"{label}: ;" for label in labels if label not in built_in_labels)
    if category == "condition_only":
        body = "\n".join(
            [
                "int ecsl_stub(long long a, long long b, long long c) {",
                local_decls,
                f"    return !!({fill});",
                "}",
            ]
        )
    elif category == "statement_list":
        stmt = normalize_statement_text(fill, category)
        body = "\n".join(
            [
                "long long ecsl_stub(long long a, long long b, long long c) {",
                "    long long x = a, y = b, z = c;",
                local_decls,
                "    while (1) {",
                "        switch ((int)a) {",
                "        case 0:",
                "label_BB_0:",
                "label_BB_1:",
                label_defs,
                "cleanup:",
                f"            {stmt}",
                "            break;",
                "        default:",
                "            break;",
                "        }",
                "        break;",
                "    }",
                "    return x + y + z;",
                "}",
            ]
        )
    elif category == "control_transfer":
        stmt = normalize_statement_text(fill, category)
        body = "\n".join(
            [
                "long long ecsl_stub(long long a) {",
                local_decls,
                "    while (a) {",
                "        if (a < 0) {",
                f"            {stmt}",
                "        }",
                "        break;",
                "    }",
                "label_BB_0:",
                "label_BB_1:",
                "label_UNKNOWN:",
                label_defs,
                "cleanup:",
                "    return a;",
                "}",
            ]
        )
    elif category == "return_expr":
        stmt = fill
        if not stmt.startswith("return"):
            stmt = "return " + stmt.rstrip(";") + ";"
        body = "\n".join(
            [
                "long long ecsl_stub(long long a, long long b, long long c) {",
                local_decls,
                f"    {stmt}",
                "}",
            ]
        )
    else:
        body = "\n".join(["long long ecsl_stub(void) {", local_decls, f"    {fill};", "    return 0;", "}"])
    return C_LOCAL_STUB_PREFIX + (extern_decls + "\n" if extern_decls else "") + body + "\n"


def local_c_stub_check(text: str, category: str, *, cc: Optional[str]) -> Dict[str, Any]:
    if not cc:
        return {
            "status": "not_available",
            "passed": None,
            "reason": "no_c_compiler_configured",
        }
    resolved = shutil.which(str(cc))
    if not resolved:
        return {
            "status": "not_available",
            "passed": None,
            "reason": f"compiler_not_found={cc}",
        }
    source = local_stub_source(text, category)
    with tempfile.TemporaryDirectory(prefix="ecsl_v0_stub_") as tmp:
        src = Path(tmp) / "candidate_stub.c"
        src.write_text(source, encoding="utf-8")
        cmd = [resolved, "-x", "c", "-std=gnu11", "-fsyntax-only", str(src)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
        except Exception as e:
            return {
                "status": "fail",
                "passed": False,
                "reason": f"local_stub_exception:{e}",
                "cmd": cmd,
            }
    return {
        "status": "pass" if proc.returncode == 0 else "fail",
        "passed": proc.returncode == 0,
        "reason": "local_c_stub_parse_type_pass" if proc.returncode == 0 else "local_c_stub_parse_type_fail",
        "cmd": cmd,
        "returncode": proc.returncode,
        "output_head": (proc.stdout or "")[:2000],
    }


def grammar_check(text: str, grammar_contract: Dict[str, Any]) -> Dict[str, Any]:
    category = str(grammar_contract.get("category") or "unknown")
    forbidden = list(grammar_contract.get("forbidden_control_transfer") or [])
    stripped = str(text or "").strip()
    errors: List[str] = []

    if not stripped:
        errors.append("empty_candidate_text")
    if PLACEHOLDER_RE.search(stripped):
        errors.append("candidate_contains_skeleton_placeholder")
    if not balanced(stripped, [("(", ")"), ("[", "]")]):
        errors.append("unbalanced_parentheses_or_brackets")
    forbidden_hits = contains_forbidden_control(stripped, forbidden)
    if forbidden_hits:
        errors.append("forbidden_control_transfer=" + ",".join(forbidden_hits))

    if category == "condition_only":
        if ";" in stripped:
            errors.append("condition_contains_semicolon")
        if "{" in stripped or "}" in stripped:
            errors.append("condition_contains_brace")
        if contains_forbidden_control(stripped, CONTROL_KEYWORDS):
            errors.append("condition_contains_control_keyword")
    elif category == "statement_list":
        if not balanced(stripped, [("{", "}")]):
            errors.append("unbalanced_braces")
    elif category == "control_transfer":
        allowed = (
            stripped.startswith("break")
            or stripped.startswith("continue")
            or stripped.startswith("return")
            or stripped.startswith("goto")
            or stripped.startswith("/*")
        )
        if not allowed:
            errors.append("control_transfer_candidate_not_template_like")
    elif category == "unsupported_region":
        errors.append("unsupported_region_not_trainable")
    elif category == "unknown":
        errors.append("unknown_grammar_category")

    return {
        "level": "V0",
        "name": "parser_type_slot_grammar",
        "status": "pass" if not errors else "fail",
        "passed": not errors,
        "errors": errors,
        "category": category,
        "note": "Heuristic local grammar check; real parser/type checking belongs to the next validator stage.",
    }


def v0_parser_type_check(text: str, grammar_contract: Dict[str, Any], *, cc: Optional[str]) -> Dict[str, Any]:
    grammar = grammar_check(text, grammar_contract)
    category = str(grammar.get("category") or "unknown")
    stub = {"status": "not_attempted", "passed": None, "reason": "grammar_failed"}
    errors = list(grammar.get("errors") or [])
    if grammar.get("passed"):
        stub = local_c_stub_check(text, category, cc=cc)
        if stub.get("passed") is False:
            errors.append(str(stub.get("reason") or "local_c_stub_failed"))
    passed = bool(grammar.get("passed")) and stub.get("passed") is not False
    status = "pass" if passed else "fail"
    if grammar.get("passed") and stub.get("status") == "not_available":
        status = "pass"
    return {
        "level": "V0",
        "name": "parser_type_local_stub_validation",
        "status": status,
        "passed": passed,
        "errors": errors,
        "category": category,
        "grammar": grammar,
        "local_stub": stub,
        "note": "V0 enforces slot grammar and, when a compiler is available, parser/type syntax in a local C stub.",
    }


def normalize_statement_text(text: str, category: str) -> str:
    out = str(text or "").strip()
    if category in {"statement_list", "control_transfer"}:
        if out and not out.endswith((";", "}", "*/")):
            out += ";"
        if out.endswith("*/"):
            out += ";"
    return out


def slot_placeholder_pattern(slot_id: str) -> re.Pattern[str]:
    return re.compile(rf"<(?P<tag>[A-Z_]+):{re.escape(slot_id)}>(?P<trailing>\s*;)?")


def render_slot_candidate(skeleton: str, slot_id: str, text: str, category: str) -> Dict[str, Any]:
    if not skeleton:
        return {
            "rendered": None,
            "status": "not_available",
            "passed": False,
            "errors": ["missing_skeleton_text"],
        }
    pattern = slot_placeholder_pattern(slot_id)
    matches = list(pattern.finditer(skeleton))
    if len(matches) != 1:
        return {
            "rendered": None,
            "status": "fail",
            "passed": False,
            "errors": [f"placeholder_match_count={len(matches)}"],
        }
    replacement = str(text or "").strip()
    if category in {"statement_list", "control_transfer"}:
        replacement = normalize_statement_text(replacement, category)
    # Use a callable replacement so C/C++ escapes such as "\xNN" in candidate
    # text are inserted literally instead of being parsed as re.sub templates.
    rendered = pattern.sub(lambda _m: replacement, skeleton, count=1)

    before_slots = [m.group("slot_ref") for m in PLACEHOLDER_RE.finditer(skeleton)]
    after_slots = [m.group("slot_ref") for m in PLACEHOLDER_RE.finditer(rendered)]
    expected_after = list(before_slots)
    try:
        expected_after.remove(slot_id)
    except ValueError:
        pass
    errors: List[str] = []
    if after_slots != expected_after:
        errors.append("non_target_placeholder_set_changed")
    if PLACEHOLDER_RE.search(replacement):
        errors.append("replacement_contains_placeholder")

    return {
        "rendered": rendered,
        "status": "pass" if not errors else "fail",
        "passed": not errors,
        "errors": errors,
        "removed_slot": slot_id,
        "remaining_placeholder_count": len(after_slots),
        "note": "V1 checks that deterministic rendering changes only the target placeholder.",
    }


def skeleton_preservation_check(
    skeleton: str,
    slot: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:
    slot_id = str(slot.get("slot_id") or "")
    category = str(q(slot, "grammar_contract", "category") or "unknown")
    rendered = render_slot_candidate(skeleton, slot_id, text, category)
    slot_type = normalize_slot_type(slot.get("slot_type"))
    control = slot.get("control_contract") if isinstance(slot.get("control_contract"), dict) else {}
    polarity = slot.get("polarity") if isinstance(slot.get("polarity"), dict) else {}
    successor_warnings: List[str] = []
    successor_errors: List[str] = []
    successor_evidence = {
        "slot_type": slot_type,
        "true_successor": control.get("true_successor"),
        "false_successor": control.get("false_successor"),
        "fallthrough_successor": control.get("fallthrough_successor"),
        "break_target": control.get("break_target"),
        "continue_target": control.get("continue_target"),
        "polarity_status": polarity.get("status"),
        "true_false_polarity_available": polarity.get("true_false_polarity_available"),
    }
    if slot_type in {"COND", "SWITCH_DISPATCH"}:
        if not control.get("true_successor") or not control.get("false_successor"):
            successor_warnings.append("v1_missing_true_false_successor_metadata")
    op_kind = module_b_slot_kind(slot)
    stripped = str(text or "").strip()
    if op_kind == "BREAK" and not control.get("break_target"):
        successor_warnings.append("v1_break_target_metadata_unavailable")
    if op_kind == "CONTINUE" and not control.get("continue_target"):
        successor_warnings.append("v1_continue_target_metadata_unavailable")
    if op_kind == "GOTO":
        edge = control.get("edge")
        target = bb_name(edge[1]) if isinstance(edge, list) and len(edge) >= 2 else None
        if not target:
            successor_warnings.append("v1_goto_target_metadata_unavailable")
        elif f"label_{target}" not in stripped and f"label_BB_{target}" not in stripped:
            successor_errors.append(f"v1_goto_target_label_mismatch_expected={target}")
    errors = list(rendered.get("errors") or []) + successor_errors
    status = "pass" if not errors and not successor_warnings else ("partial_needs_v2_v3" if not errors else "fail")
    return {
        "level": "V1",
        "name": "skeleton_preservation_cfg_successor",
        "status": status,
        "passed": bool(rendered.get("passed")) and not errors,
        "errors": errors,
        "warnings": successor_warnings,
        "successor_evidence": successor_evidence,
        "remaining_placeholder_count": rendered.get("remaining_placeholder_count"),
        "rendered_available": rendered.get("rendered") is not None,
        "note": "V1 preserves the fixed skeleton and records successor/polarity metadata required by ECSL.",
    }


def unavailable_level(level: str, name: str, reason: str) -> Dict[str, Any]:
    return {
        "level": level,
        "name": name,
        "status": "not_available",
        "passed": False,
        "errors": [],
        "reason": reason,
    }


def validation_level(
    level: str,
    name: str,
    *,
    passed: bool,
    status: str,
    errors: Optional[Sequence[str]] = None,
    reason: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "level": level,
        "name": name,
        "status": status,
        "passed": bool(passed),
        "errors": list(errors or []),
    }
    if reason:
        out["reason"] = reason
    if evidence is not None:
        out["evidence"] = evidence
    if note:
        out["note"] = note
    return out


def candidate_text(cand: Dict[str, Any]) -> str:
    for item in cand.get("c_renderings") or []:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            return str(item.get("text") or "").strip()
    return ""


def bb_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        m = re.search(r"BB_(\d+)", value)
        if m:
            return f"BB_{int(m.group(1))}"
        try:
            return f"BB_{int(value)}"
        except Exception:
            return value if value.startswith("BB_") else None
    try:
        return f"BB_{int(value)}"
    except Exception:
        return None


def nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def contract_status_available(contract: Dict[str, Any]) -> bool:
    status = str(contract.get("status") or "").lower()
    return bool(status) and "not_available" not in status and "unknown" not in status


def contract_has_known_empty_real_facts(contract: Dict[str, Any]) -> bool:
    fact_state = str(contract.get("fact_state") or "").lower()
    status = str(contract.get("status") or "").lower()
    return fact_state == "proven_empty_real" or status == "available_empty_from_real_function_ir"


def slot_contracts_prove_empty_observable_effect(slot: Dict[str, Any]) -> bool:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    return bool(contract_has_known_empty_real_facts(data) and contract_has_known_empty_real_facts(effect))


def slot_can_probe_empty_observable_effect(slot: Dict[str, Any]) -> bool:
    """Allow an empty-statement V3 probe when contracts do not disprove it.

    This expands candidate coverage without promoting the candidate to C2/C3:
    V2 marks it partial unless both data and effect contracts prove real-empty.
    """
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    if contract_has_known_empty_real_facts(data) and contract_has_known_empty_real_facts(effect):
        return True
    if any(nonempty_list(data.get(key)) for key in ("live_out", "defs")):
        return False
    if any(nonempty_list(effect.get(key)) for key in ("write_set", "call_effects", "volatile_ops")):
        return False
    if effect.get("may_throw_or_longjmp"):
        return False
    return True


def compact_probe_payload(values: Sequence[Any], *, limit: int = 6) -> str:
    parts: List[str] = []
    for value in values:
        text = sanitize_identifier(value, prefix="sym")
        if text and text not in parts:
            parts.append(text)
        if len(parts) >= limit:
            break
    return "_".join(parts) if parts else "unknown"


def block_semantic_probe_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    out: List[Dict[str, Any]] = []
    live_values = list(data.get("live_out") or data.get("defs") or [])
    if live_values:
        payload = compact_probe_payload(live_values)
        out.append(
            candidate_template(
                slot,
                text=f'ECSL_dataflow_probe("{payload}");',
                source="binary_ir",
                semantic_kind="statement_transformer",
                summary="dataflow V3 probe for non-empty block live-out contract",
                requires=["v3_dataflow_statement_synthesis_or_full_function_validation"],
                base_alignment_score=0.14,
            )
        )
    effect_values = list(effect.get("write_set") or []) + list(effect.get("call_effects") or []) + list(effect.get("volatile_ops") or [])
    if effect_values:
        payload = compact_probe_payload(effect_values)
        out.append(
            candidate_template(
                slot,
                text=f'ECSL_effect_probe("{payload}");',
                source="binary_ir",
                semantic_kind="statement_transformer",
                summary="effect V3 probe for non-empty block write/call/volatile contract",
                requires=["v3_effect_statement_synthesis_or_full_function_validation"],
                base_alignment_score=0.14,
            )
        )
    return out


def contract_has_proxy_empty_facts(contract: Dict[str, Any]) -> bool:
    fact_state = str(contract.get("fact_state") or "").lower()
    status = str(contract.get("status") or "").lower()
    return fact_state == "empty_proxy_needs_v3" or status == "partial_empty_from_proxy_function_ir"


def contract_has_proxy_facts(contract: Dict[str, Any]) -> bool:
    fact_state = str(contract.get("fact_state") or "").lower()
    status = str(contract.get("status") or "").lower()
    return "proxy" in fact_state or status.startswith("partial_") or "proxy" in status


def contract_has_real_or_partial_facts(contract: Dict[str, Any]) -> bool:
    if contract_has_known_empty_real_facts(contract):
        return True
    return contract_status_available(contract) and (
        any(nonempty_list(contract.get(key)) for key in ("live_in", "live_out", "defs", "uses", "read_set", "write_set", "call_effects", "volatile_ops"))
        or safe_int(contract.get("memory_instruction_count"), 0) > 0
    )


def has_any_dataflow_fact(data: Dict[str, Any]) -> bool:
    if contract_has_known_empty_real_facts(data):
        return True
    return any(
        nonempty_list(data.get(key))
        for key in ("live_in", "live_out", "defs", "uses")
    ) or bool(condition_ssa_from_slot({"data_contract": data}))


def has_any_effect_fact(effect: Dict[str, Any]) -> bool:
    if contract_has_known_empty_real_facts(effect):
        return True
    return any(
        nonempty_list(effect.get(key))
        for key in ("read_set", "write_set", "call_effects", "volatile_ops")
    ) or effect.get("may_throw_or_longjmp") is not None


def text_has_assignment_effect(text: str) -> bool:
    cleaned = re.sub(r"//.*|/\*.*?\*/", " ", str(text or ""), flags=re.S)
    return bool(re.search(r"(?<![=!<>])=(?!=)", cleaned) or "++" in cleaned or "--" in cleaned)


def text_has_call_effect(text: str) -> bool:
    cleaned = re.sub(r"//.*|/\*.*?\*/", " ", str(text or ""), flags=re.S)
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", cleaned))


def text_has_control_effect(text: str) -> bool:
    return bool(re.search(r"\b(return|break|continue|goto)\b", str(text or "")))


def is_semantic_probe_text(text: str) -> bool:
    stripped = str(text or "").strip()
    return stripped.startswith(("ECSL_dataflow_probe(", "ECSL_effect_probe("))


def text_references_any_alias(text: str, aliases: Sequence[str]) -> bool:
    hay = str(text or "")
    hay_norm = re.sub(r"[^A-Za-z0-9_]+", " ", hay)
    for alias in aliases:
        token = str(alias or "").strip()
        if not token:
            continue
        if token in hay:
            return True
        # Allow %cmp0/cmp0, arg.0/arg_0, and similar canonicalization noise.
        token_norm = re.sub(r"[^A-Za-z0-9_]+", " ", token).strip()
        if token_norm and re.search(rf"(?<![A-Za-z0-9_]){re.escape(token_norm)}(?![A-Za-z0-9_])", hay_norm):
            return True
    return False


def contains_unknown_placeholder_text(text: str) -> bool:
    needles = [
        "ECSL_UNKNOWN",
        "ECSL_BINARY_BLOCK",
        "ECSL_EARLY_EXIT_VALUE",
        "label_UNKNOWN",
    ]
    return any(needle in str(text or "") for needle in needles)


REGISTER_SSA_RE = re.compile(
    r"^(?:r(?:[abcd]x|[sd]i|[sb]p|[0-9]+|[0-9]+[bwd]?|ip|flags)|e(?:[abcd]x|[sd]i|[sb]p|ip|flags)|"
    r"(?:[abcd][lh])|[er]?flags|[a-z]{1,3})_[0-9]+$"
)


def is_register_ssa_name(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(REGISTER_SSA_RE.match(text))


def is_register_ssa_move(text: str) -> bool:
    cleaned = re.sub(r"//.*|/\*.*?\*/", " ", str(text or ""), flags=re.S).strip().rstrip(";").strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)$", cleaned)
    return bool(m and is_register_ssa_name(m.group(1)) and is_register_ssa_name(m.group(2)))


def load_external_v3(path: Optional[str]) -> Dict[Tuple[Any, Any, str, str, str], Dict[str, Any]]:
    """Load external V3 rows indexed by function and, when present, slot/candidate.

    Accepted row fields are intentionally permissive so outputs from a symbolic
    executor, trace runner, or differential harness can be plugged in without a
    schema migration.  Strong promotion still requires a passed result and C3+.
    """
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    out: Dict[Tuple[Any, Any, str, str, str], Dict[str, Any]] = {}
    if not p.is_file():
        raise FileNotFoundError(f"external V3 JSONL not found: {p}")
    for _line_no, row in iter_jsonl(p):
        func = str(row.get("func_name") or "")
        slot_id = canonical_slot_key(row.get("slot_id") or row.get("slot") or "")
        candidate_id = str(row.get("candidate_id") or row.get("candidate") or "")
        candidate_text_key = stable_id(str(row.get("candidate_text") or row.get("target_text") or ""), n=16)
        if not slot_id:
            continue
        if not (candidate_id or row.get("candidate_text") or row.get("target_text")):
            continue
        keys = [
            (row.get("example"), row.get("input_line"), func, slot_id, candidate_id),
            (None, row.get("input_line"), func, slot_id, candidate_id),
            (row.get("function_key"), None, "", slot_id, candidate_id),
        ]
        if not candidate_id and candidate_text_key:
            keys.extend(
                [
                    (row.get("example"), row.get("input_line"), func, slot_id, f"text:{candidate_text_key}"),
                    (None, row.get("input_line"), func, slot_id, f"text:{candidate_text_key}"),
                    (row.get("function_key"), None, "", slot_id, f"text:{candidate_text_key}"),
                ]
            )
        for key in keys:
            out[key] = row
    return out


def external_v3_row(
    index: Dict[Tuple[Any, Any, str, str, str], Dict[str, Any]],
    function_record: Dict[str, Any],
    slot: Dict[str, Any],
    cand: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not index:
        return None
    func = str(function_record.get("func_name") or "")
    slot_id = canonical_slot_key(slot.get("slot_id") or cand.get("slot_id") or "")
    candidate_id = str(cand.get("candidate_id") or "")
    text_key = stable_id(candidate_text(cand), n=16)
    keys = [
        (function_record.get("example"), function_record.get("input_line"), func, slot_id, candidate_id),
        (None, function_record.get("input_line"), func, slot_id, candidate_id),
        (function_record.get("function_key"), None, "", slot_id, candidate_id),
        (function_record.get("example"), function_record.get("input_line"), func, slot_id, f"text:{text_key}"),
        (None, function_record.get("input_line"), func, slot_id, f"text:{text_key}"),
        (function_record.get("function_key"), None, "", slot_id, f"text:{text_key}"),
    ]
    for key in keys:
        if key in index:
            return index[key]
    return None


def external_v3_passed(row: Optional[Dict[str, Any]]) -> Tuple[bool, str, str, Dict[str, Any]]:
    if not row:
        return False, "not_available", "no_external_v3_result", {}
    oracle_type = str(row.get("oracle_type") or q(row, "evidence", "oracle_type") or q(row, "validation", "V3", "oracle_type") or "")
    if oracle_type in DIAGNOSTIC_V3_ORACLE_TYPES:
        level = str(row.get("confidence_level") or row.get("tier") or "C0")
        return False, level, f"diagnostic_v3_oracle_rejected={oracle_type}", {
            "oracle_type": oracle_type,
            "diagnostic_rejected": True,
        }
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
    evidence = {
        "external_v3_row_keys": sorted(str(k) for k in row.keys())[:40],
        "raw_confidence_level": row.get("confidence_level") or row.get("tier"),
        "raw_passed": row.get("passed"),
        "raw_v3_pass": row.get("v3_pass"),
        "differential_pass": row.get("differential_pass"),
        "trace_match": row.get("trace_match"),
        "symbolic_match": row.get("symbolic_match"),
    }
    return semantic_pass and TIER_ORDER.get(level, 0) >= TIER_ORDER["C3"], level, reason, evidence


def v2_static_semantic_check(
    slot: Dict[str, Any],
    cand: Dict[str, Any],
    text: str,
    *,
    require_full_dataflow: bool,
) -> Dict[str, Any]:
    """Slot-local V2 compatibility gate.

    V2 should be a useful filter before V3, not a replacement for V3.  It fails
    explicit contradictions (bad grammar-preserving text, wrong control
    transfer template, unsafe placeholder semantics), but missing static
    SSA/effect facts are recorded as ``needs_v3_*`` warnings.  That keeps the
    V3 task pool healthy while preserving the rule that only real V3 can create
    hard C3 labels.
    """
    raw_slot_type = str(slot.get("slot_type") or "").upper()
    slot_type = normalize_slot_type(raw_slot_type)
    op_kind = module_b_slot_kind(slot)
    category = str(q(slot, "grammar_contract", "category") or "unknown")
    control = slot.get("control_contract") if isinstance(slot.get("control_contract"), dict) else {}
    data = slot.get("data_contract") if isinstance(slot.get("data_contract"), dict) else {}
    effect = slot.get("effect_contract") if isinstance(slot.get("effect_contract"), dict) else {}
    provenance = cand.get("provenance") if isinstance(cand.get("provenance"), dict) else {}
    source = ",".join(str(x) for x in (provenance.get("source") or []))
    rewrite_kind = str(provenance.get("rewrite_kind") or "")
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {
        "slot_type": slot_type,
        "raw_slot_type": raw_slot_type,
        "module_b_slot_kind": op_kind,
        "grammar_category": category,
        "candidate_source": source,
        "rewrite_kind": rewrite_kind,
        "control_status": control.get("status"),
        "data_status": data.get("status"),
        "effect_status": effect.get("status"),
        "data_fact_state": data.get("fact_state"),
        "effect_fact_state": effect.get("fact_state"),
        "real_ssa_available": q(data, "ssa_grounding", "real_ssa_available"),
        "condition_ssa": condition_ssa_from_slot(slot),
        "allowed_control_effects": control.get("allowed_control_effects") or [],
        "live_in_count": len(data.get("live_in") or []),
        "live_out_count": len(data.get("live_out") or []),
        "defs_count": len(data.get("defs") or []),
        "uses_count": len(data.get("uses") or []),
        "read_set_count": len(effect.get("read_set") or []),
        "write_set_count": len(effect.get("write_set") or []),
        "call_effect_count": len(effect.get("call_effects") or []),
    }

    if contains_unknown_placeholder_text(text):
        errors.append("candidate_contains_unknown_or_placeholder_semantics")
    semantic_probe = is_semantic_probe_text(text)

    if contract_has_proxy_facts(data):
        warnings.append("needs_v3_data_contract_proxy_not_clean_v2")
    if contract_has_proxy_facts(effect):
        warnings.append("needs_v3_effect_contract_proxy_not_clean_v2")
    if contract_has_known_empty_real_facts(data) and not semantic_probe:
        symbols = identifiers_in_text(text)
        if symbols:
            errors.append("candidate_references_symbols_against_empty_data_contract=" + ",".join(symbols[:8]))
    if contract_has_known_empty_real_facts(effect):
        unexpected_effects = []
        if text_has_assignment_effect(text):
            unexpected_effects.append("assignment")
        if text_has_call_effect(text) and not semantic_probe:
            unexpected_effects.append("call_like")
        allowed_effects = set(str(x) for x in (control.get("allowed_control_effects") or []))
        if text_has_control_effect(text) and not (allowed_effects - {"none"}):
            unexpected_effects.append("control_transfer")
        if unexpected_effects:
            errors.append("candidate_introduces_effect_against_empty_effect_contract=" + ",".join(unexpected_effects))

    if slot_type == "COND":
        condition_ssa = condition_ssa_from_slot(slot)
        symbol_aliases = condition_symbol_aliases(slot)
        evidence["condition_symbol_aliases"] = symbol_aliases[:32]
        if not condition_ssa:
            warnings.append("needs_v3_cond_missing_condition_ssa")
            if require_full_dataflow:
                errors.append("cond_missing_condition_ssa")
        elif symbol_aliases and not text_references_any_alias(text, symbol_aliases):
            warnings.append("needs_v3_cond_symbol_alpha_or_expression_equivalence")
        elif not symbol_aliases:
            warnings.append("needs_v3_cond_missing_symbol_mapping")
            if require_full_dataflow:
                errors.append("cond_missing_symbol_mapping")
        if q(data, "ssa_grounding", "real_ssa_available") is False:
            warnings.append("needs_v3_cond_real_ssa_not_available")
            if require_full_dataflow:
                errors.append("cond_real_ssa_not_available")
        if not control.get("true_successor") or not control.get("false_successor"):
            warnings.append("needs_v3_cond_missing_true_false_successors")
            if require_full_dataflow:
                errors.append("cond_missing_true_false_successors")
        if rewrite_kind == "branch_inversion":
            warnings.append("branch_inversion_needs_successor_polarity_validation")
            # Branch inversion can be valid without a literal leading "!":
            # e.g. "(x == 0)" vs "x".  Leave polarity equivalence to V3.
        if require_full_dataflow and not (nonempty_list(data.get("uses")) or condition_ssa):
            errors.append("cond_missing_uses_or_condition_dataflow")
        if effect.get("may_throw_or_longjmp"):
            warnings.append("needs_v3_cond_may_throw_or_longjmp")
            if require_full_dataflow:
                errors.append("cond_may_throw_or_longjmp")
        if nonempty_list(effect.get("volatile_ops")):
            warnings.append("needs_v3_cond_volatile_order_proof")
            if require_full_dataflow:
                errors.append("cond_volatile_ops_need_order_proof")
        if nonempty_list(effect.get("write_set")):
            warnings.append("needs_v3_cond_write_effect_order_proof")
            if require_full_dataflow:
                errors.append("cond_write_effect_not_allowed")

    elif op_kind in {"BREAK", "CONTINUE", "GOTO", "EARLY_EXIT", "CLEANUP_EXIT"}:
        allowed = set(str(x) for x in (control.get("allowed_control_effects") or []))
        if op_kind == "BREAK":
            if "break" not in allowed or not text.startswith("break"):
                errors.append("break_candidate_or_allowed_effect_mismatch")
            if not control.get("break_target"):
                warnings.append("needs_v3_break_missing_target")
                if require_full_dataflow:
                    errors.append("break_missing_target")
        elif op_kind == "CONTINUE":
            if "continue" not in allowed or not text.startswith("continue"):
                errors.append("continue_candidate_or_allowed_effect_mismatch")
            if not control.get("continue_target"):
                warnings.append("needs_v3_continue_missing_target")
                if require_full_dataflow:
                    errors.append("continue_missing_target")
        elif op_kind == "GOTO":
            edge = control.get("edge")
            target = None
            if isinstance(edge, list) and len(edge) >= 2:
                target = bb_name(edge[1])
            expected_labels = {f"label_{target}", f"label_BB_{target}"} if target else set()
            if "goto_like_internal" not in allowed or not text.startswith("goto "):
                errors.append("goto_candidate_or_allowed_effect_mismatch")
            if not target:
                warnings.append("needs_v3_goto_missing_edge_target")
                if require_full_dataflow:
                    errors.append("goto_missing_edge_target")
            elif not any(label in text for label in expected_labels):
                errors.append(f"goto_target_label_mismatch_expected={sorted(expected_labels)}")
        elif op_kind == "EARLY_EXIT":
            if "return" not in allowed or not text.startswith("return"):
                errors.append("early_exit_candidate_or_allowed_effect_mismatch")
            if "ECSL_EARLY_EXIT_VALUE" in text:
                errors.append("early_exit_return_value_not_synthesized")
            if not contract_status_available(data):
                warnings.append("needs_v3_early_exit_data_contract")
        elif op_kind == "CLEANUP_EXIT":
            if "goto_like_internal" not in allowed or not text.startswith("goto "):
                errors.append("cleanup_exit_candidate_or_allowed_effect_mismatch")
            edge = control.get("edge")
            target = None
            if isinstance(edge, list) and len(edge) >= 2:
                target = bb_name(edge[1])
            expected_labels = {f"label_{target}", f"label_BB_{target}"} if target else set()
            if "cleanup" not in text and not any(label in text for label in expected_labels):
                errors.append("cleanup_exit_missing_cleanup_or_cfg_target_label")
        simple_target_only_transfer = op_kind in {"BREAK", "CONTINUE", "GOTO"}
        if not simple_target_only_transfer and not contract_status_available(data):
            warnings.append("needs_v3_control_transfer_data_contract")
        if not simple_target_only_transfer and not contract_status_available(effect):
            warnings.append("needs_v3_control_transfer_effect_contract")
        if require_full_dataflow and (
            not simple_target_only_transfer
            and (not contract_status_available(data) or not contract_status_available(effect))
        ):
            errors.append("control_transfer_missing_full_data_effect_contract")

    elif slot_type in {"BLOCK", "CASE_LABEL", "STMT", "EXPR", "LVALUE", "CALL_ARG", "CALL_TARGET", "MEMORY_EFFECT", "PHI_OR_MERGE", "RETURN", "EXIT", "LOOP_EXIT", "SWITCH_DISPATCH"} or op_kind in {"BLOCK", "CASE", "STMT"}:
        if is_register_ssa_move(text):
            errors.append("block_candidate_register_ssa_move_not_source_level_semantics")
        if "ECSL_BINARY_BLOCK" in text:
            errors.append("block_candidate_is_comment_placeholder_not_semantic_fill")
        if contains_forbidden_control(text, q(slot, "grammar_contract", "forbidden_control_transfer") or []):
            errors.append("block_forbidden_control_transfer")
        simple_control_text = text.strip().startswith(("return", "break", "continue", "goto "))
        if slot_type == "STMT" and simple_control_text:
            # Simple control statements are checked by grammar/V1 and then V3.
            # They usually do not carry populated read/write/effect summaries in
            # the O1 contracts, so treating missing effect facts as V2 failure
            # would incorrectly block every simple statement candidate.
            pass
        elif not contract_has_real_or_partial_facts(data):
            if contract_has_proxy_empty_facts(data):
                warnings.append("needs_v3_block_data_contract_proxy_empty")
            else:
                warnings.append("needs_v3_block_data_contract")
        if slot_type == "STMT" and simple_control_text:
            pass
        elif not contract_has_real_or_partial_facts(effect):
            if contract_has_proxy_empty_facts(effect):
                warnings.append("needs_v3_block_effect_contract_proxy_empty")
            else:
                warnings.append("needs_v3_block_effect_contract")
        if text.strip() == ";":
            if slot_contracts_prove_empty_observable_effect(slot):
                warnings = [w for w in warnings if w not in {"needs_v3_block_data_contract", "needs_v3_block_effect_contract"}]
            else:
                warnings.append("needs_v3_empty_effect_probe_not_proven_empty")
        if semantic_probe:
            warnings.append("needs_v3_semantic_probe_synthesis_or_full_function_validation")
        if require_full_dataflow:
            if not has_any_dataflow_fact(data):
                errors.append("block_missing_liveout_defs_uses")
            if not has_any_effect_fact(effect):
                errors.append("block_missing_readwrite_effect_summary")
    else:
        errors.append(f"unsupported_slot_type_for_v2={slot_type or 'UNKNOWN'};module_b_slot_kind={op_kind or 'UNKNOWN'}")

    clean_pass = not errors and not warnings
    compatible = not errors
    status = "pass" if clean_pass else ("partial_needs_v3" if compatible else "fail")
    reason = (
        "v2_contract_compatible_needs_v3"
        if compatible and warnings
        else ("v2_contract_compatible" if clean_pass else ";".join(errors[:8]))
    )
    evidence["warnings"] = warnings
    evidence["compatible"] = bool(compatible)
    evidence["clean_pass"] = bool(clean_pass)
    evidence["needs_v3"] = bool(warnings)
    return validation_level(
        "V2",
        "ssa_liveout_readwrite_effect",
        passed=clean_pass,
        status=status,
        errors=errors,
        reason=reason,
        evidence=evidence,
        note=(
            "V2 checks slot-local SSA/control/effect contract compatibility. "
            "It is C2 evidence, not differential execution."
        ),
    )


def fill_remaining_placeholders_with_stubs(rendered: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tag = match.group("tag")
        trailing = ""
        if tag == "FILL_COND":
            return "1" + trailing
        if tag == "CASE_SLOT":
            return "case 0:"
        if tag in {"GOTO_SLOT", "BREAK_SLOT", "CONTINUE_SLOT", "EARLY_EXIT_SLOT", "CLEANUP_EXIT_SLOT"}:
            return "/* unresolved control slot */;" + trailing
        return "/* unresolved slot */" + trailing

    return PLACEHOLDER_RE.sub(repl, rendered)


def rendered_candidate_text(function_record: Dict[str, Any], slot: Dict[str, Any], text: str) -> Optional[str]:
    skeleton = str(function_record.get("verified_skeleton_text") or "")
    category = str(q(slot, "grammar_contract", "category") or "unknown")
    slot_id = str(slot.get("slot_id") or "")
    rendered = render_slot_candidate(skeleton, slot_id, text, category)
    value = rendered.get("rendered")
    return str(value) if value is not None else None


def compile_smoke_check(
    function_record: Dict[str, Any],
    slot: Dict[str, Any],
    cand: Dict[str, Any],
    text: str,
    *,
    cc: str,
    keep_dir: Optional[str],
) -> Dict[str, Any]:
    rendered = rendered_candidate_text(function_record, slot, text)
    if rendered is None:
        return validation_level(
            "V3",
            "compile_smoke",
            passed=False,
            status="fail",
            errors=["render_failed_before_compile"],
            reason="render_failed_before_compile",
        )
    stubbed = fill_remaining_placeholders_with_stubs(rendered)
    # The Module-B skeleton often lacks includes, typedefs, and declarations.
    # Compile smoke is therefore diagnostic evidence only; external V3 is needed
    # for semantic promotion.
    source = "\n".join(
        [
            "/* ECSL V3 compile smoke. This is not a semantic differential proof. */",
            "#include <stddef.h>",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#define label_UNKNOWN label_BB_0",
            stubbed,
            "",
        ]
    )
    tmp_obj: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        if keep_dir:
            work = Path(keep_dir).expanduser().resolve()
            work.mkdir(parents=True, exist_ok=True)
        else:
            tmp_obj = tempfile.TemporaryDirectory(prefix="ecsl_v3_compile_")
            work = Path(tmp_obj.name)
        stem = stable_id(function_record.get("function_key"), slot.get("slot_id"), cand.get("candidate_id"), n=16)
        src_path = work / f"{stem}.c"
        src_path.write_text(source, encoding="utf-8")
        cmd = [cc, "-x", "c", "-std=gnu11", "-fsyntax-only", str(src_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
        ok = proc.returncode == 0
        return validation_level(
            "V3",
            "compile_smoke",
            passed=ok,
            status="pass" if ok else "fail",
            errors=[] if ok else ["compile_smoke_failed"],
            reason="compile_smoke_pass" if ok else "compile_smoke_failed",
            evidence={
                "cmd": cmd,
                "source_path": str(src_path),
                "returncode": proc.returncode,
                "output_head": (proc.stdout or "")[:4000],
                "compile_only_not_semantic": True,
            },
            note="Compile smoke records parser/compiler evidence only; it is not trace/symbolic/differential equivalence.",
        )
    except Exception as e:
        return validation_level(
            "V3",
            "compile_smoke",
            passed=False,
            status="fail",
            errors=["compile_smoke_exception"],
            reason=f"compile_smoke_exception:{e}",
        )
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


def run_v3_command(
    command_template: str,
    function_record: Dict[str, Any],
    slot: Dict[str, Any],
    cand: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:
    rendered = rendered_candidate_text(function_record, slot, text) or ""
    with tempfile.TemporaryDirectory(prefix="ecsl_v3_command_") as tmp:
        tmp_path = Path(tmp)
        candidate_path = tmp_path / "candidate.c"
        meta_path = tmp_path / "meta.json"
        candidate_path.write_text(fill_remaining_placeholders_with_stubs(rendered), encoding="utf-8")
        meta = {
            "function_key": function_record.get("function_key"),
            "example": function_record.get("example"),
            "input_line": function_record.get("input_line"),
            "func_name": function_record.get("func_name"),
            "slot_id": slot.get("slot_id"),
            "candidate_id": cand.get("candidate_id"),
            "candidate_text": text,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        env = dict(os.environ)
        env.update(
            {
                "ECSL_FUNCTION_KEY": str(function_record.get("function_key") or ""),
                "ECSL_EXAMPLE": str(function_record.get("example") or ""),
                "ECSL_INPUT_LINE": str(function_record.get("input_line") or ""),
                "ECSL_FUNC_NAME": str(function_record.get("func_name") or ""),
                "ECSL_SLOT_ID": str(slot.get("slot_id") or ""),
                "ECSL_CANDIDATE_ID": str(cand.get("candidate_id") or ""),
                "ECSL_CANDIDATE_C": str(candidate_path),
                "ECSL_META_JSON": str(meta_path),
            }
        )
        proc = subprocess.run(
            command_template,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=300,
        )
        raw = (proc.stdout or "").strip()
        parsed: Dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw.splitlines()[-1])
            except Exception:
                parsed = {}
        ok = bool(
            strict_true(parsed.get("differential_pass"))
            or strict_true(parsed.get("trace_match"))
            or strict_true(parsed.get("symbolic_match"))
        )
        if proc.returncode != 0:
            ok = False
        level = str(parsed.get("confidence_level") or parsed.get("tier") or ("C3" if ok else "C0"))
        promoted = ok and TIER_ORDER.get(level, 0) >= TIER_ORDER["C3"]
        return validation_level(
            "V3",
            "symbolic_trace_or_differential_execution",
            passed=promoted,
            status="pass" if promoted else "fail",
            errors=[] if promoted else ["v3_command_failed_or_not_c3"],
            reason=str(parsed.get("reason") or parsed.get("status") or ("v3_command_semantic_pass" if promoted else "v3_command_missing_semantic_proof_or_not_c3")),
            evidence={
                "command": command_template,
                "returncode": proc.returncode,
                "stdout_head": raw[:4000],
                "parsed": parsed,
                "confidence_level": level,
            },
            note="External command is treated as V3 only when it reports C3+ differential/trace/symbolic pass.",
        )


def v3_validation_check(
    function_record: Dict[str, Any],
    slot: Dict[str, Any],
    cand: Dict[str, Any],
    text: str,
    args: argparse.Namespace,
    external_index: Dict[Tuple[Any, Any, str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    row = external_v3_row(external_index, function_record, slot, cand)
    ext_ok, ext_level, ext_reason, ext_evidence = external_v3_passed(row)
    compile_report = None
    if bool(args.enable_v3_compile_smoke):
        cc = str(args.v3_cc or "")
        resolved = shutil.which(cc) if cc else None
        if resolved:
            compile_report = compile_smoke_check(
                function_record,
                slot,
                cand,
                text,
                cc=resolved,
                keep_dir=args.v3_compile_keep_dir,
            )
        else:
            compile_report = validation_level(
                "V3",
                "compile_smoke",
                passed=False,
                status="not_available",
                reason=f"compiler_not_found={cc or '<empty>'}",
            )
    if ext_ok:
        return validation_level(
            "V3",
            "symbolic_trace_or_differential_execution",
            passed=True,
            status="pass",
            reason=ext_reason,
            evidence={"external_v3": ext_evidence, "compile_smoke": compile_report},
            note="Passed by external C3+ symbolic/trace/differential validator.",
        )
    if args.v3_command:
        command_report = run_v3_command(str(args.v3_command), function_record, slot, cand, text)
        if command_report.get("passed"):
            if compile_report is not None:
                command_report.setdefault("evidence", {})["compile_smoke"] = compile_report
            return command_report
        return validation_level(
            "V3",
            "symbolic_trace_or_differential_execution",
            passed=False,
            status="fail",
            errors=["external_v3_command_failed"],
            reason=str(command_report.get("reason") or "external_v3_command_failed"),
            evidence={"external_v3": ext_evidence, "v3_command": command_report, "compile_smoke": compile_report},
        )
    if bool(args.allow_compile_smoke_as_v3) and compile_report and compile_report.get("passed"):
        return validation_level(
            "V3",
            "symbolic_trace_or_differential_execution",
            passed=False,
            status="compile_smoke_only_not_v3",
            reason="compile_smoke_is_diagnostic_not_v3",
            evidence={"external_v3": ext_evidence, "compile_smoke": compile_report},
            note="Compile smoke is diagnostic only; ECSL C3 requires trace/symbolic/differential evidence.",
        )
    evidence = {"external_v3": ext_evidence, "compile_smoke": compile_report}
    return validation_level(
        "V3",
        "symbolic_trace_or_differential_execution",
        passed=False,
        status="fail" if row or compile_report else "not_available",
        errors=[] if not row else ["external_v3_not_passed"],
        reason=ext_reason,
        evidence=evidence,
        note="V3 requires external symbolic/trace/differential pass; compile smoke alone is diagnostic only.",
    )


def keep_external_candidate(
    slot: Dict[str, Any],
    *,
    source: str,
    semantic_kind: str,
    summary: str,
    requires: Sequence[str],
    rewrite_kind: Optional[str] = None,
) -> Dict[str, Any]:
    slot_id = str(slot.get("slot_id") or "")
    candidate_id = "keep_" + stable_id(slot_id, source, semantic_kind, rewrite_kind or summary)
    region_bbs = list(q(slot, "region", "region_bbs") or [])
    return {
        "schema": "SlotCandidate.v0",
        "candidate_id": candidate_id,
        "slot_id": slot_id,
        "semantic_object": {
            "kind": semantic_kind,
            "ir_refs": region_bbs,
            "summary": summary,
        },
        "typed_canonical_ir": {
            "expr_tree": None,
            "stmt_region": None,
            "transformer": {"live_in_to_live_out": None, "status": "KEEP_EXTERNAL"},
            "type": None,
        },
        "c_renderings": [{"text": "", "style": "KEEP_EXTERNAL", "requires": list(requires)}],
        "provenance": {
            "source": [source],
            "source_refs": region_bbs,
            "rewrite_kind": rewrite_kind,
            "status": "KEEP_EXTERNAL",
        },
        "matching": semantic_matching_report(
            slot,
            source=source,
            base_alignment_score=0.0,
            rewrite_kind=rewrite_kind,
            generated_text_available=False,
        ),
        "validation": {},
        "confidence": {},
        "keep": {
            "status": "KEEP_EXTERNAL",
            "reason": "external data/model/tool is required; this placeholder must not be treated as a label",
        },
    }


def candidate_template(
    slot: Dict[str, Any],
    *,
    text: str,
    source: str,
    semantic_kind: str,
    summary: str,
    style: str = "canonical",
    requires: Optional[List[str]] = None,
    rewrite_kind: Optional[str] = None,
    base_alignment_score: float = 0.1,
) -> Dict[str, Any]:
    slot_id = str(slot.get("slot_id") or "")
    candidate_id = "cand_" + stable_id(slot_id, source, semantic_kind, text, rewrite_kind or "")
    region_bbs = list(q(slot, "region", "region_bbs") or [])
    return {
        "schema": "SlotCandidate.v0",
        "candidate_id": candidate_id,
        "slot_id": slot_id,
        "semantic_object": {
            "kind": semantic_kind,
            "ir_refs": region_bbs,
            "summary": summary,
        },
        "typed_canonical_ir": {
            "expr_tree": None,
            "stmt_region": region_bbs if semantic_kind in {"statement_transformer", "memory_effect"} else None,
            "transformer": {
                "live_in_to_live_out": None,
                "status": "not_available_o1",
            },
            "type": None,
        },
        "c_renderings": [
            {
                "text": text,
                "style": style,
                "requires": list(requires or []),
            }
        ],
        "provenance": {
            "source": [source],
            "source_refs": region_bbs,
            "rewrite_kind": rewrite_kind,
        },
        "matching": semantic_matching_report(
            slot,
            source=source,
            base_alignment_score=float(base_alignment_score),
            rewrite_kind=rewrite_kind,
            generated_text_available=True,
        ),
        "validation": {},
        "confidence": {},
    }


def source_cfg_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        keep_external_candidate(
            slot,
            source="source_cfg",
            semantic_kind="source_semantic_unit",
            summary="source CFG/SSA candidate extractor placeholder",
            requires=["source_code", "source_cfg", "source_ssa", "semantic_matcher"],
        )
    ]


def execution_trace_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        keep_external_candidate(
            slot,
            source="execution_trace",
            semantic_kind="trace_observation",
            summary="execution trace or unit-test trace candidate placeholder",
            requires=["test_inputs_or_traces", "trace_to_slot_observation_mapper"],
        )
    ]


def self_training_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        keep_external_candidate(
            slot,
            source="self_training",
            semantic_kind="validated_model_prediction",
            summary="self-training candidate placeholder from future high-confidence Module-C predictions",
            requires=["model_prediction", "renderer", "validator_C3_or_above"],
        )
    ]


def cond_binary_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    ssa = condition_ssa_from_slot(slot)
    slot_id = str(slot.get("slot_id") or "")
    if ssa:
        sym = sanitize_identifier(ssa, prefix="ssa")
        text = f"({sym} != 0)"
        requires = ["ssa_symbol_mapping", "condition_polarity_validation"]
        base = 0.35
        summary = f"binary branch condition candidate from condition_ssa={ssa}"
    else:
        text = "1 /* ECSL_UNKNOWN_COND */"
        requires = ["recover_condition_expression"]
        base = 0.05
        summary = "placeholder condition prior because no condition_ssa is available"
    out = [
        candidate_template(
            slot,
            text=text,
            source="binary_ir",
            semantic_kind="predicate",
            summary=summary,
            requires=requires,
            base_alignment_score=base,
        )
    ]
    if ssa:
        out.append(
            candidate_template(
                slot,
                text=f"!({text})",
                source="rewrite_template",
                semantic_kind="predicate",
                summary="branch inversion candidate for true/false successor matching",
                requires=["condition_side_effect_free", "successor_polarity_validation"],
                rewrite_kind="branch_inversion",
                base_alignment_score=max(0.0, base - 0.08),
            )
        )
    else:
        out.append(
            candidate_template(
                slot,
                text="0 /* ECSL_UNKNOWN_INVERTED_COND */",
                source="rewrite_template",
                semantic_kind="predicate",
                summary=f"low-confidence inverted condition prior for {slot_id}",
                requires=["recover_condition_expression", "successor_polarity_validation"],
                rewrite_kind="branch_inversion",
                base_alignment_score=0.03,
            )
        )
    return out


def block_binary_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    slot_id = str(slot.get("slot_id") or "")
    bbs = list(q(slot, "region", "region_bbs") or [])
    bb_text = " ".join(str(bb) for bb in bbs) if bbs else "unknown_region"
    out: List[Dict[str, Any]] = []
    ret = return_ssa_from_slot(slot)
    if ret:
        sym = sanitize_identifier(ret, prefix="ret")
        out.append(
            candidate_template(
                slot,
                text=f"return {sym};",
                source="binary_ir",
                semantic_kind="statement_transformer",
                summary=f"concrete return candidate from return_ssa={ret}",
                requires=["return_ssa_source_mapping", "return_value_differential_validation"],
                base_alignment_score=0.34,
            )
        )
    ret_const = return_constant_from_slot(slot)
    if ret_const is not None:
        out.append(
            candidate_template(
                slot,
                text=f"return {ret_const};",
                source="binary_ir",
                semantic_kind="statement_transformer",
                summary=f"concrete return candidate from verified return_constant={ret_const}",
                requires=["return_constant_contract_validation"],
                base_alignment_score=0.36,
            )
        )
    assign = simple_assignment_from_slot(slot)
    if assign:
        lhs, rhs = assign
        out.append(
            candidate_template(
                slot,
                text=f"{lhs} = {rhs};",
                source="binary_ir",
                semantic_kind="statement_transformer",
                summary=f"concrete assignment candidate from def/use {lhs}<={rhs}",
                requires=["def_use_source_mapping", "write_effect_differential_validation"],
                base_alignment_score=0.30,
            )
        )
    if slot_can_probe_empty_observable_effect(slot):
        proven_empty = slot_contracts_prove_empty_observable_effect(slot)
        out.append(
            candidate_template(
                slot,
                text=";",
                source="binary_ir",
                semantic_kind="statement_transformer",
                summary=(
                    "empty statement candidate from proven-empty data/effect slot contract"
                    if proven_empty
                    else "empty statement V3 probe because no observable live-out/write/call effect is known"
                ),
                requires=(
                    ["empty_liveout_effect_contract_validation"]
                    if proven_empty
                    else ["v3_empty_effect_or_full_function_validation"]
                ),
                base_alignment_score=0.38 if proven_empty else 0.16,
            )
        )
    out.extend(block_semantic_probe_candidates(slot))
    text = f"/* ECSL_BINARY_BLOCK {slot_id} covers {bb_text}; recover statements from SSA/effects */"
    out.append(
        candidate_template(
            slot,
            text=text,
            source="binary_ir",
            semantic_kind="statement_transformer",
            summary="binary-region no-op prior; not a semantic teacher until V2/V3 validates effects",
            requires=["ssa_statement_synthesis", "live_out_effect_validation"],
            base_alignment_score=0.08 if bbs else 0.02,
        )
    )
    return out


def control_transfer_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    slot_type = module_b_slot_kind(slot)
    edge = q(slot, "provenance", "edge") or q(slot, "control_contract", "edge")
    target = None
    if isinstance(edge, list) and len(edge) >= 2:
        target = safe_int(edge[1])
    out: List[Dict[str, Any]] = []

    def add(text: str, summary: str, requires: List[str], *, base: float = 0.25, source: str = "rewrite_template") -> None:
        out.append(
            candidate_template(
                slot,
                text=text,
                source=source,
                semantic_kind="statement_transformer",
                summary=summary,
                requires=requires,
                rewrite_kind=slot_type.lower(),
                base_alignment_score=base,
            )
        )

    if slot_type == "BREAK":
        add("break;", "loop break template from Module-B escape edge", ["loop_break_target_validation"])
    elif slot_type == "CONTINUE":
        add("continue;", "loop continue template from Module-B escape edge", ["loop_continue_target_validation"])
    elif slot_type == "GOTO":
        label = f"label_BB_{target}" if target is not None else "label_UNKNOWN"
        add(f"goto {label};", "goto-like internal edge template", ["target_label_rendering", "structured_goto_policy"])
    elif slot_type == "EARLY_EXIT":
        ret = return_ssa_from_slot(slot)
        if ret:
            sym = sanitize_identifier(ret, prefix="ret")
            add(
                f"return {sym};",
                f"early return candidate from return_ssa={ret}",
                ["return_ssa_source_mapping", "return_value_differential_validation"],
                base=0.34,
                source="binary_ir",
            )
        ret_const = return_constant_from_slot(slot)
        if ret_const is not None:
            add(
                f"return {ret_const};",
                f"early return candidate from verified return_constant={ret_const}",
                ["return_constant_contract_validation"],
                base=0.36,
                source="binary_ir",
            )
        # Constant return values are common in guard/early-exit rewrites.  They
        # are concrete candidates only; V3 must still prove them against source
        # or binary semantics before they can become hard labels.
        for value in ("0", "1", "-1"):
            add(
                f"return {value};",
                f"early return constant candidate {value}",
                ["return_value_differential_validation"],
                base=0.22,
            )
    elif slot_type == "CLEANUP_EXIT":
        if target is not None:
            add(
                f"goto label_BB_{target};",
                "cleanup exit rendered as explicit Module-B CFG target label",
                ["target_label_rendering", "cleanup_target_validation"],
                base=0.27,
            )
        add("goto cleanup;", "cleanup exit template; cleanup label must be rendered consistently", ["cleanup_label_rendering", "cleanup_target_validation"])
    return out


def rewrite_template_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    slot_type = normalize_slot_type(slot.get("slot_type"))
    op_kind = module_b_slot_kind(slot)
    sem_type = str(slot.get("semantic_slot_type") or "").upper()
    role = str(slot.get("role") or "").lower()
    out: List[Dict[str, Any]] = []

    def add(kind: str) -> None:
        spec = REWRITE_KINDS[kind]
        out.append(
            keep_external_candidate(
                slot,
                source="rewrite_template",
                semantic_kind="rewrite_candidate_request",
                summary=str(spec["summary"]),
                requires=list(spec["requires"]),
                rewrite_kind=kind,
            )
        )

    if slot_type == "COND":
        add("branch_inversion")
        add("nested_if_conjunction")
        if sem_type == "SWITCH_DISPATCH" or "switch" in role:
            add("switch_if_chain")
        if "loop" in role:
            add("while_loop_break")
    if slot_type in {"BLOCK", "CASE_LABEL"} or op_kind in {"BLOCK", "CASE"}:
        add("early_return_if_else")
        add("continue_guarded_body")
        if slot_type == "CASE_LABEL" or op_kind == "CASE":
            add("switch_if_chain")
    if op_kind in {"BREAK", "CONTINUE"}:
        add("while_loop_break")
        add("continue_guarded_body")
    if op_kind in {"EARLY_EXIT", "CLEANUP_EXIT"}:
        add("early_return_if_else")
    return out


def llm_proposal_request(slot: Dict[str, Any]) -> Dict[str, Any]:
    slot_id = str(slot.get("slot_id") or "")
    category = str(q(slot, "grammar_contract", "category") or "unknown")
    prompt = {
        "task": "propose_slot_fill_candidates",
        "slot_id": slot_id,
        "slot_type": slot.get("slot_type"),
        "semantic_slot_type": slot.get("semantic_slot_type"),
        "grammar_category": category,
        "anchor": slot.get("anchor"),
        "region": slot.get("region"),
        "control_contract": slot.get("control_contract"),
        "data_contract": slot.get("data_contract"),
        "effect_contract": slot.get("effect_contract"),
        "constraints": [
            "do not modify skeleton tokens",
            "obey grammar_category",
            "do not output source-AST-only labels unless validator can prove compatibility",
        ],
    }
    cand = keep_external_candidate(
        slot,
        source="llm_proposal",
        semantic_kind="llm_request",
        summary="LLM proposal request; not a candidate label until generated text passes validator",
        requires=["external_llm_generation", "validator_pass"],
    )
    cand["llm_prompt"] = prompt
    return cand


def generate_candidates(
    slot: Dict[str, Any],
    *,
    emit_llm_requests: bool,
    emit_keep_external: bool,
) -> List[Dict[str, Any]]:
    slot_type = normalize_slot_type(slot.get("slot_type"))
    op_kind = module_b_slot_kind(slot)
    category = str(q(slot, "grammar_contract", "category") or "unknown")
    out: List[Dict[str, Any]] = []
    if emit_keep_external:
        out.extend(source_cfg_candidates(slot))
        out.extend(execution_trace_candidates(slot))
        out.extend(self_training_candidates(slot))
    if category == "unsupported_region" or op_kind == "UNSTRUCTURED_REGION":
        if emit_llm_requests:
            out.append(llm_proposal_request(slot))
        return out
    if slot_type == "COND":
        out.extend(cond_binary_candidates(slot))
    elif slot_type in {"BLOCK", "CASE_LABEL", "STMT", "EXPR", "LVALUE", "CALL_ARG", "CALL_TARGET", "MEMORY_EFFECT", "PHI_OR_MERGE"} or op_kind in {"BLOCK", "CASE", "STMT"}:
        out.extend(block_binary_candidates(slot))
    elif op_kind in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
        out.extend(control_transfer_candidates(slot))
    if emit_keep_external:
        out.extend(rewrite_template_candidates(slot))
    if emit_llm_requests:
        out.append(llm_proposal_request(slot))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for cand in out:
        text = candidate_text(cand)
        if q(cand, "keep", "status") == "KEEP_EXTERNAL":
            key = (
                "KEEP_EXTERNAL",
                ",".join(str(x) for x in (q(cand, "provenance", "source") or [])),
                str(q(cand, "provenance", "rewrite_kind") or ""),
                str(q(cand, "semantic_object", "kind") or ""),
            )
        else:
            key = (
                "TEXT",
                text,
                str(q(cand, "semantic_object", "kind") or ""),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def validate_candidate(
    function_record: Dict[str, Any],
    slot: Dict[str, Any],
    cand: Dict[str, Any],
    args: argparse.Namespace,
    external_v3_index: Dict[Tuple[Any, Any, str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    if q(cand, "keep", "status") == "KEEP_EXTERNAL":
        return {
            "V0": unavailable_level("V0", "parser_type_slot_grammar", "KEEP_EXTERNAL candidate has no generated text yet"),
            "V1": unavailable_level("V1", "skeleton_preservation", "KEEP_EXTERNAL candidate has no generated text yet"),
            "V2": validation_level(
                "V2",
                "ssa_liveout_readwrite_effect",
                passed=False,
                status="not_available",
                reason="KEEP_EXTERNAL candidate has no generated text yet",
            ),
            "V3": validation_level(
                "V3",
                "symbolic_trace_or_differential_execution",
                passed=False,
                status="not_available",
                reason="KEEP_EXTERNAL candidate has no generated text yet",
            ),
        }
    text = candidate_text(cand)
    grammar = v0_parser_type_check(text, slot.get("grammar_contract") or {}, cc=getattr(args, "v0_cc", None))
    skeleton = str(function_record.get("verified_skeleton_text") or "")
    preservation = skeleton_preservation_check(skeleton, slot, text)
    if bool(args.enable_v2_validator):
        v2 = v2_static_semantic_check(
            slot,
            cand,
            text,
            require_full_dataflow=bool(args.v2_require_full_dataflow),
        )
    else:
        v2 = unavailable_level(
            "V2",
            "ssa_liveout_readwrite_effect",
            "disabled by --no-enable-v2-validator",
        )
    if not grammar.get("passed") or not preservation.get("passed"):
        v3 = validation_level(
            "V3",
            "symbolic_trace_or_differential_execution",
            passed=False,
            status="not_attempted",
            reason="V0/V1 failed before V3",
            evidence={"v0_passed": grammar.get("passed"), "v1_passed": preservation.get("passed")},
        )
    else:
        v3 = v3_validation_check(function_record, slot, cand, text, args, external_v3_index)
    return {
        "V0": grammar,
        "V1": preservation,
        "V2": v2,
        "V3": v3,
    }


def score_confidence(slot: Dict[str, Any], cand: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
    if q(cand, "keep", "status") == "KEEP_EXTERNAL":
        return {
            "confidence_level": "C0",
            "confidence_weight": 0.0,
            "hard_ce_allowed": False,
            "reasons": ["keep_external_candidate_not_generated_or_validated"],
        }
    source = ",".join(cand.get("provenance", {}).get("source") or [])
    if "llm_proposal" in source:
        return {
            "confidence_level": "C0",
            "confidence_weight": 0.0,
            "hard_ce_allowed": False,
            "reasons": ["llm_request_not_validated_candidate"],
        }
    if not bool(q(slot, "b_gate", "pass")):
        return {
            "confidence_level": "C0",
            "confidence_weight": 0.0,
            "hard_ce_allowed": False,
            "reasons": ["b_gate_failed"],
        }
    if not q(validation, "V0", "passed"):
        return {
            "confidence_level": "C0",
            "confidence_weight": 0.0,
            "hard_ce_allowed": False,
            "reasons": ["v0_grammar_failed"],
        }
    if not q(validation, "V1", "passed"):
        return {
            "confidence_level": "C0",
            "confidence_weight": 0.0,
            "hard_ce_allowed": False,
            "reasons": ["v1_skeleton_preservation_failed"],
        }
    v1_clean = str(q(validation, "V1", "status") or "") == "pass"
    v2 = bool(q(validation, "V2", "passed")) and str(q(validation, "V2", "status") or "") == "pass"
    v3 = q(validation, "V3", "passed")
    if v3:
        level = "C3"
    elif v1_clean and v2:
        level = "C2"
    else:
        level = "C1"
    if level == "C1":
        validator_reason = "v2_v3_failed_or_missing_so_candidate_is_prior_only"
    elif level == "C2":
        validator_reason = "v2_contract_compatible"
    else:
        validator_reason = "v3_symbolic_trace_or_differential_validated"
    return {
        "confidence_level": level,
        "confidence_weight": TIER_WEIGHT[level],
        "hard_ce_allowed": TIER_ORDER[level] >= TIER_ORDER["C3"],
        "reasons": [
            "v0_v1_passed",
            "v1_clean_pass" if v1_clean else "v1_partial_needs_successor_or_polarity_v3",
            validator_reason,
        ],
    }


def build_prompt(function_record: Dict[str, Any], slot: Dict[str, Any]) -> str:
    skeleton = function_record.get("verified_skeleton_text") or function_record.get("verified_skeleton_preview") or ""
    global_block = {
        "func_name": function_record.get("func_name"),
        "example": function_record.get("example"),
        "input_line": function_record.get("input_line"),
        "candidate_name": q(function_record, "module_b", "candidate_name"),
    }
    contract = {
        "slot_id": slot.get("slot_id"),
        "slot_type": slot.get("slot_type"),
        "semantic_slot_type": slot.get("semantic_slot_type"),
        "anchor": slot.get("anchor"),
        "region": slot.get("region"),
        "control_contract": slot.get("control_contract"),
        "data_contract": slot.get("data_contract"),
        "effect_contract": slot.get("effect_contract"),
        "grammar_contract": slot.get("grammar_contract"),
    }
    return "\n".join(
        [
            "<global>",
            json.dumps(global_block, ensure_ascii=False, sort_keys=True),
            "</global>",
            "<skeleton>",
            str(skeleton),
            "</skeleton>",
            "<slot_contract>",
            json.dumps(contract, ensure_ascii=False, sort_keys=True),
            "</slot_contract>",
            "<task>",
            "Return JSON with one fill: {\"fills\":[{\"slot_id\":\"...\",\"text\":\"...\"}]}",
            "</task>",
        ]
    )


def train_record(
    function_record: Dict[str, Any],
    slot: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    *,
    min_train_tier: str,
    include_c1_priors: bool,
) -> Optional[Dict[str, Any]]:
    min_rank = TIER_ORDER.get(min_train_tier, TIER_ORDER["C2"])
    refs = []
    priors = []
    for cand in candidates:
        conf = cand.get("confidence") or {}
        level = str(conf.get("confidence_level") or "C0")
        text = str((cand.get("c_renderings") or [{}])[0].get("text") or "")
        if not text:
            continue
        item = {
            "candidate_id": cand.get("candidate_id"),
            "text": text,
            "weight": safe_float(conf.get("confidence_weight")),
            "confidence_level": level,
            "provenance": cand.get("provenance"),
            "validation": cand.get("validation"),
        }
        if TIER_ORDER.get(level, 0) >= min_rank and safe_float(conf.get("confidence_weight")) > 0.0:
            refs.append(item)
        elif include_c1_priors and level == "C1":
            priors.append(item)
    if not refs and not priors:
        return None
    slot_id = str(slot.get("slot_id") or "")
    return {
        "schema": TRAIN_SCHEMA,
        "function_key": function_record.get("function_key"),
        "example": function_record.get("example"),
        "input_line": function_record.get("input_line"),
        "func_name": function_record.get("func_name"),
        "slot_id": slot_id,
        "slot_type": slot.get("slot_type"),
        "prompt": build_prompt(function_record, slot),
        "references": refs,
        "candidate_priors": priors,
        "sft_eligible": bool(refs),
        "completion_format": "slot_patch_json",
        "notes": [
            "Use references for confidence-weighted multi-reference SFT.",
            "C1 priors are retrieval/reranker features and must not be hard CE labels.",
        ],
    }


def decode_seed_record(function_record: Dict[str, Any], slot_candidate_sets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    table: Dict[str, List[Dict[str, Any]]] = {}
    for scs in slot_candidate_sets:
        slot = scs.get("slot") or {}
        slot_id = str(slot.get("slot_id") or "")
        table[slot_id] = [
            {
                "candidate_id": cand.get("candidate_id"),
                "text": str((cand.get("c_renderings") or [{}])[0].get("text") or ""),
                "confidence": cand.get("confidence"),
                "validation": cand.get("validation"),
                "provenance": cand.get("provenance"),
            }
            for cand in scs.get("candidates") or []
            if str((cand.get("c_renderings") or [{}])[0].get("text") or "")
        ]
    return {
        "schema": DECODE_SCHEMA,
        "function_key": function_record.get("function_key"),
        "example": function_record.get("example"),
        "input_line": function_record.get("input_line"),
        "func_name": function_record.get("func_name"),
        "skeleton": function_record.get("verified_skeleton_text") or function_record.get("verified_skeleton_preview"),
        "candidate_table": table,
        "global_rerank_features": [
            "compile_success",
            "skeleton_preservation",
            "cfg_edge_f1",
            "slot_contract_satisfaction",
            "live_out_equivalence",
            "effect_equivalence",
            "variable_naming_coherence",
            "length_penalty",
        ],
        "notes": [
            "This is a seed table for global beam/ILP/rerank; local top-1 is not final inference.",
        ],
    }


def process_function(
    function_record: Dict[str, Any],
    *,
    max_candidates_per_slot: int,
    emit_llm_requests: bool,
    min_train_tier: str,
    include_c1_priors: bool,
    emit_keep_external: bool,
    args: argparse.Namespace,
    external_v3_index: Dict[Tuple[Any, Any, str, str, str], Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    slot_sets = []
    train_records = []
    for slot in function_record.get("semantic_slots") or []:
        candidates = generate_candidates(
            slot,
            emit_llm_requests=emit_llm_requests,
            emit_keep_external=emit_keep_external,
        )
        validated: List[Dict[str, Any]] = []
        for cand in candidates:
            validation = validate_candidate(function_record, slot, cand, args, external_v3_index)
            cand["validation"] = validation
            cand["confidence"] = score_confidence(slot, cand, validation)
            validated.append(cand)
        validated.sort(
            key=lambda c: (
                TIER_ORDER.get(str(q(c, "confidence", "confidence_level") or "C0"), 0),
                safe_float(q(c, "matching", "base_alignment_score")),
                c.get("candidate_id") or "",
            ),
            reverse=True,
        )
        if max_candidates_per_slot > 0:
            validated = validated[:max_candidates_per_slot]
        slot_set = {
            "slot": slot,
            "candidates": validated,
            "candidate_count": len(validated),
            "tier_dist": dict(Counter(str(q(c, "confidence", "confidence_level") or "C0") for c in validated)),
        }
        slot_sets.append(slot_set)
        tr = train_record(
            function_record,
            slot,
            validated,
            min_train_tier=min_train_tier,
            include_c1_priors=include_c1_priors,
        )
        if tr is not None:
            train_records.append(tr)

    candidate_record = {
        "schema": CANDIDATE_SCHEMA,
        "function_key": function_record.get("function_key"),
        "example": function_record.get("example"),
        "input_line": function_record.get("input_line"),
        "func_name": function_record.get("func_name"),
        "module_b": function_record.get("module_b"),
        "b_gate": function_record.get("b_gate"),
        "verified_skeleton_text": function_record.get("verified_skeleton_text"),
        "verified_skeleton_preview": function_record.get("verified_skeleton_preview"),
        "slot_candidate_sets": slot_sets,
        "summary": {
            "slot_count": len(slot_sets),
            "candidate_count": sum(len(s.get("candidates") or []) for s in slot_sets),
            "keep_external_count": sum(
                1
                for s in slot_sets
                for c in (s.get("candidates") or [])
                if q(c, "keep", "status") == "KEEP_EXTERNAL"
            ),
            "sft_eligible_slot_count": sum(1 for r in train_records if r.get("sft_eligible")),
        },
    }
    return candidate_record, train_records, decode_seed_record(function_record, slot_sets)


def update_stats(stats: Dict[str, Any], candidate_record: Dict[str, Any], train_records: Sequence[Dict[str, Any]]) -> None:
    stats["functions_total"] += 1
    if q(candidate_record, "b_gate", "pass"):
        stats["functions_b_gate_pass"] += 1
    if q(candidate_record, "summary", "sft_eligible_slot_count"):
        stats["functions_with_sft_eligible_slot"] += 1
    stats["slot_count"] += safe_int(q(candidate_record, "summary", "slot_count"))
    stats["candidate_count"] += safe_int(q(candidate_record, "summary", "candidate_count"))
    stats["keep_external_count"] += safe_int(q(candidate_record, "summary", "keep_external_count"))
    stats["train_record_count"] += len(train_records)
    stats["sft_eligible_train_record_count"] += sum(1 for r in train_records if r.get("sft_eligible"))
    for slot_set in candidate_record.get("slot_candidate_sets") or []:
        slot = slot_set.get("slot") or {}
        slot_type = str(slot.get("slot_type") or "UNKNOWN")
        stats["slot_type_dist"][slot_type] += 1
        stats["module_b_slot_kind_dist"][str(slot.get("module_b_slot_kind") or "UNKNOWN")] += 1
        slot_has_concrete = False
        slot_has_v2_pass = False
        slot_has_v2_partial = False
        slot_has_v2_fail = False
        slot_has_empty_effect_candidate = False
        slot_has_return_constant_candidate = False
        slot_has_dataflow_probe_candidate = False
        slot_has_effect_probe_candidate = False
        for cand in slot_set.get("candidates") or []:
            level = str(q(cand, "confidence", "confidence_level") or "C0")
            stats["candidate_tier_dist"][level] += 1
            is_keep_external = q(cand, "keep", "status") == "KEEP_EXTERNAL"
            has_text = bool(str(candidate_text(cand) or "").strip())
            if is_keep_external:
                stats["keep_external_candidate_count"] += 1
            elif has_text:
                slot_has_concrete = True
                stats["concrete_candidate_count"] += 1
                stats["concrete_candidate_tier_dist"][level] += 1
            for src in q(cand, "provenance", "source") or []:
                stats["candidate_source_dist"][str(src)] += 1
            rewrite_kind = q(cand, "provenance", "rewrite_kind")
            if rewrite_kind:
                stats["rewrite_kind_dist"][str(rewrite_kind)] += 1
            if is_keep_external:
                stats["keep_external_source_dist"][",".join(q(cand, "provenance", "source") or ["unknown"])] += 1
            for key in ("V0", "V1", "V2", "V3"):
                status = str(q(cand, "validation", key, "status") or "unknown")
                stats[f"{key}_status_dist"][status] += 1
                if has_text and not is_keep_external:
                    stats[f"concrete_{key}_status_dist"][status] += 1
            if has_text and not is_keep_external:
                v2_status = str(q(cand, "validation", "V2", "status") or "unknown")
                if v2_status == "pass":
                    slot_has_v2_pass = True
                elif v2_status == "partial_needs_v3":
                    slot_has_v2_partial = True
                elif v2_status == "fail":
                    slot_has_v2_fail = True
                text = candidate_text(cand).strip()
                source = ",".join(str(x) for x in (q(cand, "provenance", "source") or []))
                summary = str(q(cand, "semantic_object", "summary") or "")
                if text == ";" and "binary_ir" in source:
                    slot_has_empty_effect_candidate = True
                if text.startswith("return ") and "return_constant" in summary:
                    slot_has_return_constant_candidate = True
                if text.startswith("ECSL_dataflow_probe("):
                    slot_has_dataflow_probe_candidate = True
                if text.startswith("ECSL_effect_probe("):
                    slot_has_effect_probe_candidate = True
            v2_reason = str(q(cand, "validation", "V2", "reason") or "")
            v3_reason = str(q(cand, "validation", "V3", "reason") or "")
            if v2_reason:
                stats["V2_reason_dist"][v2_reason.split(";")[0][:120]] += 1
            for warning in q(cand, "validation", "V2", "evidence", "warnings") or []:
                stats["V2_warning_dist"][str(warning)[:120]] += 1
            if v3_reason:
                stats["V3_reason_dist"][v3_reason.split(";")[0][:120]] += 1
        if slot_has_concrete:
            stats["concrete_candidate_slots_by_type"][slot_type] += 1
        if slot_has_v2_pass:
            stats["v2_pass_slots_by_type"][slot_type] += 1
        if slot_has_v2_partial:
            stats["v2_partial_slots_by_type"][slot_type] += 1
        if slot_has_v2_pass or slot_has_v2_partial:
            stats["v2_taskable_slots_by_type"][slot_type] += 1
            stats["v2_taskable_slot_count"] += 1
        if slot_has_v2_fail and not slot_has_v2_pass and not slot_has_v2_partial:
            stats["v2_fail_slots_by_type"][slot_type] += 1
        if slot_has_empty_effect_candidate:
            stats["empty_effect_candidate_slots_by_type"][slot_type] += 1
        if slot_has_return_constant_candidate:
            stats["return_constant_candidate_slots_by_type"][slot_type] += 1
        if slot_has_dataflow_probe_candidate:
            stats["dataflow_probe_candidate_slots_by_type"][slot_type] += 1
        if slot_has_effect_probe_candidate:
            stats["effect_probe_candidate_slots_by_type"][slot_type] += 1


def initial_stats() -> Dict[str, Any]:
    return {
        "functions_total": 0,
        "functions_b_gate_pass": 0,
        "functions_with_sft_eligible_slot": 0,
        "slot_count": 0,
        "candidate_count": 0,
        "concrete_candidate_count": 0,
        "keep_external_candidate_count": 0,
        "keep_external_count": 0,
        "train_record_count": 0,
        "sft_eligible_train_record_count": 0,
        "slot_type_dist": Counter(),
        "module_b_slot_kind_dist": Counter(),
        "concrete_candidate_slots_by_type": Counter(),
        "v2_pass_slots_by_type": Counter(),
        "v2_partial_slots_by_type": Counter(),
        "v2_taskable_slots_by_type": Counter(),
        "v2_taskable_slot_count": 0,
        "v2_fail_slots_by_type": Counter(),
        "empty_effect_candidate_slots_by_type": Counter(),
        "return_constant_candidate_slots_by_type": Counter(),
        "dataflow_probe_candidate_slots_by_type": Counter(),
        "effect_probe_candidate_slots_by_type": Counter(),
        "candidate_tier_dist": Counter(),
        "concrete_candidate_tier_dist": Counter(),
        "candidate_source_dist": Counter(),
        "keep_external_source_dist": Counter(),
        "rewrite_kind_dist": Counter(),
        "V0_status_dist": Counter(),
        "V1_status_dist": Counter(),
        "V2_status_dist": Counter(),
        "V3_status_dist": Counter(),
        "concrete_V0_status_dist": Counter(),
        "concrete_V1_status_dist": Counter(),
        "concrete_V2_status_dist": Counter(),
        "concrete_V3_status_dist": Counter(),
        "V2_reason_dist": Counter(),
        "V2_warning_dist": Counter(),
        "V3_reason_dist": Counter(),
    }


def write_report(path: Path, *, stats: Dict[str, Any], args: argparse.Namespace) -> None:
    n = stats["functions_total"]
    lines: List[str] = []
    lines.append("=== Module C ECSL Candidate / Validator / Training Builder ===")
    lines.append(f"contracts_jsonl = {args.contracts_jsonl}")
    lines.append(f"out_candidates_jsonl = {args.out_candidates_jsonl}")
    lines.append(f"out_train_jsonl = {args.out_train_jsonl}")
    lines.append(f"out_decode_jsonl = {args.out_decode_jsonl}")
    lines.append("")
    lines.append(f"functions_total = {n}")
    lines.append(f"functions_b_gate_pass = {pct(stats['functions_b_gate_pass'], n)}")
    lines.append(f"functions_with_sft_eligible_slot = {stats['functions_with_sft_eligible_slot']}")
    lines.append(f"slot_count = {stats['slot_count']}")
    lines.append(f"candidate_count = {stats['candidate_count']}")
    lines.append(f"concrete_candidate_count = {stats['concrete_candidate_count']}")
    lines.append(f"keep_external_candidate_count = {stats['keep_external_candidate_count']}")
    lines.append(f"keep_external_count = {stats['keep_external_count']}")
    lines.append(f"train_record_count = {stats['train_record_count']}")
    lines.append(f"sft_eligible_train_record_count = {stats['sft_eligible_train_record_count']}")
    lines.append("")
    lines.append(f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}")
    lines.append(f"module_b_slot_kind_dist = {compact_counter(stats['module_b_slot_kind_dist'])}")
    lines.append(f"concrete_candidate_slots_by_type = {compact_counter(stats['concrete_candidate_slots_by_type'])}")
    lines.append(f"v2_pass_slots_by_type = {compact_counter(stats['v2_pass_slots_by_type'])}")
    lines.append(f"v2_partial_slots_by_type = {compact_counter(stats['v2_partial_slots_by_type'])}")
    lines.append(f"v2_taskable_slot_count = {pct(stats['v2_taskable_slot_count'], stats['slot_count'])}")
    lines.append(f"v2_taskable_slots_by_type = {compact_counter(stats['v2_taskable_slots_by_type'])}")
    lines.append(f"v2_fail_slots_by_type = {compact_counter(stats['v2_fail_slots_by_type'])}")
    lines.append(f"empty_effect_candidate_slots_by_type = {compact_counter(stats['empty_effect_candidate_slots_by_type'])}")
    lines.append(f"return_constant_candidate_slots_by_type = {compact_counter(stats['return_constant_candidate_slots_by_type'])}")
    lines.append(f"dataflow_probe_candidate_slots_by_type = {compact_counter(stats['dataflow_probe_candidate_slots_by_type'])}")
    lines.append(f"effect_probe_candidate_slots_by_type = {compact_counter(stats['effect_probe_candidate_slots_by_type'])}")
    lines.append(f"candidate_tier_dist = {compact_counter(stats['candidate_tier_dist'])}")
    lines.append(f"concrete_candidate_tier_dist = {compact_counter(stats['concrete_candidate_tier_dist'])}")
    lines.append(f"candidate_source_dist = {compact_counter(stats['candidate_source_dist'])}")
    lines.append(f"keep_external_source_dist = {compact_counter(stats['keep_external_source_dist'])}")
    lines.append(f"rewrite_kind_dist = {compact_counter(stats['rewrite_kind_dist'])}")
    lines.append(f"V0_status_dist = {compact_counter(stats['V0_status_dist'])}")
    lines.append(f"V1_status_dist = {compact_counter(stats['V1_status_dist'])}")
    lines.append(f"V2_status_dist = {compact_counter(stats['V2_status_dist'])}")
    lines.append(f"V3_status_dist = {compact_counter(stats['V3_status_dist'])}")
    lines.append(f"concrete_V0_status_dist = {compact_counter(stats['concrete_V0_status_dist'])}")
    lines.append(f"concrete_V1_status_dist = {compact_counter(stats['concrete_V1_status_dist'])}")
    lines.append(f"concrete_V2_status_dist = {compact_counter(stats['concrete_V2_status_dist'])}")
    lines.append(f"concrete_V3_status_dist = {compact_counter(stats['concrete_V3_status_dist'])}")
    lines.append(f"V2_reason_dist = {compact_counter(stats['V2_reason_dist'])}")
    lines.append(f"V2_warning_dist = {compact_counter(stats['V2_warning_dist'])}")
    lines.append(f"V3_reason_dist = {compact_counter(stats['V3_reason_dist'])}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- V0/V1 passing means grammar and skeleton preservation only.")
    lines.append("- V2 clean pass is a slot-local compatibility gate; missing static evidence is reported as partial_needs_v3 and is not promoted to C2/C3.")
    lines.append("- V3 accepts external symbolic/trace/differential C3+ results; optional compile smoke is diagnostic only.")
    lines.append("- Therefore C2 means contract-compatible weak evidence; C3+ means hard SFT eligibility.")
    lines.append("- SFT should use records with references; candidate_priors are retrieval/rerank features only.")
    lines.append("- KEEP_EXTERNAL entries are algorithm-required hooks for source CFG/SSA, traces, LLM proposals, self-training, and rewrite generators; they are not labels.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> Dict[str, Any]:
    stats = initial_stats()
    external_v3_index = load_external_v3(args.external_v3_jsonl)
    out_candidates = Path(args.out_candidates_jsonl)
    out_train = Path(args.out_train_jsonl)
    out_decode = Path(args.out_decode_jsonl)
    out_report = Path(args.out_txt)
    for path in (out_candidates, out_train, out_decode, out_report):
        path.parent.mkdir(parents=True, exist_ok=True)

    with out_candidates.open("w", encoding="utf-8") as cand_f, out_train.open(
        "w", encoding="utf-8"
    ) as train_f, out_decode.open("w", encoding="utf-8") as decode_f:
        for _line_no, function_record in iter_jsonl(Path(args.contracts_jsonl), limit=int(args.limit)):
            candidate_record, train_records, decode_record = process_function(
                function_record,
                max_candidates_per_slot=int(args.max_candidates_per_slot),
                emit_llm_requests=bool(args.emit_llm_proposal_requests),
                min_train_tier=str(args.min_train_tier),
                include_c1_priors=bool(args.include_c1_priors),
                emit_keep_external=bool(args.emit_keep_external),
                args=args,
                external_v3_index=external_v3_index,
            )
            cand_f.write(json.dumps(candidate_record, ensure_ascii=False, sort_keys=True) + "\n")
            for rec in train_records:
                train_f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            decode_f.write(json.dumps(decode_record, ensure_ascii=False, sort_keys=True) + "\n")
            update_stats(stats, candidate_record, train_records)

    write_report(out_report, stats=stats, args=args)
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ECSL candidates and training records from contract JSONL.")
    ap.add_argument("--contracts-jsonl", required=True)
    ap.add_argument("--out-candidates-jsonl", required=True)
    ap.add_argument("--out-train-jsonl", required=True)
    ap.add_argument("--out-decode-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates-per-slot", type=int, default=8)
    ap.add_argument("--min-train-tier", default="C2", choices=["C1", "C2", "C3", "C4", "C5"])
    ap.add_argument(
        "--include-c1-priors",
        action="store_true",
        help="Write C1 priors into training records as candidate_priors, never as hard references.",
    )
    ap.add_argument(
        "--emit-llm-proposal-requests",
        action="store_true",
        help="Emit LLM proposal request objects for later generation; they are C0 until validated.",
    )
    ap.add_argument(
        "--emit-keep-external",
        action="store_true",
        help="Emit KEEP_EXTERNAL records for source CFG/SSA, traces, rewrite requests, and self-training hooks.",
    )
    ap.add_argument(
        "--enable-v2-validator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the local conservative V2 SSA/control/effect contract validator.",
    )
    ap.add_argument(
        "--v0-cc",
        default="cc",
        help="Compiler command for V0 local parser/type stub validation. If unavailable, V0 falls back to grammar-only.",
    )
    ap.add_argument(
        "--v2-require-full-dataflow",
        action="store_true",
        help="Require populated live-in/live-out/defs/uses/effect facts for V2 pass; useful for strict audits.",
    )
    ap.add_argument(
        "--external-v3-jsonl",
        default=None,
        help="External C3+ symbolic/trace/differential validation JSONL to promote matching candidates to V3.",
    )
    ap.add_argument(
        "--v3-command",
        default=None,
        help=(
            "Optional shell command for a real V3 validator. It receives ECSL_* env vars and must print JSON with "
            "differential_pass, trace_match, or symbolic_match plus confidence_level C3+."
        ),
    )
    ap.add_argument(
        "--enable-v3-compile-smoke",
        action="store_true",
        help="Run a diagnostic full-function render + C syntax compile smoke for generated candidates.",
    )
    ap.add_argument("--v3-cc", default="cc", help="Compiler command for --enable-v3-compile-smoke.")
    ap.add_argument(
        "--v3-compile-keep-dir",
        default=None,
        help="Optional directory to keep compile-smoke rendered C files for inspection.",
    )
    ap.add_argument(
        "--allow-compile-smoke-as-v3",
        action="store_true",
        help="Deprecated compatibility flag; compile smoke is diagnostic only and never promotes to V3/C3.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stats = build(args)
    print(f"functions_total = {stats['functions_total']}")
    print(f"functions_b_gate_pass = {pct(stats['functions_b_gate_pass'], stats['functions_total'])}")
    print(f"slot_count = {stats['slot_count']}")
    print(f"candidate_count = {stats['candidate_count']}")
    print(f"keep_external_count = {stats['keep_external_count']}")
    print(f"candidate_tier_dist = {compact_counter(stats['candidate_tier_dist'])}")
    print(f"train_record_count = {stats['train_record_count']}")
    print(f"sft_eligible_train_record_count = {stats['sft_eligible_train_record_count']}")
    print(f"wrote_candidates = {args.out_candidates_jsonl}")
    print(f"wrote_train = {args.out_train_jsonl}")
    print(f"wrote_decode = {args.out_decode_jsonl}")
    print(f"wrote_report = {args.out_txt}")


if __name__ == "__main__":
    main()

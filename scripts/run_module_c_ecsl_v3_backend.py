#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the built-in Module-C ECSL V3 validator.

This backend consumes ``ecsl_v3_validation_tasks.jsonl`` and writes V3 result
rows that ``build_module_c_ecsl_strong_sft.py`` can consume directly.  It uses
the training source-teacher projection as the reference oracle and performs:

* compiled differential tests for supported boolean/integer expressions,
  returns, and simple assignments;
* symbolic structural equality for simple control transfers such as
  break/continue/goto.

It is intentionally strict: source projection by itself is never a pass, compile
smoke by itself is never a pass, and unsupported shapes are written to the audit
file instead of being promoted.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_v3_lite() -> Any:
    path = SCRIPT_DIR / "run_module_c_ecsl_v3_differential_lite.py"
    spec = importlib.util.spec_from_file_location("_ecsl_v3_lite_backend_helpers", path)
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


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def compact_counter(counter: collections.Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def function_keys(row: Dict[str, Any]) -> List[Tuple[Any, Any, str]]:
    func = str(row.get("func_name") or "")
    return [
        (row.get("function_key"), None, ""),
        (row.get("example"), row.get("input_line"), func),
        (None, row.get("input_line"), func),
    ]


def canonical_slot_id(value: Any) -> str:
    text = str(value or "").strip()
    try:
        import re

        m = re.search(r"slot_(\d+)", text)
        if m:
            return f"slot_{int(m.group(1)):03d}"
    except Exception:
        pass
    return text


def load_projection_index(path: Optional[str]) -> Dict[Tuple[Any, Any, str], Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(Path(path).expanduser().resolve()):
        for key in function_keys(row):
            out[key] = row
    return out


def lookup_projection(index: Dict[Tuple[Any, Any, str], Dict[str, Any]], task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in function_keys(task):
        if key in index:
            return index[key]
    return None


def projection_reference_text(projection: Optional[Dict[str, Any]], slot_id: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    if not isinstance(projection, dict):
        return "", None, "source_projection_not_found"
    want = canonical_slot_id(slot_id)
    for item in projection.get("slot_v3_results") or []:
        if not isinstance(item, dict):
            continue
        if canonical_slot_id(item.get("slot_id")) != want:
            continue
        text = str(item.get("target_text") or "").strip()
        if not text:
            return "", item, "source_projection_slot_empty_target"
        if not (strict_true(item.get("source_teacher_passed")) or strict_true(item.get("passed"))):
            return "", item, "source_projection_slot_not_passed"
        return text, item, "ok"
    return "", None, "source_projection_slot_not_found"


def is_source_projection_task(task: Dict[str, Any]) -> bool:
    if bool(task.get("candidate_is_source_projection")):
        return True
    prov = task.get("candidate_provenance") if isinstance(task.get("candidate_provenance"), dict) else {}
    if str(prov.get("oracle_type") or "") == "training_source_teacher_slot_projection":
        return True
    return "source_teacher_projection" in {str(x) for x in (prov.get("source") or [])}


def task_module_b_kind(task: Dict[str, Any]) -> str:
    slot_contract = task.get("slot_contract") if isinstance(task.get("slot_contract"), dict) else {}
    grammar = slot_contract.get("grammar_contract") if isinstance(slot_contract.get("grammar_contract"), dict) else {}
    return str(
        task.get("module_b_slot_kind")
        or q(task, "slot", "module_b_slot_kind")
        or grammar.get("module_b_slot_kind")
        or task.get("slot_type")
        or ""
    )


def task_slot_contract(task: Dict[str, Any]) -> Dict[str, Any]:
    return task.get("slot_contract") if isinstance(task.get("slot_contract"), dict) else {}


def symbols_from_contract_value(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
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


def sanitize_identifier(value: Any, *, prefix: str = "v") -> str:
    text = str(value or "").strip()
    text = re.sub(r"^%+", "", text)
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = prefix
    if text[0].isdigit():
        text = f"{prefix}_{text}"
    return text


def contract_symbol_aliases(value: Any, *, prefix: str = "ssa") -> List[str]:
    out: List[str] = []
    seen = set()
    for item in symbols_from_contract_value(value):
        variants = [item, sanitize_identifier(item, prefix=prefix)]
        text = str(item or "").strip()
        for sep in ("@", ".", "#"):
            if sep in text:
                head = text.split(sep, 1)[0].lstrip("%")
                if head:
                    variants.extend([head, sanitize_identifier(head, prefix=prefix)])
        if text.startswith("%"):
            variants.append(text.lstrip("%"))
        for variant in variants:
            v = str(variant or "").strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def condition_ssa_from_task(task: Dict[str, Any]) -> Any:
    contract = task_slot_contract(task)
    data = contract.get("data_contract") if isinstance(contract.get("data_contract"), dict) else {}
    observable = contract.get("observable_semantics") if isinstance(contract.get("observable_semantics"), dict) else {}
    return (
        q(data, "ssa_grounding", "condition_ssa")
        or q(data, "condition_ssa")
        or q(data, "branch_condition_ssa")
        or q(data, "predicate_ssa")
        or q(observable, "condition_ssa")
        or q(observable, "branch_condition_ssa")
    )


def return_ssa_from_task(task: Dict[str, Any]) -> Any:
    contract = task_slot_contract(task)
    data = contract.get("data_contract") if isinstance(contract.get("data_contract"), dict) else {}
    control = contract.get("control_contract") if isinstance(contract.get("control_contract"), dict) else {}
    effect = contract.get("effect_contract") if isinstance(contract.get("effect_contract"), dict) else {}
    observable = contract.get("observable_semantics") if isinstance(contract.get("observable_semantics"), dict) else {}
    return (
        q(data, "ssa_grounding", "return_ssa")
        or q(data, "return_ssa")
        or q(data, "return_value_ssa")
        or q(control, "return_ssa")
        or q(control, "return_value_ssa")
        or q(effect, "return_ssa")
        or q(observable, "return_ssa")
        or q(observable, "return_value_ssa")
    )


def return_constant_from_task(task: Dict[str, Any]) -> Any:
    contract = task_slot_contract(task)
    data = contract.get("data_contract") if isinstance(contract.get("data_contract"), dict) else {}
    control = contract.get("control_contract") if isinstance(contract.get("control_contract"), dict) else {}
    effect = contract.get("effect_contract") if isinstance(contract.get("effect_contract"), dict) else {}
    observable = contract.get("observable_semantics") if isinstance(contract.get("observable_semantics"), dict) else {}
    return (
        q(data, "ssa_grounding", "return_constant")
        or q(data, "ssa_grounding", "return_value_constant")
        or q(data, "return_constant")
        or q(data, "return_value_constant")
        or q(control, "return_constant")
        or q(control, "return_value_constant")
        or q(effect, "return_constant")
        or q(observable, "return_constant")
        or q(observable, "return_value_constant")
    )


def condition_provenance_from_task(task: Dict[str, Any]) -> Dict[str, Any]:
    contract = task_slot_contract(task)
    data = contract.get("data_contract") if isinstance(contract.get("data_contract"), dict) else {}
    grounding = data.get("ssa_grounding") if isinstance(data.get("ssa_grounding"), dict) else {}
    prov = grounding.get("condition_provenance")
    if isinstance(prov, dict):
        return prov
    out: Dict[str, Any] = {}
    ssa_def = grounding.get("condition_ssa_def")
    if isinstance(ssa_def, dict):
        out["ssa_def"] = ssa_def
    branch = grounding.get("condition_branch_instruction")
    if isinstance(branch, dict):
        out["branch_instruction_record"] = branch
        out["branch_text"] = branch.get("text")
        out["branch_mnemonic"] = branch.get("mnemonic")
    return out


def contract_condition_provenance_verified(task: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    prov = condition_provenance_from_task(task)
    ssa_def = prov.get("ssa_def") if isinstance(prov.get("ssa_def"), dict) else {}
    branch = prov.get("branch_instruction_record") if isinstance(prov.get("branch_instruction_record"), dict) else {}
    def_text = str(ssa_def.get("def_text") or "")
    branch_text = str(prov.get("branch_text") or branch.get("text") or "")
    cond = condition_ssa_from_task(task)
    control = q(task_slot_contract(task), "control_contract") or {}
    true_succ = control.get("true_successor") if isinstance(control, dict) else None
    false_succ = control.get("false_successor") if isinstance(control, dict) else None
    evidence = {
        "condition_provenance": prov,
        "condition_def_status": ssa_def.get("status"),
        "condition_def_text": def_text or None,
        "condition_def_instruction": ssa_def.get("defined_by_instruction"),
        "branch_text": branch_text or None,
        "branch_instruction": prov.get("branch_instruction") or branch.get("instruction_id"),
        "true_successor": true_succ,
        "false_successor": false_succ,
        "true_false_polarity_available": bool(prov.get("true_false_polarity_available")),
    }
    if not cond:
        return False, "missing_condition_ssa", evidence
    if ssa_def.get("status") != "available" or not def_text:
        return False, "missing_condition_ssa_def_instruction", evidence
    if not branch_text:
        return False, "missing_branch_instruction", evidence
    if not (true_succ and false_succ):
        return False, "missing_true_false_successors", evidence
    if not bool(prov.get("true_false_polarity_available")):
        return False, "missing_true_false_polarity", evidence
    return True, "condition_ssa_def_and_branch_provenance_available", evidence


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", str(text or ""), flags=re.S)
    text = re.sub(r"//.*", " ", text)
    return text.strip()


COMMON_C_CONSTANTS = {
    "NULL": "0",
    "true": "1",
    "false": "0",
    "TRUE": "1",
    "FALSE": "0",
    "EXIT_SUCCESS": "0",
    "EXIT_FAILURE": "1",
}

C_TYPE_WORDS = {
    "_Bool",
    "bool",
    "char",
    "const",
    "double",
    "enum",
    "float",
    "int",
    "long",
    "short",
    "signed",
    "size_t",
    "ssize_t",
    "static",
    "struct",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "uintptr_t",
    "unsigned",
    "void",
    "volatile",
}

C_KEYWORDS = {
    "break",
    "case",
    "continue",
    "default",
    "do",
    "else",
    "for",
    "goto",
    "if",
    "return",
    "sizeof",
    "switch",
    "while",
    *C_TYPE_WORDS,
}


def strip_balanced_outer_parens(text: str) -> str:
    text = str(text or "").strip()
    while text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        if not inner:
            break
        depth = 0
        encloses_whole = True
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    encloses_whole = False
                    break
            if depth == 0 and i < len(text) - 1:
                encloses_whole = False
                break
        if not encloses_whole:
            break
        text = inner
    return text


def strip_simple_c_casts(text: str) -> str:
    """Drop simple scalar casts such as ``(int) x`` for symbolic comparison."""
    type_pat = r"(?:const\s+|volatile\s+|signed\s+|unsigned\s+|struct\s+\w+\s+)*(?:_Bool|bool|char|short|int|long|float|double|size_t|ssize_t|u?int(?:8|16|32|64)_t|uintptr_t|void)(?:\s*\*)*"
    old = None
    out = str(text or "")
    while old != out:
        old = out
        out = re.sub(rf"\(\s*{type_pat}\s*\)", "", out)
    return out


def normalize_c_literals(text: str) -> str:
    out = str(text or "")
    for name, value in COMMON_C_CONSTANTS.items():
        out = re.sub(rf"\b{re.escape(name)}\b", value, out)
    # Integer suffixes are irrelevant for these conservative text proofs.
    out = re.sub(r"\b(0[xX][0-9A-Fa-f]+|[0-9]+)(?:[uUlL]+)\b", r"\1", out)
    return out


def strip_redundant_identifier_parens(text: str) -> str:
    out = str(text or "")
    old = None
    atom = r"(?:[A-Za-z_][A-Za-z0-9_]*|\d+)"
    while old != out:
        old = out
        # Do not rewrite function-call arguments: ``foo(x)`` must remain a
        # call, not become ``foox``.  This only removes parens around atoms at
        # expression boundaries or after operators.
        out = re.sub(rf"(^|[!~+\-*/%<>=&|?:,\[])\(\s*({atom})\s*\)", r"\1\2", out)
    return out


def canonical_c_text(text: str) -> str:
    text = strip_c_comments(text)
    text = text.strip().rstrip(";").strip()
    m = re.match(r"^(return)\s+(.+)$", text)
    if m:
        return "return " + canonical_c_text(m.group(2))
    m = re.match(r"^(if|while)\s*\((.*)\)$", text)
    if m:
        return str(m.group(1)) + "(" + canonical_c_text(m.group(2)) + ")"
    text = strip_simple_c_casts(text)
    text = normalize_c_literals(text)
    text = strip_redundant_identifier_parens(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*(->|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--)\s*", r"\1", text)
    text = re.sub(r"\s*([()[\]{}.,;,=+\-*/%<>!&|?:])\s*", r"\1", text)
    text = strip_balanced_outer_parens(text)
    return text


def alpha_canonical_c_text(text: str) -> str:
    """Canonicalize identifiers by occurrence while preserving calls/fields.

    This is intentionally narrower than arbitrary text equality: operators,
    call names, field names, and statement keywords must still match.  It only
    abstracts local variable names, which are expected to differ between binary
    SSA-derived candidates and source-level references.
    """
    text = canonical_c_text(text)
    mapping: Dict[str, str] = {}
    parts: List[str] = []
    pos = 0
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text):
        ident = match.group(0)
        start, end = match.span()
        parts.append(text[pos:start])
        next_nonspace = ""
        j = end
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text):
            next_nonspace = text[j]
        prev = text[max(0, start - 2):start]
        if ident in C_KEYWORDS or next_nonspace == "(" or prev in {".", "->"}:
            parts.append(ident)
        elif re.match(r"^[A-Z][A-Z0-9_]*$", ident):
            parts.append(ident)
        else:
            if ident not in mapping:
                mapping[ident] = f"v{len(mapping)}"
            parts.append(mapping[ident])
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def alpha_symbolic_safe(candidate_canonical: str, reference_canonical: str) -> bool:
    """Guard alpha text matching against trivial single-variable self-proof.

    ``x`` vs ``y`` is too weak as a symbolic V3 proof without an SSA/source
    binding.  Structured expressions such as ``x + 1`` vs ``y + 1`` or
    ``foo(x)`` vs ``foo(y)`` still carry operator/call/field shape and are
    useful conservative symbolic evidence.
    """
    if not candidate_canonical or not reference_canonical:
        return False
    ident_only = r"[A-Za-z_][A-Za-z0-9_]*"
    if re.fullmatch(ident_only, candidate_canonical) and re.fullmatch(ident_only, reference_canonical):
        return False
    structural_needles = ("(", ")", "[", "]", "->", ".", "+", "-", "*", "/", "%", "==", "!=", "<", ">", "&&", "||", "&", "|", "^", "!", "return ")
    return any(token in candidate_canonical for token in structural_needles) and any(
        token in reference_canonical for token in structural_needles
    )


def symbolic_text_equal(
    candidate_text: str,
    reference_text: str,
    *,
    allow_alpha: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    cand = canonical_c_text(candidate_text)
    ref = canonical_c_text(reference_text)
    if bool(cand and cand == ref):
        return True, {
            "proof_kind": "canonical_symbolic_text_equal",
            "candidate_canonical": cand,
            "reference_canonical": ref,
        }
    cand_alpha = alpha_canonical_c_text(candidate_text) if allow_alpha else ""
    ref_alpha = alpha_canonical_c_text(reference_text) if allow_alpha else ""
    ok = bool(allow_alpha and alpha_symbolic_safe(cand, ref) and cand_alpha and cand_alpha == ref_alpha)
    return ok, {
        "proof_kind": "alpha_symbolic_text_equal" if ok else "canonical_symbolic_text_mismatch",
        "candidate_canonical": cand,
        "reference_canonical": ref,
        "candidate_alpha_canonical": cand_alpha,
        "reference_alpha_canonical": ref_alpha,
    }


def symbolic_pass_result(
    candidate_text: str,
    reference_text: str,
    *,
    allow_alpha: bool,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok, evidence = symbolic_text_equal(
        candidate_text,
        reference_text,
        allow_alpha=allow_alpha,
    )
    if not ok:
        return False, "v3_backend_symbolic_text_mismatch", evidence
    proof_kind = str(evidence.get("proof_kind") or "canonical_symbolic_text_equal")
    reason = (
        "v3_backend_alpha_symbolic_text_equal_pass"
        if proof_kind == "alpha_symbolic_text_equal"
        else "v3_backend_canonical_symbolic_text_equal_pass"
    )
    evidence = {
        **evidence,
        "mode": proof_kind,
        "candidate_text": candidate_text,
        "reference_text": reference_text,
    }
    return True, reason, evidence


def candidate_provenance_source(task: Dict[str, Any]) -> str:
    prov = task.get("candidate_provenance") if isinstance(task.get("candidate_provenance"), dict) else {}
    return ",".join(str(x) for x in (prov.get("source") or []))


def candidate_rewrite_kind(task: Dict[str, Any]) -> str:
    prov = task.get("candidate_provenance") if isinstance(task.get("candidate_provenance"), dict) else {}
    return str(prov.get("rewrite_kind") or "")


def contract_has_known_empty_real_facts(contract: Dict[str, Any]) -> bool:
    fact_state = str(contract.get("fact_state") or "").lower()
    status = str(contract.get("status") or "").lower()
    return fact_state == "proven_empty_real" or status == "available_empty_from_real_function_ir"


def empty_effect_contract_result(task: Dict[str, Any], candidate_text: str) -> Tuple[bool, str, Dict[str, Any]]:
    slot_type = str(task.get("slot_type") or "").upper()
    module_b_kind = task_module_b_kind(task).upper()
    source = candidate_provenance_source(task)
    contract = task_slot_contract(task)
    data = contract.get("data_contract") if isinstance(contract.get("data_contract"), dict) else {}
    effect = contract.get("effect_contract") if isinstance(contract.get("effect_contract"), dict) else {}
    text = canonical_c_text(candidate_text)
    evidence: Dict[str, Any] = {
        "proof_kind": "contract_empty_effect_symbolic",
        "candidate_text": candidate_text,
        "candidate_canonical": text,
        "candidate_source": source,
        "slot_type": slot_type,
        "module_b_slot_kind": module_b_kind,
        "data_status": data.get("status"),
        "data_fact_state": data.get("fact_state"),
        "effect_status": effect.get("status"),
        "effect_fact_state": effect.get("fact_state"),
    }
    if "binary_ir" not in source:
        evidence["reject_reason"] = "candidate_not_from_binary_ir"
        return False, "v3_backend_empty_effect_reject:not_binary_ir", evidence
    if slot_type not in {"BLOCK", "STMT", "CASE_LABEL"} and module_b_kind not in {"BLOCK", "STMT", "CASE"}:
        evidence["reject_reason"] = "slot_type_not_empty_effect_statement"
        return False, "v3_backend_empty_effect_reject:unsupported_slot", evidence
    if text not in {"", ";"}:
        evidence["reject_reason"] = "candidate_not_empty_statement"
        return False, "v3_backend_empty_effect_reject:not_empty_statement", evidence
    if contract_has_known_empty_real_facts(data) and contract_has_known_empty_real_facts(effect):
        return True, "v3_backend_contract_empty_effect_symbolic_pass", evidence
    evidence["reject_reason"] = "data_or_effect_contract_not_proven_empty_real"
    return False, "v3_backend_empty_effect_reject:not_proven_empty", evidence


def control_contract_result(task: Dict[str, Any], candidate_text: str) -> Tuple[bool, str, Dict[str, Any]]:
    slot_type = str(task.get("slot_type") or "").upper()
    module_b_kind = task_module_b_kind(task).upper()
    contract = task_slot_contract(task)
    control = contract.get("control_contract") if isinstance(contract.get("control_contract"), dict) else {}
    text = canonical_c_text(candidate_text)
    evidence: Dict[str, Any] = {
        "proof_kind": "control_contract_symbolic",
        "candidate_text": candidate_text,
        "slot_type": slot_type,
        "module_b_slot_kind": module_b_kind,
        "control_contract": {
            "edge": control.get("edge"),
            "break_target": control.get("break_target"),
            "continue_target": control.get("continue_target"),
            "allowed_control_effects": control.get("allowed_control_effects"),
        },
    }
    allowed = {str(x) for x in (control.get("allowed_control_effects") or [])}
    if module_b_kind == "BREAK" or slot_type == "LOOP_EXIT" and text == "break":
        if text == "break" and "break" in allowed:
            return True, "v3_backend_control_contract_break_pass", evidence
        evidence["reject_reason"] = "break_text_or_allowed_effect_mismatch"
        return False, "v3_backend_control_contract_reject:break_mismatch", evidence
    if module_b_kind == "CONTINUE":
        if text == "continue" and "continue" in allowed:
            return True, "v3_backend_control_contract_continue_pass", evidence
        evidence["reject_reason"] = "continue_text_or_allowed_effect_mismatch"
        return False, "v3_backend_control_contract_reject:continue_mismatch", evidence
    if module_b_kind in {"GOTO", "CLEANUP_EXIT"} or slot_type == "EXIT":
        edge = control.get("edge")
        target = None
        if isinstance(edge, list) and len(edge) >= 2:
            target = str(edge[1])
        labels = {f"label_BB_{target}", f"label_{target}", f"BB_{target}"} if target else set()
        cleanup_ok = module_b_kind == "CLEANUP_EXIT" and text == "goto cleanup" and "goto_like_internal" in allowed
        target_ok = text.startswith("goto ") and "goto_like_internal" in allowed and (not labels or any(label in candidate_text for label in labels))
        if cleanup_ok or target_ok:
            return True, "v3_backend_control_contract_goto_pass", evidence
        evidence["reject_reason"] = "goto_text_or_target_mismatch"
        evidence["expected_labels"] = sorted(labels)
        return False, "v3_backend_control_contract_reject:goto_mismatch", evidence
    evidence["reject_reason"] = "unsupported_control_contract_slot"
    return False, "v3_backend_control_contract_reject:unsupported_slot", evidence


def references_any_alias(text: str, aliases: List[str]) -> bool:
    canon = canonical_c_text(text)
    for alias in aliases:
        raw = str(alias or "").strip()
        if not raw:
            continue
        raw_canon = canonical_c_text(raw)
        if re.search(rf"\b{re.escape(raw)}\b", str(text or "")):
            return True
        if raw_canon and re.search(rf"\b{re.escape(raw_canon)}\b", canon):
            return True
    return False


def return_constant_contract_result(task: Dict[str, Any], candidate_text: str) -> Tuple[bool, str, Dict[str, Any]]:
    slot_type = str(task.get("slot_type") or "").upper()
    source = candidate_provenance_source(task)
    expected = return_constant_from_task(task)
    cand = canonical_c_text(candidate_text)
    m = re.match(r"^return\s+([-+]?(?:0x[0-9a-fA-F]+|\d+))$", cand)
    evidence: Dict[str, Any] = {
        "proof_kind": "contract_return_constant_symbolic",
        "slot_type": slot_type,
        "candidate_text": candidate_text,
        "candidate_canonical": cand,
        "candidate_source": source,
        "expected_return_constant": expected,
    }
    if "binary_ir" not in source:
        evidence["reject_reason"] = "candidate_not_from_binary_ir"
        return False, "v3_backend_return_constant_reject:not_binary_ir", evidence
    if slot_type not in {"RETURN", "STMT", "BLOCK", "EXIT"}:
        evidence["reject_reason"] = "slot_type_not_return_capable"
        return False, "v3_backend_return_constant_reject:unsupported_slot", evidence
    if expected is None:
        evidence["reject_reason"] = "missing_return_constant_contract"
        return False, "v3_backend_return_constant_reject:missing_contract", evidence
    if not m:
        evidence["reject_reason"] = "candidate_not_return_constant"
        return False, "v3_backend_return_constant_reject:not_return_constant", evidence
    try:
        cand_value = int(m.group(1), 0)
        expected_value = int(str(expected), 0)
    except Exception:
        cand_value = str(m.group(1))
        expected_value = str(expected)
    evidence["candidate_return_constant"] = cand_value
    evidence["expected_return_constant_normalized"] = expected_value
    if cand_value == expected_value:
        return True, "v3_backend_contract_return_constant_symbolic_pass", evidence
    evidence["reject_reason"] = "return_constant_mismatch"
    return False, "v3_backend_return_constant_reject:mismatch", evidence


def contract_backed_symbolic_result(
    task: Dict[str, Any],
    candidate_text: str,
    reference_text: str,
    *,
    require_provenance: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Use slot contracts as the symbolic bridge required by the ECSL design.

    This accepts only direct binary-IR candidates grounded in the slot's own
    condition/return SSA.  It does not accept source-projection self-proof,
    branch-inversion templates, or arbitrary single-variable alpha matches.
    """
    slot_type = str(task.get("slot_type") or "").upper()
    source = candidate_provenance_source(task)
    rewrite_kind = candidate_rewrite_kind(task)
    evidence: Dict[str, Any] = {
        "proof_kind": "contract_backed_symbolic",
        "candidate_text": candidate_text,
        "reference_text": reference_text,
        "candidate_source": source,
        "rewrite_kind": rewrite_kind,
        "slot_type": slot_type,
    }
    if "binary_ir" not in source:
        evidence["reject_reason"] = "candidate_not_from_binary_ir"
        return False, "v3_backend_contract_symbolic_reject:not_binary_ir", evidence
    if rewrite_kind:
        evidence["reject_reason"] = "rewrite_candidate_requires_explicit_polarity_or_trace"
        return False, "v3_backend_contract_symbolic_reject:rewrite_needs_v3", evidence
    if slot_type == "COND":
        cond = condition_ssa_from_task(task)
        aliases = contract_symbol_aliases(cond, prefix="ssa")
        evidence["condition_ssa"] = cond
        evidence["aliases"] = aliases[:32]
        prov_ok, prov_reason, prov_evidence = contract_condition_provenance_verified(task)
        evidence.update(prov_evidence)
        evidence["provenance_check"] = prov_reason
        if cond and references_any_alias(candidate_text, aliases):
            if require_provenance and not prov_ok:
                evidence["proof_kind"] = "contract_backed_symbolic_needs_audit"
                evidence["reject_reason"] = prov_reason
                return False, f"v3_backend_contract_symbolic_reject:{prov_reason}", evidence
            evidence["proof_kind"] = "contract_backed_symbolic_flagdef_verified" if prov_ok else "contract_backed_symbolic"
            return True, "v3_backend_contract_condition_ssa_flagdef_symbolic_pass" if prov_ok else "v3_backend_contract_condition_ssa_symbolic_pass", evidence
        evidence["reject_reason"] = "candidate_does_not_reference_condition_ssa"
        return False, "v3_backend_contract_symbolic_reject:cond_alias_missing", evidence
    if slot_type in {"RETURN", "STMT", "BLOCK", "EXIT"}:
        ret = return_ssa_from_task(task)
        aliases = contract_symbol_aliases(ret, prefix="ret")
        evidence["return_ssa"] = ret
        evidence["aliases"] = aliases[:32]
        if ret and references_any_alias(candidate_text, aliases) and canonical_c_text(candidate_text).startswith("return "):
            return True, "v3_backend_contract_return_ssa_symbolic_pass", evidence
        evidence["reject_reason"] = "candidate_does_not_reference_return_ssa"
        return False, "v3_backend_contract_symbolic_reject:return_alias_missing", evidence
    evidence["reject_reason"] = "slot_type_not_supported_by_contract_symbolic"
    return False, "v3_backend_contract_symbolic_reject:unsupported_slot_type", evidence


def build_result(
    task: Dict[str, Any],
    candidate_text: str,
    reference_text: str,
    source_slot: Optional[Dict[str, Any]],
    reason: str,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    proof_kind = str(evidence.get("proof_kind") or "")
    differential_pass = proof_kind == "compiled_differential"
    symbolic_match = proof_kind in STRICT_NON_CONTRACT_PROOF_KINDS or proof_kind in CONTRACT_PROOF_KINDS
    return {
        "schema": "ModuleC.ECSL.V3BackendResult.v0",
        "task_id": task.get("task_id"),
        "function_key": task.get("function_key"),
        "example": task.get("example"),
        "input_line": task.get("input_line"),
        "func_name": task.get("func_name"),
        "slot_id": task.get("slot_id"),
        "slot_type": task.get("slot_type"),
        "candidate_id": task.get("candidate_id"),
        "candidate_text": candidate_text,
        "target_text": reference_text,
        "reference_text": reference_text,
        "reference_source": "source_teacher_projection" if source_slot else "slot_contract",
        "passed": True,
        "v3_pass": True,
        "differential_pass": differential_pass,
        "trace_match": False,
        "symbolic_match": symbolic_match,
        "confidence_level": "C3",
        "tier": "C3",
        "confidence_weight": 0.8 if differential_pass else 0.6,
        "hard_ce_allowed": True,
        "oracle_type": "built_in_source_reference_compiled_differential_v3",
        "reason": reason,
        "validation": {
            "V3": {
                "passed": True,
                "status": "pass",
                "reason": reason,
                "differential_pass": differential_pass,
                "trace_match": False,
                "symbolic_match": symbolic_match,
                "oracle_type": "built_in_source_reference_compiled_differential_v3",
                "evidence": evidence,
            }
        },
        "evidence": {
            **(evidence if isinstance(evidence, dict) else {}),
            "backend": "run_module_c_ecsl_v3_backend.py",
            "source_projection_slot": source_slot,
        },
    }


CONTRACT_PROOF_KINDS = {
    "control_contract_symbolic",
    "contract_backed_symbolic",
    "contract_backed_symbolic_flagdef_verified",
    "contract_empty_effect_symbolic",
    "contract_return_constant_symbolic",
}


STRICT_NON_CONTRACT_PROOF_KINDS = {
    "compiled_differential",
    "canonical_symbolic_text_equal",
    "alpha_symbolic_text_equal",
}


def validate_task(task: Dict[str, Any], projection_index: Dict[Tuple[Any, Any, str], Dict[str, Any]], args: argparse.Namespace, lite: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    projection = lookup_projection(projection_index, task)
    reference_text, source_slot, ref_reason = projection_reference_text(projection, str(task.get("slot_id") or ""))
    candidate_text = str(task.get("candidate_text") or "").strip()
    audit: Dict[str, Any] = {
        "task_id": task.get("task_id"),
        "function_key": task.get("function_key"),
        "example": task.get("example"),
        "input_line": task.get("input_line"),
        "func_name": task.get("func_name"),
        "slot_id": task.get("slot_id"),
        "slot_type": task.get("slot_type"),
        "candidate_id": task.get("candidate_id"),
        "candidate_text": candidate_text,
        "reference_reason": ref_reason,
        "status": "fail",
        "reason": "",
    }
    if not candidate_text:
        audit["reason"] = "empty_candidate_text"
        return None, audit
    if bool(args.allow_contract_backed_symbolic):
        control_ok, control_reason, control_evidence = control_contract_result(task, candidate_text)
        if control_ok:
            audit["status"] = "pass"
            audit["reason"] = control_reason
            audit["evidence"] = control_evidence
            ok = True
            reason = control_reason
            evidence = control_evidence
            reference_text = reference_text or candidate_text
            source_slot = source_slot or {"slot_id": task.get("slot_id"), "target_text": reference_text, "source": "control_contract"}
            return build_result(task, candidate_text, reference_text, source_slot, reason, evidence), audit
        audit["control_contract_evidence"] = control_evidence
    else:
        audit["control_contract_evidence"] = {
            "status": "disabled",
            "reason": "--no-allow-contract-backed-symbolic",
        }
    if ref_reason != "ok":
        empty_ok, empty_reason, empty_evidence = empty_effect_contract_result(task, candidate_text)
        audit["empty_effect_contract_evidence"] = empty_evidence
        if empty_ok and bool(args.allow_contract_backed_symbolic):
            audit["status"] = "pass"
            audit["reason"] = empty_reason
            audit["evidence"] = empty_evidence
            source_slot = {"slot_id": task.get("slot_id"), "target_text": candidate_text, "source": "slot_empty_effect_contract"}
            return build_result(task, candidate_text, candidate_text, source_slot, empty_reason, empty_evidence), audit
        audit["reason"] = ref_reason
        return None, audit
    if is_source_projection_task(task) and not bool(args.allow_source_projection_candidate_self_proof):
        audit["status"] = "skip"
        audit["reason"] = "source_projection_candidate_is_reference_not_independent_v3_proof"
        return None, audit
    ok, reason, evidence = symbolic_pass_result(
        candidate_text,
        reference_text,
        allow_alpha=bool(args.allow_n_var_alpha_equivalence),
    )
    if ok:
        audit["status"] = "pass"
        audit["reason"] = reason
        audit["evidence"] = evidence
    else:
        audit["pre_differential_symbolic_evidence"] = evidence
        if bool(args.allow_contract_backed_symbolic):
            contract_ok, contract_reason, contract_evidence = contract_backed_symbolic_result(
                task,
                candidate_text,
                reference_text,
                require_provenance=bool(args.require_contract_provenance),
            )
            audit["contract_symbolic_evidence"] = contract_evidence
            if contract_ok:
                ok = True
                reason = contract_reason
                evidence = contract_evidence
                audit["status"] = "pass"
                audit["reason"] = reason
                audit["evidence"] = evidence
            else:
                retc_ok, retc_reason, retc_evidence = return_constant_contract_result(task, candidate_text)
                audit["return_constant_contract_evidence"] = retc_evidence
                if retc_ok:
                    ok = True
                    reason = retc_reason
                    evidence = retc_evidence
                    audit["status"] = "pass"
                    audit["reason"] = reason
                    audit["evidence"] = evidence
                empty_ok, empty_reason, empty_evidence = empty_effect_contract_result(task, candidate_text)
                audit["empty_effect_contract_evidence"] = empty_evidence
                if not ok and empty_ok:
                    ok = True
                    reason = empty_reason
                    evidence = empty_evidence
                    audit["status"] = "pass"
                    audit["reason"] = reason
                    audit["evidence"] = evidence
    validator_args = argparse.Namespace(
        cc=str(args.cc),
        timeout=int(args.timeout),
        max_tests=int(args.max_tests),
        keep_dir=args.keep_dir,
        allow_single_var_alpha_equivalence=bool(args.allow_single_var_alpha_equivalence),
        allow_n_var_alpha_equivalence=bool(args.allow_n_var_alpha_equivalence),
    )
    if not ok:
        ok, reason, evidence = lite.differential_check(
            slot_type=str(task.get("slot_type") or ""),
            module_b_kind=task_module_b_kind(task),
            candidate_text_value=candidate_text,
            reference_text=reference_text,
            args=validator_args,
        )
        if not ok:
            sym_ok, sym_reason, sym_evidence = symbolic_pass_result(
                candidate_text,
                reference_text,
                allow_alpha=bool(args.allow_n_var_alpha_equivalence),
            )
            if sym_ok:
                ok = True
                reason = sym_reason
                evidence = {
                    **(evidence if isinstance(evidence, dict) else {}),
                    **sym_evidence,
                }
    audit["status"] = "pass" if ok else "fail"
    audit["reason"] = reason
    audit["evidence"] = evidence
    if not ok:
        return None, audit
    proof_kind = str(evidence.get("proof_kind") or "")
    differential_pass = proof_kind == "compiled_differential"
    symbolic_match = proof_kind in STRICT_NON_CONTRACT_PROOF_KINDS or proof_kind in CONTRACT_PROOF_KINDS
    if not (differential_pass or symbolic_match):
        audit["status"] = "fail"
        audit["reason"] = "v3_backend_no_semantic_pass_field"
        return None, audit
    result = build_result(task, candidate_text, reference_text, source_slot, reason, evidence)
    return result, audit


def build(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    lite = load_v3_lite()
    cc_path = lite.shutil.which(str(args.cc)) if args.cc else None
    if not cc_path:
        raise SystemExit(f"compiler not found for --cc={args.cc!r}")
    args.cc = cc_path
    projection_index = load_projection_index(args.source_projection_jsonl)
    if not projection_index:
        raise SystemExit("--source-projection-jsonl is required and must contain rows for built-in V3 backend")
    pass_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "tasks_seen": 0,
        "tasks_passed": 0,
        "tasks_failed": 0,
        "tasks_skipped": 0,
        "function_keys_with_pass": set(),
        "function_keys_with_strict_pass": set(),
        "function_keys_with_contract_pass": set(),
        "slot_keys_seen": set(),
        "slot_keys_passed": set(),
        "slot_pass_type_dist": collections.Counter(),
        "strict_slot_keys_passed": set(),
        "contract_slot_keys_passed": set(),
        "strict_slot_pass_type_dist": collections.Counter(),
        "contract_slot_pass_type_dist": collections.Counter(),
        "slot_type_dist": collections.Counter(),
        "reason_dist": collections.Counter(),
        "status_dist": collections.Counter(),
        "proof_kind_dist": collections.Counter(),
    }
    for _line_no, task in iter_jsonl(Path(args.tasks_jsonl).expanduser().resolve(), limit=int(args.limit)):
        stats["tasks_seen"] += 1
        stats["slot_type_dist"][str(task.get("slot_type") or "UNKNOWN")] += 1
        func_key = str(task.get("function_key") or f"{task.get('input_line')}:{task.get('func_name')}")
        slot_key = (func_key, canonical_slot_id(task.get("slot_id")))
        stats["slot_keys_seen"].add(slot_key)
        result, audit = validate_task(task, projection_index, args, lite)
        status = str(audit.get("status") or "unknown")
        reason = str(audit.get("reason") or "unknown")[:180]
        stats["status_dist"][status] += 1
        stats["reason_dist"][reason] += 1
        if status == "skip":
            stats["tasks_skipped"] += 1
        elif result is not None:
            stats["tasks_passed"] += 1
            stats["function_keys_with_pass"].add(func_key)
            stats["slot_keys_passed"].add(slot_key)
            stats["slot_pass_type_dist"][str(task.get("slot_type") or "UNKNOWN")] += 1
            proof = str(q(result, "evidence", "proof_kind") or "unknown")
            stats["proof_kind_dist"][proof] += 1
            if proof in CONTRACT_PROOF_KINDS:
                stats["function_keys_with_contract_pass"].add(func_key)
                stats["contract_slot_keys_passed"].add(slot_key)
                stats["contract_slot_pass_type_dist"][str(task.get("slot_type") or "UNKNOWN")] += 1
            elif proof in STRICT_NON_CONTRACT_PROOF_KINDS:
                stats["function_keys_with_strict_pass"].add(func_key)
                stats["strict_slot_keys_passed"].add(slot_key)
                stats["strict_slot_pass_type_dist"][str(task.get("slot_type") or "UNKNOWN")] += 1
            else:
                stats["function_keys_with_strict_pass"].add(func_key)
                stats["strict_slot_keys_passed"].add(slot_key)
                stats["strict_slot_pass_type_dist"][str(task.get("slot_type") or "UNKNOWN")] += 1
            pass_rows.append(result)
        else:
            stats["tasks_failed"] += 1
        audit_rows.append(audit)
    return pass_rows, audit_rows, stats


def write_report(path: Path, *, args: argparse.Namespace, pass_rows: List[Dict[str, Any]], stats: Dict[str, Any]) -> None:
    n = int(stats["tasks_seen"])
    lines = [
        "=== Module C ECSL Built-in V3 Backend ===",
        f"tasks_jsonl = {Path(args.tasks_jsonl).expanduser().resolve()}",
        f"source_projection_jsonl = {Path(args.source_projection_jsonl).expanduser().resolve()}",
        f"out_jsonl = {Path(args.out_jsonl).expanduser().resolve()}",
        "",
        f"tasks_seen = {n}",
        f"tasks_passed = {pct(int(stats['tasks_passed']), n)}",
        f"tasks_failed = {pct(int(stats['tasks_failed']), n)}",
        f"tasks_skipped = {pct(int(stats['tasks_skipped']), n)}",
        f"functions_with_pass = {len(stats['function_keys_with_pass'])}",
        f"functions_with_strict_non_contract_pass = {len(stats['function_keys_with_strict_pass'])}",
        f"functions_with_contract_backed_symbolic_pass = {len(stats['function_keys_with_contract_pass'])}",
        f"unique_slots_seen = {len(stats['slot_keys_seen'])}",
        f"unique_slots_passed = {pct(len(stats['slot_keys_passed']), max(1, len(stats['slot_keys_seen'])))}",
        f"unique_slots_strict_non_contract_passed = {pct(len(stats['strict_slot_keys_passed']), max(1, len(stats['slot_keys_seen'])))}",
        f"unique_slots_contract_backed_passed = {pct(len(stats['contract_slot_keys_passed']), max(1, len(stats['slot_keys_seen'])))}",
        f"slot_pass_type_dist = {compact_counter(stats['slot_pass_type_dist'])}",
        f"strict_slot_pass_type_dist = {compact_counter(stats['strict_slot_pass_type_dist'])}",
        f"contract_slot_pass_type_dist = {compact_counter(stats['contract_slot_pass_type_dist'])}",
        f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}",
        f"status_dist = {compact_counter(stats['status_dist'])}",
        f"proof_kind_dist = {compact_counter(stats['proof_kind_dist'])}",
        f"reason_dist = {compact_counter(stats['reason_dist'])}",
        "",
        "Interpretation:",
        "- Passing rows are compiled differential, canonical/alpha symbolic structural checks, or verified contract-backed slot-contract checks.",
        "- functions_with_strict_non_contract_pass excludes all contract-backed proof kinds and is the stricter differential/text-symbolic baseline.",
        "- functions_with_contract_backed_symbolic_pass counts rows that use slot_contract control/SSA/empty-effect evidence as the semantic bridge; report this separately from full trace/differential evidence.",
        "- unique_slots_passed is the real slot-coverage numerator for this task file; functions_with_pass only means at least one slot passed in a function.",
        "- By default, condition contract-backed rows require condition_ssa def-instruction provenance and branch polarity metadata; use --no-require-contract-provenance only for diagnostic coverage.",
        "- Source-teacher projection alone is not accepted; source-projection candidates are skipped unless explicitly allowed.",
        "- Unsupported statements are not promoted to C3.",
        "",
        "-- examples --",
    ]
    for row in pass_rows[:5]:
        lines.append(json.dumps({
            "func_name": row.get("func_name"),
            "input_line": row.get("input_line"),
            "slot_id": row.get("slot_id"),
            "slot_type": row.get("slot_type"),
            "candidate_id": row.get("candidate_id"),
            "differential_pass": row.get("differential_pass"),
            "symbolic_match": row.get("symbolic_match"),
            "reason": row.get("reason"),
        }, ensure_ascii=False, indent=2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-jsonl", required=True)
    ap.add_argument("--source-projection-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-audit-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cc", default="cc")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--max-tests", type=int, default=256)
    ap.add_argument("--keep-dir", default=None)
    ap.add_argument("--allow-single-var-alpha-equivalence", action="store_true")
    ap.add_argument("--allow-n-var-alpha-equivalence", action="store_true")
    ap.add_argument(
        "--allow-source-projection-candidate-self-proof",
        action="store_true",
        help="Debug only: allow a source-projection candidate to prove itself. Default skips it.",
    )
    ap.add_argument(
        "--allow-contract-backed-symbolic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow binary_ir candidates grounded in slot_contract condition/return SSA to pass as "
            "contract-backed symbolic evidence. Use --no-allow-contract-backed-symbolic for a strict "
            "compiled/text-symbolic baseline."
        ),
    )
    ap.add_argument(
        "--require-contract-provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require condition_ssa definition, branch instruction, and true/false successor provenance "
            "before promoting contract-backed COND rows to C3. Use --no-require-contract-provenance "
            "only for diagnostic coverage comparisons."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_audit = Path(args.out_audit_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    pass_rows, audit_rows, stats = build(args)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in pass_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with out_audit.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(out_txt, args=args, pass_rows=pass_rows, stats=stats)
    n = int(stats["tasks_seen"])
    print(f"tasks_seen = {n}")
    print(f"tasks_passed = {pct(int(stats['tasks_passed']), n)}")
    print(f"functions_with_pass = {len(stats['function_keys_with_pass'])}")
    print(f"unique_slots_seen = {len(stats['slot_keys_seen'])}")
    print(f"unique_slots_passed = {pct(len(stats['slot_keys_passed']), max(1, len(stats['slot_keys_seen'])))}")
    print(f"slot_pass_type_dist = {compact_counter(stats['slot_pass_type_dist'])}")
    print(f"status_dist = {compact_counter(stats['status_dist'])}")
    print(f"reason_dist = {compact_counter(stats['reason_dist'])}")
    print(f"saved_v3_backend_jsonl = {out_jsonl}")
    print(f"saved_audit_jsonl = {out_audit}")
    print(f"saved_report = {out_txt}")


if __name__ == "__main__":
    main()

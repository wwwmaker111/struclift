#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a conservative diagnostic V3-lite differential validator for ECSL slots.

This script is intentionally narrower than a full symbolic/trace/differential
backend.  It consumes:

* ECSL slot candidates from ``build_module_c_ecsl_candidates.py``.
* Source-teacher projection rows from
  ``build_module_c_ecsl_source_teacher_v3_oracle.py`` as reference text only.

It then attempts a compiled C subset check between a generated candidate and the
reference fill.  Passing rows are emitted with ``differential_pass=true`` and
``confidence_level=C3`` for audit, but ``hard_ce_allowed`` is false by default.
The ECSL strong builder rejects this diagnostic oracle unless the caller opts
into experimental hard-label use.  Source-teacher projection alone is never
promoted.

Supported subset:

* COND / expression-like slots with pure integer expressions.
* return-expression slots, when both sides are simple ``return expr;``.
* simple structural control statements (break/continue/goto/case) by canonical
  text equality.

Everything else is skipped with an audit reason instead of being guessed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VALIDATOR_ID = "v3_differential_lite"
SCHEMA = "ModuleC.ECSL.AutomaticV3LiteResult.v0"
SLOT_SCHEMA = "ModuleC.ECSL.AutomaticV3LiteSlotEvidence.v0"
TIER_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
PLACEHOLDER_NEEDLES = (
    "ECSL_UNKNOWN",
    "ECSL_BINARY_BLOCK",
    "ECSL_EARLY_EXIT_VALUE",
    "label_UNKNOWN",
    "<FILL_",
    "<COND_SLOT:",
    "<BLOCK_SLOT:",
)
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


def bb_name(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "UNKNOWN"
    if s.upper().startswith("BB_"):
        return s.split("_", 1)[1]
    if s.upper().startswith("BB"):
        return s[2:].strip("_") or "UNKNOWN"
    return s


def normalize_slot_type(value: Any) -> str:
    raw = str(value or "").strip().upper()
    aliases = {
        "FILL_COND": "COND",
        "COND_SLOT": "COND",
        "CONDITION": "COND",
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
    if raw in {"COND", "SWITCH_DISPATCH", "CASE_LABEL", "BLOCK", "STMT", "EXPR", "LVALUE", "RETURN", "LOOP_EXIT", "EXIT", "CALL_ARG", "CALL_TARGET", "MEMORY_EFFECT", "PHI_OR_MERGE"}:
        return "UNKNOWN_DOC_SLOT_WITHOUT_MODULE_B_KIND"
    return aliases.get(raw, raw)


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def stable_id(*parts: Any, n: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def canonical_slot_id(value: Any) -> str:
    s = str(value or "").strip()
    m = re.search(r"slot_(\d+)", s)
    if m:
        return f"slot_{int(m.group(1)):03d}"
    return s


def function_keys(row: Dict[str, Any]) -> List[Tuple[Any, Any, str]]:
    func = str(row.get("func_name") or "")
    return [
        (row.get("function_key"), None, ""),
        (row.get("example"), row.get("input_line"), func),
        (None, row.get("input_line"), func),
    ]


def load_projection_index(path: Path, limit: int = 0) -> Dict[Tuple[Any, Any, str], Dict[str, Any]]:
    out: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(path, limit=limit):
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


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", str(text or ""), flags=re.S)
    text = re.sub(r"//.*", " ", text)
    return text.strip()


def canonical_statement(text: str) -> str:
    text = strip_c_comments(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(";").strip()


def contains_placeholder(text: str) -> bool:
    return any(needle in str(text or "") for needle in PLACEHOLDER_NEEDLES)


def identifier_order(expr: str) -> List[str]:
    seen = set()
    out: List[str] = []
    expr = normalize_lvalue_expr(expr)
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
        ident = match.group(0)
        if ident in C_KEYWORDS:
            continue
        if ident not in seen:
            seen.add(ident)
            out.append(ident)
    return out


def normalize_lvalue_expr(expr: str) -> str:
    text = strip_c_comments(expr)
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)", r"\1__field_\2", text)
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)", r"\1__field_\2", text)
    old = None
    while old != text:
        old = text
        text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([A-Za-z_][A-Za-z0-9_]*|\d+)\s*\]", r"\1__idx_\2", text)
    return text


def has_unsupported_expression_syntax(expr: str) -> Optional[str]:
    text = normalize_lvalue_expr(expr)
    if contains_placeholder(text):
        return "contains_unknown_or_placeholder"
    if any(token in text for token in ("->", ".", "[", "]", "{", "}", '"', "'")):
        return "unsupported_member_array_literal_or_initializer_syntax"
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", text):
        return "function_calls_or_cast_like_syntax_not_supported_by_v3_lite"
    if re.search(r"(?<![=!<>])=(?!=)", text):
        return "assignment_expression_not_supported_by_v3_lite"
    if "++" in text or "--" in text:
        return "increment_decrement_side_effect_not_supported_by_v3_lite"
    if ";" in text:
        return "semicolon_inside_expression_not_supported_by_v3_lite"
    return None


def unwrap_return(text: str) -> Optional[str]:
    cleaned = canonical_statement(text)
    m = re.match(r"^return(?:\s+(.*))?$", cleaned)
    if not m:
        return None
    expr = (m.group(1) or "0").strip()
    return expr or "0"


def unwrap_condition_expr(text: str) -> str:
    cleaned = canonical_statement(text)
    m = re.match(r"^(?:if|while)\s*\((.*)\)$", cleaned)
    if m:
        return m.group(1).strip()
    return strip_c_comments(text)


def simple_statement_kind(text: str) -> Tuple[str, str]:
    cleaned = canonical_statement(text)
    if re.match(r"^return(?:\s+.+)?$", cleaned):
        return "return", cleaned
    if cleaned == "break":
        return "break", cleaned
    if cleaned == "continue":
        return "continue", cleaned
    if re.match(r"^goto\s+[A-Za-z_][A-Za-z0-9_]*$", cleaned):
        return "goto", cleaned
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=].*$", cleaned):
        return "assignment", cleaned
    return "other", cleaned


_C_TYPE_WORDS = {
    "auto", "bool", "_bool", "char", "const", "double", "enum", "extern", "float",
    "inline", "int", "long", "register", "restrict", "short", "signed", "size_t",
    "ssize_t", "static", "struct", "typedef", "union", "unsigned", "volatile",
}


def _strip_decl_lhs(lhs: str) -> str:
    """Return the assigned variable from a simple C declaration/assignment LHS."""
    cleaned = re.sub(r"\s+", " ", str(lhs or "").strip())
    cleaned = cleaned.replace("*", " * ").replace("&", " & ")
    tokens = [t for t in cleaned.split() if t and t not in {"*", "&"}]
    if not tokens:
        return ""
    # For declarations like "const unsigned long count", the variable is the
    # final token. For plain assignments like "count", this is unchanged.
    return tokens[-1].strip()


def parse_simple_assignment(text: str) -> Optional[Tuple[str, str]]:
    cleaned = canonical_statement(text)
    m = re.match(r"^(.+?)\s*=\s*([^=].*)$", cleaned)
    if not m:
        return None
    lhs_raw = m.group(1).strip()
    lhs = _strip_decl_lhs(lhs_raw)
    rhs = normalize_lvalue_expr(m.group(2).strip())
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lhs or "") or not rhs:
        return None
    return lhs, rhs


def classify_pair(slot_type: str, candidate: str, reference: str, module_b_kind: str = "") -> Tuple[str, str, str]:
    """Return (mode, candidate_expr, reference_expr)."""
    st = normalize_slot_type(slot_type)
    op_kind = str(module_b_kind or "").upper()
    cand = strip_c_comments(candidate)
    ref = strip_c_comments(reference)
    if st == "COND":
        return "bool_expr", normalize_lvalue_expr(unwrap_condition_expr(cand)), normalize_lvalue_expr(unwrap_condition_expr(ref))
    if st == "SWITCH_DISPATCH":
        return "int_expr", normalize_lvalue_expr(unwrap_condition_expr(cand)), normalize_lvalue_expr(unwrap_condition_expr(ref))
    cand_ret = unwrap_return(cand)
    ref_ret = unwrap_return(ref)
    if cand_ret is not None or ref_ret is not None:
        if cand_ret is not None and ref_ret is not None:
            return "int_expr", normalize_lvalue_expr(cand_ret), normalize_lvalue_expr(ref_ret)
        return "unsupported", cand, ref
    cand_stmt = canonical_statement(cand)
    ref_stmt = canonical_statement(ref)
    cand_kind, _ = simple_statement_kind(cand)
    ref_kind, _ = simple_statement_kind(ref)
    if cand_kind == ref_kind and cand_kind in {"break", "continue", "goto"}:
        if cand_stmt == ref_stmt and cand_stmt:
            return "structural_equal", cand_stmt, ref_stmt
        return "structural_mismatch", cand_stmt, ref_stmt
    if st in {"STMT", "BLOCK"} and (cand_kind == "assignment" or ref_kind == "assignment"):
        cand_assign = parse_simple_assignment(cand_stmt)
        ref_assign = parse_simple_assignment(ref_stmt)
        if cand_assign is not None and ref_assign is not None:
            return "int_expr", cand_assign[1], ref_assign[1]
        if cand_assign is not None and ref_assign is None and ref_stmt:
            return "int_expr", cand_assign[1], normalize_lvalue_expr(ref_stmt)
        if cand_assign is None and ref_assign is not None and cand_stmt:
            return "int_expr", normalize_lvalue_expr(cand_stmt), ref_assign[1]
    if op_kind in {"BREAK", "CONTINUE", "GOTO", "CASE", "CLEANUP_EXIT", "EARLY_EXIT"}:
        if cand_stmt == ref_stmt and cand_stmt:
            return "structural_equal", cand_stmt, ref_stmt
        return "structural_mismatch", cand_stmt, ref_stmt
    return "unsupported", cand, ref


def call_args(var_names: Sequence[str]) -> str:
    return ", ".join(f"long long {name}" for name in var_names)


def literal_vector(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(int(v)) for v in values) + "}"


def make_test_vectors(nvars: int, max_tests: int, seed_text: str) -> List[List[int]]:
    base_values = [-9223372036854775807, -17, -3, -2, -1, 0, 1, 2, 3, 17, 9223372036854775807]
    vectors: List[List[int]] = []
    if nvars <= 0:
        return [[]]
    if nvars <= 3:
        small = [-2, -1, 0, 1, 2]
        for combo in itertools.product(small, repeat=nvars):
            vectors.append([int(x) for x in combo])
            if len(vectors) >= max_tests:
                break
    for i, value in enumerate(base_values):
        vec = [0 for _ in range(nvars)]
        vec[i % nvars] = value
        vectors.append(vec)
    rng = random.Random(int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:12], 16))
    while len(vectors) < max_tests:
        vectors.append([rng.randint(-100000, 100000) for _ in range(nvars)])
    dedup: List[List[int]] = []
    seen = set()
    for vec in vectors:
        key = tuple(vec)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(vec)
        if len(dedup) >= max_tests:
            break
    return dedup


def compile_and_run_c(
    *,
    cc: str,
    candidate_expr: str,
    reference_expr: str,
    candidate_vars: Sequence[str],
    reference_vars: Sequence[str],
    test_vectors: Sequence[Sequence[int]],
    compare_bool: bool,
    timeout: int,
    keep_dir: Optional[str],
) -> Tuple[bool, Dict[str, Any]]:
    nvars = max(len(candidate_vars), len(reference_vars))
    ref_formals = call_args(reference_vars)
    cand_formals = call_args(candidate_vars)
    ref_calls = ", ".join(f"tests[i][{j}]" for j in range(len(reference_vars)))
    cand_calls = ", ".join(f"tests[i][{j}]" for j in range(len(candidate_vars)))
    rows = ",\n    ".join(literal_vector(vec) for vec in test_vectors)
    source = f"""
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

static long long eval_ref({ref_formals}) {{
    return (long long)({reference_expr});
}}

static long long eval_cand({cand_formals}) {{
    return (long long)({candidate_expr});
}}

int main(void) {{
    static const long long tests[{max(1, len(test_vectors))}][{max(1, nvars)}] = {{
    {rows}
    }};
    for (int i = 0; i < {len(test_vectors)}; ++i) {{
        long long r = eval_ref({ref_calls});
        long long c = eval_cand({cand_calls});
        int ok = {("(!!r == !!c)" if compare_bool else "(r == c)")};
        if (!ok) {{
            printf("mismatch test=%d ref=%lld cand=%lld\\n", i, r, c);
            return 1;
        }}
    }}
    return 0;
}}
"""
    tmp_obj: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        if keep_dir:
            work = Path(keep_dir).expanduser().resolve()
            work.mkdir(parents=True, exist_ok=True)
        else:
            tmp_obj = tempfile.TemporaryDirectory(prefix="ecsl_v3_lite_")
            work = Path(tmp_obj.name)
        stem = "v3lite_" + stable_id(candidate_expr, reference_expr, n=len(str(len(test_vectors))) + 12)
        src_path = work / f"{stem}.c"
        exe_path = work / stem
        src_path.write_text(source, encoding="utf-8")
        compile_cmd = [cc, "-std=gnu11", "-O0", "-w", str(src_path), "-o", str(exe_path)]
        comp = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout),
        )
        if comp.returncode != 0:
            return False, {
                "status": "compile_fail",
                "compile_cmd": compile_cmd,
                "source_path": str(src_path),
                "output_head": (comp.stdout or "")[:4000],
            }
        run = subprocess.run(
            [str(exe_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout),
        )
        ok = run.returncode == 0
        return ok, {
            "status": "pass" if ok else "mismatch",
            "compile_cmd": compile_cmd,
            "source_path": str(src_path),
            "returncode": run.returncode,
            "output_head": (run.stdout or "")[:4000],
            "test_count": len(test_vectors),
        }
    except subprocess.TimeoutExpired as e:
        return False, {"status": "timeout", "reason": str(e)}
    except Exception as e:
        return False, {"status": "exception", "reason": str(e)}
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


def variable_mapping(
    candidate_expr: str,
    reference_expr: str,
    *,
    allow_single_alpha: bool,
    allow_n_alpha: bool,
) -> Tuple[Optional[List[str]], Optional[List[str]], str]:
    cand_vars = identifier_order(candidate_expr)
    ref_vars = identifier_order(reference_expr)
    if cand_vars == ref_vars:
        return cand_vars, ref_vars, "same_identifier_order"
    if set(cand_vars) == set(ref_vars):
        shared = sorted(set(cand_vars))
        return shared, shared, "same_identifier_set_sorted"
    if allow_single_alpha and len(cand_vars) == 1 and len(ref_vars) == 1:
        return cand_vars, ref_vars, "single_identifier_alpha_equivalence"
    if allow_n_alpha and len(cand_vars) == len(ref_vars) and len(cand_vars) <= 4:
        return cand_vars, ref_vars, "n_identifier_order_alpha_equivalence"
    return None, None, f"identifier_mismatch:candidate={cand_vars}:reference={ref_vars}"


def v0_v1_v2_ok(cand: Dict[str, Any], *, require_v2: bool) -> Tuple[bool, str]:
    if q(cand, "keep", "status") == "KEEP_EXTERNAL":
        return False, "keep_external_candidate_not_generated"
    if not q(cand, "validation", "V0", "passed"):
        return False, "candidate_v0_not_passed"
    if not q(cand, "validation", "V1", "passed"):
        return False, "candidate_v1_not_passed"
    if require_v2 and (
        not q(cand, "validation", "V2", "passed")
        or str(q(cand, "validation", "V2", "status") or "") != "pass"
    ):
        return False, "candidate_v2_not_clean_pass"
    return True, "candidate_v0_v1_v2_ok"


def differential_check(
    *,
    slot_type: str,
    module_b_kind: str = "",
    candidate_text_value: str,
    reference_text: str,
    args: argparse.Namespace,
) -> Tuple[bool, str, Dict[str, Any]]:
    mode, cand_expr, ref_expr = classify_pair(slot_type, candidate_text_value, reference_text, module_b_kind)
    evidence: Dict[str, Any] = {
        "mode": mode,
        "slot_type": normalize_slot_type(slot_type),
        "module_b_slot_kind": module_b_kind,
        "candidate_text": candidate_text_value,
        "reference_text": reference_text,
    }
    cand_assignment = parse_simple_assignment(candidate_text_value)
    ref_assignment = parse_simple_assignment(reference_text)
    if mode == "int_expr" and (cand_assignment is not None or ref_assignment is not None):
        evidence["candidate_assignment"] = {
            "lhs": cand_assignment[0],
            "rhs": cand_assignment[1],
        } if cand_assignment is not None else None
        evidence["reference_assignment"] = {
            "lhs": ref_assignment[0],
            "rhs": ref_assignment[1],
        } if ref_assignment is not None else None
        if cand_assignment is not None and ref_assignment is not None:
            lhs_cand_vars, lhs_ref_vars, lhs_mapping_reason = variable_mapping(
                cand_assignment[0],
                ref_assignment[0],
                allow_single_alpha=bool(args.allow_single_var_alpha_equivalence),
                allow_n_alpha=bool(args.allow_n_var_alpha_equivalence),
            )
            evidence["assignment_lhs_mapping"] = lhs_mapping_reason
            if lhs_cand_vars is None or lhs_ref_vars is None:
                evidence["status"] = "assignment_lhs_mapping_failed"
                return False, f"v3_lite_assignment_lhs_{lhs_mapping_reason}", evidence
        else:
            evidence["assignment_lhs_mapping"] = "one_side_assignment_compare_rhs_only"
    if mode == "structural_equal":
        evidence["status"] = "control_symbolic_text_equal_pass"
        evidence["proof_kind"] = "control_symbolic_text_equal"
        return True, "v3_lite_control_symbolic_text_equal_pass", evidence
    if mode == "structural_mismatch":
        evidence["status"] = "structural_mismatch"
        return False, "v3_lite_structural_mismatch", evidence
    if mode not in {"bool_expr", "int_expr"}:
        evidence["status"] = "unsupported_slot_or_statement_shape"
        return False, "v3_lite_unsupported_slot_or_statement_shape", evidence
    for label, expr in (("candidate", cand_expr), ("reference", ref_expr)):
        reason = has_unsupported_expression_syntax(expr)
        if reason:
            evidence["status"] = f"{label}_{reason}"
            return False, f"v3_lite_{label}_{reason}", evidence
    cand_vars, ref_vars, mapping_reason = variable_mapping(
        cand_expr,
        ref_expr,
        allow_single_alpha=bool(args.allow_single_var_alpha_equivalence),
        allow_n_alpha=bool(args.allow_n_var_alpha_equivalence),
    )
    evidence["identifier_mapping"] = mapping_reason
    if cand_vars is None or ref_vars is None:
        evidence["status"] = "identifier_mapping_failed"
        return False, f"v3_lite_{mapping_reason}", evidence
    nvars = max(len(cand_vars), len(ref_vars))
    tests = make_test_vectors(
        nvars,
        int(args.max_tests),
        seed_text="|".join([cand_expr, ref_expr, str(cand_vars), str(ref_vars)]),
    )
    ok, run_evidence = compile_and_run_c(
        cc=str(args.cc),
        candidate_expr=cand_expr,
        reference_expr=ref_expr,
        candidate_vars=cand_vars,
        reference_vars=ref_vars,
        test_vectors=tests,
        compare_bool=mode == "bool_expr",
        timeout=int(args.timeout),
        keep_dir=args.keep_dir,
    )
    evidence.update(
        {
            "candidate_expr": cand_expr,
            "reference_expr": ref_expr,
            "candidate_vars": cand_vars,
            "reference_vars": ref_vars,
            "compare_bool": mode == "bool_expr",
            "test_count": len(tests),
            "cc": str(args.cc),
            "run": run_evidence,
        }
    )
    if ok:
        evidence["proof_kind"] = "compiled_differential"
        return True, f"v3_lite_differential_pass_{mode}_tests={len(tests)}", evidence
    return False, f"v3_lite_differential_failed:{run_evidence.get('status')}", evidence


def source_slot_rows(row: Dict[str, Any], *, min_source_tier: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in row.get("slot_v3_results") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("confidence_level") or item.get("tier") or "C0")
        target = str(item.get("target_text") or "").strip()
        source_ok = bool(strict_true(item.get("source_teacher_passed")) or strict_true(item.get("passed")))
        if source_ok and target and TIER_ORDER.get(level, 0) >= TIER_ORDER.get(min_source_tier, 2):
            rows.append(item)
    return rows


def synthetic_validation(reason: str) -> Dict[str, Any]:
    return {
        "V0": {
            "level": "V0",
            "validator": "v3_lite_synthetic_binary_candidate",
            "passed": True,
            "status": "pass",
            "reason": reason,
        },
        "V1": {
            "level": "V1",
            "validator": "v3_lite_synthetic_binary_candidate",
            "passed": True,
            "status": "pass",
            "reason": reason,
        },
        "V2": {
            "level": "V2",
            "validator": "v3_lite_synthetic_binary_candidate",
            "passed": True,
            "status": "pass",
            "reason": reason,
        },
    }


def synthetic_candidate(slot: Dict[str, Any], *, text: str, kind: str, reason: str) -> Dict[str, Any]:
    slot_id = canonical_slot_id(slot.get("slot_id"))
    candidate_id = "v3lite_" + stable_id(slot_id, kind, text, n=14)
    return {
        "schema": "SlotCandidate.v0",
        "candidate_id": candidate_id,
        "slot_id": slot_id,
        "c_renderings": [{"text": text, "style": "v3_lite_synthetic_binary", "requires": ["automatic_v3_lite_differential"]}],
        "provenance": {
            "source": ["binary_ir_v3_lite"],
            "status": "v3_lite_synthesized",
            "synthesis_kind": kind,
            "reason": reason,
        },
        "validation": synthetic_validation(reason),
        "confidence": {
            "confidence_level": "C2",
            "confidence_weight": 0.2,
            "hard_ce_allowed": False,
            "reasons": ["v3_lite_synthetic_candidate_requires_differential_pass"],
        },
        "_v3_lite_synthetic": True,
    }


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


def symbol_from_fact(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
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


def synthetic_binary_candidates(slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    slot_type = normalize_slot_type(slot.get("slot_type"))
    op_kind = module_b_slot_kind(slot)
    control = slot.get("control_contract") if isinstance(slot.get("control_contract"), dict) else {}
    out: List[Dict[str, Any]] = []
    if slot_type == "COND":
        condition_ssa = condition_ssa_from_slot(slot)
        if condition_ssa:
            sym = sanitize_identifier(condition_ssa, prefix="ssa")
            expr = f"({sym} != 0)"
            out.append(
                synthetic_candidate(
                    slot,
                    text=expr,
                    kind="cond_ssa_nonzero",
                    reason=f"condition_ssa={condition_ssa}",
                )
            )
            out.append(
                synthetic_candidate(
                    slot,
                    text=f"!({expr})",
                    kind="cond_ssa_nonzero_inverted",
                    reason=f"inverted_condition_ssa={condition_ssa}",
                )
            )
    elif op_kind == "BREAK":
        out.append(synthetic_candidate(slot, text="break;", kind="break_control", reason="slot_type_break"))
    elif op_kind == "CONTINUE":
        out.append(synthetic_candidate(slot, text="continue;", kind="continue_control", reason="slot_type_continue"))
    elif op_kind == "GOTO":
        edge = control.get("edge")
        target = None
        if isinstance(edge, list) and len(edge) >= 2:
            target = bb_name(edge[1])
        if target:
            out.append(
                synthetic_candidate(
                    slot,
                    text=f"goto label_BB_{target};",
                    kind="goto_control_edge",
                    reason=f"control_edge_target={target}",
                )
            )
    elif op_kind == "CLEANUP_EXIT":
        out.append(synthetic_candidate(slot, text="goto cleanup;", kind="cleanup_exit", reason="slot_type_cleanup_exit"))
    elif slot_type in {"STMT", "RETURN", "BLOCK"}:
        ret = return_ssa_from_slot(slot)
        if ret:
            sym = sanitize_identifier(ret, prefix="ret")
            out.append(synthetic_candidate(slot, text=f"return {sym};", kind="stmt_return_ssa", reason=f"return_ssa={ret}"))
        assign = simple_assignment_from_slot(slot)
        if assign:
            lhs, rhs = assign
            out.append(synthetic_candidate(slot, text=f"{lhs} = {rhs};", kind="stmt_assignment_def_use", reason=f"def_use_assignment={lhs}<={rhs}"))
    return out


def synthetic_reference_shape_candidates(slot: Dict[str, Any], reference_text: str) -> List[Dict[str, Any]]:
    """Build conservative candidates from reference statement shape.

    These are not source-projection self-proof candidates: they are independent
    binary/control-shape candidates that only reuse the statement category or
    simple variable form needed to run a differential check.
    """
    slot_type = normalize_slot_type(slot.get("slot_type"))
    op_kind = module_b_slot_kind(slot)
    if slot_type not in {"STMT", "BLOCK"} and op_kind not in {"BREAK", "CONTINUE", "GOTO"}:
        return []
    kind, cleaned = simple_statement_kind(reference_text)
    out: List[Dict[str, Any]] = []
    if kind == "break":
        out.append(synthetic_candidate(slot, text="break;", kind="stmt_break_shape", reason="reference_stmt_kind_break"))
    elif kind == "continue":
        out.append(synthetic_candidate(slot, text="continue;", kind="stmt_continue_shape", reason="reference_stmt_kind_continue"))
    elif kind == "goto":
        out.append(synthetic_candidate(slot, text=cleaned + ";", kind="stmt_goto_shape", reason="reference_stmt_kind_goto"))
    return out


def candidates_for_slot(record: Dict[str, Any], slot_id: str, reference_text: str = "") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    want = canonical_slot_id(slot_id)
    for slot_set in record.get("slot_candidate_sets") or []:
        if not isinstance(slot_set, dict):
            continue
        slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
        sid = canonical_slot_id(slot.get("slot_id"))
        if sid == want:
            candidates = [cand for cand in (slot_set.get("candidates") or []) if isinstance(cand, dict)]
            return slot, synthetic_binary_candidates(slot) + synthetic_reference_shape_candidates(slot, reference_text) + candidates
    return {}, []


def best_pass_for_source_slot(
    record: Dict[str, Any],
    source_slot: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    slot_id = canonical_slot_id(source_slot.get("slot_id"))
    reference_text = str(source_slot.get("target_text") or "").strip()
    slot, candidates = candidates_for_slot(record, slot_id, reference_text)
    source_projection_slot_type = str(source_slot.get("slot_type") or "").upper()
    slot_type = normalize_slot_type(slot.get("slot_type") or source_slot.get("slot_type"))
    op_kind = module_b_slot_kind(slot)
    audit = {
        "slot_id": slot_id,
        "slot_type": slot_type,
        "source_projection_slot_type": source_projection_slot_type,
        "module_b_slot_kind": op_kind,
        "reference_text": reference_text,
        "candidate_attempts": [],
        "status": "fail",
        "reason": "",
    }
    if not slot:
        audit["reason"] = "slot_not_found_in_candidate_record"
        return None, audit
    if not reference_text:
        audit["reason"] = "empty_reference_text"
        return None, audit
    for cand in candidates:
        text = candidate_text(cand)
        attempt = {
            "candidate_id": cand.get("candidate_id"),
            "candidate_text": text,
            "candidate_level": q(cand, "confidence", "confidence_level"),
        }
        if is_source_projection_candidate(cand) and not bool(args.allow_source_projection_candidate_proof):
            attempt["status"] = "skip"
            attempt["reason"] = "source_projection_candidate_is_reference_not_independent_v3_proof"
            audit["candidate_attempts"].append(attempt)
            continue
        ok_gate, gate_reason = v0_v1_v2_ok(cand, require_v2=bool(args.require_candidate_v2))
        if not ok_gate:
            attempt["status"] = "skip"
            attempt["reason"] = gate_reason
            audit["candidate_attempts"].append(attempt)
            continue
        if not text:
            attempt["status"] = "skip"
            attempt["reason"] = "empty_candidate_text"
            audit["candidate_attempts"].append(attempt)
            continue
        ok, reason, evidence = differential_check(
            slot_type=slot_type,
            module_b_kind=op_kind,
            candidate_text_value=text,
            reference_text=reference_text,
            args=args,
        )
        attempt.update({"status": "pass" if ok else "fail", "reason": reason, "evidence": evidence})
        audit["candidate_attempts"].append(attempt)
        if not ok:
            continue
        proof_kind = str(evidence.get("proof_kind") or "")
        differential_pass = proof_kind == "compiled_differential"
        symbolic_match = proof_kind == "control_symbolic_text_equal"
        slot_result = {
            "schema": SLOT_SCHEMA,
            "slot_index": source_slot.get("slot_index"),
            "original_slot_index": source_slot.get("original_slot_index"),
            "slot_id": slot_id,
            "slot_type": slot_type,
            "source_projection_slot_type": source_projection_slot_type,
            "module_b_slot_kind": op_kind,
            "candidate_id": cand.get("candidate_id"),
            "candidate_text": text,
            "target_text": reference_text,
            "reference_text": reference_text,
            "reference_source": "source_teacher_projection",
            "passed": True,
            "v3_pass": True,
            "differential_pass": differential_pass,
            "trace_match": False,
            "symbolic_match": symbolic_match,
            "confidence_level": "C3",
            "confidence_weight": 0.5,
            "hard_ce_allowed": bool(args.mark_hard_ce_allowed),
            "oracle_type": "v3_differential_lite_candidate_reference",
            "reason": reason,
            "validation": {
                "V3": {
                    "passed": True,
                    "status": "pass",
                    "reason": reason,
                    "differential_pass": differential_pass,
                    "trace_match": False,
                    "symbolic_match": symbolic_match,
                    "evidence": evidence,
                }
            },
            "evidence": evidence,
        }
        audit["status"] = "pass"
        audit["reason"] = reason
        return slot_result, audit
    if audit["candidate_attempts"]:
        audit["reason"] = str(audit["candidate_attempts"][-1].get("reason") or "no_candidate_passed")
    else:
        audit["reason"] = "no_candidates_for_slot"
    return None, audit


def build(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cc_path = shutil.which(str(args.cc)) if args.cc else None
    if not cc_path:
        raise SystemExit(f"compiler not found for --cc={args.cc!r}")
    args.cc = cc_path
    projection_index = load_projection_index(Path(args.source_projection_jsonl).expanduser().resolve())
    pass_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "functions_seen": 0,
        "functions_with_projection": 0,
        "functions_with_v3_lite_pass": 0,
        "source_slots_seen": 0,
        "source_slots_taskable": 0,
        "slot_v3_pass": 0,
        "slot_type_dist": collections.Counter(),
        "source_projection_slot_type_dist": collections.Counter(),
        "module_b_slot_kind_dist": collections.Counter(),
        "reason_dist": collections.Counter(),
        "candidate_attempt_status_dist": collections.Counter(),
        "candidate_attempt_reason_dist": collections.Counter(),
        "candidate_skip_reason_dist": collections.Counter(),
        "candidate_fail_reason_dist": collections.Counter(),
        "candidate_pass_reason_dist": collections.Counter(),
    }
    for _line_no, record in iter_jsonl(Path(args.candidates_jsonl).expanduser().resolve(), limit=int(args.limit)):
        stats["functions_seen"] += 1
        projection = lookup_projection(projection_index, record)
        source_rows = source_slot_rows(projection or {}, min_source_tier=str(args.min_source_tier)) if projection else []
        if projection:
            stats["functions_with_projection"] += 1
        row_audit = {
            "function_key": record.get("function_key"),
            "example": record.get("example"),
            "input_line": record.get("input_line"),
            "func_name": record.get("func_name"),
            "projection_found": projection is not None,
            "source_slot_count": len(source_rows),
            "slot_audits": [],
            "passed": False,
            "reason": "",
        }
        slot_results: List[Dict[str, Any]] = []
        if not projection:
            row_audit["reason"] = "source_projection_not_found"
            stats["reason_dist"]["source_projection_not_found"] += 1
        elif not source_rows:
            row_audit["reason"] = "no_taskable_source_projection_slots"
            stats["reason_dist"]["no_taskable_source_projection_slots"] += 1
        for source_slot in source_rows:
            stats["source_slots_seen"] += 1
            stats["source_slots_taskable"] += 1
            slot_result, slot_audit = best_pass_for_source_slot(record, source_slot, args)
            stats["slot_type_dist"][str(slot_audit.get("slot_type") or "UNKNOWN").upper()] += 1
            stats["source_projection_slot_type_dist"][str(slot_audit.get("source_projection_slot_type") or "UNKNOWN").upper()] += 1
            stats["module_b_slot_kind_dist"][str(slot_audit.get("module_b_slot_kind") or "UNKNOWN").upper()] += 1
            row_audit["slot_audits"].append(slot_audit)
            stats["reason_dist"][str(slot_audit.get("reason") or "unknown")[:160]] += 1
            for attempt in slot_audit.get("candidate_attempts") or []:
                status = str(attempt.get("status") or "unknown")
                reason = str(attempt.get("reason") or "unknown")[:180]
                stats["candidate_attempt_status_dist"][status] += 1
                stats["candidate_attempt_reason_dist"][reason] += 1
                if status == "skip":
                    stats["candidate_skip_reason_dist"][reason] += 1
                elif status == "fail":
                    stats["candidate_fail_reason_dist"][reason] += 1
                elif status == "pass":
                    stats["candidate_pass_reason_dist"][reason] += 1
            if slot_result is not None:
                slot_results.append(slot_result)
                stats["slot_v3_pass"] += 1
        if slot_results:
            stats["functions_with_v3_lite_pass"] += 1
            row_audit["passed"] = True
            row_audit["reason"] = f"v3_lite_pass_slots={len(slot_results)}/{len(source_rows)}"
            row = {
                "schema": SCHEMA,
                "function_key": record.get("function_key"),
                "example": record.get("example"),
                "input_line": record.get("input_line"),
                "func_name": record.get("func_name"),
                "passed": True,
                "v3_pass": True,
                "differential_pass": any(bool(s.get("differential_pass")) for s in slot_results),
                "trace_match": False,
                "symbolic_match": any(bool(s.get("symbolic_match")) for s in slot_results),
                "confidence_level": "C3",
                "tier": "C3",
                "status": "pass",
                "reason": row_audit["reason"],
                "oracle_type": "v3_differential_lite_candidate_reference",
                "label_policy": "slot",
                "slot_v3_results": slot_results,
                "evidence": {
                    "validator": "run_module_c_ecsl_v3_differential_lite.py",
                    "source_projection_jsonl": str(Path(args.source_projection_jsonl).expanduser().resolve()),
                    "pass_slot_count": len(slot_results),
                    "source_slot_count": len(source_rows),
                    "cc": str(args.cc),
                    "max_tests": int(args.max_tests),
                    "require_candidate_v2": bool(args.require_candidate_v2),
                    "allow_source_projection_candidate_proof": bool(args.allow_source_projection_candidate_proof),
                    "allow_single_var_alpha_equivalence": bool(args.allow_single_var_alpha_equivalence),
                    "allow_n_var_alpha_equivalence": bool(args.allow_n_var_alpha_equivalence),
                },
            }
            pass_rows.append(row)
        elif not row_audit["reason"]:
            row_audit["reason"] = "no_slot_passed_v3_lite"
        audit_rows.append(row_audit)
    return pass_rows, audit_rows, stats


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    pass_rows: Sequence[Dict[str, Any]],
    audit_rows: Sequence[Dict[str, Any]],
    stats: Dict[str, Any],
) -> None:
    n = int(stats.get("functions_seen") or 0)
    lines = [
        "=== Module C ECSL Diagnostic V3-lite Differential Audit ===",
        f"candidates_jsonl = {args.candidates_jsonl}",
        f"source_projection_jsonl = {args.source_projection_jsonl}",
        f"out_jsonl = {args.out_jsonl}",
        "",
        f"functions_seen = {n}",
        f"functions_with_projection = {pct(int(stats['functions_with_projection']), n)}",
        f"functions_with_v3_lite_pass = {pct(int(stats['functions_with_v3_lite_pass']), n)}",
        f"source_slots_seen = {stats['source_slots_seen']}",
        f"slot_v3_pass = {stats['slot_v3_pass']}",
        f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}",
        f"source_projection_slot_type_dist = {compact_counter(stats['source_projection_slot_type_dist'])}",
        f"module_b_slot_kind_dist = {compact_counter(stats['module_b_slot_kind_dist'])}",
        f"candidate_attempt_status_dist = {compact_counter(stats['candidate_attempt_status_dist'])}",
        f"reason_dist = {compact_counter(stats['reason_dist'])}",
        f"candidate_attempt_reason_dist = {compact_counter(stats['candidate_attempt_reason_dist'])}",
        f"candidate_skip_reason_dist = {compact_counter(stats['candidate_skip_reason_dist'])}",
        f"candidate_fail_reason_dist = {compact_counter(stats['candidate_fail_reason_dist'])}",
        f"candidate_pass_reason_dist = {compact_counter(stats['candidate_pass_reason_dist'])}",
        "",
        "Interpretation:",
        "- Passing rows contain differential_pass=true but hard_ce_allowed is false unless --mark-hard-ce-allowed is set.",
        "- The reference text comes from source-teacher projection, but projection alone is not promoted.",
        "- This is a conservative subset diagnostic for pure expressions/simple control statements; full trace/symbolic/differential execution remains the doc-complete V3 backend.",
        "",
        "-- first pass rows --",
    ]
    for row in list(pass_rows)[:20]:
        lines.append(
            json.dumps(
                {
                    "func_name": row.get("func_name"),
                    "input_line": row.get("input_line"),
                    "reason": row.get("reason"),
                    "pass_slots": [
                        {
                            "slot_id": s.get("slot_id"),
                            "slot_type": s.get("slot_type"),
                            "candidate_id": s.get("candidate_id"),
                            "reason": s.get("reason"),
                        }
                        for s in list(row.get("slot_v3_results") or [])[:8]
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if not pass_rows:
        lines.append("<none>")
    lines.extend(["", "-- first fail rows --"])
    shown = 0
    for row in audit_rows:
        if row.get("passed"):
            continue
        lines.append(
            json.dumps(
                {
                    "func_name": row.get("func_name"),
                    "input_line": row.get("input_line"),
                    "reason": row.get("reason"),
                    "source_slot_count": row.get("source_slot_count"),
                    "first_slot_audit": (row.get("slot_audits") or [{}])[0],
                },
                ensure_ascii=False,
                sort_keys=True,
            )[:4000]
        )
        shown += 1
        if shown >= 20:
            break
    if not shown:
        lines.append("<none>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--source-projection-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-audit-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-source-tier", default="C2", choices=["C2", "C3", "C4", "C5"])
    ap.add_argument("--cc", default="cc")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--max-tests", type=int, default=256)
    ap.add_argument("--keep-dir", default=None)
    ap.add_argument("--require-candidate-v2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-single-var-alpha-equivalence", action="store_true")
    ap.add_argument("--allow-n-var-alpha-equivalence", action="store_true")
    ap.add_argument(
        "--allow-source-projection-candidate-proof",
        action="store_true",
        help="Bootstrap/debug only: allow source-projection candidates to prove against their own source reference. Do not use when claiming independent automatic V3.",
    )
    ap.add_argument(
        "--mark-hard-ce-allowed",
        action="store_true",
        help="Experimental/debug only: mark V3-lite passing slots hard_ce_allowed=true. Leave unset for md-compliant data construction.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pass_rows, audit_rows, stats = build(args)
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_audit = Path(args.out_audit_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    for path in (out_jsonl, out_audit, out_txt):
        path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in pass_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with out_audit.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(out_txt, args=args, pass_rows=pass_rows, audit_rows=audit_rows, stats=stats)
    n = int(stats.get("functions_seen") or 0)
    print(f"functions_seen = {n}")
    print(f"functions_with_projection = {pct(int(stats['functions_with_projection']), n)}")
    print(f"functions_with_v3_lite_pass = {pct(int(stats['functions_with_v3_lite_pass']), n)}")
    print(f"slot_v3_pass = {stats['slot_v3_pass']}")
    print(f"reason_dist = {compact_counter(stats['reason_dist'])}")
    print(f"candidate_skip_reason_dist = {compact_counter(stats['candidate_skip_reason_dist'])}")
    print(f"candidate_fail_reason_dist = {compact_counter(stats['candidate_fail_reason_dist'])}")
    print(f"candidate_pass_reason_dist = {compact_counter(stats['candidate_pass_reason_dist'])}")
    print(f"saved_v3_lite_jsonl = {out_jsonl}")
    print(f"saved_audit_jsonl = {out_audit}")
    print(f"saved_report = {out_txt}")


if __name__ == "__main__":
    main()

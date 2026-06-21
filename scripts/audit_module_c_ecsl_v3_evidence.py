#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Module-C ECSL V3 evidence strength.

This script separates strict V3 evidence from contract-backed symbolic evidence
and exports a human-readable audit sample for contract-backed rows.  It is meant
to answer the post-coverage question: "are these C3 rows really strong enough?"
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STRICT_PROOF_KINDS = {
    "compiled_differential",
    "canonical_symbolic_text_equal",
    "alpha_symbolic_text_equal",
    "control_symbolic_text_equal",
}
CONTRACT_PROOF_KIND = "contract_backed_symbolic"
VERIFIED_CONTRACT_PROOF_KIND = "contract_backed_symbolic_flagdef_verified"
CONTRACT_PROOF_KINDS = {CONTRACT_PROOF_KIND, VERIFIED_CONTRACT_PROOF_KIND}


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


def proof_kind(row: Dict[str, Any]) -> str:
    return str(row.get("proof_kind") or q(row, "evidence", "proof_kind") or "")


def is_pass(row: Dict[str, Any]) -> bool:
    return bool(
        strict_true(row.get("passed"))
        or strict_true(row.get("v3_pass"))
        or strict_true(row.get("differential_pass"))
        or strict_true(row.get("trace_match"))
        or strict_true(row.get("symbolic_match"))
        or strict_true(q(row, "validation", "V3", "passed"))
    )


def function_key(row: Dict[str, Any]) -> str:
    return str(row.get("function_key") or f"{row.get('input_line')}:{row.get('func_name')}")


def stable_float_key(row: Dict[str, Any], seed: int) -> str:
    raw = "|".join(
        str(x)
        for x in (
            seed,
            row.get("function_key"),
            row.get("input_line"),
            row.get("func_name"),
            row.get("slot_id"),
            row.get("candidate_id"),
            row.get("candidate_text"),
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def compact_counter(counter: collections.Counter, *, limit: int = 50) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def load_pass_rows(path: Path) -> List[Dict[str, Any]]:
    return [row for _line_no, row in iter_jsonl(path) if is_pass(row)]


def load_strong_audit(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(p):
        out[function_key(row)] = row
    return out


def result_task_key(row: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any]:
    return (
        row.get("task_id"),
        row.get("function_key"),
        row.get("slot_id"),
        row.get("candidate_id"),
        row.get("candidate_text"),
    )


def load_tasks(path: Optional[str]) -> Dict[Tuple[Any, Any, Any, Any, Any], Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {}
    out: Dict[Tuple[Any, Any, Any, Any, Any], Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(p):
        out[result_task_key(row)] = row
    return out


def list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def task_context(task: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    contract = task.get("slot_contract") if isinstance(task.get("slot_contract"), dict) else {}
    control = contract.get("control_contract") if isinstance(contract.get("control_contract"), dict) else {}
    data = contract.get("data_contract") if isinstance(contract.get("data_contract"), dict) else {}
    effect = contract.get("effect_contract") if isinstance(contract.get("effect_contract"), dict) else {}
    grounding = data.get("ssa_grounding") if isinstance(data.get("ssa_grounding"), dict) else {}
    cond_prov = grounding.get("condition_provenance") if isinstance(grounding.get("condition_provenance"), dict) else {}
    ssa_def = cond_prov.get("ssa_def") if isinstance(cond_prov.get("ssa_def"), dict) else {}
    v2e = q(task, "candidate_validation", "V2", "evidence") or {}
    if not isinstance(v2e, dict):
        v2e = {}
    return {
        "task_id": task.get("task_id"),
        "candidate_validation_v2_status": q(task, "candidate_validation", "V2", "status"),
        "candidate_validation_v2_reason": q(task, "candidate_validation", "V2", "reason"),
        "candidate_validation_v2_warnings": v2e.get("warnings"),
        "control_true_successor": control.get("true_successor"),
        "control_false_successor": control.get("false_successor"),
        "control_edge": control.get("edge"),
        "control_allowed_effects": control.get("allowed_control_effects"),
        "data_status": data.get("status"),
        "data_fact_state": data.get("fact_state"),
        "data_condition_ssa": q(data, "ssa_grounding", "condition_ssa") or data.get("condition_ssa") or data.get("branch_condition_ssa"),
        "condition_def_status": ssa_def.get("status"),
        "condition_def_instruction": ssa_def.get("defined_by_instruction"),
        "condition_def_text": ssa_def.get("def_text"),
        "condition_def_mnemonic": ssa_def.get("def_mnemonic"),
        "branch_instruction": cond_prov.get("branch_instruction"),
        "branch_text": cond_prov.get("branch_text"),
        "branch_mnemonic": cond_prov.get("branch_mnemonic"),
        "branch_polarity_available": cond_prov.get("true_false_polarity_available"),
        "data_return_ssa": q(data, "ssa_grounding", "return_ssa") or data.get("return_ssa") or data.get("return_value_ssa"),
        "data_live_in_count": list_count(data.get("live_in")),
        "data_live_out_count": list_count(data.get("live_out")),
        "data_defs_count": list_count(data.get("defs")),
        "data_uses_count": list_count(data.get("uses")),
        "effect_status": effect.get("status"),
        "effect_fact_state": effect.get("fact_state"),
        "effect_read_set_count": list_count(effect.get("read_set")),
        "effect_write_set_count": list_count(effect.get("write_set")),
        "effect_call_effect_count": list_count(effect.get("call_effects")),
    }


def row_brief(
    row: Dict[str, Any],
    strong_row: Optional[Dict[str, Any]] = None,
    task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    out = {
        "function_key": row.get("function_key"),
        "input_line": row.get("input_line"),
        "func_name": row.get("func_name"),
        "slot_id": row.get("slot_id"),
        "slot_type": row.get("slot_type"),
        "candidate_id": row.get("candidate_id"),
        "proof_kind": proof_kind(row),
        "reason": row.get("reason"),
        "candidate_text": row.get("candidate_text"),
        "reference_text": row.get("reference_text") or row.get("target_text"),
        "condition_ssa": ev.get("condition_ssa"),
        "return_ssa": ev.get("return_ssa"),
        "aliases": ev.get("aliases"),
        "candidate_source": ev.get("candidate_source"),
        "rewrite_kind": ev.get("rewrite_kind"),
        "strong_tier": strong_row.get("tier") if strong_row else None,
        "strong_reason": strong_row.get("reason") if strong_row else None,
        "manual_verdict": "",
        "manual_reason": "",
    }
    out.update(task_context(task))
    return out


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_pass_rows(Path(args.v3_results_jsonl).expanduser().resolve())
    strong_by_func = load_strong_audit(args.strong_audit_jsonl)
    task_by_key = load_tasks(args.tasks_jsonl)

    proof_dist = collections.Counter(proof_kind(row) or "unknown" for row in rows)
    slot_type_dist = collections.Counter(str(row.get("slot_type") or "UNKNOWN") for row in rows)
    reason_dist = collections.Counter(str(row.get("reason") or "unknown") for row in rows)
    funcs_all = {function_key(row) for row in rows}
    funcs_strict = {function_key(row) for row in rows if proof_kind(row) in STRICT_PROOF_KINDS}
    funcs_contract = {function_key(row) for row in rows if proof_kind(row) in CONTRACT_PROOF_KINDS}
    funcs_contract_verified = {function_key(row) for row in rows if proof_kind(row) == VERIFIED_CONTRACT_PROOF_KIND}
    funcs_contract_only = funcs_contract - funcs_strict

    contract_rows = [row for row in rows if proof_kind(row) in CONTRACT_PROOF_KINDS]
    strict_rows = [row for row in rows if proof_kind(row) in STRICT_PROOF_KINDS]
    sample_rows = sorted(contract_rows, key=lambda r: stable_float_key(r, int(args.seed)))[: int(args.sample_size)]
    sample = [
        row_brief(row, strong_by_func.get(function_key(row)), task_by_key.get(result_task_key(row)))
        for row in sample_rows
    ]

    out_jsonl = Path(args.out_sample_jsonl).expanduser().resolve()
    out_md = Path(args.out_sample_md).expanduser().resolve()
    out_report = Path(args.out_report).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    md_lines = [
        "# Module C ECSL Contract-backed V3 Audit Sample",
        "",
        f"- v3_results_jsonl: `{args.v3_results_jsonl}`",
        f"- sample_size: {len(sample)}",
        f"- seed: {args.seed}",
        "",
        "Review verdict options: `pass`, `fail`, `uncertain`.",
        "",
    ]
    for i, row in enumerate(sample, 1):
        md_lines.extend(
            [
                f"## {i}. {row.get('func_name')} / {row.get('slot_id')}",
                "",
                f"- input_line: `{row.get('input_line')}`",
                f"- slot_type: `{row.get('slot_type')}`",
                f"- proof_kind: `{row.get('proof_kind')}`",
                f"- reason: `{row.get('reason')}`",
                f"- condition_ssa: `{row.get('condition_ssa')}`",
                f"- return_ssa: `{row.get('return_ssa')}`",
                f"- aliases: `{row.get('aliases')}`",
                f"- task_id: `{row.get('task_id')}`",
                f"- V2: `{row.get('candidate_validation_v2_status')}` / `{row.get('candidate_validation_v2_reason')}`",
                f"- V2 warnings: `{row.get('candidate_validation_v2_warnings')}`",
                f"- control true/false: `{row.get('control_true_successor')}` / `{row.get('control_false_successor')}`",
                f"- control edge: `{row.get('control_edge')}`",
                f"- data condition_ssa: `{row.get('data_condition_ssa')}`",
                f"- condition def: `{row.get('condition_def_instruction')}` / `{row.get('condition_def_text')}`",
                f"- branch instruction: `{row.get('branch_instruction')}` / `{row.get('branch_text')}`",
                f"- branch polarity available: `{row.get('branch_polarity_available')}`",
                f"- data return_ssa: `{row.get('data_return_ssa')}`",
                f"- data counts live_in/live_out/defs/uses: `{row.get('data_live_in_count')}/{row.get('data_live_out_count')}/{row.get('data_defs_count')}/{row.get('data_uses_count')}`",
                f"- effect counts read/write/call: `{row.get('effect_read_set_count')}/{row.get('effect_write_set_count')}/{row.get('effect_call_effect_count')}`",
                "",
                "candidate_text:",
                "```c",
                str(row.get("candidate_text") or ""),
                "```",
                "",
                "reference_text:",
                "```c",
                str(row.get("reference_text") or ""),
                "```",
                "",
                "manual_verdict:",
                "",
                "manual_reason:",
                "",
            ]
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    report = {
        "v3_results_jsonl": str(Path(args.v3_results_jsonl).expanduser().resolve()),
        "tasks_jsonl": str(Path(args.tasks_jsonl).expanduser().resolve()) if args.tasks_jsonl else None,
        "strong_audit_jsonl": str(Path(args.strong_audit_jsonl).expanduser().resolve()) if args.strong_audit_jsonl else None,
        "passed_slot_rows": len(rows),
        "strict_slot_rows": len(strict_rows),
        "contract_backed_slot_rows": len(contract_rows),
        "functions_with_any_pass": len(funcs_all),
        "functions_with_strict_pass": len(funcs_strict),
        "functions_with_contract_backed_pass": len(funcs_contract),
        "functions_with_verified_contract_backed_pass": len(funcs_contract_verified),
        "functions_with_contract_only_pass": len(funcs_contract_only),
        "proof_kind_dist": compact_counter(proof_dist),
        "slot_type_dist": compact_counter(slot_type_dist),
        "reason_dist": compact_counter(reason_dist),
        "sample_jsonl": str(out_jsonl),
        "sample_md": str(out_md),
        "interpretation": [
            "strict_pass excludes contract_backed_symbolic and is the conservative differential/text-symbolic baseline.",
            "contract_only functions depend on slot_contract SSA bridge; verified contract rows include condition def/branch provenance, but still should be audited before claiming candidate_validation_precision.",
            "source projection self-proof is not counted here unless it appears as a passing V3 result, which should be investigated.",
        ],
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-results-jsonl", required=True)
    ap.add_argument("--tasks-jsonl", default=None)
    ap.add_argument("--strong-audit-jsonl", default=None)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-sample-jsonl", required=True)
    ap.add_argument("--out-sample-md", required=True)
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260620)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = build(args)
    print(f"passed_slot_rows = {report['passed_slot_rows']}")
    print(f"strict_slot_rows = {report['strict_slot_rows']}")
    print(f"contract_backed_slot_rows = {report['contract_backed_slot_rows']}")
    print(f"functions_with_any_pass = {report['functions_with_any_pass']}")
    print(f"functions_with_strict_pass = {report['functions_with_strict_pass']}")
    print(f"functions_with_contract_backed_pass = {report['functions_with_contract_backed_pass']}")
    print(f"functions_with_verified_contract_backed_pass = {report['functions_with_verified_contract_backed_pass']}")
    print(f"functions_with_contract_only_pass = {report['functions_with_contract_only_pass']}")
    print(f"proof_kind_dist = {report['proof_kind_dist']}")
    print(f"saved_report = {report['sample_jsonl']}")
    print(f"saved_audit_md = {report['sample_md']}")
    print(f"saved_summary_json = {args.out_report}")


if __name__ == "__main__":
    main()

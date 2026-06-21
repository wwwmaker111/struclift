#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a small Module-C/D slot-task set from Module-B v3 output.

Module-B v3 emits a verified skeleton: control structure plus fillable slots
grounded to BBs, instructions, SSA, and CFG edges.  Module C should not receive
source code at inference time.  This exporter creates the bridge format that
Modules C/D need for a first smoke test:

* the verified B-v3 skeleton;
* one task per slot, with slot type, BB grounding, instruction/SSA context;
* a render policy telling Module D how the slot should be filled or templated.

It intentionally does not construct source labels.  Source/teacher labels can
be added later as a teacher-forced training target, but they are separate from
the binary-only inference input.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


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


def selected_report(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "validator", "selected_report") or q(row, "verified_skeleton", "validator_report") or {}


def ranker_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "teacher_diagnostics", "ranker_metrics") or {}


def verified_skeleton(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("verified_skeleton") if isinstance(row.get("verified_skeleton"), dict) else {}


def function_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("function_ir") if isinstance(row.get("function_ir"), dict) else {}


def slot_pointers(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    grounding = q(row, "verified_skeleton", "slot_grounding") or {}
    out: Dict[int, Dict[str, Any]] = {}
    for ptr in grounding.get("pointers") or []:
        if not isinstance(ptr, dict):
            continue
        sid = safe_int(ptr.get("slot_id"), -1)
        if sid >= 0:
            out[sid] = ptr
    return out


def bb_count(row: Dict[str, Any]) -> int:
    bbs = function_ir(row).get("basic_blocks") or []
    return len(bbs) if isinstance(bbs, list) else 0


def hard_contract_ok(row: Dict[str, Any]) -> bool:
    rep = selected_report(row)
    return bool(
        rep.get("valid")
        and rep.get("bb_coverage_ok")
        and (rep.get("edge_preservation_ok") or rep.get("strict_edge_preservation_ok"))
        and rep.get("slot_grounding_ok")
        and rep.get("parse_success")
        and not rep.get("fatal_errors")
    )


def doc_quality_ready(row: Dict[str, Any]) -> bool:
    rm = ranker_metrics(row)
    if "doc_quality_ready" in rm:
        return bool(rm.get("doc_quality_ready"))
    return bool(q(row, "document_metric_contract", "rank_quality", "doc_quality_ready"))


def is_ok_bucket(row: Dict[str, Any]) -> bool:
    buckets = row.get("failure_buckets") or []
    return list(buckets) == ["ok"]


def slot_counts(row: Dict[str, Any]) -> collections.Counter:
    return collections.Counter(
        str(s.get("slot_type") or "UNKNOWN")
        for s in (verified_skeleton(row).get("slots") or [])
        if isinstance(s, dict)
    )


def row_score(row: Dict[str, Any]) -> Tuple[Any, ...]:
    rep = selected_report(row)
    rm = ranker_metrics(row)
    grounding = q(row, "verified_skeleton", "slot_grounding") or {}
    counts = slot_counts(row)
    n = bb_count(row)
    fallback = safe_int(rep.get("fallback_bb_count"))
    goto = safe_int(rep.get("goto_count"))
    escape = safe_int(rep.get("escape_count"), goto)
    cond_ground = safe_float(grounding.get("condition_slot_grounded_rate"), 0.0)
    branch_cov = safe_float(grounding.get("branch_provenance_coverage"), 0.0)
    structured = safe_float(rm.get("structured_bb_ratio"), safe_float(rep.get("structured_block_coverage"), 0.0))
    safe_for_old_c = bool(q(row, "module_c_contract", "safe_for_modulec"))

    return (
        int(hard_contract_ok(row)),
        int(is_ok_bucket(row)),
        int(doc_quality_ready(row)),
        int(fallback == 0),
        int(goto == 0),
        int(escape == 0),
        int(cond_ground >= 0.999),
        int(branch_cov >= 0.999),
        int(safe_for_old_c),
        structured,
        -counts["UNSTRUCTURED_REGION"],
        -counts["GOTO"],
        -counts["BREAK"],
        -counts["CONTINUE"],
        -n,
        -len(verified_skeleton(row).get("slots") or []),
    )


def select_rows(
    rows: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    max_examples: int,
    max_bb: int,
    require_ok_bucket: bool,
    exclude_candidate_regex: str,
    require_no_escape: bool,
    require_modulec_safe: bool,
    require_cfillable_ranker_ready: bool,
    require_teacher_render_ready: bool,
    require_cond_match: bool,
    require_high_confidence: bool,
) -> List[Tuple[int, Dict[str, Any], str]]:
    exclude_re = re.compile(exclude_candidate_regex) if exclude_candidate_regex else None
    strong: List[Tuple[int, Dict[str, Any], str]] = []
    fallback: List[Tuple[int, Dict[str, Any], str]] = []
    for line_no, row in rows:
        if not hard_contract_ok(row):
            continue
        if not verified_skeleton(row).get("skeleton"):
            continue
        if not verified_skeleton(row).get("slots"):
            continue
        if max_bb > 0 and bb_count(row) > max_bb:
            continue
        if require_ok_bucket and not is_ok_bucket(row):
            continue
        cand_name = str((verified_skeleton(row).get("candidate_name") or ""))
        if exclude_re and exclude_re.search(cand_name):
            continue
        if require_modulec_safe and not bool(q(row, "module_c_contract", "safe_for_modulec")):
            continue
        rm = ranker_metrics(row)
        if require_high_confidence:
            if not bool(rm.get("rank_trainable")):
                continue
            if not bool(rm.get("c_fillable_ranker_ready")):
                continue
            if not bool(rm.get("modulec_teacher_render_structural_ready")):
                continue
            if not bool(rm.get("modulec_cond_count_match")):
                continue
            if safe_float(rm.get("modulec_cond_slot_match_ratio"), 0.0) < 0.98:
                continue
            if safe_int(rm.get("modulec_unsafe_nonempty_slot_count")) > 0:
                continue
            if safe_int(rm.get("modulec_missing_teacher_primary_count")) > 0:
                continue
        if require_cfillable_ranker_ready and not bool(rm.get("c_fillable_ranker_ready")):
            continue
        if require_teacher_render_ready and not bool(rm.get("modulec_teacher_render_structural_ready")):
            continue
        if require_cond_match and safe_float(rm.get("modulec_cond_slot_match_ratio"), 0.0) < 0.98:
            continue

        rep = selected_report(row)
        no_escape = (
            safe_int(rep.get("fallback_bb_count")) == 0
            and safe_int(rep.get("goto_count")) == 0
            and safe_int(rep.get("escape_count"), safe_int(rep.get("goto_count"))) == 0
        )
        if require_no_escape and not no_escape:
            continue
        if is_ok_bucket(row) and doc_quality_ready(row) and no_escape:
            strong.append((line_no, row, "strict_high_confidence"))
        else:
            fallback.append((line_no, row, "relaxed_valid"))

    strong.sort(key=lambda item: row_score(item[1]), reverse=True)
    fallback.sort(key=lambda item: row_score(item[1]), reverse=True)
    selected = strong[:max_examples]
    if len(selected) < max_examples:
        selected.extend(fallback[: max_examples - len(selected)])
    return selected


def edge_context_for_bbs(fi: Dict[str, Any], bbs: Sequence[int]) -> Dict[str, Any]:
    bb_set = {safe_int(x, -1) for x in bbs}
    incoming: List[Dict[str, Any]] = []
    outgoing: List[Dict[str, Any]] = []
    internal: List[Dict[str, Any]] = []
    for edge in fi.get("cfg_edges") or []:
        if not isinstance(edge, dict):
            continue
        src = safe_int(edge.get("src"), -1)
        dst = safe_int(edge.get("dst"), -1)
        slim = {
            "src": src,
            "dst": dst,
            "kind": edge.get("kind"),
            "condition_ssa": edge.get("condition_ssa"),
            "provenance_instruction": edge.get("provenance_instruction"),
            "confidence": edge.get("confidence"),
        }
        if src in bb_set and dst in bb_set:
            internal.append(slim)
        elif src in bb_set:
            outgoing.append(slim)
        elif dst in bb_set:
            incoming.append(slim)
    return {"incoming": incoming, "outgoing": outgoing, "internal": internal}


def lifted_slice(fi: Dict[str, Any], start: int, end: int, max_items: int) -> List[Any]:
    lifted = fi.get("lifted_instructions") or []
    if not isinstance(lifted, list) or start < 0 or end <= start:
        return []
    return lifted[start : min(end, start + max_items)]


def bb_context(fi: Dict[str, Any], bbs: Sequence[int], *, max_instructions_per_bb: int) -> List[Dict[str, Any]]:
    blocks = fi.get("basic_blocks") or []
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw_bb in bbs:
        bb = safe_int(raw_bb, -1)
        if bb in seen or not (0 <= bb < len(blocks)):
            continue
        seen.add(bb)
        block = blocks[bb] if isinstance(blocks[bb], dict) else {}
        start, end = 0, 0
        rng = block.get("instruction_index_range")
        if isinstance(rng, list) and len(rng) >= 2:
            start, end = safe_int(rng[0]), safe_int(rng[1])
        opcodes = block.get("opcode_ids") if isinstance(block.get("opcode_ids"), list) else []
        operands_t = block.get("operand_type_ids") if isinstance(block.get("operand_type_ids"), list) else []
        operands_v = block.get("operand_value_ids") if isinstance(block.get("operand_value_ids"), list) else []
        out.append(
            {
                "bb_id": bb,
                "stable_id": block.get("stable_id") or f"bb_{bb}",
                "address_range": block.get("address_range"),
                "terminator": block.get("terminator"),
                "instruction_index_range": [start, end],
                "instruction_count": safe_int(block.get("instruction_count"), max(0, end - start)),
                "instruction_ids": list(block.get("instruction_ids") or [])[:max_instructions_per_bb],
                "opcode_ids": opcodes[:max_instructions_per_bb],
                "operand_type_ids": operands_t[:max_instructions_per_bb],
                "operand_value_ids": operands_v[:max_instructions_per_bb],
                "lifted_instructions_preview": lifted_slice(fi, start, end, max_instructions_per_bb),
            }
        )
    return out


def module_c_policy(slot_type: str) -> Dict[str, Any]:
    typ = str(slot_type or "")
    if typ == "COND":
        return {
            "module_c_fill_type": "FILL_COND",
            "trainable": True,
            "required_for_render": True,
            "prediction_target": "c_condition_expression",
            "label_policy": "predict_from_condition_ssa_branch_instruction_and_true_false_edges",
            "module_d_render_action": "replace_fill_cond",
        }
    if typ == "BLOCK":
        return {
            "module_c_fill_type": "FILL_STMT",
            "trainable": True,
            "required_for_render": True,
            "prediction_target": "c_statement_sequence",
            "label_policy": "predict_from_primary_bb_instruction_span_and_dataflow",
            "module_d_render_action": "replace_fill_block",
        }
    if typ == "CASE":
        return {
            "module_c_fill_type": "TEMPLATE_CASE",
            "trainable": False,
            "required_for_render": True,
            "prediction_target": "case_or_default_label_plus_case_body_template",
            "label_policy": "do_not_project_raw_source_lines; render case/default from switch metadata or keep labeled case placeholder",
            "module_d_render_action": "emit_case_template",
        }
    if typ == "UNSTRUCTURED_REGION":
        return {
            "module_c_fill_type": "FILL_STMT",
            "trainable": False,
            "required_for_render": True,
            "prediction_target": "labeled_cfg_pseudocode_or_conservative_goto_region",
            "label_policy": "skip_free_training_in_first_smoke; render_as_labeled_cfg_region",
            "module_d_render_action": "emit_unstructured_region",
        }
    if typ in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
        return {
            "module_c_fill_type": f"TEMPLATE_{typ}",
            "trainable": False,
            "required_for_render": True,
            "prediction_target": "control_transfer_template",
            "label_policy": "do_not_train_free_text; render_from_cfg_edge",
            "module_d_render_action": f"emit_{typ.lower()}_template",
        }
    return {
        "module_c_fill_type": "FILL_STMT",
        "trainable": False,
        "required_for_render": False,
        "prediction_target": "unknown_slot",
        "label_policy": "skip_unknown_slot_type",
        "module_d_render_action": "leave_placeholder_or_comment",
    }


def make_slot_task(
    row: Dict[str, Any],
    slot: Dict[str, Any],
    pointer: Optional[Dict[str, Any]],
    *,
    max_instructions_per_bb: int,
) -> Dict[str, Any]:
    fi = function_ir(row)
    sid = safe_int(slot.get("slot_id"), -1)
    primary = [safe_int(x, -1) for x in (slot.get("primary_bbs") or []) if safe_int(x, -1) >= 0]
    context = [safe_int(x, -1) for x in (slot.get("context_bbs") or primary) if safe_int(x, -1) >= 0]
    bb_for_mask = primary or context
    typ = str(slot.get("slot_type") or "")
    policy = module_c_policy(typ)
    ptr = dict(pointer or {})
    out = {
        "slot_id": sid,
        "slot_ref": f"slot_{sid:03d}" if sid >= 0 else str(slot.get("slot_id")),
        "b3_slot_type": typ,
        "role": slot.get("role") or "",
        "owner": slot.get("owner"),
        "reason": slot.get("reason") or "",
        "primary_bbs": primary,
        "context_bbs": context,
        "slot_mask_bb_indices": bb_for_mask,
        "edge": slot.get("edge"),
        "module_c": policy,
        "grounding": {
            "pointer_available": bool(ptr),
            "grounding_ok": bool(ptr.get("grounding_ok")) if ptr else None,
            "condition_bb": ptr.get("condition_bb"),
            "condition_ssa": ptr.get("condition_ssa"),
            "branch_instruction": ptr.get("branch_instruction"),
            "branch_instruction_index": ptr.get("branch_instruction_index"),
            "branch_mnemonic": ptr.get("branch_mnemonic"),
            "branch_address": ptr.get("branch_address"),
            "branch_metadata_source": ptr.get("branch_metadata_source"),
            "branch_metadata_available": ptr.get("branch_metadata_available"),
            "branch_provenance_available": ptr.get("branch_provenance_available"),
            "true_false_polarity_available": ptr.get("true_false_polarity_available"),
            "outgoing_edges": ptr.get("outgoing_edges") or [],
            "function_ir_outgoing_edges": ptr.get("function_ir_outgoing_edges") or [],
            "instruction_ranges": ptr.get("instruction_ranges") or [],
            "instruction_count": ptr.get("instruction_count"),
            "real_ssa_available": ptr.get("real_ssa_available"),
        },
        "binary_context": {
            "basic_blocks": bb_context(fi, bb_for_mask, max_instructions_per_bb=max_instructions_per_bb),
            "cfg_edges": edge_context_for_bbs(fi, bb_for_mask),
        },
        "teacher_label": {
            "available": False,
            "label_text": None,
            "note": (
                "source is not an inference input; add labels only in a separate "
                "teacher-forced projection step if needed"
            ),
        },
    }
    return out


def compact_function_ir(fi: Dict[str, Any], *, max_lifted: int) -> Dict[str, Any]:
    out = {
        "schema": fi.get("schema") or "FunctionIR.v1",
        "function_id": fi.get("function_id"),
        "function_name": fi.get("function_name"),
        "source_path": fi.get("source_path"),
        "binary_path": fi.get("binary_path"),
        "basic_blocks": fi.get("basic_blocks") or [],
        "cfg_edges": fi.get("cfg_edges") or [],
        "ssa_level": fi.get("ssa_level"),
        "availability_source": fi.get("availability_source"),
        "branch_metadata": fi.get("branch_metadata") or {},
        "instruction_provenance": fi.get("instruction_provenance") or {},
        "def_use_graph": fi.get("def_use_graph") or {},
        "memory_access_summary": fi.get("memory_access_summary") or {},
    }
    lifted = fi.get("lifted_instructions") or []
    if isinstance(lifted, list):
        out["lifted_instructions_preview"] = lifted[:max_lifted]
        out["lifted_instruction_count"] = len(lifted)
    else:
        out["lifted_instructions_preview"] = []
        out["lifted_instruction_count"] = 0
    return out


def make_task_record(
    line_no: int,
    row: Dict[str, Any],
    selection_reason: str,
    *,
    max_instructions_per_bb: int,
    max_lifted_preview: int,
) -> Dict[str, Any]:
    vs = verified_skeleton(row)
    fi = function_ir(row)
    pointers = slot_pointers(row)
    slots = [s for s in (vs.get("slots") or []) if isinstance(s, dict)]
    slot_tasks = [
        make_slot_task(
            row,
            slot,
            pointers.get(safe_int(slot.get("slot_id"), -1)),
            max_instructions_per_bb=max_instructions_per_bb,
        )
        for slot in slots
    ]
    rep = selected_report(row)
    rm = ranker_metrics(row)
    grounding = vs.get("slot_grounding") or {}
    return {
        "schema": "ModuleCDSlotTaskFromB3.v1",
        "source": {
            "b3_input_line": line_no,
            "b3_example": row.get("example"),
            "b3_schema_version": row.get("schema_version"),
            "selection_reason": selection_reason,
        },
        "func_name": row.get("func_name") or fi.get("function_name"),
        "function_ir": compact_function_ir(fi, max_lifted=max_lifted_preview),
        "verified_skeleton": {
            "schema": vs.get("schema") or "VerifiedSkeleton.v1",
            "candidate_id": vs.get("candidate_id"),
            "candidate_name": vs.get("candidate_name"),
            "selection_source": vs.get("selection_source"),
            "skeleton": vs.get("skeleton"),
            "outline": vs.get("outline") or [],
            "escape_edges": vs.get("escape_edges") or {},
            "fallback_regions": vs.get("fallback_regions") or [],
            "validator_report": rep,
        },
        "module_b_quality": {
            "hard_contract_ok": hard_contract_ok(row),
            "failure_buckets": row.get("failure_buckets") or [],
            "doc_quality_ready": doc_quality_ready(row),
            "second_version_ready": rm.get("second_version_ready"),
            "c_fillable_ranker_ready": rm.get("c_fillable_ranker_ready"),
            "rank_trainable": rm.get("rank_trainable"),
            "rank_label_tier": rm.get("rank_label_tier"),
            "rank_label_status": rm.get("rank_label_status"),
            "rank_label_confidence": rm.get("rank_label_confidence"),
            "modulec_safe_for_rank": rm.get("modulec_safe_for_rank"),
            "modulec_teacher_render_structural_ready": rm.get("modulec_teacher_render_structural_ready"),
            "modulec_cond_slot_match_ratio": rm.get("modulec_cond_slot_match_ratio"),
            "modulec_cond_count_match": rm.get("modulec_cond_count_match"),
            "modulec_c_fillable_score": rm.get("modulec_c_fillable_score"),
            "modulec_source_like_score": rm.get("modulec_source_like_score"),
            "modulec_ast_source_like_score": rm.get("modulec_ast_source_like_score"),
            "source_cfg_metrics_used_for_rank": rm.get("source_cfg_metrics_used_for_rank"),
            "source_cfg_score": rm.get("source_cfg_score"),
            "source_cfg_struct_recall": rm.get("source_cfg_struct_recall"),
            "source_cfg_struct_f1": rm.get("source_cfg_struct_f1"),
            "modulec_unsafe_nonempty_slot_count": rm.get("modulec_unsafe_nonempty_slot_count"),
            "modulec_missing_teacher_primary_count": rm.get("modulec_missing_teacher_primary_count"),
            "modulec_case_slot_gap": rm.get("modulec_case_slot_gap"),
            "ranker_name": rm.get("name"),
            "structured_bb_ratio": rm.get("structured_bb_ratio"),
            "goto_count": rep.get("goto_count"),
            "fallback_bb_count": rep.get("fallback_bb_count"),
            "escape_count": rep.get("escape_count"),
            "slot_grounding_pointer_acc": grounding.get("slot_grounding_pointer_acc"),
            "condition_slot_grounded_rate": grounding.get("condition_slot_grounded_rate"),
            "branch_provenance_coverage": grounding.get("branch_provenance_coverage"),
            "real_condition_ssa_available_rate": grounding.get("real_condition_ssa_available_rate"),
        },
        "module_c_slots": slot_tasks,
        "module_d_contract": {
            "render_input": "verified_skeleton.skeleton + module_c slot predictions",
            "render_rule": (
                "replace FILL_COND/FILL_BLOCK placeholders with Module-C predictions; "
                "render escape/fallback slots from CFG templates unless explicitly enabled"
            ),
            "checks": [
                "no_unfilled_required_placeholder",
                "c_syntax_or_compile_reward",
                "cfg_structural_check_against_verified_skeleton",
                "optional_trace_or_unit_semantic_reward",
            ],
        },
        "inference_policy": {
            "source_code_allowed": False,
            "source_code_note": (
                "full source may be used only to construct teacher-forced labels "
                "or evaluate output, never as Module-C inference input"
            ),
        },
    }


def summarize(records: Sequence[Dict[str, Any]], *, total_seen: int, selected_lines: Sequence[int]) -> str:
    lines: List[str] = []
    n = len(records)
    lines.append("=== Module C/D slot-task export from Module B-v3 ===")
    lines.append(f"input_rows_seen = {total_seen}")
    lines.append(f"selected_rows = {n}")
    lines.append(f"selected_input_lines = {list(selected_lines)}")
    lines.append("")
    lines.append("-- selected functions --")
    for idx, rec in enumerate(records, 1):
        ql = rec.get("module_b_quality") or {}
        slots = rec.get("module_c_slots") or []
        c = collections.Counter(((s.get("module_c") or {}).get("module_c_fill_type") for s in slots))
        lines.append(
            f"{idx:02d}. func={rec.get('func_name')} "
            f"n_bb={len(((rec.get('function_ir') or {}).get('basic_blocks') or []))} "
            f"candidate={(rec.get('verified_skeleton') or {}).get('candidate_name')} "
            f"slots={len(slots)} fill_types={sorted(c.items())} "
            f"doc_ready={ql.get('doc_quality_ready')} "
            f"cfillable={ql.get('c_fillable_ranker_ready')} "
            f"modulec_safe={ql.get('modulec_safe_for_rank')} "
            f"render_ready={ql.get('modulec_teacher_render_structural_ready')} "
            f"buckets={ql.get('failure_buckets')}"
        )
    lines.append("")
    all_slots = [s for rec in records for s in (rec.get("module_c_slots") or [])]
    fill_counter = collections.Counter((s.get("module_c") or {}).get("module_c_fill_type") for s in all_slots)
    b3_counter = collections.Counter(s.get("b3_slot_type") for s in all_slots)
    trainable = sum(1 for s in all_slots if (s.get("module_c") or {}).get("trainable"))
    required = sum(1 for s in all_slots if (s.get("module_c") or {}).get("required_for_render"))
    cond_slots = [s for s in all_slots if (s.get("module_c") or {}).get("module_c_fill_type") == "FILL_COND"]
    grounded_cond = sum(1 for s in cond_slots if (s.get("grounding") or {}).get("condition_ssa"))
    branch_prov = sum(1 for s in cond_slots if (s.get("grounding") or {}).get("branch_provenance_available"))
    block_slots = [s for s in all_slots if (s.get("module_c") or {}).get("module_c_fill_type") == "FILL_STMT" and s.get("b3_slot_type") == "BLOCK"]
    block_instr = [
        safe_int((s.get("grounding") or {}).get("instruction_count"))
        for s in block_slots
    ]
    lines.append("-- slot summary --")
    lines.append(f"total_slots = {len(all_slots)}")
    lines.append(f"b3_slot_type_dist = {sorted(b3_counter.items())}")
    lines.append(f"module_c_fill_type_dist = {sorted(fill_counter.items())}")
    lines.append(
        f"c_fillable_ranker_ready_rows = {pct(sum(1 for r in records if (r.get('module_b_quality') or {}).get('c_fillable_ranker_ready')), n)}"
    )
    lines.append(
        f"rank_trainable_rows = {pct(sum(1 for r in records if (r.get('module_b_quality') or {}).get('rank_trainable')), n)}"
    )
    lines.append(
        f"modulec_safe_for_rank_rows = {pct(sum(1 for r in records if (r.get('module_b_quality') or {}).get('modulec_safe_for_rank')), n)}"
    )
    lines.append(
        "teacher_render_structural_ready_rows = "
        f"{pct(sum(1 for r in records if (r.get('module_b_quality') or {}).get('modulec_teacher_render_structural_ready')), n)}"
    )
    lines.append(
        "modulec_cond_count_match_rows = "
        f"{pct(sum(1 for r in records if (r.get('module_b_quality') or {}).get('modulec_cond_count_match')), n)}"
    )
    lines.append(f"trainable_slots = {pct(trainable, len(all_slots))}")
    lines.append(f"required_for_render_slots = {pct(required, len(all_slots))}")
    lines.append(f"cond_slots_with_condition_ssa = {pct(grounded_cond, len(cond_slots))}")
    lines.append(f"cond_slots_with_branch_provenance = {pct(branch_prov, len(cond_slots))}")
    if block_instr:
        lines.append(f"block_slot_instruction_count_mean = {statistics.mean(block_instr):.2f}")
        lines.append(f"block_slot_instruction_count_median = {statistics.median(block_instr):.2f}")
    lines.append("")
    lines.append("-- how to use this file --")
    lines.append("1. Module C inference input = function_ir + verified_skeleton + module_c_slots.")
    lines.append("2. Do not feed full source to Module C during inference.")
    lines.append("3. For teacher-forced debugging, add source-projected label_text per slot in a separate step.")
    lines.append("4. Module D renders slot predictions into the B-v3 skeleton and runs compile/semantic checks.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True, help="Module-B v3 full-pipeline JSONL")
    ap.add_argument("--out-jsonl", required=True, help="Output Module-C/D slot-task JSONL")
    ap.add_argument("--out-txt", required=True, help="Human-readable summary")
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--max-input-rows", type=int, default=0)
    ap.add_argument("--max-bb", type=int, default=18, help="Prefer small smoke-test functions; <=0 disables")
    ap.add_argument("--allow-non-ok-bucket", action="store_true")
    ap.add_argument(
        "--allow-relaxed-label-alignment",
        action="store_true",
        help=(
            "Disable the default high-confidence C-label alignment filter. "
            "Use only for debugging; teacher-forced C/D experiments should leave it off."
        ),
    )
    ap.add_argument(
        "--exclude-candidate-regex",
        default="",
        help="Skip selected B-v3 candidates whose candidate_name matches this regex, e.g. 'switch'.",
    )
    ap.add_argument(
        "--require-no-escape",
        action="store_true",
        help="Require zero fallback/goto/escape slots in the selected B-v3 report.",
    )
    ap.add_argument(
        "--require-modulec-safe",
        action="store_true",
        help="Require module_c_contract.safe_for_modulec=True; useful for teacher-forced overfit label projection.",
    )
    ap.add_argument(
        "--require-cfillable-ranker-ready",
        action="store_true",
        help="Require ranker_metrics.c_fillable_ranker_ready=True from the B-v3 second-version target.",
    )
    ap.add_argument(
        "--require-teacher-render-ready",
        action="store_true",
        help="Require ranker_metrics.modulec_teacher_render_structural_ready=True before exporting C/D slots.",
    )
    ap.add_argument(
        "--require-cond-match",
        action="store_true",
        help="Require ranker_metrics.modulec_cond_slot_match_ratio>=0.98 before exporting C/D slots.",
    )
    ap.add_argument("--max-instructions-per-bb", type=int, default=12)
    ap.add_argument("--max-lifted-preview", type=int, default=512)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl).expanduser().resolve()
    rows = list(iter_jsonl(in_path, limit=int(args.max_input_rows)))
    selected = select_rows(
        rows,
        max_examples=int(args.max_examples),
        max_bb=int(args.max_bb),
        require_ok_bucket=not bool(args.allow_non_ok_bucket),
        exclude_candidate_regex=str(args.exclude_candidate_regex or ""),
        require_no_escape=bool(args.require_no_escape),
        require_modulec_safe=bool(args.require_modulec_safe),
        require_cfillable_ranker_ready=bool(args.require_cfillable_ranker_ready),
        require_teacher_render_ready=bool(args.require_teacher_render_ready),
        require_cond_match=bool(args.require_cond_match),
        require_high_confidence=not bool(args.allow_relaxed_label_alignment),
    )
    records = [
        make_task_record(
            line_no,
            row,
            reason,
            max_instructions_per_bb=int(args.max_instructions_per_bb),
            max_lifted_preview=int(args.max_lifted_preview),
        )
        for line_no, row, reason in selected
    ]
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    out_txt.write_text(
        summarize(records, total_seen=len(rows), selected_lines=[line_no for line_no, _row, _reason in selected]),
        encoding="utf-8",
    )
    print(f"selected = {len(records)}/{len(rows)}")
    print(f"saved_jsonl = {out_jsonl}")
    print(f"saved_txt = {out_txt}")


if __name__ == "__main__":
    main()

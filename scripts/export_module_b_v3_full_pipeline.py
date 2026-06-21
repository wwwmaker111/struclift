#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a complete auditable Module-B v3 full-pipeline package.

This script is the "one file to inspect" wrapper for the practical Module-B v3
design:

Binary CFG/SSA FunctionIR
  -> deterministic normalization and analysis
  -> candidate generation
  -> optional L_rank candidate ranker
  -> validator / canonicalizer / fallback boundary
  -> verified canonical structured skeleton
  -> Module-C slot-label constructability contract

The safety contract is still deterministic.  The learned ranker only selects
among generated candidates; it never creates CFG edges, deletes BBs, or bypasses
validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_script_module(
    "module_b_v3_second_version_eval_for_full_pipeline",
    SCRIPT_DIR / "evaluate_module_b_v3_second_version.py",
)
C_AUDIT = _load_script_module(
    "module_b_v3_c_label_constructability_for_full_pipeline",
    SCRIPT_DIR / "audit_module_b_v3_c_label_constructability.py",
)


analyse_cfg = EVAL.analyse_cfg
build_candidate_records = EVAL.build_candidate_records
find_candidate = EVAL.find_candidate
iter_jsonl = EVAL.iter_jsonl
prepare_rows = EVAL.prepare_rows
prepare_rows_with_diagnostics = getattr(EVAL, "prepare_rows_with_diagnostics", None)
rank_sort_key = EVAL.rank_sort_key
render_candidate = EVAL.render_candidate
row_func_name = EVAL.row_func_name
selected_metrics = EVAL.selected_metrics
select_with_ranker = EVAL.select_with_ranker
train_pairwise_ranker = EVAL.train_pairwise_ranker
validate_candidate = EVAL.validate_candidate

audit_candidate = C_AUDIT.audit_candidate
slots_from_candidate = C_AUDIT.slots_from_candidate
slots_from_teacher_record = C_AUDIT.slots_from_teacher_record


SCHEMA_VERSION = "module_b_v3_full_pipeline_v1"


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def mean_val(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows]
    return float(statistics.mean(vals)) if vals else 0.0


def median_val(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows]
    return float(statistics.median(vals)) if vals else 0.0


def _as_int_list(x: Any) -> List[int]:
    if not isinstance(x, list):
        return []
    out: List[int] = []
    for v in x:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return float(default)
    if value != value:
        return float(default)
    return float(value)


def _normalise_region_list(regions: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(regions, list):
        return out
    for idx, reg in enumerate(regions):
        if not isinstance(reg, dict):
            continue
        nodes = _as_int_list(reg.get("nodes"))
        out.append(
            {
                "region_id": int(idx),
                "entry": _safe_int(reg.get("entry"), -1),
                "exit": _safe_int(reg.get("exit"), -1),
                "nodes": nodes,
                "size": int(len(nodes)),
                "incoming_edges": reg.get("incoming_edges") or [],
                "outgoing_edges": reg.get("outgoing_edges") or [],
            }
        )
    return out


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonify(v) for v in obj)
    return obj


def _row_has_any(row: Dict[str, Any], keys: Sequence[str]) -> bool:
    return any(k in row and row.get(k) not in (None, [], {}, "") for k in keys)


def _edge_kind(row: Dict[str, Any], idx: int) -> Optional[Any]:
    edge_types = row.get("bin_edge_type")
    if isinstance(edge_types, list) and idx < len(edge_types):
        return edge_types[idx]
    edge_kinds = row.get("cfg_edge_kinds") or row.get("edge_kinds") or row.get("edge_types")
    if isinstance(edge_kinds, list) and idx < len(edge_kinds):
        return edge_kinds[idx]
    return None


def _edge_confidence(row: Dict[str, Any], idx: int) -> Optional[float]:
    raw = row.get("cfg_edge_confidence") or row.get("edge_confidence")
    if isinstance(raw, list) and idx < len(raw):
        try:
            return float(raw[idx])
        except Exception:
            return None
    return None


def build_function_ir(row: Dict[str, Any], *, function_id: str) -> Dict[str, Any]:
    existing = row.get("function_ir")
    if isinstance(existing, dict) and existing.get("basic_blocks") is not None and existing.get("cfg_edges") is not None:
        out = dict(existing)
        out.setdefault("schema", "FunctionIR.v1")
        out.setdefault("function_id", function_id)
        out.setdefault("function_name", row.get("func_name") or row.get("function"))
        out.setdefault("lifted_instructions", row.get("lifted_instructions") or [])
        out.setdefault("ssa_values", row.get("ssa_values") or [])
        out.setdefault("def_use_graph", row.get("def_use_graph") or {})
        out.setdefault("instruction_provenance", row.get("instruction_provenance") or {})
        out.setdefault("branch_metadata", row.get("branch_metadata") or {})
        out.setdefault("call_metadata", row.get("call_metadata") or {})
        out.setdefault("memory_access_summary", row.get("memory_access_summary") or {})
        availability_keys = {
            "lifted_instructions_available": bool(out.get("lifted_instructions")),
            "ssa_available": bool(out.get("ssa_values")),
            "def_use_available": bool(out.get("def_use_graph")),
            "instruction_provenance_available": bool(out.get("instruction_provenance")),
            "branch_metadata_available": bool(out.get("branch_metadata")),
            "call_metadata_available": bool(out.get("call_metadata")),
            "memory_access_summary_available": bool(out.get("memory_access_summary")),
        }
        for key, value in availability_keys.items():
            out.setdefault(key, value)
        out.setdefault(
            "missing_optional_inputs",
            [k.replace("_available", "") for k in availability_keys if not bool(out.get(k))],
        )
        return out

    block_lengths = _as_int_list(row.get("bin_block_lengths"))
    n_bb = len(block_lengths)
    opcodes = row.get("bin_opcodes") if isinstance(row.get("bin_opcodes"), list) else []
    operand_types = row.get("bin_operand_types") if isinstance(row.get("bin_operand_types"), list) else []
    operand_values = row.get("bin_operand_values") if isinstance(row.get("bin_operand_values"), list) else []
    raw_edges = row.get("bin_edge_index") if isinstance(row.get("bin_edge_index"), list) else [[], []]
    srcs = raw_edges[0] if len(raw_edges) == 2 and isinstance(raw_edges[0], list) else []
    dsts = raw_edges[1] if len(raw_edges) == 2 and isinstance(raw_edges[1], list) else []

    offsets: List[Tuple[int, int]] = []
    cur = 0
    for length in block_lengths:
        start = cur
        cur += max(0, int(length))
        offsets.append((start, cur))

    basic_blocks: List[Dict[str, Any]] = []
    for bb, (start, end) in enumerate(offsets):
        outgoing = [i for i, s in enumerate(srcs) if _safe_int(s, -1) == bb]
        terminator = "return" if not outgoing else ("cond_branch" if len(outgoing) >= 2 else "jump")
        instr_ids = [f"bb_{bb}:ins_{i}" for i in range(start, end)]
        basic_blocks.append(
            {
                "bb_id": int(bb),
                "stable_id": f"bb_{bb}",
                "address_range": row.get("bin_block_address_ranges", [None] * n_bb)[bb]
                if isinstance(row.get("bin_block_address_ranges"), list)
                and bb < len(row.get("bin_block_address_ranges"))
                else None,
                "instruction_ids": instr_ids,
                "instruction_index_range": [int(start), int(end)],
                "instruction_count": int(max(0, end - start)),
                "opcode_ids": opcodes[start:end] if opcodes else [],
                "operand_type_ids": operand_types[start:end] if operand_types else [],
                "operand_value_ids": operand_values[start:end] if operand_values else [],
                "terminator": terminator,
                "has_call": None,
                "has_return": bool(not outgoing),
                "side_effect_summary": [],
            }
        )

    cfg_edges: List[Dict[str, Any]] = []
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        s = _safe_int(src, -1)
        d = _safe_int(dst, -1)
        if not (0 <= s < n_bb and 0 <= d < n_bb):
            continue
        cfg_edges.append(
            {
                "edge_id": int(idx),
                "src": int(s),
                "dst": int(d),
                "src_id": f"bb_{s}",
                "dst_id": f"bb_{d}",
                "kind": _edge_kind(row, idx),
                "condition_ssa": None,
                "provenance_instruction": (
                    f"bb_{s}:ins_{offsets[s][1] - 1}"
                    if 0 <= s < len(offsets) and offsets[s][1] > offsets[s][0]
                    else None
                ),
                "confidence": _edge_confidence(row, idx),
            }
        )

    availability = {
        "lifted_instructions_available": _row_has_any(
            row, ["lifted_instructions", "ir_instructions", "llvm_ir", "vex_ir", "pcode"]
        ),
        "ssa_available": _row_has_any(row, ["ssa_values", "def_use_graph", "ssa_def_use"]),
        "def_use_available": _row_has_any(row, ["def_use_graph", "ssa_def_use", "du_edges"]),
        "instruction_provenance_available": _row_has_any(
            row, ["instruction_provenance", "source_instruction_map", "asm_instruction_ids"]
        ),
        "branch_metadata_available": _row_has_any(
            row, ["branch_metadata", "cfg_edge_kinds", "edge_kinds", "bin_edge_type"]
        ),
        "call_metadata_available": _row_has_any(row, ["call_metadata", "callsites", "call_edges"]),
        "memory_access_summary_available": _row_has_any(
            row, ["memory_access_summary", "mem_accesses", "memory_ops"]
        ),
    }
    missing = [k.replace("_available", "") for k, v in availability.items() if not v]
    return {
        "schema": "FunctionIR.v1",
        "function_id": function_id,
        "function_name": row.get("func_name") or row.get("function"),
        "source_path": row.get("source_path") or row.get("src_path") or row.get("path"),
        "binary_path": row.get("binary_path") or row.get("bin_path"),
        "basic_blocks": basic_blocks,
        "cfg_edges": cfg_edges,
        "lifted_instructions": row.get("lifted_instructions") or row.get("ir_instructions") or [],
        "ssa_values": row.get("ssa_values") or [],
        "def_use_graph": row.get("def_use_graph") or row.get("ssa_def_use") or {},
        "instruction_provenance": row.get("instruction_provenance") or {},
        "branch_metadata": row.get("branch_metadata") or {},
        "call_metadata": row.get("call_metadata") or {},
        "memory_access_summary": row.get("memory_access_summary") or {},
        "upstream_confidence": {
            "module_a_available": _row_has_any(
                row,
                [
                    "module_a_logits",
                    "module_a_confidence",
                    "pattern_logits",
                    "bb_role_logits",
                    "ctrl_logits",
                ],
            ),
            "cfg_available": bool(basic_blocks),
            "edge_available": bool(cfg_edges),
            "raw": row.get("upstream_confidence") or row.get("confidence") or None,
        },
        **availability,
        "missing_optional_inputs": missing,
    }


def _dominance_pairs(idom: Sequence[int]) -> List[Dict[str, int]]:
    return [
        {"node": int(i), "parent": int(parent)}
        for i, parent in enumerate(idom)
        if int(parent) >= 0
    ]


def _compute_sccs(n_bb: int, succs: Dict[int, List[int]]) -> List[List[int]]:
    index = 0
    stack: List[int] = []
    on_stack: set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    sccs: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in succs.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while stack:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(sorted(comp))

    for v in range(n_bb):
        if v not in indices:
            strongconnect(v)
    return sorted(sccs, key=lambda xs: (len(xs), xs))


def build_analysis_context(cfg: Any) -> Dict[str, Any]:
    loop_headers = sorted(int(x) for x in cfg.loop_nodes_by_header.keys())
    loops = [
        {
            "header": int(h),
            "nodes": sorted(int(x) for x in cfg.loop_nodes_by_header.get(h, set())),
            "back_edges": [[int(s), int(d)] for s, d in cfg.back_edges if int(d) == int(h)],
        }
        for h in loop_headers
    ]
    sccs = [
        sorted(int(x) for x in comp)
        for comp in (getattr(cfg, "sccs", None) or _compute_sccs(int(cfg.n_bb), cfg.succs))
    ]
    irreducible_sccs = [
        sorted(int(x) for x in comp)
        for comp in (getattr(cfg, "irreducible_sccs", None) or [])
    ]
    sese_regions = _normalise_region_list(getattr(cfg, "sese_regions", []))
    return {
        "schema": "NormalizeAndAnalyze.v1",
        "normalized_cfg": {
            "n_bb": int(cfg.n_bb),
            "edges": [[int(s), int(d)] for s, d in cfg.edges],
            "succs": {str(k): [int(x) for x in v] for k, v in cfg.succs.items()},
            "preds": {str(k): [int(x) for x in v] for k, v in cfg.preds.items()},
            "dfs_order": [int(x) for x in cfg.dfs_order],
            "unreachable": [int(x) for x in cfg.unreachable],
            "entry": 0 if int(cfg.n_bb) > 0 else None,
            "exits": [int(i) for i in range(int(cfg.n_bb)) if not cfg.succs.get(i)],
        },
        "dominators": {
            "idom": [int(x) for x in cfg.idom],
            "tree_edges": _dominance_pairs(cfg.idom),
        },
        "postdominators": {
            "ipdom": [int(x) for x in cfg.ipdom],
            "tree_edges": _dominance_pairs(cfg.ipdom),
        },
        "sccs": [{"scc_id": i, "nodes": comp} for i, comp in enumerate(sccs)],
        "irreducible_components": [
            {"component_id": i, "nodes": comp}
            for i, comp in enumerate(irreducible_sccs)
        ],
        "natural_loops": loops,
        "sese_regions": {
            "available": bool(sese_regions),
            "region_count": int(len(sese_regions)),
            "regions": sese_regions,
        },
        "reachability": {
            "entry_reachable_count": int(cfg.n_bb - len(cfg.unreachable)),
            "unreachable_count": int(len(cfg.unreachable)),
            "unreachable": [int(x) for x in cfg.unreachable],
        },
        "unsupported_markers": [
            "advanced_switch_recovery_is_conservative",
            "irreducible_cfg_uses_fallback_or_goto_slots",
        ],
    }


def candidate_summary(candidates: Sequence[Dict[str, Any]], *, include_all: bool) -> List[Dict[str, Any]]:
    rows = candidates if include_all else list(candidates)[:20]
    out: List[Dict[str, Any]] = []
    for c in rows:
        report = c.get("validator_report") or {}
        target = c.get("rank_target") or {}
        style = c.get("source_style") or {}
        grounding = c.get("grounding") or {}
        features = c.get("features") or {}
        out.append(
            {
                "candidate_id": c.get("candidate_id"),
                "name": c.get("name"),
                "family": EVAL._candidate_family_name(c) if hasattr(EVAL, "_candidate_family_name") else None,
                "outline": c.get("outline"),
                "valid": bool(report.get("valid")),
                "hard_valid": bool(target.get("hard_valid")),
                "target_policy": target.get("target_policy"),
                "doc_mvp_ready": bool(target.get("doc_mvp_ready")),
                "doc_quality_ready": bool(target.get("doc_quality_ready")),
                "cost_feasible": bool(target.get("cost_feasible")),
                "source_soft_recovered": bool(target.get("source_soft_recovered")),
                "decompile_ready_proxy": bool(target.get("decompile_ready_proxy")),
                "fallback_bb_count": int(report.get("fallback_bb_count") or 0),
                "fallback_region_count": int(report.get("fallback_region_count") or 0),
                "goto_count": int(report.get("goto_count") or 0),
                "break_count": int(report.get("break_count") or 0),
                "continue_count": int(report.get("continue_count") or 0),
                "early_exit_count": int(report.get("early_exit_count") or 0),
                "cleanup_exit_count": int(report.get("cleanup_exit_count") or 0),
                "escape_count": int(
                    report.get("escape_count")
                    or (
                        int(report.get("goto_count") or 0)
                        + int(report.get("break_count") or 0)
                        + int(report.get("continue_count") or 0)
                        + int(report.get("early_exit_count") or 0)
                        + int(report.get("cleanup_exit_count") or 0)
                    )
                ),
                "slot_count": int(report.get("slot_count") or 0),
                "slot_locality_score": target.get("slot_locality_score", report.get("slot_locality_score")),
                "slot_locality_ready": bool(target.get("slot_locality_ready")),
                "empty_control_count": int(target.get("empty_control_count", report.get("empty_control_count") or 0) or 0),
                "empty_loop_count": int(target.get("empty_loop_count", report.get("empty_loop_count") or 0) or 0),
                "empty_if_count": int(target.get("empty_if_count", report.get("empty_if_count") or 0) or 0),
                "empty_switch_count": int(target.get("empty_switch_count", report.get("empty_switch_count") or 0) or 0),
                "empty_control_with_context_count": int(
                    target.get(
                        "empty_control_with_context_count",
                        report.get("empty_control_with_context_count") or 0,
                    )
                    or 0
                ),
                "control_body_total_bb_count": int(
                    target.get("control_body_total_bb_count", report.get("control_body_total_bb_count") or 0) or 0
                ),
                "control_body_nested_bb_count": int(
                    target.get("control_body_nested_bb_count", report.get("control_body_nested_bb_count") or 0) or 0
                ),
                "body_bb_outside_control_count": int(
                    target.get("body_bb_outside_control_count", report.get("body_bb_outside_control_count") or 0) or 0
                ),
                "structured_control_count": int(report.get("structured_control_count") or 0),
                "structured_bb_ratio": target.get("structured_bb_ratio"),
                "edge_semantic_ratio": target.get("edge_semantic_ratio"),
                "edge_interval_ratio": target.get("edge_interval_ratio"),
                "rank_quality": target.get("rank_quality"),
                "rank_label_tier": target.get("rank_label_tier"),
                "rank_label_status": target.get("rank_label_status"),
                "rank_trainable": bool(target.get("rank_trainable")),
                "rank_label_confidence": target.get("rank_label_confidence"),
                "doc_quality_score": target.get("doc_quality_score", target.get("rank_quality")),
                "ranker_score": c.get("ranker_score"),
                "second_version_ready": bool(target.get("second_version_ready")),
                "c_fillable_ranker_ready": bool(target.get("c_fillable_ranker_ready")),
                "modulec_safe_for_rank": bool(target.get("modulec_safe_for_rank")),
                "modulec_labelable_ratio": target.get("modulec_labelable_ratio"),
                "modulec_teacher_primary_coverage": target.get("modulec_teacher_primary_coverage"),
                "modulec_teacher_slot_match_ratio": target.get("modulec_teacher_slot_match_ratio"),
                "modulec_cond_slot_match_ratio": target.get("modulec_cond_slot_match_ratio"),
                "modulec_cond_count_match": bool(target.get("modulec_cond_count_match")),
                "modulec_c_fillable_score": target.get("modulec_c_fillable_score"),
                "modulec_source_like_score": target.get("modulec_source_like_score"),
                "modulec_ast_source_like_score": target.get("modulec_ast_source_like_score"),
                "source_cfg_metrics_used_for_rank": bool(target.get("source_cfg_metrics_used_for_rank")),
                "source_cfg_score": target.get("source_cfg_score"),
                "source_cfg_struct_recall": target.get("source_cfg_struct_recall"),
                "source_cfg_struct_f1": target.get("source_cfg_struct_f1"),
                "source_cfg_all_recall": target.get("source_cfg_all_recall"),
                "source_cfg_all_f1": target.get("source_cfg_all_f1"),
                "source_cfg_structured_src_stmt_coverage": target.get("source_cfg_structured_src_stmt_coverage"),
                "source_cfg_fallback_src_stmt_coverage": target.get("source_cfg_fallback_src_stmt_coverage"),
                "source_cfg_escape_src_stmt_coverage": target.get("source_cfg_escape_src_stmt_coverage"),
                "modulec_teacher_render_structural_ready": bool(target.get("modulec_teacher_render_structural_ready")),
                "modulec_branch_label_conflict_free": bool(target.get("modulec_branch_label_conflict_free")),
                "modulec_switch_case_slot_ready": bool(target.get("modulec_switch_case_slot_ready")),
                "modulec_explicit_case_slot_ready": bool(target.get("modulec_explicit_case_slot_ready")),
                "modulec_switch_edge_ready": bool(target.get("modulec_switch_edge_ready")),
                "modulec_escape_template_ready": bool(target.get("modulec_escape_template_ready")),
                "modulec_unsafe_nonempty_slot_count": int(target.get("modulec_unsafe_nonempty_slot_count") or 0),
                "modulec_missing_teacher_primary_count": int(target.get("modulec_missing_teacher_primary_count") or 0),
                "modulec_case_slot_gap": int(target.get("modulec_case_slot_gap") or 0),
                "outline_lcs_f1": style.get("outline_lcs_f1"),
                "kind_lcs_f1": style.get("kind_lcs_f1"),
                "real_ssa_available": bool(grounding.get("real_ssa_available")),
                "condition_ssa_available_rate": grounding.get("condition_ssa_available_rate"),
                "real_condition_ssa_available_rate": grounding.get("real_condition_ssa_available_rate"),
                "branch_provenance_coverage": grounding.get("branch_provenance_coverage"),
                "condition_branch_metadata_rate": grounding.get("condition_branch_metadata_rate"),
                "condition_polarity_available_rate": grounding.get("condition_polarity_available_rate"),
                "condition_slot_grounded_rate": grounding.get("condition_slot_grounded_rate"),
                "branch_header_cond_slot_coverage": features.get("branch_header_cond_slot_coverage"),
                "branch_header_cond_slot_gap": features.get("branch_header_cond_slot_gap"),
            }
        )
    return out


def build_verified_skeleton_package(
    *,
    cand: Dict[str, Any],
    report: Dict[str, Any],
    selection_source: str,
    include_skeleton: bool,
) -> Dict[str, Any]:
    slots = cand.get("slots") or []
    grounding = cand.get("grounding") or {}
    edge_audit = cand.get("edge_audit") or {}
    fallback_regions: List[Dict[str, Any]] = []
    for slot in slots:
        if str(slot.get("slot_type") or "") == "UNSTRUCTURED_REGION":
            fallback_regions.append(
                {
                    "slot_id": slot.get("slot_id"),
                    "primary_bbs": slot.get("primary_bbs") or [],
                    "context_bbs": slot.get("context_bbs") or [],
                    "reason": slot.get("reason") or "unstructured_region",
                }
            )
    goto_edges = [
        {
            "slot_id": slot.get("slot_id"),
            "edge": slot.get("edge"),
            "primary_bbs": slot.get("primary_bbs") or [],
            "context_bbs": slot.get("context_bbs") or [],
            "reason": slot.get("reason") or "explicit_goto",
        }
        for slot in slots
        if str(slot.get("slot_type") or "") == "GOTO"
    ]
    break_edges = [
        {
            "slot_id": slot.get("slot_id"),
            "edge": slot.get("edge"),
            "primary_bbs": slot.get("primary_bbs") or [],
            "context_bbs": slot.get("context_bbs") or [],
            "reason": slot.get("reason") or "explicit_break",
        }
        for slot in slots
        if str(slot.get("slot_type") or "") == "BREAK"
    ]
    continue_edges = [
        {
            "slot_id": slot.get("slot_id"),
            "edge": slot.get("edge"),
            "primary_bbs": slot.get("primary_bbs") or [],
            "context_bbs": slot.get("context_bbs") or [],
            "reason": slot.get("reason") or "explicit_continue",
        }
        for slot in slots
        if str(slot.get("slot_type") or "") == "CONTINUE"
    ]
    early_exit_edges = [
        {
            "slot_id": slot.get("slot_id"),
            "edge": slot.get("edge"),
            "primary_bbs": slot.get("primary_bbs") or [],
            "context_bbs": slot.get("context_bbs") or [],
            "reason": slot.get("reason") or "explicit_early_exit",
        }
        for slot in slots
        if str(slot.get("slot_type") or "") == "EARLY_EXIT"
    ]
    cleanup_exit_edges = [
        {
            "slot_id": slot.get("slot_id"),
            "edge": slot.get("edge"),
            "primary_bbs": slot.get("primary_bbs") or [],
            "context_bbs": slot.get("context_bbs") or [],
            "cleanup_target_bb": (slot.get("context_bbs") or [None])[-1],
            "cleanup_source_bbs": (slot.get("context_bbs") or [])[:-1],
            "reason": slot.get("reason") or "explicit_cleanup_exit",
        }
        for slot in slots
        if str(slot.get("slot_type") or "") == "CLEANUP_EXIT"
    ]
    skeleton_text = cand.get("skeleton")
    return {
        "schema": "VerifiedSkeleton.v1",
        "candidate_id": cand.get("candidate_id"),
        "candidate_name": cand.get("name"),
        "selection_source": selection_source,
        "skeleton": skeleton_text if include_skeleton else None,
        "skeleton_preview": cand.get("skeleton_preview")
        or "\n".join(str(skeleton_text or "").splitlines()[:24]),
        "outline": cand.get("outline") or [],
        "slots": slots,
        "slot_grounding": grounding,
        "escape_edges": {
            "goto_edges": goto_edges,
            "break_edges": break_edges,
            "continue_edges": continue_edges,
            "early_exit_edges": early_exit_edges,
            "cleanup_exit_edges": cleanup_exit_edges,
            "uncovered_edges": report.get("uncovered_edges") or [],
            "edge_category_counts": edge_audit.get("edge_category_counts") or {},
            "edge_category_preview": edge_audit.get("edge_category_preview") or {},
        },
        "fallback_regions": fallback_regions,
        "goto_edges": goto_edges,
        "break_edges": break_edges,
        "continue_edges": continue_edges,
        "early_exit_edges": early_exit_edges,
        "cleanup_exit_edges": cleanup_exit_edges,
        "validator_report": report,
        "audit_log": [
            {
                "stage": "candidate_generation",
                "status": "generated_by_deterministic_generator",
                "candidate_name": cand.get("name"),
            },
            {
                "stage": "validator",
                "status": "pass" if report.get("valid") else "fail",
                "fatal_errors": report.get("fatal_errors") or [],
            },
            {
                "stage": "canonicalizer",
                "status": "identity_canonicalizer",
                "note": "current canonicalizer keeps validated candidate unchanged and records explicit fallback/goto slots",
            },
        ],
    }


def build_module_c_contract(
    *,
    cand: Dict[str, Any],
    teacher: Dict[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    c_audit = audit_candidate(
        pred_slots=slots_from_candidate(cand),
        teacher_slots=slots_from_teacher_record(teacher),
        validator_report=report,
        outline_equal=(cand.get("source_style") or {}).get("outline_equal"),
    )
    slot_reports = c_audit.get("slot_reports") or []
    case_slot_ids = {
        int(s.get("slot_id"))
        for s in (cand.get("slots") or [])
        if str(s.get("slot_type") or "") == "CASE"
        and s.get("slot_id") is not None
    }
    trainable = [
        s
        for s in slot_reports
        if bool(s.get("constructable"))
        and str(s.get("kind")) in {"COND", "BLOCK", "BRANCH"}
        and str(s.get("target")) != "EMPTY"
        and int(s.get("sid") or -1) not in case_slot_ids
    ]
    skipped = [s for s in slot_reports if s not in trainable]
    pred_cond_count = int(c_audit.get("pred_cond_count") or 0)
    teacher_cond_count = int(c_audit.get("teacher_cond_count") or 0)
    cond_slot_match_ratio = (
        1.0
        if max(pred_cond_count, teacher_cond_count) <= 0
        else min(pred_cond_count, teacher_cond_count) / max(pred_cond_count, teacher_cond_count)
    )
    target = cand.get("rank_target") or {}
    return {
        "schema": "ModuleCSlotContract.v1",
        "input_contract": {
            "inputs": [
                "verified_skeleton",
                "slot_grounding",
                "binary_context",
                "optional_module_a_graph_embeddings",
            ],
            "loss_policy": "train_only_constructable_nonempty_slots; explicit empty slots are legal skip targets",
        },
        "slot_label_constructability": c_audit,
        "trainable_slot_count": int(len(trainable)),
        "skipped_slot_count": int(len(skipped)),
        "safe_for_modulec": bool(c_audit.get("safe_for_modulec")),
        "pred_slots_labelable": bool(c_audit.get("pred_slots_labelable")),
        "teacher_primary_coverage_ok": bool(c_audit.get("teacher_primary_coverage_ok")),
        "cond_count_match": bool(pred_cond_count == teacher_cond_count),
        "cond_slot_match_ratio": round(float(cond_slot_match_ratio), 6),
        "unsafe_nonempty_slot_count": int(len(c_audit.get("unsafe_nonempty_pred_slots") or [])),
        "missing_teacher_primary_count": int(len(c_audit.get("missing_teacher_primary_bbs") or [])),
        "rank_trainable": bool(target.get("rank_trainable")),
        "rank_label_tier": target.get("rank_label_tier"),
        "rank_label_status": target.get("rank_label_status"),
    }


def build_candidate_constructability_summary(
    *,
    candidates: Sequence[Dict[str, Any]],
    teacher: Dict[str, Any],
    max_preview: int = 16,
) -> Dict[str, Any]:
    teacher_slots = slots_from_teacher_record(teacher)
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        audit = audit_candidate(
            pred_slots=slots_from_candidate(cand),
            teacher_slots=teacher_slots,
            validator_report=cand.get("validator_report"),
            outline_equal=(cand.get("source_style") or {}).get("outline_equal"),
        )
        rows.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "name": cand.get("name"),
                "safe_for_modulec": bool(audit.get("safe_for_modulec")),
                "pred_slots_labelable": bool(audit.get("pred_slots_labelable")),
                "teacher_primary_coverage_ok": bool(audit.get("teacher_primary_coverage_ok")),
                "unsafe_nonempty_slot_count": int(len(audit.get("unsafe_nonempty_pred_slots") or [])),
                "missing_teacher_primary_count": int(len(audit.get("missing_teacher_primary_bbs") or [])),
                "rank_quality": (cand.get("rank_target") or {}).get("rank_quality"),
                "outline_lcs_f1": (cand.get("source_style") or {}).get("outline_lcs_f1"),
                "goto_count": (cand.get("validator_report") or {}).get("goto_count"),
                "fallback_bb_count": (cand.get("validator_report") or {}).get("fallback_bb_count"),
            }
        )
    safe = [r for r in rows if r.get("safe_for_modulec")]
    labelable = [r for r in rows if r.get("pred_slots_labelable")]
    coverage = [r for r in rows if r.get("teacher_primary_coverage_ok")]
    best_pool = safe or labelable or coverage or rows
    best = None
    if best_pool:
        best = max(
            best_pool,
            key=lambda r: (
                int(bool(r.get("safe_for_modulec"))),
                int(bool(r.get("pred_slots_labelable"))),
                int(bool(r.get("teacher_primary_coverage_ok"))),
                -int(r.get("unsafe_nonempty_slot_count") or 0),
                -int(r.get("missing_teacher_primary_count") or 0),
                _safe_float(r.get("outline_lcs_f1")),
                -int(r.get("goto_count") or 0),
            ),
        )
    return {
        "schema": "CandidateConstructabilitySummary.v1",
        "candidate_count": int(len(rows)),
        "any_candidate_safe_for_modulec": bool(safe),
        "any_candidate_slots_labelable": bool(labelable),
        "any_candidate_teacher_primary_coverage_ok": bool(coverage),
        "safe_candidate_count": int(len(safe)),
        "labelable_candidate_count": int(len(labelable)),
        "coverage_ok_candidate_count": int(len(coverage)),
        "constructability_best_candidate": best,
        "candidate_constructability_preview": rows[: max(0, int(max_preview))],
    }


def build_failure_buckets(
    *,
    ranker_metrics: Dict[str, Any],
    oracle_metrics: Dict[str, Any],
    selected_report: Dict[str, Any],
    selected_c_contract: Dict[str, Any],
    strict_valid_candidate_count: int,
) -> List[str]:
    buckets: List[str] = []
    if not ranker_metrics.get("hard_valid"):
        buckets.append("selected_not_hard_valid")
    if not selected_report.get("bb_coverage_ok"):
        buckets.append("bb_coverage_failed")
    if not (
        selected_report.get("edge_preservation_ok")
        or selected_report.get("strict_edge_preservation_ok")
    ):
        buckets.append("edge_preservation_failed")
    if selected_report.get("fatal_errors"):
        buckets.append("validator_fatal_error")
    if strict_valid_candidate_count <= 0:
        buckets.append("candidate_pool_no_strict_valid")
    if oracle_metrics.get("doc_quality_ready") and not ranker_metrics.get("doc_quality_ready"):
        buckets.append("ranker_doc_quality_selection_miss")
    if not oracle_metrics.get("doc_quality_ready"):
        buckets.append("candidate_pool_doc_quality_limited")
    if oracle_metrics.get("modulec_safe_for_rank") and not ranker_metrics.get("modulec_safe_for_rank"):
        buckets.append("ranker_modulec_safe_selection_miss")
    if oracle_metrics.get("rank_trainable") and not ranker_metrics.get("rank_trainable"):
        buckets.append("ranker_rank_trainable_selection_miss")
    if not oracle_metrics.get("rank_trainable"):
        buckets.append("candidate_pool_no_rank_trainable")
    if not selected_c_contract.get("rank_trainable"):
        buckets.append("not_rank_trainable_selected")
    if not selected_c_contract.get("safe_for_modulec"):
        buckets.append("not_modulec_safe_selected")
    if not ranker_metrics.get("modulec_teacher_render_structural_ready"):
        buckets.append("teacher_render_structural_not_ready")
    if int(ranker_metrics.get("modulec_unsafe_nonempty_slot_count") or 0) > 0:
        buckets.append("unsafe_nonempty_slot_labels")
    if int(ranker_metrics.get("modulec_missing_teacher_primary_count") or 0) > 0:
        buckets.append("missing_teacher_primary_for_slots")
    if float(ranker_metrics.get("modulec_cond_slot_match_ratio") or 0.0) < 0.98:
        buckets.append("condition_slot_teacher_alignment_gap")
    if int(ranker_metrics.get("modulec_case_slot_gap") or 0) > 0:
        buckets.append("case_slot_not_explicit")
    if float(ranker_metrics.get("structured_bb_ratio") or 0.0) < 0.60:
        buckets.append("low_structured_block_coverage")
    if float(ranker_metrics.get("fallback_bb_ratio") or 0.0) > 0.30:
        buckets.append("high_fallback_bb_ratio")
    if int(ranker_metrics.get("goto_count") or 0) > 2:
        buckets.append("high_goto_count")
    if int(ranker_metrics.get("escape_count") or ranker_metrics.get("goto_count") or 0) > 4:
        buckets.append("high_escape_count")
    if not buckets:
        buckets.append("ok")
    return buckets


def add_scores_to_candidates(
    *,
    records: Sequence[Dict[str, Any]],
    rankers: Sequence[Any],
) -> None:
    for i, rec in enumerate(records):
        ranker = rankers[i]
        cfg_feat = rec.get("cfg_features") or {}
        for cand in rec.get("candidates") or []:
            try:
                cand["ranker_score"] = round(float(ranker.score_candidate(cand, cfg_feat)), 6)
            except Exception:
                cand["ranker_score"] = None


def choose_selected_candidate(
    *,
    args: argparse.Namespace,
    rec: Dict[str, Any],
    ranker_selected: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]],
    oracle: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    mode = str(getattr(args, "selection_mode", "ranker") or "ranker")
    candidates = rec.get("candidates") or []
    fallback = next(
        (c for c in candidates if str(c.get("name") or "") == "full_fallback"),
        candidates[0] if candidates else None,
    )
    if mode == "oracle":
        return (
            oracle or baseline or ranker_selected or fallback,
            "oracle_teacher_forced_source_like",
        )
    if mode == "deterministic":
        return (
            baseline or ranker_selected or fallback,
            "deterministic_binary_cost",
        )
    return (
        ranker_selected or baseline or fallback,
        "ranker_leave_one_out" if not args.train_on_all else "ranker_train_on_all",
    )


def build_full_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Optional[Any], float]:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    teacher_jsonl = Path(args.teacher_jsonl).expanduser().resolve()
    if callable(prepare_rows_with_diagnostics):
        raw_rows, input_join_diagnostics = prepare_rows_with_diagnostics(
            source_jsonl,
            teacher_jsonl,
            int(args.max_examples),
        )
    else:
        raw_rows = prepare_rows(source_jsonl, teacher_jsonl, int(args.max_examples))
        input_join_diagnostics = {
            "teacher_rows_seen": int(len(raw_rows)),
            "matched_rows": int(len(raw_rows)),
            "missing_rows": 0,
            "match_method_dist": [("legacy_prepare_rows", int(len(raw_rows)))],
        }
    setattr(args, "_input_join_diagnostics", input_join_diagnostics)
    t0 = time.time()

    base_records: List[Dict[str, Any]] = []
    row_triplets: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for example, (line_no, row, teacher) in enumerate(raw_rows):
        rec = build_candidate_records(
            row=row,
            teacher=teacher,
            max_region_nodes=int(args.max_region_nodes),
            max_mixed_if_headers=int(args.max_mixed_if_headers),
            candidate_beam=int(args.candidate_beam),
            include_skeletons=bool(args.include_skeletons),
            include_switch_chain=bool(args.include_switch_chain),
            include_sese_interval=not bool(args.disable_sese_interval),
            max_candidates_final=int(args.max_candidates_final),
            candidate_family_cap=int(args.candidate_family_cap),
            candidate_signature_cap=int(args.candidate_signature_cap),
            mixed_candidate_cap=int(args.mixed_candidate_cap),
        )
        rec.update(
            {
                "example": int(example),
                "input_line": int(line_no),
                "func_name": row_func_name(row, teacher),
                "n_bb": int(rec["cfg_features"].get("n_bb") or 0),
                "n_edges": int(rec["cfg_features"].get("n_edges") or 0),
            }
        )
        row_triplets.append((line_no, row, teacher))
        rec.pop("cfg", None)
        base_records.append(rec)

    if bool(args.train_on_all):
        full_ranker = train_pairwise_ranker(
            base_records,
            epochs=int(args.ranker_epochs),
            lr=float(args.ranker_lr),
            l2=float(args.ranker_l2),
            seed=int(args.seed),
        )
        rankers = [full_ranker for _ in base_records]
    else:
        rankers = []
        for i in range(len(base_records)):
            train_rows = base_records[:i] + base_records[i + 1 :]
            rankers.append(
                train_pairwise_ranker(
                    train_rows,
                    epochs=int(args.ranker_epochs),
                    lr=float(args.ranker_lr),
                    l2=float(args.ranker_l2),
                    seed=int(args.seed) + i,
                )
            )
        full_ranker = train_pairwise_ranker(
            base_records,
            epochs=int(args.ranker_epochs),
            lr=float(args.ranker_lr),
            l2=float(args.ranker_l2),
            seed=int(args.seed),
        )

    add_scores_to_candidates(records=base_records, rankers=rankers)

    full_records: List[Dict[str, Any]] = []
    for i, rec in enumerate(base_records):
        line_no, row, teacher = row_triplets[i]
        cfg = analyse_cfg(row)
        baseline = find_candidate(rec, rec.get("deterministic_candidate_id"))
        oracle = find_candidate(rec, rec.get("oracle_candidate_id"))
        ranker_selected = select_with_ranker(
            rec.get("candidates") or [],
            rec.get("cfg_features") or {},
            rankers[i],
        )
        selected, selection_source = choose_selected_candidate(
            args=args,
            rec=rec,
            ranker_selected=ranker_selected,
            baseline=baseline,
            oracle=oracle,
        )
        if selected is None:
            continue
        candidate_records = rec.get("candidates") or []
        baseline_metrics = selected_metrics(rec, baseline)
        oracle_metrics = selected_metrics(rec, oracle)
        ranker_metrics = selected_metrics(rec, selected)
        selected_report = selected.get("validator_report") or {}
        selected_grounding = selected.get("grounding") or {}
        c_contract = build_module_c_contract(
            cand=selected,
            teacher=teacher,
            report=selected_report,
        )
        candidate_c_summary = build_candidate_constructability_summary(
            candidates=candidate_records,
            teacher=teacher,
            max_preview=int(args.candidate_constructability_preview),
        )
        function_id = (
            str(row.get("function_id") or row.get("func_id") or "")
            or f"line_{int(line_no)}:{row_func_name(row, teacher) or 'func'}"
        )
        verified_skeleton = build_verified_skeleton_package(
            cand=selected,
            report=selected_report,
            selection_source=selection_source,
            include_skeleton=bool(args.include_skeletons),
        )
        full: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "example": int(rec.get("example") or i),
            "input_line": int(line_no),
            "func_name": row_func_name(row, teacher),
            "function_ir": build_function_ir(row, function_id=function_id),
            "analysis_context": build_analysis_context(cfg),
            "candidate_generation": {
                "schema": "CandidateGeneration.v1",
                "include_switch_chain": bool(args.include_switch_chain),
                "include_sese_interval": not bool(args.disable_sese_interval),
                "candidate_count": int(rec.get("candidate_count") or len(candidate_records)),
                "candidate_compaction": rec.get("candidate_compaction") or {},
                "strict_valid_candidate_count": int(rec.get("strict_valid_candidate_count") or 0),
                "rank_trainable_candidate_count": int(rec.get("rank_trainable_candidate_count") or 0),
                "c_fillable_candidate_count": int(rec.get("c_fillable_candidate_count") or 0),
                "modulec_safe_candidate_count": int(rec.get("modulec_safe_candidate_count") or 0),
                "second_version_candidate_count": int(rec.get("second_version_candidate_count") or 0),
                "any_rank_trainable_candidate": bool(rec.get("any_rank_trainable_candidate")),
                "any_c_fillable_candidate": bool(rec.get("any_c_fillable_candidate")),
                "any_modulec_safe_candidate": bool(rec.get("any_modulec_safe_candidate")),
                "any_second_version_candidate": bool(rec.get("any_second_version_candidate")),
                "oracle_rank_trainable": bool(rec.get("oracle_rank_trainable")),
                "oracle_c_fillable_ranker_ready": bool(rec.get("oracle_c_fillable_ranker_ready")),
                "oracle_modulec_safe_for_rank": bool(rec.get("oracle_modulec_safe_for_rank")),
                "top16_contains_rank_trainable": bool(rec.get("top16_contains_rank_trainable")),
                "top32_contains_rank_trainable": bool(rec.get("top32_contains_rank_trainable")),
                "top16_contains_c_fillable": bool(rec.get("top16_contains_c_fillable")),
                "top32_contains_c_fillable": bool(rec.get("top32_contains_c_fillable")),
                "fallback_candidate_present": any(
                    str(c.get("name") or "") == "full_fallback"
                    for c in candidate_records
                ),
                "invalid_candidate_entering_training": int(
                    sum(
                        1
                        for c in candidate_records
                        if bool((c.get("rank_target") or {}).get("hard_valid")) is False
                        and _safe_float((c.get("rank_target") or {}).get("rank_quality"), -10.0) > -9.0
                    )
                ),
                "deterministic_candidate_id": rec.get("deterministic_candidate_id"),
                "deterministic_name": rec.get("deterministic_name"),
                "oracle_candidate_id": rec.get("oracle_candidate_id"),
                "oracle_name": rec.get("oracle_name"),
                "oracle_selection": rec.get("oracle_selection"),
                "top3_oracle_candidate_ids": rec.get("top3_oracle_candidate_ids") or [],
                "top5_oracle_candidate_ids": rec.get("top5_oracle_candidate_ids") or [],
                "top16_oracle_candidate_ids": [
                    c.get("candidate_id")
                    for c in sorted(candidate_records, key=rank_sort_key, reverse=True)[:16]
                ],
                "top32_oracle_candidate_ids": [
                    c.get("candidate_id")
                    for c in sorted(candidate_records, key=rank_sort_key, reverse=True)[:32]
                ],
                "candidate_summary": candidate_summary(
                    candidate_records,
                    include_all=bool(args.include_all_candidate_summaries),
                ),
                "candidates": candidate_records if bool(args.include_candidates) else None,
            },
            "ranker": {
                "schema": "RankerSelection.v1",
                "enabled": True,
                "loss": "L_rank",
                "mode": "train_on_all" if args.train_on_all else "leave_one_out",
                "selection_mode": str(args.selection_mode),
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_name": selected.get("name"),
                "selected_ranker_score": selected.get("ranker_score"),
                "binary_only_ranker_candidate_id": ranker_selected.get("candidate_id") if ranker_selected else None,
                "binary_only_ranker_name": ranker_selected.get("name") if ranker_selected else None,
                "oracle_candidate_id": oracle.get("candidate_id") if oracle else None,
                "oracle_name": oracle.get("name") if oracle else None,
                "rank_supervision": {
                    "target_policy": (oracle.get("rank_target") or {}).get("target_policy")
                    if oracle
                    else "document_cfg_ssa_cfillable_quality_v3",
                    "oracle_selection": rec.get("oracle_selection"),
                    "target_candidate_id": rec.get("oracle_candidate_id"),
                    "target_name": rec.get("oracle_name"),
                    "target_rank_quality": (oracle.get("rank_target") or {}).get("rank_quality")
                    if oracle
                    else None,
                    "source_style_quality_diagnostic": (oracle.get("source_style") or {}).get("quality")
                    if oracle
                    else None,
                    "hard_valid": bool((oracle.get("rank_target") or {}).get("hard_valid"))
                    if oracle
                    else False,
                    "loss_formula": "L_rank over strict-valid candidates; source-like and Module-C-fillable signals are preference terms",
                    "uses_modulec_constructability": bool(
                        (oracle.get("rank_target") or {}).get("modulec_c_fillable_score") is not None
                    ) if oracle else False,
                },
            },
            "validator": {
                "schema": "Validator.v1",
                "selected_report": selected_report,
                "global_pass": bool(selected_report.get("valid")),
                "fatal_error_zero": not bool(selected_report.get("fatal_errors") or []),
                "hard_correctness_contract": {
                    "bb_coverage_ok": bool(selected_report.get("bb_coverage_ok")),
                    "edge_preservation_ok": bool(selected_report.get("edge_preservation_ok")),
                    "strict_edge_preservation_ok": bool(
                        selected_report.get("strict_edge_preservation_ok")
                    ),
                    "slot_grounding_ok": bool(selected_report.get("slot_grounding_ok")),
                    "parse_success": bool(selected_report.get("parse_success")),
                },
            },
            "canonicalizer": {
                "schema": "Canonicalizer.v1",
                "enabled": True,
                "actions": [
                    "identity_keep_validated_candidate",
                    "explicit_fallback_region_slots",
                    "explicit_escape_slots_for_unstructured_cross_region_edges",
                ],
                "rerun_validator_after_canonicalize": True,
                "validator_report_after_canonicalize": selected_report,
            },
            "verified_skeleton": verified_skeleton,
            "module_c_contract": c_contract,
            "candidate_constructability": candidate_c_summary,
            "document_metric_contract": {
                "hard_correctness": {
                    "bb_coverage_100": bool(selected_report.get("bb_coverage_ok")),
                    "cfg_edge_preservation_100_or_explicit_fallback": bool(
                        selected_report.get("edge_preservation_ok")
                    ),
                    "accepted_cfg_equivalence_proxy": bool(selected_report.get("valid")),
                    "false_verified_proxy": bool(
                        selected_report.get("valid")
                        and (
                            not selected_report.get("bb_coverage_ok")
                            or not selected_report.get("edge_preservation_ok")
                            or bool(selected_report.get("fatal_errors") or [])
                        )
                    ),
                    "validator_fatal_error_zero_after_fallback": not bool(
                        selected_report.get("fatal_errors") or []
                    ),
                    "skeleton_parse_success": bool(selected_report.get("parse_success")),
                },
                "structuredness": {
                    "verified_skeleton": bool(selected_report.get("valid")),
                    "fallback_region_count": int(selected_report.get("fallback_region_count") or 0),
                    "fallback_bb_count": int(selected_report.get("fallback_bb_count") or 0),
                    "goto_count": int(selected_report.get("goto_count") or 0),
                    "break_count": int(selected_report.get("break_count") or 0),
                    "continue_count": int(selected_report.get("continue_count") or 0),
                    "early_exit_count": int(selected_report.get("early_exit_count") or 0),
                    "cleanup_exit_count": int(selected_report.get("cleanup_exit_count") or 0),
                    "escape_count": int(
                        selected_report.get("escape_count")
                        or (
                            int(selected_report.get("goto_count") or 0)
                            + int(selected_report.get("break_count") or 0)
                            + int(selected_report.get("continue_count") or 0)
                            + int(selected_report.get("early_exit_count") or 0)
                            + int(selected_report.get("cleanup_exit_count") or 0)
                        )
                    ),
                    "slot_locality_score": ranker_metrics.get(
                        "slot_locality_score",
                        selected_report.get("slot_locality_score"),
                    ),
                    "slot_locality_ready": bool(ranker_metrics.get("slot_locality_ready")),
                    "empty_control_count": int(
                        ranker_metrics.get("empty_control_count", selected_report.get("empty_control_count") or 0)
                        or 0
                    ),
                    "empty_loop_count": int(
                        ranker_metrics.get("empty_loop_count", selected_report.get("empty_loop_count") or 0)
                        or 0
                    ),
                    "empty_if_count": int(
                        ranker_metrics.get("empty_if_count", selected_report.get("empty_if_count") or 0)
                        or 0
                    ),
                    "empty_switch_count": int(
                        ranker_metrics.get("empty_switch_count", selected_report.get("empty_switch_count") or 0)
                        or 0
                    ),
                    "empty_control_with_context_count": int(
                        ranker_metrics.get(
                            "empty_control_with_context_count",
                            selected_report.get("empty_control_with_context_count") or 0,
                        )
                        or 0
                    ),
                    "body_bb_outside_control_count": int(
                        ranker_metrics.get(
                            "body_bb_outside_control_count",
                            selected_report.get("body_bb_outside_control_count") or 0,
                        )
                        or 0
                    ),
                    "structured_block_coverage": round(
                        float(selected_report.get("structured_bb_count") or 0)
                        / max(1, int(rec.get("n_bb") or 0)),
                        6,
                    ),
                    "average_nesting_depth_proxy": max(
                        [int(x[1]) for x in (selected.get("outline") or [])] or [0]
                    ),
                    "repair_count": 0,
                },
                "model_effect": {
                    "candidate_top1_accuracy_proxy": bool(
                        selected.get("candidate_id") == rec.get("oracle_candidate_id")
                    ),
                    "candidate_top3_oracle_hit": bool(
                        selected.get("candidate_id") in (rec.get("top3_oracle_candidate_ids") or [])
                    ),
                    "candidate_top5_oracle_hit": bool(
                        selected.get("candidate_id") in (rec.get("top5_oracle_candidate_ids") or [])
                    ),
                    "second_version_ready": bool(ranker_metrics.get("second_version_ready")),
                    "c_fillable_ranker_ready": bool(ranker_metrics.get("c_fillable_ranker_ready")),
                    "rank_trainable": bool(ranker_metrics.get("rank_trainable")),
                    "rank_label_tier": ranker_metrics.get("rank_label_tier"),
                    "rank_label_status": ranker_metrics.get("rank_label_status"),
                    "rank_label_confidence": ranker_metrics.get("rank_label_confidence"),
                    "modulec_safe_for_rank": bool(ranker_metrics.get("modulec_safe_for_rank")),
                    "modulec_pred_slots_labelable": bool(ranker_metrics.get("modulec_pred_slots_labelable")),
                    "modulec_teacher_primary_coverage_ok": bool(
                        ranker_metrics.get("modulec_teacher_primary_coverage_ok")
                    ),
                    "modulec_teacher_render_structural_ready": bool(
                        ranker_metrics.get("modulec_teacher_render_structural_ready")
                    ),
                    "modulec_branch_label_conflict_free": bool(
                        ranker_metrics.get("modulec_branch_label_conflict_free")
                    ),
                    "modulec_switch_case_slot_ready": bool(ranker_metrics.get("modulec_switch_case_slot_ready")),
                    "modulec_explicit_case_slot_ready": bool(
                        ranker_metrics.get("modulec_explicit_case_slot_ready")
                    ),
                    "modulec_switch_edge_ready": bool(ranker_metrics.get("modulec_switch_edge_ready")),
                    "modulec_escape_template_ready": bool(ranker_metrics.get("modulec_escape_template_ready")),
                    "modulec_labelable_ratio": ranker_metrics.get("modulec_labelable_ratio"),
                    "modulec_teacher_primary_coverage": ranker_metrics.get("modulec_teacher_primary_coverage"),
                    "modulec_teacher_slot_match_ratio": ranker_metrics.get("modulec_teacher_slot_match_ratio"),
                    "modulec_cond_slot_match_ratio": ranker_metrics.get("modulec_cond_slot_match_ratio"),
                    "modulec_cond_count_match": bool(ranker_metrics.get("modulec_cond_count_match")),
                    "modulec_c_fillable_score": ranker_metrics.get("modulec_c_fillable_score"),
                    "modulec_source_like_score": ranker_metrics.get("modulec_source_like_score"),
                    "modulec_ast_source_like_score": ranker_metrics.get("modulec_ast_source_like_score"),
                    "slot_locality_score": ranker_metrics.get("slot_locality_score"),
                    "slot_locality_ready": bool(ranker_metrics.get("slot_locality_ready")),
                    "empty_control_count": ranker_metrics.get("empty_control_count"),
                    "empty_control_with_context_count": ranker_metrics.get("empty_control_with_context_count"),
                    "body_bb_outside_control_count": ranker_metrics.get("body_bb_outside_control_count"),
                    "source_cfg_metrics_used_for_rank": bool(ranker_metrics.get("source_cfg_metrics_used_for_rank")),
                    "source_cfg_score": ranker_metrics.get("source_cfg_score"),
                    "source_cfg_struct_recall": ranker_metrics.get("source_cfg_struct_recall"),
                    "source_cfg_struct_f1": ranker_metrics.get("source_cfg_struct_f1"),
                    "source_cfg_all_recall": ranker_metrics.get("source_cfg_all_recall"),
                    "source_cfg_all_f1": ranker_metrics.get("source_cfg_all_f1"),
                    "source_cfg_structured_src_stmt_coverage": ranker_metrics.get("source_cfg_structured_src_stmt_coverage"),
                    "source_cfg_fallback_src_stmt_coverage": ranker_metrics.get("source_cfg_fallback_src_stmt_coverage"),
                    "source_cfg_escape_src_stmt_coverage": ranker_metrics.get("source_cfg_escape_src_stmt_coverage"),
                    "modulec_unsafe_nonempty_slot_count": ranker_metrics.get("modulec_unsafe_nonempty_slot_count"),
                    "modulec_missing_teacher_primary_count": ranker_metrics.get("modulec_missing_teacher_primary_count"),
                    "modulec_case_slot_gap": ranker_metrics.get("modulec_case_slot_gap"),
                    "slot_grounding_accuracy": (
                        selected_grounding.get("slot_grounding_pointer_acc")
                    ),
                    "condition_polarity_error": selected_grounding.get("condition_polarity_error_proxy"),
                    "condition_polarity_available_rate": selected_grounding.get("condition_polarity_available_rate"),
                    "real_ssa_available": bool(selected_grounding.get("real_ssa_available")),
                    "condition_ssa_available_rate": selected_grounding.get("condition_ssa_available_rate"),
                    "real_condition_ssa_available_rate": selected_grounding.get("real_condition_ssa_available_rate"),
                    "branch_provenance_coverage": selected_grounding.get("branch_provenance_coverage"),
                    "condition_branch_metadata_rate": selected_grounding.get("condition_branch_metadata_rate"),
                    "condition_slot_grounded_rate": selected_grounding.get("condition_slot_grounded_rate"),
                    "branch_header_cond_slot_coverage": ranker_metrics.get(
                        "branch_header_cond_slot_coverage"
                    ),
                    "branch_header_cond_slot_gap": ranker_metrics.get(
                        "branch_header_cond_slot_gap"
                    ),
                    "if_cond_grounded_rate": selected_grounding.get("if_cond_grounded_rate"),
                    "loop_cond_grounded_rate": selected_grounding.get("loop_cond_grounded_rate"),
                    "switch_cond_grounded_rate": selected_grounding.get("switch_cond_grounded_rate"),
                },
            },
            "teacher_diagnostics": {
                "teacher_outline": (rec.get("teacher") or {}).get("outline") or [],
                "teacher_outline_kindseq": (rec.get("teacher") or {}).get("outline_kindseq") or [],
                "teacher_slot_kindseq": (rec.get("teacher") or {}).get("slot_kindseq") or [],
                "source_ast_metrics_are_diagnostic_not_hard_gate": True,
                "baseline_metrics": baseline_metrics,
                "binary_ranker_metrics": selected_metrics(rec, ranker_selected),
                "ranker_metrics": ranker_metrics,
                "oracle_metrics": oracle_metrics,
            },
            "failure_buckets": build_failure_buckets(
                ranker_metrics=ranker_metrics,
                oracle_metrics=oracle_metrics,
                selected_report=selected_report,
                selected_c_contract=c_contract,
                strict_valid_candidate_count=int(rec.get("strict_valid_candidate_count") or 0),
            ),
        }
        full_records.append(_jsonify(full))

    return full_records, full_ranker, time.time() - t0


def _metrics_view(records: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        diag = r.get("teacher_diagnostics") or {}
        obj = diag.get(key) or {}
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _selected_reports(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        ((r.get("validator") or {}).get("selected_report") or {})
        for r in records
    ]


def build_summary(records: Sequence[Dict[str, Any]], args: argparse.Namespace, elapsed: float) -> str:
    n = len(records)
    reports = _selected_reports(records)
    ranker_rows = _metrics_view(records, "ranker_metrics")
    oracle_rows = _metrics_view(records, "oracle_metrics")
    baseline_rows = _metrics_view(records, "baseline_metrics")
    contracts = [(r.get("module_c_contract") or {}) for r in records]
    candidate_contracts = [(r.get("candidate_constructability") or {}) for r in records]
    cand_gen = [(r.get("candidate_generation") or {}) for r in records]
    analyses = [(r.get("analysis_context") or {}) for r in records]
    buckets = Counter(b for r in records for b in (r.get("failure_buckets") or []))
    doc_contracts = [(r.get("document_metric_contract") or {}) for r in records]
    hard_contracts = [(d.get("hard_correctness") or {}) for d in doc_contracts]
    struct_contracts = [(d.get("structuredness") or {}) for d in doc_contracts]
    model_contracts = [(d.get("model_effect") or {}) for d in doc_contracts]
    function_irs = [(r.get("function_ir") or {}) for r in records]

    def bool_count(rows: Sequence[Dict[str, Any]], key: str) -> int:
        return sum(1 for r in rows if bool(r.get(key)))

    def metric_block(name: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
        return [
            f"-- {name} source/quality diagnostics --",
            f"hard_valid = {pct(bool_count(rows, 'hard_valid'), len(rows))}",
            f"doc_mvp_ready = {pct(bool_count(rows, 'doc_mvp_ready'), len(rows))}",
            f"doc_quality_ready = {pct(bool_count(rows, 'doc_quality_ready'), len(rows))}",
            f"second_version_ready = {pct(bool_count(rows, 'second_version_ready'), len(rows))}",
            f"c_fillable_ranker_ready = {pct(bool_count(rows, 'c_fillable_ranker_ready'), len(rows))}",
            f"rank_trainable = {pct(bool_count(rows, 'rank_trainable'), len(rows))}",
            f"modulec_safe_for_rank = {pct(bool_count(rows, 'modulec_safe_for_rank'), len(rows))}",
            f"modulec_teacher_render_structural_ready = {pct(bool_count(rows, 'modulec_teacher_render_structural_ready'), len(rows))}",
            f"modulec_cond_count_match = {pct(bool_count(rows, 'modulec_cond_count_match'), len(rows))}",
            f"slot_locality_ready = {pct(bool_count(rows, 'slot_locality_ready'), len(rows))}",
            f"cost_feasible = {pct(bool_count(rows, 'cost_feasible'), len(rows))}",
            f"source_soft_recovered = {pct(bool_count(rows, 'source_soft_recovered'), len(rows))}",
            f"source_ast_metrics_used_for_rank = {pct(sum(1 for r in rows if r.get('target_policy') == 'document_cfg_ssa_cfillable_quality_v3'), len(rows))}",
            f"source_cfg_metrics_used_for_rank = {pct(sum(1 for r in rows if r.get('source_cfg_metrics_used_for_rank')), len(rows))}",
            f"doc_quality_score_mean = {mean_val(rows, 'doc_quality_score'):.4f}",
            f"source_cfg_score_mean = {mean_val(rows, 'source_cfg_score'):.4f}",
            f"source_cfg_struct_recall_mean = {mean_val(rows, 'source_cfg_struct_recall'):.4f}",
            f"source_cfg_struct_f1_mean = {mean_val(rows, 'source_cfg_struct_f1'):.4f}",
            f"source_cfg_structured_src_stmt_coverage_mean = {mean_val(rows, 'source_cfg_structured_src_stmt_coverage'):.4f}",
            f"source_cfg_fallback_src_stmt_coverage_mean = {mean_val(rows, 'source_cfg_fallback_src_stmt_coverage'):.4f}",
            f"source_cfg_escape_src_stmt_coverage_mean = {mean_val(rows, 'source_cfg_escape_src_stmt_coverage'):.4f}",
            f"modulec_c_fillable_score_mean = {mean_val(rows, 'modulec_c_fillable_score'):.4f}",
            f"modulec_source_like_score_mean = {mean_val(rows, 'modulec_source_like_score'):.4f}",
            f"modulec_ast_source_like_score_mean = {mean_val(rows, 'modulec_ast_source_like_score'):.4f}",
            f"modulec_cond_slot_match_ratio_mean = {mean_val(rows, 'modulec_cond_slot_match_ratio'):.4f}",
            f"modulec_unsafe_nonempty_slot_count_mean = {mean_val(rows, 'modulec_unsafe_nonempty_slot_count'):.4f}",
            f"modulec_missing_teacher_primary_count_mean = {mean_val(rows, 'modulec_missing_teacher_primary_count'):.4f}",
            f"modulec_case_slot_gap_mean = {mean_val(rows, 'modulec_case_slot_gap'):.4f}",
            f"slot_locality_score_mean = {mean_val(rows, 'slot_locality_score'):.4f}",
            f"empty_control_count_mean = {mean_val(rows, 'empty_control_count'):.4f}",
            f"empty_control_with_context_count_mean = {mean_val(rows, 'empty_control_with_context_count'):.4f}",
            f"body_bb_outside_control_count_mean = {mean_val(rows, 'body_bb_outside_control_count'):.4f}",
            f"rank_label_tier_dist = {Counter(r.get('rank_label_tier') for r in rows).most_common()}",
            f"rank_label_status_dist = {Counter(r.get('rank_label_status') for r in rows).most_common(20)}",
            f"structured_bb_ratio_mean = {mean_val(rows, 'structured_bb_ratio'):.4f}",
            f"edge_semantic_ratio_mean = {mean_val(rows, 'edge_semantic_ratio'):.4f}",
            f"edge_interval_ratio_mean = {mean_val(rows, 'edge_interval_ratio'):.4f}",
            f"outline_equal = {pct(bool_count(rows, 'outline_equal'), len(rows))}",
            f"kindseq_equal = {pct(bool_count(rows, 'kindseq_equal'), len(rows))}",
            f"outline_lcs_f1_mean = {mean_val(rows, 'outline_lcs_f1'):.4f}",
            f"outline_lcs_f1_median = {median_val(rows, 'outline_lcs_f1'):.4f}",
            f"kind_lcs_f1_mean = {mean_val(rows, 'kind_lcs_f1'):.4f}",
            f"fallback_bb_ratio_mean = {mean_val(rows, 'fallback_bb_ratio'):.4f}",
            f"goto_count_mean = {mean_val(rows, 'goto_count'):.4f}",
            f"real_ssa_available = {pct(bool_count(rows, 'real_ssa_available'), len(rows))}",
            f"condition_ssa_available_rate_mean = {mean_val(rows, 'condition_ssa_available_rate'):.4f}",
            f"real_condition_ssa_available_rate_mean = {mean_val(rows, 'real_condition_ssa_available_rate'):.4f}",
            f"branch_provenance_coverage_mean = {mean_val(rows, 'branch_provenance_coverage'):.4f}",
            f"condition_branch_metadata_rate_mean = {mean_val(rows, 'condition_branch_metadata_rate'):.4f}",
            f"condition_slot_grounded_rate_mean = {mean_val(rows, 'condition_slot_grounded_rate'):.4f}",
            f"if_cond_grounded_rate_mean = {mean_val(rows, 'if_cond_grounded_rate'):.4f}",
            f"loop_cond_grounded_rate_mean = {mean_val(rows, 'loop_cond_grounded_rate'):.4f}",
            f"switch_cond_grounded_rate_mean = {mean_val(rows, 'switch_cond_grounded_rate'):.4f}",
            f"selected_name_dist = {Counter(r.get('name') for r in rows).most_common(30)}",
        ]

    def real_ssa_function_ir(fi: Dict[str, Any]) -> bool:
        availability = fi.get("availability_source") if isinstance(fi.get("availability_source"), dict) else {}
        return bool(
            str(availability.get("ssa") or "").startswith("binary_")
            or str(fi.get("ssa_level") or "").startswith("binary_")
        )

    lines: List[str] = []
    lines.append("===== MODULE B-v3 FULL PIPELINE EXPORT =====")
    lines.append(f"schema_version = {SCHEMA_VERSION}")
    lines.append(f"source_jsonl = {Path(args.source_jsonl).expanduser().resolve()}")
    lines.append(f"teacher_jsonl = {Path(args.teacher_jsonl).expanduser().resolve()}")
    lines.append(f"n = {n}")
    join_diag = getattr(args, "_input_join_diagnostics", {}) or {}
    if join_diag:
        lines.append(
            "input_join = "
            f"matched {join_diag.get('matched_rows')}/{join_diag.get('teacher_rows_seen')} "
            f"missing={join_diag.get('missing_rows')} "
            f"methods={join_diag.get('match_method_dist')}"
        )
    lines.append(
        "core_claim = verified CFG-faithful canonical structured skeleton; "
        "source AST similarity is diagnostic, not the hard correctness gate."
    )
    lines.append(f"selection_mode = {str(args.selection_mode)}")
    lines.append(
        "ranker_feature_policy = binary_only_cfg_ssa_candidate_validator_provenance; "
        "source_teacher_metrics_are_L_rank_labels_and_audit_only"
    )
    lines.append(f"include_switch_chain = {bool(args.include_switch_chain)}")
    lines.append(f"include_sese_interval = {not bool(args.disable_sese_interval)}")
    lines.append("")
    sese_contexts = [
        (a.get("sese_regions") or {})
        for a in analyses
    ]
    sese_counts = [int(s.get("region_count") or len(s.get("regions") or [])) for s in sese_contexts]
    irreducible_counts = [
        len(a.get("irreducible_components") or [])
        for a in analyses
    ]
    lines.append("-- B0 deterministic analysis --")
    lines.append(f"sese_regions_available = {pct(sum(1 for x in sese_counts if x > 0), n)}")
    lines.append(f"sese_region_count_mean = {statistics.mean(sese_counts) if sese_counts else 0:.4f}")
    lines.append(f"sese_region_count_dist = {sorted(Counter(sese_counts).items())}")
    lines.append(f"irreducible_component_function_rate = {pct(sum(1 for x in irreducible_counts if x > 0), n)}")
    lines.append(f"irreducible_component_count_dist = {sorted(Counter(irreducible_counts).items())}")
    lines.append("")
    lines.append("-- 14.1 hard correctness, selected ranker output --")
    lines.append(f"BB coverage = {pct(bool_count(hard_contracts, 'bb_coverage_100'), n)}")
    lines.append(
        "CFG edge preservation or explicit fallback = "
        f"{pct(bool_count(hard_contracts, 'cfg_edge_preservation_100_or_explicit_fallback'), n)}"
    )
    lines.append(
        f"accepted CFG equivalence proxy = {pct(bool_count(hard_contracts, 'accepted_cfg_equivalence_proxy'), n)}"
    )
    lines.append(
        f"False verified rate proxy = {pct(bool_count(hard_contracts, 'false_verified_proxy'), n)}"
    )
    lines.append(
        "validator fatal error after fallback = "
        f"{pct(n - bool_count(hard_contracts, 'validator_fatal_error_zero_after_fallback'), n)}"
    )
    lines.append(
        f"skeleton parse success = {pct(bool_count(hard_contracts, 'skeleton_parse_success'), n)}"
    )
    lines.append(f"timeout rate = {pct(0, n)}")
    lines.append("")
    lines.append("-- raw validator fields --")
    lines.append(f"valid = {pct(sum(1 for r in reports if r.get('valid')), n)}")
    lines.append(f"bb_coverage_ok = {pct(sum(1 for r in reports if r.get('bb_coverage_ok')), n)}")
    lines.append(f"edge_preservation_ok = {pct(sum(1 for r in reports if r.get('edge_preservation_ok')), n)}")
    lines.append(
        f"strict_edge_preservation_ok = {pct(sum(1 for r in reports if r.get('strict_edge_preservation_ok')), n)}"
    )
    lines.append(f"slot_grounding_ok = {pct(sum(1 for r in reports if r.get('slot_grounding_ok')), n)}")
    lines.append(f"parse_success = {pct(sum(1 for r in reports if r.get('parse_success')), n)}")
    lines.append(f"validator_fatal_error_zero = {pct(sum(1 for r in reports if not r.get('fatal_errors')), n)}")
    lines.append(
        "first_version_contract_pass = "
        + pct(
            sum(
                1
                for r in reports
                if r.get("valid")
                and r.get("bb_coverage_ok")
                and r.get("edge_preservation_ok")
                and r.get("slot_grounding_ok")
                and r.get("parse_success")
                and not r.get("fatal_errors")
            ),
            n,
        )
    )
    lines.append("")
    lines.append("-- 14.2 structuredness --")
    lines.append(f"Verified Skeleton Rate = {pct(bool_count(struct_contracts, 'verified_skeleton'), n)}")
    fallback_region_counts = [int(s.get("fallback_region_count") or 0) for s in struct_contracts]
    fallback_bb_counts = [int(s.get("fallback_bb_count") or 0) for s in struct_contracts]
    goto_counts = [int(s.get("goto_count") or 0) for s in struct_contracts]
    break_counts = [int(s.get("break_count") or 0) for s in struct_contracts]
    continue_counts = [int(s.get("continue_count") or 0) for s in struct_contracts]
    early_exit_counts = [int(s.get("early_exit_count") or 0) for s in struct_contracts]
    cleanup_exit_counts = [int(s.get("cleanup_exit_count") or 0) for s in struct_contracts]
    escape_counts = [int(s.get("escape_count") or s.get("goto_count") or 0) for s in struct_contracts]
    slot_locality_scores = [_safe_float(s.get("slot_locality_score")) for s in struct_contracts]
    empty_control_counts = [int(s.get("empty_control_count") or 0) for s in struct_contracts]
    empty_context_counts = [int(s.get("empty_control_with_context_count") or 0) for s in struct_contracts]
    outside_body_counts = [int(s.get("body_bb_outside_control_count") or 0) for s in struct_contracts]
    structured_cov = [float(s.get("structured_block_coverage") or 0.0) for s in struct_contracts]
    depths = [int(s.get("average_nesting_depth_proxy") or 0) for s in struct_contracts]
    repairs = [int(s.get("repair_count") or 0) for s in struct_contracts]
    lines.append(f"Fallback Region Rate = {pct(sum(1 for x in fallback_region_counts if x > 0), n)}")
    lines.append(f"fallback_region_count_mean = {statistics.mean(fallback_region_counts) if fallback_region_counts else 0:.4f}")
    lines.append(f"fallback_bb_count_mean = {statistics.mean(fallback_bb_counts) if fallback_bb_counts else 0:.4f}")
    lines.append(f"GOTO count per function mean = {statistics.mean(goto_counts) if goto_counts else 0:.4f}")
    lines.append(f"GOTO count dist = {sorted(Counter(goto_counts).items())}")
    lines.append(f"BREAK count per function mean = {statistics.mean(break_counts) if break_counts else 0:.4f}")
    lines.append(f"CONTINUE count per function mean = {statistics.mean(continue_counts) if continue_counts else 0:.4f}")
    lines.append(f"EARLY_EXIT count per function mean = {statistics.mean(early_exit_counts) if early_exit_counts else 0:.4f}")
    lines.append(f"EARLY_EXIT count dist = {sorted(Counter(early_exit_counts).items())}")
    lines.append(f"CLEANUP_EXIT count per function mean = {statistics.mean(cleanup_exit_counts) if cleanup_exit_counts else 0:.4f}")
    lines.append(f"CLEANUP_EXIT count dist = {sorted(Counter(cleanup_exit_counts).items())}")
    lines.append(f"Escape slot count per function mean = {statistics.mean(escape_counts) if escape_counts else 0:.4f}")
    lines.append(f"Escape slot count dist = {sorted(Counter(escape_counts).items())}")
    lines.append(f"Slot locality ready = {pct(bool_count(struct_contracts, 'slot_locality_ready'), n)}")
    lines.append(f"Slot locality score mean = {statistics.mean(slot_locality_scores) if slot_locality_scores else 0:.4f}")
    lines.append(f"Empty control count mean = {statistics.mean(empty_control_counts) if empty_control_counts else 0:.4f}")
    lines.append(f"Empty control with context dist = {sorted(Counter(empty_context_counts).items())}")
    lines.append(f"Body BB outside control dist = {sorted(Counter(outside_body_counts).items())}")
    lines.append(f"Structured block coverage mean = {statistics.mean(structured_cov) if structured_cov else 0:.4f}")
    lines.append(f"Average nesting depth proxy mean = {statistics.mean(depths) if depths else 0:.4f}")
    lines.append(f"Repair count per function mean = {statistics.mean(repairs) if repairs else 0:.4f}")
    lines.append("")
    lines.append("-- candidate pool --")
    cand_counts = [int(c.get("candidate_count") or 0) for c in cand_gen]
    lines.append(f"candidate_count_dist = {sorted(Counter(c.get('candidate_count') for c in cand_gen).items())}")
    lines.append(f"average_candidate_count = {statistics.mean(cand_counts) if cand_counts else 0:.4f}")
    lines.append(f"average_candidate_count_target_le_32 = {statistics.mean(cand_counts) <= 32 if cand_counts else False}")
    compactions = [c.get("candidate_compaction") or {} for c in cand_gen]
    lines.append(
        f"candidate_compaction_enabled = {pct(sum(1 for c in compactions if c.get('enabled')), n)}"
    )
    lines.append(
        f"candidate_compaction_before_mean = {statistics.mean([int(c.get('before') or 0) for c in compactions]) if compactions else 0:.4f}"
    )
    lines.append(
        f"candidate_compaction_after_mean = {statistics.mean([int(c.get('after') or 0) for c in compactions]) if compactions else 0:.4f}"
    )
    lines.append(
        f"candidate_compaction_dropped_mean = {statistics.mean([int(c.get('dropped') or 0) for c in compactions]) if compactions else 0:.4f}"
    )
    lines.append(
        f"strict_valid_candidate_count_dist = {sorted(Counter(c.get('strict_valid_candidate_count') for c in cand_gen).items())}"
    )
    lines.append(
        f"rank_trainable_candidate_count_dist = {sorted(Counter(c.get('rank_trainable_candidate_count') for c in cand_gen).items())}"
    )
    lines.append(
        f"c_fillable_candidate_count_dist = {sorted(Counter(c.get('c_fillable_candidate_count') for c in cand_gen).items())}"
    )
    lines.append(
        f"modulec_safe_candidate_count_dist = {sorted(Counter(c.get('modulec_safe_candidate_count') for c in cand_gen).items())}"
    )
    lines.append(
        f"any_rank_trainable_candidate = {pct(sum(1 for c in cand_gen if c.get('any_rank_trainable_candidate')), n)}"
    )
    lines.append(
        f"any_c_fillable_candidate = {pct(sum(1 for c in cand_gen if c.get('any_c_fillable_candidate')), n)}"
    )
    lines.append(
        f"oracle_rank_trainable = {pct(sum(1 for c in cand_gen if c.get('oracle_rank_trainable')), n)}"
    )
    lines.append(
        f"top16_contains_rank_trainable = {pct(sum(1 for c in cand_gen if c.get('top16_contains_rank_trainable')), n)}"
    )
    lines.append(
        f"top32_contains_rank_trainable = {pct(sum(1 for c in cand_gen if c.get('top32_contains_rank_trainable')), n)}"
    )
    lines.append(
        f"top16_contains_c_fillable = {pct(sum(1 for c in cand_gen if c.get('top16_contains_c_fillable')), n)}"
    )
    lines.append(
        f"top32_contains_c_fillable = {pct(sum(1 for c in cand_gen if c.get('top32_contains_c_fillable')), n)}"
    )
    lines.append(
        f"fallback_candidate_coverage = {pct(sum(1 for c in cand_gen if c.get('fallback_candidate_present')), n)}"
    )
    lines.append(
        f"invalid_candidate_entering_training = {sum(int(c.get('invalid_candidate_entering_training') or 0) for c in cand_gen)}"
    )
    lines.append(
        f"oracle_selection_dist = {Counter(c.get('oracle_selection') for c in cand_gen).most_common()}"
    )
    family_counter = Counter()
    for c in cand_gen:
        for item in c.get("candidate_summary") or []:
            family_counter[str(item.get("family") or "unknown")] += 1
    lines.append(f"candidate_summary_family_dist = {family_counter.most_common(30)}")
    lines.append("")
    lines.append("-- 14.3 model/ranker effect --")
    lines.append(
        f"candidate top-1 accuracy proxy = {pct(bool_count(model_contracts, 'candidate_top1_accuracy_proxy'), n)}"
    )
    lines.append(
        f"candidate top-3 oracle hit = {pct(bool_count(model_contracts, 'candidate_top3_oracle_hit'), n)}"
    )
    lines.append(
        f"candidate top-5 oracle hit = {pct(bool_count(model_contracts, 'candidate_top5_oracle_hit'), n)}"
    )
    baseline_ready = bool_count(baseline_rows, "doc_quality_ready")
    ranker_ready = bool_count(ranker_rows, "doc_quality_ready")
    lines.append(f"ranker doc-quality-ready improvement over deterministic baseline = {ranker_ready - baseline_ready:+d}")
    lines.append(
        f"second_version_ready = {pct(bool_count(model_contracts, 'second_version_ready'), n)}"
    )
    lines.append(
        f"c_fillable_ranker_ready = {pct(bool_count(model_contracts, 'c_fillable_ranker_ready'), n)}"
    )
    lines.append(
        f"rank_trainable = {pct(bool_count(model_contracts, 'rank_trainable'), n)}"
    )
    lines.append(
        f"modulec_safe_for_rank = {pct(bool_count(model_contracts, 'modulec_safe_for_rank'), n)}"
    )
    lines.append(
        "modulec_teacher_render_structural_ready = "
        f"{pct(bool_count(model_contracts, 'modulec_teacher_render_structural_ready'), n)}"
    )
    lines.append(
        "modulec_branch_label_conflict_free = "
        f"{pct(bool_count(model_contracts, 'modulec_branch_label_conflict_free'), n)}"
    )
    lines.append(
        f"modulec_cond_count_match = {pct(bool_count(model_contracts, 'modulec_cond_count_match'), n)}"
    )
    lines.append(
        f"modulec_switch_case_slot_ready = {pct(bool_count(model_contracts, 'modulec_switch_case_slot_ready'), n)}"
    )
    lines.append(
        "modulec_explicit_case_slot_ready = "
        f"{pct(bool_count(model_contracts, 'modulec_explicit_case_slot_ready'), n)}"
    )
    lines.append(
        f"modulec_switch_edge_ready = {pct(bool_count(model_contracts, 'modulec_switch_edge_ready'), n)}"
    )
    lines.append(
        f"modulec_escape_template_ready = {pct(bool_count(model_contracts, 'modulec_escape_template_ready'), n)}"
    )
    for key in (
        "source_cfg_score",
        "source_cfg_struct_recall",
        "source_cfg_struct_f1",
        "source_cfg_structured_src_stmt_coverage",
        "source_cfg_fallback_src_stmt_coverage",
        "source_cfg_escape_src_stmt_coverage",
        "modulec_labelable_ratio",
        "modulec_teacher_primary_coverage",
        "modulec_teacher_slot_match_ratio",
        "modulec_cond_slot_match_ratio",
        "modulec_c_fillable_score",
        "modulec_source_like_score",
        "modulec_unsafe_nonempty_slot_count",
        "modulec_missing_teacher_primary_count",
        "modulec_case_slot_gap",
    ):
        lines.append(f"{key}_mean = {mean_val(model_contracts, key):.4f}")
    sg_vals = [
        _safe_float(m.get("slot_grounding_accuracy"))
        for m in model_contracts
        if m.get("slot_grounding_accuracy") is not None
    ]
    lines.append(f"slot grounding accuracy pointer proxy = {statistics.mean(sg_vals) if sg_vals else 0:.4f}")
    lines.append(f"real_ssa_available selected = {pct(bool_count(model_contracts, 'real_ssa_available'), n)}")
    for key in (
        "condition_ssa_available_rate",
        "real_condition_ssa_available_rate",
        "branch_provenance_coverage",
        "condition_branch_metadata_rate",
        "condition_polarity_available_rate",
        "condition_slot_grounded_rate",
        "branch_header_cond_slot_coverage",
        "branch_header_cond_slot_gap",
        "if_cond_grounded_rate",
        "loop_cond_grounded_rate",
        "switch_cond_grounded_rate",
    ):
        vals = [
            _safe_float(m.get(key))
            for m in model_contracts
            if m.get(key) is not None
        ]
        lines.append(f"{key}_mean = {statistics.mean(vals) if vals else 0:.4f}")
    lines.append("condition polarity error = unavailable_without_true_false_edge_or_ssa_metadata")
    lines.append("")
    lines.append("-- FunctionIR / data completeness --")
    for key in (
        "lifted_instructions_available",
        "ssa_available",
        "def_use_available",
        "instruction_provenance_available",
        "branch_metadata_available",
        "call_metadata_available",
        "memory_access_summary_available",
    ):
        lines.append(f"{key} = {pct(sum(1 for fi in function_irs if fi.get(key)), n)}")
    lines.append(
        "full_cfg_functionir_parse = "
        f"{pct(sum(1 for fi in function_irs if fi.get('basic_blocks') and fi.get('cfg_edges') is not None), n)}"
    )
    lines.append(f"real_register_ssa_functionir = {pct(sum(1 for fi in function_irs if real_ssa_function_ir(fi)), n)}")
    lines.append(
        f"functionir_ssa_level_dist = {Counter(str(fi.get('ssa_level') or 'unknown') for fi in function_irs).most_common()}"
    )
    lines.append(
        "functionir_ssa_source_dist = "
        f"{Counter(str(((fi.get('availability_source') or {}) if isinstance(fi.get('availability_source'), dict) else {}).get('ssa') or 'unknown') for fi in function_irs).most_common()}"
    )
    lines.append("")
    lines.append("-- source-AST diagnostics only, not hard gates --")
    lines.extend(metric_block("baseline", baseline_rows))
    lines.append("")
    lines.extend(metric_block("ranker", ranker_rows))
    lines.append("")
    lines.extend(metric_block("oracle", oracle_rows))
    lines.append("")
    lines.append("-- ranker versus oracle --")
    lines.append(
        "note = top-k metrics compare the selected output against the source/teacher oracle; "
        "use --selection-mode oracle only for upper-bound or Module-C teacher-forced export."
    )
    lines.append(
        "ranker_top1_matches_oracle = "
        + pct(
            sum(
                1
                for r in records
                if ((r.get("ranker") or {}).get("selected_candidate_id"))
                == ((r.get("candidate_generation") or {}).get("oracle_candidate_id"))
            ),
            n,
        )
    )
    lines.append(
        "ranker_selected_in_oracle_top3 = "
        + pct(
            sum(
                1
                for r in records
                if ((r.get("ranker") or {}).get("selected_candidate_id"))
                in ((r.get("candidate_generation") or {}).get("top3_oracle_candidate_ids") or [])
            ),
            n,
        )
    )
    lines.append(
        "ranker_selected_in_oracle_top5 = "
        + pct(
            sum(
                1
                for r in records
                if ((r.get("ranker") or {}).get("selected_candidate_id"))
                in ((r.get("candidate_generation") or {}).get("top5_oracle_candidate_ids") or [])
            ),
            n,
        )
    )
    g1_rate = sum(1 for c in cand_gen if c.get("top16_contains_rank_trainable")) / max(1, n)
    g4_rate = sum(
        1
        for r in records
        if ((r.get("ranker") or {}).get("selected_candidate_id"))
        == ((r.get("candidate_generation") or {}).get("oracle_candidate_id"))
    ) / max(1, n)
    g5_rate = bool_count(model_contracts, "rank_trainable") / max(1, n)
    if g1_rate < 0.90:
        stop_go = "STOP_G1_FIX_CANDIDATE_GENERATOR_OR_LABEL_ALIGNMENT"
    elif g4_rate < 0.90 or g5_rate < 0.90:
        stop_go = "STOP_G4_FIX_RANKER_SELECTION"
    else:
        stop_go = "GO_SECOND_VERSION_RANKER_READY_FOR_C_D_SMOKE"
    lines.append(f"stop_go_decision = {stop_go}")
    lines.append("")
    lines.append("-- Module C contract, selected ranker output --")
    lines.append(f"safe_for_modulec = {pct(sum(1 for c in contracts if c.get('safe_for_modulec')), n)}")
    lines.append(f"pred_slots_labelable = {pct(sum(1 for c in contracts if c.get('pred_slots_labelable')), n)}")
    lines.append(
        f"teacher_primary_coverage_ok = {pct(sum(1 for c in contracts if c.get('teacher_primary_coverage_ok')), n)}"
    )
    lines.append(
        f"trainable_slot_count_mean = {statistics.mean([int(c.get('trainable_slot_count') or 0) for c in contracts]) if contracts else 0:.4f}"
    )
    lines.append(
        f"unsafe_nonempty_slot_count_mean = {statistics.mean([int(c.get('unsafe_nonempty_slot_count') or 0) for c in contracts]) if contracts else 0:.4f}"
    )
    lines.append("")
    lines.append("-- Module C constructability over full candidate pool --")
    lines.append(
        f"any_candidate_safe_for_modulec = {pct(sum(1 for c in candidate_contracts if c.get('any_candidate_safe_for_modulec')), n)}"
    )
    lines.append(
        f"any_candidate_slots_labelable = {pct(sum(1 for c in candidate_contracts if c.get('any_candidate_slots_labelable')), n)}"
    )
    lines.append(
        "any_candidate_teacher_primary_coverage_ok = "
        f"{pct(sum(1 for c in candidate_contracts if c.get('any_candidate_teacher_primary_coverage_ok')), n)}"
    )
    lines.append(
        f"safe_candidate_count_mean = {statistics.mean([int(c.get('safe_candidate_count') or 0) for c in candidate_contracts]) if candidate_contracts else 0:.4f}"
    )
    lines.append("")
    lines.append(f"-- failure buckets --")
    lines.append(f"failure_bucket_dist = {buckets.most_common(50)}")
    lines.append("")
    lines.append(f"-- badcase preview, first {int(args.max_preview)} --")
    bad = sorted(
        records,
        key=lambda r: (
            0 if ((r.get("teacher_diagnostics") or {}).get("ranker_metrics") or {}).get("doc_quality_ready") else 1,
            -float(((r.get("teacher_diagnostics") or {}).get("ranker_metrics") or {}).get("structured_bb_ratio") or 0.0),
            int(((r.get("teacher_diagnostics") or {}).get("ranker_metrics") or {}).get("goto_count") or 0),
            float(((r.get("teacher_diagnostics") or {}).get("ranker_metrics") or {}).get("fallback_bb_ratio") or 0.0),
        ),
        reverse=True,
    )
    for r in bad[: int(args.max_preview)]:
        rm = (r.get("teacher_diagnostics") or {}).get("ranker_metrics") or {}
        om = (r.get("teacher_diagnostics") or {}).get("oracle_metrics") or {}
        cg = r.get("candidate_generation") or {}
        cc = r.get("module_c_contract") or {}
        lines.append("-" * 120)
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"n_bb={(r.get('function_ir') or {}).get('basic_blocks') and len((r.get('function_ir') or {}).get('basic_blocks'))} "
            f"candidates={cg.get('candidate_count')} strict_valid={cg.get('strict_valid_candidate_count')} "
            f"ranker={rm.get('name')} oracle={om.get('name')} buckets={r.get('failure_buckets')}"
        )
        lines.append(
            f"  doc_quality_ready={rm.get('doc_quality_ready')} oracle_doc_quality_ready={om.get('doc_quality_ready')} "
            f"structured={rm.get('structured_bb_ratio')} oracle_structured={om.get('structured_bb_ratio')} "
            f"goto={rm.get('goto_count')} break={rm.get('break_count')} continue={rm.get('continue_count')} "
            f"escape={rm.get('escape_count')} fallback_ratio={rm.get('fallback_bb_ratio')} "
            f"source_outline_f1_diag={rm.get('outline_lcs_f1')} modulec_safe_diag={cc.get('safe_for_modulec')} "
            f"cond_ssa={rm.get('condition_ssa_available_rate')} "
            f"branch_cov={rm.get('branch_provenance_coverage')}"
        )
        lines.append(
            f"  teacher_outline={((r.get('teacher_diagnostics') or {}).get('teacher_outline') or [])} "
            f"pred_outline={((r.get('verified_skeleton') or {}).get('outline') or [])}"
        )
        unsafe = (((cc.get("slot_label_constructability") or {}).get("unsafe_nonempty_pred_slots")) or [])
        missing = (((cc.get("slot_label_constructability") or {}).get("missing_teacher_primary_bbs")) or [])
        if unsafe:
            lines.append(f"  unsafe_nonempty_pred_slots={unsafe[:5]}")
        if missing:
            lines.append(f"  missing_teacher_primary_bbs={missing[:20]}")
    lines.append("")
    lines.append(f"elapsed_sec = {elapsed:.1f}")
    return "\n".join(lines) + "\n"


def export(args: argparse.Namespace) -> None:
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_ranker = Path(args.out_ranker).expanduser().resolve() if args.out_ranker else None
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if out_ranker:
        out_ranker.parent.mkdir(parents=True, exist_ok=True)

    records, ranker, elapsed = build_full_records(args)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out_ranker is not None and ranker is not None:
        out_ranker.write_text(
            json.dumps(ranker.asdict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = build_summary(records, args, elapsed)
    out_txt.write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)
    print(f"saved_txt = {out_txt}", flush=True)
    print(f"saved_jsonl = {out_jsonl}", flush=True)
    if out_ranker is not None:
        print(f"saved_ranker = {out_ranker}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export full Module-B v3 verified skeleton pipeline package")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--out-ranker", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=100)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=4)
    ap.add_argument("--candidate-beam", type=int, default=200)
    ap.add_argument("--ranker-epochs", type=int, default=80)
    ap.add_argument("--ranker-lr", type=float, default=0.05)
    ap.add_argument("--ranker-l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-on-all", action="store_true")
    ap.add_argument(
        "--selection-mode",
        choices=["ranker", "oracle", "deterministic"],
        default="ranker",
        help=(
            "ranker = binary-only learned selection; oracle = source/teacher-forced upper bound "
            "for audits or Module-C label construction; deterministic = no learned ranker."
        ),
    )
    ap.add_argument("--max-preview", type=int, default=80)
    ap.add_argument("--include-skeletons", action="store_true")
    ap.add_argument("--include-candidates", action="store_true")
    ap.add_argument("--include-all-candidate-summaries", action="store_true")
    ap.add_argument("--include-switch-chain", action="store_true")
    ap.add_argument("--disable-sese-interval", action="store_true")
    ap.add_argument("--candidate-constructability-preview", type=int, default=16)
    ap.add_argument("--max-candidates-final", type=int, default=0)
    ap.add_argument("--candidate-family-cap", type=int, default=0)
    ap.add_argument("--candidate-signature-cap", type=int, default=0)
    ap.add_argument("--mixed-candidate-cap", type=int, default=0)
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()

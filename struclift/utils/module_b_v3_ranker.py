"""Small auditable ranker utilities for Module-B v3 candidate selection.

The ranker is intentionally lightweight.  It never creates or repairs a
skeleton; it only scores candidates that were already generated and validated by
the deterministic Module-B v3 MVP layer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FAMILY_NAMES = [
    "recursive",
    "shallow",
    "sese_interval",
    "then_only",
    "else_only",
    "mixed_if_policy",
    "switch_fanin",
    "switch_chain",
    "linear_goto",
    "fallback",
    "other",
]


SOURCE_TEACHER_DERIVED_FEATURES = {
    # These fields are computed from source/teacher labels.  They are valid
    # L_rank supervision targets and audit metrics, but must not be fed back
    # into the ranker feature vector used at binary-only inference time.
    "modulec_safe_for_rank",
    "modulec_pred_slots_labelable",
    "modulec_teacher_primary_coverage_ok",
    "modulec_teacher_render_structural_ready",
    "modulec_branch_label_conflict_free",
    "modulec_switch_case_slot_ready",
    "modulec_explicit_case_slot_ready",
    "modulec_switch_edge_ready",
    "modulec_escape_template_ready",
    "modulec_labelable_ratio",
    "modulec_teacher_primary_coverage",
    "modulec_teacher_slot_match_ratio",
    "modulec_cond_slot_match_ratio",
    "modulec_cond_count_match",
    "modulec_c_fillable_score",
    "modulec_source_like_score",
    "modulec_ast_source_like_score",
    "source_cfg_metrics_used_for_rank",
    "source_cfg_score",
    "source_cfg_struct_recall",
    "source_cfg_struct_f1",
    "source_cfg_all_recall",
    "source_cfg_all_f1",
    "source_cfg_structured_src_stmt_coverage",
    "source_cfg_fallback_src_stmt_coverage",
    "source_cfg_escape_src_stmt_coverage",
    "modulec_unsafe_nonempty_slot_count",
    "modulec_missing_teacher_primary_count",
    "modulec_case_slot_gap",
    "second_version_ready",
    "c_fillable_ranker_ready",
}


def assert_no_source_teacher_feature_leak(feature_names: Sequence[str]) -> None:
    """Fail loudly if ranker inference features include teacher/source signals."""
    leaks = sorted(set(str(name) for name in feature_names) & SOURCE_TEACHER_DERIVED_FEATURES)
    if leaks:
        raise ValueError(
            "ranker feature leak: source/teacher-derived fields cannot be used "
            f"as binary-only ranker inputs: {leaks}"
        )


def candidate_family(name: str) -> str:
    name = str(name or "")
    if name == "full_fallback":
        return "fallback"
    if name == "linear_goto":
        return "linear_goto"
    if "sese_interval" in name or "interval" in name:
        return "sese_interval"
    if "switch_fanin" in name:
        return "switch_fanin"
    if "switch_chain" in name:
        return "switch_chain"
    if "mixed" in name:
        return "mixed_if_policy"
    if "then_only" in name:
        return "then_only"
    if "else_only" in name:
        return "else_only"
    if "recursive" in name:
        return "recursive"
    if "shallow" in name:
        return "shallow"
    return "other"


def _candidate_name(cand: Dict[str, Any]) -> str:
    return str(cand.get("name") or "")


def _is_switch_chain_candidate(cand: Dict[str, Any]) -> bool:
    name = _candidate_name(cand)
    return "switch_chain" in name or "switch_fanin" in name


def _outline_switch_count(cand: Dict[str, Any]) -> int:
    feat = cand.get("features")
    if isinstance(feat, dict) and "outline_switch_count" in feat:
        return _safe_int(feat.get("outline_switch_count"))
    outline = cand.get("outline") or []
    return sum(
        1
        for item in outline
        if isinstance(item, (list, tuple)) and item and str(item[0]) == "switch"
    )


def _safe_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
    return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _counter(items: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        key = str(item)
        out[key] = out.get(key, 0) + 1
    return out


def cfg_features(cfg: Any) -> Dict[str, Any]:
    out_degrees = [len(cfg.succs.get(i, [])) for i in range(cfg.n_bb)]
    in_degrees = [len(cfg.preds.get(i, [])) for i in range(cfg.n_bb)]
    branch_headers = sum(1 for d in out_degrees if d >= 2)
    switch_like = sum(1 for d in out_degrees if d >= 3)
    sese_regions = list(getattr(cfg, "sese_regions", []) or [])
    sccs = list(getattr(cfg, "sccs", []) or [])
    irreducible_sccs = list(getattr(cfg, "irreducible_sccs", []) or [])
    return {
        "n_bb": int(cfg.n_bb),
        "n_edges": int(len(cfg.edges)),
        "n_back_edges": int(len(cfg.back_edges)),
        "n_loop_headers": int(len(cfg.loop_nodes_by_header)),
        "n_sccs": int(len(sccs)),
        "n_irreducible_sccs": int(len(irreducible_sccs)),
        "n_sese_regions": int(len(sese_regions)),
        "n_unreachable": int(len(cfg.unreachable)),
        "max_out_degree": int(max(out_degrees) if out_degrees else 0),
        "max_in_degree": int(max(in_degrees) if in_degrees else 0),
        "branch_header_count": int(branch_headers),
        "switch_like_header_count": int(switch_like),
        "edge_density": float(len(cfg.edges) / max(1, cfg.n_bb)),
    }


def slot_grounding_pointers(row: Dict[str, Any], slots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build deterministic BB/instruction pointer metadata for slots.

    This is not a learned grounding model yet.  It is the auditable provenance
    layer needed before a learned grounding pointer can be evaluated.
    """
    block_lengths = [int(x) for x in (row.get("bin_block_lengths") or [])]
    edge_index = row.get("bin_edge_index") or [[], []]
    edge_types = list(row.get("bin_edge_type") or [])
    opcodes = row.get("bin_opcodes") or []
    operand_types = row.get("bin_operand_types") or []
    operand_values = row.get("bin_operand_values") or []
    n_bb = len(block_lengths)
    offsets: List[Tuple[int, int]] = []
    cur = 0
    for length in block_lengths:
        start = cur
        cur += max(0, int(length))
        offsets.append((start, cur))

    out_edges_by_src: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_bb)}
    if isinstance(edge_index, list) and len(edge_index) == 2:
        srcs = edge_index[0] if isinstance(edge_index[0], list) else []
        dsts = edge_index[1] if isinstance(edge_index[1], list) else []
        for i, (src, dst) in enumerate(zip(srcs, dsts)):
            try:
                s = int(src)
                d = int(dst)
            except Exception:
                continue
            if 0 <= s < n_bb and 0 <= d < n_bb:
                out_edges_by_src.setdefault(s, []).append(
                    {
                        "src": s,
                        "dst": d,
                        "edge_type": int(edge_types[i]) if i < len(edge_types) else None,
                    }
                )

    pointer_rows: List[Dict[str, Any]] = []
    function_ir = row.get("function_ir") if isinstance(row.get("function_ir"), dict) else {}
    availability_source = (
        function_ir.get("availability_source")
        if isinstance(function_ir.get("availability_source"), dict)
        else {}
    )
    enrichment = (
        row.get("function_ir_enrichment")
        if isinstance(row.get("function_ir_enrichment"), dict)
        else {}
    )
    ssa_values = function_ir.get("ssa_values") or row.get("ssa_values") or []
    real_ssa_available = bool(
        str(availability_source.get("ssa") or "").startswith("binary_")
        or str(function_ir.get("ssa_level") or "").startswith("binary_")
        or str(enrichment.get("mode") or "").startswith("binary_")
        or any(isinstance(v, dict) and v.get("is_proxy") is False for v in ssa_values)
    )
    branch_metadata_raw = function_ir.get("branch_metadata") or row.get("branch_metadata") or []
    branch_by_bb: Dict[int, Dict[str, Any]] = {}
    branch_source_counts: Dict[str, int] = {}
    if isinstance(branch_metadata_raw, list):
        for meta in branch_metadata_raw:
            if not isinstance(meta, dict):
                continue
            bb = _safe_int(meta.get("bb_id"), -1)
            if 0 <= bb < n_bb:
                branch_by_bb[bb] = meta
                src = str(meta.get("source") or "unknown")
                branch_source_counts[src] = branch_source_counts.get(src, 0) + 1
    elif isinstance(branch_metadata_raw, dict):
        for key, meta in branch_metadata_raw.items():
            if not isinstance(meta, dict):
                continue
            bb = _safe_int(meta.get("bb_id", key), -1)
            if 0 <= bb < n_bb:
                branch_by_bb[bb] = meta
                src = str(meta.get("source") or "unknown")
                branch_source_counts[src] = branch_source_counts.get(src, 0) + 1
    cfg_edges_raw = function_ir.get("cfg_edges") or []
    cfg_edges_by_src: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_bb)}
    if isinstance(cfg_edges_raw, list):
        for edge in cfg_edges_raw:
            if not isinstance(edge, dict):
                continue
            src = _safe_int(edge.get("src"), -1)
            dst = _safe_int(edge.get("dst"), -1)
            if 0 <= src < n_bb and 0 <= dst < n_bb:
                cfg_edges_by_src.setdefault(src, []).append(edge)
    instruction_prov = function_ir.get("instruction_provenance") or row.get("instruction_provenance") or {}
    if not isinstance(instruction_prov, dict):
        instruction_prov = {}

    def _first_condition_ssa(branch_meta: Dict[str, Any], ir_edges: Sequence[Dict[str, Any]]) -> Any:
        if branch_meta.get("condition_ssa") not in (None, "", []):
            return branch_meta.get("condition_ssa")
        for edge in ir_edges:
            if isinstance(edge, dict) and edge.get("condition_ssa") not in (None, "", []):
                return edge.get("condition_ssa")
        return None

    def _branch_provenance_available(branch_meta: Dict[str, Any], branch_instruction: Any) -> bool:
        if branch_instruction is not None and str(branch_instruction) in instruction_prov:
            return True
        return bool(
            branch_meta.get("branch_address") is not None
            or branch_meta.get("address") is not None
            or branch_meta.get("source")
        )
    counts = {
        "total": 0,
        "ok": 0,
        "cond_total": 0,
        "cond_ok": 0,
        "cond_polarity_checkable": 0,
        "cond_polarity_available": 0,
        "cond_polarity_error": 0,
        "block_total": 0,
        "block_ok": 0,
        "goto_total": 0,
        "goto_ok": 0,
        "ssa_available": 0,
        "real_ssa_available": 0,
        "cond_condition_ssa": 0,
        "cond_real_condition_ssa": 0,
        "cond_branch_provenance": 0,
        "cond_branch_metadata": 0,
        "cond_slot_grounded": 0,
        "if_cond_total": 0,
        "if_cond_grounded": 0,
        "loop_cond_total": 0,
        "loop_cond_grounded": 0,
        "switch_cond_total": 0,
        "switch_cond_grounded": 0,
    }
    ssa_available = any(
        key in row
        for key in (
            "ssa_values",
            "def_use_graph",
            "branch_metadata",
            "lifted_instructions",
            "instruction_provenance",
        )
    ) or any(
        function_ir.get(key)
        for key in (
            "ssa_values",
            "def_use_graph",
            "branch_metadata",
            "lifted_instructions",
            "instruction_provenance",
        )
    )
    for slot in slots:
        counts["total"] += 1
        sid = _safe_int(slot.get("slot_id"), -1)
        typ = str(slot.get("slot_type") or "")
        role = str(slot.get("role") or "")
        primary = [int(x) for x in slot.get("primary_bbs") or [] if 0 <= int(x) < n_bb]
        context = [int(x) for x in slot.get("context_bbs") or [] if 0 <= int(x) < n_bb]
        bb_union = sorted(set(primary) | set(context))
        instr_ranges = [
            {"bb": bb, "start": offsets[bb][0], "end": offsets[bb][1]}
            for bb in bb_union
            if 0 <= bb < len(offsets)
        ]
        instr_count = sum(max(0, item["end"] - item["start"]) for item in instr_ranges)
        pointer: Dict[str, Any] = {
            "slot_id": sid,
            "slot_type": typ,
            "role": role,
            "primary_bbs": primary,
            "context_bbs": context,
            "instruction_ranges": instr_ranges,
            "instruction_count": int(instr_count),
            "ssa_available": bool(ssa_available),
            "real_ssa_available": bool(real_ssa_available),
        }
        ok = False
        if typ == "COND":
            counts["cond_total"] += 1
            header = primary[0] if primary else (context[0] if context else -1)
            outgoing = list(out_edges_by_src.get(header, [])) if 0 <= header < n_bb else []
            ir_edges = list(cfg_edges_by_src.get(header, [])) if 0 <= header < n_bb else []
            branch_meta = dict(branch_by_bb.get(header) or {})
            branch_instruction = branch_meta.get("branch_instruction")
            if branch_instruction is None and ir_edges:
                branch_instruction = ir_edges[0].get("provenance_instruction")
            condition_ssa = _first_condition_ssa(branch_meta, ir_edges)
            branch_metadata_available = bool(branch_meta)
            branch_provenance_available = _branch_provenance_available(branch_meta, branch_instruction)
            ssa_grounding_ok = bool(condition_ssa and branch_provenance_available)
            polarity_available = bool(branch_meta.get("true_false_polarity_available"))
            polarity_note = (
                "true_false_edge_metadata_available"
                if polarity_available
                else (
                    "true_false_edge_metadata_unavailable"
                    if len(outgoing) >= 2 else "condition_has_less_than_two_outgoing_edges"
                )
            )
            start, end = offsets[header] if 0 <= header < len(offsets) else (0, 0)
            pointer.update(
                {
                    "condition_bb": header,
                    "branch_instruction_index": int(end - 1) if end > start else None,
                    "branch_instruction": branch_instruction,
                    "branch_mnemonic": branch_meta.get("branch_mnemonic"),
                    "branch_address": branch_meta.get("branch_address") or branch_meta.get("address"),
                    "branch_metadata_source": branch_meta.get("source"),
                    "branch_metadata_available": bool(branch_metadata_available),
                    "branch_provenance_available": bool(branch_provenance_available),
                    "condition_ssa": condition_ssa,
                    "condition_ssa_available": bool(condition_ssa),
                    "real_condition_ssa_available": bool(real_ssa_available and condition_ssa),
                    "ssa_grounding_ok": bool(ssa_grounding_ok),
                    "outgoing_edges": outgoing,
                    "function_ir_outgoing_edges": ir_edges,
                    "polarity_checkable": bool(len(outgoing) >= 2),
                    "true_false_polarity_available": bool(polarity_available),
                    "polarity_error_proxy": False if polarity_available else None,
                    "polarity_error_note": polarity_note,
                    "opcode_window_available": bool(opcodes and end > start),
                }
            )
            if len(outgoing) >= 2:
                counts["cond_polarity_checkable"] += 1
            if polarity_available:
                counts["cond_polarity_available"] += 1
            if branch_metadata_available:
                counts["cond_branch_metadata"] += 1
            if branch_provenance_available:
                counts["cond_branch_provenance"] += 1
            if condition_ssa:
                counts["cond_condition_ssa"] += 1
            if real_ssa_available and condition_ssa:
                counts["cond_real_condition_ssa"] += 1
            if ssa_grounding_ok:
                counts["cond_slot_grounded"] += 1
            if "if_cond" in role:
                counts["if_cond_total"] += 1
                if ssa_grounding_ok:
                    counts["if_cond_grounded"] += 1
            if "loop_cond" in role:
                counts["loop_cond_total"] += 1
                if ssa_grounding_ok:
                    counts["loop_cond_grounded"] += 1
            if "switch_cond" in role or "switch" in role:
                counts["switch_cond_total"] += 1
                if ssa_grounding_ok:
                    counts["switch_cond_grounded"] += 1
            ok = bool(0 <= header < n_bb and outgoing and end > start)
            if ok:
                counts["cond_ok"] += 1
        elif typ in {"BLOCK", "UNSTRUCTURED_REGION"}:
            counts["block_total"] += 1
            pointer.update(
                {
                    "opcode_window_available": bool(opcodes and instr_count > 0),
                    "operand_window_available": bool(
                        operand_types and operand_values and instr_count > 0
                    ),
                }
            )
            ok = bool(primary and instr_count > 0)
            if ok:
                counts["block_ok"] += 1
        elif typ == "CLEANUP_EXIT":
            counts["goto_total"] += 1
            ctx = list(context or primary)
            target = ctx[-1] if ctx else -1
            sources = ctx[:-1]
            edge_ok = bool(
                target >= 0
                and sources
                and all(any(e.get("dst") == target for e in out_edges_by_src.get(src, [])) for src in sources)
            )
            pointer.update(
                {
                    "edge": None,
                    "edge_exists_in_cfg": bool(edge_ok),
                    "cleanup_sources": list(sources),
                    "cleanup_target": target if target >= 0 else None,
                    "escape_slot_kind": typ,
                }
            )
            ok = bool(edge_ok)
            if ok:
                counts["goto_ok"] += 1
        elif typ in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT"}:
            counts["goto_total"] += 1
            edge = slot.get("edge") if isinstance(slot.get("edge"), list) else None
            edge_ok = False
            if edge and len(edge) >= 2:
                s, d = int(edge[0]), int(edge[1])
                edge_ok = any(e.get("dst") == d for e in out_edges_by_src.get(s, []))
            pointer.update({"edge": edge, "edge_exists_in_cfg": bool(edge_ok), "escape_slot_kind": typ})
            ok = bool(edge_ok)
            if ok:
                counts["goto_ok"] += 1
        else:
            ok = bool(instr_count > 0 or typ)
        if ssa_available:
            counts["ssa_available"] += 1
        if real_ssa_available:
            counts["real_ssa_available"] += 1
        pointer["grounding_ok"] = bool(ok)
        if ok:
            counts["ok"] += 1
        pointer_rows.append(pointer)

    total = max(1, counts["total"])
    cond_total = max(1, counts["cond_total"])
    block_total = max(1, counts["block_total"])
    goto_total = max(1, counts["goto_total"])
    if_cond_total = max(1, counts["if_cond_total"])
    loop_cond_total = max(1, counts["loop_cond_total"])
    switch_cond_total = max(1, counts["switch_cond_total"])
    polarity_avail = max(1, counts["cond_polarity_available"])
    return {
        "slot_grounding_pointer_ok": counts["ok"] == counts["total"],
        "slot_grounding_pointer_acc": round(float(counts["ok"] / total), 6),
        "cond_pointer_acc": round(float(counts["cond_ok"] / cond_total), 6),
        "cond_polarity_checkable_rate": round(
            float(counts["cond_polarity_checkable"] / cond_total),
            6,
        ),
        "condition_polarity_available_rate": round(
            float(counts["cond_polarity_available"] / cond_total),
            6,
        ),
        "condition_polarity_error_proxy": round(
            float(counts["cond_polarity_error"] / polarity_avail),
            6,
        ) if counts["cond_polarity_available"] > 0 else None,
        "condition_polarity_note": (
            "true_false_edge_metadata_available"
            if counts["cond_polarity_available"] > 0
            else "true_false_edge_metadata_unavailable"
        ),
        "block_pointer_acc": round(float(counts["block_ok"] / block_total), 6),
        "goto_pointer_acc": round(float(counts["goto_ok"] / goto_total), 6),
        "ssa_available": bool(ssa_available),
        "real_ssa_available": bool(real_ssa_available),
        "branch_metadata_available": bool(branch_by_bb),
        "branch_metadata_source_dist": dict(branch_source_counts),
        "condition_ssa_available_rate": round(float(counts["cond_condition_ssa"] / cond_total), 6),
        "real_condition_ssa_available_rate": round(float(counts["cond_real_condition_ssa"] / cond_total), 6),
        "branch_provenance_coverage": round(float(counts["cond_branch_provenance"] / cond_total), 6),
        "condition_branch_metadata_rate": round(float(counts["cond_branch_metadata"] / cond_total), 6),
        "condition_slot_grounded_rate": round(
            float(counts["cond_slot_grounded"] / cond_total),
            6,
        ),
        "if_cond_grounded_rate": round(float(counts["if_cond_grounded"] / if_cond_total), 6),
        "loop_cond_grounded_rate": round(float(counts["loop_cond_grounded"] / loop_cond_total), 6),
        "switch_cond_grounded_rate": round(float(counts["switch_cond_grounded"] / switch_cond_total), 6),
        "counts": counts,
        "pointers": pointer_rows,
    }


def candidate_features(cand_rec: Dict[str, Any], cfg_feat: Dict[str, Any]) -> Dict[str, Any]:
    report = cand_rec.get("validator_report") or {}
    edge = cand_rec.get("edge_audit") or {}
    grounding = cand_rec.get("grounding") or {}
    target = cand_rec.get("rank_target") or {}
    slots = cand_rec.get("slots") or []
    outline = [(str(k), int(d)) for k, d in cand_rec.get("outline") or []]
    slot_types = _counter(str(s.get("slot_type") or "") for s in slots)
    outline_kinds = _counter(k for k, _d in outline)
    n_bb = max(1, _safe_int(cfg_feat.get("n_bb"), 1))
    n_edges = max(1, _safe_int(cfg_feat.get("n_edges"), 1))
    cfg_branch_headers = max(0, _safe_int(cfg_feat.get("branch_header_count")))
    cond_slot_count = _safe_int(slot_types.get("COND"))
    branch_slot_coverage = (
        float(cond_slot_count / cfg_branch_headers) if cfg_branch_headers > 0 else 1.0
    )
    edge_counts = edge.get("edge_category_counts") or {}
    sem_reasons = edge.get("structured_semantic_reasons") or {}
    return {
        "validator_valid": bool(report.get("valid", False)),
        "bb_coverage_ok": bool(report.get("bb_coverage_ok", False)),
        "edge_preservation_ok": bool(report.get("edge_preservation_ok", False)),
        "strict_edge_preservation_ok": bool(
            report.get("strict_edge_preservation_ok", False)
            or edge.get("strict_edge_preservation_ok", False)
        ),
        "slot_grounding_ok": bool(report.get("slot_grounding_ok", False)),
        "parse_success": bool(report.get("parse_success", False)),
        "validator_score": _safe_float(report.get("score")),
        "fallback_region_count": _safe_int(report.get("fallback_region_count")),
        "fallback_bb_count": _safe_int(report.get("fallback_bb_count")),
        "fallback_bb_ratio": _safe_float(report.get("fallback_bb_count")) / n_bb,
        "goto_count": _safe_int(report.get("goto_count")),
        "break_count": _safe_int(report.get("break_count")),
        "continue_count": _safe_int(report.get("continue_count")),
        "early_exit_count": _safe_int(report.get("early_exit_count")),
        "cleanup_exit_count": _safe_int(report.get("cleanup_exit_count")),
        "escape_count": _safe_int(report.get("escape_count"), _safe_int(report.get("goto_count"))),
        "goto_per_edge": _safe_float(report.get("goto_count")) / n_edges,
        "escape_per_edge": _safe_float(report.get("escape_count"), _safe_float(report.get("goto_count"))) / n_edges,
        "structured_control_count": _safe_int(report.get("structured_control_count")),
        "structured_control_ratio": _safe_float(report.get("structured_control_count")) / n_bb,
        "structured_bb_count": _safe_int(report.get("structured_bb_count")),
        "structured_bb_ratio": _safe_float(report.get("structured_bb_count")) / n_bb,
        "slot_count": _safe_int(report.get("slot_count"), len(slots)),
        "slot_count_ratio": _safe_float(report.get("slot_count"), len(slots)) / n_bb,
        "slot_locality_score": _safe_float(report.get("slot_locality_score"), 1.0),
        "empty_control_count": _safe_int(report.get("empty_control_count")),
        "empty_loop_count": _safe_int(report.get("empty_loop_count")),
        "empty_if_count": _safe_int(report.get("empty_if_count")),
        "empty_switch_count": _safe_int(report.get("empty_switch_count")),
        "empty_control_with_context_count": _safe_int(report.get("empty_control_with_context_count")),
        "control_body_total_bb_count": _safe_int(report.get("control_body_total_bb_count")),
        "control_body_nested_bb_count": _safe_int(report.get("control_body_nested_bb_count")),
        "body_bb_outside_control_count": _safe_int(report.get("body_bb_outside_control_count")),
        "body_bb_outside_control_ratio": _safe_float(report.get("body_bb_outside_control_count"))
        / max(1, _safe_int(report.get("control_body_total_bb_count"), 1)),
        "cond_slot_count": cond_slot_count,
        "cfg_branch_header_count": cfg_branch_headers,
        "branch_header_cond_slot_coverage": min(1.0, branch_slot_coverage),
        "branch_header_cond_slot_gap": max(0.0, 1.0 - min(1.0, branch_slot_coverage)),
        "block_slot_count": _safe_int(slot_types.get("BLOCK")),
        "case_slot_count": _safe_int(slot_types.get("CASE")),
        "unstructured_slot_count": _safe_int(slot_types.get("UNSTRUCTURED_REGION")),
        "goto_slot_count": _safe_int(slot_types.get("GOTO")),
        "break_slot_count": _safe_int(slot_types.get("BREAK")),
        "continue_slot_count": _safe_int(slot_types.get("CONTINUE")),
        "early_exit_slot_count": _safe_int(slot_types.get("EARLY_EXIT")),
        "cleanup_exit_slot_count": _safe_int(slot_types.get("CLEANUP_EXIT")),
        "outline_count": len(outline),
        "outline_count_ratio": float(len(outline) / n_bb),
        "outline_max_depth": max((int(d) for _k, d in outline), default=0),
        "outline_if_count": _safe_int(outline_kinds.get("if")),
        "outline_loop_count": _safe_int(outline_kinds.get("loop")),
        "outline_switch_count": _safe_int(outline_kinds.get("switch")),
        "edge_goto_ratio": _safe_float(edge_counts.get("goto_slot")) / n_edges,
        "edge_break_ratio": _safe_float(edge_counts.get("break_slot")) / n_edges,
        "edge_continue_ratio": _safe_float(edge_counts.get("continue_slot")) / n_edges,
        "edge_early_exit_ratio": _safe_float(edge_counts.get("early_exit_slot")) / n_edges,
        "edge_cleanup_exit_ratio": _safe_float(edge_counts.get("cleanup_exit_slot")) / n_edges,
        "edge_fallback_ratio": _safe_float(edge_counts.get("fallback_internal")) / n_edges,
        "edge_linear_ratio": _safe_float(edge_counts.get("linear_slot")) / n_edges,
        "edge_explicit_ratio": _safe_float(edge_counts.get("explicit_structured")) / n_edges,
        "edge_interval_ratio": _safe_float(edge_counts.get("interval_internal")) / n_edges,
        "edge_semantic_ratio": _safe_float(edge_counts.get("structured_semantic")) / n_edges,
        "edge_uncovered_ratio": _safe_float(edge_counts.get("uncovered")) / n_edges,
        "if_semantic_ratio": _safe_float(sem_reasons.get("if_else_edge")) / n_edges
        + _safe_float(sem_reasons.get("if_then_edge")) / n_edges,
        "loop_semantic_ratio": _safe_float(sem_reasons.get("loop_backedge")) / n_edges
        + _safe_float(sem_reasons.get("loop_body_edge")) / n_edges,
        "switch_semantic_ratio": _safe_float(sem_reasons.get("switch_case_edge")) / n_edges,
        "slot_pointer_acc": _safe_float(grounding.get("slot_grounding_pointer_acc")),
        "cond_pointer_acc": _safe_float(grounding.get("cond_pointer_acc")),
        "cond_polarity_checkable_rate": _safe_float(grounding.get("cond_polarity_checkable_rate")),
        "condition_polarity_available_rate": _safe_float(grounding.get("condition_polarity_available_rate")),
        "block_pointer_acc": _safe_float(grounding.get("block_pointer_acc")),
        "goto_pointer_acc": _safe_float(grounding.get("goto_pointer_acc")),
        "ssa_available": bool(grounding.get("ssa_available")),
        "real_ssa_available": bool(grounding.get("real_ssa_available")),
        "condition_ssa_available_rate": _safe_float(grounding.get("condition_ssa_available_rate")),
        "real_condition_ssa_available_rate": _safe_float(grounding.get("real_condition_ssa_available_rate")),
        "branch_provenance_coverage": _safe_float(grounding.get("branch_provenance_coverage")),
        "condition_branch_metadata_rate": _safe_float(grounding.get("condition_branch_metadata_rate")),
        "condition_slot_grounded_rate": _safe_float(grounding.get("condition_slot_grounded_rate")),
        "if_cond_grounded_rate": _safe_float(grounding.get("if_cond_grounded_rate")),
        "loop_cond_grounded_rate": _safe_float(grounding.get("loop_cond_grounded_rate")),
        "switch_cond_grounded_rate": _safe_float(grounding.get("switch_cond_grounded_rate")),
        "modulec_safe_for_rank": bool(target.get("modulec_safe_for_rank")),
        "modulec_pred_slots_labelable": bool(target.get("modulec_pred_slots_labelable")),
        "modulec_teacher_primary_coverage_ok": bool(target.get("modulec_teacher_primary_coverage_ok")),
        "modulec_teacher_render_structural_ready": bool(target.get("modulec_teacher_render_structural_ready")),
        "modulec_branch_label_conflict_free": bool(target.get("modulec_branch_label_conflict_free")),
        "modulec_switch_case_slot_ready": bool(target.get("modulec_switch_case_slot_ready")),
        "modulec_explicit_case_slot_ready": bool(target.get("modulec_explicit_case_slot_ready")),
        "modulec_switch_edge_ready": bool(target.get("modulec_switch_edge_ready")),
        "modulec_escape_template_ready": bool(target.get("modulec_escape_template_ready")),
        "modulec_labelable_ratio": _safe_float(target.get("modulec_labelable_ratio")),
        "modulec_teacher_primary_coverage": _safe_float(target.get("modulec_teacher_primary_coverage")),
        "modulec_teacher_slot_match_ratio": _safe_float(target.get("modulec_teacher_slot_match_ratio")),
        "modulec_cond_slot_match_ratio": _safe_float(target.get("modulec_cond_slot_match_ratio")),
        "modulec_cond_count_match": bool(target.get("modulec_cond_count_match")),
        "modulec_c_fillable_score": _safe_float(target.get("modulec_c_fillable_score")),
        "modulec_source_like_score": _safe_float(target.get("modulec_source_like_score")),
        "modulec_ast_source_like_score": _safe_float(target.get("modulec_ast_source_like_score")),
        "source_cfg_metrics_used_for_rank": bool(target.get("source_cfg_metrics_used_for_rank")),
        "source_cfg_score": _safe_float(target.get("source_cfg_score")),
        "source_cfg_struct_recall": _safe_float(target.get("source_cfg_struct_recall")),
        "source_cfg_struct_f1": _safe_float(target.get("source_cfg_struct_f1")),
        "source_cfg_all_recall": _safe_float(target.get("source_cfg_all_recall")),
        "source_cfg_all_f1": _safe_float(target.get("source_cfg_all_f1")),
        "source_cfg_structured_src_stmt_coverage": _safe_float(target.get("source_cfg_structured_src_stmt_coverage")),
        "source_cfg_fallback_src_stmt_coverage": _safe_float(target.get("source_cfg_fallback_src_stmt_coverage")),
        "source_cfg_escape_src_stmt_coverage": _safe_float(target.get("source_cfg_escape_src_stmt_coverage")),
        "modulec_unsafe_nonempty_slot_count": _safe_int(target.get("modulec_unsafe_nonempty_slot_count")),
        "modulec_missing_teacher_primary_count": _safe_int(target.get("modulec_missing_teacher_primary_count")),
        "modulec_case_slot_gap": _safe_int(target.get("modulec_case_slot_gap")),
        "second_version_ready": bool(target.get("second_version_ready")),
        "c_fillable_ranker_ready": bool(target.get("c_fillable_ranker_ready")),
    }


def feature_vector(cand_rec: Dict[str, Any], cfg_feat: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    feat = cand_rec.get("features") or candidate_features(cand_rec, cfg_feat)
    name = str(cand_rec.get("name") or "")
    family = candidate_family(name)
    values: Dict[str, float] = {
        "bias": 1.0,
        "cfg_log_n_bb": math.log1p(_safe_float(cfg_feat.get("n_bb"))),
        "cfg_log_n_edges": math.log1p(_safe_float(cfg_feat.get("n_edges"))),
        "cfg_edge_density": _safe_float(cfg_feat.get("edge_density")),
        "cfg_back_edges": _safe_float(cfg_feat.get("n_back_edges")),
        "cfg_loop_headers": _safe_float(cfg_feat.get("n_loop_headers")),
        "cfg_sccs": _safe_float(cfg_feat.get("n_sccs")),
        "cfg_irreducible_sccs": _safe_float(cfg_feat.get("n_irreducible_sccs")),
        "cfg_sese_regions": _safe_float(cfg_feat.get("n_sese_regions")),
        "cfg_branch_headers": _safe_float(cfg_feat.get("branch_header_count")),
        "cfg_switch_like_headers": _safe_float(cfg_feat.get("switch_like_header_count")),
    }
    for fam in FAMILY_NAMES:
        values[f"family_{fam}"] = 1.0 if family == fam else 0.0
    ordered = [
        "validator_valid",
        "bb_coverage_ok",
        "edge_preservation_ok",
        "strict_edge_preservation_ok",
        "slot_grounding_ok",
        "parse_success",
        "fallback_region_count",
        "fallback_bb_ratio",
        "goto_count",
        "break_count",
        "continue_count",
        "early_exit_count",
        "cleanup_exit_count",
        "escape_count",
        "goto_per_edge",
        "escape_per_edge",
        "structured_control_count",
        "structured_control_ratio",
        "structured_bb_ratio",
        "slot_count_ratio",
        "cond_slot_count",
        "cfg_branch_header_count",
        "branch_header_cond_slot_coverage",
        "branch_header_cond_slot_gap",
        "block_slot_count",
        "case_slot_count",
        "unstructured_slot_count",
        "goto_slot_count",
        "break_slot_count",
        "continue_slot_count",
        "early_exit_slot_count",
        "cleanup_exit_slot_count",
        "outline_count",
        "outline_count_ratio",
        "outline_max_depth",
        "outline_if_count",
        "outline_loop_count",
        "outline_switch_count",
        "edge_goto_ratio",
        "edge_break_ratio",
        "edge_continue_ratio",
        "edge_early_exit_ratio",
        "edge_cleanup_exit_ratio",
        "edge_fallback_ratio",
        "edge_linear_ratio",
        "edge_explicit_ratio",
        "edge_interval_ratio",
        "edge_semantic_ratio",
        "edge_uncovered_ratio",
        "if_semantic_ratio",
        "loop_semantic_ratio",
        "switch_semantic_ratio",
        "slot_pointer_acc",
        "cond_pointer_acc",
        "condition_polarity_available_rate",
        "block_pointer_acc",
        "goto_pointer_acc",
        "ssa_available",
        "real_ssa_available",
        "condition_ssa_available_rate",
        "real_condition_ssa_available_rate",
        "branch_provenance_coverage",
        "condition_branch_metadata_rate",
        "condition_slot_grounded_rate",
        "if_cond_grounded_rate",
        "loop_cond_grounded_rate",
        "switch_cond_grounded_rate",
    ]
    ordered = [key for key in ordered if key not in SOURCE_TEACHER_DERIVED_FEATURES]
    for key in ordered:
        values[key] = _safe_float(feat.get(key))
    names = list(values.keys())
    assert_no_source_teacher_feature_leak(names)
    return names, [float(values[k]) for k in names]


@dataclass
class LinearRanker:
    feature_names: List[str]
    mean: List[float]
    scale: List[float]
    weights: List[float]

    def score_vector(self, xs: Sequence[float]) -> float:
        total = 0.0
        for i, value in enumerate(xs):
            if i >= len(self.weights):
                break
            scale = self.scale[i] if i < len(self.scale) and self.scale[i] else 1.0
            mean = self.mean[i] if i < len(self.mean) else 0.0
            total += self.weights[i] * ((float(value) - mean) / scale)
        return float(total)

    def score_candidate(self, cand_rec: Dict[str, Any], cfg_feat: Dict[str, Any]) -> float:
        assert_no_source_teacher_feature_leak(self.feature_names)
        names, values = feature_vector(cand_rec, cfg_feat)
        if names != self.feature_names:
            remap = {name: value for name, value in zip(names, values)}
            values = [float(remap.get(name, 0.0)) for name in self.feature_names]
        return self.score_vector(values)

    def asdict(self) -> Dict[str, Any]:
        return {
            "kind": "linear_pairwise_ranker",
            "feature_names": list(self.feature_names),
            "mean": [float(x) for x in self.mean],
            "scale": [float(x) for x in self.scale],
            "weights": [float(x) for x in self.weights],
        }


def _standard_stats(vectors: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    if not vectors:
        return [], []
    dim = len(vectors[0])
    mean = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            mean[i] += float(value)
    mean = [x / max(1, len(vectors)) for x in mean]
    var = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            diff = float(value) - mean[i]
            var[i] += diff * diff
    scale = [math.sqrt(v / max(1, len(vectors))) or 1.0 for v in var]
    return mean, scale


def _strict_valid_for_rank(cand: Dict[str, Any]) -> bool:
    report = cand.get("validator_report") or {}
    edge = cand.get("edge_audit") or {}
    return bool(
        report.get("valid")
        and report.get("bb_coverage_ok")
        and report.get("unique_primary_ok")
        and report.get("slot_grounding_ok")
        and report.get("parse_success")
        and not (report.get("fatal_errors") or [])
        and (
            edge.get("strict_edge_preservation_ok")
            or report.get("strict_edge_preservation_ok")
        )
    )


def _rank_quality_for_training(cand: Dict[str, Any]) -> float:
    target = cand.get("rank_target") or {}
    if "rank_quality" in target:
        return _safe_float(target.get("rank_quality"), -1e9)
    # Backward-compatible fallback for old audit JSONL.  This is deliberately
    # only used after strict-valid filtering, so invalid candidates cannot become
    # positive labels.
    return _safe_float((cand.get("source_style") or {}).get("quality"), -1e9)


def _rank_trainable_for_training(cand: Dict[str, Any]) -> bool:
    target = cand.get("rank_target") or {}
    if "rank_trainable" in target:
        return bool(target.get("rank_trainable"))
    # Backward-compatible behavior for older audit JSONL that predates explicit
    # teacher-confidence labels.
    return bool(_strict_valid_for_rank(cand))


def _fallback_ratio_for_rank(cand: Dict[str, Any], cfg_feat: Optional[Dict[str, Any]] = None) -> float:
    feat = cand.get("features")
    if isinstance(feat, dict) and "fallback_bb_ratio" in feat:
        return _safe_float(feat.get("fallback_bb_ratio"))
    report = cand.get("validator_report") or {}
    n_bb = max(1, _safe_int((cfg_feat or {}).get("n_bb"), 1))
    return _safe_float(report.get("fallback_bb_count")) / n_bb


def _goto_count_for_rank(cand: Dict[str, Any]) -> int:
    feat = cand.get("features")
    if isinstance(feat, dict) and "goto_count" in feat:
        return _safe_int(feat.get("goto_count"))
    return _safe_int((cand.get("validator_report") or {}).get("goto_count"))


def _escape_count_for_rank(cand: Dict[str, Any]) -> int:
    feat = cand.get("features")
    if isinstance(feat, dict) and "escape_count" in feat:
        return _safe_int(feat.get("escape_count"))
    report = cand.get("validator_report") or {}
    return _safe_int(
        report.get("escape_count"),
        _safe_int(report.get("goto_count"))
        + _safe_int(report.get("break_count"))
        + _safe_int(report.get("continue_count"))
        + _safe_int(report.get("early_exit_count"))
        + _safe_int(report.get("cleanup_exit_count")),
    )


def _cond_slot_branch_coverage_for_rank(cand: Dict[str, Any], cfg_feat: Dict[str, Any]) -> float:
    feat = cand.get("features") or candidate_features(cand, cfg_feat)
    if "branch_header_cond_slot_coverage" in feat:
        return _safe_float(feat.get("branch_header_cond_slot_coverage"), 1.0)
    branch_headers = max(0, _safe_int(cfg_feat.get("branch_header_count")))
    if branch_headers <= 0:
        return 1.0
    return min(1.0, _safe_float(feat.get("cond_slot_count")) / float(branch_headers))


def _binary_candidate_score(cand: Dict[str, Any], cfg_feat: Dict[str, Any]) -> float:
    """Deterministic binary-only tie-breaker for safe ranker selection.

    It intentionally uses only candidate, validator, CFG/SSA, and provenance
    features that are available at inference.  Source/teacher-derived target
    fields are excluded so a high offline oracle score cannot leak into the
    selected skeleton.
    """
    feat = cand.get("features") or candidate_features(cand, cfg_feat)
    n_bb = max(1.0, _safe_float(cfg_feat.get("n_bb"), 1.0))
    branch_gap = max(0.0, 1.0 - _cond_slot_branch_coverage_for_rank(cand, cfg_feat))
    score = (
        1000.0 * float(_strict_valid_for_rank(cand))
        + 28.0 * float(feat.get("structured_bb_ratio") or 0.0)
        + 8.0 * float(feat.get("edge_semantic_ratio") or 0.0)
        + 4.0 * float(feat.get("edge_explicit_ratio") or 0.0)
        + 3.0 * float(feat.get("edge_interval_ratio") or 0.0)
        + 10.0 * float(feat.get("slot_pointer_acc") or 0.0)
        + 7.0 * float(feat.get("condition_slot_grounded_rate") or 0.0)
        + 4.0 * float(feat.get("branch_provenance_coverage") or 0.0)
        + 1.5 * float(feat.get("condition_polarity_available_rate") or 0.0)
        + 1.0 * float(feat.get("edge_cleanup_exit_ratio") or 0.0)
        + 3.0 * float(feat.get("switch_semantic_ratio") or 0.0)
        - 24.0 * float(feat.get("fallback_bb_ratio") or 0.0)
        - 2.5 * float(feat.get("fallback_region_count") or 0.0)
        - 1.25 * float(feat.get("goto_count") or 0.0)
        - 0.35 * float(feat.get("break_count") or 0.0)
        - 0.25 * float(feat.get("continue_count") or 0.0)
        - 0.55 * float(feat.get("early_exit_count") or 0.0)
        - 0.35 * float(feat.get("cleanup_exit_count") or 0.0)
        - 0.35 * float(feat.get("slot_count") or 0.0) / n_bb
        - 12.0 * branch_gap
    )
    if _is_switch_chain_candidate(cand):
        if "switch_fanin" in _candidate_name(cand):
            score += (
                1.25
                + 5.0 * float(feat.get("switch_semantic_ratio") or 0.0)
                - 0.75 * float(feat.get("edge_goto_ratio") or 0.0)
                - 1.0 * float(feat.get("fallback_bb_ratio") or 0.0)
            )
        else:
            score -= (
                0.75
                + 1.5 * float(feat.get("edge_goto_ratio") or 0.0)
                + 1.0 * float(feat.get("fallback_bb_ratio") or 0.0)
            )
    return float(score)


def train_pairwise_ranker(
    records: Sequence[Dict[str, Any]],
    *,
    epochs: int = 80,
    lr: float = 0.05,
    l2: float = 1e-4,
    seed: int = 13,
    max_pairs_per_example: int = 64,
) -> LinearRanker:
    vectors_by_key: Dict[Tuple[int, int], List[float]] = {}
    feature_names: Optional[List[str]] = None
    all_vectors: List[List[float]] = []
    for ridx, rec in enumerate(records):
        cfg_feat = rec.get("cfg_features") or {}
        for cand in rec.get("candidates") or []:
            names, vec = feature_vector(cand, cfg_feat)
            if feature_names is None:
                feature_names = names
            vectors_by_key[(ridx, _safe_int(cand.get("candidate_id")))] = vec
            all_vectors.append(vec)
    if feature_names is None:
        return LinearRanker([], [], [], [])
    mean, scale = _standard_stats(all_vectors)

    def norm(vec: Sequence[float]) -> List[float]:
        return [
            (float(value) - mean[i]) / (scale[i] if scale[i] else 1.0)
            for i, value in enumerate(vec)
        ]

    pairs: List[Tuple[List[float], float]] = []
    rng = random.Random(seed)
    for ridx, rec in enumerate(records):
        candidates = [c for c in (rec.get("candidates") or []) if _strict_valid_for_rank(c)]
        if len(candidates) < 2:
            continue
        trainable = [c for c in candidates if _rank_trainable_for_training(c)]
        if not trainable:
            continue
        ranked = sorted(
            candidates,
            key=lambda c: (_rank_quality_for_training(c), -_safe_int(c.get("candidate_id"))),
            reverse=True,
        )
        pos = max(
            trainable,
            key=lambda c: (_rank_quality_for_training(c), -_safe_int(c.get("candidate_id"))),
        )
        pos_quality = _rank_quality_for_training(pos)
        negs = [
            c for c in ranked
            if _safe_int(c.get("candidate_id")) != _safe_int(pos.get("candidate_id"))
            if pos_quality - _rank_quality_for_training(c) > 1e-6
        ]
        if len(negs) > max_pairs_per_example:
            keep_head = max(0, int(max_pairs_per_example) // 2)
            tail = negs[keep_head:]
            sample_n = min(len(tail), max(0, int(max_pairs_per_example) - keep_head))
            negs = negs[:keep_head] + rng.sample(tail, sample_n)
        pos_vec = norm(vectors_by_key[(ridx, _safe_int(pos.get("candidate_id")))])
        for neg in negs:
            neg_vec = norm(vectors_by_key[(ridx, _safe_int(neg.get("candidate_id")))])
            diff = [p - n for p, n in zip(pos_vec, neg_vec)]
            gap = pos_quality - _rank_quality_for_training(neg)
            pairs.append((diff, min(2.0, max(0.1, gap))))
    dim = len(feature_names)
    weights = [0.0] * dim
    if not pairs:
        return LinearRanker(list(feature_names), mean, scale, weights)
    for _epoch in range(max(1, int(epochs))):
        rng.shuffle(pairs)
        for diff, pair_weight in pairs:
            margin = sum(w * x for w, x in zip(weights, diff))
            if margin >= 40:
                coeff = 0.0
            else:
                coeff = 1.0 / (1.0 + math.exp(margin))
            for i, x in enumerate(diff):
                grad = -coeff * x * pair_weight + float(l2) * weights[i]
                weights[i] -= float(lr) * grad
    return LinearRanker(list(feature_names), mean, scale, weights)


def select_with_ranker(
    candidates: Sequence[Dict[str, Any]],
    cfg_feat: Dict[str, Any],
    ranker: LinearRanker,
) -> Optional[Dict[str, Any]]:
    valid = [
        c for c in candidates
        if _strict_valid_for_rank(c)
    ]
    # Keep the learned ranker inside the first-version safety envelope.  These
    # filters only use binary-visible candidate/validator costs.
    tight_cost = [
        c for c in valid
        if _fallback_ratio_for_rank(c, cfg_feat) <= 0.30
        and _goto_count_for_rank(c) <= 2
        and _escape_count_for_rank(c) <= 4
        and _cond_slot_branch_coverage_for_rank(c, cfg_feat) >= 0.90
    ]
    grounded_cost = [
        c for c in valid
        if _fallback_ratio_for_rank(c, cfg_feat) <= 0.30
        and _cond_slot_branch_coverage_for_rank(c, cfg_feat) >= 0.90
    ]
    low_fallback = [
        c for c in valid
        if _fallback_ratio_for_rank(c, cfg_feat) <= 0.30
    ]
    pool = tight_cost or grounded_cost or low_fallback or valid
    if not pool:
        return None
    no_learned_signal = not ranker.weights or all(abs(float(w)) < 1e-12 for w in ranker.weights)
    best: Optional[Tuple[float, int, Dict[str, Any]]] = None
    for cand in pool:
        if no_learned_signal:
            score = _binary_candidate_score(cand, cfg_feat)
        else:
            score = ranker.score_candidate(cand, cfg_feat)
            score += 0.08 * _binary_candidate_score(cand, cfg_feat)
        cand["ranker_score"] = round(float(score), 6)
        key = (float(score), -_safe_int(cand.get("candidate_id")), cand)
        if best is None or (key[0], key[1]) > (best[0], best[1]):
            best = key
    if best is not None:
        return best[2]
    return None

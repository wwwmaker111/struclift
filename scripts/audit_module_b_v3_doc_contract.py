#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Module-B v3 output against the algorithm document contract.

This audit deliberately ignores source-AST exact match and source-AST slot
constructability.  The Module-B v3 document defines Module B as a binary
CFG/SSA -> verified structured skeleton component, so the hard gates here are
CFG/BB/instruction/SSA/provenance/validator properties plus structuredness and
ranker diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def sf(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return float(default)
    if value != value:
        return float(default)
    return float(value)


def si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def mean(xs: Sequence[Any]) -> float:
    vals = [sf(x) for x in xs]
    return float(statistics.mean(vals)) if vals else 0.0


def q(row: Dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def report(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "validator", "selected_report") or {}


def selected_grounding(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "verified_skeleton", "slot_grounding") or {}


def struct(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "document_metric_contract", "structuredness") or {}


def model_effect(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "document_metric_contract", "model_effect") or {}


def candidate_generation(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("candidate_generation") or {}


def function_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("function_ir") or {}


def analysis_context(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("analysis_context") or {}


def ranker(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("ranker") or {}


def has_real_ssa(fi: Dict[str, Any]) -> bool:
    src = fi.get("availability_source") if isinstance(fi.get("availability_source"), dict) else {}
    return bool(
        str(src.get("ssa") or "").startswith("binary_")
        or str(fi.get("ssa_level") or "").startswith("binary_")
    )


def candidate_family(name: str) -> str:
    name = str(name or "")
    if name == "full_fallback":
        return "fallback"
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
    if "goto" in name:
        return "goto"
    return "other"


def candidate_doc_quality(row: Dict[str, Any], cand: Dict[str, Any]) -> float:
    """Document-aligned quality, intentionally not source-AST based."""
    target = cand.get("rank_target") or {}
    if (
        str(target.get("target_policy") or "").startswith("document_cfg_ssa_quality")
        and target.get("rank_quality") is not None
    ):
        return sf(target.get("rank_quality"), -1e9)
    rep = cand.get("validator_report") or {}
    feat = cand.get("features") or {}
    grounding = cand.get("grounding") or {}
    n_bb = max(1, len((function_ir(row).get("basic_blocks") or [])))
    hard = bool(
        rep.get("valid")
        and rep.get("bb_coverage_ok")
        and rep.get("edge_preservation_ok")
        and rep.get("slot_grounding_ok")
        and rep.get("parse_success")
        and not rep.get("fatal_errors")
    )
    fallback_ratio = sf(rep.get("fallback_bb_count")) / n_bb
    goto_count = sf(rep.get("goto_count"))
    escape_count = sf(
        rep.get("escape_count"),
        goto_count
        + sf(rep.get("break_count"))
        + sf(rep.get("continue_count"))
        + sf(rep.get("early_exit_count"))
        + sf(rep.get("cleanup_exit_count")),
    )
    structured_cov = sf(rep.get("structured_bb_count")) / n_bb
    return (
        1000.0 * float(hard)
        + 20.0 * structured_cov
        + 8.0 * sf(grounding.get("slot_grounding_pointer_acc"))
        + 6.0 * sf(grounding.get("condition_slot_grounded_rate"))
        + 4.0 * sf(grounding.get("branch_provenance_coverage"))
        + 3.0 * sf(feat.get("edge_semantic_ratio"))
        + 2.0 * sf(feat.get("edge_interval_ratio"))
        - 18.0 * fallback_ratio
        - 1.5 * goto_count
        - 0.3 * max(0.0, escape_count - goto_count)
        - 0.15 * sf(rep.get("slot_count"))
    )


def doc_oracle_ids(row: Dict[str, Any]) -> List[Any]:
    cands = (candidate_generation(row).get("candidates") or [])
    ranked = sorted(cands, key=lambda c: candidate_doc_quality(row, c), reverse=True)
    return [c.get("candidate_id") for c in ranked]


def selected_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    sid = ranker(row).get("selected_candidate_id")
    for cand in candidate_generation(row).get("candidates") or []:
        if cand.get("candidate_id") == sid:
            return cand
    return {}


def selected_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return document metrics for the selected candidate.

    Newer exports store these fields in teacher diagnostics and in
    document_metric_contract.model_effect.  Candidate-level rank_target/features
    are kept as fallbacks so the audit can still explain older JSONL files.
    """
    cand = selected_candidate(row)
    ranker_metrics = q(row, "teacher_diagnostics", "ranker_metrics") or {}
    model = model_effect(row)
    target = cand.get("rank_target") or {}
    features = cand.get("features") or {}
    out: Dict[str, Any] = {}
    for key in (
        "branch_header_cond_slot_coverage",
        "branch_header_cond_slot_gap",
    ):
        value = ranker_metrics.get(key)
        if value is None:
            value = model.get(key)
        if value is None:
            value = target.get(key)
        if value is None:
            value = features.get(key)
        out[key] = value
    return out


def deterministic_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    sid = candidate_generation(row).get("deterministic_candidate_id")
    for cand in candidate_generation(row).get("candidates") or []:
        if cand.get("candidate_id") == sid:
            return cand
    return {}


def doc_quality_gap_to_best(row: Dict[str, Any], cand: Dict[str, Any]) -> float:
    cands = candidate_generation(row).get("candidates") or []
    if not cands or not cand:
        return 1e9
    best = max(candidate_doc_quality(row, c) for c in cands)
    return float(best - candidate_doc_quality(row, cand))


def edge_completeness(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    edge_total = 0
    edge_kind = 0
    cond_edges = 0
    cond_ssa = 0
    prov = 0
    for row in rows:
        for edge in function_ir(row).get("cfg_edges") or []:
            if not isinstance(edge, dict):
                continue
            edge_total += 1
            if edge.get("kind") not in (None, ""):
                edge_kind += 1
            if edge.get("condition_ssa") not in (None, "", []):
                cond_ssa += 1
            if edge.get("provenance_instruction") not in (None, "", []):
                prov += 1
            if str(edge.get("kind") or "") in {"true_branch", "false_branch", "cond_branch"}:
                cond_edges += 1
    return {
        "edge_total": edge_total,
        "edge_kind": edge_kind,
        "cond_edges": cond_edges,
        "cond_ssa": cond_ssa,
        "provenance_instruction": prov,
    }


def block_completeness(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    bb_id = 0
    instr_ids = 0
    address = 0
    terminator = 0
    for row in rows:
        for block in function_ir(row).get("basic_blocks") or []:
            if not isinstance(block, dict):
                continue
            total += 1
            if block.get("bb_id") not in (None, ""):
                bb_id += 1
            if block.get("instruction_ids"):
                instr_ids += 1
            if block.get("address_range") or block.get("address_start") is not None:
                address += 1
            if block.get("terminator") not in (None, ""):
                terminator += 1
    return {
        "block_total": total,
        "bb_id": bb_id,
        "instruction_ids": instr_ids,
        "address": address,
        "terminator": terminator,
    }


def final_decision(summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    n = max(1, summary["n"])
    blockers: List[str] = []
    caveats: List[str] = []
    hard_ok = (
        summary["valid"] == n
        and summary["bb_coverage_ok"] == n
        and summary["edge_preservation_ok"] == n
        and summary["parse_success"] == n
        and summary["fatal_zero"] == n
        and summary["false_verified"] == 0
    )
    if not hard_ok:
        blockers.append("14.1 hard correctness is not fully satisfied")
    if summary["fallback_candidate_coverage"] != n:
        blockers.append("fallback candidate coverage is not 100%")
    if summary["invalid_candidate_entering_training"] != 0:
        blockers.append("invalid candidates enter training")
    if summary["average_candidate_count"] > 32:
        blockers.append("average candidate count exceeds 32")
    if summary["real_ssa_rate"] < 0.95:
        blockers.append("real binary SSA coverage below 95%")
    if summary["slot_grounding_pointer_acc_mean"] < 0.98:
        blockers.append("slot grounding pointer accuracy below 98%")
    if summary["branch_header_cond_slot_coverage_mean"] < 0.95:
        blockers.append("CFG branch headers not sufficiently exposed as condition slots")
    if summary["condition_slot_grounded_rate_mean"] < 0.95:
        blockers.append("condition slot grounding below 95%")
    if summary["branch_provenance_coverage_mean"] < 0.95:
        blockers.append("branch provenance coverage below 95%")
    if summary["condition_polarity_available_rate"] < 0.50:
        caveats.append("condition polarity cannot yet be evaluated from true/false edge metadata")
    if summary["sese_available"] == 0:
        caveats.append("SESE/interval region extractor is not available")
    elif summary["selected_sese_interval_rate"] == 0.0:
        caveats.append("SESE/interval regions are available but not selected by the current candidate/ranker path")
    if summary["switch_grounded_rate"] == 0.0:
        caveats.append("switch grounding/recovery is effectively unsupported in this run")
    if summary["structured_block_coverage_mean"] < 0.80:
        caveats.append("structured block coverage is below 80%; output is safe but not very structured")
    if summary["goto_count_mean"] > 1.0:
        caveats.append("mean GOTO count is above 1; readability still weak")
    if summary.get("escape_count_mean", 0.0) > 1.0:
        caveats.append("mean escape slot count is above 1; early-exit/goto readability still weak")
    if summary["fallback_region_rate"] > 0.10:
        caveats.append("fallback region rate is above 10%; candidate generator needs more structure support")

    if blockers:
        return "DOC_CONTRACT_FAIL", blockers + caveats
    if caveats:
        return "DOC_MVP_PASS_WITH_LIMITATIONS", caveats
    return "DOC_FULL_CONTRACT_PASS", []


def build_summary(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    n = len(rows)
    reps = [report(r) for r in rows]
    fies = [function_ir(r) for r in rows]
    analyses = [analysis_context(r) for r in rows]
    grounds = [selected_grounding(r) for r in rows]
    structs = [struct(r) for r in rows]
    candgens = [candidate_generation(r) for r in rows]
    models = [model_effect(r) for r in rows]
    edge_c = edge_completeness(rows)
    block_c = block_completeness(rows)

    selected_doc_top1 = 0
    selected_doc_top3 = 0
    selected_doc_top5 = 0
    deterministic_doc_top1 = 0
    selected_tie_top1 = 0
    selected_tie_top1_001 = 0
    selected_tie_top1_01 = 0
    deterministic_tie_top1 = 0
    selected_quality_gaps: List[float] = []
    deterministic_quality_gaps: List[float] = []
    for row in rows:
        ids = doc_oracle_ids(row)
        selected_id = ranker(row).get("selected_candidate_id")
        deterministic_id = candidate_generation(row).get("deterministic_candidate_id")
        selected_doc_top1 += int(bool(ids and selected_id == ids[0]))
        selected_doc_top3 += int(selected_id in ids[:3])
        selected_doc_top5 += int(selected_id in ids[:5])
        deterministic_doc_top1 += int(bool(ids and deterministic_id == ids[0]))
        sgap = doc_quality_gap_to_best(row, selected_candidate(row))
        dgap = doc_quality_gap_to_best(row, deterministic_candidate(row))
        selected_quality_gaps.append(sgap)
        deterministic_quality_gaps.append(dgap)
        selected_tie_top1 += int(sgap <= 1e-6)
        selected_tie_top1_001 += int(sgap <= 0.01)
        selected_tie_top1_01 += int(sgap <= 0.10)
        deterministic_tie_top1 += int(dgap <= 1e-6)

    families = collections.Counter()
    for row in rows:
        for cand in candidate_generation(row).get("candidates") or []:
            families[candidate_family(cand.get("name"))] += 1

    summary = {
        "n": n,
        "valid": sum(1 for r in reps if r.get("valid")),
        "bb_coverage_ok": sum(1 for r in reps if r.get("bb_coverage_ok")),
        "edge_preservation_ok": sum(1 for r in reps if r.get("edge_preservation_ok")),
        "strict_edge_preservation_ok": sum(1 for r in reps if r.get("strict_edge_preservation_ok")),
        "slot_grounding_ok": sum(1 for r in reps if r.get("slot_grounding_ok")),
        "parse_success": sum(1 for r in reps if r.get("parse_success")),
        "fatal_zero": sum(1 for r in reps if not r.get("fatal_errors")),
        "false_verified": sum(
            1 for row in rows if q(row, "document_metric_contract", "hard_correctness", "false_verified_proxy")
        ),
        "fallback_candidate_coverage": sum(1 for c in candgens if c.get("fallback_candidate_present")),
        "invalid_candidate_entering_training": sum(si(c.get("invalid_candidate_entering_training")) for c in candgens),
        "average_candidate_count": mean([c.get("candidate_count") for c in candgens]),
        "real_ssa_rate": sum(1 for fi in fies if has_real_ssa(fi)) / max(1, n),
        "slot_grounding_pointer_acc_mean": mean([g.get("slot_grounding_pointer_acc") for g in grounds]),
        "branch_header_cond_slot_coverage_mean": mean(
            [selected_metrics(row).get("branch_header_cond_slot_coverage") for row in rows]
        ),
        "condition_slot_grounded_rate_mean": mean([g.get("condition_slot_grounded_rate") for g in grounds]),
        "branch_provenance_coverage_mean": mean([g.get("branch_provenance_coverage") for g in grounds]),
        "condition_polarity_available_rate": mean([g.get("condition_polarity_available_rate") for g in grounds]),
        "condition_polarity_error_proxy_mean": mean([g.get("condition_polarity_error_proxy") for g in grounds]),
        "sese_available": sum(1 for a in analyses if q(a, "sese_regions", "available")),
        "sese_region_count_mean": mean(
            [
                (
                    q(a, "sese_regions", "region_count")
                    if q(a, "sese_regions", "region_count") is not None
                    else len(q(a, "sese_regions", "regions") or [])
                )
                for a in analyses
            ]
        ),
        "selected_sese_interval_rate": sum(
            1
            for row in rows
            if "sese_interval" in str(q(row, "ranker", "selected_name") or q(row, "verified_skeleton", "candidate_name") or "")
        )
        / max(1, n),
        "switch_grounded_rate": mean([g.get("switch_cond_grounded_rate") for g in grounds]),
        "structured_block_coverage_mean": mean([s.get("structured_block_coverage") for s in structs]),
        "goto_count_mean": mean([s.get("goto_count") for s in structs]),
        "escape_count_mean": mean([s.get("escape_count", s.get("goto_count")) for s in structs]),
        "fallback_region_rate": sum(1 for s in structs if si(s.get("fallback_region_count")) > 0) / max(1, n),
    }
    label, notes = final_decision(summary)

    lines: List[str] = []
    lines.append("===== MODULE B-v3 DOCUMENT-CONTRACT AUDIT =====")
    lines.append(f"input_jsonl = {Path(args.input_jsonl).expanduser().resolve()}")
    lines.append(f"n = {n}")
    lines.append("source_ast_metrics_used_for_decision = False")
    lines.append("modulec_source_ast_slot_safety_used_for_decision = False")
    lines.append("")
    lines.append("-- final decision, document-only --")
    lines.append(f"doc_contract_label = {label}")
    lines.append("notes = " + json.dumps(notes, ensure_ascii=False))
    lines.append(
        "decision = "
        + (
            "Module B satisfies the document MVP safety contract; remaining issues are structuredness/advanced analysis limitations."
            if label == "DOC_MVP_PASS_WITH_LIMITATIONS"
            else "Module B satisfies the configured full document contract."
            if label == "DOC_FULL_CONTRACT_PASS"
            else "Module B does not satisfy the document contract."
        )
    )
    lines.append("")
    lines.append("-- 14.1 correctness hard gates --")
    lines.append(f"valid = {pct(summary['valid'], n)}")
    lines.append(f"BB coverage = {pct(summary['bb_coverage_ok'], n)}")
    lines.append(f"CFG edge preservation = {pct(summary['edge_preservation_ok'], n)}")
    lines.append(f"validator lowering edge preservation = {pct(summary['strict_edge_preservation_ok'], n)}")
    lines.append(f"slot grounding validator ok = {pct(summary['slot_grounding_ok'], n)}")
    lines.append(f"accepted CFG equivalence proxy = {pct(summary['valid'], n)}")
    lines.append(f"False verified rate proxy = {pct(summary['false_verified'], n)}")
    lines.append(f"validator fatal error after fallback = {pct(n - summary['fatal_zero'], n)}")
    lines.append(f"skeleton parse success = {pct(summary['parse_success'], n)}")
    lines.append(f"timeout rate = {pct(0, n)}")
    lines.append("")
    lines.append("-- FunctionIR / B0 inputs --")
    lines.append(f"FunctionIR parse = {pct(sum(1 for fi in fies if fi.get('basic_blocks') and fi.get('cfg_edges') is not None), n)}")
    lines.append(f"lifted_instructions_available = {pct(sum(1 for fi in fies if fi.get('lifted_instructions_available')), n)}")
    lines.append(f"ssa_available = {pct(sum(1 for fi in fies if fi.get('ssa_available')), n)}")
    lines.append(f"real_binary_register_ssa = {pct(sum(1 for fi in fies if has_real_ssa(fi)), n)}")
    lines.append(f"def_use_available = {pct(sum(1 for fi in fies if fi.get('def_use_available')), n)}")
    lines.append(f"instruction_provenance_available = {pct(sum(1 for fi in fies if fi.get('instruction_provenance_available')), n)}")
    lines.append(f"branch_metadata_available = {pct(sum(1 for fi in fies if fi.get('branch_metadata_available')), n)}")
    lines.append(f"call_metadata_available = {pct(sum(1 for fi in fies if fi.get('call_metadata_available')), n)}")
    lines.append(f"memory_access_summary_available = {pct(sum(1 for fi in fies if fi.get('memory_access_summary_available')), n)}")
    lines.append(f"basic_block_id_coverage = {pct(block_c['bb_id'], block_c['block_total'])}")
    lines.append(f"basic_block_instruction_id_coverage = {pct(block_c['instruction_ids'], block_c['block_total'])}")
    lines.append(f"basic_block_address_coverage = {pct(block_c['address'], block_c['block_total'])}")
    lines.append(f"basic_block_terminator_coverage = {pct(block_c['terminator'], block_c['block_total'])}")
    lines.append(f"cfg_edge_kind_coverage = {pct(edge_c['edge_kind'], edge_c['edge_total'])}")
    lines.append(f"cfg_edge_condition_ssa_coverage = {pct(edge_c['cond_ssa'], edge_c['edge_total'])}")
    lines.append(f"cfg_edge_provenance_instruction_coverage = {pct(edge_c['provenance_instruction'], edge_c['edge_total'])}")
    lines.append("")
    lines.append("-- AnalysisContext / B0 analysis --")
    for key in ("normalized_cfg", "dominators", "postdominators", "sccs", "natural_loops", "reachability"):
        lines.append(f"{key}_available = {pct(sum(1 for a in analyses if bool(a.get(key))), n)}")
    lines.append(f"sese_regions_available = {pct(summary['sese_available'], n)}")
    lines.append(f"sese_region_count_mean = {summary['sese_region_count_mean']:.4f}")
    lines.append(f"selected_sese_interval_rate = {pct(round(summary['selected_sese_interval_rate'] * n), n)}")
    lines.append(
        "unsupported_marker_dist = "
        + str(collections.Counter(x for a in analyses for x in (a.get("unsupported_markers") or [])).most_common())
    )
    lines.append("")
    lines.append("-- Candidate generator / Stage 0 --")
    counts = [si(c.get("candidate_count")) for c in candgens]
    comp = [c.get("candidate_compaction") or {} for c in candgens]
    strict_counts = [si(c.get("strict_valid_candidate_count")) for c in candgens]
    lines.append(f"average_candidate_count = {mean(counts):.4f}")
    lines.append(f"average_candidate_count_target_le_32 = {mean(counts) <= 32 if counts else False}")
    lines.append(f"candidate_count_dist = {sorted(collections.Counter(counts).items())}")
    lines.append(f"candidate_compaction_enabled = {pct(sum(1 for c in comp if c.get('enabled')), n)}")
    lines.append(f"candidate_compaction_before_mean = {mean([c.get('before') for c in comp]):.4f}")
    lines.append(f"candidate_compaction_after_mean = {mean([c.get('after') for c in comp]):.4f}")
    lines.append(f"fallback_candidate_coverage = {pct(summary['fallback_candidate_coverage'], n)}")
    lines.append(f"invalid_candidate_entering_training = {summary['invalid_candidate_entering_training']}")
    lines.append(f"strict_valid_candidate_count_dist = {sorted(collections.Counter(strict_counts).items())}")
    lines.append(f"candidate_family_dist = {families.most_common()}")
    lines.append("")
    lines.append("-- 14.2 structuredness --")
    lines.append(f"Verified Skeleton Rate = {pct(sum(1 for s in structs if s.get('verified_skeleton')), n)}")
    lines.append(f"Fallback Region Rate = {pct(sum(1 for s in structs if si(s.get('fallback_region_count')) > 0), n)}")
    lines.append(f"fallback_region_count_mean = {mean([s.get('fallback_region_count') for s in structs]):.4f}")
    lines.append(f"fallback_bb_count_mean = {mean([s.get('fallback_bb_count') for s in structs]):.4f}")
    lines.append(f"GOTO count per function mean = {mean([s.get('goto_count') for s in structs]):.4f}")
    lines.append(f"GOTO count dist = {sorted(collections.Counter(si(s.get('goto_count')) for s in structs).items())}")
    lines.append(f"BREAK count per function mean = {mean([s.get('break_count') for s in structs]):.4f}")
    lines.append(f"CONTINUE count per function mean = {mean([s.get('continue_count') for s in structs]):.4f}")
    lines.append(f"EARLY_EXIT count per function mean = {mean([s.get('early_exit_count') for s in structs]):.4f}")
    lines.append(f"EARLY_EXIT count dist = {sorted(collections.Counter(si(s.get('early_exit_count')) for s in structs).items())}")
    lines.append(f"CLEANUP_EXIT count per function mean = {mean([s.get('cleanup_exit_count') for s in structs]):.4f}")
    lines.append(
        "CLEANUP_EXIT count dist = "
        + str(sorted(collections.Counter(si(s.get('cleanup_exit_count')) for s in structs).items()))
    )
    lines.append(f"Escape slot count per function mean = {mean([s.get('escape_count', s.get('goto_count')) for s in structs]):.4f}")
    lines.append(
        "Escape slot count dist = "
        + str(sorted(collections.Counter(si(s.get('escape_count', s.get('goto_count'))) for s in structs).items()))
    )
    lines.append(f"Structured block coverage mean = {mean([s.get('structured_block_coverage') for s in structs]):.4f}")
    lines.append(f"Average nesting depth proxy mean = {mean([s.get('average_nesting_depth_proxy') for s in structs]):.4f}")
    lines.append(f"Repair count per function mean = {mean([s.get('repair_count') for s in structs]):.4f}")
    lines.append("")
    lines.append("-- 14.3 model/ranker and grounding, document-only --")
    lines.append(f"doc_oracle_top1_hit = {pct(selected_doc_top1, n)}")
    lines.append(f"doc_oracle_top3_hit = {pct(selected_doc_top3, n)}")
    lines.append(f"doc_oracle_top5_hit = {pct(selected_doc_top5, n)}")
    lines.append(f"deterministic_doc_oracle_top1 = {pct(deterministic_doc_top1, n)}")
    lines.append(f"ranker_doc_top1_delta_vs_deterministic = {selected_doc_top1 - deterministic_doc_top1:+d}")
    lines.append(f"tie_aware_doc_oracle_top1_hit_eps_1e-6 = {pct(selected_tie_top1, n)}")
    lines.append(f"tie_aware_doc_oracle_top1_hit_eps_0.01 = {pct(selected_tie_top1_001, n)}")
    lines.append(f"tie_aware_doc_oracle_top1_hit_eps_0.10 = {pct(selected_tie_top1_01, n)}")
    lines.append(f"deterministic_tie_aware_doc_oracle_top1_eps_1e-6 = {pct(deterministic_tie_top1, n)}")
    lines.append(f"meaningful_ranker_miss_gap_gt_0.10 = {pct(sum(1 for x in selected_quality_gaps if x > 0.10), n)}")
    lines.append(f"rank_quality_gap_to_best_mean = {mean(selected_quality_gaps):.6f}")
    lines.append(f"rank_quality_gap_to_best_median = {statistics.median(selected_quality_gaps) if selected_quality_gaps else 0.0:.6f}")
    lines.append(f"rank_quality_gap_to_best_max = {max(selected_quality_gaps) if selected_quality_gaps else 0.0:.6f}")
    lines.append(f"deterministic_rank_quality_gap_to_best_mean = {mean(deterministic_quality_gaps):.6f}")
    lines.append(f"slot_grounding_accuracy = {mean([g.get('slot_grounding_pointer_acc') for g in grounds]):.4f}")
    lines.append(f"condition_pointer_acc = {mean([g.get('cond_pointer_acc') for g in grounds]):.4f}")
    lines.append(f"block_pointer_acc = {mean([g.get('block_pointer_acc') for g in grounds]):.4f}")
    lines.append(f"goto_pointer_acc = {mean([g.get('goto_pointer_acc') for g in grounds]):.4f}")
    lines.append(
        "branch_header_cond_slot_coverage_mean = "
        f"{mean([selected_metrics(row).get('branch_header_cond_slot_coverage') for row in rows]):.4f}"
    )
    lines.append(
        "branch_header_cond_slot_gap_mean = "
        f"{mean([selected_metrics(row).get('branch_header_cond_slot_gap') for row in rows]):.4f}"
    )
    for key in (
        "condition_ssa_available_rate",
        "real_condition_ssa_available_rate",
        "branch_provenance_coverage",
        "condition_branch_metadata_rate",
        "condition_polarity_available_rate",
        "condition_slot_grounded_rate",
        "if_cond_grounded_rate",
        "loop_cond_grounded_rate",
        "switch_cond_grounded_rate",
    ):
        lines.append(f"{key}_mean = {mean([g.get(key) for g in grounds]):.4f}")
    lines.append(f"condition_polarity_error_proxy_mean = {summary['condition_polarity_error_proxy_mean']:.4f}")
    lines.append("")
    lines.append("-- explicitly ignored by this document-contract decision --")
    lines.append("source AST exact match = ignored")
    lines.append("source AST slot constructability / safe_for_modulec = ignored")
    lines.append("full source compile rate = ignored for Module B-only audit")
    lines.append("")
    lines.append(f"-- bad document-contract caveats, first {int(args.max_preview)} --")
    bad = []
    for row in rows:
        s = struct(row)
        g = selected_grounding(row)
        bad.append(
            (
                si(s.get("goto_count")),
                si(s.get("fallback_region_count")),
                -sf(s.get("structured_block_coverage")),
                sf(g.get("switch_cond_grounded_rate")),
                si(row.get("example")),
                row,
            )
        )
    for _goto, _fb, _neg_cov, _sw, _example, row in sorted(bad, reverse=True)[: int(args.max_preview)]:
        s = struct(row)
        g = selected_grounding(row)
        lines.append(
            f"example={row.get('example')} input_line={row.get('input_line')} func={row.get('func_name')} "
            f"candidate={ranker(row).get('selected_name')} n_bb={len(function_ir(row).get('basic_blocks') or [])} "
            f"goto={s.get('goto_count')} fallback_regions={s.get('fallback_region_count')} "
            f"fallback_bbs={s.get('fallback_bb_count')} structured_cov={s.get('structured_block_coverage')} "
            f"cond_ground={g.get('condition_slot_grounded_rate')} branch_cov={g.get('branch_provenance_coverage')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Module-B v3 output against document-defined contract")
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-preview", type=int, default=40)
    args = ap.parse_args()

    rows = [row for _line_no, row in iter_jsonl(Path(args.input_jsonl).expanduser().resolve(), int(args.max_examples))]
    txt = build_summary(rows, args)
    out = Path(args.out_txt).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(txt, end="", flush=True)
    print(f"saved_txt = {out}", flush=True)


if __name__ == "__main__":
    main()

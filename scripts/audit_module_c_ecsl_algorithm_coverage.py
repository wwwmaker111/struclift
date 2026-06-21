#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit implementation coverage for the Module-C ECSL design document.

This audit is deliberately honest: it distinguishes runnable local code from
external hooks and future training stages.  It should be used after code changes
to answer "which parts of the ECSL algorithm are implemented, partial, kept as
external hooks, or still missing?"
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


STATUS_ORDER = {
    "implemented": 0,
    "partial": 1,
    "keep_external": 2,
    "keep_training_stage": 3,
    "missing": 4,
}

CODE_PATHS = [
    "scripts/build_module_c_ecsl_slot_contracts.py",
    "scripts/build_module_c_ecsl_candidates.py",
    "scripts/pack_module_c_ecsl_training_manifest.py",
    "scripts/audit_module_c_ecsl_global_decode_seed.py",
    "scripts/audit_module_c_ecsl_algorithm_coverage.py",
    "scripts/run_module_c_ecsl_pipeline_local.py",
    "scripts/inject_module_c_ecsl_source_projection_candidates.py",
    "scripts/export_module_c_ecsl_v3_validation_tasks.py",
    "scripts/run_module_c_ecsl_v3_differential_lite.py",
    "scripts/export_module_c_ecsl_human_audit_tasks.py",
    "scripts/pack_module_c_ecsl_human_audit_v3_results.py",
]


def coverage_matrix() -> List[Dict[str, Any]]:
    return [
        {
            "id": "semantic_slot_schema",
            "phase": "Phase 1",
            "requirement": "SemanticSlot schema for skeleton-compatible slot contracts",
            "expected_status": "implemented",
            "keywords": ["SLOT_SCHEMA", "SemanticSlotLite.v0", "semantic_slots"],
            "notes": "O0/Lite schema is implemented without source AST projection.",
        },
        {
            "id": "skeleton_cfg_schema",
            "phase": "Phase 1",
            "requirement": "SkeletonCFG schema around fixed Module-B skeleton",
            "expected_status": "implemented",
            "keywords": ["skeleton_cfg", "SkeletonCFG.o0_metadata_only", "renderer_policy"],
            "notes": "Metadata-only CFG wrapper exists; full AST parser is not needed for O0.",
        },
        {
            "id": "skeleton_slot_lowerer",
            "phase": "Phase 1",
            "requirement": "Skeleton Slot Lowerer from placeholders to anchors",
            "expected_status": "implemented",
            "keywords": ["LOWERER_SCHEMA", "SkeletonSlotLowerer.v0", "lower_skeleton_slots"],
            "notes": "Text-based lowerer preserves Module-B skeleton tokens.",
        },
        {
            "id": "slot_grammar_checker",
            "phase": "Phase 1 / Phase 3",
            "requirement": "Slot grammar checker",
            "expected_status": "implemented",
            "keywords": ["def grammar_check", "def local_c_stub_check", "parser_type_local_stub_validation"],
            "notes": "Local V0 now combines slot grammar with a compiler-backed C stub when --v0-cc is available.",
        },
        {
            "id": "deterministic_renderer",
            "phase": "Phase 1 / Phase 5",
            "requirement": "Deterministic renderer preserving non-placeholder skeleton tokens",
            "expected_status": "implemented",
            "keywords": ["render_slot_candidate", "skeleton_preservation_check", "preserve_all_non_placeholder_skeleton_tokens"],
            "notes": "Renderer is used for V1 smoke validation and decode-seed audit.",
        },
        {
            "id": "observable_semantics",
            "phase": "Phase 1",
            "requirement": "Obs(slot) observable semantics contract",
            "expected_status": "implemented",
            "keywords": ["observable_semantics_contract", "observable_items", "do_not_compare"],
            "notes": "Observable items are declared; equivalence proof is V2/V3.",
        },
        {
            "id": "unified_ir_lite",
            "phase": "Phase 1",
            "requirement": "Unified IR lowering for source/binary/skeleton comparison",
            "expected_status": "partial",
            "keywords": ["unified_ir_contract", "source_teacher_ir_status", "binary_ir_status", "skeleton_ir_status"],
            "notes": "Skeleton IR and partial binary refs exist; source lowering is explicitly KEEP_EXTERNAL.",
        },
        {
            "id": "attention_mask_contract",
            "phase": "Phase 1 / Phase 11",
            "requirement": "Three-ring attention mask contract",
            "expected_status": "implemented",
            "keywords": ["attention_mask_contract", "region_bb", "boundary_bb", "dataflow_symbols"],
            "notes": "Region/boundary rings exist; dataflow ring becomes richer after V2 extraction.",
        },
        {
            "id": "source_cfg_ssa_candidate_extractor",
            "phase": "Phase 2",
            "requirement": "Source CFG/SSA candidate extractor",
            "expected_status": "partial",
            "keywords": ["source_cfg_candidates", "source_cfg", "source_ssa", "KEEP_EXTERNAL", "source_teacher_projection_fill"],
            "notes": "Source CFG/SSA remains external, but concrete source-teacher projection candidates can now be injected as C2 candidates for downstream validators.",
        },
        {
            "id": "binary_ir_candidate_extractor",
            "phase": "Phase 2",
            "requirement": "Binary IR candidate extractor",
            "expected_status": "partial",
            "keywords": ["cond_binary_candidates", "block_binary_candidates", "source=\"binary_ir\"", "synthetic_binary_candidates"],
            "notes": "O1 candidates plus V3-lite synthetic binary candidates cover simple condition/control cases; block/effect synthesis remains partial.",
        },
        {
            "id": "execution_trace_candidate_extractor",
            "phase": "Phase 2",
            "requirement": "Execution trace candidate extractor",
            "expected_status": "keep_external",
            "keywords": ["execution_trace_candidates", "test_inputs_or_traces", "trace_to_slot_observation_mapper"],
            "notes": "Trace candidate hook is present; actual trace collection is external.",
        },
        {
            "id": "rewrite_candidate_generators",
            "phase": "Phase 2 / Section 8",
            "requirement": "Explicit rewrite candidate generators",
            "expected_status": "partial",
            "keywords": ["REWRITE_KINDS", "early_return_if_else", "while_loop_break", "switch_if_chain", "branch_inversion"],
            "notes": "Rewrite request objects are emitted; concrete text rewrites require external verifier-backed synthesis.",
        },
        {
            "id": "llm_proposal_generator",
            "phase": "Phase 2",
            "requirement": "LLM proposal generator",
            "expected_status": "keep_external",
            "keywords": ["llm_proposal_request", "external_llm_generation", "validator_pass"],
            "notes": "Prompt/request records are emitted; no model call is performed here.",
        },
        {
            "id": "self_training_candidate_hook",
            "phase": "Phase 2 / Phase 4",
            "requirement": "Self-training candidate hook",
            "expected_status": "keep_training_stage",
            "keywords": ["self_training_candidates", "validated_model_prediction", "self_training"],
            "notes": "Future model predictions can be routed in; current script does not self-train.",
        },
        {
            "id": "semantic_match_cost",
            "phase": "Phase 2 / Section 9",
            "requirement": "Semantic matching cost matrix components",
            "expected_status": "implemented",
            "keywords": ["semantic_matching_report", "MATCHING_COMPONENT_NAMES", "cost_components"],
            "notes": "Cost object is auditable; unavailable semantic dimensions are marked not_available.",
        },
        {
            "id": "uot_matching",
            "phase": "Phase 2 / Section 9",
            "requirement": "Unbalanced Optimal Transport matching",
            "expected_status": "keep_external",
            "keywords": ["KEEP_EXTERNAL_uot_solver_not_connected", "unmatched_source_unit_policy", "unmatched_b_slot_policy"],
            "notes": "UOT contract and policies are recorded; actual solver is not connected.",
        },
        {
            "id": "v0_validator",
            "phase": "Phase 3",
            "requirement": "V0 parser/type/slot grammar validator",
            "expected_status": "implemented",
            "keywords": ["v0_parser_type_check", "local_c_stub_check", "parser_type_local_stub_validation"],
            "notes": "V0 implements parser/type/local-stub validation locally; if the compiler is missing it reports grammar-only pass instead of silent failure.",
        },
        {
            "id": "v1_validator",
            "phase": "Phase 3",
            "requirement": "V1 skeleton/CFG successor preservation validator",
            "expected_status": "partial",
            "keywords": ["skeleton_preservation_cfg_successor", "render_slot_candidate", "successor_evidence"],
            "notes": "Skeleton immutability is implemented and successor metadata is audited; missing/polarity-ambiguous successor facts are demoted to needs-V3 rather than treated as proof.",
        },
        {
            "id": "v2_validator",
            "phase": "Phase 3",
            "requirement": "V2 live-out/read-write/effect validator",
            "expected_status": "partial",
            "keywords": ["v2_static_semantic_check", "fact_state", "needs_v3_data_contract_proxy_not_clean_v2", "ssa_liveout_readwrite_effect"],
            "notes": "V2 distinguishes real empty/nonempty facts from proxy or missing facts; proxy evidence can feed V3 tasks but is not clean C2 proof.",
        },
        {
            "id": "v3_validator",
            "phase": "Phase 3",
            "requirement": "V3 trace/symbolic/differential execution validator",
            "expected_status": "keep_external",
            "keywords": ["v3_validation_check", "external_v3_jsonl", "v3_command", "V3ValidationTask", "DIAGNOSTIC_V3_ORACLE_TYPES"],
            "notes": "External JSONL/command hook and task export are implemented; real full trace/symbolic/differential backend remains external. Source projection/V3-lite are rejected as hard V3 by default.",
        },
        {
            "id": "human_semantic_audit_workflow",
            "phase": "Phase 3",
            "requirement": "Manual semantic audit path for strong slot evidence when automatic V3 backend is unavailable",
            "expected_status": "implemented",
            "keywords": ["HumanAuditTask", "human_audit_pass", "HumanAuditedStrongEvidence", "evidence_kind"],
            "notes": "Human audit can promote reviewed slot fills into hard SFT evidence; it is reported separately from automatic V3.",
        },
        {
            "id": "confidence_scorer",
            "phase": "Phase 3",
            "requirement": "Confidence scorer C0-C5 with hard CE policy",
            "expected_status": "implemented",
            "keywords": ["score_confidence", "confidence_level", "hard_ce_allowed", "TIER_WEIGHT"],
            "notes": "C1 priors are kept out of hard CE; C2 comes from V2 and C3+ from V3 evidence.",
        },
        {
            "id": "module_c_input_output_format",
            "phase": "Phase 4 / Section 11",
            "requirement": "Module-C input prompt and structured slot-patch output",
            "expected_status": "implemented",
            "keywords": ["build_prompt", "slot_patch_json", "Return JSON with one fill"],
            "notes": "Training record uses structured slot patch output rather than free full-function text.",
        },
        {
            "id": "multi_reference_sft_manifest",
            "phase": "Phase 4",
            "requirement": "Confidence-weighted multi-reference SFT records/manifest",
            "expected_status": "implemented",
            "keywords": ["confidence_weighted_multi_reference_sft", "normalize_reference_weights", "ecsl_training_manifest"],
            "notes": "Data packer is implemented; it does not run the trainer.",
        },
        {
            "id": "latent_em_plan",
            "phase": "Phase 4",
            "requirement": "Latent EM optional plan",
            "expected_status": "keep_training_stage",
            "keywords": ["latent_em", "KEEP_TRAINING_STAGE", "entropy floor"],
            "notes": "Manifest records the training-stage plan and guardrails.",
        },
        {
            "id": "consistency_training_plan",
            "phase": "Phase 4",
            "requirement": "Consistency training plan",
            "expected_status": "keep_training_stage",
            "keywords": ["consistency_training", "views", "Obs(slot)"],
            "notes": "Manifest records views and requirements; training loop is not implemented here.",
        },
        {
            "id": "self_training_plan",
            "phase": "Phase 4",
            "requirement": "Self-training plan",
            "expected_status": "keep_training_stage",
            "keywords": ["self_training", "promotion_rule", "C3/C4/C5"],
            "notes": "Promotion rule exists; actual iterative self-training is a later stage.",
        },
        {
            "id": "global_decode_seed",
            "phase": "Phase 5",
            "requirement": "Per-slot k-candidate table for global decoding",
            "expected_status": "implemented",
            "keywords": ["decode_seed_record", "candidate_table", "global_rerank_features"],
            "notes": "Seed table exists; it is not the final ILP/beam solver.",
        },
        {
            "id": "global_rerank_audit",
            "phase": "Phase 5",
            "requirement": "Global decode/render smoke audit",
            "expected_status": "partial",
            "keywords": ["audit_module_c_ecsl_global_decode_seed", "render_top1", "remaining_placeholders"],
            "notes": "Top-1 deterministic render audit exists; full compile/execution reranker is external.",
        },
        {
            "id": "full_function_validation",
            "phase": "Phase 5",
            "requirement": "Full-function compile/execution validation",
            "expected_status": "keep_external",
            "keywords": ["compile_success", "effect_equivalence", "live_out_equivalence", "global_rerank_features"],
            "notes": "Feature slots are named in decode seed; compiler/execution harness is not run.",
        },
        {
            "id": "grpo_reward_plan",
            "phase": "Phase 5",
            "requirement": "GRPO/RL reward plan",
            "expected_status": "keep_training_stage",
            "keywords": ["grpo_rl", "KEEP_LATER_STAGE", "Module C adapter/LoRA"],
            "notes": "Manifest records the reward-stage constraints; no RL training loop is implemented.",
        },
        {
            "id": "b_gate",
            "phase": "Gate",
            "requirement": "B skeleton reliability gate",
            "expected_status": "implemented",
            "keywords": ["b_gate_report", "strict_edge_preservation_ok", "slot_locality_ready", "contract_consistency_pass"],
            "notes": "Gate prevents failed B skeletons from becoming hard supervision.",
        },
        {
            "id": "curriculum_manifest",
            "phase": "Curriculum",
            "requirement": "Curriculum shards/manifest",
            "expected_status": "implemented",
            "keywords": ["curriculum_bucket", "curriculum_shards", "recommended_sequence"],
            "notes": "Packer writes bucketed JSONL shards.",
        },
        {
            "id": "metrics_ablation_hooks",
            "phase": "Experiment",
            "requirement": "Metrics and ablation hooks",
            "expected_status": "partial",
            "keywords": ["ecsl_algorithm_coverage", "stats", "bucket_dist", "candidate_source_dist"],
            "notes": "Reports expose counts/distributions; full experimental ablation runner is not implemented.",
        },
    ]


def load_texts(repo_root: Path, paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in paths:
        path = repo_root / rel
        if path.exists():
            out[rel] = path.read_text(encoding="utf-8", errors="replace")
    return out


def find_keyword_evidence(texts: Dict[str, str], keywords: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    evidence: Dict[str, List[Dict[str, Any]]] = {}
    for keyword in keywords:
        hits: List[Dict[str, Any]] = []
        pattern = re.escape(keyword)
        for rel, text in texts.items():
            for line_no, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    hits.append({"file": rel, "line": line_no, "text": line.strip()[:180]})
                    if len(hits) >= 6:
                        break
            if len(hits) >= 6:
                break
        evidence[keyword] = hits
    return evidence


def doc_requirement_present(doc_text: str, requirement_id: str, requirement: str) -> bool:
    probes = [requirement_id.replace("_", " "), requirement]
    lower_doc = doc_text.lower()
    for probe in probes:
        words = [w for w in re.split(r"[^A-Za-z0-9]+", probe.lower()) if len(w) >= 4]
        if words and sum(1 for word in words if word in lower_doc) >= min(2, len(words)):
            return True
    return False


def resolve_status(item: Dict[str, Any], evidence: Dict[str, List[Dict[str, Any]]]) -> str:
    expected = str(item["expected_status"])
    found = sum(1 for hits in evidence.values() if hits)
    required = len(item.get("keywords") or [])
    if found == 0:
        return "missing"
    if expected == "implemented" and found < max(1, min(required, 2)):
        return "partial"
    if expected in {"partial", "keep_external", "keep_training_stage"} and found < max(1, min(required, 2)):
        return "partial"
    return expected


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    doc_path = Path(args.doc_md).resolve()
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""
    code_texts = load_texts(repo_root, CODE_PATHS)

    rows: List[Dict[str, Any]] = []
    for item in coverage_matrix():
        evidence = find_keyword_evidence(code_texts, item["keywords"])
        status = resolve_status(item, evidence)
        rows.append(
            {
                "id": item["id"],
                "phase": item["phase"],
                "requirement": item["requirement"],
                "status": status,
                "expected_status": item["expected_status"],
                "doc_requirement_detected": doc_requirement_present(doc_text, item["id"], item["requirement"]),
                "keywords": item["keywords"],
                "evidence": evidence,
                "notes": item["notes"],
            }
        )

    status_dist = Counter(row["status"] for row in rows)
    phase_dist = Counter(row["phase"] for row in rows)
    report = {
        "schema": "ModuleC.ECSL.AlgorithmCoverageAudit.v0",
        "doc_md": str(doc_path),
        "repo_root": str(repo_root),
        "code_paths": [path for path in CODE_PATHS if (repo_root / path).exists()],
        "total_items": len(rows),
        "status_dist": dict(status_dist),
        "phase_dist": dict(phase_dist),
        "items": rows,
        "warnings": [
            "V2 is a conservative local contract gate; real empty facts, proxy facts, and missing facts are reported separately.",
            "V3-lite is diagnostic only by default; full trace/symbolic/differential coverage requires an external backend.",
            "Compile smoke and source-teacher projection are not semantic V3 proof.",
            "Human audit workflow can provide hard SFT evidence for reviewed slots, but it is not automatic V3.",
            "LLM generation, actual SFT/EM/consistency/self-training/GRPO loops are not run by these builders.",
            "Global decode audit is render-only unless a compiler/execution validator is supplied through the V3 hooks.",
        ],
    }

    out_json = Path(args.out_json)
    out_txt = Path(args.out_txt)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "=== Module C ECSL Algorithm Coverage Audit ===",
        f"doc_md = {doc_path}",
        f"repo_root = {repo_root}",
        "",
        f"total_items = {len(rows)}",
        f"status_dist = {status_dist.most_common()}",
        f"phase_dist = {phase_dist.most_common()}",
        "",
        "-- items --",
    ]
    for row in sorted(rows, key=lambda r: (STATUS_ORDER.get(r["status"], 99), r["phase"], r["id"])):
        hit_count = sum(1 for hits in row["evidence"].values() if hits)
        lines.append(
            f"[{row['status']}] {row['phase']} {row['id']} :: {row['requirement']} "
            f"(evidence_keywords={hit_count}/{len(row['keywords'])})"
        )
        lines.append(f"  note: {row['notes']}")
    lines.extend(["", "-- warnings --"])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit coverage of Module-C ECSL algorithm requirements.")
    ap.add_argument("--doc-md", default="StrucLift_ModuleC_ECSL_方案.md")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-txt", required=True)
    return ap.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()

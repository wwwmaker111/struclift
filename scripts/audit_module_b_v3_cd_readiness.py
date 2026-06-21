#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit whether Module-B v3 output is ready to feed Modules C/D.

This script evaluates the practical question after the Module-B full-pipeline
export: can the verified skeleton be used as a stable interface for downstream
expression/statement recovery and final decompilation?

It intentionally separates:

* safety readiness: B output is legal and CFG-faithful;
* limited C/D readiness: only constructable slots can be trained/filled;
* full C/D readiness: most functions have safe slot contracts and low escapes.

Source-AST similarity is reported only as a diagnostic.  It is not the primary
correctness criterion for Module B.
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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return float(default)
    if value != value:
        return float(default)
    return float(value)


def mean_val(vals: Sequence[Any]) -> float:
    xs = [safe_float(x) for x in vals]
    return float(statistics.mean(xs)) if xs else 0.0


def median_val(vals: Sequence[Any]) -> float:
    xs = [safe_float(x) for x in vals]
    return float(statistics.median(xs)) if xs else 0.0


def q(rows: Dict[str, Any], *keys: str) -> Any:
    cur: Any = rows
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def ranker_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "teacher_diagnostics", "ranker_metrics") or {}


def baseline_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "teacher_diagnostics", "baseline_metrics") or {}


def oracle_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "teacher_diagnostics", "oracle_metrics") or {}


def selected_report(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "validator", "selected_report") or {}


def modulec(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("module_c_contract") or {}


def constructability(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("candidate_constructability") or {}


def model_effect(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "document_metric_contract", "model_effect") or {}


def structuredness(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "document_metric_contract", "structuredness") or {}


def hard_correctness(row: Dict[str, Any]) -> Dict[str, Any]:
    return q(row, "document_metric_contract", "hard_correctness") or {}


def n_bb(row: Dict[str, Any]) -> int:
    blocks = q(row, "function_ir", "basic_blocks") or []
    if isinstance(blocks, list):
        return len(blocks)
    return 0


def has_teacher_kind(row: Dict[str, Any], kind: str) -> bool:
    outline = q(row, "teacher_diagnostics", "teacher_outline") or []
    return any(isinstance(item, list) and item and str(item[0]) == kind for item in outline)


def readiness_label(summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    n = max(1, int(summary["n"]))
    hard_ok = summary["hard_contract_pass"] == n
    false_verified_zero = summary["false_verified"] == 0
    real_ssa_rate = summary["real_ssa"] / n
    cond_ground = summary["condition_slot_grounded_rate_mean"]
    branch_cov = summary["branch_provenance_coverage_mean"]
    safe_rate = summary["safe_for_modulec"] / n
    any_safe_rate = summary["any_candidate_safe_for_modulec"] / n
    ready_rate = summary["decompile_ready_proxy"] / n
    fallback_rate = summary["fallback_region_functions"] / n
    goto_mean = summary["goto_mean"]
    structured_cov = summary["structured_coverage_mean"]
    switch_ready_rate = (
        summary["switch_ready"] / summary["switch_n"]
        if summary["switch_n"]
        else 1.0
    )
    large_ready_rate = (
        summary["large_ready"] / summary["large_n"]
        if summary["large_n"]
        else 1.0
    )

    if not hard_ok:
        reasons.append("hard safety contract is not 100%")
    if not false_verified_zero:
        reasons.append("false verified proxy is nonzero")
    if real_ssa_rate < 0.95:
        reasons.append("real SSA coverage is below 95%")
    if cond_ground < 0.90 or branch_cov < 0.90:
        reasons.append("condition grounding/provenance is below 90%")
    if safe_rate < 0.50:
        reasons.append("selected Module-C-safe functions below 50%")
    if any_safe_rate < 0.50:
        reasons.append("candidate pool Module-C-safe upper bound below 50%")
    if ready_rate < 0.60:
        reasons.append("decompile-ready proxy below 60%")
    if fallback_rate > 0.10:
        reasons.append("fallback region function rate above 10%")
    if goto_mean > 1.0:
        reasons.append("mean goto count above 1")
    if structured_cov < 0.80:
        reasons.append("structured block coverage below 80%")
    if summary["switch_n"] and switch_ready_rate < 0.50:
        reasons.append("switch-containing functions are not ready")
    if summary["large_n"] and large_ready_rate < 0.50:
        reasons.append("large functions are not ready")

    if hard_ok and false_verified_zero and real_ssa_rate >= 0.95 and cond_ground >= 0.90 and branch_cov >= 0.90:
        if (
            safe_rate >= 0.70
            and any_safe_rate >= 0.80
            and ready_rate >= 0.70
            and fallback_rate <= 0.10
            and goto_mean <= 1.0
            and structured_cov >= 0.80
            and switch_ready_rate >= 0.50
            and large_ready_rate >= 0.50
        ):
            return "B_READY_FOR_FULL_CD", reasons
        if any_safe_rate >= 0.15 and ready_rate >= 0.40:
            return "B_READY_FOR_LIMITED_CD", reasons
        return "B_SAFE_ONLY", reasons
    return "B_NOT_READY", reasons


def build_summary(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    n = len(rows)
    reports = [selected_report(r) for r in rows]
    hard_rows = [hard_correctness(r) for r in rows]
    rank_rows = [ranker_metrics(r) for r in rows]
    base_rows = [baseline_metrics(r) for r in rows]
    oracle_rows = [oracle_metrics(r) for r in rows]
    model_rows = [model_effect(r) for r in rows]
    struct_rows = [structuredness(r) for r in rows]
    c_rows = [modulec(r) for r in rows]
    cand_rows = [constructability(r) for r in rows]

    switch_rows = [r for r in rows if has_teacher_kind(r, "switch")]
    loop_rows = [r for r in rows if has_teacher_kind(r, "loop")]
    if_rows = [r for r in rows if has_teacher_kind(r, "if")]
    large_rows = [r for r in rows if n_bb(r) >= int(args.large_bb_threshold)]

    def bcount(seq: Sequence[Dict[str, Any]], key: str) -> int:
        return sum(1 for x in seq if bool(x.get(key)))

    def ready(seq: Sequence[Dict[str, Any]]) -> int:
        return sum(1 for r in seq if ranker_metrics(r).get("decompile_ready_proxy"))

    summary = {
        "n": n,
        "hard_contract_pass": sum(
            1
            for rep in reports
            if rep.get("valid")
            and rep.get("bb_coverage_ok")
            and rep.get("edge_preservation_ok")
            and rep.get("slot_grounding_ok")
            and rep.get("parse_success")
            and not rep.get("fatal_errors")
        ),
        "false_verified": bcount(hard_rows, "false_verified_proxy"),
        "real_ssa": bcount(model_rows, "real_ssa_available"),
        "condition_slot_grounded_rate_mean": mean_val(
            [m.get("condition_slot_grounded_rate") for m in model_rows]
        ),
        "branch_provenance_coverage_mean": mean_val(
            [m.get("branch_provenance_coverage") for m in model_rows]
        ),
        "safe_for_modulec": bcount(c_rows, "safe_for_modulec"),
        "any_candidate_safe_for_modulec": bcount(cand_rows, "any_candidate_safe_for_modulec"),
        "decompile_ready_proxy": bcount(rank_rows, "decompile_ready_proxy"),
        "fallback_region_functions": sum(
            1 for s in struct_rows if int(s.get("fallback_region_count") or 0) > 0
        ),
        "goto_mean": mean_val([s.get("goto_count") for s in struct_rows]),
        "escape_mean": mean_val([s.get("escape_count", s.get("goto_count")) for s in struct_rows]),
        "structured_coverage_mean": mean_val([s.get("structured_block_coverage") for s in struct_rows]),
        "switch_n": len(switch_rows),
        "switch_ready": ready(switch_rows),
        "loop_n": len(loop_rows),
        "loop_ready": ready(loop_rows),
        "if_n": len(if_rows),
        "if_ready": ready(if_rows),
        "large_n": len(large_rows),
        "large_ready": ready(large_rows),
    }
    label, reasons = readiness_label(summary)

    unsafe_reason = collections.Counter()
    unsafe_kind = collections.Counter()
    missing_dist = collections.Counter()
    unsafe_dist = collections.Counter()
    for r in rows:
        audit = q(r, "module_c_contract", "slot_label_constructability") or {}
        unsafe = audit.get("unsafe_nonempty_pred_slots") or []
        missing = audit.get("missing_teacher_primary_bbs") or []
        unsafe_dist[len(unsafe)] += 1
        missing_dist[len(missing)] += 1
        for item in unsafe:
            if isinstance(item, dict):
                unsafe_reason[str(item.get("reason") or "unknown")] += 1
                unsafe_kind[str(item.get("kind") or "unknown")] += 1

    bucket_dist = collections.Counter(
        b for r in rows for b in (r.get("failure_buckets") or [])
    )
    exclusive_bucket_dist = collections.Counter(
        tuple(r.get("failure_buckets") or []) for r in rows
    )
    safe_selected = [r for r in rows if modulec(r).get("safe_for_modulec")]
    any_safe_not_selected = [
        r
        for r in rows
        if constructability(r).get("any_candidate_safe_for_modulec")
        and not modulec(r).get("safe_for_modulec")
    ]
    ranker_changed = [
        r
        for r in rows
        if baseline_metrics(r).get("candidate_id") != ranker_metrics(r).get("candidate_id")
    ]
    ranker_bad = [
        r
        for r in rows
        if ranker_metrics(r).get("candidate_id") != oracle_metrics(r).get("candidate_id")
    ]

    lines: List[str] = []
    lines.append("===== MODULE B-v3 C/D READINESS AUDIT =====")
    lines.append(f"source_jsonl = {Path(args.input_jsonl).expanduser().resolve()}")
    lines.append(f"n = {n}")
    lines.append("")
    lines.append("-- final decision --")
    lines.append(f"readiness_label = {label}")
    lines.append(
        "decision = "
        + (
            "Module B is safe as a verified CFG-faithful skeleton layer, but not enough for full C/D decompilation."
            if label in {"B_SAFE_ONLY", "B_READY_FOR_LIMITED_CD"}
            else "Module B satisfies the configured C/D readiness thresholds."
            if label == "B_READY_FOR_FULL_CD"
            else "Module B does not satisfy the configured safety/readiness thresholds."
        )
    )
    lines.append("blocking_reasons = " + json.dumps(reasons, ensure_ascii=False))
    lines.append("")
    lines.append("-- hard gates --")
    lines.append(f"hard_contract_pass = {pct(summary['hard_contract_pass'], n)}")
    lines.append(f"false_verified_proxy = {pct(summary['false_verified'], n)}")
    lines.append(f"validator_valid = {pct(sum(1 for r in reports if r.get('valid')), n)}")
    lines.append(f"bb_coverage_ok = {pct(sum(1 for r in reports if r.get('bb_coverage_ok')), n)}")
    lines.append(f"edge_preservation_ok = {pct(sum(1 for r in reports if r.get('edge_preservation_ok')), n)}")
    lines.append(f"parse_success = {pct(sum(1 for r in reports if r.get('parse_success')), n)}")
    lines.append(f"validator_fatal_zero = {pct(sum(1 for r in reports if not r.get('fatal_errors')), n)}")
    lines.append("")
    lines.append("-- C/D interface readiness --")
    lines.append(f"selected_safe_for_modulec = {pct(summary['safe_for_modulec'], n)}")
    lines.append(f"any_candidate_safe_for_modulec = {pct(summary['any_candidate_safe_for_modulec'], n)}")
    lines.append(f"selected_pred_slots_labelable = {pct(bcount(c_rows, 'pred_slots_labelable'), n)}")
    lines.append(f"selected_teacher_primary_coverage_ok = {pct(bcount(c_rows, 'teacher_primary_coverage_ok'), n)}")
    lines.append(f"any_candidate_slots_labelable = {pct(bcount(cand_rows, 'any_candidate_slots_labelable'), n)}")
    lines.append(f"any_candidate_teacher_primary_coverage_ok = {pct(bcount(cand_rows, 'any_candidate_teacher_primary_coverage_ok'), n)}")
    lines.append(f"unsafe_nonempty_slot_count_mean = {mean_val([c.get('unsafe_nonempty_slot_count') for c in c_rows]):.4f}")
    lines.append(f"missing_teacher_primary_count_mean = {mean_val([c.get('missing_teacher_primary_count') for c in c_rows]):.4f}")
    lines.append(f"unsafe_slot_reason_dist = {unsafe_reason.most_common(20)}")
    lines.append(f"unsafe_slot_kind_dist = {unsafe_kind.most_common(20)}")
    lines.append(f"unsafe_slot_count_dist = {sorted(unsafe_dist.items())}")
    lines.append(f"missing_teacher_primary_count_dist = {sorted(missing_dist.items())}")
    lines.append("")
    lines.append("-- control-flow / structuredness --")
    lines.append(f"verified_skeleton = {pct(bcount(struct_rows, 'verified_skeleton'), n)}")
    lines.append(f"decompile_ready_proxy = {pct(summary['decompile_ready_proxy'], n)}")
    lines.append(f"fallback_region_function_rate = {pct(summary['fallback_region_functions'], n)}")
    lines.append(f"fallback_bb_count_mean = {mean_val([s.get('fallback_bb_count') for s in struct_rows]):.4f}")
    lines.append(f"goto_count_mean = {summary['goto_mean']:.4f}")
    lines.append(f"goto_count_median = {median_val([s.get('goto_count') for s in struct_rows]):.4f}")
    lines.append(f"escape_slot_count_mean = {summary['escape_mean']:.4f}")
    lines.append(f"escape_slot_count_median = {median_val([s.get('escape_count', s.get('goto_count')) for s in struct_rows]):.4f}")
    lines.append(f"structured_block_coverage_mean = {summary['structured_coverage_mean']:.4f}")
    lines.append(f"structured_block_coverage_median = {median_val([s.get('structured_block_coverage') for s in struct_rows]):.4f}")
    lines.append("")
    lines.append("-- SSA/provenance grounding --")
    lines.append(f"real_ssa_available_selected = {pct(summary['real_ssa'], n)}")
    for key in (
        "condition_ssa_available_rate",
        "real_condition_ssa_available_rate",
        "branch_provenance_coverage",
        "condition_branch_metadata_rate",
        "condition_slot_grounded_rate",
        "if_cond_grounded_rate",
        "loop_cond_grounded_rate",
        "switch_cond_grounded_rate",
    ):
        lines.append(f"{key}_mean = {mean_val([m.get(key) for m in model_rows]):.4f}")
    lines.append("")
    lines.append("-- source-AST diagnostics only --")
    for name, seq in (("baseline", base_rows), ("ranker", rank_rows), ("oracle", oracle_rows)):
        lines.append(
            f"{name}: ready={pct(bcount(seq, 'decompile_ready_proxy'), len(seq))} "
            f"outline_equal={pct(bcount(seq, 'outline_equal'), len(seq))} "
            f"source_soft={pct(bcount(seq, 'source_soft_recovered'), len(seq))} "
            f"outline_f1_mean={mean_val([x.get('outline_lcs_f1') for x in seq]):.4f} "
            f"goto_mean={mean_val([x.get('goto_count') for x in seq]):.4f}"
        )
    lines.append(f"ranker_changed_selection = {pct(len(ranker_changed), n)}")
    lines.append(f"ranker_missed_oracle = {pct(len(ranker_bad), n)}")
    lines.append("")
    lines.append("-- difficult buckets --")
    lines.append(
        f"switch: n={len(switch_rows)} ready={pct(summary['switch_ready'], len(switch_rows))} "
        f"safe={pct(sum(1 for r in switch_rows if modulec(r).get('safe_for_modulec')), len(switch_rows))} "
        f"outline_f1_mean={mean_val([ranker_metrics(r).get('outline_lcs_f1') for r in switch_rows]):.4f}"
    )
    lines.append(
        f"loop: n={len(loop_rows)} ready={pct(summary['loop_ready'], len(loop_rows))} "
        f"safe={pct(sum(1 for r in loop_rows if modulec(r).get('safe_for_modulec')), len(loop_rows))} "
        f"outline_f1_mean={mean_val([ranker_metrics(r).get('outline_lcs_f1') for r in loop_rows]):.4f}"
    )
    lines.append(
        f"if: n={len(if_rows)} ready={pct(summary['if_ready'], len(if_rows))} "
        f"safe={pct(sum(1 for r in if_rows if modulec(r).get('safe_for_modulec')), len(if_rows))} "
        f"outline_f1_mean={mean_val([ranker_metrics(r).get('outline_lcs_f1') for r in if_rows]):.4f}"
    )
    lines.append(
        f"large_bb_ge_{int(args.large_bb_threshold)}: n={len(large_rows)} "
        f"ready={pct(summary['large_ready'], len(large_rows))} "
        f"safe={pct(sum(1 for r in large_rows if modulec(r).get('safe_for_modulec')), len(large_rows))} "
        f"outline_f1_mean={mean_val([ranker_metrics(r).get('outline_lcs_f1') for r in large_rows]):.4f}"
    )
    lines.append("")
    lines.append("-- failure bucket distribution --")
    lines.append(f"failure_bucket_dist = {bucket_dist.most_common(50)}")
    lines.append(f"exclusive_failure_bucket_dist = {exclusive_bucket_dist.most_common(50)}")
    lines.append("")
    lines.append("-- action recommendation --")
    if label == "B_READY_FOR_FULL_CD":
        lines.append("1. Proceed to full Module C/D integration on this data slice.")
        lines.append("2. Keep validator and provenance gates as hard runtime checks.")
    elif label == "B_READY_FOR_LIMITED_CD":
        lines.append("1. Use Module B as the safe skeleton boundary.")
        lines.append("2. Train/evaluate Module C only on constructable/safe slots, not all slots.")
        lines.append("3. Do not claim full decompilation readiness yet.")
        lines.append("4. Improve candidate pool before expecting large gains: switch recovery, SESE/interval regions, loop exit structuring.")
        lines.append("5. Add a Module-C-aware selection mode only as a small patch; current upper bound is low.")
    elif label == "B_SAFE_ONLY":
        lines.append("1. Accept Module B only as a safety MVP.")
        lines.append("2. Do not connect full Module C/D training yet except for a tiny constructable subset.")
        lines.append("3. Fix candidate pool expressiveness and label alignment first.")
    else:
        lines.append("1. Fix hard validator/safety failures before downstream integration.")
    lines.append("")
    lines.append(f"-- selected safe examples, first {int(args.max_examples_preview)} --")
    for r in safe_selected[: int(args.max_examples_preview)]:
        m = ranker_metrics(r)
        lines.append(
            f"example={r.get('example')} func={r.get('func_name')} n_bb={n_bb(r)} "
            f"ready={m.get('decompile_ready_proxy')} f1={m.get('outline_lcs_f1')} "
            f"goto={m.get('goto_count')} fallback={m.get('fallback_bb_ratio')} "
            f"candidate={m.get('name')}"
        )
    lines.append("")
    lines.append(f"-- any-safe-but-selected-unsafe examples, first {int(args.max_examples_preview)} --")
    for r in any_safe_not_selected[: int(args.max_examples_preview)]:
        m = ranker_metrics(r)
        best = constructability(r).get("constructability_best_candidate") or {}
        lines.append(
            f"example={r.get('example')} func={r.get('func_name')} selected={m.get('name')} "
            f"best_safe={best.get('name')} selected_ready={m.get('decompile_ready_proxy')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Module-B v3 readiness for Module C/D integration")
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--large-bb-threshold", type=int, default=21)
    ap.add_argument("--max-examples-preview", type=int, default=30)
    args = ap.parse_args()

    rows = [row for _line_no, row in iter_jsonl(Path(args.input_jsonl).expanduser().resolve(), int(args.max_examples))]
    txt = build_summary(rows, args)
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt, encoding="utf-8")
    print(txt, end="", flush=True)
    print(f"saved_txt = {out_txt}", flush=True)


if __name__ == "__main__":
    main()

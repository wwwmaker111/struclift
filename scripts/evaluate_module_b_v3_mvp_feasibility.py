#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate whether the Module-B v3 MVP skeleton algorithm is feasible.

This evaluator reports where each CFG edge is explained:

* GOTO slot
* explicit structured edge recorded by the structurer
* fallback region
* linear adjacent slots
* structured semantic edge lowered from IF/LOOP/SWITCH
* uncovered

Use this script to judge the MVP as a safe skeleton layer, not as a final source
AST exact-match model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MVP_PATH = ROOT / "struclift" / "utils" / "module_b_v3_mvp.py"
MVP_SPEC = importlib.util.spec_from_file_location("module_b_v3_mvp_standalone", MVP_PATH)
if MVP_SPEC is None or MVP_SPEC.loader is None:
    raise RuntimeError(f"cannot load {MVP_PATH}")
MVP_MODULE = importlib.util.module_from_spec(MVP_SPEC)
sys.modules[MVP_SPEC.name] = MVP_MODULE
MVP_SPEC.loader.exec_module(MVP_MODULE)

analyse_cfg = MVP_MODULE.analyse_cfg
candidate_outline = MVP_MODULE.candidate_outline
generate_candidates = MVP_MODULE.generate_candidates
render_candidate = MVP_MODULE.render_candidate
select_candidate = MVP_MODULE.select_candidate
validate_candidate = MVP_MODULE.validate_candidate
explain_candidate_edges = MVP_MODULE.explain_candidate_edges


Edge = Tuple[int, int]


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def read_source_by_line(path: Path, wanted: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    wanted_set = {int(x) for x in wanted if int(x) > 0}
    out: Dict[int, Dict[str, Any]] = {}
    if not wanted_set:
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no not in wanted_set:
                continue
            line = line.strip()
            if line:
                out[line_no] = json.loads(line)
            if len(out) >= len(wanted_set):
                break
    return out


def teacher_outline(teacher: Optional[Dict[str, Any]]) -> List[Tuple[str, int]]:
    if not teacher:
        return []
    pairs = teacher.get("teacher_outline_pairs")
    if isinstance(pairs, list):
        out: List[Tuple[str, int]] = []
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((str(item[0]), int(item[1])))
                except Exception:
                    continue
        return out
    controls = teacher.get("controls") if isinstance(teacher.get("controls"), list) else []
    return [
        (str(c.get("kind") or "unknown"), int(c.get("depth") or 0))
        for c in controls
        if str(c.get("kind") or "") in {"if", "loop", "switch"}
    ]


def decl_from_teacher(teacher: Optional[Dict[str, Any]]) -> str:
    text = str((teacher or {}).get("teacher_skeleton") or "")
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.endswith("{"):
        first = first[:-1].rstrip()
    if first and "(" in first and ")" in first:
        return first
    return "int func(...)"


def normalize_depths(outline: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not outline:
        return []
    levels = {d: i for i, d in enumerate(sorted({int(d) for _k, d in outline}))}
    return [(str(k), int(levels[int(d)])) for k, d in outline]


def kindseq(outline: Sequence[Tuple[str, int]]) -> List[str]:
    return [str(k) for k, _d in outline]


def lcs_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return int(prev[-1])


def seq_f1(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    hit = lcs_len(pred, gold)
    precision = hit / max(1, len(pred))
    recall = hit / max(1, len(gold))
    return float(2.0 * precision * recall / max(1e-8, precision + recall))


def outline_metrics(pred: Sequence[Tuple[str, int]], gold: Sequence[Tuple[str, int]]) -> Dict[str, Any]:
    pn = normalize_depths(pred)
    gn = normalize_depths(gold)
    pk = kindseq(pred)
    gk = kindseq(gold)
    p_depth = max((d for _k, d in pn), default=-1) + 1
    g_depth = max((d for _k, d in gn), default=-1) + 1
    return {
        "has_teacher_outline": bool(gold),
        "outline_equal": bool(gold) and pn == gn,
        "kindseq_equal": bool(gold) and pk == gk,
        "outline_lcs_f1": round(seq_f1(pn, gn), 6),
        "kind_lcs_f1": round(seq_f1(pk, gk), 6),
        "control_count_gap": int(len(pred) - len(gold)),
        "max_depth_gap": int(p_depth - g_depth),
        "pred_outline": [[k, d] for k, d in pred],
        "teacher_outline": [[k, d] for k, d in gold],
    }


def edge_explanation_audit(cfg: Any, cand: Any) -> Dict[str, Any]:
    core = explain_candidate_edges(cfg, cand)
    n_edges = max(1, len(cfg.edges))
    counts = dict(core.get("counts") or {})
    explained = len(cfg.edges) - int(counts.get("uncovered") or 0)
    explicit_or_fallback = (
        int(counts.get("goto_slot") or 0)
        + int(counts.get("explicit_structured") or 0)
        + int(counts.get("fallback_internal") or 0)
        + int(counts.get("linear_slot") or 0)
    )
    return {
        "edge_count": int(len(cfg.edges)),
        "edge_category_counts": counts,
        "edge_category_preview": dict(core.get("preview") or {}),
        "structured_semantic_reasons": dict(core.get("semantic_reasons") or {}),
        "strict_edge_preservation_ok": int(counts.get("uncovered") or 0) == 0,
        "edge_preservation_ratio": round(float(explained / n_edges), 6),
        "explicit_or_fallback_edge_ratio": round(float(explicit_or_fallback / n_edges), 6),
        "structured_semantic_edge_ratio": round(float(int(counts.get("structured_semantic") or 0) / n_edges), 6),
        "fallback_edge_ratio": round(float(int(counts.get("fallback_internal") or 0) / n_edges), 6),
        "goto_edge_ratio": round(float(int(counts.get("goto_slot") or 0) / n_edges), 6),
    }


def source_style_score(metrics: Dict[str, Any], report: Dict[str, Any]) -> float:
    valid = 1.0 if bool(report.get("valid")) else 0.0
    fallback_bb = float(report.get("fallback_bb_count") or 0.0)
    goto_count = float(report.get("goto_count") or 0.0)
    return float(
        0.48 * float(metrics.get("outline_lcs_f1") or 0.0)
        + 0.32 * float(metrics.get("kind_lcs_f1") or 0.0)
        + 0.20 * valid
        - 0.01 * fallback_bb
        - 0.01 * goto_count
    )


def classify_feasibility(
    *,
    report: Dict[str, Any],
    edge_audit: Dict[str, Any],
    metrics: Dict[str, Any],
    n_bb: int,
    max_fallback_bb_ratio: float,
    max_goto_count: int,
    min_outline_f1: float,
    min_kind_f1: float,
) -> Dict[str, Any]:
    fallback_bb_ratio = float(report.get("fallback_bb_count") or 0) / max(1, int(n_bb))
    hard_valid = bool(
        report.get("valid")
        and report.get("bb_coverage_ok")
        and report.get("unique_primary_ok")
        and report.get("slot_grounding_ok")
        and report.get("parse_success")
        and not (report.get("fatal_errors") or [])
        and edge_audit.get("strict_edge_preservation_ok")
    )
    cost_feasible = bool(
        hard_valid
        and fallback_bb_ratio <= float(max_fallback_bb_ratio)
        and int(report.get("goto_count") or 0) <= int(max_goto_count)
    )
    source_soft = bool(
        hard_valid
        and bool(metrics.get("has_teacher_outline"))
        and float(metrics.get("outline_lcs_f1") or 0.0) >= float(min_outline_f1)
        and float(metrics.get("kind_lcs_f1") or 0.0) >= float(min_kind_f1)
        and abs(int(metrics.get("control_count_gap") or 0)) <= 1
    )
    decompile_ready_proxy = bool(cost_feasible and source_soft)
    return {
        "hard_valid": hard_valid,
        "cost_feasible": cost_feasible,
        "source_soft_recovered": source_soft,
        "decompile_ready_proxy": decompile_ready_proxy,
        "fallback_bb_ratio": round(fallback_bb_ratio, 6),
        "thresholds": {
            "max_fallback_bb_ratio": float(max_fallback_bb_ratio),
            "max_goto_count": int(max_goto_count),
            "min_outline_f1": float(min_outline_f1),
            "min_kind_f1": float(min_kind_f1),
        },
    }


def row_func_name(row: Dict[str, Any], teacher: Optional[Dict[str, Any]]) -> Any:
    return row.get("func_name") or row.get("function") or (teacher or {}).get("func_name")


def prepare_rows(source_jsonl: Path, teacher_jsonl: Optional[Path], max_examples: int) -> List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
    if teacher_jsonl is None:
        return [(line_no, row, None) for line_no, row in iter_jsonl(source_jsonl, max_examples)]
    teachers = [teacher for _line_no, teacher in iter_jsonl(teacher_jsonl, max_examples)]
    source_by_line = read_source_by_line(source_jsonl, [int(t.get("input_line") or 0) for t in teachers])
    rows: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for teacher in teachers:
        line_no = int(teacher.get("input_line") or 0)
        row = source_by_line.get(line_no)
        if row is not None:
            rows.append((line_no, row, teacher))
    return rows


def summarize_numeric(records: Sequence[Dict[str, Any]], path: Sequence[str]) -> Tuple[float, float]:
    vals: List[float] = []
    for rec in records:
        obj: Any = rec
        for key in path:
            obj = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(obj, (int, float)):
            vals.append(float(obj))
    if not vals:
        return 0.0, 0.0
    return float(statistics.mean(vals)), float(statistics.median(vals))


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def dist(records: Sequence[Dict[str, Any]], path: Sequence[str]) -> List[Tuple[Any, int]]:
    c = Counter()
    for rec in records:
        obj: Any = rec
        for key in path:
            obj = obj.get(key) if isinstance(obj, dict) else None
        c[obj] += 1
    return sorted(c.items(), key=lambda kv: (str(kv[0]), kv[1]))


def build_report(records: Sequence[Dict[str, Any]], args: argparse.Namespace, elapsed: float) -> str:
    n = len(records)
    counters = Counter()
    name_dist = Counter()
    edge_cat_total = Counter()
    for rec in records:
        report = rec.get("validator_report") or {}
        edge = rec.get("edge_audit") or {}
        feas = rec.get("feasibility") or {}
        metrics = rec.get("outline_metrics") or {}
        name_dist[str(rec.get("selected_candidate_name"))] += 1
        for key in (
            "valid",
            "bb_coverage_ok",
            "unique_primary_ok",
            "edge_preservation_ok",
            "slot_grounding_ok",
            "parse_success",
        ):
            if report.get(key):
                counters[key] += 1
        if not (report.get("fatal_errors") or []):
            counters["fatal_error_zero"] += 1
        for key in (
            "strict_edge_preservation_ok",
        ):
            if edge.get(key):
                counters[key] += 1
        for key in (
            "hard_valid",
            "cost_feasible",
            "source_soft_recovered",
            "decompile_ready_proxy",
        ):
            if feas.get(key):
                counters[key] += 1
        for key in ("outline_equal", "kindseq_equal"):
            if metrics.get(key):
                counters[key] += 1
        for key, value in (edge.get("edge_category_counts") or {}).items():
            edge_cat_total[key] += int(value)

    outline_mean, outline_median = summarize_numeric(records, ["outline_metrics", "outline_lcs_f1"])
    kind_mean, kind_median = summarize_numeric(records, ["outline_metrics", "kind_lcs_f1"])
    fallback_ratio_mean, fallback_ratio_median = summarize_numeric(records, ["feasibility", "fallback_bb_ratio"])
    semantic_mean, semantic_median = summarize_numeric(records, ["edge_audit", "structured_semantic_edge_ratio"])
    explicit_mean, explicit_median = summarize_numeric(records, ["edge_audit", "explicit_or_fallback_edge_ratio"])
    strict_mean, strict_median = summarize_numeric(records, ["edge_audit", "edge_preservation_ratio"])

    lines: List[str] = []
    lines.append("===== MODULE B-v3 MVP FEASIBILITY EVALUATION =====")
    lines.append(f"source_jsonl = {Path(args.source_jsonl).expanduser().resolve()}")
    lines.append(f"teacher_jsonl = {Path(args.teacher_jsonl).expanduser().resolve() if args.teacher_jsonl else None}")
    lines.append(f"n = {n}")
    lines.append(
        "note = strict MVP validator requires every CFG edge to be explained by "
        "structured semantics, linear slots, fallback regions, or explicit GOTO."
    )
    lines.append("")
    lines.append("-- code / validator sanity --")
    lines.append("code_check = syntax/runtime checked by py_compile before running this script")
    lines.append("validator_limitation = SSA/value grounding is not checked in this CFG-only MVP")
    lines.append("recommended_primary_metric = hard_valid + fallback/goto cost + source_soft_recovered")
    lines.append("")
    lines.append("-- hard safety --")
    for key in (
        "valid",
        "bb_coverage_ok",
        "unique_primary_ok",
        "edge_preservation_ok",
        "slot_grounding_ok",
        "parse_success",
        "fatal_error_zero",
        "strict_edge_preservation_ok",
    ):
        lines.append(f"{key} = {pct(counters[key], n)}")
    lines.append("")
    lines.append("-- feasibility rates --")
    for key in (
        "hard_valid",
        "cost_feasible",
        "source_soft_recovered",
        "decompile_ready_proxy",
    ):
        lines.append(f"{key} = {pct(counters[key], n)}")
    lines.append(
        f"thresholds = fallback_bb_ratio<={args.max_fallback_bb_ratio}, "
        f"goto_count<={args.max_goto_count}, outline_f1>={args.min_outline_f1}, "
        f"kind_f1>={args.min_kind_f1}"
    )
    lines.append("")
    lines.append("-- source-structure proxy, audit only --")
    lines.append(f"outline_equal = {pct(counters['outline_equal'], n)}")
    lines.append(f"kindseq_equal = {pct(counters['kindseq_equal'], n)}")
    lines.append(f"outline_lcs_f1_mean = {outline_mean:.4f}")
    lines.append(f"outline_lcs_f1_median = {outline_median:.4f}")
    lines.append(f"kind_lcs_f1_mean = {kind_mean:.4f}")
    lines.append(f"kind_lcs_f1_median = {kind_median:.4f}")
    lines.append(f"control_count_gap_dist = {dist(records, ['outline_metrics', 'control_count_gap'])}")
    lines.append(f"max_depth_gap_dist = {dist(records, ['outline_metrics', 'max_depth_gap'])}")
    lines.append("")
    lines.append("-- fallback / goto / structure cost --")
    lines.append(f"selected_candidate_name_dist = {name_dist.most_common(30)}")
    lines.append(f"fallback_region_count_dist = {dist(records, ['validator_report', 'fallback_region_count'])}")
    lines.append(f"fallback_bb_count_dist = {dist(records, ['validator_report', 'fallback_bb_count'])}")
    lines.append(f"fallback_bb_ratio_mean = {fallback_ratio_mean:.4f}")
    lines.append(f"fallback_bb_ratio_median = {fallback_ratio_median:.4f}")
    lines.append(f"goto_count_dist = {dist(records, ['validator_report', 'goto_count'])}")
    lines.append(f"structured_control_count_dist = {dist(records, ['validator_report', 'structured_control_count'])}")
    lines.append("")
    lines.append("-- edge explanation audit --")
    lines.append(f"edge_category_total = {sorted(edge_cat_total.items())}")
    semantic_reasons_total = Counter()
    for rec in records:
        semantic_reasons_total.update((rec.get("edge_audit") or {}).get("structured_semantic_reasons") or {})
    lines.append(f"structured_semantic_reason_total = {semantic_reasons_total.most_common()}")
    lines.append(f"strict_edge_preservation_ratio_mean = {strict_mean:.4f}")
    lines.append(f"strict_edge_preservation_ratio_median = {strict_median:.4f}")
    lines.append(f"structured_semantic_edge_ratio_mean = {semantic_mean:.4f}")
    lines.append(f"structured_semantic_edge_ratio_median = {semantic_median:.4f}")
    lines.append(f"explicit_or_fallback_edge_ratio_mean = {explicit_mean:.4f}")
    lines.append(f"explicit_or_fallback_edge_ratio_median = {explicit_median:.4f}")
    lines.append("")
    lines.append(f"-- badcase preview, first {int(args.max_bad_preview)} --")

    def bad_key(rec: Dict[str, Any]) -> Tuple[int, float, int, float]:
        feas = rec.get("feasibility") or {}
        metrics = rec.get("outline_metrics") or {}
        report = rec.get("validator_report") or {}
        return (
            0 if feas.get("decompile_ready_proxy") else 1,
            -float(metrics.get("outline_lcs_f1") or 0.0),
            int(report.get("fallback_bb_count") or 0),
            float((rec.get("edge_audit") or {}).get("edge_preservation_ratio") or 0.0),
        )

    bad = sorted(records, key=bad_key, reverse=True)
    for rec in bad[: int(args.max_bad_preview)]:
        report = rec.get("validator_report") or {}
        feas = rec.get("feasibility") or {}
        metrics = rec.get("outline_metrics") or {}
        edge = rec.get("edge_audit") or {}
        reasons: List[str] = []
        if not feas.get("hard_valid"):
            reasons.append("not_hard_valid")
        if float(feas.get("fallback_bb_ratio") or 0.0) > float(args.max_fallback_bb_ratio):
            reasons.append("high_fallback")
        if int(report.get("goto_count") or 0) > int(args.max_goto_count):
            reasons.append("high_goto")
        if metrics.get("has_teacher_outline") and float(metrics.get("outline_lcs_f1") or 0.0) < float(args.min_outline_f1):
            reasons.append("low_outline_f1")
        if not edge.get("strict_edge_preservation_ok"):
            reasons.append("unexplained_edges")
        lines.append(
            f"example={rec.get('example')} input_line={rec.get('input_line')} "
            f"func={rec.get('func_name')} n_bb={rec.get('n_bb')} "
            f"candidate={rec.get('selected_candidate_name')} reasons={reasons} "
            f"hard={feas.get('hard_valid')} cost={feas.get('cost_feasible')} "
            f"soft={feas.get('source_soft_recovered')} ready={feas.get('decompile_ready_proxy')} "
            f"outline_f1={metrics.get('outline_lcs_f1')} kind_f1={metrics.get('kind_lcs_f1')} "
            f"fallback_ratio={feas.get('fallback_bb_ratio')} goto={report.get('goto_count')} "
            f"edge_preservation_ratio={edge.get('edge_preservation_ratio')} "
            f"semantic_edge_ratio={edge.get('structured_semantic_edge_ratio')} "
            f"teacher_outline={metrics.get('teacher_outline')} pred_outline={metrics.get('pred_outline')}"
        )
        preview = edge.get("edge_category_preview") or {}
        if preview.get("uncovered"):
            lines.append(f"  uncovered_edges={preview.get('uncovered')}")
        if preview.get("structured_semantic"):
            lines.append(f"  structured_semantic_edges={preview.get('structured_semantic')}")

    lines.append("")
    lines.append(f"elapsed_sec = {elapsed:.1f}")
    return "\n".join(lines) + "\n"


def evaluate(args: argparse.Namespace) -> None:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    teacher_jsonl = Path(args.teacher_jsonl).expanduser().resolve() if args.teacher_jsonl else None
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = prepare_rows(source_jsonl, teacher_jsonl, int(args.max_examples))
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for example, (line_no, row, teacher) in enumerate(rows):
            start = time.time()
            cfg = analyse_cfg(row)
            candidates = generate_candidates(
                cfg,
                max_region_nodes=int(args.max_region_nodes),
                max_mixed_headers=int(args.max_mixed_if_headers),
                candidate_beam=int(args.candidate_beam),
            )
            cand, report_obj, _text = select_candidate(cfg, candidates)
            rendered = render_candidate(cand, decl_line=decl_from_teacher(teacher))
            report = validate_candidate(cfg, cand, rendered_text=rendered).asdict()
            pred_outline = candidate_outline(cand)
            gold_outline = teacher_outline(teacher)
            metrics = outline_metrics(pred_outline, gold_outline)
            edge_audit = edge_explanation_audit(cfg, cand)
            feasibility = classify_feasibility(
                report=report,
                edge_audit=edge_audit,
                metrics=metrics,
                n_bb=int(cfg.n_bb),
                max_fallback_bb_ratio=float(args.max_fallback_bb_ratio),
                max_goto_count=int(args.max_goto_count),
                min_outline_f1=float(args.min_outline_f1),
                min_kind_f1=float(args.min_kind_f1),
            )
            rec = {
                "example": example,
                "input_line": line_no,
                "func_name": row_func_name(row, teacher),
                "n_bb": int(cfg.n_bb),
                "n_edges": int(len(cfg.edges)),
                "selected_candidate_name": cand.name,
                "candidate_count": int(len(candidates)),
                "validator_report": report,
                "edge_audit": edge_audit,
                "outline_metrics": metrics,
                "feasibility": feasibility,
                "elapsed_sec": round(float(time.time() - start), 6),
            }
            if bool(args.include_skeletons):
                rec["skeleton"] = rendered
            records.append(rec)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = build_report(records, args, time.time() - t0)
    out_txt.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"saved_txt = {out_txt}", flush=True)
    print(f"saved_jsonl = {out_jsonl}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Module-B v3 MVP skeleton feasibility")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, default=None)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=100)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=4)
    ap.add_argument("--candidate-beam", type=int, default=160)
    ap.add_argument("--max-fallback-bb-ratio", type=float, default=0.30)
    ap.add_argument("--max-goto-count", type=int, default=2)
    ap.add_argument("--min-outline-f1", type=float, default=0.80)
    ap.add_argument("--min-kind-f1", type=float, default=0.80)
    ap.add_argument("--max-bad-preview", type=int, default=80)
    ap.add_argument("--include-skeletons", action="store_true")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Module-B v3 candidates with a source-CFG-aware metric.

This is a diagnostic metric, not a replacement for the strict binary-CFG
validator.  It projects binary CFG edges to source-statement CFG edges through
``alignment_matrix`` and reports whether the projected edges are explained by
structured skeleton regions, fallback regions, or explicit goto slots.
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
generate_candidates = MVP_MODULE.generate_candidates


Edge = Tuple[int, int]
SrcEdge = Tuple[int, int]


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def read_source_by_input_line(path: Path, wanted: Sequence[int]) -> Dict[int, Dict[str, Any]]:
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


def coerce_edge_index(raw: Any) -> Set[Edge]:
    edges: Set[Edge] = set()
    if isinstance(raw, list) and len(raw) == 2:
        srcs = raw[0] if isinstance(raw[0], list) else []
        dsts = raw[1] if isinstance(raw[1], list) else []
        for s, d in zip(srcs, dsts):
            try:
                edges.add((int(s), int(d)))
            except Exception:
                continue
    return edges


def source_edges(row: Dict[str, Any]) -> Set[SrcEdge]:
    return coerce_edge_index(row.get("src_edge_index"))


def bb_to_source_statements(
    row: Dict[str, Any],
    *,
    n_bb: int,
    topk: int,
    min_frac: float,
    min_weight: float,
) -> List[Set[int]]:
    """Map each BB to one or more source statement ids via alignment_matrix."""
    n_src = len(row.get("src_stmt_lengths") or [])
    matrix = row.get("alignment_matrix")
    out: List[Set[int]] = [set() for _ in range(n_bb)]
    if not isinstance(matrix, list) or not matrix or n_src <= 0:
        return out
    for bb in range(min(n_bb, len(matrix))):
        vals = matrix[bb]
        if not isinstance(vals, list) or not vals:
            continue
        pairs: List[Tuple[float, int]] = []
        for j, val in enumerate(vals[:n_src]):
            try:
                weight = float(val)
            except Exception:
                continue
            if weight > 0.0:
                pairs.append((weight, int(j)))
        if not pairs:
            continue
        pairs.sort(reverse=True)
        peak = pairs[0][0]
        keep_threshold = max(float(min_weight), float(peak) * float(min_frac))
        kept = [j for weight, j in pairs if weight >= keep_threshold]
        if topk > 0:
            kept = kept[: int(topk)]
        out[bb].update(kept)
    return out


def prf(pred: Set[SrcEdge], gold: Set[SrcEdge]) -> Dict[str, float]:
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "match": 0.0}
    hit = len(pred & gold)
    precision = hit / max(1, len(pred))
    recall = hit / max(1, len(gold))
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "match": float(hit),
    }


def edge_categories(cfg: Any, cand: Any) -> Dict[str, Set[Edge]]:
    goto_edges = {
        (int(slot.edge[0]), int(slot.edge[1]))
        for slot in cand.slots
        if slot.slot_type == "GOTO" and slot.edge is not None
    }
    explicit_edges = {(int(s), int(d)) for s, d in cand.explicit_edges}
    explicit_non_goto = explicit_edges - goto_edges
    fallback_regions = [set(int(x) for x in reg) for reg in cand.fallback_regions]
    owners = MVP_MODULE._slot_owners(cand.slots)
    structured_semantic: Dict[Edge, str] = {}
    MVP_MODULE._collect_structured_semantic_edges(cfg, cand.root, structured_semantic)
    categories: Dict[str, Set[Edge]] = {
        "goto_slot": set(),
        "explicit_structured": set(),
        "fallback_internal": set(),
        "linear_slot": set(),
        "structured_semantic": set(),
        "uncovered": set(),
    }
    for raw_edge in cfg.edges:
        edge = (int(raw_edge[0]), int(raw_edge[1]))
        if edge in goto_edges:
            categories["goto_slot"].add(edge)
        elif edge in explicit_non_goto:
            categories["explicit_structured"].add(edge)
        elif MVP_MODULE._edge_explained_by_region(edge, fallback_regions):
            categories["fallback_internal"].add(edge)
        elif MVP_MODULE._edge_explained_by_linear_slots(edge, owners, cand.slots):
            categories["linear_slot"].add(edge)
        elif edge in structured_semantic:
            categories["structured_semantic"].add(edge)
        else:
            categories["uncovered"].add(edge)
    return categories


def project_bin_edges(edges: Iterable[Edge], bb_to_src: Sequence[Set[int]]) -> Set[SrcEdge]:
    out: Set[SrcEdge] = set()
    for b0, b1 in edges:
        if not (0 <= int(b0) < len(bb_to_src) and 0 <= int(b1) < len(bb_to_src)):
            continue
        for s0 in bb_to_src[int(b0)]:
            for s1 in bb_to_src[int(b1)]:
                if int(s0) != int(s1):
                    out.add((int(s0), int(s1)))
    return out


def project_bbs(bbs: Iterable[int], bb_to_src: Sequence[Set[int]]) -> Set[int]:
    out: Set[int] = set()
    for bb in bbs:
        if 0 <= int(bb) < len(bb_to_src):
            out.update(int(x) for x in bb_to_src[int(bb)])
    return out


def candidate_source_cfg_metrics(
    *,
    row: Dict[str, Any],
    cfg: Any,
    cand: Any,
    bb_to_src: Sequence[Set[int]],
    gold_src_edges: Set[SrcEdge],
) -> Dict[str, Any]:
    cats = edge_categories(cfg, cand)
    structured_edges = (
        cats["structured_semantic"]
        | cats["explicit_structured"]
        | cats["linear_slot"]
    )
    fallback_edges = cats["fallback_internal"]
    goto_edges = cats["goto_slot"]
    all_projected = project_bin_edges(set(cfg.edges), bb_to_src)
    structured_projected = project_bin_edges(structured_edges, bb_to_src)
    fallback_projected = project_bin_edges(fallback_edges, bb_to_src)
    goto_projected = project_bin_edges(goto_edges, bb_to_src)

    aligned_src = set().union(*bb_to_src) if bb_to_src else set()
    fallback_bbs: Set[int] = set()
    for reg in cand.fallback_regions:
        fallback_bbs.update(int(x) for x in reg)
    goto_bbs: Set[int] = set()
    structured_bbs: Set[int] = set()
    for slot in cand.slots:
        bbs = [int(x) for x in (slot.primary_bbs or []) + (slot.context_bbs or [])]
        if slot.slot_type == "GOTO":
            goto_bbs.update(bbs)
        elif slot.slot_type == "UNSTRUCTURED_REGION":
            fallback_bbs.update(bbs)
        else:
            structured_bbs.update(bbs)
    fallback_src = project_bbs(fallback_bbs, bb_to_src)
    goto_src = project_bbs(goto_bbs, bb_to_src)
    structured_src = project_bbs(structured_bbs - fallback_bbs, bb_to_src)

    all_prf = prf(all_projected, gold_src_edges)
    struct_prf = prf(structured_projected, gold_src_edges)
    fallback_prf = prf(fallback_projected, gold_src_edges)
    goto_prf = prf(goto_projected, gold_src_edges)
    n_aligned = max(1, len(aligned_src))
    return {
        "src_cfg_available": bool(gold_src_edges and aligned_src),
        "src_stmt_count": int(len(row.get("src_stmt_lengths") or [])),
        "src_edge_count": int(len(gold_src_edges)),
        "aligned_src_stmt_count": int(len(aligned_src)),
        "bin_to_src_projected_edge_count": int(len(all_projected)),
        "src_cfg_all_precision": all_prf["precision"],
        "src_cfg_all_recall": all_prf["recall"],
        "src_cfg_all_f1": all_prf["f1"],
        "src_cfg_struct_precision": struct_prf["precision"],
        "src_cfg_struct_recall": struct_prf["recall"],
        "src_cfg_struct_f1": struct_prf["f1"],
        "src_cfg_fallback_recall": fallback_prf["recall"],
        "src_cfg_goto_recall": goto_prf["recall"],
        "structured_src_stmt_coverage": round(float(len(structured_src) / n_aligned), 6),
        "fallback_src_stmt_coverage": round(float(len(fallback_src) / n_aligned), 6),
        "goto_src_stmt_coverage": round(float(len(goto_src) / n_aligned), 6),
        "projected_struct_edge_count": int(len(structured_projected)),
        "projected_fallback_edge_count": int(len(fallback_projected)),
        "projected_goto_edge_count": int(len(goto_projected)),
        "binary_edge_category_counts": {k: int(len(v)) for k, v in cats.items()},
    }


def find_result_candidate(record: Dict[str, Any], candidate_id: Any) -> Optional[Dict[str, Any]]:
    for cand in record.get("candidates") or []:
        if cand.get("candidate_id") == candidate_id:
            return cand
    return None


def mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows if r.get("src_cfg_available")]
    return float(statistics.mean(vals)) if vals else 0.0


def source_cfg_good(src: Dict[str, Any], *, recall_threshold: float, f1_threshold: float) -> bool:
    return (
        bool(src.get("src_cfg_available"))
        and float(src.get("src_cfg_struct_recall") or 0.0) >= float(recall_threshold)
        and float(src.get("src_cfg_struct_f1") or 0.0) >= float(f1_threshold)
    )


def ast_soft_good(metrics: Dict[str, Any]) -> bool:
    return bool(metrics.get("source_soft_recovered"))


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def build_summary(records: Sequence[Dict[str, Any]], elapsed: float, args: argparse.Namespace) -> str:
    n = len(records)
    avail = [r for r in records if r.get("src_cfg_available")]
    views = {
        "baseline": [r["baseline_source_cfg"] for r in records],
        "ranker": [r["ranker_source_cfg"] for r in records],
        "oracle": [r["oracle_source_cfg"] for r in records],
    }
    lines: List[str] = []
    lines.append("===== MODULE B-v3 SOURCE-CFG-AWARE AUDIT =====")
    lines.append(f"source_jsonl = {Path(args.source_jsonl).expanduser().resolve()}")
    lines.append(f"result_jsonl = {Path(args.result_jsonl).expanduser().resolve()}")
    lines.append(f"n = {n}")
    lines.append(f"src_cfg_available = {pct(len(avail), n)}")
    lines.append(
        "metric_note = src CFG edges are projected through alignment_matrix; "
        "structured metrics exclude fallback_internal and goto_slot explanations."
    )
    lines.append("")
    for name, rows in views.items():
        lines.append(f"-- {name} --")
        lines.append(f"src_cfg_all_f1_mean = {mean(rows, 'src_cfg_all_f1'):.4f}")
        lines.append(f"src_cfg_all_recall_mean = {mean(rows, 'src_cfg_all_recall'):.4f}")
        lines.append(f"src_cfg_struct_f1_mean = {mean(rows, 'src_cfg_struct_f1'):.4f}")
        lines.append(f"src_cfg_struct_recall_mean = {mean(rows, 'src_cfg_struct_recall'):.4f}")
        lines.append(f"src_cfg_fallback_recall_mean = {mean(rows, 'src_cfg_fallback_recall'):.4f}")
        lines.append(f"src_cfg_goto_recall_mean = {mean(rows, 'src_cfg_goto_recall'):.4f}")
        lines.append(f"structured_src_stmt_coverage_mean = {mean(rows, 'structured_src_stmt_coverage'):.4f}")
        lines.append(f"fallback_src_stmt_coverage_mean = {mean(rows, 'fallback_src_stmt_coverage'):.4f}")
        lines.append(f"goto_src_stmt_coverage_mean = {mean(rows, 'goto_src_stmt_coverage'):.4f}")
        lines.append("")

    high_recall = float(args.high_src_recall)
    high_f1 = float(args.high_src_f1)
    ranker_ast_good = [
        r for r in records
        if ast_soft_good(r.get("ranker_metrics") or {})
    ]
    ranker_src_good = [
        r for r in records
        if source_cfg_good(
            r.get("ranker_source_cfg") or {},
            recall_threshold=high_recall,
            f1_threshold=high_f1,
        )
    ]
    ast_bad_src_good = [
        r for r in records
        if not ast_soft_good(r.get("ranker_metrics") or {})
        and source_cfg_good(
            r.get("ranker_source_cfg") or {},
            recall_threshold=high_recall,
            f1_threshold=high_f1,
        )
    ]
    ast_bad_src_bad = [
        r for r in records
        if r.get("src_cfg_available")
        and not ast_soft_good(r.get("ranker_metrics") or {})
        and not source_cfg_good(
            r.get("ranker_source_cfg") or {},
            recall_threshold=high_recall,
            f1_threshold=high_f1,
        )
    ]
    oracle_missing = [
        r for r in records
        if r.get("src_cfg_available")
        and not ast_soft_good(r.get("oracle_metrics") or {})
        and not source_cfg_good(
            r.get("oracle_source_cfg") or {},
            recall_threshold=high_recall,
            f1_threshold=high_f1,
        )
    ]
    ranker_miss = [
        r for r in records
        if (
            ast_soft_good(r.get("oracle_metrics") or {})
            or source_cfg_good(
                r.get("oracle_source_cfg") or {},
                recall_threshold=high_recall,
                f1_threshold=high_f1,
            )
        )
        and not (
            ast_soft_good(r.get("ranker_metrics") or {})
            or source_cfg_good(
                r.get("ranker_source_cfg") or {},
                recall_threshold=high_recall,
                f1_threshold=high_f1,
            )
        )
    ]
    lines.append("-- diagnosis buckets, ranker --")
    lines.append(f"ranker_ast_soft_good = {pct(len(ranker_ast_good), n)}")
    lines.append(f"ranker_source_cfg_good = {pct(len(ranker_src_good), n)}")
    lines.append(
        "ast_bad_but_source_cfg_good = "
        f"{pct(len(ast_bad_src_good), n)}  # CFG-semantic candidate; AST outline metric may be harsh"
    )
    lines.append(
        "ast_bad_and_source_cfg_bad = "
        f"{pct(len(ast_bad_src_bad), n)}  # genuinely weak recovered structure"
    )
    lines.append(
        "candidate_pool_missing_proxy = "
        f"{pct(len(oracle_missing), n)}  # even oracle lacks AST/source-CFG-good candidate"
    )
    lines.append(
        "ranker_selection_miss_proxy = "
        f"{pct(len(ranker_miss), n)}  # oracle has a usable candidate but ranker did not pick it"
    )
    lines.append("")

    lines.append("-- possible AST-metric false negatives, ranker --")
    interesting = sorted(
        ast_bad_src_good,
        key=lambda r: (
            -float((r.get("ranker_source_cfg") or {}).get("src_cfg_struct_recall") or 0.0),
            -float((r.get("ranker_source_cfg") or {}).get("src_cfg_struct_f1") or 0.0),
        ),
    )
    for r in interesting[: int(args.max_preview)]:
        m = r.get("ranker_metrics") or {}
        s = r.get("ranker_source_cfg") or {}
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"ranker={m.get('name')} outline_f1={m.get('outline_lcs_f1')} "
            f"src_struct_recall={s.get('src_cfg_struct_recall')} src_struct_f1={s.get('src_cfg_struct_f1')} "
            f"fallback_src_cov={s.get('fallback_src_stmt_coverage')} goto_src_cov={s.get('goto_src_stmt_coverage')}"
        )
    lines.append("")

    lines.append("-- candidate-pool missing proxy cases --")
    missing_sorted = sorted(
        oracle_missing,
        key=lambda r: (
            float((r.get("oracle_source_cfg") or {}).get("src_cfg_struct_f1") or 0.0),
            float((r.get("oracle_metrics") or {}).get("outline_lcs_f1") or 0.0),
        ),
    )
    for r in missing_sorted[: int(args.max_preview)]:
        m = r.get("oracle_metrics") or {}
        s = r.get("oracle_source_cfg") or {}
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"oracle={m.get('name')} oracle_outline_f1={m.get('outline_lcs_f1')} "
            f"oracle_src_struct_recall={s.get('src_cfg_struct_recall')} "
            f"oracle_src_struct_f1={s.get('src_cfg_struct_f1')} "
            f"oracle_fallback_src_cov={s.get('fallback_src_stmt_coverage')} "
            f"oracle_goto_src_cov={s.get('goto_src_stmt_coverage')}"
        )
    lines.append("")

    lines.append("-- ranker selection miss proxy cases --")
    miss_sorted = sorted(
        ranker_miss,
        key=lambda r: (
            -float((r.get("oracle_source_cfg") or {}).get("src_cfg_struct_f1") or 0.0),
            -float((r.get("oracle_metrics") or {}).get("outline_lcs_f1") or 0.0),
        ),
    )
    for r in miss_sorted[: int(args.max_preview)]:
        rm = r.get("ranker_metrics") or {}
        rs = r.get("ranker_source_cfg") or {}
        om = r.get("oracle_metrics") or {}
        os = r.get("oracle_source_cfg") or {}
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"ranker={rm.get('name')} r_outline_f1={rm.get('outline_lcs_f1')} "
            f"r_src_f1={rs.get('src_cfg_struct_f1')} "
            f"oracle={om.get('name')} o_outline_f1={om.get('outline_lcs_f1')} "
            f"o_src_f1={os.get('src_cfg_struct_f1')}"
        )
    lines.append("")

    lines.append("-- high fallback source-CFG cases, ranker --")
    bad = sorted(
        [r for r in records if r.get("src_cfg_available")],
        key=lambda r: (
            float((r.get("ranker_source_cfg") or {}).get("fallback_src_stmt_coverage") or 0.0),
            float((r.get("ranker_metrics") or {}).get("fallback_bb_ratio") or 0.0),
        ),
        reverse=True,
    )
    for r in bad[: int(args.max_preview)]:
        m = r.get("ranker_metrics") or {}
        s = r.get("ranker_source_cfg") or {}
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"ranker={m.get('name')} fallback_bb={m.get('fallback_bb_ratio')} "
            f"src_all_f1={s.get('src_cfg_all_f1')} src_struct_f1={s.get('src_cfg_struct_f1')} "
            f"fallback_src_cov={s.get('fallback_src_stmt_coverage')} "
            f"fallback_src_edge_recall={s.get('src_cfg_fallback_recall')}"
        )
    lines.append("")
    lines.append(f"elapsed_sec = {elapsed:.1f}")
    return "\n".join(lines) + "\n"


def audit(args: argparse.Namespace) -> None:
    result_jsonl = Path(args.result_jsonl).expanduser().resolve()
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    raw_records = [rec for _line, rec in iter_jsonl(result_jsonl)]
    if args.max_examples > 0:
        raw_records = raw_records[: int(args.max_examples)]
    source_rows = read_source_by_input_line(source_jsonl, [int(r.get("input_line") or 0) for r in raw_records])

    t0 = time.time()
    out_records: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for rec in raw_records:
            row = source_rows.get(int(rec.get("input_line") or 0))
            base_out = {
                "example": rec.get("example"),
                "input_line": rec.get("input_line"),
                "func_name": rec.get("func_name"),
                "n_bb": rec.get("n_bb"),
                "baseline_metrics": rec.get("baseline_metrics") or {},
                "ranker_metrics": rec.get("ranker_metrics") or {},
                "oracle_metrics": rec.get("oracle_metrics") or {},
            }
            if row is None:
                base_out["src_cfg_available"] = False
                base_out["skip_reason"] = "source_row_missing"
                for name in ("baseline", "ranker", "oracle"):
                    base_out[f"{name}_source_cfg"] = {"src_cfg_available": False}
                jf.write(json.dumps(base_out, ensure_ascii=False) + "\n")
                out_records.append(base_out)
                continue
            cfg = analyse_cfg(row)
            gold = source_edges(row)
            bb_to_src = bb_to_source_statements(
                row,
                n_bb=int(cfg.n_bb),
                topk=int(args.align_topk),
                min_frac=float(args.align_min_frac),
                min_weight=float(args.align_min_weight),
            )
            src_available = bool(gold and any(bb_to_src))
            try:
                cands = generate_candidates(
                    cfg,
                    max_region_nodes=int(args.max_region_nodes),
                    max_mixed_headers=int(args.max_mixed_if_headers),
                    candidate_beam=int(args.candidate_beam),
                )
            except Exception as exc:
                base_out["src_cfg_available"] = False
                base_out["skip_reason"] = f"candidate_regen_failed:{exc}"
                for name in ("baseline", "ranker", "oracle"):
                    base_out[f"{name}_source_cfg"] = {"src_cfg_available": False}
                jf.write(json.dumps(base_out, ensure_ascii=False) + "\n")
                out_records.append(base_out)
                continue

            base_out["src_cfg_available"] = bool(src_available)
            base_out["src_edge_count"] = int(len(gold))
            base_out["aligned_src_stmt_count"] = int(len(set().union(*bb_to_src)) if bb_to_src else 0)
            for name, metric_key in (
                ("baseline", "baseline_metrics"),
                ("ranker", "ranker_metrics"),
                ("oracle", "oracle_metrics"),
            ):
                cid = (rec.get(metric_key) or {}).get("candidate_id")
                cand = cands[int(cid)] if isinstance(cid, int) and 0 <= int(cid) < len(cands) else None
                if cand is None or not src_available:
                    base_out[f"{name}_source_cfg"] = {"src_cfg_available": False}
                else:
                    base_out[f"{name}_source_cfg"] = candidate_source_cfg_metrics(
                        row=row,
                        cfg=cfg,
                        cand=cand,
                        bb_to_src=bb_to_src,
                        gold_src_edges=gold,
                    )
            jf.write(json.dumps(base_out, ensure_ascii=False) + "\n")
            out_records.append(base_out)

    summary = build_summary(out_records, time.time() - t0, args)
    out_txt.write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)
    print(f"saved_txt = {out_txt}", flush=True)
    print(f"saved_jsonl = {out_jsonl}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Source-CFG-aware audit for Module-B v3 outputs")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--result-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=4)
    ap.add_argument("--candidate-beam", type=int, default=160)
    ap.add_argument("--align-topk", type=int, default=3)
    ap.add_argument("--align-min-frac", type=float, default=0.5)
    ap.add_argument("--align-min-weight", type=float, default=1e-8)
    ap.add_argument("--high-src-recall", type=float, default=0.80)
    ap.add_argument("--high-src-f1", type=float, default=0.50)
    ap.add_argument("--max-preview", type=int, default=30)
    audit(ap.parse_args())


if __name__ == "__main__":
    main()

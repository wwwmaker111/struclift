#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Module-B v3 candidate-ranker training data.

This script implements the supervision layer described in
``module_b_v3_practical_candidate_ranker_strategy.md``:

    binary CFG -> deterministic candidates -> validator -> teacher scoring
    -> L_rank labels

The candidate generator and validator are source-free.  Source/teacher skeletons
are used only offline to build a source-style preference label for the small
ranker.  If no valid candidate is close enough to the teacher, the label is
downgraded to the safe fallback candidate with a low teacher weight.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def teacher_outline(teacher: Dict[str, Any]) -> List[Tuple[str, int]]:
    pairs = teacher.get("teacher_outline_pairs")
    if isinstance(pairs, list):
        out: List[Tuple[str, int]] = []
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((str(item[0]), int(item[1])))
        return out
    controls = teacher.get("controls") if isinstance(teacher.get("controls"), list) else []
    return [
        (str(c.get("kind") or "unknown"), int(c.get("depth") or 0))
        for c in controls
        if str(c.get("kind") or "") in {"if", "loop", "switch"}
    ]


def teacher_slot_kindseq(teacher: Dict[str, Any]) -> List[str]:
    slots = teacher.get("slots") if isinstance(teacher.get("slots"), list) else []
    out: List[str] = []
    for slot in slots:
        if not bool(slot.get("trainable", True)):
            continue
        raw = str(slot.get("slot_type") or "")
        if raw == "FILL_COND":
            out.append("COND")
        elif raw == "FILL_BLOCK":
            out.append("BLOCK")
        elif raw == "FILL_BRANCH":
            out.append("BRANCH")
    return out


def pred_slot_kindseq(slots: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for slot in slots:
        raw = str(slot.get("slot_type") or "")
        if raw in {"COND", "BLOCK"}:
            out.append(raw)
        elif raw == "UNSTRUCTURED_REGION":
            out.append("BLOCK")
        elif raw in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
            out.append("BRANCH")
    return out


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


def count_similarity(pred_n: int, gold_n: int) -> float:
    if pred_n == gold_n == 0:
        return 1.0
    return max(0.0, 1.0 - abs(int(pred_n) - int(gold_n)) / max(1, int(gold_n)))


def decl_from_teacher(teacher: Dict[str, Any]) -> str:
    text = str(teacher.get("teacher_skeleton") or "")
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.endswith("{"):
        first = first[:-1].rstrip()
    if first and "(" in first and ")" in first:
        return first
    return "int func(...)"


def candidate_family(name: str) -> str:
    if name == "full_fallback":
        return "fallback"
    if name == "linear_goto":
        return "linear_goto"
    if "sese_interval" in name or "interval" in name:
        return "sese_interval"
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


def cfg_features(cfg: Any) -> Dict[str, Any]:
    out_degrees = [len(cfg.succs.get(i, [])) for i in range(cfg.n_bb)]
    in_degrees = [len(cfg.preds.get(i, [])) for i in range(cfg.n_bb)]
    return {
        "n_bb": int(cfg.n_bb),
        "n_edges": int(len(cfg.edges)),
        "n_back_edges": int(len(cfg.back_edges)),
        "n_loop_headers": int(len(cfg.loop_nodes_by_header)),
        "n_unreachable": int(len(cfg.unreachable)),
        "max_out_degree": int(max(out_degrees) if out_degrees else 0),
        "max_in_degree": int(max(in_degrees) if in_degrees else 0),
        "branch_header_count": int(sum(1 for d in out_degrees if d >= 2)),
        "switch_like_header_count": int(sum(1 for d in out_degrees if d >= 3)),
    }


def max_outline_depth(outline: Sequence[Tuple[str, int]]) -> int:
    return int(max((int(d) for _k, d in outline), default=0))


def candidate_features(cand_rec: Dict[str, Any]) -> Dict[str, Any]:
    vr = cand_rec.get("validator_report") or {}
    slots = cand_rec.get("slots") or []
    outline = [(str(k), int(d)) for k, d in cand_rec.get("outline") or []]
    slot_types = Counter(str(s.get("slot_type") or "") for s in slots)
    kinds = Counter(k for k, _d in outline)
    return {
        "candidate_id": int(cand_rec.get("candidate_id") or 0),
        "name": str(cand_rec.get("name") or ""),
        "family": candidate_family(str(cand_rec.get("name") or "")),
        "validator_valid": bool(vr.get("valid", False)),
        "bb_coverage_ok": bool(vr.get("bb_coverage_ok", False)),
        "edge_preservation_ok": bool(vr.get("edge_preservation_ok", False)),
        "slot_grounding_ok": bool(vr.get("slot_grounding_ok", False)),
        "parse_success": bool(vr.get("parse_success", False)),
        "validator_score": float(vr.get("score") or 0.0),
        "fallback_region_count": int(vr.get("fallback_region_count") or 0),
        "fallback_bb_count": int(vr.get("fallback_bb_count") or 0),
        "goto_count": int(vr.get("goto_count") or 0),
        "cleanup_exit_count": int(vr.get("cleanup_exit_count") or 0),
        "escape_count": int(vr.get("escape_count") or 0),
        "structured_control_count": int(vr.get("structured_control_count") or 0),
        "structured_bb_count": int(vr.get("structured_bb_count") or 0),
        "slot_count": int(vr.get("slot_count") or len(slots)),
        "cond_slot_count": int(slot_types.get("COND", 0)),
        "block_slot_count": int(slot_types.get("BLOCK", 0)),
        "unstructured_slot_count": int(slot_types.get("UNSTRUCTURED_REGION", 0)),
        "goto_slot_count": int(slot_types.get("GOTO", 0)),
        "break_slot_count": int(slot_types.get("BREAK", 0)),
        "continue_slot_count": int(slot_types.get("CONTINUE", 0)),
        "early_exit_slot_count": int(slot_types.get("EARLY_EXIT", 0)),
        "cleanup_exit_slot_count": int(slot_types.get("CLEANUP_EXIT", 0)),
        "outline_count": int(len(outline)),
        "outline_max_depth": max_outline_depth(outline),
        "outline_if_count": int(kinds.get("if", 0)),
        "outline_loop_count": int(kinds.get("loop", 0)),
        "outline_switch_count": int(kinds.get("switch", 0)),
    }


def source_style_score(
    cand_rec: Dict[str, Any],
    *,
    t_outline: Sequence[Tuple[str, int]],
    t_slot_kind: Sequence[str],
) -> Dict[str, Any]:
    vr = cand_rec.get("validator_report") or {}
    slots = cand_rec.get("slots") or []
    p_outline = [(str(k), int(d)) for k, d in cand_rec.get("outline") or []]
    p_norm = normalize_depths(p_outline)
    t_norm = normalize_depths(t_outline)
    p_kind = kindseq(p_outline)
    t_kind = kindseq(t_outline)
    p_slot_kind = pred_slot_kindseq(slots)

    outline_f1 = seq_f1(p_norm, t_norm)
    kind_f1 = seq_f1(p_kind, t_kind)
    slot_kind_f1 = seq_f1(p_slot_kind, t_slot_kind)
    control_count_score = count_similarity(len(p_outline), len(t_outline))
    valid = bool(vr.get("valid", False))
    fallback_bb = int(vr.get("fallback_bb_count") or 0)
    goto_count = int(vr.get("goto_count") or 0)
    slot_count = int(vr.get("slot_count") or len(slots))
    safety = 1.0 if valid else 0.0
    compact_penalty = min(0.25, 0.01 * fallback_bb + 0.008 * goto_count + 0.002 * slot_count)
    quality = (
        0.42 * outline_f1
        + 0.22 * kind_f1
        + 0.14 * control_count_score
        + 0.12 * slot_kind_f1
        + 0.10 * safety
        - compact_penalty
    )
    if not valid:
        quality -= 1.0
    outline_equal = bool(t_outline) and p_norm == t_norm
    kindseq_equal = bool(t_outline) and p_kind == t_kind
    return {
        "quality": round(float(quality), 6),
        "outline_lcs_f1": round(float(outline_f1), 6),
        "kind_lcs_f1": round(float(kind_f1), 6),
        "slot_kind_lcs_f1": round(float(slot_kind_f1), 6),
        "control_count_score": round(float(control_count_score), 6),
        "outline_equal": bool(outline_equal),
        "kindseq_equal": bool(kindseq_equal),
        "outline_count_gap": int(len(p_outline) - len(t_outline)),
        "slot_kind_count_gap": int(len(p_slot_kind) - len(t_slot_kind)),
    }


def soft_targets(candidates: Sequence[Dict[str, Any]], *, tau: float) -> List[Dict[str, Any]]:
    valid = [c for c in candidates if bool((c.get("features") or {}).get("validator_valid"))]
    pool = valid or list(candidates)
    if not pool:
        return []
    scores = [float((c.get("source_style") or {}).get("quality") or -1e9) / max(1e-6, tau) for c in pool]
    mx = max(scores)
    exps = [math.exp(min(60.0, s - mx)) for s in scores]
    denom = sum(exps) or 1.0
    probs_by_id = {
        int(c.get("candidate_id") or 0): float(e / denom)
        for c, e in zip(pool, exps)
    }
    return [
        {
            "candidate_id": int(c.get("candidate_id") or 0),
            "prob": round(float(probs_by_id.get(int(c.get("candidate_id") or 0), 0.0)), 8),
        }
        for c in candidates
    ]


def choose_label(
    candidates: Sequence[Dict[str, Any]],
    *,
    quality_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], str, float]:
    valid = [c for c in candidates if bool((c.get("features") or {}).get("validator_valid"))]
    pool = valid or list(candidates)
    if not pool:
        return None, "no_candidate", 0.0
    oracle = max(
        pool,
        key=lambda c: (
            float((c.get("source_style") or {}).get("quality") or -1e9),
            -int(c.get("candidate_id") or 0),
        ),
    )
    oracle_quality = float((oracle.get("source_style") or {}).get("quality") or 0.0)
    if oracle_quality >= float(quality_threshold):
        return oracle, "teacher_preference", 1.0

    fallback = next((c for c in candidates if str(c.get("name")) == "full_fallback"), None)
    if fallback is not None and bool((fallback.get("features") or {}).get("validator_valid", True)):
        return fallback, "safe_fallback_low_confidence", 0.35
    return oracle, "low_confidence_best_available", 0.5


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def _dist(values: Iterable[Any]) -> List[Tuple[Any, int]]:
    return sorted(Counter(values).items(), key=lambda kv: (kv[0], kv[1]))


def export(args: argparse.Namespace) -> None:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    teacher_jsonl = Path(args.teacher_jsonl).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    teachers = [t for _ln, t in iter_jsonl(teacher_jsonl, int(args.max_examples))]
    source_by_line = read_source_by_line(source_jsonl, [int(t.get("input_line") or 0) for t in teachers])

    t0 = time.time()
    records: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for example, teacher in enumerate(teachers):
            line_no = int(teacher.get("input_line") or 0)
            row = source_by_line.get(line_no)
            if row is None:
                continue
            cfg = analyse_cfg(row)
            t_outline = teacher_outline(teacher)
            t_slot_kind = teacher_slot_kindseq(teacher)
            candidate_objs = generate_candidates(
                cfg,
                max_region_nodes=int(args.max_region_nodes),
                max_mixed_headers=int(args.max_mixed_if_headers),
                candidate_beam=int(args.candidate_beam),
            )
            selected_cand, selected_report, _selected_text = select_candidate(cfg, candidate_objs)
            candidates: List[Dict[str, Any]] = []
            for cid, cand in enumerate(candidate_objs):
                text = render_candidate(cand, decl_line=decl_from_teacher(teacher))
                report = validate_candidate(cfg, cand, rendered_text=text)
                slots = [s.asdict() for s in cand.slots]
                cand_rec: Dict[str, Any] = {
                    "candidate_id": cid,
                    "name": cand.name,
                    "outline": [[k, d] for k, d in candidate_outline(cand)],
                    "validator_report": report.asdict(),
                    "slots": slots,
                }
                cand_rec["features"] = candidate_features(cand_rec)
                cand_rec["source_style"] = source_style_score(
                    cand_rec,
                    t_outline=t_outline,
                    t_slot_kind=t_slot_kind,
                )
                if bool(args.include_skeletons):
                    cand_rec["skeleton"] = text
                elif bool(args.keep_skeleton_for_label):
                    cand_rec["skeleton_preview"] = "\n".join(text.splitlines()[:12])
                candidates.append(cand_rec)

            hard, label_mode, teacher_weight = choose_label(
                candidates,
                quality_threshold=float(args.quality_threshold),
            )
            distribution = soft_targets(candidates, tau=float(args.softmax_tau))
            deterministic_id = None
            for cand_rec in candidates:
                if cand_rec["name"] == selected_cand.name:
                    deterministic_id = int(cand_rec["candidate_id"])
                    break

            rec = {
                "example": example,
                "input_line": line_no,
                "func_name": row.get("func_name") or teacher.get("func_name"),
                "n_bb": int(cfg.n_bb),
                "cfg_features": cfg_features(cfg),
                "teacher": {
                    "outline": [[k, d] for k, d in t_outline],
                    "outline_kindseq": kindseq(t_outline),
                    "slot_kindseq": list(t_slot_kind),
                },
                "candidate_count": int(len(candidates)),
                "deterministic_selected_candidate_id": deterministic_id,
                "deterministic_selected_name": selected_cand.name,
                "deterministic_validator_report": selected_report.asdict(),
                "rank_supervision": {
                    "loss": "L_rank",
                    "hard_target_candidate_id": hard.get("candidate_id") if hard else None,
                    "hard_target_name": hard.get("name") if hard else None,
                    "label_mode": label_mode,
                    "teacher_weight": round(float(teacher_weight), 4),
                    "quality_threshold": float(args.quality_threshold),
                    "softmax_tau": float(args.softmax_tau),
                    "soft_targets": distribution,
                    "oracle_quality": (
                        float((hard.get("source_style") or {}).get("quality") or 0.0)
                        if hard else 0.0
                    ),
                    "oracle_outline_lcs_f1": (
                        float((hard.get("source_style") or {}).get("outline_lcs_f1") or 0.0)
                        if hard else 0.0
                    ),
                },
                "candidates": candidates,
            }
            records.append(rec)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    candidate_count_dist = Counter(int(r.get("candidate_count") or 0) for r in records)
    label_mode_dist = Counter((r.get("rank_supervision") or {}).get("label_mode") for r in records)
    hard_name_dist = Counter((r.get("rank_supervision") or {}).get("hard_target_name") for r in records)
    deterministic_name_dist = Counter(r.get("deterministic_selected_name") for r in records)
    high_conf = sum(1 for r in records if float((r.get("rank_supervision") or {}).get("teacher_weight") or 0.0) >= 0.999)
    deterministic_valid = sum(
        1 for r in records if bool((r.get("deterministic_validator_report") or {}).get("valid"))
    )
    hard_quality = [float((r.get("rank_supervision") or {}).get("oracle_quality") or 0.0) for r in records]
    hard_outline_f1 = [float((r.get("rank_supervision") or {}).get("oracle_outline_lcs_f1") or 0.0) for r in records]
    hard_fallback = []
    hard_goto = []
    hard_outline_equal = 0
    for r in records:
        target_id = (r.get("rank_supervision") or {}).get("hard_target_candidate_id")
        target = next((c for c in r.get("candidates") or [] if c.get("candidate_id") == target_id), None)
        if target is None:
            continue
        f = target.get("features") or {}
        s = target.get("source_style") or {}
        hard_fallback.append(int(f.get("fallback_region_count") or 0))
        hard_goto.append(int(f.get("goto_count") or 0))
        if s.get("outline_equal"):
            hard_outline_equal += 1

    with out_txt.open("w", encoding="utf-8") as wf:
        wf.write("===== MODULE B-v3 RANKER DATASET =====\n")
        wf.write(f"source_jsonl = {source_jsonl}\n")
        wf.write(f"teacher_jsonl = {teacher_jsonl}\n")
        wf.write(f"out_jsonl = {out_jsonl}\n")
        wf.write(f"n = {n}\n")
        wf.write(f"quality_threshold = {float(args.quality_threshold):.4f}\n")
        wf.write(f"softmax_tau = {float(args.softmax_tau):.4f}\n")
        wf.write("\n-- MVP safety, deterministic selected candidate --\n")
        wf.write(f"deterministic_selected_valid = {_pct(deterministic_valid, n)}\n")
        wf.write(f"deterministic_selected_name_dist = {deterministic_name_dist.most_common()}\n")
        wf.write("\n-- candidate pool --\n")
        wf.write(f"candidate_count_dist = {sorted(candidate_count_dist.items())}\n")
        wf.write("\n-- L_rank supervision --\n")
        wf.write(f"high_conf_teacher_labels = {_pct(high_conf, n)}\n")
        wf.write(f"label_mode_dist = {label_mode_dist.most_common()}\n")
        wf.write(f"hard_target_name_dist = {hard_name_dist.most_common()}\n")
        wf.write(f"hard_target_outline_equal = {_pct(hard_outline_equal, n)}\n")
        wf.write(f"hard_target_quality_mean = {mean(hard_quality) if hard_quality else 0.0:.4f}\n")
        wf.write(f"hard_target_quality_median = {median(hard_quality) if hard_quality else 0.0:.4f}\n")
        wf.write(f"hard_target_outline_lcs_f1_mean = {mean(hard_outline_f1) if hard_outline_f1 else 0.0:.4f}\n")
        wf.write(f"hard_target_outline_lcs_f1_median = {median(hard_outline_f1) if hard_outline_f1 else 0.0:.4f}\n")
        wf.write(f"hard_target_fallback_dist = {_dist(hard_fallback)}\n")
        wf.write(f"hard_target_goto_dist = {_dist(hard_goto)}\n")
        wf.write(f"elapsed_sec = {time.time() - t0:.1f}\n")
        wf.write("\n-- per-example preview --\n")
        for r in records[: int(args.preview_examples)]:
            sup = r.get("rank_supervision") or {}
            wf.write("-" * 120 + "\n")
            wf.write(
                f"example={r.get('example')} input_line={r.get('input_line')} "
                f"func={r.get('func_name')} n_bb={r.get('n_bb')} "
                f"candidates={r.get('candidate_count')} deterministic={r.get('deterministic_selected_name')} "
                f"target={sup.get('hard_target_name')} mode={sup.get('label_mode')} "
                f"w={sup.get('teacher_weight')} q={sup.get('oracle_quality'):.4f} "
                f"outline_f1={sup.get('oracle_outline_lcs_f1'):.4f}\n"
            )
            wf.write(f"teacher_outline={((r.get('teacher') or {}).get('outline'))}\n")
            top = sorted(
                r.get("candidates") or [],
                key=lambda c: float((c.get("source_style") or {}).get("quality") or -1e9),
                reverse=True,
            )[:8]
            for c in top:
                f = c.get("features") or {}
                s = c.get("source_style") or {}
                wf.write(
                    f"  cand={c.get('candidate_id')} name={c.get('name')} "
                    f"valid={f.get('validator_valid')} q={s.get('quality')} "
                    f"outline_f1={s.get('outline_lcs_f1')} kind_f1={s.get('kind_lcs_f1')} "
                    f"fallback={f.get('fallback_region_count')} goto={f.get('goto_count')} "
                    f"outline={c.get('outline')}\n"
                )

    print(f"n = {n}", flush=True)
    print(f"deterministic_selected_valid = {_pct(deterministic_valid, n)}", flush=True)
    print(f"candidate_count_dist = {sorted(candidate_count_dist.items())}", flush=True)
    print(f"high_conf_teacher_labels = {_pct(high_conf, n)}", flush=True)
    print(f"label_mode_dist = {label_mode_dist.most_common()}", flush=True)
    print(f"hard_target_name_dist = {hard_name_dist.most_common()}", flush=True)
    print(f"hard_target_outline_equal = {_pct(hard_outline_equal, n)}", flush=True)
    print(f"hard_target_outline_lcs_f1_mean = {mean(hard_outline_f1) if hard_outline_f1 else 0.0:.4f}", flush=True)
    print(f"saved_txt = {out_txt}", flush=True)
    print(f"saved_jsonl = {out_jsonl}", flush=True)
    print(f"elapsed_sec = {time.time() - t0:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Module-B v3 L_rank candidate dataset")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=4)
    ap.add_argument("--candidate-beam", type=int, default=160)
    ap.add_argument("--quality-threshold", type=float, default=0.62)
    ap.add_argument("--softmax-tau", type=float, default=0.15)
    ap.add_argument("--preview-examples", type=int, default=30)
    ap.add_argument("--include-skeletons", action="store_true")
    ap.add_argument(
        "--keep-skeleton-for-label",
        action="store_true",
        help="Store a short skeleton preview even when full skeletons are disabled.",
    )
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()

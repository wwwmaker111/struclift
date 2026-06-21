#!/usr/bin/env python3
"""Train/evaluate a structural source-like ranker for Module B-v3.

This script is intentionally standalone: it reads an exported Module-B v3
full-pipeline JSONL with full candidate summaries and trains a small pairwise
linear ranker whose supervision is source-structure similarity only.

The ranker features are binary/candidate-only.  Source/teacher-derived fields
such as outline_lcs_f1 and kind_lcs_f1 are used only as labels/evaluation
targets and never as inference features.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


FAMILIES = [
    "recursive",
    "shallow",
    "sese_interval",
    "then_only",
    "else_only",
    "mixed_if_policy",
    "depth_limited",
    "sparse_control",
    "wide_region",
    "switch_fanin",
    "switch_chain",
    "linear_goto",
    "fallback",
    "other",
]


def safe_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        value = float(x)
        return value if math.isfinite(value) else float(default)
    try:
        value = float(x)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def finite_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        value = float(x)
        return value if math.isfinite(value) else None
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def pct(n: int, d: int) -> str:
    return f"{int(n)}/{int(d)} ({100.0 * int(n) / max(1, int(d)):.1f}%)"


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def med(xs: Sequence[float]) -> float:
    return float(median(xs)) if xs else 0.0


def family_name(name: Any) -> str:
    s = str(name or "")
    if s == "full_fallback":
        return "fallback"
    if "linear_goto" in s:
        return "linear_goto"
    if "sese_interval" in s or "interval" in s:
        return "sese_interval"
    if "switch_fanin" in s:
        return "switch_fanin"
    if "switch_chain" in s:
        return "switch_chain"
    if "mixed" in s:
        return "mixed_if_policy"
    if "depth" in s:
        return "depth_limited"
    if "skip_tiny" in s or "skip_small" in s:
        return "sparse_control"
    if "wide" in s:
        return "wide_region"
    if "then_only" in s:
        return "then_only"
    if "else_only" in s:
        return "else_only"
    if "recursive" in s:
        return "recursive"
    if "shallow" in s:
        return "shallow"
    return "other"


def candidate_ast_proxy_score(cand: Dict[str, Any]) -> float:
    """Source-derived AST-outline proxy used only as a label/eval fallback."""
    outline = safe_float(candidate_field(cand, "outline_lcs_f1"))
    kind = safe_float(candidate_field(cand, "kind_lcs_f1"))
    fallback_penalty = 0.08 * min(1.0, safe_int(candidate_field(cand, "fallback_bb_count")) / 3.0)
    goto_penalty = 0.04 * min(1.0, safe_int(candidate_field(cand, "goto_count")) / 3.0)
    escape_penalty = 0.03 * min(1.0, safe_int(candidate_field(cand, "escape_count")) / 5.0)
    exact_bonus = 0.08 * float(bool(candidate_field(cand, "outline_equal"))) + 0.05 * float(bool(candidate_field(cand, "kindseq_equal")))
    score = 0.55 * outline + 0.37 * kind + exact_bonus - fallback_penalty - goto_penalty - escape_penalty
    return max(0.0, min(1.0, float(score)))


def candidate_export_preference_score(cand: Dict[str, Any]) -> Optional[float]:
    """Approximate the full-pipeline teacher preference from exported fields.

    The JSONL candidate summary intentionally stores source/teacher-derived
    values only as labels.  The ranker features below never consume these fields
    at inference time, but the L_rank supervision should follow the exported
    rank target instead of inventing a second oracle.
    """
    if not bool(candidate_field(cand, "hard_valid")):
        return 0.0
    rank_quality = finite_float(candidate_field(cand, "rank_quality"))
    tier = safe_int(candidate_field(cand, "rank_label_tier"), 0)
    has_export_label = rank_quality is not None or tier > 0
    if not has_export_label:
        return None

    tier_score = {
        6: 0.96,
        5: 0.88,
        4: 0.78,
        3: 0.58,
        2: 0.42,
        1: 0.28,
        0: 0.0,
    }.get(max(0, min(6, int(tier))), 0.0)
    ast = candidate_ast_proxy_score(cand)
    source_cfg = (
        0.50 * safe_float(candidate_field(cand, "source_cfg_score"))
        + 0.30 * safe_float(candidate_field(cand, "source_cfg_struct_recall"))
        + 0.20 * safe_float(candidate_field(cand, "source_cfg_struct_f1"))
    )
    bonus = (
        0.04 * float(bool(candidate_field(cand, "rank_trainable")))
        + 0.025 * float(bool(candidate_field(cand, "source_soft_recovered")))
        + 0.015 * float(bool(candidate_field(cand, "outline_equal")))
        + 0.010 * float(bool(candidate_field(cand, "kindseq_equal")))
    )
    locality_penalty = (
        0.10 * max(0.0, 1.0 - safe_float(candidate_field(cand, "slot_locality_score"), 1.0))
        + 0.08 * min(1.0, safe_float(candidate_field(cand, "empty_control_with_context_count")) / 2.0)
        + 0.06 * min(1.0, safe_float(candidate_field(cand, "body_bb_outside_control_count")) / 4.0)
    )
    residual = 0.0
    if rank_quality is not None:
        # rank_quality is tier-major (roughly 10000 * tier + preference).
        # Keep only a tiny bounded within-tier tie-break so the exported tier
        # remains the main supervision signal.
        residual = max(-0.04, min(0.04, (float(rank_quality) - 10000.0 * float(tier)) / 800.0))
    score = 0.64 * tier_score + 0.24 * ast + 0.12 * source_cfg + bonus + residual - locality_penalty
    return max(0.0, min(1.0, float(score)))


def candidate_structural_score(cand: Dict[str, Any]) -> float:
    """Teacher-derived structural label used for supervision/eval only."""
    exported = candidate_export_preference_score(cand)
    if exported is not None:
        return exported
    return candidate_ast_proxy_score(cand)


def candidate_field(cand: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in cand:
        return cand.get(key)
    target = cand.get("rank_target") if isinstance(cand.get("rank_target"), dict) else {}
    if key in target:
        return target.get(key)
    style = cand.get("source_style") if isinstance(cand.get("source_style"), dict) else {}
    if key in style:
        return style.get(key)
    report = cand.get("validator_report") if isinstance(cand.get("validator_report"), dict) else {}
    if key in report:
        return report.get(key)
    features = cand.get("features") if isinstance(cand.get("features"), dict) else {}
    if key in features:
        return features.get(key)
    return default


def candidate_pool(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    cg = row.get("candidate_generation") or {}
    full = cg.get("candidates")
    if isinstance(full, list) and full:
        return full
    summary = cg.get("candidate_summary")
    if isinstance(summary, list):
        return summary
    return []


def is_structural_source_like(cand: Dict[str, Any], threshold: float = 0.72) -> bool:
    if not bool(candidate_field(cand, "hard_valid")):
        return False
    if bool(candidate_field(cand, "structural_source_like")):
        return True
    if bool(candidate_field(cand, "rank_trainable")) or safe_int(candidate_field(cand, "rank_label_tier"), 0) >= 5:
        return True
    if bool(candidate_field(cand, "source_soft_recovered")) and safe_int(candidate_field(cand, "rank_label_tier"), 0) >= 4:
        return True
    outline = safe_float(candidate_field(cand, "outline_lcs_f1"))
    kind = safe_float(candidate_field(cand, "kind_lcs_f1"))
    if candidate_field(cand, "outline_equal") or candidate_field(cand, "kindseq_equal"):
        return True
    return bool(candidate_structural_score(cand) >= threshold and (outline >= 0.65 or kind >= 0.78))


def selected_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    selected_id = (row.get("ranker") or {}).get("selected_candidate_id")
    for cand in candidate_pool(row):
        if cand.get("candidate_id") == selected_id:
            return cand
    return {}


def outline_stats(cand: Dict[str, Any]) -> Dict[str, float]:
    outline = cand.get("outline") or []
    kinds: List[str] = []
    depths: List[int] = []
    if isinstance(outline, list):
        for item in outline:
            if isinstance(item, (list, tuple)) and item:
                kinds.append(str(item[0]))
                if len(item) >= 2:
                    depths.append(safe_int(item[1]))
    n = max(1, len(kinds))
    return {
        "outline_count": float(len(kinds)),
        "outline_if_count": float(sum(1 for k in kinds if k == "if")),
        "outline_loop_count": float(sum(1 for k in kinds if k == "loop")),
        "outline_switch_count": float(sum(1 for k in kinds if k == "switch")),
        "outline_max_depth": float(max(depths) if depths else 0),
        "outline_avg_depth": float(sum(depths) / n if depths else 0.0),
    }


def candidate_features(row: Dict[str, Any], cand: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """Binary/candidate-only features. Do not add source/teacher label fields."""
    cg = row.get("candidate_generation") or {}
    analysis = row.get("analysis_context") or {}
    function_ir = row.get("function_ir") or {}
    cfg = analysis.get("cfg") if isinstance(analysis.get("cfg"), dict) else {}
    cfeat = cand.get("features") if isinstance(cand.get("features"), dict) else {}
    fam = family_name(cand.get("name"))
    outline = outline_stats(cand)

    candidate_count = max(1.0, safe_float(cg.get("candidate_count"), 1.0))
    n_bb = max(
        1.0,
        safe_float(cfg.get("n_bb"), safe_float(function_ir.get("n_basic_blocks"), 0.0)),
    )
    fallback_bb = safe_float(candidate_field(cand, "fallback_bb_count"))
    fallback_region = safe_float(candidate_field(cand, "fallback_region_count"))
    goto_count = safe_float(candidate_field(cand, "goto_count"))
    break_count = safe_float(candidate_field(cand, "break_count"))
    continue_count = safe_float(candidate_field(cand, "continue_count"))
    early_exit_count = safe_float(candidate_field(cand, "early_exit_count"))
    cleanup_exit_count = safe_float(candidate_field(cand, "cleanup_exit_count"))
    escape_count = safe_float(
        candidate_field(cand, "escape_count"),
        goto_count + break_count + continue_count + early_exit_count + cleanup_exit_count,
    )
    slot_count = max(0.0, safe_float(candidate_field(cand, "slot_count")))
    structured_control = max(0.0, safe_float(candidate_field(cand, "structured_control_count")))
    slot_locality = safe_float(candidate_field(cand, "slot_locality_score"), 1.0)
    empty_control = safe_float(candidate_field(cand, "empty_control_count"))
    empty_context = safe_float(candidate_field(cand, "empty_control_with_context_count"))
    outside_body = safe_float(candidate_field(cand, "body_bb_outside_control_count"))
    control_body_total = max(1.0, safe_float(candidate_field(cand, "control_body_total_bb_count"), 1.0))

    values: Dict[str, float] = {
        "bias": 1.0,
        "log_candidate_count": math.log1p(candidate_count),
        "log_n_bb": math.log1p(n_bb),
        "candidate_id_norm": safe_float(cand.get("candidate_id")) / candidate_count,
        "valid": float(bool(candidate_field(cand, "valid"))),
        "hard_valid": float(bool(candidate_field(cand, "hard_valid"))),
        "doc_quality_ready": float(bool(candidate_field(cand, "doc_quality_ready"))),
        "cost_feasible": float(bool(candidate_field(cand, "cost_feasible"))),
        "fallback_bb_ratio": fallback_bb / n_bb,
        "fallback_region_count": fallback_region,
        "goto_count": goto_count,
        "break_count": break_count,
        "continue_count": continue_count,
        "early_exit_count": early_exit_count,
        "cleanup_exit_count": cleanup_exit_count,
        "escape_count": escape_count,
        "slot_count_ratio": slot_count / n_bb,
        "slot_locality_score": slot_locality,
        "empty_control_ratio": empty_control / n_bb,
        "empty_control_with_context_ratio": empty_context / n_bb,
        "body_bb_outside_control_ratio": outside_body / control_body_total,
        "structured_control_ratio": structured_control / n_bb,
        "structured_bb_ratio": safe_float(candidate_field(cand, "structured_bb_ratio"), safe_float(cfeat.get("structured_bb_ratio"))),
        "edge_semantic_ratio": safe_float(candidate_field(cand, "edge_semantic_ratio"), safe_float(cfeat.get("edge_semantic_ratio"))),
        "edge_interval_ratio": safe_float(candidate_field(cand, "edge_interval_ratio"), safe_float(cfeat.get("edge_interval_ratio"))),
        "ranker_score_old": safe_float(cand.get("ranker_score")),
        "outline_count_ratio": outline["outline_count"] / n_bb,
        "outline_if_ratio": outline["outline_if_count"] / n_bb,
        "outline_loop_ratio": outline["outline_loop_count"] / n_bb,
        "outline_switch_ratio": outline["outline_switch_count"] / n_bb,
        "outline_max_depth": outline["outline_max_depth"],
        "outline_avg_depth": outline["outline_avg_depth"],
        "name_has_guard": float("guard" in str(cand.get("name") or "")),
        "name_has_mixed": float("mixed" in str(cand.get("name") or "")),
        "name_has_switch": float("switch" in str(cand.get("name") or "")),
        "name_has_then": float("then" in str(cand.get("name") or "")),
        "name_has_else": float("else" in str(cand.get("name") or "")),
    }
    for fam_name in FAMILIES:
        active = 1.0 if fam == fam_name else 0.0
        values[f"family_{fam_name}"] = active
        if active:
            for key in (
                "log_candidate_count",
                "log_n_bb",
                "candidate_id_norm",
                "fallback_bb_ratio",
                "goto_count",
                "escape_count",
                "slot_count_ratio",
                "slot_locality_score",
                "empty_control_ratio",
                "empty_control_with_context_ratio",
                "body_bb_outside_control_ratio",
                "structured_control_ratio",
                "structured_bb_ratio",
                "edge_semantic_ratio",
                "edge_interval_ratio",
                "outline_count_ratio",
                "outline_if_ratio",
                "outline_loop_ratio",
                "outline_switch_ratio",
                "outline_max_depth",
            ):
                values[f"family_{fam_name}_x_{key}"] = values[key]
        else:
            for key in (
                "log_candidate_count",
                "log_n_bb",
                "candidate_id_norm",
                "fallback_bb_ratio",
                "goto_count",
                "escape_count",
                "slot_count_ratio",
                "slot_locality_score",
                "empty_control_ratio",
                "empty_control_with_context_ratio",
                "body_bb_outside_control_ratio",
                "structured_control_ratio",
                "structured_bb_ratio",
                "edge_semantic_ratio",
                "edge_interval_ratio",
                "outline_count_ratio",
                "outline_if_ratio",
                "outline_loop_ratio",
                "outline_switch_ratio",
                "outline_max_depth",
            ):
                values[f"family_{fam_name}_x_{key}"] = 0.0
    names = sorted(values)
    return names, [float(values[k]) for k in names]


def standard_stats(vectors: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    if not vectors:
        return [], []
    dim = len(vectors[0])
    mu: List[float] = []
    scale: List[float] = []
    for j in range(dim):
        col = [float(v[j]) for v in vectors]
        m = sum(col) / max(1, len(col))
        var = sum((x - m) ** 2 for x in col) / max(1, len(col))
        sd = math.sqrt(var)
        mu.append(m)
        scale.append(sd if sd > 1e-8 else 1.0)
    return mu, scale


def normalize(vec: Sequence[float], mean_vec: Sequence[float], scale_vec: Sequence[float]) -> List[float]:
    return [
        (float(x) - float(mean_vec[i])) / (float(scale_vec[i]) if float(scale_vec[i]) else 1.0)
        for i, x in enumerate(vec)
    ]


def dot(w: Sequence[float], v: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(w, v)))


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows: Sequence[Dict[str, Any]], dev_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_dev = max(1, int(round(len(idx) * float(dev_ratio)))) if len(idx) > 1 else 0
    dev = sorted(idx[:n_dev])
    train = sorted(idx[n_dev:])
    if not train and dev:
        train, dev = dev, []
    return train, dev


def build_pairs(
    rows: Sequence[Dict[str, Any]],
    row_indices: Sequence[int],
    *,
    max_pairs_per_function: int,
    positive_top_k: int,
    hard_negative_k: int,
    max_negatives_per_family: int,
    min_margin: float,
    seed: int,
) -> Tuple[List[Tuple[int, int, int]], Counter]:
    rng = random.Random(seed)
    pairs: List[Tuple[int, int, int]] = []
    stats: Counter = Counter()
    for ridx in row_indices:
        row = rows[ridx]
        cands = [
            c
            for c in candidate_pool(row)
            if bool(candidate_field(c, "hard_valid"))
        ]
        if len(cands) < 2:
            stats["skip_lt2"] += 1
            continue
        scored = [(safe_int(c.get("candidate_id")), candidate_structural_score(c), c) for c in cands]
        scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        best_score = float(scored[0][1])
        positives = [
            item
            for item in scored
            if is_structural_source_like(item[2])
            and best_score - float(item[1]) <= max(0.04, float(min_margin) * 2.0)
        ][: max(1, int(positive_top_k))]
        if not positives:
            stats["skip_no_structural_positive"] += 1
            continue
        function_pairs: List[Tuple[int, int, int]] = []
        seen_pairs: Set[Tuple[int, int]] = set()

        def add_pair(pos_id: int, neg_id: int) -> None:
            key = (int(pos_id), int(neg_id))
            if key in seen_pairs or int(pos_id) == int(neg_id):
                return
            seen_pairs.add(key)
            function_pairs.append((ridx, int(pos_id), int(neg_id)))

        for pos_id, pos_score, _pos in positives:
            negs = [
                item
                for item in scored
                if int(item[0]) != int(pos_id)
                and float(pos_score) - float(item[1]) >= float(min_margin)
            ]
            if not negs:
                continue
            for neg_id, _score_value, _cand in negs[: max(0, int(hard_negative_k))]:
                add_pair(pos_id, neg_id)

            fam_counts: Counter = Counter()
            for neg_id, _score_value, neg_cand in negs:
                fam = family_name(neg_cand.get("name"))
                if fam_counts[fam] >= max(1, int(max_negatives_per_family)):
                    continue
                fam_counts[fam] += 1
                add_pair(pos_id, neg_id)

        if not function_pairs:
            stats["skip_no_margin_neg"] += 1
            continue
        if len(function_pairs) > int(max_pairs_per_function):
            head = max(1, int(max_pairs_per_function) // 2)
            tail = function_pairs[head:]
            sample_n = max(0, int(max_pairs_per_function) - head)
            function_pairs = function_pairs[:head] + rng.sample(tail, min(sample_n, len(tail)))
        pairs.extend(function_pairs)
        stats["functions_with_pairs"] += 1
        stats["pairs"] += len(function_pairs)
        stats["positive_candidates"] += len(positives)
        stats["positive_family_" + family_name(positives[0][2].get("name"))] += 1
    return pairs, stats


def train_ranker(
    rows: Sequence[Dict[str, Any]],
    train_indices: Sequence[int],
    *,
    epochs: int,
    lr: float,
    l2: float,
    max_pairs_per_function: int,
    positive_top_k: int,
    hard_negative_k: int,
    max_negatives_per_family: int,
    min_margin: float,
    seed: int,
) -> Dict[str, Any]:
    feature_names: List[str] = []
    raw_vectors: Dict[Tuple[int, int], List[float]] = {}
    all_vectors: List[List[float]] = []
    for ridx in train_indices:
        row = rows[ridx]
        for cand in candidate_pool(row):
            names, vec = candidate_features(row, cand)
            if not feature_names:
                feature_names = names
            raw_vectors[(ridx, safe_int(cand.get("candidate_id")))] = vec
            all_vectors.append(vec)
    if not feature_names:
        raise RuntimeError("no candidate features found")
    mean_vec, scale_vec = standard_stats(all_vectors)
    vectors = {k: normalize(v, mean_vec, scale_vec) for k, v in raw_vectors.items()}
    pairs, pair_stats = build_pairs(
        rows,
        train_indices,
        max_pairs_per_function=max_pairs_per_function,
        positive_top_k=positive_top_k,
        hard_negative_k=hard_negative_k,
        max_negatives_per_family=max_negatives_per_family,
        min_margin=min_margin,
        seed=seed,
    )
    if not pairs:
        raise RuntimeError(f"no training pairs built: {dict(pair_stats)}")

    rng = random.Random(seed)
    w = [0.0 for _ in feature_names]
    for epoch in range(int(epochs)):
        rng.shuffle(pairs)
        eta = float(lr) / math.sqrt(1.0 + epoch)
        for ridx, pos_id, neg_id in pairs:
            pv = vectors.get((ridx, pos_id))
            nv = vectors.get((ridx, neg_id))
            if pv is None or nv is None:
                continue
            diff = [p - n for p, n in zip(pv, nv)]
            z = dot(w, diff)
            # logistic pairwise loss: log(1 + exp(-z))
            grad_scale = -1.0 / (1.0 + math.exp(max(-40.0, min(40.0, z))))
            for j, value in enumerate(diff):
                w[j] -= eta * (grad_scale * value + float(l2) * w[j])

    return {
        "feature_names": feature_names,
        "weights": w,
        "mean": mean_vec,
        "scale": scale_vec,
        "pair_stats": dict(pair_stats),
        "train_pair_count": len(pairs),
    }


def model_score(model: Dict[str, Any], row: Dict[str, Any], cand: Dict[str, Any]) -> float:
    _names, vec = candidate_features(row, cand)
    norm = normalize(vec, model["mean"], model["scale"])
    return dot(model["weights"], norm)


def pick_by_model(row: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    cands = [
        c
        for c in candidate_pool(row)
        if bool(candidate_field(c, "hard_valid"))
    ]
    if not cands:
        return {}
    return max(
        cands,
        key=lambda c: (
            model_score(model, row, c),
            -float(family_name(c.get("name")) == "fallback"),
            -float(family_name(c.get("name")) == "linear_goto"),
            safe_float(candidate_field(c, "slot_locality_score"), 1.0),
            -safe_float(candidate_field(c, "empty_control_with_context_count")),
            -safe_float(candidate_field(c, "body_bb_outside_control_count")),
            safe_float(candidate_field(c, "structured_bb_ratio")),
            -safe_float(candidate_field(c, "fallback_bb_count")),
            -safe_float(candidate_field(c, "goto_count")),
            -safe_int(c.get("candidate_id")),
        ),
    )


def pick_oracle(row: Dict[str, Any]) -> Dict[str, Any]:
    cands = [
        c
        for c in candidate_pool(row)
        if bool(candidate_field(c, "hard_valid"))
    ]
    if not cands:
        return {}
    return max(cands, key=lambda c: (candidate_structural_score(c), -safe_int(c.get("candidate_id"))))


def evaluate(rows: Sequence[Dict[str, Any]], indices: Sequence[int], model: Dict[str, Any]) -> Dict[str, Any]:
    selected_old_like = 0
    selected_new_like = 0
    oracle_like = 0
    any_like = 0
    exact_oracle = 0
    new_in_oracle_top3 = 0
    new_in_oracle_top5 = 0
    old_scores: List[float] = []
    new_scores: List[float] = []
    oracle_scores: List[float] = []
    miss_examples: List[Dict[str, Any]] = []
    no_pool_examples: List[Dict[str, Any]] = []
    selected_name_dist: Counter = Counter()
    model_name_dist: Counter = Counter()
    oracle_name_dist: Counter = Counter()
    candidate_count_dist: Counter = Counter()

    for ridx in indices:
        row = rows[ridx]
        cg = row.get("candidate_generation") or {}
        cands = candidate_pool(row)
        valid = [c for c in cands if bool(candidate_field(c, "hard_valid"))]
        candidate_count_dist[safe_int(cg.get("candidate_count"), len(cands))] += 1
        old = selected_candidate(row)
        new = pick_by_model(row, model)
        oracle = pick_oracle(row)
        ranked_oracle = sorted(
            valid,
            key=lambda c: (candidate_structural_score(c), -safe_int(c.get("candidate_id"))),
            reverse=True,
        )
        any_has = any(is_structural_source_like(c) for c in valid)
        if any_has:
            any_like += 1
        if is_structural_source_like(old):
            selected_old_like += 1
        if is_structural_source_like(new):
            selected_new_like += 1
        if is_structural_source_like(oracle):
            oracle_like += 1
        if safe_int(new.get("candidate_id"), -1) == safe_int(oracle.get("candidate_id"), -2):
            exact_oracle += 1
        oracle_ids3 = {safe_int(c.get("candidate_id")) for c in ranked_oracle[:3]}
        oracle_ids5 = {safe_int(c.get("candidate_id")) for c in ranked_oracle[:5]}
        if safe_int(new.get("candidate_id"), -1) in oracle_ids3:
            new_in_oracle_top3 += 1
        if safe_int(new.get("candidate_id"), -1) in oracle_ids5:
            new_in_oracle_top5 += 1
        old_scores.append(candidate_structural_score(old))
        new_scores.append(candidate_structural_score(new))
        oracle_scores.append(candidate_structural_score(oracle))
        selected_name_dist[str(old.get("name") or "MISSING")] += 1
        model_name_dist[str(new.get("name") or "MISSING")] += 1
        oracle_name_dist[str(oracle.get("name") or "MISSING")] += 1

        if is_structural_source_like(oracle) and not is_structural_source_like(new) and len(miss_examples) < 30:
            miss_examples.append(
                {
                    "example": row.get("example"),
                    "input_line": row.get("input_line"),
                    "func": row.get("func_name"),
                    "old": old.get("name"),
                    "old_score": round(candidate_structural_score(old), 4),
                    "model": new.get("name"),
                    "model_score": round(candidate_structural_score(new), 4),
                    "oracle": oracle.get("name"),
                    "oracle_score": round(candidate_structural_score(oracle), 4),
                    "candidate_count": safe_int(cg.get("candidate_count"), len(cands)),
                }
            )
        if not any_has and len(no_pool_examples) < 20:
            no_pool_examples.append(
                {
                    "example": row.get("example"),
                    "input_line": row.get("input_line"),
                    "func": row.get("func_name"),
                    "candidate_count": safe_int(cg.get("candidate_count"), len(cands)),
                    "oracle": oracle.get("name"),
                    "oracle_score": round(candidate_structural_score(oracle), 4),
                }
            )

    n = len(indices)
    return {
        "n": n,
        "old_selected_like": selected_old_like,
        "model_selected_like": selected_new_like,
        "oracle_like": oracle_like,
        "any_like": any_like,
        "model_exact_oracle": exact_oracle,
        "model_in_oracle_top3": new_in_oracle_top3,
        "model_in_oracle_top5": new_in_oracle_top5,
        "old_score_mean": mean(old_scores),
        "old_score_median": med(old_scores),
        "model_score_mean": mean(new_scores),
        "model_score_median": med(new_scores),
        "oracle_score_mean": mean(oracle_scores),
        "oracle_score_median": med(oracle_scores),
        "old_selected_name_dist": selected_name_dist.most_common(20),
        "model_selected_name_dist": model_name_dist.most_common(20),
        "oracle_name_dist": oracle_name_dist.most_common(20),
        "candidate_count_dist": sorted(candidate_count_dist.items()),
        "miss_examples": miss_examples,
        "no_pool_examples": no_pool_examples,
    }


def write_report(path: Path, args: argparse.Namespace, model: Dict[str, Any], train_eval: Dict[str, Any], dev_eval: Dict[str, Any]) -> None:
    def block(title: str, ev: Dict[str, Any]) -> List[str]:
        n = int(ev["n"])
        return [
            f"-- {title} --",
            f"rows = {n}",
            f"old_ranker_selected_structural_source_like = {pct(ev['old_selected_like'], n)}",
            f"structural_ranker_selected_source_like = {pct(ev['model_selected_like'], n)}",
            f"oracle_structural_source_like = {pct(ev['oracle_like'], n)}",
            f"any_structural_source_like_candidate = {pct(ev['any_like'], n)}",
            f"structural_ranker_exact_oracle = {pct(ev['model_exact_oracle'], n)}",
            f"structural_ranker_in_oracle_top3 = {pct(ev['model_in_oracle_top3'], n)}",
            f"structural_ranker_in_oracle_top5 = {pct(ev['model_in_oracle_top5'], n)}",
            f"old_structural_score_mean = {ev['old_score_mean']:.4f}",
            f"structural_ranker_score_mean = {ev['model_score_mean']:.4f}",
            f"oracle_structural_score_mean = {ev['oracle_score_mean']:.4f}",
            f"old_structural_score_median = {ev['old_score_median']:.4f}",
            f"structural_ranker_score_median = {ev['model_score_median']:.4f}",
            f"oracle_structural_score_median = {ev['oracle_score_median']:.4f}",
            f"candidate_count_dist = {ev['candidate_count_dist']}",
            f"old_selected_name_dist = {ev['old_selected_name_dist']}",
            f"structural_ranker_selected_name_dist = {ev['model_selected_name_dist']}",
            f"oracle_name_dist = {ev['oracle_name_dist']}",
            "",
        ]

    weights = list(zip(model["feature_names"], model["weights"]))
    top_pos = sorted(weights, key=lambda x: x[1], reverse=True)[:20]
    top_neg = sorted(weights, key=lambda x: x[1])[:20]
    lines: List[str] = [
        "=== Module B-v3 structural source-like ranker ===",
        f"input_jsonl = {args.input_jsonl}",
        "feature_policy = binary_candidate_only_no_source_teacher_features",
        "label_policy = exported_rank_target_preference_with_ast_proxy_fallback",
        f"epochs = {args.epochs}",
        f"lr = {args.lr}",
        f"l2 = {args.l2}",
        f"seed = {args.seed}",
        f"dev_ratio = {args.dev_ratio}",
        f"positive_top_k = {args.positive_top_k}",
        f"hard_negative_k = {args.hard_negative_k}",
        f"max_negatives_per_family = {args.max_negatives_per_family}",
        f"train_pair_count = {model['train_pair_count']}",
        f"pair_stats = {model['pair_stats']}",
        "",
    ]
    lines.extend(block("train", train_eval))
    lines.extend(block("dev", dev_eval))
    lines.append("-- top positive weights --")
    lines.extend(f"{name} = {value:.6f}" for name, value in top_pos)
    lines.append("")
    lines.append("-- top negative weights --")
    lines.extend(f"{name} = {value:.6f}" for name, value in top_neg)
    lines.append("")
    lines.append("-- dev misses: oracle source-like but structural ranker missed --")
    for item in dev_eval.get("miss_examples") or []:
        lines.append(
            "example={example} input_line={input_line} func={func} "
            "old={old} old_score={old_score} model={model} model_score={model_score} "
            "oracle={oracle} oracle_score={oracle_score} candidate_count={candidate_count}".format(**item)
        )
    lines.append("")
    lines.append("-- dev no source-like in pool examples --")
    for item in dev_eval.get("no_pool_examples") or []:
        lines.append(
            "example={example} input_line={input_line} func={func} "
            "candidate_count={candidate_count} oracle={oracle} oracle_score={oracle_score}".format(**item)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-ranker", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dev-ratio", type=float, default=0.2)
    ap.add_argument("--max-pairs-per-function", type=int, default=64)
    ap.add_argument("--positive-top-k", type=int, default=3)
    ap.add_argument("--hard-negative-k", type=int, default=12)
    ap.add_argument("--max-negatives-per-family", type=int, default=8)
    ap.add_argument("--min-margin", type=float, default=0.03)
    args = ap.parse_args()

    rows = load_rows(Path(args.input_jsonl))
    train_idx, dev_idx = split_rows(rows, dev_ratio=args.dev_ratio, seed=args.seed)
    model = train_ranker(
        rows,
        train_idx,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        max_pairs_per_function=args.max_pairs_per_function,
        positive_top_k=args.positive_top_k,
        hard_negative_k=args.hard_negative_k,
        max_negatives_per_family=args.max_negatives_per_family,
        min_margin=args.min_margin,
        seed=args.seed,
    )
    train_eval = evaluate(rows, train_idx, model)
    dev_eval = evaluate(rows, dev_idx, model) if dev_idx else evaluate(rows, train_idx, model)
    payload = {
        "schema": "module_b_v3_structural_linear_ranker_v1",
        "feature_policy": "binary_candidate_only_no_source_teacher_features",
        "label_policy": "exported_rank_target_preference_with_ast_proxy_fallback",
        "args": vars(args),
        "feature_names": model["feature_names"],
        "weights": model["weights"],
        "mean": model["mean"],
        "scale": model["scale"],
        "pair_stats": model["pair_stats"],
        "train_pair_count": model["train_pair_count"],
        "train_eval": train_eval,
        "dev_eval": dev_eval,
    }
    Path(args.out_ranker).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(Path(args.out_txt), args, model, train_eval, dev_eval)
    print(f"rows = {len(rows)}")
    print(f"train_rows = {len(train_idx)} dev_rows = {len(dev_idx)}")
    print(f"train_pair_count = {model['train_pair_count']}")
    print(f"train_old_selected_source_like = {pct(train_eval['old_selected_like'], train_eval['n'])}")
    print(f"train_structural_ranker_selected_source_like = {pct(train_eval['model_selected_like'], train_eval['n'])}")
    print(f"train_oracle_source_like = {pct(train_eval['oracle_like'], train_eval['n'])}")
    print(f"dev_old_selected_source_like = {pct(dev_eval['old_selected_like'], dev_eval['n'])}")
    print(f"dev_structural_ranker_selected_source_like = {pct(dev_eval['model_selected_like'], dev_eval['n'])}")
    print(f"dev_oracle_source_like = {pct(dev_eval['oracle_like'], dev_eval['n'])}")
    print(f"saved_ranker = {args.out_ranker}")
    print(f"saved_txt = {args.out_txt}")


if __name__ == "__main__":
    main()

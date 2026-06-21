#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Similarity audit for Module-B v3 candidate pools.

The candidate-oracle exporter reports exact-match metrics.  This audit adds
continuous similarity numbers so rule/MVP candidates can be judged as a
source-structure approximation rather than only correct/incorrect.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MVP_PATH = ROOT / "struclift" / "utils" / "module_b_v3_mvp.py"
MVP_SPEC = importlib.util.spec_from_file_location("module_b_v3_mvp_standalone", MVP_PATH)
if MVP_SPEC is None or MVP_SPEC.loader is None:
    raise RuntimeError(f"cannot load {MVP_PATH}")
MVP_MODULE = importlib.util.module_from_spec(MVP_SPEC)
sys.modules[MVP_SPEC.name] = MVP_MODULE
MVP_SPEC.loader.exec_module(MVP_MODULE)

normalize_depths = None


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def as_outline(raw: Any) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not isinstance(raw, list):
        return out
    for x in raw:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            try:
                out.append((str(x[0]), int(x[1])))
            except Exception:
                continue
    return out


def norm_depth(outline: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
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
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def seq_f1(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    hit = lcs_len(pred, gold)
    p = hit / max(1, len(pred))
    r = hit / max(1, len(gold))
    return 2.0 * p * r / max(1e-8, p + r)


def multiset_f1(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    cp = collections.Counter(pred)
    cg = collections.Counter(gold)
    hit = sum(min(cp[k], cg[k]) for k in set(cp) | set(cg))
    if not pred and not gold:
        return 1.0
    p = hit / max(1, len(pred))
    r = hit / max(1, len(gold))
    return 2.0 * p * r / max(1e-8, p + r)


def candidate_by_id(row: Dict[str, Any], cid: Any) -> Optional[Dict[str, Any]]:
    try:
        target = int(cid)
    except Exception:
        return None
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return None
    for c in candidates:
        try:
            if int(c.get("candidate_id")) == target:
                return c
        except Exception:
            continue
    return None


def best_by_constructability(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return None
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for c in candidates:
        oracle = c.get("oracle") or {}
        vr = c.get("validator_report") or {}
        score = (
            3.0 * float(bool(oracle.get("outline_equal")))
            + 2.0 * float(oracle.get("outline_f1") or 0.0)
            + 1.0 * float(oracle.get("kind_f1") or 0.0)
            + 0.5 * float(oracle.get("slot_kind_f1") or 0.0)
            + 0.5 * float(bool(vr.get("valid")))
            - 0.03 * float(vr.get("fallback_region_count") or 0)
            - 0.02 * float(vr.get("goto_count") or 0)
        )
        if best is None or score > best[0]:
            best = (score, c)
    return best[1] if best else None


def candidate_similarity(row: Dict[str, Any], cand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    teacher = norm_depth(as_outline(row.get("teacher_outline")))
    if cand is None:
        pred = []
        name = None
        cid = None
        oracle = {}
        vr = {}
    else:
        pred = norm_depth(as_outline(cand.get("outline")))
        name = cand.get("name")
        cid = cand.get("candidate_id")
        oracle = cand.get("oracle") or {}
        vr = cand.get("validator_report") or {}
    pred_kind = kindseq(pred)
    teacher_kind = kindseq(teacher)
    return {
        "candidate_id": cid,
        "candidate_name": name,
        "control_count_gap": len(pred) - len(teacher),
        "control_count_equal": len(pred) == len(teacher),
        "kindseq_equal": pred_kind == teacher_kind,
        "outline_equal": pred == teacher and bool(teacher),
        "outline_lcs_f1": round(seq_f1(pred, teacher), 6),
        "kind_lcs_f1": round(seq_f1(pred_kind, teacher_kind), 6),
        "kind_bag_f1": round(multiset_f1(pred_kind, teacher_kind), 6),
        "max_depth_gap": (
            (max([d for _k, d in pred], default=-1) + 1)
            - (max([d for _k, d in teacher], default=-1) + 1)
        ),
        "fallback_region_count": int(vr.get("fallback_region_count") or oracle.get("fallback_count") or 0),
        "goto_count": int(vr.get("goto_count") or oracle.get("goto_count") or 0),
        "validator_valid": bool(vr.get("valid", True)),
        "oracle_score": oracle.get("score"),
    }


def mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def summarize(records: Sequence[Dict[str, Any]], prefix: str) -> List[str]:
    n = len(records)
    c = collections.Counter()
    gaps = collections.Counter()
    depth_gaps = collections.Counter()
    names = collections.Counter()
    fallback = collections.Counter()
    gotos = collections.Counter()
    outline_f1s: List[float] = []
    kind_lcs_f1s: List[float] = []
    kind_bag_f1s: List[float] = []
    for r in records:
        obj = r[prefix]
        names[str(obj.get("candidate_name"))] += 1
        gaps[int(obj.get("control_count_gap") or 0)] += 1
        depth_gaps[int(obj.get("max_depth_gap") or 0)] += 1
        fallback[int(obj.get("fallback_region_count") or 0)] += 1
        gotos[int(obj.get("goto_count") or 0)] += 1
        for key in ("control_count_equal", "kindseq_equal", "outline_equal", "validator_valid"):
            if obj.get(key):
                c[key] += 1
        outline_f1s.append(float(obj.get("outline_lcs_f1") or 0.0))
        kind_lcs_f1s.append(float(obj.get("kind_lcs_f1") or 0.0))
        kind_bag_f1s.append(float(obj.get("kind_bag_f1") or 0.0))

    lines: List[str] = []
    lines.append(f"-- {prefix} --")
    lines.append(f"{prefix}_control_count_equal = {pct(c['control_count_equal'], n)}")
    lines.append(f"{prefix}_kindseq_equal = {pct(c['kindseq_equal'], n)}")
    lines.append(f"{prefix}_outline_equal = {pct(c['outline_equal'], n)}")
    lines.append(f"{prefix}_validator_valid = {pct(c['validator_valid'], n)}")
    lines.append(f"{prefix}_outline_lcs_f1_mean = {mean(outline_f1s):.4f}")
    lines.append(f"{prefix}_outline_lcs_f1_median = {median(outline_f1s):.4f}")
    lines.append(f"{prefix}_kind_lcs_f1_mean = {mean(kind_lcs_f1s):.4f}")
    lines.append(f"{prefix}_kind_bag_f1_mean = {mean(kind_bag_f1s):.4f}")
    lines.append(f"{prefix}_candidate_name_dist = {names.most_common(20)}")
    lines.append(f"{prefix}_control_count_gap_dist = {sorted(gaps.items())}")
    lines.append(f"{prefix}_max_depth_gap_dist = {sorted(depth_gaps.items())}")
    lines.append(f"{prefix}_fallback_region_count_dist = {sorted(fallback.items())}")
    lines.append(f"{prefix}_goto_count_dist = {sorted(gotos.items())}")
    return lines


def audit(args: argparse.Namespace) -> None:
    pred_jsonl = Path(args.pred_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve() if args.out_txt else None
    out_jsonl = Path(args.out_jsonl).expanduser().resolve() if args.out_jsonl else None

    records: List[Dict[str, Any]] = []
    for _line_no, row in iter_jsonl(pred_jsonl, int(args.max_examples)):
        top1 = candidate_by_id(row, row.get("top1_candidate_id"))
        oracle = candidate_by_id(row, row.get("oracle_best_candidate_id"))
        construct_best = best_by_constructability(row)
        rec = {
            "example": row.get("example"),
            "input_line": row.get("input_line"),
            "func_name": row.get("func_name"),
            "n_bb": row.get("n_bb"),
            "candidate_count": row.get("candidate_count"),
            "teacher_outline": row.get("teacher_outline"),
            "top1": candidate_similarity(row, top1),
            "oracle": candidate_similarity(row, oracle),
            "constructability_best": candidate_similarity(row, construct_best),
        }
        records.append(rec)

    n = len(records)
    lines: List[str] = []
    lines.append("===== MODULE B-v3 CANDIDATE SIMILARITY AUDIT =====")
    lines.append(f"pred_jsonl = {pred_jsonl}")
    lines.append(f"n = {n}")
    cand_counts = collections.Counter(int(r.get("candidate_count") or 0) for r in records)
    lines.append(f"candidate_count_dist = {sorted(cand_counts.items())}")
    lines.append("")
    for prefix in ("top1", "oracle", "constructability_best"):
        lines.extend(summarize(records, prefix))
        lines.append("")

    lines.append(f"-- low-similarity oracle preview, first {int(args.max_bad_preview)} --")
    bad = sorted(records, key=lambda r: (r["oracle"]["outline_lcs_f1"], r["oracle"]["kind_lcs_f1"]))
    for r in bad[: int(args.max_bad_preview)]:
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"n_bb={r.get('n_bb')} cand_count={r.get('candidate_count')} "
            f"oracle={r['oracle'].get('candidate_name')} outline_f1={r['oracle'].get('outline_lcs_f1')} "
            f"kind_f1={r['oracle'].get('kind_lcs_f1')} count_gap={r['oracle'].get('control_count_gap')} "
            f"fallback={r['oracle'].get('fallback_region_count')} goto={r['oracle'].get('goto_count')} "
            f"teacher_outline={r.get('teacher_outline')}"
        )

    report = "\n".join(lines) + "\n"
    print(report, end="")

    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(report, encoding="utf-8")
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit continuous source-structure similarity for Module-B v3 candidates")
    ap.add_argument("--pred-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-bad-preview", type=int, default=40)
    args = ap.parse_args()
    audit(args)


if __name__ == "__main__":
    main()

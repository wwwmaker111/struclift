#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Module-B v3 candidate-set oracle top-k quality.

This script answers one question before training any ranker:

    Does the rule-generated candidate pool contain a good skeleton?

Generation remains binary-only.  The teacher JSONL is used only to score and
rank candidates offline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
        for x in pairs:
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append((str(x[0]), int(x[1])))
        return out
    controls = teacher.get("controls") if isinstance(teacher.get("controls"), list) else []
    return [
        (str(c.get("kind") or "unknown"), int(c.get("depth") or 0))
        for c in controls
        if str(c.get("kind") or "") in {"if", "loop", "switch"}
    ]


def teacher_slot_layout(teacher: Dict[str, Any]) -> List[Tuple[str, int]]:
    slots = teacher.get("slots") if isinstance(teacher.get("slots"), list) else []
    out: List[Tuple[str, int]] = []
    for s in slots:
        if not bool(s.get("trainable", True)):
            continue
        typ = str(s.get("slot_type") or "")
        if typ == "FILL_COND":
            kind = "COND"
        elif typ == "FILL_BLOCK":
            kind = "BLOCK"
        elif typ == "FILL_BRANCH":
            kind = "BRANCH"
        else:
            continue
        out.append((kind, int(s.get("render_depth") or 0)))
    return out


def pred_slot_layout(slots: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for s in slots:
        typ = str(s.get("slot_type") or "")
        if typ in {"COND", "BLOCK"}:
            out.append((typ, 0))
        elif typ == "UNSTRUCTURED_REGION":
            out.append(("BLOCK", 0))
        elif typ == "GOTO":
            out.append(("BRANCH", 0))
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
    precision = hit / max(1, len(pred))
    recall = hit / max(1, len(gold))
    return 2.0 * precision * recall / max(1e-8, precision + recall)


def decl_from_teacher(teacher: Dict[str, Any]) -> str:
    text = str(teacher.get("teacher_skeleton") or "")
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.endswith("{"):
        first = first[:-1].rstrip()
    if first and "(" in first and ")" in first:
        return first
    return "int func(...)"


def score_candidate(
    cand_rec: Dict[str, Any],
    *,
    t_outline: Sequence[Tuple[str, int]],
    t_layout: Sequence[Tuple[str, int]],
) -> Dict[str, Any]:
    p_outline = [(str(k), int(d)) for k, d in cand_rec.get("outline") or []]
    p_norm = normalize_depths(p_outline)
    t_norm = normalize_depths(t_outline)
    p_layout = [(str(k), int(d)) for k, d in cand_rec.get("slot_layout") or []]
    p_kind = kindseq(p_outline)
    t_kind = kindseq(t_outline)
    p_slot_kind = [k for k, _d in p_layout]
    t_slot_kind = [k for k, _d in t_layout]
    vr = cand_rec.get("validator_report") or {}
    outline_equal = bool(t_outline) and p_norm == t_norm
    kindseq_equal = bool(t_outline) and p_kind == t_kind
    slot_layout_equal = bool(t_layout) and p_layout == list(t_layout)
    slot_kindseq_equal = bool(t_layout) and p_slot_kind == t_slot_kind
    outline_f1 = seq_f1(p_norm, t_norm)
    kind_f1 = seq_f1(p_kind, t_kind)
    slot_f1 = seq_f1(p_slot_kind, t_slot_kind)
    fallback = int(vr.get("fallback_region_count") or 0)
    goto = int(vr.get("goto_count") or 0)
    structured = int(vr.get("structured_control_count") or 0)
    valid_bonus = 1.0 if bool(vr.get("valid")) else -1.0
    # Oracle score is intentionally teacher-heavy.  Safety is a hard filter in
    # training later, but here invalid candidates should still sort last.
    score = (
        3.0 * float(outline_equal)
        + 2.0 * float(kindseq_equal)
        + 1.5 * float(slot_layout_equal)
        + 0.75 * float(slot_kindseq_equal)
        + 2.0 * outline_f1
        + 1.0 * kind_f1
        + 0.75 * slot_f1
        + 0.03 * structured
        - 0.05 * fallback
        - 0.03 * goto
        + valid_bonus
    )
    return {
        "score": round(float(score), 6),
        "outline_equal": outline_equal,
        "kindseq_equal": kindseq_equal,
        "slot_layout_equal": slot_layout_equal,
        "slot_kindseq_equal": slot_kindseq_equal,
        "outline_f1": round(float(outline_f1), 6),
        "kind_f1": round(float(kind_f1), 6),
        "slot_kind_f1": round(float(slot_f1), 6),
        "fallback_count": fallback,
        "goto_count": goto,
        "structured_control_count": structured,
        "outline_gap": len(p_outline) - len(t_outline),
        "slot_gap": len(p_layout) - len(t_layout),
    }


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
    rows: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for example, teacher in enumerate(teachers):
            line_no = int(teacher.get("input_line") or 0)
            row = source_by_line.get(line_no)
            if row is None:
                continue
            cfg = analyse_cfg(row)
            t_outline = teacher_outline(teacher)
            t_layout = teacher_slot_layout(teacher)
            candidate_objs = generate_candidates(
                cfg,
                max_region_nodes=int(args.max_region_nodes),
                max_mixed_headers=int(args.max_mixed_if_headers),
                candidate_beam=int(args.candidate_beam),
            )
            candidates = []
            for cid, cand in enumerate(candidate_objs):
                text = render_candidate(cand, decl_line=decl_from_teacher(teacher))
                vr = validate_candidate(cfg, cand, rendered_text=text)
                slots = [s.asdict() for s in cand.slots]
                cand_rec = {
                    "candidate_id": cid,
                    "name": cand.name,
                    "skeleton": text if bool(args.include_skeletons) else None,
                    "outline": [[k, d] for k, d in candidate_outline(cand)],
                    "slot_layout": [[k, d] for k, d in pred_slot_layout(slots)],
                    "slots": slots if bool(args.include_slots) else None,
                    "validator_report": vr.asdict(),
                }
                cand_rec["oracle"] = score_candidate(
                    cand_rec,
                    t_outline=t_outline,
                    t_layout=t_layout,
                )
                candidates.append(cand_rec)
            selected_cand, selected_report, _selected_text = select_candidate(cfg, candidate_objs)
            selected_idx = None
            for c in candidates:
                if c.get("name") == selected_cand.name:
                    selected_idx = c.get("candidate_id")
                    break

            ranked = sorted(
                candidates,
                key=lambda c: (
                    float((c.get("oracle") or {}).get("score") or -1e9),
                    -int(c.get("candidate_id") or 0),
                ),
                reverse=True,
            )
            top1 = next(
                (c for c in candidates if c.get("candidate_id") == selected_idx),
                candidates[0] if candidates else None,
            )
            oracle_best = ranked[0] if ranked else None
            rec = {
                "example": example,
                "input_line": line_no,
                "func_name": row.get("func_name") or teacher.get("func_name"),
                "n_bb": len(row.get("bin_block_lengths") or []),
                "teacher_outline": [[k, d] for k, d in t_outline],
                "teacher_slot_layout": [[k, d] for k, d in t_layout],
                "candidate_count": len(candidates),
                "top1_candidate_id": top1.get("candidate_id") if top1 else None,
                "top1_name": top1.get("name") if top1 else None,
                "top1_oracle": top1.get("oracle") if top1 else None,
                "selected_validator_report": selected_report.asdict(),
                "oracle_best_candidate_id": oracle_best.get("candidate_id") if oracle_best else None,
                "oracle_best_name": oracle_best.get("name") if oracle_best else None,
                "oracle_best": oracle_best.get("oracle") if oracle_best else None,
                "top3_contains_outline_equal": any((c.get("oracle") or {}).get("outline_equal") for c in ranked[:3]),
                "top5_contains_outline_equal": any((c.get("oracle") or {}).get("outline_equal") for c in ranked[:5]),
                "top3_contains_slot_layout_equal": any((c.get("oracle") or {}).get("slot_layout_equal") for c in ranked[:3]),
                "top5_contains_slot_layout_equal": any((c.get("oracle") or {}).get("slot_layout_equal") for c in ranked[:5]),
                "ranked_candidate_ids": [c.get("candidate_id") for c in ranked],
                "candidates": candidates,
            }
            rows.append(rec)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(rows)
    c = Counter()
    best_name = Counter()
    top1_name = Counter()
    cand_count = Counter()
    oracle_fallback = Counter()
    oracle_goto = Counter()
    top1_fallback = Counter()
    top1_goto = Counter()
    oracle_valid = 0
    top1_valid = 0
    oracle_outline_f1: List[float] = []
    top1_outline_f1: List[float] = []
    for r in rows:
        cand_count[int(r.get("candidate_count") or 0)] += 1
        top1 = r.get("top1_oracle") or {}
        best = r.get("oracle_best") or {}
        top1_name[str(r.get("top1_name"))] += 1
        best_name[str(r.get("oracle_best_name"))] += 1
        for prefix, obj in (("top1", top1), ("oracle", best)):
            for key in (
                "outline_equal",
                "kindseq_equal",
                "slot_layout_equal",
                "slot_kindseq_equal",
            ):
                if obj.get(key):
                    c[f"{prefix}_{key}"] += 1
        if r.get("top3_contains_outline_equal"):
            c["top3_outline_equal"] += 1
        if r.get("top5_contains_outline_equal"):
            c["top5_outline_equal"] += 1
        if r.get("top3_contains_slot_layout_equal"):
            c["top3_slot_layout_equal"] += 1
        if r.get("top5_contains_slot_layout_equal"):
            c["top5_slot_layout_equal"] += 1
        best_cand_id = r.get("oracle_best_candidate_id")
        top1_cand_id = r.get("top1_candidate_id")
        best_cand = next((x for x in r.get("candidates") or [] if x.get("candidate_id") == best_cand_id), None)
        top1_cand = next((x for x in r.get("candidates") or [] if x.get("candidate_id") == top1_cand_id), None)
        if bool(((best_cand or {}).get("validator_report") or {}).get("valid")):
            oracle_valid += 1
        if bool(((top1_cand or {}).get("validator_report") or {}).get("valid")):
            top1_valid += 1
        oracle_outline_f1.append(float(best.get("outline_f1") or 0.0))
        top1_outline_f1.append(float(top1.get("outline_f1") or 0.0))
        oracle_fallback[int(best.get("fallback_count") or 0)] += 1
        oracle_goto[int(best.get("goto_count") or 0)] += 1
        top1_fallback[int(top1.get("fallback_count") or 0)] += 1
        top1_goto[int(top1.get("goto_count") or 0)] += 1

    def pct(num: int) -> str:
        return f"{num}/{n} ({100.0 * num / max(1, n):.1f}%)"

    with out_txt.open("w", encoding="utf-8") as wf:
        wf.write("===== MODULE B-v3 CANDIDATE ORACLE TOP-K =====\n")
        wf.write(f"source_jsonl = {source_jsonl}\n")
        wf.write(f"teacher_jsonl = {teacher_jsonl}\n")
        wf.write(f"out_jsonl = {out_jsonl}\n")
        wf.write(f"n = {n}\n")
        wf.write(f"candidate_count_dist = {sorted(cand_count.items())}\n")
        wf.write(
            "note = top1 is the deterministic source-free selector; oracle is "
            "the source-teacher upper bound for L_rank supervision.\n"
        )
        wf.write("\n-- top1/current generator --\n")
        wf.write(f"top1_validator_valid = {pct(top1_valid)}\n")
        wf.write(f"top1_outline_f1_mean = {sum(top1_outline_f1) / max(1, len(top1_outline_f1)):.4f}\n")
        for key in ("outline_equal", "kindseq_equal", "slot_layout_equal", "slot_kindseq_equal"):
            wf.write(f"top1_{key} = {pct(c[f'top1_{key}'])}\n")
        wf.write(f"top1_name_dist = {top1_name.most_common()}\n")
        wf.write(f"top1_fallback_dist = {sorted(top1_fallback.items())}\n")
        wf.write(f"top1_goto_dist = {sorted(top1_goto.items())}\n")
        wf.write("\n-- oracle candidate-set upper bound --\n")
        wf.write(f"oracle_validator_valid = {pct(oracle_valid)}\n")
        wf.write(f"oracle_outline_f1_mean = {sum(oracle_outline_f1) / max(1, len(oracle_outline_f1)):.4f}\n")
        for key in ("outline_equal", "kindseq_equal", "slot_layout_equal", "slot_kindseq_equal"):
            wf.write(f"oracle_{key} = {pct(c[f'oracle_{key}'])}\n")
        wf.write(f"top3_contains_outline_equal = {pct(c['top3_outline_equal'])}\n")
        wf.write(f"top5_contains_outline_equal = {pct(c['top5_outline_equal'])}\n")
        wf.write(f"top3_contains_slot_layout_equal = {pct(c['top3_slot_layout_equal'])}\n")
        wf.write(f"top5_contains_slot_layout_equal = {pct(c['top5_slot_layout_equal'])}\n")
        wf.write(f"oracle_best_name_dist = {best_name.most_common()}\n")
        wf.write(f"oracle_fallback_dist = {sorted(oracle_fallback.items())}\n")
        wf.write(f"oracle_goto_dist = {sorted(oracle_goto.items())}\n")
        wf.write(f"elapsed_sec = {time.time() - t0:.1f}\n")
        wf.write("\n")
        wf.write("-- bad / informative examples preview --\n")
        bad = [
            r for r in rows
            if not ((r.get("oracle_best") or {}).get("outline_equal"))
            or not ((r.get("oracle_best") or {}).get("slot_layout_equal"))
        ]
        bad.sort(
            key=lambda r: (
                0 if (r.get("oracle_best") or {}).get("outline_equal") else 1,
                abs(int((r.get("oracle_best") or {}).get("outline_gap") or 0)),
                abs(int((r.get("oracle_best") or {}).get("slot_gap") or 0)),
            ),
            reverse=True,
        )
        for r in bad[:80]:
            wf.write("-" * 120 + "\n")
            wf.write(
                f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
                f"n_bb={r.get('n_bb')} top1={r.get('top1_name')} best={r.get('oracle_best_name')} "
                f"best_score={(r.get('oracle_best') or {}).get('score')}\n"
            )
            wf.write(f"teacher_outline={r.get('teacher_outline')}\n")
            wf.write(f"teacher_slot_layout={r.get('teacher_slot_layout')}\n")
            for cand in sorted(r.get("candidates") or [], key=lambda x: x.get("candidate_id")):
                wf.write(
                    f"  cand={cand.get('candidate_id')} name={cand.get('name')} "
                    f"oracle={cand.get('oracle')} outline={cand.get('outline')} "
                    f"slot_layout={cand.get('slot_layout')}\n"
                )

    print(f"n = {n}", flush=True)
    print(f"top1_validator_valid = {pct(top1_valid)}", flush=True)
    for key in ("outline_equal", "kindseq_equal", "slot_layout_equal", "slot_kindseq_equal"):
        print(f"top1_{key} = {pct(c[f'top1_{key}'])}", flush=True)
    print(f"oracle_validator_valid = {pct(oracle_valid)}", flush=True)
    for key in ("outline_equal", "kindseq_equal", "slot_layout_equal", "slot_kindseq_equal"):
        print(f"oracle_{key} = {pct(c[f'oracle_{key}'])}", flush=True)
    print(f"oracle_outline_f1_mean = {sum(oracle_outline_f1) / max(1, len(oracle_outline_f1)):.4f}", flush=True)
    print(f"top3_contains_outline_equal = {pct(c['top3_outline_equal'])}", flush=True)
    print(f"top5_contains_outline_equal = {pct(c['top5_outline_equal'])}", flush=True)
    print(f"top3_contains_slot_layout_equal = {pct(c['top3_slot_layout_equal'])}", flush=True)
    print(f"top5_contains_slot_layout_equal = {pct(c['top5_slot_layout_equal'])}", flush=True)
    print("oracle_best_name_dist =", best_name.most_common(), flush=True)
    print("saved_txt =", out_txt, flush=True)
    print("saved_jsonl =", out_jsonl, flush=True)
    print(f"elapsed_sec = {time.time() - t0:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Module-B v3 candidate oracle top-k audit")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=3)
    ap.add_argument("--candidate-beam", type=int, default=80)
    ap.add_argument("--include-skeletons", action="store_true")
    ap.add_argument("--include-slots", action="store_true")
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()

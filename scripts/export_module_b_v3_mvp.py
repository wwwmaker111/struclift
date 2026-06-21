#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export safe Module-B v3 MVP skeletons from binary CFG JSONL.

This is the model-free first version:

    binary CFG -> rule candidates -> validator -> fallback -> verified skeleton

Source/teacher data is optional and is used only for audit comparisons.  The
generated skeleton never reads source AST or teacher labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
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
build_verified_skeleton = MVP_MODULE.build_verified_skeleton


FILL_RE = re.compile(r"<FILL_(COND|BLOCK|BRANCH):slot_(\d+)>")
UNSTRUCTURED_RE = re.compile(r"<UNSTRUCTURED_REGION:slot_(\d+)>")
GOTO_RE = re.compile(r"<GOTO_SLOT:slot_(\d+)>")


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
    if not wanted_set:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
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


def normalize_depths(outline: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not outline:
        return []
    levels = {d: i for i, d in enumerate(sorted({int(d) for _k, d in outline}))}
    return [(str(k), int(levels[int(d)])) for k, d in outline]


def kindseq(outline: Sequence[Tuple[str, int]]) -> List[str]:
    return [str(k) for k, _d in outline]


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


def predicted_slot_layout(slots: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
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


def decl_from_teacher(teacher: Optional[Dict[str, Any]]) -> str:
    text = str((teacher or {}).get("teacher_skeleton") or "")
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.endswith("{"):
        first = first[:-1].rstrip()
    if first and "(" in first and ")" in first:
        return first
    return "int func(...)"


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def _render_outline(outline: Sequence[Tuple[str, int]]) -> str:
    if not outline:
        return "<none>"
    return "\n".join(f"{'  ' * int(d)}{k}" for k, d in outline)


def _slot_grounding_hygiene(slots: Sequence[Dict[str, Any]], n_bb: int) -> Dict[str, Any]:
    owners: Dict[int, List[int]] = {}
    invalid: List[Dict[str, Any]] = []
    missing_grounding: List[int] = []
    for s in slots:
        sid = int(s.get("slot_id") or 0)
        typ = str(s.get("slot_type") or "")
        primary = [int(x) for x in s.get("primary_bbs") or []]
        context = [int(x) for x in s.get("context_bbs") or []]
        if typ != "GOTO" and not primary:
            missing_grounding.append(sid)
        for key, vals in (("primary_bbs", primary), ("context_bbs", context)):
            for bb in vals:
                if bb < 0 or bb >= n_bb:
                    invalid.append({"slot_id": sid, "key": key, "bb": bb})
        if typ != "GOTO":
            for bb in primary:
                owners.setdefault(bb, []).append(sid)
    duplicate = {str(bb): ids for bb, ids in owners.items() if len(ids) > 1}
    missing_bbs = [bb for bb in range(n_bb) if bb not in owners]
    return {
        "missing_grounding_slots": missing_grounding,
        "invalid_bb_refs": invalid,
        "duplicate_primary_bbs": duplicate,
        "missing_primary_bbs": missing_bbs,
    }


def export(args: argparse.Namespace) -> None:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    teachers: List[Dict[str, Any]] = []
    if args.teacher_jsonl is not None:
        for _line_no, teacher in iter_jsonl(Path(args.teacher_jsonl).expanduser().resolve(), int(args.max_examples)):
            teachers.append(teacher)
    rows: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    if teachers:
        by_line = read_source_by_line(source_jsonl, [int(t.get("input_line") or 0) for t in teachers])
        for i, t in enumerate(teachers):
            line_no = int(t.get("input_line") or 0)
            src = by_line.get(line_no)
            if src is not None:
                rows.append((line_no, src, t))
    else:
        for line_no, row in iter_jsonl(source_jsonl, int(args.max_examples)):
            rows.append((line_no, row, None))

    t0 = time.time()
    records: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for idx, (line_no, row, teacher) in enumerate(rows):
            n_bb = len(row.get("bin_block_lengths") or [])
            decl = decl_from_teacher(teacher)
            built = build_verified_skeleton(
                row,
                decl_line=decl,
                max_region_nodes=int(args.max_region_nodes),
                max_mixed_headers=int(args.max_mixed_if_headers),
                candidate_beam=int(args.candidate_beam),
            )
            report = dict(built.get("validator_report") or {})
            pred_outline = [(str(k), int(d)) for k, d in (built.get("outline") or [])]
            teach_outline = teacher_outline(teacher or {})
            teach_norm = normalize_depths(teach_outline)
            pred_norm = normalize_depths(pred_outline)
            teach_layout = teacher_slot_layout(teacher or {})
            pred_layout = predicted_slot_layout(built.get("slots") or [])
            hygiene = _slot_grounding_hygiene(built.get("slots") or [], n_bb)
            rec = {
                "example": idx,
                "input_line": line_no,
                "func_name": row.get("func_name") or (teacher or {}).get("func_name"),
                "n_bb": n_bb,
                "candidate_name": built.get("candidate_name"),
                "teacher_outline": teach_outline,
                "teacher_outline_normalized": teach_norm,
                "pred_outline": pred_outline,
                "pred_outline_normalized": pred_norm,
                "count_equal": bool(teach_outline) and len(teach_outline) == len(pred_outline),
                "kindseq_equal": bool(teach_outline) and kindseq(teach_outline) == kindseq(pred_outline),
                "outline_equal": bool(teach_outline) and teach_norm == pred_norm,
                "teacher_slot_layout": teach_layout,
                "pred_slot_layout": pred_layout,
                "slot_kindseq_equal": bool(teach_layout) and [k for k, _ in teach_layout] == [k for k, _ in pred_layout],
                "slot_layout_equal": bool(teach_layout) and teach_layout == pred_layout,
                "function_decl": decl,
                "teacher_skeleton": (teacher or {}).get("teacher_skeleton"),
                "predicted_skeleton": built.get("skeleton"),
                "slots": built.get("slots"),
                "slot_grounding_hygiene": hygiene,
                "validator_report": report,
                "cfg": built.get("cfg"),
                "candidates": built.get("candidates"),
            }
            records.append(rec)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    counters = Counter()
    cand_counter = Counter()
    fallback_dist = Counter()
    goto_dist = Counter()
    outline_gap = Counter()
    for r in records:
        cand_counter[str(r.get("candidate_name"))] += 1
        vr = r.get("validator_report") or {}
        for key in (
            "valid",
            "bb_coverage_ok",
            "unique_primary_ok",
            "edge_preservation_ok",
            "slot_grounding_ok",
            "parse_success",
        ):
            if vr.get(key):
                counters[key] += 1
        if not (vr.get("fatal_errors") or []):
            counters["validator_fatal_error_zero"] += 1
        if r.get("count_equal"):
            counters["count_equal"] += 1
        if r.get("kindseq_equal"):
            counters["kindseq_equal"] += 1
        if r.get("outline_equal"):
            counters["outline_equal"] += 1
        if r.get("slot_kindseq_equal"):
            counters["slot_kindseq_equal"] += 1
        if r.get("slot_layout_equal"):
            counters["slot_layout_equal"] += 1
        fallback_dist[int(vr.get("fallback_region_count") or 0)] += 1
        goto_dist[int(vr.get("goto_count") or 0)] += 1
        outline_gap[len(r.get("pred_outline") or []) - len(r.get("teacher_outline") or [])] += 1

    elapsed = time.time() - t0
    with out_txt.open("w", encoding="utf-8") as wf:
        wf.write("===== MODULE B-v3 SAFE MVP EXPORT =====\n")
        wf.write(f"source_jsonl = {source_jsonl}\n")
        wf.write(f"teacher_jsonl = {args.teacher_jsonl}\n")
        wf.write(f"out_jsonl = {out_jsonl}\n")
        wf.write(f"n = {n}\n")
        wf.write("\n-- safety / validator --\n")
        for key in (
            "valid",
            "bb_coverage_ok",
            "unique_primary_ok",
            "edge_preservation_ok",
            "slot_grounding_ok",
            "parse_success",
            "validator_fatal_error_zero",
        ):
            wf.write(f"{key} = {_pct(counters[key], n)}\n")
        wf.write("\n-- source-teacher comparison, audit only --\n")
        for key in (
            "count_equal",
            "kindseq_equal",
            "outline_equal",
            "slot_kindseq_equal",
            "slot_layout_equal",
        ):
            wf.write(f"{key} = {_pct(counters[key], n)}\n")
        wf.write("\n-- distributions --\n")
        wf.write(f"candidate_dist = {cand_counter.most_common()}\n")
        wf.write(f"fallback_region_count_dist = {sorted(fallback_dist.items())}\n")
        wf.write(f"goto_count_dist = {sorted(goto_dist.items())}\n")
        wf.write(f"outline_count_gap_dist = {sorted(outline_gap.items())}\n")
        wf.write(f"elapsed_sec = {elapsed:.1f}\n")
        wf.write("\n")

        for r in records:
            vr = r.get("validator_report") or {}
            wf.write("=" * 120 + "\n")
            wf.write(
                f"example={r['example']} input_line={r['input_line']} func={r.get('func_name')} "
                f"n_bb={r['n_bb']} candidate={r.get('candidate_name')} "
                f"valid={vr.get('valid')} fallback_regions={vr.get('fallback_region_count')} "
                f"goto_count={vr.get('goto_count')} outline_equal={r.get('outline_equal')} "
                f"slot_layout_equal={r.get('slot_layout_equal')}\n"
            )
            wf.write("\n-- VALIDATOR REPORT --\n")
            wf.write(json.dumps(vr, ensure_ascii=False, indent=2) + "\n")
            wf.write("\n-- TEACHER OUTLINE, AUDIT ONLY --\n")
            wf.write(_render_outline(r.get("teacher_outline") or []) + "\n")
            wf.write("\n-- PREDICTED OUTLINE --\n")
            wf.write(_render_outline(r.get("pred_outline") or []) + "\n")
            wf.write("\n-- TEACHER SLOT LAYOUT, AUDIT ONLY --\n")
            wf.write(str(r.get("teacher_slot_layout") or []) + "\n")
            wf.write("\n-- PREDICTED SLOT LAYOUT --\n")
            wf.write(str(r.get("pred_slot_layout") or []) + "\n")
            if r.get("teacher_skeleton"):
                wf.write("\n-- TEACHER CANONICAL SKELETON, AUDIT ONLY --\n")
                wf.write(str(r.get("teacher_skeleton") or "").rstrip() + "\n")
            wf.write("\n-- B-v3 VERIFIED SKELETON --\n")
            wf.write(str(r.get("predicted_skeleton") or "").rstrip() + "\n\n")

    print(f"n = {n}", flush=True)
    for key in (
        "valid",
        "bb_coverage_ok",
        "edge_preservation_ok",
        "slot_grounding_ok",
        "parse_success",
        "validator_fatal_error_zero",
    ):
        print(f"{key} = {_pct(counters[key], n)}", flush=True)
    if teachers:
        print(f"outline_equal = {_pct(counters['outline_equal'], n)}", flush=True)
        print(f"slot_layout_equal = {_pct(counters['slot_layout_equal'], n)}", flush=True)
    print("candidate_dist =", cand_counter.most_common(), flush=True)
    print("fallback_region_count_dist =", sorted(fallback_dist.items()), flush=True)
    print("goto_count_dist =", sorted(goto_dist.items()), flush=True)
    print(f"saved_txt = {out_txt}", flush=True)
    print(f"saved_jsonl = {out_jsonl}", flush=True)
    print(f"elapsed_sec = {elapsed:.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export safe Module-B v3 MVP skeletons")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=3)
    ap.add_argument("--candidate-beam", type=int, default=80)
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()

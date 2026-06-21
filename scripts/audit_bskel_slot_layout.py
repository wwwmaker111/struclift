#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit slot-aware B_skel predicted skeletons.

This diagnostic is intentionally stricter than outline metrics.  It compares
the teacher and predicted FILL slot layout, checks BB grounding hygiene, and
estimates whether predicted slots can be matched to teacher slots by BB overlap.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FILL_RE = re.compile(r"<FILL_(COND|BLOCK|BRANCH):slot_(\d+)>")
PRIMARY_RE = re.compile(r"primary\s+\[([^\]]*)\]")
CONTEXT_RE = re.compile(r"context\s+\[([^\]]*)\]")


@dataclass
class Slot:
    kind: str
    sid: int
    indent: int
    primary: Optional[List[int]]
    context: Optional[List[int]]
    line: str
    order: int


def _parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x) for x in re.findall(r"-?\d+", text)]


def parse_slots(skeleton: str) -> List[Slot]:
    slots: List[Slot] = []
    for line in skeleton.splitlines():
        m = FILL_RE.search(line)
        if not m:
            continue
        pm = PRIMARY_RE.search(line)
        cm = CONTEXT_RE.search(line)
        leading = len(line) - len(line.lstrip(" "))
        slots.append(
            Slot(
                kind=m.group(1),
                sid=int(m.group(2)),
                indent=leading // 4,
                primary=_parse_int_list(pm.group(1)) if pm else None,
                context=_parse_int_list(cm.group(1)) if cm else None,
                line=line.strip(),
                order=len(slots),
            )
        )
    return slots


def _valid_bbs(slots: Sequence[Slot], n_bb: int) -> List[Tuple[int, str, int]]:
    bad: List[Tuple[int, str, int]] = []
    for s in slots:
        for key, vals in (("primary", s.primary), ("context", s.context)):
            for bb in vals or []:
                if bb < 0 or bb >= n_bb:
                    bad.append((s.sid, key, bb))
    return bad


def _slot000_bad(slots: Sequence[Slot]) -> bool:
    for s in slots:
        if s.sid == 0 and ((s.primary or []) or (s.context or [])):
            return True
    return False


def _dup_primary(slots: Sequence[Slot]) -> Dict[int, List[int]]:
    owners: Dict[int, List[int]] = collections.defaultdict(list)
    for s in slots:
        if s.sid == 0:
            continue
        for bb in s.primary or []:
            owners[int(bb)].append(s.sid)
    return {bb: ids for bb, ids in owners.items() if len(ids) > 1}


def _layout_signature(slots: Sequence[Slot], *, include_slot000: bool = False) -> List[Tuple[str, int]]:
    return [
        (s.kind, s.indent)
        for s in slots
        if include_slot000 or s.sid != 0
    ]


def _kind_sequence(slots: Sequence[Slot], *, include_slot000: bool = False) -> List[str]:
    return [
        s.kind
        for s in slots
        if include_slot000 or s.sid != 0
    ]


def _control_body_pattern(slots: Sequence[Slot]) -> List[bool]:
    """For every COND slot, record whether it has an immediate body BLOCK slot.

    This is a structural diagnostic, not a parser.  It uses the canonical pretty
    indentation emitted by the B_skel exporter: a direct body slot should appear
    at indent ``cond_indent + 1`` before the next slot at ``cond_indent`` or
    shallower.
    """
    pattern: List[bool] = []
    for i, s in enumerate(slots):
        if s.kind != "COND":
            continue
        has_body = False
        for t in slots[i + 1 :]:
            if t.indent <= s.indent:
                break
            if t.kind == "BLOCK" and t.indent == s.indent + 1:
                has_body = True
                break
        pattern.append(has_body)
    return pattern


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    aa = set(int(x) for x in a)
    bb = set(int(x) for x in b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _slot_score(pred: Slot, teacher: Slot) -> float:
    if pred.kind != teacher.kind:
        return -1.0
    p_primary = pred.primary or []
    t_primary = teacher.primary or []
    p_context = pred.context or []
    t_context = teacher.context or []
    primary_score = _jaccard(p_primary, t_primary)
    context_score = _jaccard(p_context, t_context)
    if p_primary or t_primary:
        return max(primary_score, 0.50 * context_score)
    return context_score


def match_slots(
    pred_slots: Sequence[Slot],
    teacher_slots: Sequence[Slot],
    *,
    threshold: float,
    min_gap: float,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[Dict[str, Any]]]:
    """Greedy one-to-one predicted-slot to teacher-slot matching."""
    pred = [
        s for s in pred_slots
        if s.sid != 0 and not (s.kind == "BRANCH" and not (s.primary or s.context))
    ]
    teacher = [s for s in teacher_slots if s.sid != 0]
    used_pred: set[int] = set()
    used_teacher: set[int] = set()
    matches: List[Dict[str, Any]] = []

    # Teacher COND slots often carry no BB comments in the canonical skeleton.
    # For Module C targets these are naturally matched by control order; BLOCK
    # slots are the ones that need BB-overlap matching.
    pred_cond = [i for i, s in enumerate(pred) if s.kind == "COND"]
    teacher_cond = [i for i, s in enumerate(teacher) if s.kind == "COND"]
    for pi, ti in zip(pred_cond, teacher_cond):
        used_pred.add(pi)
        used_teacher.add(ti)
        matches.append(
            {
                "pred_sid": pred[pi].sid,
                "teacher_sid": teacher[ti].sid,
                "kind": "COND",
                "score": 1.0,
                "pred_primary": pred[pi].primary or [],
                "teacher_primary": teacher[ti].primary or [],
                "match_method": "cond_order",
            }
        )

    candidates: List[Tuple[float, int, int]] = []
    ambiguity: List[Dict[str, Any]] = []
    for pi, ps in enumerate(pred):
        if pi in used_pred:
            continue
        scored = sorted(
            [
                (_slot_score(ps, ts), ti)
                for ti, ts in enumerate(teacher)
                if ti not in used_teacher
            ],
            reverse=True,
        )
        if scored:
            best, best_ti = scored[0]
            second = scored[1][0] if len(scored) > 1 else -1.0
            if best >= threshold and second >= threshold and best - second < min_gap:
                ambiguity.append(
                    {
                        "pred_sid": ps.sid,
                        "pred_kind": ps.kind,
                        "best_score": round(best, 4),
                        "second_score": round(second, 4),
                        "best_teacher_sid": teacher[best_ti].sid,
                    }
                )
            if best >= threshold:
                candidates.append((best, pi, best_ti))

    candidates.sort(reverse=True)
    for score, pi, ti in candidates:
        if pi in used_pred or ti in used_teacher:
            continue
        used_pred.add(pi)
        used_teacher.add(ti)
        matches.append(
            {
                "pred_sid": pred[pi].sid,
                "teacher_sid": teacher[ti].sid,
                "kind": pred[pi].kind,
                "score": round(float(score), 4),
                "pred_primary": pred[pi].primary or [],
                "teacher_primary": teacher[ti].primary or [],
                "match_method": "bb_overlap",
            }
        )

    unmatched_pred = [pred[i].sid for i in range(len(pred)) if i not in used_pred]
    unmatched_teacher = [teacher[i].sid for i in range(len(teacher)) if i not in used_teacher]
    return matches, unmatched_pred, unmatched_teacher, ambiguity


def audit_row(row: Dict[str, Any], *, threshold: float, min_gap: float) -> Dict[str, Any]:
    pred_slots = parse_slots(str(row.get("predicted_skeleton") or ""))
    teacher_slots = parse_slots(str(row.get("teacher_skeleton") or ""))
    n_bb = int(row.get("n_bb") or 0)

    pred_layout = _layout_signature(pred_slots)
    teacher_layout = _layout_signature(teacher_slots)
    pred_kinds = _kind_sequence(pred_slots)
    teacher_kinds = _kind_sequence(teacher_slots)
    pred_body = _control_body_pattern(pred_slots)
    teacher_body = _control_body_pattern(teacher_slots)

    matches, unmatched_pred, unmatched_teacher, ambiguity = match_slots(
        pred_slots,
        teacher_slots,
        threshold=threshold,
        min_gap=min_gap,
    )
    pred_dup = _dup_primary(pred_slots)
    teacher_dup = _dup_primary(teacher_slots)
    invalid_pred = _valid_bbs(pred_slots, n_bb)
    missing_alignment = [
        s.sid for s in pred_slots
        if s.primary is None or s.context is None
    ]
    empty_pred = [
        {"sid": s.sid, "kind": s.kind, "line": s.line}
        for s in pred_slots
        if s.sid != 0 and s.primary == [] and s.context == []
    ]

    reasons: List[str] = []
    if not bool(row.get("outline_equal")):
        reasons.append("outline_not_equal")
    if pred_layout != teacher_layout:
        reasons.append("slot_layout_not_equal")
    if pred_body != teacher_body:
        reasons.append("control_body_slot_pattern_not_equal")
    if pred_dup:
        reasons.append("pred_duplicate_primary")
    if invalid_pred:
        reasons.append("invalid_pred_bb_ref")
    if _slot000_bad(pred_slots):
        reasons.append("slot000_has_bb")
    if missing_alignment:
        reasons.append("missing_pred_primary_or_context")
    if unmatched_pred:
        reasons.append("unmatched_pred_slots")
    if ambiguity:
        reasons.append("ambiguous_slot_matches")

    safe_strict = (
        bool(row.get("outline_equal"))
        and pred_layout == teacher_layout
        and not pred_dup
        and not invalid_pred
        and not _slot000_bad(pred_slots)
        and not missing_alignment
    )
    safe_overlap = (
        bool(row.get("outline_equal"))
        and not pred_dup
        and not invalid_pred
        and not _slot000_bad(pred_slots)
        and not missing_alignment
        and not unmatched_pred
        and not ambiguity
    )

    return {
        "example": row.get("example"),
        "input_line": row.get("input_line"),
        "func_name": row.get("func_name"),
        "n_bb": n_bb,
        "count_equal": bool(row.get("count_equal")),
        "kindseq_equal": bool(row.get("kindseq_equal")),
        "outline_equal": bool(row.get("outline_equal")),
        "pred_slot_count": len(pred_slots),
        "teacher_slot_count": len(teacher_slots),
        "slot_count_equal": len(pred_slots) == len(teacher_slots),
        "pred_cond_count": sum(s.kind == "COND" for s in pred_slots),
        "teacher_cond_count": sum(s.kind == "COND" for s in teacher_slots),
        "cond_count_equal": sum(s.kind == "COND" for s in pred_slots)
        == sum(s.kind == "COND" for s in teacher_slots),
        "pred_block_count": sum(s.kind == "BLOCK" for s in pred_slots),
        "teacher_block_count": sum(s.kind == "BLOCK" for s in teacher_slots),
        "block_count_equal": sum(s.kind == "BLOCK" for s in pred_slots)
        == sum(s.kind == "BLOCK" for s in teacher_slots),
        "slot_kindseq_equal": pred_kinds == teacher_kinds,
        "slot_layout_equal": pred_layout == teacher_layout,
        "control_body_slot_pattern_equal": pred_body == teacher_body,
        "pred_body_pattern": pred_body,
        "teacher_body_pattern": teacher_body,
        "pred_duplicate_primary": pred_dup,
        "teacher_duplicate_primary": teacher_dup,
        "invalid_pred_bb_refs": invalid_pred,
        "slot000_bad": _slot000_bad(pred_slots),
        "missing_pred_alignment_slots": missing_alignment,
        "empty_pred_slots": empty_pred,
        "matched_slot_count": len(matches),
        "unmatched_pred_slots": unmatched_pred,
        "unmatched_teacher_slots": unmatched_teacher,
        "ambiguous_matches": ambiguity,
        "matches": matches,
        "safe_for_modulec_strict": safe_strict,
        "safe_for_modulec_overlap": safe_overlap,
        "reasons": reasons,
        "pred_layout": pred_layout,
        "teacher_layout": teacher_layout,
    }


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / den:.1f}%)" if den else "0/0 (0.0%)"


def render_report(results: Sequence[Dict[str, Any]], *, max_bad_preview: int) -> str:
    n = len(results)
    c = collections.Counter()
    reason_counts = collections.Counter()
    slot_diff = collections.Counter()
    block_diff = collections.Counter()
    cond_diff = collections.Counter()
    bad_rows: List[Dict[str, Any]] = []
    for r in results:
        for key in (
            "count_equal",
            "kindseq_equal",
            "outline_equal",
            "slot_count_equal",
            "cond_count_equal",
            "block_count_equal",
            "slot_kindseq_equal",
            "slot_layout_equal",
            "control_body_slot_pattern_equal",
            "safe_for_modulec_strict",
            "safe_for_modulec_overlap",
        ):
            if r.get(key):
                c[key] += 1
        if r.get("slot000_bad"):
            c["slot000_bad"] += 1
        if r.get("invalid_pred_bb_refs"):
            c["invalid_pred_bb_examples"] += 1
        if r.get("pred_duplicate_primary"):
            c["pred_duplicate_primary_examples"] += 1
        if r.get("teacher_duplicate_primary"):
            c["teacher_duplicate_primary_examples"] += 1
        if r.get("empty_pred_slots"):
            c["empty_pred_slot_examples"] += 1
        slot_diff[int(r["pred_slot_count"]) - int(r["teacher_slot_count"])] += 1
        cond_diff[int(r["pred_cond_count"]) - int(r["teacher_cond_count"])] += 1
        block_diff[int(r["pred_block_count"]) - int(r["teacher_block_count"])] += 1
        for reason in r.get("reasons") or []:
            reason_counts[str(reason)] += 1
        if r.get("reasons"):
            bad_rows.append(r)

    lines: List[str] = []
    lines.append("===== B_skel slot-aware layout audit =====")
    lines.append(f"n = {n}")
    lines.append("")
    lines.append("-- outline / layout --")
    for key in (
        "count_equal",
        "kindseq_equal",
        "outline_equal",
        "slot_count_equal",
        "cond_count_equal",
        "block_count_equal",
        "slot_kindseq_equal",
        "slot_layout_equal",
        "control_body_slot_pattern_equal",
    ):
        lines.append(f"{key} = {_pct(c[key], n)}")
    lines.append("")
    lines.append("-- BB grounding hygiene --")
    lines.append(f"slot000_bad_examples = {c['slot000_bad']}")
    lines.append(f"invalid_pred_bb_examples = {c['invalid_pred_bb_examples']}")
    lines.append(f"pred_duplicate_primary_examples = {c['pred_duplicate_primary_examples']}")
    lines.append(f"teacher_duplicate_primary_examples = {c['teacher_duplicate_primary_examples']}")
    lines.append(f"empty_pred_slot_examples = {c['empty_pred_slot_examples']}")
    lines.append("")
    lines.append("-- Module C usability estimates --")
    lines.append(f"safe_for_modulec_strict = {_pct(c['safe_for_modulec_strict'], n)}")
    lines.append(f"safe_for_modulec_overlap = {_pct(c['safe_for_modulec_overlap'], n)}")
    lines.append("")
    lines.append("-- distributions --")
    lines.append(f"slot_count_diff(pred-teacher) = {sorted(slot_diff.items(), key=lambda kv: (abs(kv[0]), kv[0]))}")
    lines.append(f"cond_count_diff(pred-teacher) = {sorted(cond_diff.items(), key=lambda kv: (abs(kv[0]), kv[0]))}")
    lines.append(f"block_count_diff(pred-teacher) = {sorted(block_diff.items(), key=lambda kv: (abs(kv[0]), kv[0]))}")
    lines.append(f"reason_counts = {dict(reason_counts)}")

    lines.append("")
    lines.append(f"-- bad/example preview, first {max_bad_preview} --")
    for r in bad_rows[:max_bad_preview]:
        lines.append(
            "example={example} func={func_name} reasons={reasons} "
            "outline={outline_equal} layout={slot_layout_equal} body={control_body_slot_pattern_equal} "
            "pred_slots={pred_slot_count} teacher_slots={teacher_slot_count} matched={matched_slot_count} "
            "unmatched_pred={unmatched_pred_slots} dup_primary={pred_duplicate_primary}".format(**r)
        )
        if r.get("ambiguous_matches"):
            lines.append(f"  ambiguous_matches={r['ambiguous_matches'][:5]}")
        if r.get("empty_pred_slots"):
            lines.append(f"  empty_pred_slots={r['empty_pred_slots'][:5]}")
        if r.get("pred_layout") != r.get("teacher_layout"):
            lines.append(f"  teacher_layout={r.get('teacher_layout')}")
            lines.append(f"  pred_layout   ={r.get('pred_layout')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit B_skel slot-aware predicted skeleton layout")
    ap.add_argument("--pred-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--match-threshold", type=float, default=0.25)
    ap.add_argument("--match-min-gap", type=float, default=0.05)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-bad-preview", type=int, default=100)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    with args.pred_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_examples > 0 and i >= args.max_examples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            results.append(
                audit_row(
                    row,
                    threshold=float(args.match_threshold),
                    min_gap=float(args.match_min_gap),
                )
            )

    report = render_report(results, max_bad_preview=int(args.max_bad_preview))
    print(report, end="")

    if args.out_txt is not None:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(report, encoding="utf-8")
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

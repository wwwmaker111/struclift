#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit whether Module-B v3 outputs can build Module-C slot labels.

This is deliberately different from exact slot-layout equality.  Module C does
not require the predicted slot sequence to be identical to the teacher skeleton:
an extra branch/body slot is acceptable if it is explicitly empty.  What Module
C needs is simpler and stricter:

* every non-empty predicted fill slot must have a plausible source label;
* empty predicted slots must be explicit empty targets, not hidden missing BBs;
* teacher/source block BBs should be covered by predicted non-empty slots;
* the hard validator must still pass for B-v3 candidates.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


FILL_RE = re.compile(r"<FILL_(COND|BLOCK|BRANCH):slot_(\d+)>")
CASE_RE = re.compile(r"<CASE_SLOT:slot_(\d+)>")
UNSTRUCTURED_RE = re.compile(r"<UNSTRUCTURED_REGION:slot_(\d+)>")
GOTO_RE = re.compile(r"<(?:GOTO|BREAK|CONTINUE|EARLY_EXIT|CLEANUP_EXIT)_SLOT:slot_(\d+)>")
PRIMARY_RE = re.compile(r"primary\s+\[([^\]]*)\]")
CONTEXT_RE = re.compile(r"context\s+\[([^\]]*)\]")


@dataclass
class SlotRec:
    sid: int
    kind: str
    primary: List[int]
    context: List[int]
    role: str = ""
    source: str = ""
    trainable: bool = True
    order: int = 0


def _parse_ints(text: str) -> List[int]:
    if not text.strip():
        return []
    return sorted({int(x) for x in re.findall(r"-?\d+", text)})


def _as_int_list(x: Any) -> List[int]:
    if not isinstance(x, list):
        return []
    out: Set[int] = set()
    for v in x:
        try:
            out.add(int(v))
        except Exception:
            continue
    return sorted(out)


def _norm_kind(raw: str) -> str:
    raw = str(raw or "")
    if raw in {"FILL_COND", "COND"}:
        return "COND"
    if raw in {"FILL_BLOCK", "BLOCK", "UNSTRUCTURED_REGION"}:
        return "BLOCK"
    if raw in {"FILL_BRANCH", "BRANCH", "CASE"}:
        return "BRANCH"
    if raw in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
        return "BRANCH"
    return raw or "UNKNOWN"


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def read_teacher_by_line(path: Optional[Path], max_examples: int = 0) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(path, max_examples):
        input_line = int(row.get("input_line") or 0)
        if input_line > 0:
            out[input_line] = row
    return out


def slots_from_skeleton(text: str, *, trainable: bool = True) -> List[SlotRec]:
    out: List[SlotRec] = []
    for line in str(text or "").splitlines():
        m = FILL_RE.search(line)
        kind = ""
        sid: Optional[int] = None
        if m:
            kind = _norm_kind(m.group(1))
            sid = int(m.group(2))
        else:
            um = UNSTRUCTURED_RE.search(line)
            gm = GOTO_RE.search(line)
            cm_case = CASE_RE.search(line)
            if um:
                kind = "BLOCK"
                sid = int(um.group(1))
            elif cm_case:
                kind = "BRANCH"
                sid = int(cm_case.group(1))
            elif gm:
                kind = "GOTO"
                sid = int(gm.group(1))
        if sid is None:
            continue
        pm = PRIMARY_RE.search(line)
        cm = CONTEXT_RE.search(line)
        out.append(
            SlotRec(
                sid=sid,
                kind=kind,
                primary=_parse_ints(pm.group(1)) if pm else [],
                context=_parse_ints(cm.group(1)) if cm else [],
                source="skeleton",
                trainable=trainable,
                order=len(out),
            )
        )
    return out


def slots_from_teacher_record(row: Optional[Dict[str, Any]]) -> List[SlotRec]:
    if not row:
        return []
    raw_slots = row.get("slots")
    if isinstance(raw_slots, list):
        out: List[SlotRec] = []
        for i, s in enumerate(raw_slots):
            if not isinstance(s, dict):
                continue
            trainable = bool(s.get("trainable", True))
            kind = _norm_kind(str(s.get("slot_type") or s.get("kind") or ""))
            if kind not in {"COND", "BLOCK", "BRANCH"}:
                continue
            sid_raw = s.get("slot_id", s.get("sid", i))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = i
            out.append(
                SlotRec(
                    sid=sid,
                    kind=kind,
                    primary=_as_int_list(s.get("primary_bbs")),
                    context=_as_int_list(s.get("context_bbs")),
                    role=str(s.get("role") or ""),
                    source="teacher_jsonl",
                    trainable=trainable,
                    order=len(out),
                )
            )
        return out
    return slots_from_skeleton(str(row.get("teacher_skeleton") or ""), trainable=True)


def slots_from_candidate(candidate: Dict[str, Any]) -> List[SlotRec]:
    raw_slots = candidate.get("slots")
    if isinstance(raw_slots, list):
        out: List[SlotRec] = []
        for i, s in enumerate(raw_slots):
            if not isinstance(s, dict):
                continue
            kind = _norm_kind(str(s.get("slot_type") or s.get("kind") or ""))
            sid_raw = s.get("slot_id", s.get("sid", i))
            try:
                sid = int(sid_raw)
            except Exception:
                sid = i
            out.append(
                SlotRec(
                    sid=sid,
                    kind=kind,
                    primary=_as_int_list(s.get("primary_bbs")),
                    context=_as_int_list(s.get("context_bbs")),
                    role=str(s.get("role") or ""),
                    source="candidate_slots",
                    trainable=True,
                    order=len(out),
                )
            )
        return out
    skel = candidate.get("skeleton") or candidate.get("predicted_skeleton") or ""
    return slots_from_skeleton(str(skel), trainable=True)


def slots_from_export_row(row: Dict[str, Any]) -> List[SlotRec]:
    raw_slots = row.get("slots")
    if isinstance(raw_slots, list):
        return slots_from_candidate({"slots": raw_slots})
    return slots_from_skeleton(str(row.get("predicted_skeleton") or ""), trainable=True)


def _teacher_units(teacher_slots: Sequence[SlotRec]) -> Tuple[List[SlotRec], List[SlotRec]]:
    trainable = [s for s in teacher_slots if s.trainable]
    conds = [s for s in trainable if s.kind == "COND"]
    blocks = [s for s in trainable if s.kind in {"BLOCK", "BRANCH"} and s.primary]
    return conds, blocks


def _owner_map(slots: Sequence[SlotRec]) -> Dict[int, List[int]]:
    owners: Dict[int, List[int]] = collections.defaultdict(list)
    for s in slots:
        for bb in s.primary:
            owners[int(bb)].append(int(s.sid))
    return dict(owners)


def _slot_label_text(slot: SlotRec, teacher_owner: Dict[int, List[int]]) -> Tuple[bool, List[int], List[int]]:
    covered: Set[int] = set()
    matched_sids: Set[int] = set()
    for bb in slot.primary:
        ids = teacher_owner.get(int(bb), [])
        if ids:
            covered.add(int(bb))
            matched_sids.update(int(x) for x in ids)
    missing = [bb for bb in slot.primary if bb not in covered]
    return not missing, sorted(matched_sids), missing


def audit_candidate(
    *,
    pred_slots: Sequence[SlotRec],
    teacher_slots: Sequence[SlotRec],
    validator_report: Optional[Dict[str, Any]] = None,
    outline_equal: Optional[bool] = None,
) -> Dict[str, Any]:
    teacher_conds, teacher_blocks = _teacher_units(teacher_slots)
    teacher_block_owner = _owner_map(teacher_blocks)
    teacher_block_primary = sorted({bb for s in teacher_blocks for bb in s.primary})
    teacher_cond_has_primary = any(s.primary for s in teacher_conds)
    teacher_cond_by_sid = {int(s.sid): s for s in teacher_conds}
    unmatched_teacher_cond_sids: Set[int] = {int(s.sid) for s in teacher_conds}

    pred_fill = [s for s in pred_slots if s.kind in {"COND", "BLOCK", "BRANCH"}]
    pred_nonempty_blocks = [s for s in pred_fill if s.kind in {"BLOCK", "BRANCH"} and s.primary]
    pred_empty_slots = [s for s in pred_fill if s.kind in {"BLOCK", "BRANCH"} and not s.primary]
    pred_conds = [s for s in pred_fill if s.kind == "COND"]

    slot_reports: List[Dict[str, Any]] = []
    labelable = True
    unsafe_nonempty: List[Dict[str, Any]] = []
    covered_teacher_bbs: Set[int] = set()
    matched_teacher_slot_ids: Set[int] = set()

    for i, slot in enumerate(pred_conds):
        matched_teacher: Optional[SlotRec] = None
        match_mode = ""
        if slot.primary and teacher_cond_has_primary:
            slot_primary = set(int(x) for x in slot.primary)
            scored: List[Tuple[int, int, SlotRec]] = []
            for tc in teacher_conds:
                if int(tc.sid) not in unmatched_teacher_cond_sids:
                    continue
                overlap = len(slot_primary & set(int(x) for x in tc.primary))
                if overlap > 0:
                    scored.append((overlap, -int(tc.order), tc))
            if scored:
                scored.sort(reverse=True)
                matched_teacher = scored[0][2]
                match_mode = "COND_PRIMARY_BB"
        elif i < len(teacher_conds) and int(teacher_conds[i].sid) in unmatched_teacher_cond_sids:
            matched_teacher = teacher_conds[i]
            match_mode = "COND_ORDER_NO_PRIMARY"

        ok = matched_teacher is not None
        if not ok:
            labelable = False
            reason = (
                "condition_primary_not_in_teacher_conditions"
                if slot.primary and teacher_cond_has_primary
                else "extra_condition_slot"
            )
            unsafe_nonempty.append(
                {
                    "sid": slot.sid,
                    "kind": slot.kind,
                    "primary": slot.primary,
                    "reason": reason,
                }
            )
        else:
            matched_teacher_slot_ids.add(int(matched_teacher.sid))
            unmatched_teacher_cond_sids.discard(int(matched_teacher.sid))
        slot_reports.append(
            {
                "sid": slot.sid,
                "kind": slot.kind,
                "primary": slot.primary,
                "target": match_mode if ok else "UNMATCHED_COND",
                "constructable": ok,
                "teacher_sid": matched_teacher.sid if matched_teacher is not None else None,
                "teacher_primary": (
                    teacher_cond_by_sid[int(matched_teacher.sid)].primary
                    if matched_teacher is not None and int(matched_teacher.sid) in teacher_cond_by_sid
                    else []
                ),
            }
        )

    for slot in pred_empty_slots:
        slot_reports.append(
            {
                "sid": slot.sid,
                "kind": slot.kind,
                "primary": [],
                "target": "EMPTY",
                "constructable": True,
                "teacher_sids": [],
                "missing_primary_bbs": [],
            }
        )

    for slot in pred_nonempty_blocks:
        ok, teacher_sids, missing = _slot_label_text(slot, teacher_block_owner)
        if not ok:
            labelable = False
            unsafe_nonempty.append(
                {
                    "sid": slot.sid,
                    "kind": slot.kind,
                    "primary": slot.primary,
                    "missing_primary_bbs": missing,
                    "reason": "primary_bbs_not_in_teacher_block_labels",
                }
            )
        for bb in slot.primary:
            if bb in teacher_block_owner:
                covered_teacher_bbs.add(int(bb))
        matched_teacher_slot_ids.update(int(x) for x in teacher_sids)
        slot_reports.append(
            {
                "sid": slot.sid,
                "kind": slot.kind,
                "primary": slot.primary,
                "target": "TEACHER_BLOCKS",
                "constructable": ok,
                "teacher_sids": teacher_sids,
                "missing_primary_bbs": missing,
            }
        )

    missing_teacher_primary = [bb for bb in teacher_block_primary if bb not in covered_teacher_bbs]
    teacher_primary_coverage_ok = not missing_teacher_primary
    validator_ok = True
    if validator_report is not None:
        validator_ok = bool(validator_report.get("valid", False))
    safe_for_modulec = bool(labelable and teacher_primary_coverage_ok and validator_ok)

    return {
        "pred_slot_count": len(pred_slots),
        "pred_fill_slot_count": len(pred_fill),
        "pred_cond_count": len(pred_conds),
        "teacher_cond_count": len(teacher_conds),
        "pred_nonempty_block_count": len(pred_nonempty_blocks),
        "pred_empty_slot_count": len(pred_empty_slots),
        "teacher_block_slot_count": len(teacher_blocks),
        "pred_slots_labelable": bool(labelable),
        "teacher_primary_coverage_ok": bool(teacher_primary_coverage_ok),
        "safe_for_modulec": bool(safe_for_modulec),
        "validator_ok": bool(validator_ok),
        "outline_equal": bool(outline_equal) if outline_equal is not None else None,
        "teacher_primary_bbs": teacher_block_primary,
        "covered_teacher_primary_bbs": sorted(covered_teacher_bbs),
        "missing_teacher_primary_bbs": missing_teacher_primary,
        "matched_teacher_slot_ids": sorted(matched_teacher_slot_ids),
        "unsafe_nonempty_pred_slots": unsafe_nonempty,
        "slot_reports": sorted(slot_reports, key=lambda x: int(x.get("sid") or 0)),
    }


def _candidate_by_id(candidates: Sequence[Dict[str, Any]], cid: Any) -> Optional[Dict[str, Any]]:
    try:
        cid_int = int(cid)
    except Exception:
        return None
    for c in candidates:
        try:
            if int(c.get("candidate_id")) == cid_int:
                return c
        except Exception:
            continue
    return None


def audit_row(
    row: Dict[str, Any],
    *,
    teacher_by_line: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    input_line = int(row.get("input_line") or 0)
    teacher = teacher_by_line.get(input_line)
    if teacher is None and row.get("teacher_skeleton"):
        teacher = {"teacher_skeleton": row.get("teacher_skeleton"), "slots": row.get("teacher_slots")}
    teacher_slots = slots_from_teacher_record(teacher)

    candidates = row.get("candidates")
    out: Dict[str, Any] = {
        "example": row.get("example"),
        "input_line": input_line,
        "func_name": row.get("func_name"),
        "n_bb": row.get("n_bb"),
        "has_teacher_slots": bool(teacher_slots),
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
    }

    if isinstance(candidates, list) and candidates:
        top1 = _candidate_by_id(candidates, row.get("top1_candidate_id")) or candidates[0]
        oracle = _candidate_by_id(candidates, row.get("oracle_best_candidate_id")) or top1
        audits: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for name, cand in (("top1", top1), ("oracle", oracle)):
            audit = audit_candidate(
                pred_slots=slots_from_candidate(cand),
                teacher_slots=teacher_slots,
                validator_report=cand.get("validator_report"),
                outline_equal=(cand.get("oracle") or {}).get("outline_equal"),
            )
            audits.append((name, cand, audit))
            out[f"{name}_candidate_id"] = cand.get("candidate_id")
            out[f"{name}_candidate_name"] = cand.get("name")
            out[f"{name}_c_label"] = audit

        any_safe = False
        best_score = -1
        best_any: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        for cand in candidates:
            audit = audit_candidate(
                pred_slots=slots_from_candidate(cand),
                teacher_slots=teacher_slots,
                validator_report=cand.get("validator_report"),
                outline_equal=(cand.get("oracle") or {}).get("outline_equal"),
            )
            score = (
                1000 * int(bool(audit.get("safe_for_modulec")))
                + 100 * int(bool(audit.get("pred_slots_labelable")))
                + 100 * int(bool(audit.get("teacher_primary_coverage_ok")))
                + 10 * int(bool(audit.get("validator_ok")))
                - len(audit.get("unsafe_nonempty_pred_slots") or [])
                - len(audit.get("missing_teacher_primary_bbs") or [])
            )
            if audit.get("safe_for_modulec"):
                any_safe = True
            if score > best_score:
                best_score = score
                best_any = (cand, audit)
        out["any_candidate_safe_for_modulec"] = bool(any_safe)
        if best_any is not None:
            cand, audit = best_any
            out["constructability_best_candidate_id"] = cand.get("candidate_id")
            out["constructability_best_candidate_name"] = cand.get("name")
            out["constructability_best_c_label"] = audit
        return out

    audit = audit_candidate(
        pred_slots=slots_from_export_row(row),
        teacher_slots=teacher_slots,
        validator_report=row.get("validator_report"),
        outline_equal=row.get("outline_equal"),
    )
    out["selected_candidate_name"] = row.get("candidate_name")
    out["selected_c_label"] = audit
    out["safe_for_modulec"] = audit.get("safe_for_modulec")
    return out


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def render_report(rows: Sequence[Dict[str, Any]], *, max_bad_preview: int) -> str:
    n = len(rows)
    c = collections.Counter()
    bad: List[Dict[str, Any]] = []
    name_dist = collections.Counter()
    for r in rows:
        if r.get("candidate_count"):
            for prefix in ("top1", "oracle", "constructability_best"):
                obj = r.get(f"{prefix}_c_label") or {}
                if obj.get("pred_slots_labelable"):
                    c[f"{prefix}_pred_slots_labelable"] += 1
                if obj.get("teacher_primary_coverage_ok"):
                    c[f"{prefix}_teacher_primary_coverage_ok"] += 1
                if obj.get("safe_for_modulec"):
                    c[f"{prefix}_safe_for_modulec"] += 1
                name = r.get(f"{prefix}_candidate_name")
                if name:
                    name_dist[f"{prefix}:{name}"] += 1
            if r.get("any_candidate_safe_for_modulec"):
                c["any_candidate_safe_for_modulec"] += 1
            if not ((r.get("constructability_best_c_label") or {}).get("safe_for_modulec")):
                bad.append(r)
        else:
            obj = r.get("selected_c_label") or {}
            if obj.get("pred_slots_labelable"):
                c["selected_pred_slots_labelable"] += 1
            if obj.get("teacher_primary_coverage_ok"):
                c["selected_teacher_primary_coverage_ok"] += 1
            if obj.get("safe_for_modulec"):
                c["selected_safe_for_modulec"] += 1
            name = r.get("selected_candidate_name")
            if name:
                name_dist[f"selected:{name}"] += 1
            if not obj.get("safe_for_modulec"):
                bad.append(r)

    lines: List[str] = []
    lines.append("===== MODULE B-v3 MODULE-C LABEL CONSTRUCTABILITY AUDIT =====")
    lines.append(f"n = {n}")
    lines.append("")
    if any(r.get("candidate_count") for r in rows):
        lines.append("-- candidate-set constructability --")
        for prefix in ("top1", "oracle", "constructability_best"):
            lines.append(f"{prefix}_pred_slots_labelable = {_pct(c[f'{prefix}_pred_slots_labelable'], n)}")
            lines.append(f"{prefix}_teacher_primary_coverage_ok = {_pct(c[f'{prefix}_teacher_primary_coverage_ok'], n)}")
            lines.append(f"{prefix}_safe_for_modulec = {_pct(c[f'{prefix}_safe_for_modulec'], n)}")
        lines.append(f"any_candidate_safe_for_modulec = {_pct(c['any_candidate_safe_for_modulec'], n)}")
    else:
        lines.append("-- selected-output constructability --")
        lines.append(f"selected_pred_slots_labelable = {_pct(c['selected_pred_slots_labelable'], n)}")
        lines.append(f"selected_teacher_primary_coverage_ok = {_pct(c['selected_teacher_primary_coverage_ok'], n)}")
        lines.append(f"selected_safe_for_modulec = {_pct(c['selected_safe_for_modulec'], n)}")
    lines.append("")
    lines.append(f"candidate_name_dist = {name_dist.most_common()}")
    lines.append("")
    lines.append(f"-- unsafe preview, first {max_bad_preview} --")
    for r in bad[:max_bad_preview]:
        if r.get("candidate_count"):
            obj = r.get("constructability_best_c_label") or {}
            lines.append(
                f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
                f"best={r.get('constructability_best_candidate_name')} "
                f"labelable={obj.get('pred_slots_labelable')} coverage={obj.get('teacher_primary_coverage_ok')} "
                f"safe={obj.get('safe_for_modulec')}"
            )
        else:
            obj = r.get("selected_c_label") or {}
            lines.append(
                f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
                f"candidate={r.get('selected_candidate_name')} "
                f"labelable={obj.get('pred_slots_labelable')} coverage={obj.get('teacher_primary_coverage_ok')} "
                f"safe={obj.get('safe_for_modulec')}"
            )
        if obj.get("unsafe_nonempty_pred_slots"):
            lines.append(f"  unsafe_nonempty_pred_slots={obj.get('unsafe_nonempty_pred_slots')[:6]}")
        if obj.get("missing_teacher_primary_bbs"):
            lines.append(f"  missing_teacher_primary_bbs={obj.get('missing_teacher_primary_bbs')[:20]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Module-B v3 outputs for Module-C label constructability")
    ap.add_argument("--pred-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, default=None)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-bad-preview", type=int, default=80)
    args = ap.parse_args()

    teacher_by_line = read_teacher_by_line(args.teacher_jsonl, int(args.max_examples))
    rows = [
        audit_row(row, teacher_by_line=teacher_by_line)
        for _line_no, row in iter_jsonl(args.pred_jsonl, int(args.max_examples))
    ]
    report = render_report(rows, max_bad_preview=int(args.max_bad_preview))
    print(report, end="")

    if args.out_txt is not None:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(report, encoding="utf-8")
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

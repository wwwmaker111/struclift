#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit B_skel teacher JSONL exported from source AST + M_gt."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def outline_from_teacher(controls: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
    rows = []
    for c in controls:
        kind = str(c.get("kind") or "")
        if kind not in {"if", "loop", "switch"}:
            continue
        depth = 0
        parent = c.get("parent_control")
        seen = set()
        while parent is not None:
            try:
                p = int(parent)
            except Exception:
                break
            if p in seen or p < 0 or p >= len(controls):
                break
            seen.add(p)
            depth += 1
            parent = controls[p].get("parent_control")
        rows.append((int(c.get("control_id", len(rows))), kind, depth))
    rows.sort(key=lambda x: x[0])
    return [(kind, depth) for _cid, kind, depth in rows]


def source_outline(row: Dict[str, Any], source_control_filter: str) -> List[Tuple[str, int]]:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    outline = effect._source_control_outline(
        rows,
        active_lines=active,
        source_control_filter=source_control_filter,
    )
    if not outline and active:
        outline = effect._active_line_control_outline(
            row,
            active,
            source_control_filter=source_control_filter,
        )
    return [(str(t), int(d)) for t, d, _ln in outline]


def check_record(rec: Dict[str, Any]) -> Tuple[Counter[str], Dict[str, Any]]:
    bad: Counter[str] = Counter()
    n_bb = int(rec.get("n_bb") or 0)
    controls = rec.get("controls") if isinstance(rec.get("controls"), list) else []
    slots = rec.get("slots") if isinstance(rec.get("slots"), list) else []
    skeleton = str(rec.get("teacher_skeleton") or "")
    if n_bb <= 0:
        bad["bad_n_bb"] += 1
    if not controls:
        bad["no_controls"] += 1
    preamble = [s for s in slots if str(s.get("role") or "") == "preamble"]
    if len(preamble) != 1:
        bad["bad_preamble_count"] += 1
    elif int(preamble[0].get("slot_id", -1)) != 0:
        bad["preamble_not_slot0"] += 1
    if "<FILL_BLOCK:slot_000>" not in skeleton:
        bad["missing_preamble_in_skeleton"] += 1
    ctrl_ids = set()
    for c in controls:
        cid = c.get("control_id")
        if not isinstance(cid, int):
            bad["bad_control_id"] += 1
            continue
        ctrl_ids.add(cid)
        if c.get("kind") not in {"if", "loop", "switch"}:
            bad["bad_control_kind"] += 1
        for key in ("condition_bbs", "region_primary_bbs"):
            xs = c.get(key, c.get("region_bbs") if key == "region_primary_bbs" else [])
            if not isinstance(xs, list):
                bad[f"{key}_not_list"] += 1
                continue
            for bb in xs:
                if not isinstance(bb, int) or bb < 0 or bb >= n_bb:
                    bad[f"{key}_oob"] += 1
    owner_primary: Dict[int, int] = {}
    for s in slots:
        sid = int(s.get("slot_id", -1))
        trainable = bool(s.get("trainable", True))
        primary = s.get("primary_bbs", s.get("bbs"))
        context = s.get("context_bbs", s.get("bbs"))
        if not isinstance(primary, list):
            bad["primary_not_list"] += 1
            primary = []
        if not isinstance(context, list):
            bad["context_not_list"] += 1
            context = []
        if trainable and not primary:
            bad["empty_trainable_primary"] += 1
        if trainable and not context:
            bad["empty_trainable_context"] += 1
        owner = s.get("owner_control")
        if owner is not None and owner not in ctrl_ids:
            bad["slot_owner_missing"] += 1
        for bb in primary:
            if not isinstance(bb, int) or bb < 0 or bb >= n_bb:
                bad["primary_oob"] += 1
                continue
            if trainable:
                owner_primary[bb] = owner_primary.get(bb, 0) + 1
        for bb in context:
            if not isinstance(bb, int) or bb < 0 or bb >= n_bb:
                bad["context_oob"] += 1
    dup_primary = sum(1 for v in owner_primary.values() if v > 1)
    meta = {
        "n_bb": n_bb,
        "n_controls": len(controls),
        "n_slots": len(slots),
        "bb_multi_primary_owner": dup_primary,
        "teacher_outline": outline_from_teacher(controls),
    }
    return bad, meta


def audit(args: argparse.Namespace) -> None:
    jsonl = Path(args.teacher_jsonl).expanduser().resolve()
    source_path = Path(args.source_jsonl).expanduser().resolve() if args.source_jsonl else None
    source_by_line: Dict[int, Dict[str, Any]] = {}
    if source_path is not None:
        wanted = []
        records = list(iter_jsonl(jsonl))
        for r in records:
            try:
                wanted.append(int(r.get("input_line")))
            except Exception:
                pass
        want = set(wanted)
        with source_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line_no in want:
                    source_by_line[line_no] = json.loads(line)
                if len(source_by_line) >= len(want):
                    break
    else:
        records = list(iter_jsonl(jsonl))

    n = 0
    clean = 0
    bad_total: Counter[str] = Counter()
    kind_ok = 0
    outline_ok = 0
    gap0 = 0
    ast_kind_ok = 0
    ast_outline_ok = 0
    ast_gap0 = 0
    ast_available = 0
    no_src = 0
    multi_primary_examples = 0
    multi_primary_bbs = 0
    examples_bad: List[str] = []
    for rec in records:
        n += 1
        bad, meta = check_record(rec)
        if not bad:
            clean += 1
        else:
            bad_total.update(bad)
            if len(examples_bad) < 20:
                examples_bad.append(
                    f"example={rec.get('example')} line={rec.get('input_line')} func={rec.get('func_name')} bad={dict(bad)}",
                )
        if int(meta.get("bb_multi_primary_owner") or 0) > 0:
            multi_primary_examples += 1
            multi_primary_bbs += int(meta.get("bb_multi_primary_owner") or 0)
        pred = meta["teacher_outline"]
        ast_src_raw = rec.get("source_ast_outline_pairs")
        if isinstance(ast_src_raw, list):
            ast_src: List[Tuple[str, int]] = []
            for x in ast_src_raw:
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    ast_src.append((str(x[0]), int(x[1])))
            if ast_src:
                ast_available += 1
                if [x[0] for x in pred] == [x[0] for x in ast_src]:
                    ast_kind_ok += 1
                if pred == ast_src:
                    ast_outline_ok += 1
                pc = Counter(k for k, _d in pred)
                sc = Counter(k for k, _d in ast_src)
                if (pc["if"] - sc["if"], pc["loop"] - sc["loop"], pc["switch"] - sc["switch"]) == (0, 0, 0):
                    ast_gap0 += 1
        if source_path is not None:
            row = source_by_line.get(int(rec.get("input_line") or -1))
            if row is None:
                no_src += 1
                continue
            src = source_outline(row, str(args.source_control_filter))
            if [x[0] for x in pred] == [x[0] for x in src]:
                kind_ok += 1
            if pred == src:
                outline_ok += 1
            pc = Counter(k for k, _d in pred)
            sc = Counter(k for k, _d in src)
            if (pc["if"] - sc["if"], pc["loop"] - sc["loop"], pc["switch"] - sc["switch"]) == (0, 0, 0):
                gap0 += 1

    print("===== B_skel teacher audit =====")
    print(f"teacher_jsonl = {jsonl}")
    if source_path is not None:
        print(f"source_jsonl = {source_path}")
    print(f"n = {n}")
    print(f"format_clean = {clean}/{n} ({clean/max(1,n)*100:.1f}%)")
    print(f"bad = {dict(bad_total)}")
    print(f"multi_primary_owner_diagnostic = examples {multi_primary_examples}/{n}, bbs {multi_primary_bbs}")
    if ast_available:
        print(f"source_ast_available = {ast_available}/{n}")
        print(f"source_ast_control_count_gap0 = {ast_gap0}/{ast_available} ({ast_gap0/ast_available*100:.1f}%)")
        print(f"source_ast_kindseq_equal = {ast_kind_ok}/{ast_available} ({ast_kind_ok/ast_available*100:.1f}%)")
        print(f"source_ast_outline_equal = {ast_outline_ok}/{ast_available} ({ast_outline_ok/ast_available*100:.1f}%)")
    if source_path is not None:
        denom = max(1, n - no_src)
        print(f"source_available = {n-no_src}/{n}")
        print(f"bb_filtered_control_count_gap0 = {gap0}/{denom} ({gap0/denom*100:.1f}%)")
        print(f"bb_filtered_kindseq_equal = {kind_ok}/{denom} ({kind_ok/denom*100:.1f}%)")
        print(f"bb_filtered_outline_equal = {outline_ok}/{denom} ({outline_ok/denom*100:.1f}%)")
    if examples_bad:
        print()
        print("===== first bad examples =====")
        for x in examples_bad:
            print(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit B_skel teacher JSONL.")
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--source-jsonl", type=Path)
    ap.add_argument("--source-control-filter", default="bb-start-line", choices=["raw", "bb-start-line", "bb-range"])
    args = ap.parse_args()
    audit(args)


if __name__ == "__main__":
    main()

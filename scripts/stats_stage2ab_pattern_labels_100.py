#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count 10-class BB pattern labels in a Stage2AB/jsonl sample."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Counter, Dict, Iterable, List, Sequence, Tuple


PATTERN_NAMES = {
    0: "sequential",
    1: "if_header",
    2: "if_then_body",
    3: "if_else_body",
    4: "loop_header",
    5: "loop_body",
    6: "loop_exit",
    7: "switch_header",
    8: "switch_case_body",
    9: "function_prologue_epilogue",
}

CONTROL_HEADER_IDS = {1, 4, 7}
BODY_ROLE_IDS = {2, 3, 5, 6, 8}


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _pattern_labels(row: Dict[str, Any]) -> List[int]:
    raw = row.get("pattern_labels")
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _pct(n: int, d: int) -> str:
    return "0.0%" if d <= 0 else f"{100.0 * n / d:.2f}%"


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _write_report(rows: List[Tuple[int, Dict[str, Any], List[int]]], out: Path) -> None:
    total_funcs = len(rows)
    total_bbs = sum(len(labels) for _line, _row, labels in rows)
    counts: Counter[int] = collections.Counter()
    unknown: Counter[int] = collections.Counter()
    per_func_control_headers: List[int] = []
    per_func_nonseq: List[int] = []
    funcs_with_pattern: Counter[str] = collections.Counter()
    funcs_with_header = 0
    funcs_with_loop = 0
    funcs_with_switch = 0
    funcs_straightline_by_labels = 0

    for _line, _row, labels in rows:
        c = collections.Counter(labels)
        for lab, n in c.items():
            if lab in PATTERN_NAMES:
                counts[lab] += n
            else:
                unknown[lab] += n
        header_n = sum(c.get(i, 0) for i in CONTROL_HEADER_IDS)
        nonseq_n = sum(n for lab, n in c.items() if lab != 0)
        per_func_control_headers.append(header_n)
        per_func_nonseq.append(nonseq_n)
        if header_n > 0:
            funcs_with_header += 1
        if c.get(4, 0) > 0:
            funcs_with_loop += 1
        if c.get(7, 0) > 0:
            funcs_with_switch += 1
        if header_n == 0:
            funcs_straightline_by_labels += 1
        for lab in PATTERN_NAMES:
            if c.get(lab, 0) > 0:
                funcs_with_pattern[PATTERN_NAMES[lab]] += 1

    header_bbs = sum(counts.get(i, 0) for i in CONTROL_HEADER_IDS)
    body_role_bbs = sum(counts.get(i, 0) for i in BODY_ROLE_IDS)
    seq_bbs = counts.get(0, 0)
    pro_epi_bbs = counts.get(9, 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== STAGE2AB 10-CLASS BB PATTERN LABEL STATS =====\n")
        wf.write(f"functions = {total_funcs}\n")
        wf.write(f"total_bbs = {total_bbs}\n")
        wf.write(f"known_label_bbs = {sum(counts.values())}\n")
        wf.write(f"unknown_label_bbs = {sum(unknown.values())}\n")
        wf.write(f"control_header_bbs(if/loop/switch headers) = {header_bbs} ({_pct(header_bbs, total_bbs)})\n")
        wf.write(f"body_role_bbs(then/else/loop/case/exit roles) = {body_role_bbs} ({_pct(body_role_bbs, total_bbs)})\n")
        wf.write(f"sequential_bbs = {seq_bbs} ({_pct(seq_bbs, total_bbs)})\n")
        wf.write(f"function_prologue_epilogue_bbs = {pro_epi_bbs} ({_pct(pro_epi_bbs, total_bbs)})\n")
        wf.write(
            "control_headers_per_func min/median/max = "
            f"{min(per_func_control_headers, default=0)} / "
            f"{_median(per_func_control_headers):.4g} / "
            f"{max(per_func_control_headers, default=0)}\n"
        )
        wf.write(
            "non_sequential_labels_per_func min/median/max = "
            f"{min(per_func_nonseq, default=0)} / "
            f"{_median(per_func_nonseq):.4g} / "
            f"{max(per_func_nonseq, default=0)}\n"
        )
        wf.write(f"functions_with_any_control_header = {funcs_with_header} ({_pct(funcs_with_header, total_funcs)})\n")
        wf.write(f"functions_without_control_header_by_labels = {funcs_straightline_by_labels} ({_pct(funcs_straightline_by_labels, total_funcs)})\n")
        wf.write(f"functions_with_loop_header = {funcs_with_loop} ({_pct(funcs_with_loop, total_funcs)})\n")
        wf.write(f"functions_with_switch_header = {funcs_with_switch} ({_pct(funcs_with_switch, total_funcs)})\n")

        wf.write("\n===== BB COUNT BY 10 PATTERN CLASSES =====\n")
        wf.write("id  name                         bb_count   bb_pct    funcs_with_label\n")
        for lab in range(10):
            name = PATTERN_NAMES[lab]
            n = int(counts.get(lab, 0))
            f = int(funcs_with_pattern.get(name, 0))
            wf.write(f"{lab:2d}  {name:<28s} {n:8d}  {_pct(n, total_bbs):>8s}  {f:8d} ({_pct(f, total_funcs)})\n")
        if unknown:
            wf.write("\n===== UNKNOWN LABELS =====\n")
            for lab, n in sorted(unknown.items()):
                wf.write(f"{lab}: {n}\n")

        wf.write("\n===== PER FUNCTION =====\n")
        for idx, (line_no, row, labels) in enumerate(rows):
            c = collections.Counter(labels)
            parts = [
                f"{PATTERN_NAMES[lab]}={int(c.get(lab, 0))}"
                for lab in range(10)
                if int(c.get(lab, 0)) > 0
            ]
            header_n = sum(c.get(i, 0) for i in CONTROL_HEADER_IDS)
            wf.write(
                f"{idx:4d} line={line_no} func={row.get('func_name')} "
                f"bb={len(labels)} control_headers={header_n} "
                f"labels: {', '.join(parts) if parts else '<none>'}\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", "--input", dest="jsonl", type=Path, required=True)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    rows = [
        (line_no, row, _pattern_labels(row))
        for line_no, row in _iter_jsonl(args.jsonl.expanduser().resolve(), int(args.limit))
    ]
    _write_report(rows, args.out.expanduser().resolve())
    print(f"saved_to = {args.out.expanduser().resolve()}")


if __name__ == "__main__":
    main()

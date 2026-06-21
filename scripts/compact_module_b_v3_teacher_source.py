#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact a teacher-referenced source JSONL into a small remapped smoke set.

Module-B v3 evaluators use ``teacher.input_line`` as the physical 1-based line
number in the source JSONL.  Full-size enrichment preserves those physical line
numbers by copying every source row, which is wasteful for 100-row smoke runs.

This helper writes:

* a compact source JSONL containing only the teacher-referenced rows;
* a compact teacher JSONL where ``input_line`` is remapped to 1..N;
* metadata fields preserving the original line numbers for audits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        return 0.0


def compact(args: argparse.Namespace) -> str:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    teacher_jsonl = Path(args.teacher_jsonl).expanduser().resolve()
    out_source = Path(args.out_source_jsonl).expanduser().resolve()
    out_teacher = Path(args.out_teacher_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve() if args.out_txt else None
    out_source.parent.mkdir(parents=True, exist_ok=True)
    out_teacher.parent.mkdir(parents=True, exist_ok=True)
    if out_txt:
        out_txt.parent.mkdir(parents=True, exist_ok=True)

    teachers: List[Dict[str, Any]] = []
    original_lines: List[int] = []
    for _teacher_line_no, teacher in iter_jsonl(teacher_jsonl, int(args.max_examples)):
        input_line = _safe_int(teacher.get("input_line"), 0)
        if input_line <= 0:
            continue
        teachers.append(teacher)
        original_lines.append(input_line)
    if not teachers:
        raise RuntimeError(f"no valid teacher input_line found in {teacher_jsonl}")

    want = set(original_lines)
    source_rows: Dict[int, Dict[str, Any]] = {}
    max_line = max(want)
    for line_no, row in iter_jsonl(source_jsonl, max_line):
        if line_no in want:
            source_rows[line_no] = row
            if len(source_rows) == len(want):
                break

    missing = [x for x in original_lines if x not in source_rows]
    if missing:
        raise RuntimeError(f"source rows missing for teacher input_line values: {missing[:20]}")

    t0 = time.time()
    with out_source.open("w", encoding="utf-8") as sf, out_teacher.open("w", encoding="utf-8") as tf:
        for new_line, (teacher, old_line) in enumerate(zip(teachers, original_lines), 1):
            row = dict(source_rows[old_line])
            row.setdefault("module_b_v3_compact", {})
            row["module_b_v3_compact"] = {
                **(row.get("module_b_v3_compact") if isinstance(row.get("module_b_v3_compact"), dict) else {}),
                "remapped_input_line": int(new_line),
                "original_input_line": int(old_line),
                "source_jsonl": str(source_jsonl),
            }
            row.setdefault("original_input_line", int(old_line))
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")

            remapped_teacher = dict(teacher)
            remapped_teacher["original_input_line"] = int(old_line)
            remapped_teacher["input_line"] = int(new_line)
            remapped_teacher.setdefault("module_b_v3_compact", {})
            remapped_teacher["module_b_v3_compact"] = {
                **(
                    remapped_teacher.get("module_b_v3_compact")
                    if isinstance(remapped_teacher.get("module_b_v3_compact"), dict)
                    else {}
                ),
                "remapped_input_line": int(new_line),
                "original_input_line": int(old_line),
                "source_jsonl": str(source_jsonl),
                "teacher_jsonl": str(teacher_jsonl),
            }
            tf.write(json.dumps(remapped_teacher, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    lines = [
        "===== MODULE B-v3 COMPACT TEACHER/SOURCE =====",
        f"source_jsonl = {source_jsonl}",
        f"teacher_jsonl = {teacher_jsonl}",
        f"out_source_jsonl = {out_source}",
        f"out_teacher_jsonl = {out_teacher}",
        f"n = {len(teachers)}",
        f"original_input_line_min = {min(original_lines)}",
        f"original_input_line_max = {max(original_lines)}",
        f"source_size_mb = {_size_mb(source_jsonl):.3f}",
        f"out_source_size_mb = {_size_mb(out_source):.3f}",
        f"out_teacher_size_mb = {_size_mb(out_teacher):.3f}",
        f"size_reduction_x = {(_size_mb(source_jsonl) / max(0.001, _size_mb(out_source))):.2f}",
        f"elapsed_sec = {elapsed:.1f}",
    ]
    text = "\n".join(lines) + "\n"
    if out_txt:
        out_txt.write_text(text, encoding="utf-8")
    print(text, end="")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact Module-B v3 teacher/source JSONLs by remapping input_line")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-source-jsonl", type=Path, required=True)
    ap.add_argument("--out-teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    compact(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit B-v3 Module-C overfit SFT rows by rendering teacher-filled skeletons."""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FILL_RE = re.compile(r"<FILL_([A-Z_]+):([^>]+)>")


PRELUDE = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef long idx_t;
typedef long ssize_t;
typedef unsigned long uintptr_t;
typedef long ptrdiff_t;
typedef unsigned long uintmax_t;
typedef long intmax_t;

struct stat { unsigned long st_mode; unsigned long st_ino; unsigned long st_dev; };
struct argv_iterator { int dummy; };
struct F_triple { int dummy; };
enum argv_iter_err { AI_ERR_OK = 0 };

#ifndef DT_UNKNOWN
# define DT_UNKNOWN 0
#endif
#ifndef S_IFMT
# define S_IFMT 0170000
#endif
#ifndef S_IFREG
# define S_IFREG 0100000
#endif
#ifndef S_IFDIR
# define S_IFDIR 0040000
#endif
#ifndef S_IFLNK
# define S_IFLNK 0120000
#endif
#ifndef S_IFIFO
# define S_IFIFO 0010000
#endif
#ifndef S_IFSOCK
# define S_IFSOCK 0140000
#endif
#ifndef S_IFCHR
# define S_IFCHR 0020000
#endif
#ifndef S_IFBLK
# define S_IFBLK 0060000
#endif
"""


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def target_for_slot(record: Dict[str, Any], slot_id: str, fallback_idx: int) -> str:
    targets = list(record.get("sft_teacher_targets") or [])
    prompt_ids = [str(x) for x in (record.get("sft_prompt_slot_ids") or [])]
    if slot_id in prompt_ids:
        idx = prompt_ids.index(slot_id)
        if 0 <= idx < len(targets):
            return str(targets[idx] or "")
    m = re.fullmatch(r"slot_(\d+)", str(slot_id or ""))
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(targets):
            return str(targets[idx] or "")
    if 0 <= fallback_idx < len(targets):
        return str(targets[fallback_idx] or "")
    return ""


def render_teacher(record: Dict[str, Any]) -> str:
    skeleton = str(record.get("sft_skeleton_text_pretty") or record.get("b3_verified_skeleton_text") or "")
    occurrence = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal occurrence
        target = target_for_slot(record, m.group(2), occurrence)
        occurrence += 1
        if m.group(1) == "STMT" and not target.strip():
            return ";"
        if m.group(1) == "COND" and not target.strip():
            return "1"
        return target

    out = FILL_RE.sub(repl, skeleton)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.strip() + "\n"


def structural_reasons(code: str) -> List[str]:
    reasons: List[str] = []
    if "<FILL_" in code:
        reasons.append("unfilled_placeholder")
    if re.search(r"\b(?:if|while|for|switch)\s*\(\s*\)", code):
        reasons.append("empty_control_condition")
    if re.search(r"\b(?:if|while|for|switch)\s*\(\s*(?:if|while|for|switch)\s*\(", code):
        reasons.append("nested_control_header_in_condition")
    par = code.count("(") - code.count(")")
    brk = code.count("[") - code.count("]")
    brc = code.count("{") - code.count("}")
    if par or brk or brc:
        reasons.append(f"delimiter_balance_par{par}_brk{brk}_brc{brc}")
    return reasons


def compile_check(code: str, *, timeout: int) -> Tuple[bool, str]:
    if shutil.which("gcc") is None:
        return False, "gcc_unavailable"
    full = PRELUDE + "\n" + code
    try:
        res = subprocess.run(
            ["gcc", "-fsyntax-only", "-x", "c", "-", "-w"],
            input=full.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if res.returncode == 0:
        return True, ""
    return False, res.stderr.decode(errors="replace")


def summarize(rows: List[Dict[str, Any]], rendered: List[Tuple[str, str, List[str], bool, str]]) -> str:
    lines: List[str] = []
    n = len(rows)
    reason_counter = collections.Counter(r for _name, _code, reasons, _ok, _err in rendered for r in reasons)
    compile_ok = sum(1 for _name, _code, _reasons, ok, _err in rendered if ok)
    structural_ok = sum(1 for _name, _code, reasons, _ok, _err in rendered if not reasons)
    slot_total = sum(len(r.get("slot_types") or []) for r in rows)
    supervised = sum(sum(1 for x in (r.get("sft_teacher_slot_supervise") or []) if bool(x)) for r in rows)
    lines.append("=== B-v3 Module-C overfit teacher audit ===")
    lines.append(f"rows = {n}")
    lines.append(f"supervised_slots = {supervised}/{slot_total} ({100.0 * supervised / max(1, slot_total):.1f}%)")
    lines.append(f"structural_render_ok = {structural_ok}/{n} ({100.0 * structural_ok / max(1, n):.1f}%)")
    lines.append(f"gcc_syntax_ok = {compile_ok}/{n} ({100.0 * compile_ok / max(1, n):.1f}%)")
    lines.append(f"structural_reason_dist = {sorted(reason_counter.items())}")
    lines.append("")
    lines.append("-- per function --")
    for name, code, reasons, ok, err in rendered:
        lines.append(f"{name}: structural={'ok' if not reasons else reasons} gcc={'ok' if ok else 'fail'}")
        if err and err not in {"gcc_unavailable", "timeout"}:
            first = next((x for x in err.splitlines() if "error:" in x), err.splitlines()[0] if err.splitlines() else "")
            lines.append(f"  first_error={first[:220]}")
    lines.append("")
    lines.append("-- rendered preview --")
    for name, code, reasons, ok, err in rendered[:5]:
        lines.append(f"----- {name} -----")
        lines.extend(code.splitlines()[:60])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--compile-timeout", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = [r for _line, r in iter_jsonl(Path(args.input_jsonl).expanduser().resolve(), int(args.max_examples))]
    out_render_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_render_dir:
        out_render_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Tuple[str, str, List[str], bool, str]] = []
    for idx, row in enumerate(rows):
        name = str(row.get("func_name") or f"func_{idx:03d}")
        code = render_teacher(row)
        reasons = structural_reasons(code)
        ok, err = compile_check(code, timeout=int(args.compile_timeout))
        rendered.append((name, code, reasons, ok, err))
        if out_render_dir:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            (out_render_dir / f"{idx:02d}_{safe}.teacher.c").write_text(code, encoding="utf-8")
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(summarize(rows, rendered), encoding="utf-8")
    print(f"rows = {len(rows)}")
    print(f"saved_txt = {out_txt}")
    if out_render_dir:
        print(f"saved_rendered_dir = {out_render_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Module-C teacher JSONL by rendering slots and running C syntax checks.

This is a diagnostic script for data-builder quality. A clean result here does
not prove semantic equivalence, but it separates obvious teacher corruption
(empty conditions, semicolons inside conditions, glued returns, broken C syntax)
from expected harness issues such as project-specific typedefs or globals.
"""

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

DEFAULT_PRELUDE = r"""
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

struct quoting_options { int dummy; };
extern struct quoting_options quote_quoting_options;
extern char const *program_name;
extern char const *Version;
extern int optind;
extern char *optarg;

#ifndef _IO_IN_BACKUP
# define _IO_IN_BACKUP 0x100
#endif
#ifndef ATTRIBUTE_NORETURN
# define ATTRIBUTE_NORETURN
#endif
#ifndef ATTRIBUTE_PURE
# define ATTRIBUTE_PURE
#endif
#ifndef _GL_ATTRIBUTE_PURE
# define _GL_ATTRIBUTE_PURE
#endif
#ifndef _GL_ATTRIBUTE_CONST
# define _GL_ATTRIBUTE_CONST
#endif
#ifndef SIZE_MAX
# define SIZE_MAX ((size_t)-1)
#endif
#ifndef XNMALLOC
# define XNMALLOC(n, type) ((type *) malloc ((n) * sizeof (type)))
#endif
#ifndef XMALLOC
# define XMALLOC(type) ((type *) malloc (sizeof (type)))
#endif
"""


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _target_for_placeholder(
    targets: List[Any],
    slot_id: str,
    fallback_idx: int,
) -> str:
    m = re.fullmatch(r"slot_(\d+)", str(slot_id or ""))
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(targets):
            return str(targets[idx] or "")
    if 0 <= fallback_idx < len(targets):
        return str(targets[fallback_idx] or "")
    return ""


def render_teacher(record: Dict[str, Any]) -> str:
    skeleton = str(record.get("sft_skeleton_text_pretty") or "")
    targets = list(record.get("sft_teacher_targets") or [])
    occurrence = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal occurrence
        out = _target_for_placeholder(targets, m.group(2), occurrence)
        occurrence += 1
        return out

    rendered = FILL_RE.sub(repl, skeleton)
    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
    return rendered.strip() + "\n"


def _count_semicolons_outside_strings(text: str) -> int:
    n = 0
    in_s = False
    in_c = False
    esc = False
    for ch in str(text or ""):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == ";":
            n += 1
    return n


def _iter_control_conditions(text: str) -> Iterable[Tuple[str, str]]:
    s = str(text or "")
    for m in re.finditer(r"\b(if|while|for|switch)\s*\(", s):
        kw = m.group(1)
        start = s.find("(", m.end() - 1)
        if start < 0:
            continue
        depth = 0
        in_s = False
        in_c = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if in_s:
                if ch == '"':
                    in_s = False
                continue
            if in_c:
                if ch == "'":
                    in_c = False
                continue
            if ch == '"':
                in_s = True
            elif ch == "'":
                in_c = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    yield kw, s[start + 1 : i].strip()
                    break


def structural_reasons(rendered: str) -> List[str]:
    reasons: List[str] = []
    s = str(rendered or "")
    if "<FILL_" in s:
        reasons.append("unfilled_placeholder")
    if re.search(r"\b(?:if|while|for|switch)\s*\(\s*\)", s):
        reasons.append("empty_control_condition")
    if re.search(r"\b(?:if|while|for|switch)\s*\(\s*(?:if|while|for|switch)\s*\(", s):
        reasons.append("nested_control_header_in_condition")
    for kw, cond in _iter_control_conditions(s):
        if kw != "for" and _count_semicolons_outside_strings(cond) > 0:
            reasons.append(f"{kw}_condition_has_semicolon")
            break
    if re.search(r"\{\s*return\b[^{};]*;?\s*\{\s*return\b", s):
        reasons.append("glued_return_blocks")
    if re.search(r"\breturn\s+return\b", s):
        reasons.append("double_return")
    if re.search(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b", s, flags=re.M):
        reasons.append("preprocessor_directive_in_teacher")
    par = s.count("(") - s.count(")")
    brk = s.count("[") - s.count("]")
    brc = s.count("{") - s.count("}")
    if par or brk or brc:
        reasons.append(f"delimiter_balance_par{par}_brk{brk}_brc{brc}")
    return reasons


def classify_compile(stderr: str, structural: List[str]) -> str:
    if structural:
        return "structural_teacher_bug"
    e = str(stderr or "")
    if not e:
        return "ok"
    if "unknown type name" in e:
        return "harness_unknown_type"
    if "has incomplete type" in e or "incomplete type" in e:
        return "harness_unknown_type"
    if "use of undeclared identifier" in e or "undeclared identifier" in e:
        return "harness_undeclared_identifier"
    if "call to undeclared function" in e:
        return "harness_undeclared_function"
    if "no member named" in e or "incomplete definition of type" in e:
        return "harness_incomplete_struct"
    if "redefinition of" in e:
        return "teacher_redefinition"
    if "expected" in e or "extraneous" in e or "invalid suffix" in e:
        return "teacher_syntax"
    if "too few arguments" in e or "too many arguments" in e:
        return "teacher_call_signature_mismatch"
    if "void function" in e or "non-void function" in e:
        return "teacher_return_type_mismatch"
    return "other_compile_failure"


def first_error_lines(stderr: str, max_lines: int = 8) -> str:
    lines = []
    for line in str(stderr or "").splitlines():
        if "error:" in line or "warning:" in line:
            lines.append(line.strip())
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def resolve_compiler(cc: str) -> Tuple[str, str]:
    if cc and cc != "auto":
        resolved = shutil.which(cc)
        if resolved:
            return resolved, ""
        p = Path(cc)
        if p.is_file():
            return str(p), ""
        return "", f"compiler_not_found: {cc}"
    for cand in ("clang", "cc", "gcc"):
        resolved = shutil.which(cand)
        if resolved:
            return resolved, ""
    return "", "compiler_not_found: tried clang, cc, gcc"


def _compiler_kind(cc: str) -> str:
    name = Path(str(cc or "")).name.lower()
    if "clang" in name:
        return "clang"
    if "gcc" in name or name in {"cc", "c89", "c99"}:
        return "gcc"
    return "unknown"


def compile_one(cc: str, c_path: Path, std: str) -> Tuple[bool, str]:
    cmd = [
        cc,
        "-fsyntax-only",
        "-x",
        "c",
        f"-std={std}",
    ]
    kind = _compiler_kind(cc)
    if kind == "clang":
        cmd.extend(
            [
                "-Wno-everything",
                "-Wno-error=implicit-function-declaration",
                "-Wno-error=incompatible-function-pointer-types",
            ]
        )
    elif kind == "gcc":
        cmd.extend(
            [
                "-w",
                "-Wno-error=implicit-function-declaration",
                "-Wno-error=incompatible-pointer-types",
            ]
        )
    else:
        cmd.append("-w")
    cmd.append(str(c_path))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0, (proc.stdout + proc.stderr)


def safe_name(name: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "func")).strip("_")
    return s[:80] or "func"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render and compile-audit Module-C teacher JSONL")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/structlift_modulec_teacher_compile"))
    ap.add_argument("--cc", default="auto", help="Compiler executable, or auto to try clang, cc, gcc")
    ap.add_argument("--std", default="gnu11")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--prelude", type=Path, default=None, help="Optional C prelude file prepended before each rendered teacher")
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prelude = DEFAULT_PRELUDE
    if args.prelude is not None:
        prelude = args.prelude.expanduser().read_text(encoding="utf-8")
    cc, cc_err = resolve_compiler(str(args.cc))
    if cc_err:
        print("== teacher compile audit ==")
        print("jsonl:", args.jsonl.expanduser().resolve())
        print("out_dir:", out_dir)
        print("records:", 0)
        print("error:", cc_err)
        print("hint: install/load clang or rerun with --cc gcc / --cc cc when available")
        raise SystemExit(2)
    print("compiler:", cc)

    n = 0
    structural_ok = 0
    compile_ok = 0
    categories: "collections.Counter[str]" = collections.Counter()
    structural_counter: "collections.Counter[str]" = collections.Counter()
    examples: List[Dict[str, Any]] = []

    for line_no, rec in _iter_jsonl(args.jsonl.expanduser().resolve(), args.limit):
        n += 1
        rendered = render_teacher(rec)
        reasons = structural_reasons(rendered)
        if not reasons:
            structural_ok += 1
        structural_counter.update(reasons)

        c_path = out_dir / f"teacher_{line_no:05d}_{safe_name(rec.get('func_name'))}.c"
        c_path.write_text(prelude + "\n\n" + rendered, encoding="utf-8")
        ok, stderr = compile_one(cc, c_path, args.std)
        if ok:
            compile_ok += 1
            cat = "ok"
        else:
            cat = classify_compile(stderr, reasons)
        categories[cat] += 1
        if cat != "ok" and len(examples) < args.max_examples:
            examples.append(
                {
                    "line": line_no,
                    "func": rec.get("func_name"),
                    "category": cat,
                    "structural": reasons,
                    "c_path": str(c_path),
                    "errors": first_error_lines(stderr),
                    "rendered_head": rendered[:1200],
                }
            )

    print("== teacher compile audit ==")
    print("jsonl:", args.jsonl.expanduser().resolve())
    print("out_dir:", out_dir)
    print("records:", n)
    print("structural_ok:", f"{structural_ok}/{n}", f"({100.0 * structural_ok / max(1, n):.1f}%)")
    print("clang_compile_ok:", f"{compile_ok}/{n}", f"({100.0 * compile_ok / max(1, n):.1f}%)")
    print("categories:", dict(categories))
    print("structural_reasons:", dict(structural_counter))
    print("\n== examples ==")
    for ex in examples:
        print(json.dumps(ex, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

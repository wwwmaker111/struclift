#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From CMake Ninja's compile_commands.json, emit one .ll per translation unit
(same flags as the real build). For use with build_binskel_dataset_md.py --llvm-ir-dir.

Usage:
  python scripts/emit_ir_from_compile_commands.py <build_dir> <ir_out_dir>

Skips test/fuzz/googletest paths by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SKIP_SUBSTR = (
    "/test/",
    "/fuzz/",
    "googletest",
    "googlemock",
    "/third_party/googletest",
    "/util/fipstool/",
)


def should_skip(src: str) -> bool:
    s = src.replace("\\", "/")
    return any(x in s for x in SKIP_SUBSTR)


def ll_name_for_source(src_path: Path) -> str:
    h = hashlib.sha256(os.path.normpath(str(src_path)).encode("utf-8", errors="replace")).hexdigest()[:14]
    return f"{h}_{src_path.stem}.ll"


def _strip_ccache(argv: list[str]) -> list[str]:
    out = list(argv)
    while len(out) > 0 and os.path.basename(out[0]).lower() == "ccache":
        out = out[1:]
    return out


def patch_cmd_to_emit_llvm(argv: list[str], out_ll: Path) -> list[str] | None:
    """Turn a compile argv into clang/clang++ -emit-llvm -S ... -o out.ll"""
    if not argv:
        return None
    argv = _strip_ccache(argv)
    if not argv:
        return None
    compiler = argv[0]
    low = compiler.lower()
    base = os.path.basename(low)
    if "clang" not in base and not (base.endswith("clang++") or base.endswith("clang")):
        return None

    rest: list[str] = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-c":
            i += 1
            continue
        if a == "-o" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("-o") and len(a) > 2:
            i += 1
            continue
        rest.append(a)
        i += 1

    src = None
    src_strip_idx = -1
    for j in range(len(rest) - 1, -1, -1):
        if rest[j].startswith("-"):
            continue
        if any(rest[j].endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx")):
            src = rest[j]
            src_strip_idx = j
            break
    if not src or src_strip_idx < 0:
        return None

    flags_only = rest[:src_strip_idx] + rest[src_strip_idx + 1 :]
    final = [compiler]
    if "-g" not in flags_only and "-gline-tables-only" not in flags_only and "-gdwarf" not in " ".join(flags_only):
        final.append("-g3")
    final.extend(flags_only)
    final.extend(["-emit-llvm", "-S", src, "-o", str(out_ll)])
    return final


def cmd_argv(entry: dict) -> tuple[list[str], Path]:
    directory = Path(entry.get("directory") or ".")
    if "arguments" in entry and entry["arguments"]:
        argv = list(entry["arguments"])
    elif "command" in entry and entry["command"]:
        argv = shlex.split(entry["command"])
    else:
        return [], directory
    return argv, directory


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit .ll files from compile_commands.json")
    ap.add_argument("build_dir", type=Path, help="Directory containing compile_commands.json")
    ap.add_argument("ir_out", type=Path, help="Output directory for .ll files")
    args = ap.parse_args()

    cc_json = args.build_dir / "compile_commands.json"
    if not cc_json.is_file():
        print(f"ERROR: missing {cc_json}", file=sys.stderr)
        return 1

    args.ir_out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(cc_json.read_text(encoding="utf-8"))
    ok = 0
    fail = 0
    for entry in entries:
        src = entry.get("file")
        if not src or should_skip(src):
            continue
        sp = Path(src)
        if sp.suffix.lower() not in (".c", ".cc", ".cpp", ".cxx"):
            continue
        argv, workdir = cmd_argv(entry)
        if not argv:
            fail += 1
            continue
        out_ll = args.ir_out / ll_name_for_source(sp)
        new_argv = patch_cmd_to_emit_llvm(argv, out_ll)
        if not new_argv:
            fail += 1
            continue
        r = subprocess.run(
            new_argv,
            cwd=str(workdir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0 and out_ll.is_file() and out_ll.stat().st_size > 0:
            ok += 1
        else:
            fail += 1
            if out_ll.is_file():
                out_ll.unlink(missing_ok=True)

    print(f"emit_ir_from_compile_commands: ok={ok} fail={fail} -> {args.ir_out}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

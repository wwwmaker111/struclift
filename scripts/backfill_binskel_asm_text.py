#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill readable per-basic-block assembly text into existing binskel JSONL.

Older C-stage JSONL files only stored hashed ``bin_opcodes`` and operand ids.
Those ids are fine for Module A/B embeddings, but they are poor textual context
for Module C.  This script reopens each sample's ``binary`` ELF, locates
``func_name`` in the symbol table, disassembles it with the same CFG builder,
and writes:

    bin_instr_text: list[list[str]]

The output preserves all existing keys.  If the reconstructed CFG does not
match the original ``bin_block_lengths``, the record is still written but no
``bin_instr_text`` is attached by default, avoiding silently misaligned slot
context.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from elftools.elf.elffile import ELFFile  # type: ignore
except Exception:
    ELFFile = None  # type: ignore

try:
    if ELFFile is not None:
        from build_binskel_dataset_md import build_cfg_for_function, iter_functions
    else:
        build_cfg_for_function = None  # type: ignore
        iter_functions = None  # type: ignore
except Exception:
    if ELFFile is not None:
        from build_binskel_dataset import build_cfg_for_function, iter_functions
    else:
        build_cfg_for_function = None  # type: ignore
        iter_functions = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _BinaryCache:
    def __init__(self, max_open: int = 4, num_opcodes: int = 1024) -> None:
        self.max_open = max(1, int(max_open))
        self.num_opcodes = int(num_opcodes)
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _load(self, binary: Path) -> Dict[str, Any]:
        if ELFFile is None or build_cfg_for_function is None or iter_functions is None:
            return {"objdump_only": True}
        key = str(binary)
        if key in self._cache:
            obj = self._cache.pop(key)
            self._cache[key] = obj
            return obj
        with binary.open("rb") as f:
            data = f.read()
        import io

        ef = ELFFile(io.BytesIO(data))
        text = ef.get_section_by_name(".text")
        if text is None:
            raise RuntimeError("missing .text section")
        funcs: Dict[str, Tuple[int, int]] = {}
        for name, addr, size in iter_functions(ef):
            if name and int(size) > 0:
                funcs.setdefault(str(name), (int(addr), int(size)))
        obj = {
            "data": data,
            "text_bytes": text.data(),
            "base_addr": int(text["sh_addr"]),
            "funcs": funcs,
        }
        self._cache[key] = obj
        while len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return obj

    def disasm_func(self, binary: str, func_name: str) -> Tuple[Optional[List[List[str]]], str]:
        if not binary or not func_name:
            return None, "missing binary or func_name"
        p = Path(binary)
        if not p.is_file():
            return None, f"binary not found: {binary}"
        if ELFFile is None or build_cfg_for_function is None or iter_functions is None:
            return None, "pyelftools unavailable; objdump fallback requires block lengths"
        obj = self._load(p)
        loc = obj["funcs"].get(func_name)
        if loc is None:
            return None, f"symbol not found: {func_name}"
        addr, size = loc
        base = obj["base_addr"]
        start = addr - base
        text_bytes: bytes = obj["text_bytes"]
        if start < 0 or start + size > len(text_bytes):
            return None, f"function range outside .text: addr=0x{addr:x} size={size}"
        cfg, _ranges = build_cfg_for_function(
            text_bytes[start : start + size],
            addr,
            func_name,
            num_opcodes=self.num_opcodes,
        )
        if cfg.num_blocks == 0:
            return None, "empty CFG"
        return (
            [[str(inst.raw_text or "").strip() for inst in bb.instructions] for bb in cfg.blocks],
            "",
        )


def _parse_objdump_func_instructions(text: str, func_name: str) -> Tuple[Optional[List[str]], str]:
    funcs = _parse_objdump_all_functions(text)
    out = funcs.get(func_name) or funcs.get(func_name.split("@", 1)[0])
    if out:
        return out, ""
    return None, f"objdump symbol not found or empty: {func_name}"


def _parse_objdump_all_functions(text: str) -> Dict[str, List[str]]:
    header_re = re.compile(r"^\s*[0-9a-fA-F]+\s+<(.+)>:\s*$")
    instr_re = re.compile(r"^\s*[0-9a-fA-F]+:\s*(?:[0-9a-fA-F]{2}\s+)+\s*(\S.*)$")
    cur_names: List[str] = []
    funcs: Dict[str, List[str]] = {}
    for line in text.splitlines():
        hm = header_re.match(line)
        if hm:
            name = hm.group(1)
            bare = name.split("@", 1)[0]
            cur_names = [name]
            if bare != name:
                cur_names.append(bare)
            for nm in cur_names:
                funcs.setdefault(nm, [])
            continue
        if not cur_names:
            continue
        m = instr_re.match(line)
        if not m:
            continue
        asm = re.sub(r"\s+", " ", m.group(1).strip())
        if asm:
            for nm in cur_names:
                funcs.setdefault(nm, []).append(asm)
    return funcs


class _ObjdumpCache:
    def __init__(self, max_open: int = 4, objdump_bin: str = "objdump") -> None:
        self.max_open = max(1, int(max_open))
        self.objdump_bin = objdump_bin
        self._cache: "OrderedDict[str, Dict[str, List[str]]]" = OrderedDict()

    def funcs_for_binary(self, binary: str) -> Tuple[Optional[Dict[str, List[str]]], str]:
        key = str(binary)
        if key in self._cache:
            obj = self._cache.pop(key)
            self._cache[key] = obj
            return obj, ""
        try:
            proc = subprocess.run(
                [self.objdump_bin, "-d", "-M", "intel", binary],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
        except FileNotFoundError:
            return None, f"{self.objdump_bin} not found"
        if proc.returncode != 0 and not proc.stdout:
            return None, proc.stderr.strip() or f"{self.objdump_bin} failed rc={proc.returncode}"
        funcs = _parse_objdump_all_functions(proc.stdout)
        self._cache[key] = funcs
        while len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return funcs, ""


def _objdump_backfill_by_lengths(
    binary: str,
    func_name: str,
    block_lengths: Any,
    *,
    objdump_bin: str = "objdump",
    objdump_cache: Optional[_ObjdumpCache] = None,
) -> Tuple[Optional[List[List[str]]], str, str]:
    if not isinstance(block_lengths, list) or not block_lengths:
        return None, "missing bin_block_lengths for objdump fallback", "error"
    if objdump_cache is None:
        objdump_cache = _ObjdumpCache(max_open=1, objdump_bin=objdump_bin)
    funcs, err = objdump_cache.funcs_for_binary(binary)
    if funcs is None:
        return None, err, "error"
    insns = funcs.get(func_name) or funcs.get(func_name.split("@", 1)[0])
    if not insns:
        return None, f"objdump symbol not found or empty: {func_name}", "error"
    lengths = [int(x) for x in block_lengths]
    need = sum(lengths)
    status = "ok"
    if need != len(insns):
        status = "truncated" if len(insns) > need else "short"
    out: List[List[str]] = []
    off = 0
    for n in lengths:
        out.append(insns[off : min(off + n, len(insns))])
        off += n
    msg = "" if status == "ok" else f"objdump instruction count mismatch json={need} disasm={len(insns)}"
    return out, msg, status


def _same_block_lengths(instr_text: List[List[str]], block_lengths: Any) -> bool:
    if not isinstance(block_lengths, list):
        return False
    got = [len(x) for x in instr_text]
    exp = [int(x) for x in block_lengths]
    return got == exp


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill bin_instr_text into binskel JSONL")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-opcodes", type=int, default=1024)
    ap.add_argument("--max-open-binaries", type=int, default=4)
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--objdump", type=str, default="objdump")
    ap.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Write bin_instr_text even when block lengths differ from JSONL.",
    )
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    if not inp.is_file():
        logger.error("input not found: %s", inp)
        sys.exit(1)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cache = _BinaryCache(max_open=args.max_open_binaries, num_opcodes=args.num_opcodes)
    objdump_cache = _ObjdumpCache(max_open=args.max_open_binaries, objdump_bin=args.objdump)
    n_ok = 0
    n_mismatch = 0
    n_err = 0
    n_total = 0
    examples: List[str] = []

    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for li, line in enumerate(fin):
            if args.max_lines is not None and li >= args.max_lines:
                break
            raw = line.strip()
            if not raw:
                continue
            n_total += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                n_err += 1
                continue
            instr_text, err = cache.disasm_func(
                str(rec.get("binary") or ""),
                str(rec.get("func_name") or ""),
            )
            fallback_status: Optional[str] = None
            if instr_text is None and "pyelftools unavailable" in err:
                instr_text, err, fallback_status = _objdump_backfill_by_lengths(
                    str(rec.get("binary") or ""),
                    str(rec.get("func_name") or ""),
                    rec.get("bin_block_lengths"),
                    objdump_bin=args.objdump,
                    objdump_cache=objdump_cache,
                )
            status = "ok"
            if instr_text is None:
                n_err += 1
                status = "error"
                if len(examples) < 8:
                    examples.append(f"line {li + 1} {rec.get('func_name', '?')}: {err}")
            elif not _same_block_lengths(instr_text, rec.get("bin_block_lengths")):
                n_mismatch += 1
                status = fallback_status or "mismatch"
                if args.allow_mismatch or status in ("truncated", "short"):
                    rec["bin_instr_text"] = instr_text
                    n_ok += 1
                if len(examples) < 8:
                    exp = rec.get("bin_block_lengths")
                    got = [len(x) for x in instr_text]
                    examples.append(
                        f"line {li + 1} {rec.get('func_name', '?')}: block length mismatch "
                        f"json={exp} disasm={got}"
                    )
            else:
                rec["bin_instr_text"] = instr_text
                n_ok += 1
            rec["bin_instr_text_backfill_status"] = status
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_total % 1000 == 0:
                logger.info(
                    "processed=%d ok=%d mismatch=%d err=%d",
                    n_total,
                    n_ok,
                    n_mismatch,
                    n_err,
                )

    logger.info(
        "done: total=%d ok=%d mismatch=%d err=%d -> %s",
        n_total,
        n_ok,
        n_mismatch,
        n_err,
        outp,
    )
    if examples:
        logger.info("examples:\n  %s", "\n  ".join(examples))


if __name__ == "__main__":
    main()

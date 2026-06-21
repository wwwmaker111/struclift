#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrich the teacher-smoke rows with binary-derived FunctionIR/SSA.

This is the 100-row practical bridge for Module-B v3:

* read teacher ``input_line`` values;
* preserve the source JSONL line numbering;
* for referenced rows, try to rebuild FunctionIR from the real ELF function;
* emit register-level SSA / def-use and instruction provenance from Capstone;
* if the ELF/function cannot be resolved, fall back to the auditable CFG proxy
  enrichment instead of silently pretending real SSA exists.

The produced SSA is binary-disassembly register SSA.  It is stronger than the
CFG-only proxy because it comes from real instruction addresses and register
read/write sets, but it is still not a full VEX/P-code memory SSA lifter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_CALL, CS_GRP_JUMP, CS_GRP_RET, CS_OP_IMM
except Exception as exc:  # pragma: no cover - exercised on the server if missing.
    Cs = None  # type: ignore[assignment]
    CS_ARCH_X86 = CS_MODE_64 = CS_GRP_CALL = CS_GRP_JUMP = CS_GRP_RET = CS_OP_IMM = -1  # type: ignore[assignment]
    _CAPSTONE_IMPORT_ERROR = exc
else:
    _CAPSTONE_IMPORT_ERROR = None

try:
    from elftools.elf.elffile import ELFFile
except Exception as exc:  # pragma: no cover - exercised on the server if missing.
    ELFFile = None  # type: ignore[assignment]
    _ELFTOOLS_IMPORT_ERROR = exc
else:
    _ELFTOOLS_IMPORT_ERROR = None

from enrich_module_b_v3_function_ir import enrich_row as proxy_enrich_row


STT_FUNC = 2


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


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _as_int_list(x: Any) -> List[int]:
    out: List[int] = []
    for v in _as_list(x):
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / max(1, d):.1f}%)"


def _load_teacher_targets(path: Path, max_examples: int) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for _line_no, teacher in iter_jsonl(path, max_examples):
        src_line = _safe_int(teacher.get("input_line"), 0)
        if src_line > 0:
            out[src_line] = teacher
    return out


def _parse_root_maps(raw_items: Sequence[str]) -> List[Tuple[str, str]]:
    maps: List[Tuple[str, str]] = []
    for raw in raw_items:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"--binary-root-map must be OLD=NEW, got {raw!r}")
        old, new = raw.split("=", 1)
        old = old.strip()
        new = new.strip()
        if old:
            maps.append((old, new))
    maps.sort(key=lambda x: len(x[0]), reverse=True)
    return maps


_BINARY_KEYWORDS = (
    "binary",
    "binary_path",
    "bin_path",
    "elf",
    "elf_path",
    "exe",
    "exe_path",
    "executable",
    "executable_path",
    "artifact",
    "artifact_path",
    "program",
    "program_path",
)


def _resolve_path(raw: Any, root_maps: Sequence[Tuple[str, str]], search_roots: Sequence[Path] = ()) -> Optional[Path]:
    if raw in (None, "", [], {}):
        return None
    text = str(raw)
    candidates = [Path(text).expanduser()]
    for old, new in root_maps:
        if text.startswith(old):
            candidates.append(Path(new + text[len(old):]).expanduser())
    raw_path = Path(text)
    if not raw_path.is_absolute():
        for root in search_roots:
            candidates.append(Path(root).expanduser() / raw_path)
    else:
        # Also try suffix joins for stale absolute paths.  This handles rows
        # produced on a different mount, e.g. /old/root/proj/bin/foo while the
        # current host has /nasdata/WQM/.../proj/bin/foo.
        parts = raw_path.parts
        for root in search_roots:
            root = Path(root).expanduser()
            for keep in range(1, min(len(parts), 8) + 1):
                candidates.append(root.joinpath(*parts[-keep:]))
    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except OSError:
            continue
    return None


def _iter_binary_path_values(obj: Any, *, max_depth: int = 5, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if max_depth < 0:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            skey = str(key)
            lkey = skey.lower()
            path = f"{prefix}.{skey}" if prefix else skey
            if any(token in lkey for token in _BINARY_KEYWORDS):
                yield path, value
            elif any(token in lkey for token in ("path", "file")) and isinstance(value, str):
                text = value.lower()
                if any(token in text for token in ("/bin/", ".elf", ".so", ".out", "o0", "o1", "o2", "o3")):
                    yield path, value
            if isinstance(value, (dict, list)):
                yield from _iter_binary_path_values(value, max_depth=max_depth - 1, prefix=path)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj[:64]):
            if isinstance(value, (dict, list)):
                yield from _iter_binary_path_values(value, max_depth=max_depth - 1, prefix=f"{prefix}[{idx}]")


def _row_binary_path(
    row: Dict[str, Any],
    teacher: Optional[Dict[str, Any]],
    root_maps: Sequence[Tuple[str, str]],
    search_roots: Sequence[Path] = (),
) -> Optional[Path]:
    seen = set()
    for source in (row, teacher or {}):
        if not isinstance(source, dict):
            continue
        for key, raw in _iter_binary_path_values(source):
            marker = (key, str(raw))
            if marker in seen:
                continue
            seen.add(marker)
            path = _resolve_path(raw, root_maps, search_roots)
            if path is not None:
                return path
    return None


def _row_binary_path_debug(
    row: Dict[str, Any],
    teacher: Optional[Dict[str, Any]],
    *,
    max_items: int = 20,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for label, source in (("source", row), ("teacher", teacher or {})):
        if not isinstance(source, dict):
            continue
        for key, raw in _iter_binary_path_values(source):
            out.append({"source": label, "key": key, "value": str(raw)[:300]})
            if len(out) >= max_items:
                return out
    return out


def _legacy_row_binary_path(row: Dict[str, Any], root_maps: Sequence[Tuple[str, str]]) -> Optional[Path]:
    for key in ("binary", "binary_path", "bin_path", "elf", "elf_path", "exe_path"):
        path = _resolve_path(row.get(key), root_maps)
        if path is not None:
            return path
    return None


def _row_func_name(row: Dict[str, Any], teacher: Optional[Dict[str, Any]]) -> Optional[str]:
    for raw in (
        row.get("func_name"),
        row.get("function"),
        row.get("name"),
        (teacher or {}).get("func_name"),
        (teacher or {}).get("function"),
        (teacher or {}).get("name"),
    ):
        if raw not in (None, "", [], {}):
            return str(raw)
    return None


def _iter_functions(elffile: Any) -> Iterable[Tuple[str, int, int]]:
    symtab = elffile.get_section_by_name(".symtab") or elffile.get_section_by_name(".dynsym")
    if symtab is None:
        return
    for sym in symtab.iter_symbols():
        try:
            st_info = sym.entry["st_info"]
            st_type = getattr(st_info, "type", None)
            if st_type is None and isinstance(st_info, dict):
                st_type = st_info.get("type")
            is_func = st_type == "STT_FUNC" or (isinstance(st_type, int) and (st_type & 0xF) == STT_FUNC)
            size = int(sym.entry["st_size"])
            value = int(sym.entry["st_value"])
            if is_func and size > 0 and value > 0:
                yield str(sym.name), value, size
        except Exception:
            continue


def _norm_symbol(name: str) -> str:
    return name.strip().lstrip("_")


def _find_function(elffile: Any, func_name: str) -> Optional[Tuple[str, int, int]]:
    funcs = list(_iter_functions(elffile))
    if not funcs:
        return None
    for name, addr, size in funcs:
        if name == func_name:
            return name, addr, size
    target = _norm_symbol(func_name)
    for name, addr, size in funcs:
        if _norm_symbol(name) == target:
            return name, addr, size
    for name, addr, size in funcs:
        if name.endswith("." + func_name) or name.endswith("_" + func_name):
            return name, addr, size
    return None


class ElfFunctionCache:
    """Cache ELF symbol tables and .text bytes for repeated function extraction."""

    def __init__(self) -> None:
        self.md = _get_capstone()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats: Counter = Counter()

    def _load(self, binary: Path) -> Dict[str, Any]:
        key = str(binary)
        if key in self.cache:
            self.stats["elf_cache_hit"] += 1
            return self.cache[key]
        self.stats["elf_cache_miss"] += 1
        with binary.open("rb") as f:
            elffile = ELFFile(f)
            text_sec = elffile.get_section_by_name(".text")
            if text_sec is None:
                raise RuntimeError("elf_has_no_text_section")
            funcs = list(_iter_functions(elffile))
            text_bytes = text_sec.data()
            info = {
                "text_base": int(text_sec["sh_addr"]),
                "text_bytes": text_bytes,
                "funcs": funcs,
            }
        self.cache[key] = info
        return info

    def find_function(self, binary: Path, func_name: str) -> Optional[Tuple[str, int, int]]:
        info = self._load(binary)
        funcs = list(info.get("funcs") or [])
        if not funcs:
            return None
        for name, addr, size in funcs:
            if name == func_name:
                return name, addr, size
        target = _norm_symbol(func_name)
        for name, addr, size in funcs:
            if _norm_symbol(name) == target:
                return name, addr, size
        for name, addr, size in funcs:
            if name.endswith("." + func_name) or name.endswith("_" + func_name):
                return name, addr, size
        return None

    def disasm_function(self, binary: Path, addr: int, size: int) -> Tuple[List[Any], str]:
        info = self._load(binary)
        base_addr = int(info["text_base"])
        text_bytes = info["text_bytes"]
        start = int(addr) - base_addr
        if start < 0 or start + int(size) > len(text_bytes):
            return [], "function_address_out_of_text_range"
        code = text_bytes[start:start + int(size)]
        insns = list(self.md.disasm(code, int(addr)))
        if not insns:
            return [], "capstone_disassembly_empty"
        return insns, "ok"


def _get_capstone() -> Any:
    if Cs is None:
        raise RuntimeError(f"capstone unavailable: {_CAPSTONE_IMPORT_ERROR}")
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    return md


def _insn_reg_access(md: Any, insn: Any) -> Tuple[List[str], List[str]]:
    try:
        reads, writes = insn.regs_access()
    except Exception:
        reads, writes = [], []
    read_names = [str(md.reg_name(r) or f"reg_{r}") for r in reads]
    write_names = [str(md.reg_name(r) or f"reg_{r}") for r in writes]
    return sorted(set(read_names)), sorted(set(write_names))


def _op_summary(insn: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for op in getattr(insn, "operands", []) or []:
        item: Dict[str, Any] = {"type": int(getattr(op, "type", 0) or 0)}
        try:
            if op.type == CS_OP_IMM:
                item["imm"] = int(op.imm)
            elif hasattr(op, "reg") and op.reg:
                item["reg"] = int(op.reg)
            elif hasattr(op, "mem"):
                item["mem"] = {
                    "base": int(getattr(op.mem, "base", 0) or 0),
                    "index": int(getattr(op.mem, "index", 0) or 0),
                    "scale": int(getattr(op.mem, "scale", 0) or 0),
                    "disp": int(getattr(op.mem, "disp", 0) or 0),
                }
        except Exception:
            pass
        out.append(item)
    return out


def _build_blocks(insns: Sequence[Any]) -> Tuple[List[List[Any]], Dict[int, int], List[Tuple[int, int, int]]]:
    if not insns:
        return [], {}, []
    addr2idx = {int(insn.address): i for i, insn in enumerate(insns)}
    block_starts = {int(insns[0].address)}
    for insn in insns:
        addr = int(insn.address)
        groups = set(getattr(insn, "groups", []) or [])
        ni = addr2idx[addr] + 1
        if CS_GRP_JUMP in groups or CS_GRP_RET in groups:
            if ni < len(insns):
                block_starts.add(int(insns[ni].address))
            ops = getattr(insn, "operands", []) or []
            if CS_GRP_JUMP in groups and ops and ops[0].type == CS_OP_IMM:
                target = int(ops[0].imm)
                if target in addr2idx:
                    block_starts.add(target)

    sorted_starts = sorted(block_starts)
    addr2block: Dict[int, int] = {}
    blocks: List[List[Any]] = []
    for b_id, start_addr in enumerate(sorted_starts):
        start_i = addr2idx[start_addr]
        end_i = addr2idx[sorted_starts[b_id + 1]] if b_id + 1 < len(sorted_starts) else len(insns)
        block = list(insns[start_i:end_i])
        blocks.append(block)
        for insn in block:
            addr2block[int(insn.address)] = b_id

    edges_set = set()
    for idx, insn in enumerate(insns):
        addr = int(insn.address)
        b_id = addr2block[addr]
        ni = idx + 1
        is_last = ni >= len(insns) or addr2block.get(int(insns[ni].address), -1) != b_id
        if not is_last:
            continue
        groups = set(getattr(insn, "groups", []) or [])
        if CS_GRP_RET in groups:
            continue
        if CS_GRP_CALL in groups:
            if ni < len(insns):
                edges_set.add((b_id, addr2block[int(insns[ni].address)], 3))
            continue
        if CS_GRP_JUMP in groups:
            is_uncond = str(insn.mnemonic) in {"jmp", "ljmp"}
            ops = getattr(insn, "operands", []) or []
            if ops and ops[0].type == CS_OP_IMM:
                target = int(ops[0].imm)
                if target in addr2block:
                    tb = addr2block[target]
                    if tb <= b_id:
                        edge_kind = 2
                    elif is_uncond:
                        edge_kind = 1
                    else:
                        edge_kind = 0
                    edges_set.add((b_id, tb, edge_kind))
            if not is_uncond and ni < len(insns):
                edges_set.add((b_id, addr2block[int(insns[ni].address)], 0))
            continue
        if ni < len(insns):
            edges_set.add((b_id, addr2block[int(insns[ni].address)], 1))
    return blocks, addr2block, sorted(edges_set)


def _flatten_bin_instr_text(row: Dict[str, Any]) -> List[str]:
    raw = row.get("bin_instr_text")
    if isinstance(raw, list) and raw and all(isinstance(x, list) for x in raw):
        return [str(y) for block in raw for y in block]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _shape_matches_source(row: Dict[str, Any], blocks: Sequence[Sequence[Any]], edges: Sequence[Tuple[int, int, int]]) -> bool:
    lengths = _as_int_list(row.get("bin_block_lengths"))
    if lengths and lengths != [len(b) for b in blocks]:
        return False
    raw_edges = row.get("bin_edge_index")
    edge_types = _as_int_list(row.get("bin_edge_type"))
    if isinstance(raw_edges, list) and len(raw_edges) == 2 and isinstance(raw_edges[0], list) and isinstance(raw_edges[1], list):
        src_edges = sorted(
            (int(s), int(d), int(edge_types[i]) if i < len(edge_types) else -1)
            for i, (s, d) in enumerate(zip(raw_edges[0], raw_edges[1]))
        )
        got_edges = sorted((int(s), int(d), int(t)) for s, d, t in edges)
        if src_edges != got_edges:
            return False
    return True


def _looks_like_elf(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


class BinarySymbolIndex:
    """Resolve missing binary_path values by symbol name, guarded by CFG shape."""

    def __init__(self, roots: Sequence[Path]) -> None:
        if ELFFile is None:
            raise RuntimeError(f"pyelftools unavailable: {_ELFTOOLS_IMPORT_ERROR}")
        self.roots = [Path(x).expanduser() for x in roots]
        self.symbols: Dict[str, List[Tuple[Path, str, int, int]]] = defaultdict(list)
        self.path_order: Dict[str, int] = {}
        self.shape_cache: Dict[Tuple[str, str, int, int], Tuple[Optional[List[List[Any]]], Optional[List[Tuple[int, int, int]]]]] = {}
        self.stats: Counter = Counter()
        self._build()

    def _iter_elf_candidates(self) -> Iterable[Path]:
        seen = set()
        for root in self.roots:
            if root.is_file():
                candidates = [root]
            elif root.is_dir():
                candidates = []
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = sorted(
                        d for d in dirnames
                        if not d.startswith(".") and d not in {"__pycache__", "autom4te.cache"}
                    )
                    for name in sorted(filenames):
                        candidates.append(Path(dirpath) / name)
            else:
                self.stats["missing_search_root"] += 1
                continue
            for path in candidates:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                self.stats["files_examined"] += 1
                if _looks_like_elf(path):
                    yield path

    def _build(self) -> None:
        for path in self._iter_elf_candidates():
            self.path_order.setdefault(str(path), len(self.path_order))
            self.stats["elf_binaries"] += 1
            try:
                with path.open("rb") as f:
                    elffile = ELFFile(f)
                    for name, addr, size in _iter_functions(elffile):
                        if not name:
                            continue
                        cand = (path.resolve(), str(name), int(addr), int(size))
                        self.symbols[str(name)].append(cand)
                        nname = _norm_symbol(str(name))
                        if nname != str(name):
                            self.symbols[nname].append(cand)
                        self.stats["function_symbols"] += 1
            except Exception:
                self.stats["elf_index_errors"] += 1
        self.stats["symbol_keys"] = len(self.symbols)

    def _sort_key(self, cand: Tuple[Path, str, int, int]) -> Tuple[int, str, str, int, int]:
        path, name, addr, size = cand
        return (
            int(self.path_order.get(str(path), 1_000_000_000)),
            str(path),
            str(name),
            int(addr),
            int(size),
        )

    def candidates(self, func_name: str) -> List[Tuple[Path, str, int, int]]:
        raw: List[Tuple[Path, str, int, int]] = []
        raw.extend(self.symbols.get(str(func_name), []))
        nname = _norm_symbol(str(func_name))
        if nname != str(func_name):
            raw.extend(self.symbols.get(nname, []))
        seen = set()
        out: List[Tuple[Path, str, int, int]] = []
        for cand in raw:
            key = (str(cand[0]), str(cand[1]), int(cand[2]), int(cand[3]))
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        return sorted(out, key=self._sort_key)

    def _candidate_shape(
        self,
        cand: Tuple[Path, str, int, int],
    ) -> Tuple[Optional[List[List[Any]]], Optional[List[Tuple[int, int, int]]]]:
        path, name, addr, size = cand
        key = (str(path), str(name), int(addr), int(size))
        if key in self.shape_cache:
            return self.shape_cache[key]
        try:
            md = _get_capstone()
            with path.open("rb") as f:
                elffile = ELFFile(f)
                text_sec = elffile.get_section_by_name(".text")
                if text_sec is None:
                    self.shape_cache[key] = (None, None)
                    return self.shape_cache[key]
                base_addr = int(text_sec["sh_addr"])
                text_bytes = text_sec.data()
                start = int(addr) - base_addr
                if start < 0 or start + int(size) > len(text_bytes):
                    self.shape_cache[key] = (None, None)
                    return self.shape_cache[key]
                code = text_bytes[start:start + int(size)]
                insns = list(md.disasm(code, int(addr)))
            blocks, _addr2block, edges = _build_blocks(insns)
            self.shape_cache[key] = (blocks, edges)
        except Exception:
            self.stats["shape_rebuild_errors"] += 1
            self.shape_cache[key] = (None, None)
        return self.shape_cache[key]

    def resolve(
        self,
        row: Dict[str, Any],
        teacher: Optional[Dict[str, Any]],
        *,
        require_cfg_shape_match: bool,
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]], str]:
        func_name = _row_func_name(row, teacher)
        if not func_name:
            return None, None, "missing_func_name"
        candidates = self.candidates(func_name)
        if not candidates:
            return None, None, f"binary_recovery_no_symbol_candidate:{func_name}"

        exact_matches: List[Tuple[Path, str, int, int]] = []
        nbb_matches: List[Tuple[Path, str, int, int]] = []
        text_matches: List[Tuple[Path, str, int, int]] = []
        source_text = _flatten_bin_instr_text(row)
        src_nbb = len(_as_int_list(row.get("bin_block_lengths")))
        for cand in candidates:
            blocks, edges = self._candidate_shape(cand)
            if not blocks or edges is None:
                continue
            if src_nbb and len(blocks) == src_nbb:
                nbb_matches.append(cand)
            if _shape_matches_source(row, blocks, edges):
                exact_matches.append(cand)
                if source_text:
                    real_text = [
                        (str(insn.mnemonic) + (" " + str(insn.op_str) if str(insn.op_str) else "")).strip()
                        for block in blocks
                        for insn in block
                    ]
                    if real_text == source_text:
                        text_matches.append(cand)

        chosen: Optional[Tuple[Path, str, int, int]] = None
        method = ""
        if text_matches:
            chosen = sorted(text_matches, key=self._sort_key)[0]
            method = "symbol_plus_instruction_text"
        elif exact_matches:
            chosen = sorted(exact_matches, key=self._sort_key)[0]
            method = "symbol_plus_cfg_shape"
        elif require_cfg_shape_match:
            return None, None, f"binary_recovery_no_cfg_shape_match:{func_name}"
        elif len(candidates) == 1:
            chosen = candidates[0]
            method = "single_symbol_no_shape"
        elif len(nbb_matches) == 1:
            chosen = nbb_matches[0]
            method = "single_nbb_match"
        else:
            return None, None, f"binary_recovery_ambiguous:{func_name}:{len(candidates)}"

        path, symbol_name, addr, size = chosen
        meta = {
            "method": method,
            "binary_path": str(path),
            "func_name": str(func_name),
            "symbol_name": str(symbol_name),
            "symbol_address": int(addr),
            "symbol_size": int(size),
            "candidate_count": int(len(candidates)),
            "cfg_shape_match_count": int(len(exact_matches)),
            "instruction_text_match_count": int(len(text_matches)),
            "nbb_match_count": int(len(nbb_matches)),
        }
        return path, meta, "ok"


def _make_function_ir_from_binary(
    row: Dict[str, Any],
    teacher: Optional[Dict[str, Any]],
    *,
    input_line: int,
    root_maps: Sequence[Tuple[str, str]],
    path_search_roots: Sequence[Path],
    require_cfg_shape_match: bool,
    binary_index: Optional[BinarySymbolIndex] = None,
    elf_cache: Optional[ElfFunctionCache] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if ELFFile is None:
        return None, f"pyelftools_unavailable:{_ELFTOOLS_IMPORT_ERROR}"
    binary = _row_binary_path(row, teacher, root_maps, path_search_roots)
    binary_recovery_meta: Optional[Dict[str, Any]] = None
    if binary is None and binary_index is not None:
        recovered, meta, status = binary_index.resolve(
            row,
            teacher,
            require_cfg_shape_match=require_cfg_shape_match,
        )
        if recovered is not None:
            row = dict(row)
            row["binary_path"] = str(recovered)
            row["module_b_v3_binary_path_recovery"] = dict(meta or {})
            binary = recovered
            binary_recovery_meta = dict(meta or {})
        else:
            return None, status
    if binary is None:
        hints = _row_binary_path_debug(row, teacher, max_items=6)
        if hints:
            keys = ",".join(f"{h['source']}:{h['key']}" for h in hints[:6])
            return None, f"unresolved_binary_path_candidates:{keys}"
        return None, "missing_or_unresolved_binary_path"
    func_name = _row_func_name(row, teacher)
    if not func_name:
        return None, "missing_func_name"

    elf_cache = elf_cache or ElfFunctionCache()
    try:
        match = elf_cache.find_function(binary, func_name)
    except RuntimeError as exc:
        return None, str(exc)
    if match is None:
        return None, f"function_symbol_not_found:{func_name}"
    symbol_name, addr, size = match
    insns, disasm_status = elf_cache.disasm_function(binary, addr, size)
    if disasm_status != "ok":
        return None, disasm_status
    md = elf_cache.md

    blocks, _addr2block, edges = _build_blocks(insns)
    if not blocks:
        return None, "basic_block_rebuild_empty"
    if require_cfg_shape_match and not _shape_matches_source(row, blocks, edges):
        return None, "rebuilt_cfg_shape_mismatch"

    global_index = 0
    basic_blocks: List[Dict[str, Any]] = []
    lifted_instructions: List[Dict[str, Any]] = []
    instruction_provenance: Dict[str, Dict[str, Any]] = {}
    instr_to_bb: Dict[int, int] = {}
    instr_id_by_addr: Dict[int, str] = {}
    instr_ids_by_addr: Dict[int, str] = {}
    block_ranges: List[Tuple[int, int]] = []

    for bb_id, block in enumerate(blocks):
        start_idx = global_index
        instr_ids: List[str] = []
        for local_idx, insn in enumerate(block):
            addr = int(insn.address)
            instr_id = f"bb_{bb_id}:ins_{global_index}"
            instr_ids.append(instr_id)
            instr_to_bb[global_index] = bb_id
            instr_id_by_addr[addr] = instr_id
            instr_ids_by_addr[global_index] = instr_id
            reads, writes = _insn_reg_access(md, insn)
            groups = [int(g) for g in (getattr(insn, "groups", []) or [])]
            record = {
                "lifted_id": f"lir_{global_index}",
                "instruction_id": instr_id,
                "bb_id": int(bb_id),
                "index": int(global_index),
                "local_index": int(local_idx),
                "address": int(addr),
                "size": int(getattr(insn, "size", 0) or 0),
                "bytes": bytes(getattr(insn, "bytes", b"") or b"").hex(),
                "mnemonic": str(insn.mnemonic),
                "op_str": str(insn.op_str),
                "text": (str(insn.mnemonic) + (" " + str(insn.op_str) if str(insn.op_str) else "")).strip(),
                "kind": "x86_disasm",
                "groups": groups,
                "regs_read": reads,
                "regs_write": writes,
                "operands": _op_summary(insn),
                "is_proxy": False,
            }
            lifted_instructions.append(record)
            instruction_provenance[instr_id] = {
                "instruction_id": instr_id,
                "lifted_id": f"lir_{global_index}",
                "bb_id": int(bb_id),
                "global_instruction_index": int(global_index),
                "address": int(addr),
                "size": int(getattr(insn, "size", 0) or 0),
                "source": "binary_capstone",
            }
            global_index += 1
        end_idx = global_index
        block_ranges.append((start_idx, end_idx))
        outgoing = [e for e in edges if e[0] == bb_id]
        terminator = "return" if not outgoing else ("cond_branch" if len(outgoing) >= 2 else "jump")
        start_addr = int(block[0].address)
        end_addr = int(block[-1].address) + int(getattr(block[-1], "size", 0) or 0)
        basic_blocks.append(
            {
                "bb_id": int(bb_id),
                "stable_id": f"bb_{bb_id}",
                "address_range": [start_addr, end_addr],
                "instruction_ids": instr_ids,
                "instruction_index_range": [int(start_idx), int(end_idx)],
                "instruction_count": int(max(0, end_idx - start_idx)),
                "terminator": terminator,
                "has_call": any(CS_GRP_CALL in (getattr(i, "groups", []) or []) for i in block),
                "has_return": any(CS_GRP_RET in (getattr(i, "groups", []) or []) for i in block),
                "side_effect_summary": [],
            }
        )

    ssa_values: List[Dict[str, Any]] = []
    def_use_edges: List[Dict[str, Any]] = []
    instruction_uses: Dict[str, List[str]] = defaultdict(list)
    instruction_defs: Dict[str, List[str]] = defaultdict(list)
    reg_version: Dict[str, int] = {}
    current_ssa: Dict[str, str] = {}

    def entry_value(reg: str) -> str:
        sid = f"{reg}@entry"
        if sid not in current_ssa.values():
            ssa_values.append(
                {
                    "ssa_id": sid,
                    "kind": "register_entry",
                    "register": reg,
                    "defined_by_instruction": None,
                    "bb_id": None,
                    "uses": [],
                    "is_proxy": False,
                }
            )
        current_ssa.setdefault(reg, sid)
        return current_ssa[reg]

    for rec in lifted_instructions:
        instr_id = str(rec["instruction_id"])
        bb_id = int(rec["bb_id"])
        for reg in rec.get("regs_read") or []:
            sid = current_ssa.get(reg) or entry_value(reg)
            instruction_uses[instr_id].append(sid)
            def_use_edges.append({"def": sid, "use": instr_id, "kind": "reg_read", "register": reg, "is_proxy": False})
        for reg in rec.get("regs_write") or []:
            reg_version[reg] = reg_version.get(reg, 0) + 1
            sid = f"{reg}@{reg_version[reg]}"
            current_ssa[reg] = sid
            instruction_defs[instr_id].append(sid)
            ssa_values.append(
                {
                    "ssa_id": sid,
                    "kind": "register",
                    "register": reg,
                    "defined_by_instruction": instr_id,
                    "bb_id": int(bb_id),
                    "uses": [],
                    "is_proxy": False,
                }
            )
            def_use_edges.append({"def": instr_id, "use": sid, "kind": "reg_write", "register": reg, "is_proxy": False})

    ssa_by_id = {str(v.get("ssa_id")): v for v in ssa_values if isinstance(v, dict)}
    for edge in def_use_edges:
        sid = edge.get("def")
        if isinstance(sid, str) and sid in ssa_by_id and str(edge.get("kind")) == "reg_read":
            ssa_by_id[sid].setdefault("uses", []).append(edge.get("use"))

    branch_metadata: List[Dict[str, Any]] = []
    branch_by_bb: Dict[int, Dict[str, Any]] = {}
    for bb_id, block in enumerate(blocks):
        outgoing = [e for e in edges if e[0] == bb_id]
        if not outgoing:
            continue
        term = block[-1]
        term_idx = block_ranges[bb_id][1] - 1
        term_id = instr_ids_by_addr.get(term_idx) or instr_id_by_addr.get(int(term.address))
        cond_ssa = None
        if len(outgoing) >= 2 and term_id:
            for sid in instruction_uses.get(term_id, []):
                reg = str((ssa_by_id.get(sid) or {}).get("register") or "")
                if reg in {"rflags", "eflags", "flags"} or "flag" in reg:
                    cond_ssa = sid
                    break
            if cond_ssa is None:
                uses = instruction_uses.get(term_id, [])
                cond_ssa = uses[0] if uses else None
        meta = {
            "bb_id": int(bb_id),
            "condition_ssa": cond_ssa,
            "branch_instruction": term_id,
            "branch_mnemonic": str(term.mnemonic),
            "branch_address": int(term.address),
            "outgoing_edges": [
                {"edge_id": i, "src": int(s), "dst": int(d), "kind": int(t)}
                for i, (s, d, t) in enumerate(edges)
                if s == bb_id
            ],
            "true_false_polarity_available": bool(len(outgoing) >= 2),
            "source": "binary_capstone_register_ssa",
        }
        branch_metadata.append(meta)
        branch_by_bb[bb_id] = meta

    cfg_edges: List[Dict[str, Any]] = []
    for edge_id, (src, dst, kind) in enumerate(edges):
        meta = branch_by_bb.get(int(src)) or {}
        src_end = block_ranges[src][1] if src < len(block_ranges) else 0
        prov = meta.get("branch_instruction")
        if prov is None and src_end > 0:
            prov = instr_ids_by_addr.get(src_end - 1)
        cfg_edges.append(
            {
                "edge_id": int(edge_id),
                "src": int(src),
                "dst": int(dst),
                "src_id": f"bb_{src}",
                "dst_id": f"bb_{dst}",
                "kind": int(kind),
                "condition_ssa": meta.get("condition_ssa") if int(kind) == 0 else None,
                "provenance_instruction": prov,
                "confidence": 1.0,
            }
        )

    func_id = str(row.get("function_id") or row.get("func_id") or f"line_{input_line}:{func_name}")
    source_bin_text = _flatten_bin_instr_text(row)
    real_text = [str(x.get("text") or "") for x in lifted_instructions]
    text_match = not source_bin_text or source_bin_text == real_text
    function_ir = {
        "schema": "FunctionIR.v1",
        "function_id": func_id,
        "function_name": func_name,
        "symbol_name": symbol_name,
        "source_path": row.get("source_file") or row.get("source_path") or row.get("src_path") or row.get("path"),
        "binary_path": str(binary),
        "binary_function_address": int(addr),
        "binary_function_size": int(size),
        "basic_blocks": basic_blocks,
        "cfg_edges": cfg_edges,
        "lifted_instructions": lifted_instructions,
        "ssa_values": ssa_values,
        "def_use_graph": {
            "schema": "RegisterDefUse.v1",
            "edges": def_use_edges,
            "instruction_uses": {k: list(v) for k, v in instruction_uses.items()},
            "instruction_defs": {k: list(v) for k, v in instruction_defs.items()},
        },
        "instruction_provenance": instruction_provenance,
        "branch_metadata": branch_metadata,
        "call_metadata": {
            "calls": [
                {
                    "instruction_id": rec.get("instruction_id"),
                    "address": rec.get("address"),
                    "text": rec.get("text"),
                }
                for rec in lifted_instructions
                if CS_GRP_CALL in (rec.get("groups") or [])
            ]
        },
        "memory_access_summary": {
            "memory_instruction_count": sum(
                1 for rec in lifted_instructions for op in (rec.get("operands") or []) if (op or {}).get("type") == 3
            )
        },
        "lifted_instructions_available": True,
        "ssa_available": True,
        "def_use_available": True,
        "instruction_provenance_available": True,
        "branch_metadata_available": True,
        "call_metadata_available": True,
        "memory_access_summary_available": True,
        "availability_source": {
            "lifted_instructions": "binary_capstone_disassembly",
            "ssa": "binary_capstone_register_ssa",
            "def_use": "binary_capstone_register_def_use",
            "instruction_provenance": "binary_capstone_address",
        },
        "ssa_level": "binary_register_ssa",
        "limitations": [
            "register-level SSA only",
            "memory SSA is summarized, not fully renamed",
            "not a VEX/P-code semantic lifter",
        ],
    }

    out = dict(row)
    out["input_line"] = int(input_line)
    if teacher is not None:
        out.setdefault("module_b_teacher_input_line", teacher.get("input_line"))
        out.setdefault("module_b_teacher_func_name", teacher.get("func_name"))
    out["function_ir"] = function_ir
    out["lifted_instructions"] = lifted_instructions
    out["ssa_values"] = ssa_values
    out["def_use_graph"] = function_ir["def_use_graph"]
    out["instruction_provenance"] = instruction_provenance
    out["branch_metadata"] = branch_metadata
    out["function_ir_enrichment"] = {
        "schema": "ModuleBv3RealFunctionIR100.v1",
        "mode": "binary_capstone_register_ssa",
        "input_line": int(input_line),
        "binary": str(binary),
        "func_name": func_name,
        "symbol_name": symbol_name,
        "n_basic_blocks": int(len(basic_blocks)),
        "n_cfg_edges": int(len(cfg_edges)),
        "n_lifted_instructions": int(len(lifted_instructions)),
        "n_ssa_values": int(len(ssa_values)),
        "n_def_use_edges": int(len(def_use_edges)),
        "real_lifted_available": True,
        "real_ssa_available": True,
        "real_def_use_available": True,
        "real_instruction_provenance_available": True,
        "cfg_shape_matches_source_jsonl": bool(_shape_matches_source(row, blocks, edges)),
        "instr_text_matches_source_jsonl": bool(text_match),
        "note": "Real binary-derived register SSA from Capstone; memory SSA is summarized, not full VEX/P-code SSA.",
    }
    if binary_recovery_meta:
        out["function_ir_enrichment"]["binary_path_recovery"] = binary_recovery_meta
    return out, "ok"


def _fallback_proxy(row: Dict[str, Any], *, input_line: int, reason: str) -> Dict[str, Any]:
    out = proxy_enrich_row(row, input_line=input_line, add_proxy_ssa=True)
    e = dict(out.get("function_ir_enrichment") or {})
    e["schema"] = "ModuleBv3RealFunctionIR100.v1"
    e["mode"] = "cfg_proxy_fallback"
    e["real_extraction_error"] = reason
    e["real_lifted_available"] = False
    e["real_ssa_available"] = False
    e["real_def_use_available"] = False
    e["real_instruction_provenance_available"] = False
    out["function_ir_enrichment"] = e
    return out


def enrich_file(args: argparse.Namespace) -> List[Dict[str, Any]]:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    teacher_jsonl = Path(args.teacher_jsonl).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    root_maps = _parse_root_maps(args.binary_root_map or [])
    path_search_roots = [Path(x).expanduser() for x in (args.path_search_root or [])]
    binary_roots = [Path(x).expanduser() for x in (args.binary_search_root or [])]
    binary_index: Optional[BinarySymbolIndex] = None
    elf_cache = ElfFunctionCache()
    if bool(args.recover_missing_binary_path):
        if not binary_roots:
            raise RuntimeError("--recover-missing-binary-path requires at least one --binary-search-root")
        binary_index = BinarySymbolIndex(binary_roots)
    targets = _load_teacher_targets(teacher_jsonl, int(args.max_examples))
    if not targets:
        raise RuntimeError(f"no teacher input_line targets found in {teacher_jsonl}")
    target_lines = set(int(x) for x in targets)
    max_target_line = max(target_lines) if target_lines else 0

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    counters = Counter()
    error_counter = Counter()
    t0 = time.time()
    with source_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            counters["source_rows_scanned"] += 1
            if target_lines and line_no > max_target_line and int(counters.get("target_rows", 0)) >= len(target_lines):
                counters["source_early_stop"] += 1
                break
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if line_no not in target_lines:
                continue
            teacher = targets[line_no]
            counters["target_rows"] += 1
            try:
                enriched, status = _make_function_ir_from_binary(
                    row,
                    teacher,
                    input_line=line_no,
                    root_maps=root_maps,
                    path_search_roots=path_search_roots,
                    require_cfg_shape_match=bool(args.require_cfg_shape_match),
                    binary_index=binary_index,
                    elf_cache=elf_cache,
                )
            except Exception as exc:
                enriched = None
                status = f"exception:{type(exc).__name__}:{exc}"
            if enriched is None:
                counters["proxy_fallback"] += 1
                error_counter[status] += 1
                if bool(args.no_proxy_fallback):
                    enriched = dict(row)
                    enriched["function_ir_enrichment"] = {
                        "schema": "ModuleBv3RealFunctionIR100.v1",
                        "mode": "real_failed_no_proxy",
                        "input_line": int(line_no),
                        "real_extraction_error": status,
                    }
                else:
                    enriched = _fallback_proxy(row, input_line=line_no, reason=status)
            else:
                counters["real_register_ssa"] += 1
                recovery = ((enriched.get("function_ir_enrichment") or {}).get("binary_path_recovery") or {})
                if recovery:
                    counters["binary_path_recovered"] += 1
                    method = str(recovery.get("method") or "unknown")
                    counters[f"binary_path_recovery_method:{method}"] += 1
            dst.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            records.append(enriched)

    counters["elapsed_sec_x10"] = int(round((time.time() - t0) * 10))
    if binary_index is not None:
        for key, value in binary_index.stats.items():
            counters[f"binary_index:{key}"] = int(value)
    for key, value in elf_cache.stats.items():
        counters[f"elf_cache:{key}"] = int(value)
    summary = render_summary(
        records,
        counters=counters,
        error_counter=error_counter,
        source_jsonl=source_jsonl,
        teacher_jsonl=teacher_jsonl,
        out_jsonl=out_jsonl,
        max_examples=int(args.max_examples),
    )
    if args.out_txt:
        out_txt = Path(args.out_txt).expanduser().resolve()
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return records


def render_summary(
    records: Sequence[Dict[str, Any]],
    *,
    counters: Counter,
    error_counter: Counter,
    source_jsonl: Path,
    teacher_jsonl: Path,
    out_jsonl: Path,
    max_examples: int,
) -> str:
    n = len(records)
    real = 0
    proxy = 0
    lifted = 0
    ssa = 0
    du = 0
    prov = 0
    shape = 0
    text = 0
    modes = Counter()
    for r in records:
        e = r.get("function_ir_enrichment") or {}
        fir = r.get("function_ir") or {}
        modes[str(e.get("mode") or "missing")] += 1
        if e.get("real_ssa_available"):
            real += 1
        if str(e.get("mode")) == "cfg_proxy_fallback":
            proxy += 1
        if fir.get("lifted_instructions_available"):
            lifted += 1
        if fir.get("ssa_available"):
            ssa += 1
        if fir.get("def_use_available"):
            du += 1
        if fir.get("instruction_provenance_available"):
            prov += 1
        if e.get("cfg_shape_matches_source_jsonl"):
            shape += 1
        if e.get("instr_text_matches_source_jsonl"):
            text += 1
    elapsed = counters.get("elapsed_sec_x10", 0) / 10.0
    lines = [
        "===== MODULE B-v3 REAL FUNCTIONIR/SSA 100 ENRICHMENT =====",
        f"source_jsonl = {source_jsonl}",
        f"teacher_jsonl = {teacher_jsonl}",
        f"out_jsonl = {out_jsonl}",
        f"max_examples = {max_examples}",
        f"target_rows_seen = {_pct(int(counters.get('target_rows', 0)), max(1, max_examples))}",
        f"source_rows_scanned = {counters.get('source_rows_scanned', 0)}",
        f"source_early_stop = {counters.get('source_early_stop', 0)}",
        "",
        "-- extraction mode --",
        f"real_register_ssa = {_pct(real, n)}",
        f"proxy_fallback = {_pct(proxy, n)}",
        f"mode_dist = {sorted(modes.items())}",
        "",
        "-- FunctionIR availability on target rows --",
        f"lifted_instructions_available = {_pct(lifted, n)}",
        f"ssa_available = {_pct(ssa, n)}",
        f"def_use_available = {_pct(du, n)}",
        f"instruction_provenance_available = {_pct(prov, n)}",
        "",
        "-- real extraction consistency --",
        f"cfg_shape_matches_source_jsonl = {_pct(shape, n)}",
        f"instr_text_matches_source_jsonl = {_pct(text, n)}",
        "",
        "-- binary path recovery --",
        f"binary_path_recovered = {_pct(int(counters.get('binary_path_recovered', 0)), n)}",
    ]
    recovery_methods = sorted(
        (str(key).split(":", 1)[1], int(value))
        for key, value in counters.items()
        if str(key).startswith("binary_path_recovery_method:")
    )
    if recovery_methods:
        lines.append(f"binary_path_recovery_method_dist = {recovery_methods}")
    index_items = sorted(
        (str(key).split(":", 1)[1], int(value))
        for key, value in counters.items()
        if str(key).startswith("binary_index:")
    )
    if index_items:
        lines.append(f"binary_index_stats = {index_items}")
    elf_items = sorted(
        (str(key).split(":", 1)[1], int(value))
        for key, value in counters.items()
        if str(key).startswith("elf_cache:")
    )
    if elf_items:
        lines.append(f"elf_cache_stats = {elf_items}")
    lines.extend(
        [
            "",
            "-- fallback reasons --",
        ]
    )
    if error_counter:
        for reason, count in error_counter.most_common(20):
            lines.append(f"{reason} = {count}")
    else:
        lines.append("none")
    lines.extend(
        [
            "",
            "note = real_register_ssa is real binary-derived register SSA from Capstone; "
            "memory SSA is summarized, not full VEX/P-code SSA.",
            f"elapsed_sec = {elapsed:.1f}",
            f"saved_jsonl = {out_jsonl}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich teacher-smoke rows with binary-derived FunctionIR/register SSA")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=100)
    ap.add_argument("--binary-root-map", action="append", default=[], help="optional OLD=NEW path prefix rewrite")
    ap.add_argument(
        "--path-search-root",
        action="append",
        default=[],
        help="directory to try when binary paths in JSONL are relative or have stale absolute prefixes",
    )
    ap.add_argument(
        "--binary-search-root",
        action="append",
        default=[],
        help="directory/file to scan for ELF binaries when a row has no resolvable binary_path",
    )
    ap.add_argument(
        "--recover-missing-binary-path",
        action="store_true",
        help="recover missing binary_path values by symbol lookup plus CFG shape matching",
    )
    ap.add_argument("--require-cfg-shape-match", action="store_true")
    ap.add_argument("--no-proxy-fallback", action="store_true")
    args = ap.parse_args()
    enrich_file(args)


if __name__ == "__main__":
    main()

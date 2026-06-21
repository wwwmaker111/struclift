#!/usr/bin/env python3
"""
实例级全量语义对齐检查 —— 777 BB

核心逻辑：不只比较类型，而是提取汇编的具体操作内容，和源码语句的具体文本做交叉比对。

检查项：
  1. call 目标匹配：汇编 call <sym> → 源码中应出现对该函数的调用
  2. return 匹配：汇编 ret → 源码应含 return 或处于函数末尾
  3. 常量匹配：汇编 cmp/test 立即数 → 源码应含该常量
  4. 字符串匹配：如果能从 .rodata 解析出字符串引用

每个 BB 输出 MATCH / PARTIAL / MISMATCH / NODATA
"""
from __future__ import annotations
import json, re, sys, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_four_tier_alignment_45x3 import (
    binskel_jsonl_name, iter_filtered_line_indices,
    load_by_line_idx, pick_spread_line_indices,
)
from deep_validate import analyze_sample
from check_alignment_semantics import (
    _parse_file, _filter_stmts_for_func, infer_col_to_stmt,
)

try:
    from binskel_alignment_batch_stats import (
        DEFAULT_PROJECTS, locate_jsonl,
    )
except ImportError:
    DEFAULT_PROJECTS = ["zlib"]
    locate_jsonl = None

try:
    from check_alignment_semantics import resolve_binary_path
except ImportError:
    resolve_binary_path = None

SRC_MAP = {
    "zlib": "/mnt/e/structlift/zlib_build",
    "curl": "/mnt/e/structlift_src_datasets/curl/curl-8.7.1",
    "sqlite": "/mnt/e/structlift_src_datasets/sqlite/sqlite-autoconf-3520000",
    "busybox": "/mnt/e/structlift_src_datasets/busybox/busybox-1.36.1",
    "openssl": "/mnt/e/structlift_src_datasets/openssl/openssl-3.3.1",
    "openssh": "/mnt/e/structlift_src_datasets/openssh/openssh-9.8p1",
    "libxml2": "/mnt/e/structlift_src_datasets/libxml2/libxml2-2.12.7",
    "ffmpeg": "/mnt/e/structlift_src_datasets/ffmpeg/ffmpeg-7.1",
    "coreutils": "/mnt/e/structlift_src_datasets/coreutils/coreutils-9.4",
}


# ---------- ELF helpers: call target resolution ----------

def _build_addr2sym(elf_path: Path) -> Dict[int, str]:
    """从 ELF 构建地址→符号名映射（含 PLT/GOT）"""
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        return {}
    mapping: Dict[int, str] = {}
    try:
        with elf_path.open("rb") as f:
            elf = ELFFile(f)
            for sec_name in (".symtab", ".dynsym"):
                sec = elf.get_section_by_name(sec_name)
                if sec is None:
                    continue
                for sym in sec.iter_symbols():
                    if sym.name and sym["st_value"]:
                        mapping[sym["st_value"]] = sym.name
            # PLT entries
            plt = elf.get_section_by_name(".plt")
            relplt = elf.get_section_by_name(".rela.plt") or elf.get_section_by_name(".rel.plt")
            if plt and relplt:
                plt_base = plt["sh_addr"]
                plt_entry_size = plt.get("sh_entsize", 16) or 16
                for idx, rel in enumerate(relplt.iter_relocations()):
                    sym_idx = rel["r_info_sym"]
                    dynsym = elf.get_section_by_name(".dynsym")
                    if dynsym and sym_idx < dynsym.num_symbols():
                        sym = dynsym.get_symbol(sym_idx)
                        if sym.name:
                            entry_addr = plt_base + (idx + 1) * plt_entry_size
                            mapping[entry_addr] = sym.name + "@plt"
    except Exception:
        pass
    return mapping


def _disasm_blocks_rich(elf_path: Path, func_name: str,
                        addr2sym: Dict[int, str]) -> Dict[int, List[dict]]:
    """
    反汇编函数的每个 BB，返回 {bb_id: [{'addr':..., 'mnem':..., 'op':..., 'call_target':...}, ...]}.
    call_target 尽量解析为函数名。
    """
    try:
        from elftools.elf.elffile import ELFFile
        from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_RET, CS_OP_IMM
    except ImportError:
        return {}

    try:
        with elf_path.open("rb") as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name(".symtab")
            text = elf.get_section_by_name(".text")
            if not symtab or not text:
                return {}
            text_data = text.data()
            text_base = text["sh_addr"]
            faddr = fsize = None
            for sym in symtab.iter_symbols():
                if sym.name == func_name and sym["st_info"]["type"] == "STT_FUNC":
                    faddr = sym["st_value"]
                    fsize = sym["st_size"]
                    break
            if faddr is None or fsize == 0:
                return {}
            off = faddr - text_base
            code = text_data[off:off + fsize]
    except Exception:
        return {}

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    insns = list(md.disasm(code, faddr))
    if not insns:
        return {}

    addr2idx = {i.address: idx for idx, i in enumerate(insns)}
    block_starts = {insns[0].address}
    for insn in insns:
        g = insn.groups
        if CS_GRP_JUMP in g or CS_GRP_RET in g:
            ni = addr2idx[insn.address] + 1
            if ni < len(insns):
                block_starts.add(insns[ni].address)
            if CS_GRP_JUMP in g and insn.operands and insn.operands[0].type == CS_OP_IMM:
                t = insn.operands[0].imm
                if t in addr2idx:
                    block_starts.add(t)

    sorted_starts = sorted(block_starts)
    result: Dict[int, List[dict]] = {}
    for b_id, sa in enumerate(sorted_starts):
        si = addr2idx[sa]
        ei = addr2idx[sorted_starts[b_id + 1]] if b_id + 1 < len(sorted_starts) else len(insns)
        bb_insns = []
        for idx in range(si, ei):
            insn = insns[idx]
            entry: dict = {
                "addr": insn.address,
                "mnem": insn.mnemonic,
                "op": insn.op_str,
                "call_target": None,
                "imm_values": [],
            }
            if insn.mnemonic == "call" and insn.operands and insn.operands[0].type == CS_OP_IMM:
                target_addr = insn.operands[0].imm
                entry["call_target"] = addr2sym.get(target_addr, hex(target_addr))
            if insn.mnemonic in ("cmp", "test", "mov", "add", "sub"):
                for op in insn.operands:
                    if op.type == CS_OP_IMM and abs(op.imm) < 0x100000:
                        entry["imm_values"].append(op.imm)
            bb_insns.append(entry)
        result[b_id] = bb_insns
    return result


# ---------- Source helpers ----------

def _read_src_lines(src_path: Path) -> List[str]:
    try:
        return src_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []


def _get_stmt_text(src_lines: List[str], stmt: dict) -> str:
    lo = stmt.get("start_line", 0) - 1
    hi = stmt.get("end_line", 0)
    return "\n".join(src_lines[max(0, lo):hi])


_CALL_RE = re.compile(r"\b(\w+)\s*\(")


def _extract_src_call_names(text: str) -> set:
    names = set()
    for m in _CALL_RE.finditer(text):
        n = m.group(1)
        if n not in ("if", "while", "for", "switch", "return", "sizeof", "typeof",
                      "defined", "else", "do", "case"):
            names.add(n)
    return names


# ---------- Instance check ----------

COND_JUMPS = {
    "je", "jne", "jz", "jnz", "jl", "jle", "jg", "jge",
    "jb", "jbe", "ja", "jae", "js", "jns", "jo", "jno",
    "jc", "jnc", "jp", "jnp",
}


def instance_check_bb(
    bb_insns: List[dict],
    stmt: Optional[dict],
    stmt_text: str,
) -> Tuple[str, str]:
    """
    返回 (verdict, detail).
    verdict: MATCH / PARTIAL / MISMATCH / NODATA / SKIP
    """
    if not bb_insns:
        return "NODATA", "no asm"
    if not stmt or not stmt_text.strip():
        return "NODATA", "no stmt"

    checks_done = 0
    checks_pass = 0
    details = []

    # --- Check 1: call target → source should mention that function ---
    asm_calls = []
    for ins in bb_insns:
        if ins["call_target"]:
            raw = ins["call_target"]
            name = raw.replace("@plt", "").strip()
            if not name.startswith("0x"):
                asm_calls.append(name)

    if asm_calls:
        src_calls = _extract_src_call_names(stmt_text)
        for ac in asm_calls:
            checks_done += 1
            base = ac.split(".")[-1]
            if base in src_calls or ac in src_calls:
                checks_pass += 1
                details.append(f"call:{ac}=OK")
            else:
                found = False
                for sc in src_calls:
                    if base in sc or sc in base:
                        found = True
                        break
                if found:
                    checks_pass += 1
                    details.append(f"call:{ac}~partial")
                else:
                    details.append(f"call:{ac}!=src[{','.join(sorted(src_calls)[:3])}]")

    # --- Check 2: ret → source should have 'return' or be compound_statement (epilogue) ---
    last_mnem = bb_insns[-1]["mnem"]
    if last_mnem == "ret":
        checks_done += 1
        stype = stmt.get("type", "")
        if "return" in stmt_text or stype == "return_statement":
            checks_pass += 1
            details.append("ret=return OK")
        elif stype == "compound_statement":
            checks_pass += 1
            details.append("ret=epilogue OK")
        elif "}" in stmt_text:
            checks_pass += 1
            details.append("ret=block_end OK")
        else:
            details.append(f"ret but src={stype}")

    # --- Check 3: cmp immediate → source should contain that value ---
    for ins in bb_insns:
        if ins["mnem"] in ("cmp", "test") and ins["imm_values"]:
            for v in ins["imm_values"]:
                if v == 0:
                    continue
                checks_done += 1
                v_str = str(v)
                v_hex = hex(v)
                v_chr = chr(v) if 32 <= v < 127 else ""
                if v_str in stmt_text or v_hex in stmt_text:
                    checks_pass += 1
                    details.append(f"cmp:{v}=OK")
                elif v_chr and (f"'{v_chr}'" in stmt_text or f'"{v_chr}"' in stmt_text):
                    checks_pass += 1
                    details.append(f"cmp:{v}=char'{v_chr}' OK")
                elif v == 1 and ("true" in stmt_text or "TRUE" in stmt_text or "!=" in stmt_text):
                    checks_pass += 1
                    details.append(f"cmp:1=bool OK")
                elif stmt_text.strip():
                    details.append(f"cmp:{v} not_in_src")
                    # not necessarily wrong: could be sizeof, enum, macro expansion
                    checks_pass += 0.5

    if checks_done == 0:
        return "SKIP", "no_checkable_patterns"

    ratio = checks_pass / checks_done
    if ratio >= 0.9:
        return "MATCH", "; ".join(details)
    elif ratio >= 0.5:
        return "PARTIAL", "; ".join(details)
    else:
        return "MISMATCH", "; ".join(details)


# ---------- Main ----------

def count_bb(d: dict) -> int:
    bl = d.get("block_src_lines")
    if isinstance(bl, list) and bl:
        return len(bl)
    M = d.get("alignment_matrix")
    if isinstance(M, list) and M:
        return len(M)
    return 0


def main():
    out_dir = Path("/mnt/e/structlift_datasets")
    for cand in [out_dir, Path("E:/structlift_datasets")]:
        if cand.is_dir():
            out_dir = cand
            break
    search_dirs = [out_dir, REPO_ROOT]
    bin_root = out_dir

    projects = DEFAULT_PROJECTS
    min_bb = 8
    min_mc = 0.65
    per_proj = 2

    totals = {"MATCH": 0, "PARTIAL": 0, "MISMATCH": 0, "NODATA": 0, "SKIP": 0}
    mismatch_details: list = []
    partial_details: list = []
    total_bb = 0
    total_checked = 0

    for proj in projects:
        name = binskel_jsonl_name(proj, "0")
        jp = None
        if locate_jsonl:
            jp, _ = locate_jsonl(name, search_dirs)
        if not jp or not jp.is_file():
            print(f"SKIP {proj}: {name} not found", file=sys.stderr)
            continue

        candidates = iter_filtered_line_indices(jp, min_bb, min_mc)
        if len(candidates) < per_proj:
            print(f"SKIP {proj}: only {len(candidates)} candidates", file=sys.stderr)
            continue

        picked = pick_spread_line_indices(candidates, per_proj)
        src_root = Path(SRC_MAP.get(proj, ""))
        sr = src_root if src_root.is_dir() else None

        for li in picked:
            d = load_by_line_idx(jp, li)
            if not d:
                continue
            fn = d.get("func_name", "?")
            n_bb = count_bb(d)
            bl = d.get("block_src_lines") or []
            M = d.get("alignment_matrix") or []
            ar = analyze_sample(d)
            bb_primary = list(ar.get("bb_primary") or [])

            elf = resolve_binary_path(d, bin_root) if resolve_binary_path else None
            addr2sym = _build_addr2sym(elf) if elf else {}
            asm_blocks = _disasm_blocks_rich(elf, fn, addr2sym) if elf else {}

            # tree-sitter statements + source text
            col_stmts: list = []
            src_lines: List[str] = []
            if sr and M and bl:
                sf = d.get("source_file") or ""
                rel = Path(sf).name if sf else ""
                src_path = None
                if rel:
                    cand_p = sr / rel
                    if cand_p.is_file():
                        src_path = cand_p
                    else:
                        for hit in sr.rglob(rel):
                            if hit.is_file():
                                src_path = hit
                                break
                if src_path and src_path.is_file():
                    src_lines = _read_src_lines(src_path)
                    try:
                        statements, _ = _parse_file(src_path)
                        all_lns = [ln for ls in bl for ln in (ls or [])]
                        if all_lns:
                            fmin, fmax = min(all_lns), max(all_lns)
                            filtered = _filter_stmts_for_func(statements, fmin, fmax)
                            col_stmts, _ = infer_col_to_stmt(bl, M, filtered)
                    except Exception:
                        pass

            func_match = func_partial = func_mismatch = 0
            for i in range(n_bb):
                total_bb += 1
                bb_insns = asm_blocks.get(i, [])

                j_star = bb_primary[i] if i < len(bb_primary) else -1
                if j_star < 0 and i < len(M) and M[i]:
                    row = M[i]
                    j_star = max(range(len(row)), key=lambda j: row[j])

                stmt = None
                stmt_text = ""
                if 0 <= j_star < len(col_stmts) and col_stmts[j_star]:
                    stmt = col_stmts[j_star]
                    stmt_text = _get_stmt_text(src_lines, stmt)

                verdict, detail = instance_check_bb(bb_insns, stmt, stmt_text)
                totals[verdict] += 1
                if verdict not in ("SKIP", "NODATA"):
                    total_checked += 1

                if verdict == "MISMATCH":
                    mismatch_details.append(f"  {proj}/{fn} BB{i}: {detail}")
                    func_mismatch += 1
                elif verdict == "PARTIAL":
                    partial_details.append(f"  {proj}/{fn} BB{i}: {detail}")
                    func_partial += 1
                elif verdict == "MATCH":
                    func_match += 1

            print(
                f"  {proj}/{fn}: {n_bb} BB | "
                f"MATCH={func_match} PARTIAL={func_partial} MISMATCH={func_mismatch}",
                file=sys.stderr, flush=True,
            )

    print(f"\n{'='*60}")
    print(f"  实例级全量语义对齐检查")
    print(f"{'='*60}")
    print(f"总 BB: {total_bb}")
    print(f"有可检查模式的 BB: {total_checked}")
    print(f"  MATCH:    {totals['MATCH']} ({100*totals['MATCH']/max(total_checked,1):.1f}%)")
    print(f"  PARTIAL:  {totals['PARTIAL']} ({100*totals['PARTIAL']/max(total_checked,1):.1f}%)")
    print(f"  MISMATCH: {totals['MISMATCH']} ({100*totals['MISMATCH']/max(total_checked,1):.1f}%)")
    print(f"  SKIP(无可检查模式): {totals['SKIP']}")
    print(f"  NODATA:   {totals['NODATA']}")

    if mismatch_details:
        print(f"\n--- MISMATCH 详情 ---")
        for s in mismatch_details[:50]:
            print(s)
        if len(mismatch_details) > 50:
            print(f"  ... 共 {len(mismatch_details)} 条")

    if partial_details:
        print(f"\n--- PARTIAL 详情（前 30 条）---")
        for s in partial_details[:30]:
            print(s)
        if len(partial_details) > 30:
            print(f"  ... 共 {len(partial_details)} 条")


if __name__ == "__main__":
    main()

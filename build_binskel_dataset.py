#!/usr/bin/env python3
"""
BinSkel 完整数据集构建脚本 — 二进制基本块与源码对齐

【编译要求】必须用以下参数，少一个都会增加对齐难度：
  clang -O0 -g3 -fno-inline -fno-unroll-loops \\
        -fno-vectorize -fno-slp-vectorize \\
        -fstandalone-debug \\
        source.c -o source.bin

【对齐规则】DWARF .debug_line 中只用 is_stmt=True 的记录：
  - is_stmt 表示「此地址是推荐的断点位置」，即语句边界
  - 用全部行号会引入大量噪声；只用 is_stmt=True 可显著提升准确率

【产出】BB↔源码行号映射：block_src_lines, alignment_matrix
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from elftools.elf.elffile import ELFFile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_RET, CS_OP_REG, CS_OP_IMM, CS_OP_MEM

# ELF symbol type: STT_FUNC = 2
STT_FUNC = 2

from struclift.config import VocabConfig
from struclift.data.dataset_ids import stable_embedding_id
from struclift.data.structures import (
    Instruction,
    BasicBlock,
    BinaryCFG,
    EdgeType,
    PatternType,
)
from struclift.utils.graph_utils import extract_structural_features, find_loop_headers


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 符号表与函数迭代
# ═══════════════════════════════════════════════════════════════════════════════


def iter_functions(elffile: ELFFile):
    """迭代所有有 size 的函数符号。兼容 pyelftools，.symtab 缺失时用 .dynsym。"""
    symtab = elffile.get_section_by_name(".symtab")
    if symtab is None:
        symtab = elffile.get_section_by_name(".dynsym")
    if symtab is None:
        return
    for sym in symtab.iter_symbols():
        try:
            st_info = sym.entry["st_info"]
            st_type = getattr(st_info, "type", None) or (st_info.get("type", st_info) if isinstance(st_info, dict) else st_info)
            ok = False
            if st_type == "STT_FUNC":
                ok = True
            elif isinstance(st_type, int) and (st_type & 0xF) == STT_FUNC:
                ok = True
            if not ok:
                continue
            size = sym.entry["st_size"]
            if size and size > 0:
                yield sym.name, sym.entry["st_value"], size
        except (KeyError, TypeError, AttributeError):
            continue


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DWARF 地址→行号映射
# ═══════════════════════════════════════════════════════════════════════════════


def build_addr_to_line_map(elffile: ELFFile) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, str]]:
    """
    从 .debug_line 构建 虚拟地址 → (file_idx, line) 映射。
    file_idx 为 DWARF 1-based 索引。
    返回:
      addr2line: addr -> (file_idx, line)
      file_idx2path: file_idx -> 源文件路径
    """
    addr2line: Dict[int, Tuple[int, int]] = {}
    file_idx2path: Dict[int, str] = {}

    if not elffile.has_dwarf_info():
        return addr2line, file_idx2path

    try:
        dwarf = elffile.get_dwarf_info()
    except Exception:
        return addr2line, file_idx2path

    for cu in dwarf.iter_CUs():
        line_prog = dwarf.line_program_for_CU(cu)
        if line_prog is None:
            continue

        try:
            include_dirs = list(line_prog["include_directory"] or [])
        except (KeyError, TypeError):
            include_dirs = []
        try:
            file_entries = list(line_prog["file_entry"] or [])
        except (KeyError, TypeError):
            file_entries = []
        comp_dir = ""
        try:
            top_die = cu.get_top_DIE()
            if "DW_AT_comp_dir" in top_die.attributes:
                v = top_die.attributes["DW_AT_comp_dir"].value
                comp_dir = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        except Exception:
            pass

        for one_idx, fe in enumerate(file_entries):
            file_idx = one_idx + 1
            try:
                name = getattr(fe, "name", None)
                if name is None and hasattr(fe, "__getitem__"):
                    name = fe["name"]
            except (KeyError, TypeError, AttributeError):
                name = ""
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            name = str(name) if name else ""
            try:
                dir_idx = getattr(fe, "dir_index", None)
                if dir_idx is None and hasattr(fe, "__getitem__"):
                    dir_idx = fe["dir_index"]
            except (KeyError, TypeError, AttributeError):
                dir_idx = 0
            if dir_idx and 1 <= dir_idx <= len(include_dirs):
                inc = include_dirs[dir_idx - 1]
                if isinstance(inc, bytes):
                    inc = inc.decode("utf-8", errors="replace")
                path = Path(inc) / name
            else:
                path = Path(name)
            if comp_dir:
                path = Path(comp_dir) / path
            file_idx2path[file_idx] = str(path)

        for entry in line_prog.get_entries():
            if entry.state is None:
                continue
            # 只用 is_stmt=True 的记录（语句边界/推荐断点），过滤噪声
            if not getattr(entry.state, "is_stmt", False):
                continue
            addr = getattr(entry.state, "address", 0) or 0
            line = getattr(entry.state, "line", 0) or 0
            file_idx = getattr(entry.state, "file", 1) or 1
            if addr and line:
                addr2line[addr] = (file_idx, line)

    return addr2line, file_idx2path


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Binary CFG 构建（含 block 地址范围 + operand 信息）
# ═══════════════════════════════════════════════════════════════════════════════


def _operand_type_to_id(op_type: int) -> int:
    """Capstone operand type -> vocab id (0=invalid/pad, 1=REG, 2=IMM, 3=MEM, 4=FP)."""
    mapping = {0: 0, CS_OP_REG: 1, CS_OP_IMM: 2, CS_OP_MEM: 3, 4: 4}  # 4 = CS_OP_FP
    return mapping.get(op_type, 0)


def _value_to_bucket(val: int, num_buckets: int = 64) -> int:
    """将立即数/值映射到 log 分桶。"""
    if val == 0:
        return 0
    abs_val = abs(val)
    if abs_val < 2:
        return 1
    bucket = min(int(math.log2(abs_val)) + 1, num_buckets - 1)
    return max(0, bucket)


def build_cfg_for_function(
    code_bytes: bytes,
    base_addr: int,
    func_name: str,
    num_opcodes: int = 1024,
) -> Tuple[BinaryCFG, List[Tuple[int, int]]]:
    """
    构建 Binary CFG，并返回每个 block 的 (start_addr, end_addr) 地址范围。
    """
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    insns = list(md.disasm(code_bytes, base_addr))
    if not insns:
        return BinaryCFG(func_name=func_name, blocks=[], edges=[]), []

    addr2idx = {insn.address: idx for idx, insn in enumerate(insns)}

    block_starts = {insns[0].address}
    for insn in insns:
        groups = insn.groups
        is_jump = CS_GRP_JUMP in groups
        is_ret = CS_GRP_RET in groups
        if is_jump or is_ret:
            next_idx = addr2idx[insn.address] + 1
            if next_idx < len(insns):
                block_starts.add(insns[next_idx].address)
            if is_jump and insn.operands and insn.operands[0].type == 1:
                tgt = insn.operands[0].imm
                if tgt in addr2idx:
                    block_starts.add(tgt)

    sorted_starts = sorted(block_starts)
    addr2block = {}
    blocks: List[BasicBlock] = []
    block_addr_ranges: List[Tuple[int, int]] = []

    for b_id, start_addr in enumerate(sorted_starts):
        start_idx = addr2idx[start_addr]
        if b_id + 1 < len(sorted_starts):
            end_addr_next = sorted_starts[b_id + 1]
            end_idx = addr2idx[end_addr_next]
        else:
            end_idx = len(insns)
        end_addr = insns[end_idx - 1].address + insns[end_idx - 1].size if end_idx > 0 else start_addr

        insts: List[Instruction] = []
        for i in range(start_idx, end_idx):
            insn = insns[i]
            addr2block[insn.address] = b_id
            op_types = [0, 0, 0]
            value_buckets = [0, 0, 0]
            for j, op in enumerate(insn.operands[:3]):
                op_types[j] = _operand_type_to_id(op.type)
                if op.type == CS_OP_IMM:
                    value_buckets[j] = _value_to_bucket(op.imm)
                elif op.type == CS_OP_REG:
                    value_buckets[j] = op.reg & 0x3F
            insts.append(
                Instruction(
                    opcode=stable_embedding_id(str(insn.mnemonic), num_opcodes),
                    operand_types=op_types,
                    value_buckets=value_buckets,
                    raw_text=f"{insn.mnemonic} {insn.op_str}",
                )
            )
        blocks.append(BasicBlock(block_id=b_id, instructions=insts))
        block_addr_ranges.append((start_addr, end_addr))

    edges: List[Tuple[int, int, int]] = []
    for idx, insn in enumerate(insns):
        b_id = addr2block[insn.address]
        groups = insn.groups
        is_jump = CS_GRP_JUMP in groups
        is_ret = CS_GRP_RET in groups
        if is_jump:
            if insn.operands and insn.operands[0].type == 1:
                tgt = insn.operands[0].imm
                if tgt in addr2block:
                    edges.append((b_id, addr2block[tgt], int(EdgeType.COND)))
            next_idx = idx + 1
            if next_idx < len(insns):
                nb = addr2block[insns[next_idx].address]
                edges.append((b_id, nb, int(EdgeType.COND)))
        elif is_ret:
            continue
        else:
            next_idx = idx + 1
            if next_idx < len(insns):
                nb = addr2block[insns[next_idx].address]
                edges.append((b_id, nb, int(EdgeType.UNCOND)))

    edges = list({(s, d, t) for (s, d, t) in edges})
    return BinaryCFG(func_name=func_name, blocks=blocks, edges=edges), block_addr_ranges


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BB → 行号映射（监督：Module B）
# ═══════════════════════════════════════════════════════════════════════════════


def get_block_lines(
    block_addr_ranges: List[Tuple[int, int]],
    addr2line: Dict[int, Tuple[int, int]],
    file_idx2path: Optional[Dict[int, str]] = None,
) -> Tuple[List[List[int]], Optional[int], Optional[str]]:
    """
    根据地址范围与 addr2line 计算每个 BB 对应的源码行号列表。
    返回: (block_lines, primary_file_idx, primary_source_path)
    """
    block_lines: List[List[int]] = []
    file_counts: Dict[int, int] = {}
    file_idx2path = file_idx2path or {}

    for start_addr, end_addr in block_addr_ranges:
        lines = set()
        addr = start_addr
        while addr <= end_addr:
            if addr in addr2line:
                fi, ln = addr2line[addr]
                lines.add(ln)
                file_counts[fi] = file_counts.get(fi, 0) + 1
            addr += 4
        block_lines.append(sorted(lines) if lines else [])

    if not file_counts:
        return block_lines, None, None
    primary_file_idx = max(file_counts, key=file_counts.get)
    path = file_idx2path.get(primary_file_idx)
    return block_lines, primary_file_idx, path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 源码读取
# ═══════════════════════════════════════════════════════════════════════════════


def read_source_lines(
    source_path: Path,
    all_line_numbers: List[int],
    compile_dir: Optional[Path] = None,
) -> Tuple[List[str], Dict[int, str]]:
    """
    从源文件读取指定行。all_line_numbers 为函数内出现的所有行号。
    返回: (lines_in_order, line_num_to_text)
    """
    path = Path(source_path)
    candidates = [path]
    if compile_dir:
        candidates.append(compile_dir / path.name)
        candidates.append(compile_dir / path)
    for p in candidates:
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                line_to_text: Dict[int, str] = {}
                for i, ln in enumerate(text.splitlines(), 1):
                    if i in all_line_numbers:
                        line_to_text[i] = ln
                return [line_to_text[ln] for ln in sorted(line_to_text)], line_to_text
            except Exception:
                pass
    return [], {}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. pattern_labels 启发式（监督：Module A pattern head）
# ═══════════════════════════════════════════════════════════════════════════════


def assign_pattern_labels(cfg: BinaryCFG) -> List[int]:
    """为每个 block 分配 pattern label。"""
    import torch
    n = cfg.num_blocks
    labels = [int(PatternType.SEQUENTIAL)] * n
    if n == 0 or not cfg.edges:
        return labels
    src_l = [e[0] for e in cfg.edges]
    dst_l = [e[1] for e in cfg.edges]
    edge_index = torch.tensor([src_l, dst_l], dtype=torch.int64)
    loop_headers = find_loop_headers(edge_index, n)
    for i in loop_headers:
        labels[i] = int(PatternType.LOOP_HEADER)
    for s, d, et in cfg.edges:
        if et == int(EdgeType.BACK) and d in loop_headers:
            if labels[s] == int(PatternType.SEQUENTIAL):
                labels[s] = int(PatternType.LOOP_BODY)
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 简化的 Source 表示（按行作为 statement，无 tree-sitter）
# ═══════════════════════════════════════════════════════════════════════════════


def build_simple_source_repr(
    block_lines: List[List[int]],
    line_to_text: Dict[int, str],
    src_vocab_size: int = 32000,
) -> Tuple[List[int], List[int], List[List[int]], List[int]]:
    """
    将 block_lines 转为简单 source 表示：
      src_token_ids: 展平的所有行 token（这里用字符 hash 模拟，实际应接 tokenizer）
      src_stmt_lengths: 每行 token 数
      src_edge_index: 空的（无 AST 时无控制流边）
      src_ast_types: 每行类型，默认 OTHER=15
    """
    line_nums = sorted(set(ln for lines in block_lines for ln in lines))
    if not line_nums:
        return [], [], [], []
    stmt_lengths = []
    token_ids = []
    for ln in line_nums:
        t = line_to_text.get(ln, "")
        ids = [stable_embedding_id(w, src_vocab_size) for w in t.split()]
        stmt_lengths.append(len(ids) if ids else 1)
        token_ids.extend(ids if ids else [0])
    edge_src, edge_dst = [], []
    ast_types = [15] * len(line_nums)  # 15 = ASTType.OTHER
    return token_ids, stmt_lengths, [edge_src, edge_dst], ast_types


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 对齐矩阵（监督：Module B）
# ═══════════════════════════════════════════════════════════════════════════════


def build_alignment_matrix(
    block_lines: List[List[int]],
    line_to_stmt_idx: Dict[int, int],
) -> List[List[float]]:
    """构建 [n_blocks, n_statements] 软对齐矩阵。"""
    line_nums = sorted(line_to_stmt_idx.keys())
    n_stmt = len(line_nums)
    n_block = len(block_lines)
    if n_stmt == 0 or n_block == 0:
        return []
    matrix = [[0.0] * n_stmt for _ in range(n_block)]
    for b_id, lines in enumerate(block_lines):
        if not lines:
            continue
        for ln in lines:
            if ln in line_to_stmt_idx:
                s_idx = line_to_stmt_idx[ln]
                matrix[b_id][s_idx] = 1.0
        row_sum = sum(matrix[b_id])
        if row_sum > 0:
            for j in range(n_stmt):
                matrix[b_id][j] /= row_sum
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 9. 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    import argparse
    ap = argparse.ArgumentParser(description="构建 BinSkel 完整数据集")
    ap.add_argument("--elf", default=None, help="ELF 路径 (默认: ~/struclift_wsl/.../sqlite3)")
    ap.add_argument("--src", default=None, help="源码根目录 (默认: ELF 同级的 sqlite-autoconf 目录)")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径")
    ap.add_argument("--opt", default="O0", help="优化级别标签")
    ap.add_argument("--verbose", "-v", action="store_true", help="打印诊断信息")
    _vc = VocabConfig()
    ap.add_argument(
        "--num-opcodes",
        type=int,
        default=_vc.num_opcodes,
        help="Opcode 词表大小（与 VocabConfig / 训练一致）",
    )
    ap.add_argument(
        "--src-vocab-size",
        type=int,
        default=32000,
        help="源码 token 词表大小（与 SourceCFGEncoder 默认一致）",
    )
    args = ap.parse_args()
    if args.num_opcodes < 2:
        ap.error("--num-opcodes 必须 >= 2")
    if args.src_vocab_size < 2:
        ap.error("--src-vocab-size 必须 >= 2")

    base = Path.home()
    elf_path = Path(args.elf) if args.elf else (base / "struclift_wsl" / "datasets" / "sqlite" / "sqlite-autoconf-3520000" / "sqlite3")
    if args.src:
        src_root = Path(args.src)
    else:
        src_root = elf_path.parent
    if args.out:
        out_path = Path(args.out)
        out_dir = out_path.parent
    else:
        out_dir = base / "structlift_datasets"
        out_path = out_dir / "sqlite_O0_binskel_full.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"词表: num_opcodes={args.num_opcodes}, src_vocab_size={args.src_vocab_size}",
        flush=True,
    )

    if not elf_path.exists():
        print(f"ELF 不存在: {elf_path}")
        return

    if args.verbose:
        import sys
        f = open(elf_path, "rb")
        ef = ELFFile(f)
        st = ef.get_section_by_name(".symtab")
        ds = ef.get_section_by_name(".dynsym")
        print(f".symtab 存在: {st is not None}", file=sys.stderr)
        print(f".dynsym 存在: {ds is not None}", file=sys.stderr)
        if st:
            n = sum(1 for _ in st.iter_symbols())
            print(f".symtab 符号数: {n}", file=sys.stderr)
        if ds:
            n = sum(1 for _ in ds.iter_symbols())
            print(f".dynsym 符号数: {n}", file=sys.stderr)
        f.close()

    with elf_path.open("rb") as f, out_path.open("w", encoding="utf-8") as out_f:
        elffile = ELFFile(f)
        text_sec = elffile.get_section_by_name(".text")
        text_bytes = text_sec.data()
        base_addr = text_sec["sh_addr"]
        print("解析 DWARF .debug_line (is_stmt=True) ...", flush=True)
        addr2line, file_idx2path = build_addr_to_line_map(elffile)
        print(f"DWARF 解析完成, addr2line 记录数: {len(addr2line)}", flush=True)

        comp_dir = src_root
        written = 0
        skipped = 0
        total = 0

        for func_name, addr, size in iter_functions(elffile):
            total += 1
            if total % 500 == 0:
                print(f"  已处理 {total} 个函数, 已写出 {written} ...", flush=True)
            start = addr - base_addr
            if start < 0 or start + size > len(text_bytes):
                skipped += 1
                continue
            code_bytes = text_bytes[start : start + size]
            cfg, block_addr_ranges = build_cfg_for_function(
                code_bytes, addr, func_name, num_opcodes=args.num_opcodes
            )
            if cfg.num_blocks == 0:
                skipped += 1
                continue

            struct_feats = extract_structural_features(cfg)
            block_lines, primary_file_idx, primary_source_path = get_block_lines(
                block_addr_ranges, addr2line, file_idx2path
            )
            # 确定主要源文件
            all_lines = [ln for lines in block_lines for ln in lines]
            if not all_lines:
                source_path = None
                line_to_text = {}
            else:
                if primary_source_path:
                    sp = Path(primary_source_path)
                    source_path = sp if sp.is_absolute() else (comp_dir / sp)
                else:
                    source_path = None
                if source_path and source_path.exists():
                    _, line_to_text = read_source_lines(source_path, all_lines, comp_dir)
                else:
                    line_to_text = {}

            bin_opcodes = []
            bin_operand_types = []
            bin_operand_values = []
            bin_block_lengths = []
            for bb in cfg.blocks:
                bin_block_lengths.append(len(bb.instructions))
                for inst in bb.instructions:
                    bin_opcodes.append(inst.opcode)
                    ot = (inst.operand_types + [0, 0, 0])[:3]
                    vb = (inst.value_buckets + [0, 0, 0])[:3]
                    bin_operand_types.extend(ot)
                    bin_operand_values.extend(vb)

            src = [e[0] for e in cfg.edges]
            dst = [e[1] for e in cfg.edges]
            etype = [e[2] for e in cfg.edges]

            pattern_labels = assign_pattern_labels(cfg)

            line_nums = sorted(set(ln for lines in block_lines for ln in lines))
            line_to_stmt_idx = {ln: i for i, ln in enumerate(line_nums)}
            alignment_matrix = build_alignment_matrix(block_lines, line_to_stmt_idx)

            src_token_ids, src_stmt_lengths, src_edge_index, src_ast_types = build_simple_source_repr(
                block_lines, line_to_text, src_vocab_size=args.src_vocab_size
            )

            sample = {
                "binary": str(elf_path),
                "func_name": func_name,
                "opt_level": args.opt,
                # Module A 输入 - binary
                "bin_opcodes": bin_opcodes,
                "bin_operand_types": bin_operand_types,
                "bin_operand_values": bin_operand_values,
                "bin_instr_text": [
                    [inst.raw_text for inst in block.instructions]
                    for block in cfg.blocks
                ],
                "bin_block_lengths": bin_block_lengths,
                "bin_edge_index": [src, dst],
                "bin_edge_type": etype,
                "bin_struct_features": struct_feats.tolist(),
                # Module A 输入 - source（简化版）
                "src_token_ids": src_token_ids,
                "src_stmt_lengths": src_stmt_lengths,
                "src_edge_index": src_edge_index,
                "src_edge_type": [],
                "src_ast_types": src_ast_types,
                # 监督：BB↔源码行号
                "block_src_lines": block_lines,
                "alignment_matrix": alignment_matrix,
                "pattern_labels": pattern_labels,
                # 辅助
                "source_file": str(source_path) if source_path else None,
                "source_lines": line_nums,
            }
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

        print(f"写出: {out_path}")
        print(f"样本数: {written}, 跳过: {skipped}")


if __name__ == "__main__":
    main()

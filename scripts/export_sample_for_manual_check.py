#!/usr/bin/env python3
"""
导出指定样本的完整数据供人工核查：原函数源码、BB 反汇编、tree-sitter 语句、对齐矩阵。
用法: python scripts/export_sample_for_manual_check.py <jsonl> <func_name>
      python scripts/export_sample_for_manual_check.py <jsonl> --index 0
      python scripts/export_sample_for_manual_check.py <jsonl> --batch "func1" "func2" -o out.txt
"""
import json
import sys
from pathlib import Path

# 复用 build 脚本的 disasm 逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elftools.elf.elffile import ELFFile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_RET, CS_OP_IMM

AST_TYPE_NAMES = {
    0: "?", 2: "if_statement", 3: "while_statement", 4: "for_statement",
    5: "do_statement", 6: "switch_statement", 7: "case_statement",
    8: "return_statement", 9: "break_statement", 10: "continue_statement",
    11: "goto_statement", 12: "compound_statement", 15: "other",
}


def disasm_function(elf_path: str, func_name: str) -> list:
    """返回 [(bb_id, [(addr, mnemonic, op_str), ...]), ...]"""
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        text = elf.get_section_by_name(".text")
        text_data = text.data()
        text_base = text["sh_addr"]
        addr, size = None, 0
        for sym in symtab.iter_symbols():
            if sym.name == func_name and sym["st_info"]["type"] == "STT_FUNC":
                addr = sym["st_value"]
                size = sym["st_size"]
                break
        if addr is None or size == 0:
            return []
        offset = addr - text_base
        code = text_data[offset : offset + size]
        insns = list(md.disasm(code, addr))
    if not insns:
        return []

    addr2idx = {insn.address: i for i, insn in enumerate(insns)}
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
    addr2block = {}
    result = []
    for b_id, sa in enumerate(sorted_starts):
        si = addr2idx[sa]
        ei = addr2idx[sorted_starts[b_id + 1]] if b_id + 1 < len(sorted_starts) else len(insns)
        block_insns = []
        for i in range(si, ei):
            insn = insns[i]
            addr2block[insn.address] = b_id
            block_insns.append((hex(insn.address), insn.mnemonic, insn.op_str))
        result.append((b_id, block_insns))
    return result


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    jsonl_path = Path(sys.argv[1]).expanduser()
    arg2 = sys.argv[2]

    target = None
    if arg2 == "--index":
        idx = int(sys.argv[3])
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if i == idx:
                    target = json.loads(line)
                    break
    elif arg2 == "--line" and len(sys.argv) >= 4:
        line_num = int(sys.argv[3])  # 1-based line number in file
        import subprocess
        r = subprocess.run(
            ["sed", "-n", f"{line_num}p", str(jsonl_path)],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout:
            target = json.loads(r.stdout)
    else:
        with open(jsonl_path) as f:
            for line in f:
                s = json.loads(line)
                if s.get("func_name") == arg2:
                    target = s
                    break
    if not target:
        print(f"未找到: {arg2}")
        sys.exit(1)

    print(format_sample_report(target))


def format_sample_report(target: dict) -> str:
    """将单条 binskel 样本格式化为人工核查文本（供 export_* / 审计脚本复用）。"""
    fn = target.get("func_name", "?")
    sf = target.get("source_file", "")
    sl = target.get("source_lines", [])
    bl = target.get("block_src_lines", [])
    M = target.get("alignment_matrix", [])
    W = target.get("alignment_confidence", [])
    ast_types = target.get("src_ast_types", [])
    stmt_lens = target.get("src_stmt_lengths", [])
    bin_path = target.get("binary", "")

    out = []
    out.append("=" * 80)
    out.append(f"函数: {fn}")
    out.append(f"源文件: {sf}")
    out.append(f"二进制: {bin_path}")
    out.append("=" * 80)

    # 1. 源码
    all_ln = sorted(set(sl) | set(ln for ls in bl for ln in ls))
    out.append("\n## 1. 源码 (source_lines + block_src_lines)")
    path = Path(sf).expanduser()
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in all_ln:
            if 1 <= ln <= len(lines):
                out.append(f"  {ln:6d} | {lines[ln-1].rstrip()}")
    else:
        out.append(f"  [文件不存在: {path}]")

    # 2. BB 反汇编
    out.append("\n## 2. 二进制 BB 反汇编")
    bb_asm = disasm_function(bin_path, fn) if bin_path and Path(bin_path).exists() else []
    for b_id, insns in bb_asm:
        out.append(f"\n  --- BB {b_id} ---")
        for addr, mnem, op in insns:
            out.append(f"    {addr}  {mnem:8s} {op}")

    # 3. Tree-sitter 语句
    out.append("\n## 3. Tree-sitter 语句 (按列)")
    for j in range(len(ast_types)):
        t = ast_types[j] if j < len(ast_types) else 15
        L = stmt_lens[j] if j < len(stmt_lens) else 0
        out.append(f"  语句 {j}: type={AST_TYPE_NAMES.get(t, t)}, tokens={L}")

    # 4. 对齐
    out.append("\n## 4. 对齐矩阵")
    out.append("  block_src_lines (每 BB 对应行号):")
    for i, ln_list in enumerate(bl):
        out.append(f"    BB{i}: {ln_list}")
    out.append("  alignment_matrix M_gt (BB×语句):")
    for i in range(len(M)):
        row = [f"{M[i][j]:.2f}" for j in range(len(M[i]))]
        out.append(f"    BB{i}: {row}")
    out.append("  alignment_confidence W:")
    for i in range(len(W)):
        row = [f"{W[i][j]:.2f}" for j in range(len(W[i]))]
        out.append(f"    BB{i}: {row}")

    # 5. 其他
    out.append("\n## 5. 其他")
    out.append(f"  signal_stats: {target.get('signal_stats', {})}")
    out.append(f"  pattern_labels: {target.get('pattern_labels', [])}")
    out.append(f"  bin_block_lengths: {target.get('bin_block_lengths', [])}")
    out.append(f"  bin_edge_index: {target.get('bin_edge_index', [])}")
    out.append(f"  bin_edge_type: {target.get('bin_edge_type', [])}")

    return "\n".join(out)


if __name__ == "__main__":
    main()

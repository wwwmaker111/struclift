#!/usr/bin/env python3
"""Debug: trace edge type generation for first few functions"""
import sys, os
sys.path.insert(0, '/mnt/e/structlift')

from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_CALL, CS_GRP_RET
from elftools.elf.elffile import ELFFile
from struclift.data.structures import EdgeType

print(f"EdgeType values: COND={int(EdgeType.COND)}, UNCOND={int(EdgeType.UNCOND)}, BACK={int(EdgeType.BACK)}, CALL={int(EdgeType.CALL)}, RET={int(EdgeType.RET)}")

elf_path = os.path.expanduser('~/struclift_wsl/datasets/zlib/zlib-1.3.1/zlib_o0')
md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True

with open(elf_path, 'rb') as f:
    elf = ELFFile(f)
    symtab = elf.get_section_by_name('.symtab')
    text = elf.get_section_by_name('.text')
    text_data = text.data()
    text_base = text['sh_addr']

    count = 0
    for sym in symtab.iter_symbols():
        if sym['st_info']['type'] != 'STT_FUNC' or sym['st_size'] == 0:
            continue
        fname = sym.name
        addr = sym['st_value']
        size = sym['st_size']
        if addr < text_base or addr + size > text_base + len(text_data):
            continue

        offset = addr - text_base
        code = text_data[offset:offset+size]
        insns = list(md.disasm(code, addr))
        if len(insns) < 5:
            continue

        addr2idx = {insn.address: i for i, insn in enumerate(insns)}
        block_starts = {insns[0].address}
        for insn in insns:
            g = insn.groups
            if CS_GRP_JUMP in g or CS_GRP_RET in g:
                ni = addr2idx[insn.address] + 1
                if ni < len(insns):
                    block_starts.add(insns[ni].address)
                if CS_GRP_JUMP in g and insn.operands and insn.operands[0].type == 2:
                    t = insn.operands[0].imm
                    if t in addr2idx:
                        block_starts.add(t)

        sorted_starts = sorted(block_starts)
        addr2block = {}
        for b_id, sa in enumerate(sorted_starts):
            si = addr2idx[sa]
            ei = addr2idx[sorted_starts[b_id + 1]] if b_id + 1 < len(sorted_starts) else len(insns)
            for i in range(si, ei):
                addr2block[insns[i].address] = b_id

        print(f"\n=== {fname} ({len(insns)} insns, {len(sorted_starts)} BBs) ===")
        edges = []
        for idx, insn in enumerate(insns):
            b_id = addr2block[insn.address]
            ni_idx = idx + 1
            is_last = (ni_idx >= len(insns)) or (addr2block.get(insns[ni_idx].address, -1) != b_id)
            if not is_last:
                continue
            g = insn.groups
            grp_str = []
            if CS_GRP_JUMP in g: grp_str.append('JUMP')
            if CS_GRP_CALL in g: grp_str.append('CALL')
            if CS_GRP_RET in g: grp_str.append('RET')
            print(f"  BB{b_id} last: {insn.mnemonic} {insn.op_str}  groups=[{','.join(grp_str)}]")

            if CS_GRP_RET in g:
                print(f"    -> no edge (RET)")
                continue
            if CS_GRP_CALL in g:
                if ni_idx < len(insns):
                    tb = addr2block[insns[ni_idx].address]
                    print(f"    -> ({b_id},{tb}) CALL")
                    edges.append((b_id, tb, int(EdgeType.CALL)))
                continue
            if CS_GRP_JUMP in g:
                is_uncond = insn.mnemonic in ('jmp', 'ljmp')
                if insn.operands and insn.operands[0].type == 2:
                    t = insn.operands[0].imm
                    if t in addr2block:
                        tb = addr2block[t]
                        if tb <= b_id:
                            print(f"    -> ({b_id},{tb}) BACK  (target<=src)")
                            edges.append((b_id, tb, int(EdgeType.BACK)))
                        elif is_uncond:
                            print(f"    -> ({b_id},{tb}) UNCOND (jmp forward)")
                            edges.append((b_id, tb, int(EdgeType.UNCOND)))
                        else:
                            print(f"    -> ({b_id},{tb}) COND (cond jump target)")
                            edges.append((b_id, tb, int(EdgeType.COND)))
                    else:
                        print(f"    -> jump target 0x{t:x} NOT in function")
                else:
                    print(f"    -> indirect jump (no IMM operand)")
                if not is_uncond and ni_idx < len(insns):
                    tb = addr2block[insns[ni_idx].address]
                    print(f"    -> ({b_id},{tb}) COND (fall-through)")
                    edges.append((b_id, tb, int(EdgeType.COND)))
                continue
            if ni_idx < len(insns):
                tb = addr2block[insns[ni_idx].address]
                print(f"    -> ({b_id},{tb}) UNCOND (sequential)")
                edges.append((b_id, tb, int(EdgeType.UNCOND)))

        from collections import Counter
        ec = Counter(e[2] for e in edges)
        NAMES = {0:'COND',1:'UNCOND',2:'BACK',3:'CALL',4:'RET'}
        print(f"  Summary: {dict((NAMES.get(k,k),v) for k,v in sorted(ec.items()))}")

        count += 1
        if count >= 3:
            break

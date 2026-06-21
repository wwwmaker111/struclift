from pathlib import Path
import json
from typing import List, Optional, Tuple

from elftools.elf.elffile import ELFFile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_RET

from struclift.config import VocabConfig
from struclift.data.dataset_ids import stable_embedding_id
from struclift.data.structures import Instruction, BasicBlock, BinaryCFG, EdgeType
from struclift.utils.graph_utils import extract_structural_features


def get_text_section_bytes(elffile: ELFFile):
    text = elffile.get_section_by_name(".text")
    return text, text.data()


def iter_functions(elffile: ELFFile):
    symtab = elffile.get_section_by_name(".symtab")
    if symtab is None:
        return
    for sym in symtab.iter_symbols():
        if sym["st_info"]["type"] == "STT_FUNC" and sym.entry["st_size"] > 0:
            yield sym.name, sym.entry["st_value"], sym.entry["st_size"]


def build_cfg_for_function(
    code_bytes: bytes,
    base_addr: int,
    func_name: str,
    num_opcodes: Optional[int] = None,
) -> BinaryCFG:
    if num_opcodes is None:
        num_opcodes = VocabConfig().num_opcodes
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    insns = list(md.disasm(code_bytes, base_addr))
    if not insns:
        return BinaryCFG(func_name=func_name, blocks=[], edges=[])

    addr2idx = {insn.address: idx for idx, insn in enumerate(insns)}

    # 1) basic block 起点
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

    # 2) 划分 basic block
    for b_id, start_addr in enumerate(sorted_starts):
        start_idx = addr2idx[start_addr]
        if b_id + 1 < len(sorted_starts):
            end_addr_next = sorted_starts[b_id + 1]
            end_idx = addr2idx[end_addr_next]
        else:
            end_idx = len(insns)

        insts: List[Instruction] = []
        for i in range(start_idx, end_idx):
            insn = insns[i]
            addr2block[insn.address] = b_id
            insts.append(
                Instruction(
                    opcode=stable_embedding_id(str(insn.mnemonic), num_opcodes),
                    operand_types=[0, 0, 0],
                    value_buckets=[0, 0, 0],
                    raw_text=f"{insn.mnemonic} {insn.op_str}",
                )
            )
        blocks.append(BasicBlock(block_id=b_id, instructions=insts))

    # 3) 构建 CFG 边
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
    return BinaryCFG(func_name=func_name, blocks=blocks, edges=edges)


def main():
    elf_path = (
        Path.home()
        / "struclift_wsl"
        / "datasets"
        / "sqlite"
        / "sqlite-autoconf-3520000"
        / "sqlite3"
    )

    out_path = Path.home() / "structlift_datasets" / "sqlite_O0_cfg.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with elf_path.open("rb") as f, out_path.open("w", encoding="utf-8") as out_f:
        elffile = ELFFile(f)
        text_sec, text_bytes = get_text_section_bytes(elffile)
        base = text_sec["sh_addr"]

        for func_name, addr, size in iter_functions(elffile):
            start = addr - base
            if start < 0 or start + size > len(text_bytes):
                continue
            code_bytes = text_bytes[start : start + size]
            cfg = build_cfg_for_function(code_bytes, addr, func_name)
            struct_feats = extract_structural_features(cfg)

            opcodes = []
            block_lengths = []
            for bb in cfg.blocks:
                block_lengths.append(len(bb.instructions))
                for inst in bb.instructions:
                    opcodes.append(inst.opcode)

            if cfg.edges:
                src = [e[0] for e in cfg.edges]
                dst = [e[1] for e in cfg.edges]
                etype = [e[2] for e in cfg.edges]
            else:
                src, dst, etype = [], [], []

            sample = {
                "binary": str(elf_path),
                "func_name": func_name,
                "bin_opcodes": opcodes,
                "bin_block_lengths": block_lengths,
                "bin_edge_index": [src, dst],
                "bin_edge_type": etype,
                "bin_struct_features": struct_feats.tolist(),
            }
            out_f.write(json.dumps(sample) + "\n")

    print("写出:", out_path)


if __name__ == "__main__":
    main()

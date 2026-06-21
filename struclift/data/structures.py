"""
Core data structures used throughout StrucLift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch


# ═══════════════════════════ enumerations ═══════════════════════════════════

class EdgeType(IntEnum):
    COND = 0
    UNCOND = 1
    BACK = 2
    CALL = 3
    RET = 4


class PatternType(IntEnum):
    SEQUENTIAL = 0
    IF_HEADER = 1
    IF_THEN_BODY = 2
    IF_ELSE_BODY = 3
    LOOP_HEADER = 4
    LOOP_BODY = 5
    LOOP_EXIT = 6
    SWITCH_HEADER = 7
    SWITCH_CASE_BODY = 8
    FUNC_PROLOGUE_EPILOGUE = 9


class ASTType(IntEnum):
    EXPR_STMT = 0
    DECL_STMT = 1
    IF_STMT = 2
    WHILE_STMT = 3
    FOR_STMT = 4
    DO_WHILE_STMT = 5
    SWITCH_STMT = 6
    CASE_STMT = 7
    RETURN_STMT = 8
    BREAK_STMT = 9
    CONTINUE_STMT = 10
    GOTO_STMT = 11
    COMPOUND_STMT = 12
    CALL_EXPR = 13
    ASSIGN_EXPR = 14
    OTHER = 15


# compatible mapping: which PatternType can align to which ASTType
PATTERN_AST_COMPATIBILITY: Dict[int, List[int]] = {
    PatternType.SEQUENTIAL:             [ASTType.EXPR_STMT, ASTType.DECL_STMT,
                                         ASTType.ASSIGN_EXPR, ASTType.CALL_EXPR,
                                         ASTType.OTHER],
    PatternType.IF_HEADER:              [ASTType.IF_STMT],
    PatternType.IF_THEN_BODY:           [ASTType.EXPR_STMT, ASTType.DECL_STMT,
                                         ASTType.ASSIGN_EXPR, ASTType.CALL_EXPR,
                                         ASTType.COMPOUND_STMT, ASTType.OTHER],
    PatternType.IF_ELSE_BODY:           [ASTType.EXPR_STMT, ASTType.DECL_STMT,
                                         ASTType.ASSIGN_EXPR, ASTType.CALL_EXPR,
                                         ASTType.COMPOUND_STMT, ASTType.OTHER],
    PatternType.LOOP_HEADER:            [ASTType.WHILE_STMT, ASTType.FOR_STMT,
                                         ASTType.DO_WHILE_STMT],
    PatternType.LOOP_BODY:              [ASTType.EXPR_STMT, ASTType.DECL_STMT,
                                         ASTType.ASSIGN_EXPR, ASTType.CALL_EXPR,
                                         ASTType.COMPOUND_STMT, ASTType.OTHER],
    PatternType.LOOP_EXIT:              [ASTType.BREAK_STMT, ASTType.CONTINUE_STMT,
                                         ASTType.RETURN_STMT, ASTType.OTHER],
    PatternType.SWITCH_HEADER:          [ASTType.SWITCH_STMT],
    PatternType.SWITCH_CASE_BODY:       [ASTType.CASE_STMT, ASTType.EXPR_STMT,
                                         ASTType.BREAK_STMT, ASTType.OTHER],
    PatternType.FUNC_PROLOGUE_EPILOGUE: [ASTType.RETURN_STMT, ASTType.DECL_STMT,
                                         ASTType.OTHER],
}


# ═══════════════════════════ binary side ════════════════════════════════════

@dataclass
class Instruction:
    opcode: int                      # opcode id
    operand_types: List[int]         # up to 3 operand type ids
    value_buckets: List[int]         # up to 3 bucketed values
    raw_text: str = ""               # optional original disassembly text


@dataclass
class BasicBlock:
    block_id: int
    instructions: List[Instruction]
    # structural features (precomputed)
    in_degree: int = 0
    out_degree: int = 0
    is_loop_header: bool = False
    is_loop_exit: bool = False
    dominance_depth: int = 0
    post_dominance_depth: int = 0
    instruction_entropy: float = 0.0
    has_call: bool = False
    has_indirect_jump: bool = False
    dfs_position: float = 0.0
    bfs_position: float = 0.0

    def struct_features(self) -> List[float]:
        return [
            float(self.in_degree),
            float(self.out_degree),
            float(self.is_loop_header),
            float(self.is_loop_exit),
            float(self.dominance_depth),
            float(self.post_dominance_depth),
            float(len(self.instructions)) / 64.0,   # normalised block size
            self.instruction_entropy,
            float(self.has_call),
            float(self.has_indirect_jump),
            self.dfs_position,
            self.bfs_position,
        ]


@dataclass
class BinaryCFG:
    """Function-level binary control-flow graph."""
    func_name: str
    blocks: List[BasicBlock]
    edges: List[Tuple[int, int, int]]          # (src_idx, dst_idx, EdgeType)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


# ═══════════════════════════ source side ════════════════════════════════════

@dataclass
class SourceStatement:
    stmt_id: int
    ast_type: int                              # ASTType
    token_ids: List[int]                       # tokenised source (placeholder-ised)
    text: str = ""

    @property
    def max_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class SourceCFG:
    """Function-level source control-flow graph."""
    func_name: str
    statements: List[SourceStatement]
    edges: List[Tuple[int, int, int]]          # (src_idx, dst_idx, edge_type)

    @property
    def num_statements(self) -> int:
        return len(self.statements)


# ═══════════════════════════ paired data ════════════════════════════════════

@dataclass
class AlignmentLabel:
    """Ground-truth node alignment from DWARF."""
    matrix: torch.Tensor                       # [n_b, n_s] float32 (soft)
    confidence: torch.Tensor                   # [n_b, n_s] float32


@dataclass
class FunctionPair:
    binary_cfg: BinaryCFG
    source_cfg: Optional[SourceCFG] = None     # None at inference
    alignment: Optional[AlignmentLabel] = None
    pattern_labels: Optional[torch.Tensor] = None   # [n_b] int64
    compiler: str = ""
    opt_level: str = ""
    arch: str = ""


# ═══════════════════════════ skeleton ═══════════════════════════════════════

@dataclass
class SlotInfo:
    slot_id: str                               # e.g. "slot_0"
    slot_type: str                             # FILL_COND / FILL_STMT / FILL_EXPR / …
    aligned_bb_indices: List[int]              # which BBs this slot maps to
    binary_context_text: str = ""              # key instructions as text


@dataclass
class Skeleton:
    skeleton_text: str                         # code with <FILL_*> placeholders
    slots: List[SlotInfo] = field(default_factory=list)
    # §6.5 推理时的「骨架可靠性分数」R ∈ [0, 1]，由 ``compute_pattern_reliability``
    # 根据 Module A 的 pattern 分布与建树自检统计算出。训练数据路径（pattern 由 GT
    # 给出）通常为 ``None``。详见 ``struclift/utils/skeleton.py``。
    reliability: Optional[float] = None
    # 建树过程统计（``n_body_unattached`` 等），用于诊断与可选的 RL/解码门控。
    build_stats: Optional[Dict[str, int]] = None

    def get_slots_in_order(self) -> List[SlotInfo]:
        return list(self.slots)

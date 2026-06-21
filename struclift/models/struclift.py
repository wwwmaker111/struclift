"""
StrucLift — Full Model Assembly.

Combines all four modules into a single ``nn.Module`` for end-to-end training
and inference:

  Module A  →  Structure-Aware CFG Encoders  (binary + source)
  Module B  →  Multi-Granularity Cross-Graph Alignment (SCOT)
  Module C  →  Structure-Conditioned LLM Decoder
  Module D  →  Structural Consistency RL  (standalone trainer, not part of this nn.Module)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import itertools
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from struclift.config import StrucLiftConfig
from struclift.data.structures import PatternType, Skeleton
from struclift.models.module_a import BinaryCFGEncoder, SourceCFGEncoder
from struclift.models.module_b import CrossGraphAlignmentModule, ModuleBOutput
from struclift.models.module_c import StructureConditionedDecoder
from struclift.utils.graph_utils import (
    compute_binary_regions_dominator_tree,
    compute_binary_regions_structural,
    compute_region_segmentation,
    compute_source_regions_ast,
    compute_source_regions_compound,
)
from struclift.utils.skeleton import build_bb_opcode_hints_from_batch


# ═══════════════════════════ output containers ════════════════════════════════

@dataclass
class StrucLiftEncoderOutput:
    """Output of the encoder pipeline (Modules A + B)."""
    binary_node_embs: torch.Tensor      # [n_b, d]
    source_node_embs: torch.Tensor      # [n_s, d]  (None during inference)
    binary_graph_emb: torch.Tensor      # [1, d]
    source_graph_emb: Optional[torch.Tensor]  # [1, d]
    pattern_logits: torch.Tensor        # [n_b, C]
    alignment: Optional[ModuleBOutput]  # None in stage-1 pretraining
    # Per-graph (batch index) lists when run_alignment=True and batch>1
    cross_attn_weights_list: Optional[List[torch.Tensor]] = None
    region_bin_embs_list: Optional[List[torch.Tensor]] = None
    region_src_embs_list: Optional[List[torch.Tensor]] = None
    region_positive_mask_list: Optional[List[torch.Tensor]] = None
    # §6.5：与 batch 中每个图一一对应（可能含 None）
    skeleton_list: Optional[List[Optional[Skeleton]]] = None
    # §6.5.3：推理时预测的源码语句数
    predicted_n_s_list: Optional[List[Optional[torch.Tensor]]] = None


@dataclass
class StrucLiftOutput:
    """Output of the full StrucLift forward pass."""
    encoder_out: StrucLiftEncoderOutput
    logits: Optional[torch.Tensor]           # [B, L, V]  from LLM
    lm_loss: Optional[torch.Tensor]          # scalar
    pattern_loss: Optional[torch.Tensor]     # scalar
    contrastive_loss: Optional[torch.Tensor] # scalar


def batched_graph_kv_from_nodes(
    node_embs: torch.Tensor,
    bin_batch: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Pack flat ``[total_nodes, d]`` encoder outputs into ``[B, max_n_bb, d]``.

    Used when Module B has concatenated all graphs back into one tensor but the
    decoder needs one row of KV per batch element (proposal §7.3.2 slot mask).
    """
    d = node_embs.size(-1)
    device = node_embs.device
    dtype = node_embs.dtype
    counts = [(bin_batch == gi).sum().item() for gi in range(batch_size)]
    max_n = max(counts) if counts else 1
    kv = torch.zeros(batch_size, max_n, d, device=device, dtype=dtype)
    for gi in range(batch_size):
        sel = node_embs[bin_batch == gi]
        n = sel.size(0)
        if n > 0:
            kv[gi, :n] = sel
    return kv


# ═══════════════════════════ main model ═══════════════════════════════════════

class StrucLift(nn.Module):
    """Full StrucLift model combining Modules A, B, and C.

    Module D (GRPO RL trainer) is a standalone class and not included here;
    it wraps this model externally during Stage 4 training.

    Parameters
    ----------
    config : StrucLiftConfig
        Top-level configuration object.
    """

    def __init__(self, config: StrucLiftConfig) -> None:
        super().__init__()
        self.config = config

        # ── Module A: encoders ───────────────────────────────────────────
        self.binary_encoder = BinaryCFGEncoder(
            vcfg=config.vocab, ma_cfg=config.module_a,
        )
        self.source_encoder = SourceCFGEncoder(
            vcfg=config.vocab, ma_cfg=config.module_a,
        )

        # 图级向量仅使用各 encoder 内部的 PMAPooling（与提案 / module_a 文档一致）

        # ── Module B: alignment ──────────────────────────────────────────
        self.alignment_module = CrossGraphAlignmentModule(
            cfg=config.module_b,
            vocab_cfg=config.vocab,
        )

        # ── Module C: decoder (adapters only; LLM loaded externally) ─────
        self.decoder = StructureConditionedDecoder(config=config.module_c)

    # ─────────────────────────────────────────────────────────────────────
    # Encoder-only forward (Stages 1 & 2)
    # ─────────────────────────────────────────────────────────────────────

    def encode(
        self,
        # binary side
        bin_opcodes: torch.Tensor,
        bin_operand_types: torch.Tensor,
        bin_operand_values: torch.Tensor,
        bin_block_lengths: torch.Tensor,
        bin_edge_index: torch.Tensor,
        bin_edge_type: torch.Tensor,
        bin_struct_features: torch.Tensor,
        bin_batch: Optional[torch.Tensor] = None,
        # source side (optional — absent during inference)
        src_token_ids: Optional[torch.Tensor] = None,
        src_stmt_lengths: Optional[torch.Tensor] = None,
        src_edge_index: Optional[torch.Tensor] = None,
        src_edge_type: Optional[torch.Tensor] = None,
        src_ast_types: Optional[torch.Tensor] = None,
        src_compound_id: Optional[torch.Tensor] = None,
        src_batch: Optional[torch.Tensor] = None,
        # alignment inputs
        bin_adj: Optional[torch.Tensor] = None,
        src_adj: Optional[torch.Tensor] = None,
        M_gt: Optional[torch.Tensor] = None,
        run_alignment: bool = False,
        # ── Stage2 encoder cache 旁路（与 Module A forward 数值等价）─────────
        # 全部为 None → 走完整 Module A；bin 三件套同时非 None → 跳过
        # binary_encoder（含 SubgraphPatternClassifier）；src 二件套同时非
        # None → 跳过 source_encoder。仅在 Module A 冻结的 Stage2 / 推理路径
        # 下使用；几何字段（edge_index / batch / adj 等）仍从 batch 传入。
        precomputed_bin_node_embs: Optional[torch.Tensor] = None,
        precomputed_pattern_logits: Optional[torch.Tensor] = None,
        precomputed_bin_graph_emb: Optional[torch.Tensor] = None,
        precomputed_src_node_embs: Optional[torch.Tensor] = None,
        precomputed_src_graph_emb: Optional[torch.Tensor] = None,
    ) -> StrucLiftEncoderOutput:
        """Run Modules A and optionally B.

        Parameters
        ----------
        bin_opcodes : [N_instr]
        bin_operand_types : [N_instr, 3]
        bin_operand_values : [N_instr, 3]
        bin_block_lengths : [n_b]  number of instructions per block
        bin_edge_index : [2, E_b]
        bin_edge_type : [E_b]
        bin_struct_features : [n_b, 12]
        bin_batch : [n_b]  batch assignment (for PyG batching)
        src_token_ids : [N_stmt_tokens]
        src_stmt_lengths : [n_s]
        src_edge_index : [2, E_s]
        src_edge_type : [E_s]
        src_ast_types : [n_s]
        src_compound_id : [n_s]  optional, -1 = 无 compound 父节点
        src_batch : [n_s]
        bin_adj : [n_b, n_b]  adjacency / distance matrix
        src_adj : [n_s, n_s]
        run_alignment : bool  whether to run Module B

        Returns
        -------
        StrucLiftEncoderOutput
        """
        device = bin_opcodes.device
        n_b = bin_block_lengths.shape[0]
        max_L = self.config.vocab.max_instructions_per_block   # 64

        # ── Stage2 encoder cache 旁路：bin 三件套都给 → 完全跳过 binary_encoder
        _bin_cached = (
            precomputed_bin_node_embs is not None
            and precomputed_pattern_logits is not None
            and precomputed_bin_graph_emb is not None
        )

        if bin_batch is None:
            bin_batch = torch.zeros(n_b, dtype=torch.long, device=device)

        # Stage2 冻结 Module A 且无需反传 encoder 时，用 no_grad 跳过无谓 autograd 图构建，
        # 缩短 encoder 前向与后续 backward（alignment）的有效计算路径。
        _ma_grad = any(
            p.requires_grad
            for p in itertools.chain(
                self.binary_encoder.parameters(),
                self.source_encoder.parameters(),
            )
        )
        _ma_ctx = nullcontext if _ma_grad else torch.no_grad

        # 可选：encoder 冻结时用 bf16 跑 forward，alignment 仍 fp32（不同于 fp16，bf16 指数位
        # 与 fp32 一致，较稳；环境变量 ``STRUCLIFT_ENCODER_BF16=1`` 启用）。仅 Stage2/推理
        # 这种「encoder 不反传」的场景安全，因为 bwd 路径里没有 bf16 算子。
        _encoder_bf16 = (
            (not _ma_grad)
            and device.type == "cuda"
            and os.environ.get("STRUCLIFT_ENCODER_BF16", "").strip() == "1"
            and torch.cuda.is_bf16_supported()
        )

        if _bin_cached:
            # 直接采用预计算（与 binary_encoder.forward 数值等价；要求生成 cache 时使用
            # 与本次 run **同一份** Stage1 权重）。dtype 强制 fp32，与 alignment 一致。
            bin_node_embs = precomputed_bin_node_embs.to(device=device, dtype=torch.float32)
            pattern_logits = precomputed_pattern_logits.to(device=device, dtype=torch.float32)
            bin_graph_emb = precomputed_bin_graph_emb.to(device=device, dtype=torch.float32)
        else:
            # ── 二进制侧：Doc §5.2.2 归一化位置 i/|BB| + 正弦编码 ──
            # 与原 host-side for+小 kernel 等价；纯张量构造，避免 launch 过多小 kernel。
            n_total_instr = bin_opcodes.shape[0]
            if n_b > 0:
                bl_long = bin_block_lengths.long()
                block_id_per_instr = torch.repeat_interleave(
                    torch.arange(n_b, device=device, dtype=torch.long), bl_long,
                )  # [n_total_instr]
                block_starts = bl_long.cumsum(0) - bl_long  # [n_b]
                pos_within = (
                    torch.arange(n_total_instr, device=device, dtype=torch.long)
                    - block_starts[block_id_per_instr]
                )
                denom = bl_long.to(torch.float32).clamp(min=1.0)
                instr_norm_positions = pos_within.to(torch.float32) / denom[block_id_per_instr]
            else:
                instr_norm_positions = torch.zeros(0, dtype=torch.float32, device=device)

            # block_instr_idx / instr_mask: 每个 block 取前 min(bl, max_L) 条指令
            # 与原版完全等价：padding 处 mask=False，gather 索引取本 block 起点（被 mask 屏蔽，不影响计算）。
            raw_max = int(bin_block_lengths.max().item()) if n_b > 0 else 1
            capped_L = min(raw_max, max_L)
            if n_b > 0 and capped_L > 0:
                positions = torch.arange(capped_L, device=device, dtype=torch.long).unsqueeze(0)  # [1, L]
                bl_capped = bin_block_lengths.clamp(max=max_L)  # [n_b]
                instr_mask = positions < bl_capped.unsqueeze(1)  # [n_b, L] bool
                offsets = bin_block_lengths.cumsum(0) - bin_block_lengths  # [n_b]
                block_instr_idx = offsets.unsqueeze(1) + positions  # [n_b, L]
                # padding 处 mask=False，但 gather 仍会读 idx；钳到合法范围避免越界
                if n_total_instr > 0:
                    block_instr_idx = block_instr_idx.clamp(max=n_total_instr - 1)
            else:
                block_instr_idx = torch.zeros(n_b, capped_L, dtype=torch.long, device=device)
                instr_mask = torch.zeros(n_b, capped_L, dtype=torch.bool, device=device)

            # ── Module A: binary encoder ────────────────────────────────────
            with _ma_ctx(), torch.amp.autocast(
                "cuda", enabled=_encoder_bf16, dtype=torch.bfloat16,
            ):
                bin_enc_out = self.binary_encoder(
                    opcodes=bin_opcodes,
                    operand_types=bin_operand_types,
                    value_buckets=bin_operand_values,
                    instr_norm_positions=instr_norm_positions,
                    block_instr_idx=block_instr_idx,
                    instr_mask=instr_mask,
                    struct_feats=bin_struct_features,
                    edge_index=bin_edge_index,
                    edge_type=bin_edge_type,
                    batch=bin_batch,
                )
            bin_node_embs = bin_enc_out["node_embeddings"]      # [n_b, d]
            pattern_logits = bin_enc_out["pattern_logits"]      # [n_b, C]
            bin_graph_emb = bin_enc_out["graph_embeddings"]     # [B_graphs, d]
            if _encoder_bf16:
                # 把 alignment 的输入 cast 回 fp32，避免 bf16/fp32 混用引发 alignment 内部错误
                bin_node_embs = bin_node_embs.to(torch.float32)
                pattern_logits = pattern_logits.to(torch.float32)
                bin_graph_emb = bin_graph_emb.to(torch.float32)

        # ── Module A: 源码侧 §5.4（token 已占位化 → 词表 id）────────────────
        src_node_embs = None
        src_graph_emb = None
        # 缓存命中：src_node_embs / src_graph_emb 同时给出即可跳过 source_encoder
        _src_cached = (
            precomputed_src_node_embs is not None
            and precomputed_src_graph_emb is not None
        )
        if _src_cached:
            src_node_embs = precomputed_src_node_embs.to(device=device, dtype=torch.float32)
            src_graph_emb = precomputed_src_graph_emb.to(device=device, dtype=torch.float32)
            if src_batch is None and src_stmt_lengths is not None:
                n_s = src_stmt_lengths.shape[0]
                src_batch = torch.zeros(n_s, dtype=torch.long, device=device)
        elif src_token_ids is not None:
            n_s = src_stmt_lengths.shape[0]
            max_tok_limit = self.config.vocab.max_instructions_per_block  # 64
            max_tok_raw = int(src_stmt_lengths.max().item()) if n_s > 0 else 1
            max_tok = min(max_tok_raw, max_tok_limit)

            # 与原版等价：每条 stmt 的前 min(sl, max_tok) 个 token；其余位置 mask=False
            if n_s > 0 and max_tok > 0:
                sl_long = src_stmt_lengths.long()
                sl_capped = sl_long.clamp(max=max_tok_limit)  # [n_s]
                positions_t = torch.arange(max_tok, device=device, dtype=torch.long).unsqueeze(0)
                src_token_mask = positions_t < sl_capped.unsqueeze(1)  # [n_s, max_tok]
                tok_offsets = sl_long.cumsum(0) - sl_long  # [n_s]
                idx_full = tok_offsets.unsqueeze(1) + positions_t  # [n_s, max_tok]
                n_tok_total = src_token_ids.shape[0]
                if n_tok_total > 0:
                    idx_safe = idx_full.clamp(max=n_tok_total - 1)
                else:
                    idx_safe = torch.zeros_like(idx_full)
                src_token_2d = src_token_ids[idx_safe.reshape(-1)].reshape(n_s, max_tok)
                # padding 处用 0 token（与原 zeros 初始化一致）
                src_token_2d = torch.where(
                    src_token_mask, src_token_2d, torch.zeros_like(src_token_2d),
                )
            else:
                src_token_2d = torch.zeros(n_s, max_tok, dtype=torch.long, device=device)
                src_token_mask = torch.zeros(n_s, max_tok, dtype=torch.bool, device=device)

            if src_batch is None:
                src_batch = torch.zeros(n_s, dtype=torch.long, device=device)

            with _ma_ctx(), torch.amp.autocast(
                "cuda", enabled=_encoder_bf16, dtype=torch.bfloat16,
            ):
                src_enc_out = self.source_encoder(
                    token_ids=src_token_2d,
                    token_mask=src_token_mask,
                    ast_types=src_ast_types,
                    edge_index=src_edge_index,
                    edge_type=src_edge_type,
                    batch=src_batch,
                )
                src_node_embs = src_enc_out["node_embeddings"]
                src_graph_emb = src_enc_out["graph_embeddings"]
            if _encoder_bf16:
                src_node_embs = src_node_embs.to(torch.float32)
                if src_graph_emb is not None:
                    src_graph_emb = src_graph_emb.to(torch.float32)

        # ── Module B: alignment (optional) ──────────────────────────────
        # Module B 逐样本运行（每个样本有独立的 adj 和 alignment 矩阵）
        alignment_out = None
        cross_attn_weights_list: Optional[List[torch.Tensor]] = None
        region_bin_embs_list: Optional[List[Optional[torch.Tensor]]] = None
        region_src_embs_list: Optional[List[Optional[torch.Tensor]]] = None
        region_positive_mask_list: Optional[List[Optional[torch.Tensor]]] = None
        skeleton_list: Optional[List[Optional[Skeleton]]] = None
        predicted_n_s_list: Optional[List[Optional[torch.Tensor]]] = None
        if run_alignment and src_node_embs is not None:
            patterns_all = pattern_logits.argmax(dim=-1)
            ast_types_for_b = src_ast_types if src_ast_types is not None else torch.zeros(
                src_node_embs.size(0), dtype=torch.long, device=device,
            )
            compound_for_b = src_compound_id

            B_graphs = int(bin_batch.max().item()) + 1 if bin_batch.numel() > 0 else 1

            per_sample_bin_embs = []
            per_sample_src_embs = []
            per_sample_M = []
            per_sample_cross_attn: List[torch.Tensor] = []
            per_sample_region_bin: List[Optional[torch.Tensor]] = []
            per_sample_region_src: List[Optional[torch.Tensor]] = []
            per_sample_region_mask: List[Optional[torch.Tensor]] = []
            per_sample_skeleton: List[Optional[Skeleton]] = []
            per_sample_pred_ns: List[Optional[torch.Tensor]] = []

            # 逐图筛边：原先对每条全局边 Python 循环 + .item()，GPU 侧利用率极低。
            # PyG batch 下用 bin_batch[src]==gi & bin_batch[dst]==gi 一次向量化掩码，
            # 再用 inv_* 将全局节点号映射到该图局部下标（避免逐边 Python remap）。
            inv_bin_buf = torch.full(
                (bin_batch.numel(),), -1, dtype=torch.long, device=device,
            )
            inv_src_buf = torch.full(
                (src_batch.numel(),), -1, dtype=torch.long, device=device,
            )

            # —— Cross-graph batched alignment (D) ——
            # 默认走「prepare → cross-graph batched SCOT → finalize」三阶段路径，
            # 把 batch 内 64 图的 SCOT 子问题（每图 1 background + N region pair 或
            # 1 整图）一次 padded batched 求解，节省 per-graph alignment forward 的
            # launch 累积。环境变量 ``STRUCLIFT_ALIGN_PER_GRAPH=1`` 可回退到原版
            # 单图 forward 路径（用于回归测试 / 数值对比）。
            _align_per_graph = (
                os.environ.get("STRUCLIFT_ALIGN_PER_GRAPH", "").strip() == "1"
            )

            # 第一遍循环：每图准备输入（mask / regions / edge / adj / hints），
            # 走 prepare 路径时同时收集 SCOT 子问题；走老路径时直接 forward。
            prepared_list: List[Optional[Dict]] = []
            # gi -> (nb_i, ns_i) for graphs that are empty (n_b==0 or n_s==0)
            empty_shapes: Dict[int, Tuple[int, int]] = {}

            for gi in range(B_graphs):
                b_mask = (bin_batch == gi)
                s_mask = (src_batch == gi)
                b_embs_i = bin_node_embs[b_mask]
                s_embs_i = src_node_embs[s_mask]
                patterns_i = patterns_all[b_mask]
                pattern_logits_i = pattern_logits[b_mask]
                ast_types_i = ast_types_for_b[s_mask]
                compound_i: Optional[torch.Tensor] = None
                if (
                    compound_for_b is not None
                    and compound_for_b.numel() == ast_types_for_b.numel()
                ):
                    compound_i = compound_for_b[s_mask]
                nb_i = b_embs_i.size(0)
                ns_i = s_embs_i.size(0)

                b_nodes_i = b_mask.nonzero(as_tuple=True)[0]
                inv_bin_buf[b_nodes_i] = torch.arange(
                    nb_i, device=device, dtype=torch.long,
                )

                be_mask = (bin_batch[bin_edge_index[0]] == gi) & (
                    bin_batch[bin_edge_index[1]] == gi
                )
                if be_mask.any():
                    be_glob = bin_edge_index[:, be_mask]
                    be_i = torch.stack(
                        [inv_bin_buf[be_glob[0]], inv_bin_buf[be_glob[1]]], dim=0,
                    )
                else:
                    be_i = torch.zeros(2, 0, dtype=torch.long, device=device)

                s_nodes_i = s_mask.nonzero(as_tuple=True)[0]
                inv_src_buf[s_nodes_i] = torch.arange(
                    ns_i, device=device, dtype=torch.long,
                )

                if src_edge_index is not None:
                    se_mask = (src_batch[src_edge_index[0]] == gi) & (
                        src_batch[src_edge_index[1]] == gi
                    )
                    if se_mask.any():
                        se_glob = src_edge_index[:, se_mask]
                        se_i = torch.stack(
                            [inv_src_buf[se_glob[0]], inv_src_buf[se_glob[1]]], dim=0,
                        )
                    else:
                        se_i = torch.zeros(2, 0, dtype=torch.long, device=device)
                else:
                    se_i = torch.zeros(2, 0, dtype=torch.long, device=device)

                # 逐样本 adj
                ba_i = None
                sa_i = None
                if bin_adj is not None and bin_adj.dim() == 3:
                    ba_i = bin_adj[gi, :nb_i, :nb_i]
                elif bin_adj is not None and bin_adj.dim() == 2:
                    ba_i = bin_adj[:nb_i, :nb_i]
                if src_adj is not None and src_adj.dim() == 3:
                    sa_i = src_adj[gi, :ns_i, :ns_i]
                elif src_adj is not None and src_adj.dim() == 2:
                    sa_i = src_adj[:ns_i, :ns_i]

                # Doc §6.2.2: binary — dominator + natural loops; source — AST headers
                try:
                    bin_regions_i = compute_binary_regions_structural(
                        patterns_i, be_i, nb_i,
                    )
                except Exception:
                    try:
                        bin_regions_i = compute_binary_regions_dominator_tree(
                            patterns_i, be_i, nb_i,
                        )
                    except Exception:
                        bin_regions_i = compute_region_segmentation(
                            patterns_i, be_i, nb_i,
                        )
                if se_i.size(1) > 0:
                    try:
                        if compound_i is not None and compound_i.numel() == ns_i:
                            src_regions_i = compute_source_regions_compound(
                                ast_types_i, se_i, ns_i, compound_i,
                            )
                        else:
                            src_regions_i = compute_source_regions_ast(
                                ast_types_i, se_i, ns_i,
                            )
                    except Exception:
                        src_regions_i = compute_region_segmentation(
                            ast_types_i, se_i, ns_i,
                        )
                else:
                    src_regions_i = []

                M_gt_i: Optional[torch.Tensor] = None
                if M_gt is not None:
                    if M_gt.dim() == 3:
                        M_gt_i = M_gt[gi, :nb_i, :ns_i]
                    elif B_graphs == 1:
                        M_gt_i = M_gt[:nb_i, :ns_i]
                    else:
                        M_gt_i = None

                bb_hints = build_bb_opcode_hints_from_batch(
                    gi, bin_batch, bin_block_lengths, bin_opcodes,
                )

                if _align_per_graph:
                    # —— 老路径：单图 forward —— (留作 fallback / 数值回归对比)
                    align_i = self.alignment_module(
                        binary_node_embs=b_embs_i,
                        source_node_embs=s_embs_i,
                        patterns=patterns_i,
                        ast_types=ast_types_i,
                        binary_adj=ba_i,
                        source_adj=sa_i,
                        binary_regions=bin_regions_i if bin_regions_i else None,
                        source_regions=src_regions_i if src_regions_i else None,
                        M_gt=M_gt_i,
                        pattern_logits=pattern_logits_i,
                        binary_edge_index=be_i,
                        block_instructions_text=bb_hints,
                    )
                    per_sample_bin_embs.append(align_i.binary_node_embs)
                    per_sample_src_embs.append(align_i.source_node_embs)
                    per_sample_M.append(align_i.alignment_matrix)
                    per_sample_cross_attn.append(
                        align_i.cross_attn_weights
                        if align_i.cross_attn_weights is not None
                        else torch.zeros(nb_i, ns_i, device=device),
                    )
                    per_sample_region_bin.append(align_i.region_bin_embs)
                    per_sample_region_src.append(align_i.region_src_embs)
                    per_sample_region_mask.append(align_i.region_positive_mask)
                    per_sample_skeleton.append(align_i.skeleton)
                    per_sample_pred_ns.append(align_i.predicted_n_s)
                    prepared_list.append(None)  # 占位（新路径不读）
                    continue

                # 新路径：空图占位（finalize 阶段补零张量）
                if nb_i == 0 or ns_i == 0:
                    prepared_list.append(None)
                    empty_shapes[gi] = (nb_i, ns_i)
                    continue

                prep = self.alignment_module.prepare(
                    binary_node_embs=b_embs_i,
                    source_node_embs=s_embs_i,
                    patterns=patterns_i,
                    ast_types=ast_types_i,
                    binary_adj=ba_i,
                    source_adj=sa_i,
                    binary_regions=bin_regions_i if bin_regions_i else None,
                    source_regions=src_regions_i if src_regions_i else None,
                    M_gt=M_gt_i,
                    pattern_logits=pattern_logits_i,
                    binary_edge_index=be_i,
                    block_instructions_text=bb_hints,
                )
                prepared_list.append(prep)

            # 老路径已经填好 per_sample_* 列表，跳过新路径剩余阶段
            if not _align_per_graph:
                # —— 跨图收集所有 SCOT 子问题，一次 batched call —— ——
                all_sub_problems: List[Dict] = []
                graph_sub_offsets: List[Tuple[int, int]] = []
                for prep in prepared_list:
                    start = len(all_sub_problems)
                    if prep is not None:
                        all_sub_problems.extend(prep["sub_problems"])
                    graph_sub_offsets.append((start, len(all_sub_problems)))

                all_M_subs = self.alignment_module._run_scot_batch_cross(
                    all_sub_problems,
                )

                # 第二遍循环：每图 finalize（cross_attn / region_pos_mask / skeleton 等）
                for gi in range(B_graphs):
                    prep = prepared_list[gi]
                    if prep is None:
                        # 空图占位
                        nb_i, ns_i = empty_shapes.get(gi, (0, 0))
                        per_sample_bin_embs.append(
                            torch.zeros(nb_i, bin_node_embs.size(1), device=device,
                                        dtype=bin_node_embs.dtype),
                        )
                        per_sample_src_embs.append(
                            torch.zeros(ns_i, src_node_embs.size(1), device=device,
                                        dtype=src_node_embs.dtype),
                        )
                        per_sample_M.append(
                            torch.zeros(nb_i, ns_i, device=device, dtype=bin_node_embs.dtype),
                        )
                        per_sample_cross_attn.append(
                            torch.zeros(nb_i, ns_i, device=device),
                        )
                        per_sample_region_bin.append(None)
                        per_sample_region_src.append(None)
                        per_sample_region_mask.append(None)
                        per_sample_skeleton.append(None)
                        per_sample_pred_ns.append(None)
                        continue
                    st, ed = graph_sub_offsets[gi]
                    align_i = self.alignment_module.finalize(
                        prep, all_M_subs[st:ed],
                    )
                    nb_i = prep["n_b"]
                    ns_i = prep["n_s"]
                    per_sample_bin_embs.append(align_i.binary_node_embs)
                    per_sample_src_embs.append(align_i.source_node_embs)
                    per_sample_M.append(align_i.alignment_matrix)
                    per_sample_cross_attn.append(
                        align_i.cross_attn_weights
                        if align_i.cross_attn_weights is not None
                        else torch.zeros(nb_i, ns_i, device=device),
                    )
                    per_sample_region_bin.append(align_i.region_bin_embs)
                    per_sample_region_src.append(align_i.region_src_embs)
                    per_sample_region_mask.append(align_i.region_positive_mask)
                    per_sample_skeleton.append(align_i.skeleton)
                    per_sample_pred_ns.append(align_i.predicted_n_s)

            bin_node_embs = torch.cat(per_sample_bin_embs, dim=0)
            src_node_embs = torch.cat(per_sample_src_embs, dim=0)

            from struclift.models.module_b import ModuleBOutput
            alignment_out = ModuleBOutput(
                alignment_matrix=per_sample_M,
                binary_node_embs=bin_node_embs,
                source_node_embs=src_node_embs,
                region_alignment=None,
                skeleton=None,
            )
            cross_attn_weights_list = per_sample_cross_attn
            region_bin_embs_list = per_sample_region_bin
            region_src_embs_list = per_sample_region_src
            region_positive_mask_list = per_sample_region_mask
            skeleton_list = per_sample_skeleton
            predicted_n_s_list = per_sample_pred_ns

        return StrucLiftEncoderOutput(
            binary_node_embs=bin_node_embs,
            source_node_embs=src_node_embs,
            binary_graph_emb=bin_graph_emb,
            source_graph_emb=src_graph_emb,
            pattern_logits=pattern_logits,
            alignment=alignment_out,
            cross_attn_weights_list=cross_attn_weights_list,
            region_bin_embs_list=region_bin_embs_list,
            region_src_embs_list=region_src_embs_list,
            region_positive_mask_list=region_positive_mask_list,
            skeleton_list=skeleton_list,
            predicted_n_s_list=predicted_n_s_list if run_alignment else None,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Full forward (Stage 3 SFT)
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        # binary encoder inputs
        bin_opcodes: torch.Tensor,
        bin_operand_types: torch.Tensor,
        bin_operand_values: torch.Tensor,
        bin_block_lengths: torch.Tensor,
        bin_edge_index: torch.Tensor,
        bin_edge_type: torch.Tensor,
        bin_struct_features: torch.Tensor,
        bin_batch: Optional[torch.Tensor] = None,
        # source encoder inputs
        src_token_ids: Optional[torch.Tensor] = None,
        src_stmt_lengths: Optional[torch.Tensor] = None,
        src_edge_index: Optional[torch.Tensor] = None,
        src_edge_type: Optional[torch.Tensor] = None,
        src_ast_types: Optional[torch.Tensor] = None,
        src_compound_id: Optional[torch.Tensor] = None,
        src_batch: Optional[torch.Tensor] = None,
        # alignment
        bin_adj: Optional[torch.Tensor] = None,
        src_adj: Optional[torch.Tensor] = None,
        M_gt: Optional[torch.Tensor] = None,
        run_alignment: bool = True,
        # decoder inputs
        input_ids: Optional[torch.Tensor] = None,
        graph_kv: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        slot_weight_mask: Optional[torch.Tensor] = None,
        # encoder cache 旁路（透传 encode）
        precomputed_bin_node_embs: Optional[torch.Tensor] = None,
        precomputed_pattern_logits: Optional[torch.Tensor] = None,
        precomputed_bin_graph_emb: Optional[torch.Tensor] = None,
        precomputed_src_node_embs: Optional[torch.Tensor] = None,
        precomputed_src_graph_emb: Optional[torch.Tensor] = None,
    ) -> StrucLiftOutput:
        """Full forward pass through all modules.

        Parameters
        ----------
        (encoder params): see :meth:`encode`
        input_ids : [B, L]  tokenised skeleton
        graph_kv : [B, N, d_graph]  (if None, built from encoder output)
        slot_mask : [B, L, N]
        labels : [B, L]  target tokens (-100 for ignored positions)
        slot_weight_mask : [B, L]  per-token weights

        Returns
        -------
        StrucLiftOutput
        """
        # ── encode ──────────────────────────────────────────────────────
        enc_out = self.encode(
            bin_opcodes=bin_opcodes,
            bin_operand_types=bin_operand_types,
            bin_operand_values=bin_operand_values,
            bin_block_lengths=bin_block_lengths,
            bin_edge_index=bin_edge_index,
            bin_edge_type=bin_edge_type,
            bin_struct_features=bin_struct_features,
            bin_batch=bin_batch,
            src_token_ids=src_token_ids,
            src_stmt_lengths=src_stmt_lengths,
            src_edge_index=src_edge_index,
            src_edge_type=src_edge_type,
            src_ast_types=src_ast_types,
            src_compound_id=src_compound_id,
            src_batch=src_batch,
            bin_adj=bin_adj,
            src_adj=src_adj,
            M_gt=M_gt,
            run_alignment=run_alignment,
            precomputed_bin_node_embs=precomputed_bin_node_embs,
            precomputed_pattern_logits=precomputed_pattern_logits,
            precomputed_bin_graph_emb=precomputed_bin_graph_emb,
            precomputed_src_node_embs=precomputed_src_node_embs,
            precomputed_src_graph_emb=precomputed_src_graph_emb,
        )

        # ── decode (if decoder inputs provided) ─────────────────────────
        logits = None
        lm_loss = None
        if input_ids is not None:
            if graph_kv is None:
                bb = bin_batch
                if bb is None:
                    n = enc_out.binary_node_embs.size(0)
                    bb = torch.zeros(
                        n, dtype=torch.long, device=enc_out.binary_node_embs.device,
                    )
                B_dec = input_ids.size(0)
                graph_kv = batched_graph_kv_from_nodes(
                    enc_out.binary_node_embs, bb, batch_size=B_dec,
                )

            logits, lm_loss = self.decoder(
                input_ids=input_ids,
                graph_kv=graph_kv,
                slot_mask=slot_mask if slot_mask is not None else torch.ones(
                    input_ids.size(0), input_ids.size(1),
                    graph_kv.size(1), device=input_ids.device,
                ),
                labels=labels,
                slot_weight_mask=slot_weight_mask,
            )

        return StrucLiftOutput(
            encoder_out=enc_out,
            logits=logits,
            lm_loss=lm_loss,
            pattern_loss=None,       # computed externally in training loop
            contrastive_loss=None,   # computed externally in training loop
        )

    # ─────────────────────────────────────────────────────────────────────
    # Utility: freeze / unfreeze module groups
    # ─────────────────────────────────────────────────────────────────────

    def freeze_encoders(self) -> None:
        """Freeze Module A (both binary and source encoders)."""
        for p in self.binary_encoder.parameters():
            p.requires_grad = False
        for p in self.source_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze Module A."""
        for p in self.binary_encoder.parameters():
            p.requires_grad = True
        for p in self.source_encoder.parameters():
            p.requires_grad = True

    def freeze_alignment(self) -> None:
        """Freeze Module B."""
        for p in self.alignment_module.parameters():
            p.requires_grad = False

    def unfreeze_alignment(self) -> None:
        """Unfreeze Module B."""
        for p in self.alignment_module.parameters():
            p.requires_grad = True

    def freeze_decoder_adapters(self) -> None:
        """Freeze Module C adapters."""
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_decoder_adapters(self) -> None:
        """Unfreeze Module C adapters."""
        for p in self.decoder.parameters():
            p.requires_grad = True

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return all parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module group."""
        def _count(module: nn.Module) -> Tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        t_bin, tr_bin = _count(self.binary_encoder)
        t_src, tr_src = _count(self.source_encoder)
        t_align, tr_align = _count(self.alignment_module)
        t_dec, tr_dec = _count(self.decoder)

        return {
            "binary_encoder_total": t_bin,
            "binary_encoder_trainable": tr_bin,
            "source_encoder_total": t_src,
            "source_encoder_trainable": tr_src,
            "alignment_total": t_align,
            "alignment_trainable": tr_align,
            "decoder_total": t_dec,
            "decoder_trainable": tr_dec,
            "total": t_bin + t_src + t_align + t_dec,
            "trainable": tr_bin + tr_src + tr_align + tr_dec,
        }

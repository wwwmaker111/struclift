"""
Module A -- Structure-Aware Binary CFG Encoder & Source CFG Encoder.

This module encodes binary and source control-flow graphs into fixed-dimension
vector representations suitable for downstream alignment (Module B) and LLM
decoding (Module C).

Binary side pipeline
--------------------
1. InstructionEmbedding    -> per-instruction vectors   [*, d]
2. BlockTransformer        -> contextualised instr vecs [*, d]
3. AttentivePooling        -> one vector per block      [N, d]
4. StructuralFeatureMLP    -> structural bias           [N, d]
5. EdgeTypedGATLayer x L   -> graph-level node embeds   [N, d]  (有向 CFG：消息沿 src→dst)
6. SubgraphPatternClassifier -> pattern logits          [N, C]  (k-hop = 有向最短路 ≤ k)
7. PMAPooling              -> graph-level vector        [B, d]

Source side pipeline mirrors the binary side but uses a simple token
Transformer (instead of the instruction embedding) and adds AST-type
embeddings.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import softmax as pyg_softmax

from struclift.config import ModuleAConfig, StrucLiftConfig, VocabConfig


def sinusoidal_pe_from_normalized(norm: Tensor, d_model: int, max_len: int) -> Tensor:
    """Sinusoidal PE from continuous normalized coordinates in ``[0, 1]``.

    Maps ``norm`` to the same frequency layout as Vaswani et al., with
    ``norm=0`` → index 0 and ``norm→1`` → index ``max_len - 1`` scale.

    Parameters
    ----------
    norm : Tensor[..., ]  float32/64
        Typically ``i / |BB|`` (instruction index within block over block size).
    d_model : int
    max_len : int
        ``vcfg.max_instructions_per_block`` (or token sequence cap on source side).

    Returns
    -------
    Tensor[..., d_model]
    """
    scale = float(max(1, max_len - 1))
    pos = norm.unsqueeze(-1) * scale
    device = norm.device
    dtype = torch.float32
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=dtype, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(*norm.shape, d_model, device=device, dtype=dtype)
    pe[..., 0::2] = torch.sin(pos * div_term)
    pe[..., 1::2] = torch.cos(pos * div_term)
    return pe


# ════════════════════════════════════════════════════════════════════════════
# 1.  InstructionEmbedding
# ════════════════════════════════════════════════════════════════════════════

class InstructionEmbedding(nn.Module):
    """Composite embedding for a single machine instruction.

    Each instruction is described by:
      - opcode        : int   (index into opcode vocabulary)
      - operand_types : int[3] (padded with 0 = PAD)
      - value_buckets : int[3] (padded with 0 = PAD)

    The embedding is the *sum* of four components:
      opcode_emb(opcode) + mean(operand_type_emb) + mean(value_bucket_emb)
      + sinusoidal_position(i / |BB|)

    Doc §5.2.2: 归一化位置 ``i / |BB|`` 后接正弦编码（与 Vaswani 频率形式一致，
    经 ``max_instructions_per_block`` 缩放到连续索引尺度）。

    Parameters
    ----------
    vcfg : VocabConfig
        Vocabulary sizes.
    d_model : int
        Embedding / hidden dimension.

    Inputs
    ------
    opcodes              : Tensor[..., ]       int64
    operand_types        : Tensor[..., 3]      int64
    value_buckets        : Tensor[..., 3]      int64
    instr_norm_positions : Tensor[..., ]       float32  ``i / |BB|`` in ``[0, 1)``

    Returns
    -------
    Tensor[..., d_model]   float32
    """

    def __init__(self, vcfg: VocabConfig, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self._pe_max_len = vcfg.max_instructions_per_block

        self.opcode_emb = nn.Embedding(
            vcfg.num_opcodes, d_model, padding_idx=vcfg.pad_token_id,
        )
        self.operand_type_emb = nn.Embedding(
            vcfg.num_operand_types, d_model, padding_idx=vcfg.pad_token_id,
        )
        self.value_bucket_emb = nn.Embedding(
            vcfg.num_value_buckets, d_model, padding_idx=vcfg.pad_token_id,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
        """Standard sinusoidal positional encoding (Vaswani et al., 2017).

        Used by legacy helpers; binary instruction PE uses
        :func:`sinusoidal_pe_from_normalized` with ``i/|BB|``.

        Returns
        -------
        Tensor[max_len, d_model]  float32
        """
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)   # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )                                                                     # [d/2]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe                                                             # [L, d]

    # ------------------------------------------------------------------ #
    def forward(
        self,
        opcodes: Tensor,
        operand_types: Tensor,
        value_buckets: Tensor,
        instr_norm_positions: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        opcodes              : [*, ]       int64
        operand_types        : [*, 3]      int64
        value_buckets        : [*, 3]      int64
        instr_norm_positions : [*, ]       float32  ``i/|BB|``

        Returns
        -------
        [*, d_model]   float32
        """
        opc = self.opcode_emb(opcodes)                               # [*, d]

        # Mean-pool across the 3 operand slots (ignoring padding via mask).
        opt = self.operand_type_emb(operand_types)                   # [*, 3, d]
        opt_mask = (operand_types != 0).unsqueeze(-1).float()        # [*, 3, 1]
        opt_count = opt_mask.sum(dim=-2).clamp(min=1.0)              # [*, 1]
        opt_pooled = (opt * opt_mask).sum(dim=-2) / opt_count        # [*, d]

        vb = self.value_bucket_emb(value_buckets)                    # [*, 3, d]
        vb_mask = (value_buckets != 0).unsqueeze(-1).float()         # [*, 3, 1]
        vb_count = vb_mask.sum(dim=-2).clamp(min=1.0)               # [*, 1]
        vb_pooled = (vb * vb_mask).sum(dim=-2) / vb_count           # [*, d]

        pos_enc = sinusoidal_pe_from_normalized(
            instr_norm_positions, self.d_model, self._pe_max_len,
        )                                                             # [*, d]

        return opc + opt_pooled + vb_pooled + pos_enc                # [*, d]


# ════════════════════════════════════════════════════════════════════════════
# 2.  BlockTransformer
# ════════════════════════════════════════════════════════════════════════════

class BlockTransformer(nn.Module):
    """2-layer Pre-LN Transformer with causal mask.

    Processes the sequence of instruction embeddings within a single basic
    block.  The causal mask reflects the sequential nature of instructions.

    Parameters
    ----------
    d_model  : int   (256)
    n_heads  : int   (4)
    ffn_dim  : int   (1024)
    n_layers : int   (2)
    dropout  : float (0.1)

    Inputs
    ------
    x    : Tensor[B_blocks, seq_len, d_model]  float32  -- instruction embeds
    mask : Tensor[B_blocks, seq_len]            bool     -- True = valid token

    Returns
    -------
    Tensor[B_blocks, seq_len, d_model]  float32 -- contextualised embeddings
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Build Pre-LN Transformer encoder layers manually so we have
        # control over the normalisation order.
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                _PreLNTransformerBlock(d_model, n_heads, ffn_dim, dropout)
            )
        self.final_ln = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x    : [B, L, d]  float32
        mask : [B, L]     bool  (True = valid position)

        Returns
        -------
        [B, L, d]  float32
        """
        B, L, _ = x.shape

        # Causal mask: shape [L, L], True = **blocked** (PyTorch convention).
        causal = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )                                                                    # [L, L]

        # Key-padding mask: True = padding position to ignore.
        key_pad = ~mask                                                      # [B, L]

        for layer in self.layers:
            x = layer(x, attn_mask=causal, key_padding_mask=key_pad)

        x = self.final_ln(x)
        return x                                                             # [B, L, d]


class _PreLNTransformerBlock(nn.Module):
    """Single Pre-LayerNorm Transformer block.

    Pre-LN layout: x -> LN -> MHA -> residual -> LN -> FFN -> residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        # Self-attention with Pre-LN.
        h = self.ln1(x)
        h, _ = self.mha(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(h)

        # Feed-forward with Pre-LN.
        x = x + self.ffn(self.ln2(x))
        return x


# ════════════════════════════════════════════════════════════════════════════
# 3.  AttentivePooling
# ════════════════════════════════════════════════════════════════════════════

class AttentivePooling(nn.Module):
    """Learnable single-query attentive pooling.

    Given a set of vectors ``{z_1, ..., z_L}`` and a learnable query
    ``w in R^d``, computes:

        alpha_j = softmax(w^T z_j)          (over valid positions)
        h       = sum_j alpha_j * z_j

    Parameters
    ----------
    d_model : int  (256)

    Inputs
    ------
    z    : Tensor[B, L, d]  float32
    mask : Tensor[B, L]     bool  (True = valid)

    Returns
    -------
    Tensor[B, d]  float32
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))              # [d]

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z    : [B, L, d]  float32
        mask : [B, L]     bool

        Returns
        -------
        [B, d]  float32
        """
        # Compute raw scores.
        scores = torch.einsum("bld,d->bl", z, self.query)           # [B, L]

        # Mask invalid positions with -inf before softmax.
        scores = scores.masked_fill(~mask, float("-inf"))            # [B, L]
        alpha = F.softmax(scores, dim=-1)                            # [B, L]

        # Handle edge case where entire sequence is masked (all -inf -> nan).
        alpha = alpha.nan_to_num(0.0)

        h = torch.einsum("bl,bld->bd", alpha, z)                    # [B, d]
        return h


# ════════════════════════════════════════════════════════════════════════════
# 4.  StructuralFeatureMLP
# ════════════════════════════════════════════════════════════════════════════

class StructuralFeatureMLP(nn.Module):
    """Encodes the 12-dim precomputed structural feature vector of a basic
    block into the model's hidden dimension.

    Doc §5.2.5（与实现一致）: ``Linear(12→d) → ReLU → Linear(d→d) → LayerNorm``。

    Parameters
    ----------
    feat_dim : int  (12)
    d_model  : int  (256)

    Inputs
    ------
    feats : Tensor[N, feat_dim]  float32

    Returns
    -------
    Tensor[N, d_model]  float32
    """

    def __init__(self, feat_dim: int = 12, d_model: int = 256) -> None:
        super().__init__()
        # Doc §5.2.5
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, feats: Tensor) -> Tensor:
        """
        Parameters
        ----------
        feats : [N, 12]  float32

        Returns
        -------
        [N, d_model]  float32
        """
        return self.net(feats)


# ════════════════════════════════════════════════════════════════════════════
# 5.  EdgeTypedGATLayer
# ════════════════════════════════════════════════════════════════════════════

class EdgeTypedGATLayer(nn.Module):
    """Single Graph Attention layer with edge-type embeddings.

    Attention mechanism（StrucLift_Research_Proposal.md §5.3.1）：
    **有向 CFG**：``edge_index`` 为 ``src → dst``；消息沿边聚合到 ``dst``，
    即节点 *i* 从其 **前驱**（指向 *i* 的邻居）接收加权消息。

        e_ij = LeakyReLU( a^T [ W h_i || W h_j || E_edge(t_ij) ] )
        alpha_ij = softmax_j(e_ij)
        逐头：``h_i^k = σ( Σ_j alpha_ij^k W^k h_j )``，其中 ``σ`` 为 ELU；
        再 ``||`` 拼接 K 个头后经 ``Linear`` 投回 ``d_model``，残差 + LayerNorm。

    Parameters
    ----------
    d_model       : int  (256)
    n_heads       : int  (4)
    edge_emb_dim  : int  (32)
    num_edge_types: int  (5)
    dropout       : float (0.1)

    Inputs
    ------
    x          : Tensor[N, d_model]        float32  -- node features
    edge_index : Tensor[2, E]              int64    -- COO format
    edge_type  : Tensor[E]                 int64    -- edge type ids

    Returns
    -------
    Tensor[N, d_model]  float32  -- updated node features
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        edge_emb_dim: int = 32,
        num_edge_types: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections (shared across heads conceptually but we
        # reshape for multi-head).
        self.W = nn.Linear(d_model, d_model, bias=False)             # -> [N, K*d_k]

        # Edge type embedding.
        self.edge_emb = nn.Embedding(num_edge_types, edge_emb_dim)

        # Attention vector per head: consumes [d_k + d_k + edge_emb_dim].
        attn_in = 2 * self.d_head + edge_emb_dim
        self.attn = nn.Parameter(torch.randn(n_heads, attn_in))      # [K, attn_in]

        # Output projection (after head concatenation).
        self.out_proj = nn.Linear(d_model, d_model)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x          : [N, d]  float32
        edge_index : [2, E]  int64
        edge_type  : [E]     int64

        Returns
        -------
        [N, d]  float32
        """
        N = x.size(0)
        src, dst = edge_index                                        # [E], [E]

        # Project and reshape to multi-head.
        Wx = self.W(x).view(N, self.n_heads, self.d_head)            # [N, K, d_k]

        Wx_i = Wx[src]                                               # [E, K, d_k]
        Wx_j = Wx[dst]                                               # [E, K, d_k]

        # Edge embeddings broadcast to all heads.
        e_emb = self.edge_emb(edge_type)                             # [E, edge_dim]
        e_emb = e_emb.unsqueeze(1).expand(-1, self.n_heads, -1)     # [E, K, edge_dim]

        # Concatenate for attention computation.
        cat = torch.cat([Wx_i, Wx_j, e_emb], dim=-1)                # [E, K, attn_in]

        # Raw attention logits.
        logits = (cat * self.attn.unsqueeze(0)).sum(dim=-1)          # [E, K]
        logits = self.leaky_relu(logits)                             # [E, K]

        # Softmax per destination node, per head.
        # Use PyG's softmax which scatters over dst.
        alpha = torch.zeros_like(logits)
        for k in range(self.n_heads):
            alpha[:, k] = pyg_softmax(logits[:, k], dst, num_nodes=N)
        alpha = self.dropout(alpha)                                  # [E, K]

        # Aggregate: weighted sum of source features per dst node.
        msg = alpha.unsqueeze(-1) * Wx_j                             # [E, K, d_k]
        out = torch.zeros(N, self.n_heads, self.d_head,
                          device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)  # [N, K, d_k]

        # 提案公式中的 σ：逐头聚合后对特征用 ELU，再拼接为 [N, K*d_k]（与 GAT 常见写法一致）
        out = F.elu(out)                                             # [N, K, d_k]
        out = out.reshape(N, self.d_model)                           # [N, d]
        out = self.out_proj(out)                                     # [N, d]

        # Residual + LayerNorm.
        out = self.ln(x + self.dropout(out))                         # [N, d]
        return out


# ════════════════════════════════════════════════════════════════════════════
# 6.  SubgraphPatternClassifier
# ════════════════════════════════════════════════════════════════════════════

class SubgraphPatternClassifier(nn.Module):
    """Classifies each node into one of C structural pattern categories
    by inspecting its k-hop neighbourhood（§5.3.2 结构模式分类头）。

    For each centre node *i*:
      1. k-hop 子图：有向全源最短路 ``D`` 上取 ``D[i,j] <= k``（从 *i* 沿箭头可达的 *j*）。
      2. ``attention_pool(H̃[sub], query=H̃[node_i])``：与 §5.2.4 ``AttentivePooling``
         同构——可学习 ``w ∈ R^{2d}``，对 ``[z_j \\| H̃[node_i]]`` 做 ``w^T`` 再 softmax 池化。
      3. 拓扑特征（13 维）：边类型分布、回边、边界最短路、规模、中心到子图内各点的最短路
         统计（均值/最大/最小正值/标准差）、子图直径（全对最短路最大），均归一化。
      4. Concatenate ``[center, subgraph_repr, MLP_topo(topo)]`` → ClassifierMLP.

    实现按 **PyG batch 中的每个独立图** 构图，在单图内用 Floyd–Warshall + 张量 gather
    一次处理该图所有中心节点（不再对全局 ``N`` 做 Python 逐节点 ``k_hop_subgraph`` 循环）。
    """

    def __init__(
        self,
        d_model: int = 256,
        num_patterns: int = 10,
        topo_feat_dim: int = 13,
        k: int = 3,
        num_edge_types: int = 5,
        max_blocks: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.num_edge_types = num_edge_types
        self.max_blocks = max_blocks
        self.num_patterns = num_patterns

        # §5.3.2：与 AttentivePooling 同构的线性打分，输入为 [z_j || center]（2d）
        self.subgraph_pool_w = nn.Parameter(torch.randn(2 * d_model))

        self.topo_mlp = nn.Sequential(
            nn.Linear(topo_feat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_patterns),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _shortest_path_to_boundary(
        center_idx: int,
        sub_set: set,
        sub_edge_index: Tensor,
        original_edge_index: Tensor,
    ) -> float:
        """中心沿子图内边到任意「边界」节点的最短有向跳数，归一化到 ``[0,1]``。"""
        sub_size = len(sub_set)
        if sub_size <= 1:
            return 0.0
        boundary: set = set()
        if original_edge_index.numel() > 0:
            for e in range(original_edge_index.size(1)):
                u = int(original_edge_index[0, e].item())
                v = int(original_edge_index[1, e].item())
                if u in sub_set and v not in sub_set:
                    boundary.add(u)
        if not boundary:
            return 0.0
        adj: dict[int, list[int]] = {n: [] for n in sub_set}
        if sub_edge_index.numel() > 0:
            for e in range(sub_edge_index.size(1)):
                u = int(sub_edge_index[0, e].item())
                v = int(sub_edge_index[1, e].item())
                if u in sub_set and v in sub_set:
                    adj[u].append(v)
        dist: dict[int, int] = {}
        q: deque[int] = deque([center_idx])
        dist[center_idx] = 0
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        reachable = [dist[b] for b in boundary if b in dist]
        if not reachable:
            return 1.0
        min_d = min(reachable)
        denom = float(max(sub_size - 1, 1))
        return min(min_d / denom, 1.0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _floyd_warshall(
        n: int,
        edge_index: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """全对最短路 ``[n,n]``（边权 1，**有向**：仅 ``s→t`` 方向）。"""
        inf = 1e6
        dmat = torch.full((n, n), inf, device=device, dtype=dtype)
        dmat.fill_diagonal_(0.0)
        if edge_index.numel() > 0:
            s = edge_index[0].long()
            t = edge_index[1].long()
            one = torch.ones(1, device=device, dtype=dtype)
            dmat[s, t] = torch.minimum(dmat[s, t], one.expand_as(s).to(dtype))
        for kk in range(n):
            dmat = torch.minimum(dmat, dmat[:, kk : kk + 1] + dmat[kk : kk + 1, :])
        return dmat

    def _topo_vector_for_graph_legacy(
        self,
        dmat: Tensor,
        k_hop_m: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """**Legacy**（带 Python 逐节点循环）实现，仅保留供等价性验证。

        正式 forward 使用下面的 :meth:`_topo_vector_for_graph`（向量化版）。
        """
        fin = dmat < (1e5 / 2)
        m = k_hop_m & fin
        raw_size = m.sum(dim=1).float()
        eye = torch.eye(n, device=device, dtype=torch.bool)
        mask_other = m & (~eye)

        sum_dist = (dmat * mask_other.float()).sum(dim=1)
        denom_others = (raw_size - 1.0).clamp(min=1.0)
        mean_sp = sum_dist / denom_others
        mean_sp = torch.where(raw_size > 1, mean_sp, torch.zeros_like(mean_sp))

        d_row_max = dmat.clone()
        d_row_max[~m] = 0.0
        max_sp = d_row_max.max(dim=1).values

        d_inf = dmat.masked_fill(~m | eye, float("inf"))
        min_sp = d_inf.min(dim=1).values
        min_sp = torch.where(torch.isfinite(min_sp), min_sp, torch.zeros_like(min_sp))

        mean_exp = mean_sp.unsqueeze(1)
        var = ((dmat - mean_exp) ** 2 * mask_other.float()).sum(dim=1) / denom_others
        std_sp = torch.sqrt(var.clamp(min=0.0) + 1e-8)
        std_sp = torch.where(raw_size > 1, std_sp, torch.zeros_like(std_sp))

        norm_d = float(max(self.k, 1))
        mean_n = mean_sp / norm_d
        max_n = max_sp / norm_d
        min_n = min_sp / norm_d
        std_n = std_sp / norm_d

        size_norm = raw_size / float(max(self.max_blocks, 1))

        E_loc = edge_index.size(1)
        counts = torch.zeros(n, self.num_edge_types, device=device, dtype=dtype)
        if E_loc > 0:
            src_e = edge_index[0].long()
            dst_e = edge_index[1].long()
            et = edge_type.long()
            I = torch.arange(n, device=device).unsqueeze(1).expand(n, E_loc)
            J = src_e.unsqueeze(0).expand(n, E_loc)
            m_src_ok = m[I, J]
            inc = m_src_ok & m[I, dst_e.unsqueeze(0).expand(n, E_loc)]
            for t in range(self.num_edge_types):
                counts[:, t] = (inc & (et.unsqueeze(0) == t)).float().sum(dim=1)
            counts = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-6)

        is_back = edge_type == 2
        has_back = torch.zeros(n, device=device, dtype=dtype)
        if E_loc > 0 and is_back.any():
            dst_e = edge_index[1].long()
            src_e = edge_index[0].long()
            I = torch.arange(n, device=device).unsqueeze(1).expand(n, E_loc)
            J = src_e.unsqueeze(0).expand(n, E_loc)
            m_src_ok = m[I, J]
            hits = (
                (dst_e.unsqueeze(0) == torch.arange(n, device=device).unsqueeze(1))
                & is_back.unsqueeze(0)
                & m_src_ok
            )
            has_back = hits.any(dim=1).to(dtype)

        diameter_n = torch.zeros(n, device=device, dtype=dtype)
        denom_diam = float(max(2 * self.k, 1))
        for i in range(n):
            idx = m[i].nonzero(as_tuple=True)[0]
            if idx.numel() <= 1:
                continue
            sub = dmat[idx][:, idx]
            finite = sub[torch.isfinite(sub) & (sub < 1e5)]
            if finite.numel() > 0:
                diameter_n[i] = (finite.max() / denom_diam).clamp(max=1.0)

        boundary_list: list[float] = []
        for i in range(n):
            sub_nodes = m[i].nonzero(as_tuple=True)[0]
            sub_set = set(sub_nodes.tolist())
            if sub_nodes.numel() == 0:
                boundary_list.append(0.0)
                continue
            mask_e = torch.zeros(E_loc, dtype=torch.bool, device=device)
            for e in range(E_loc):
                u = int(edge_index[0, e].item())
                v = int(edge_index[1, e].item())
                mask_e[e] = u in sub_set and v in sub_set
            sub_ei = edge_index[:, mask_e]
            sub_et = edge_type[mask_e]
            boundary_list.append(
                self._shortest_path_to_boundary(i, sub_set, sub_ei, edge_index),
            )
        min_b = torch.tensor(boundary_list, device=device, dtype=dtype)

        topo = torch.cat(
            [
                counts,
                has_back.unsqueeze(1),
                min_b.unsqueeze(1),
                size_norm.unsqueeze(1),
                mean_n.unsqueeze(1),
                max_n.unsqueeze(1),
                min_n.unsqueeze(1),
                std_n.unsqueeze(1),
                diameter_n.unsqueeze(1),
            ],
            dim=1,
        )
        return topo

    # ------------------------------------------------------------------ #
    def _topo_vector_for_graph(
        self,
        dmat: Tensor,
        k_hop_m: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """拓扑特征 ``[n, topo_feat_dim]``（§5.3.2：最短路统计 + 边分布等）。

        向量化版：与 :meth:`_topo_vector_for_graph_legacy` **数值等价**（浮点尾部差异 < 1e-6），
        但消除了所有 ``for i in range(n)`` / ``for e in range(E_loc)`` / ``.item()`` 同步点。

        数学等价性关键证明（``min_b`` 这一维）::

            设 sub_i = {v | dmat[i,v] ≤ k}。对任意 u ∈ sub_i，原图 i→u 的最短路径
            所有中间点 w 都满足 dmat[i,w] < dmat[i,u] ≤ k，故 w ∈ sub_i。因此
            "子图诱导图最短路 center→u" 等于全图 dmat[i,u]，**无需 Python BFS**。

        其余维度仅替换 ``for i`` 为 3D broadcast（``diameter``）或 ``scatter`` 类算子
        （``counts``），数学完全等价。
        """
        # -- 保持与 legacy 相同的 m 定义 --
        fin = dmat < (1e5 / 2)
        m = k_hop_m & fin  # [n, n] bool
        raw_size = m.sum(dim=1).float()  # [n]
        eye = torch.eye(n, device=device, dtype=torch.bool)
        mask_other = m & (~eye)  # [n, n] bool

        # -- 最短路统计（4 维）：本就是纯张量，与 legacy 逐行一致 --
        sum_dist = (dmat * mask_other.float()).sum(dim=1)
        denom_others = (raw_size - 1.0).clamp(min=1.0)
        mean_sp = sum_dist / denom_others
        mean_sp = torch.where(raw_size > 1, mean_sp, torch.zeros_like(mean_sp))

        d_row_max = dmat.clone()
        d_row_max[~m] = 0.0
        max_sp = d_row_max.max(dim=1).values

        d_inf = dmat.masked_fill(~m | eye, float("inf"))
        min_sp = d_inf.min(dim=1).values
        min_sp = torch.where(
            torch.isfinite(min_sp), min_sp, torch.zeros_like(min_sp),
        )

        mean_exp = mean_sp.unsqueeze(1)
        var = ((dmat - mean_exp) ** 2 * mask_other.float()).sum(dim=1) / denom_others
        std_sp = torch.sqrt(var.clamp(min=0.0) + 1e-8)
        std_sp = torch.where(raw_size > 1, std_sp, torch.zeros_like(std_sp))

        norm_d = float(max(self.k, 1))
        mean_n = mean_sp / norm_d
        max_n = max_sp / norm_d
        min_n = min_sp / norm_d
        std_n = std_sp / norm_d

        size_norm = raw_size / float(max(self.max_blocks, 1))

        # -- counts（5 维边类型分布）：one-hot + matmul 替代 ``for t in range(T)`` --
        E_loc = edge_index.size(1)
        counts = torch.zeros(n, self.num_edge_types, device=device, dtype=dtype)
        if E_loc > 0:
            src_e = edge_index[0].long()
            dst_e = edge_index[1].long()
            et = edge_type.long()
            I = torch.arange(n, device=device).unsqueeze(1).expand(n, E_loc)
            J = src_e.unsqueeze(0).expand(n, E_loc)
            m_src_ok = m[I, J]
            inc = m_src_ok & m[I, dst_e.unsqueeze(0).expand(n, E_loc)]  # [n, E_loc]
            # legacy 用 ``.float().sum`` 做 fp32 累加再 downcast，这里严格复现
            et_oh = F.one_hot(et, num_classes=self.num_edge_types).float()  # [E_loc, T]
            counts_f = inc.float() @ et_oh                                   # [n, T] fp32
            counts = counts_f.to(dtype)
            counts = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # -- has_back（1 维回边）：已是向量化，保留 --
        is_back = edge_type == 2
        has_back = torch.zeros(n, device=device, dtype=dtype)
        if E_loc > 0 and is_back.any():
            dst_e_b = edge_index[1].long()
            src_e_b = edge_index[0].long()
            I = torch.arange(n, device=device).unsqueeze(1).expand(n, E_loc)
            J = src_e_b.unsqueeze(0).expand(n, E_loc)
            m_src_ok = m[I, J]
            hits = (
                (dst_e_b.unsqueeze(0) == torch.arange(n, device=device).unsqueeze(1))
                & is_back.unsqueeze(0)
                & m_src_ok
            )
            has_back = hits.any(dim=1).to(dtype)

        # -- diameter（1 维）：消除 ``for i``，用 3D broadcast + masked max --
        # pair_ok[i, u, v] = (u ∈ sub_i) ∧ (v ∈ sub_i) ∧ finite(dmat[u,v])
        # diameter[i] = max over (u, v) of dmat[u, v]
        denom_diam = float(max(2 * self.k, 1))
        finite_pair = torch.isfinite(dmat) & (dmat < 1e5)          # [n, n]
        pair_in_sub = m.unsqueeze(2) & m.unsqueeze(1)              # [n, n, n]
        pair_ok = pair_in_sub & finite_pair.unsqueeze(0)           # [n, n, n]
        dmat_b = dmat.unsqueeze(0).expand(n, -1, -1)               # [n, n, n]
        neg_inf = torch.full_like(dmat_b, float("-inf"))
        d_masked = torch.where(pair_ok, dmat_b, neg_inf)
        any_ok = pair_ok.reshape(n, -1).any(dim=1)                 # [n]
        diam_raw = d_masked.reshape(n, -1).max(dim=1).values       # [n]
        diam_raw = torch.where(any_ok, diam_raw, torch.zeros_like(diam_raw))
        diameter_n = (diam_raw / denom_diam).clamp(max=1.0).to(dtype)

        # -- min_b（1 维边界最短路）：消除 Python BFS，用 dmat + 布尔掩码 --
        # boundary(i) = {u | u ∈ sub_i ∧ ∃v: (u,v) ∈ E ∧ v ∉ sub_i}
        if E_loc > 0:
            A = torch.zeros(n, n, dtype=torch.float32, device=device)
            src_e2 = edge_index[0].long()
            dst_e2 = edge_index[1].long()
            A[src_e2, dst_e2] = 1.0
            not_k = (~k_hop_m).float()                              # [n, n]
            # has_out_outside[i, u] = Σ_v A[u, v] · ~k_hop_m[i, v]  > 0
            has_out_outside = (not_k @ A.t()) > 0                   # [n, n]
            boundary_mask = k_hop_m & has_out_outside               # [n, n]
            d_b = dmat.masked_fill(~boundary_mask, float("inf"))
            min_d = d_b.min(dim=1).values                           # [n]
            has_any_boundary = boundary_mask.any(dim=1)             # [n]
            denom_bd = (raw_size - 1.0).clamp(min=1.0)
            min_b_raw = (min_d / denom_bd).clamp(max=1.0)
            small = raw_size <= 1
            no_bd = ~has_any_boundary
            min_b = torch.where(
                small | no_bd, torch.zeros_like(min_b_raw), min_b_raw,
            ).to(dtype)
        else:
            min_b = torch.zeros(n, device=device, dtype=dtype)

        # -- 拼接顺序与 legacy 完全一致（classifier 的输入对齐依赖于此）--
        topo = torch.cat(
            [
                counts,
                has_back.unsqueeze(1),
                min_b.unsqueeze(1),
                size_norm.unsqueeze(1),
                mean_n.unsqueeze(1),
                max_n.unsqueeze(1),
                min_n.unsqueeze(1),
                std_n.unsqueeze(1),
                diameter_n.unsqueeze(1),
            ],
            dim=1,
        )
        return topo

    def _forward_single_graph_legacy(
        self,
        node_feats: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """**Legacy**（带 ``for i in range(n)`` + ``.item()``）实现，仅供等价性验证。"""
        n, d = node_feats.shape
        device = node_feats.device
        dtype = node_feats.dtype
        if n == 0:
            return torch.zeros(0, self.num_patterns, device=device, dtype=dtype)

        dmat = self._floyd_warshall(n, edge_index, device, dtype)
        k_hop_m = (dmat <= float(self.k)) & (dmat < 1e5)

        max_s = int(k_hop_m.sum(dim=1).max().item())
        max_s = max(max_s, 1)
        idx_pad = torch.zeros(n, max_s, dtype=torch.long, device=device)
        mask_pad = torch.zeros(n, max_s, dtype=torch.bool, device=device)
        for i in range(n):
            js = k_hop_m[i].nonzero(as_tuple=True)[0]
            ln = js.numel()
            if ln > 0:
                idx_pad[i, :ln] = js
                mask_pad[i, :ln] = True

        idx_safe = idx_pad.clamp(min=0)
        gathered = node_feats[idx_safe]
        gathered = gathered * mask_pad.unsqueeze(-1).to(dtype)

        center_row = node_feats.unsqueeze(1).expand(-1, max_s, -1)
        cat = torch.cat([gathered, center_row], dim=-1)
        scores = (cat * self.subgraph_pool_w.view(1, 1, -1)).sum(dim=-1)
        scores = scores.masked_fill(~mask_pad, float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        alpha = alpha.nan_to_num(0.0)
        subgraph_repr = (alpha.unsqueeze(-1) * gathered).sum(dim=1)

        topo = self._topo_vector_for_graph_legacy(
            dmat, k_hop_m, edge_index, edge_type, n, device, dtype,
        )
        topo_repr = self.topo_mlp(topo)

        combined = torch.cat([node_feats, subgraph_repr, topo_repr], dim=-1)
        return self.classifier(combined)

    # ------------------------------------------------------------------ #
    def _forward_single_graph(
        self,
        node_feats: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """单张 CFG（局部编号 0..n-1）上一次性算所有节点的 pattern logits ``[n,C]``。

        向量化版：与 :meth:`_forward_single_graph_legacy` **数值等价**（浮点尾部差异 < 1e-6）。

        关键等价性（subgraph attention pooling）::

            legacy 把 ``k_hop_m`` 紧致成 ``[n, max_s]`` 再做 attention；新版直接在
            ``[n, n]`` 上做 attention，非子图位置 fill -inf。``softmax`` 后这些位置
            权重为 0，对有效位置的归一化**完全相同**——attention 输出按位等价。

        拓扑特征通过 :meth:`_topo_vector_for_graph`（向量化版）得到，亦与 legacy 等价。
        """
        n, d = node_feats.shape
        device = node_feats.device
        dtype = node_feats.dtype
        if n == 0:
            return torch.zeros(0, self.num_patterns, device=device, dtype=dtype)

        dmat = self._floyd_warshall(n, edge_index, device, dtype)
        k_hop_m = (dmat <= float(self.k)) & (dmat < 1e5)  # [n, n] bool

        # ---- Subgraph attention pooling：不再紧致化，直接 [n, n] mask ----
        # legacy: scores[i, l] = (gathered || center) · w = gathered·w1 + center·w2
        # new:    scores[i, j] = node_feats[j]·w1 + node_feats[i]·w2
        # 按位等价（仅把 dot product 拆成 per-node 分量 + broadcast，省内存）
        w1 = self.subgraph_pool_w[:d]
        w2 = self.subgraph_pool_w[d:]
        per_j = node_feats @ w1                            # [n]
        per_i = node_feats @ w2                            # [n]
        scores_full = per_j.unsqueeze(0) + per_i.unsqueeze(1)   # [n, n]
        scores_full = scores_full.masked_fill(~k_hop_m, float("-inf"))
        alpha_full = F.softmax(scores_full, dim=-1)
        alpha_full = alpha_full.nan_to_num(0.0)
        # subgraph_repr[i] = Σ_j alpha[i, j] · node_feats[j]   （非子图位置 α=0 不贡献）
        subgraph_repr = alpha_full @ node_feats            # [n, d]

        topo = self._topo_vector_for_graph(
            dmat, k_hop_m, edge_index, edge_type, n, device, dtype,
        )
        topo_repr = self.topo_mlp(topo)

        combined = torch.cat([node_feats, subgraph_repr, topo_repr], dim=-1)
        return self.classifier(combined)

    def forward(
        self,
        node_feats: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor,
    ) -> Tensor:
        N = node_feats.size(0)
        device = node_feats.device
        dtype = node_feats.dtype
        if N == 0:
            return torch.zeros(0, self.num_patterns, device=device, dtype=dtype)

        logits_full = torch.empty(N, self.num_patterns, device=device, dtype=dtype)
        num_graphs = int(batch.max().item()) + 1

        for g in range(num_graphs):
            nodes = (batch == g).nonzero(as_tuple=True)[0]
            if nodes.numel() == 0:
                continue
            nodes = nodes.sort().values
            n = nodes.numel()
            feats_loc = node_feats[nodes]

            old_to_new = torch.full((N,), -1, dtype=torch.long, device=device)
            old_to_new[nodes] = torch.arange(n, device=device)

            src, dst = edge_index[0], edge_index[1]
            m_e = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)
            ei_loc = torch.stack([old_to_new[src[m_e]], old_to_new[dst[m_e]]], dim=0)
            et_loc = edge_type[m_e]

            logits_loc = self._forward_single_graph(feats_loc, ei_loc, et_loc)
            # autocast 下 logits_loc 可能为 float16，logits_full 与 node_feats 同 dtype（常为 float32）
            logits_full[nodes] = logits_loc.to(dtype=logits_full.dtype)

        return logits_full


# ════════════════════════════════════════════════════════════════════════════
# 7.  PMAPooling  (Pooling by Multihead Attention)
# ═══════════════════════════════════════════════════════════════════���════════

class PMAPooling(nn.Module):
    """§5.3.3 图级表示（Pooling by Multihead Attention, Lee et al., 2019）。

    可学习 seed :math:`S \\in \\mathbb{R}^{1 \\times d}`：
    :math:`\\bar{h}_b = \\mathrm{MultiHeadAttn}(S, \\tilde{H}_b, \\tilde{H}_b)`。

    Parameters
    ----------
    d_model : int  (256)
    n_heads : int  (4)

    Inputs
    ------
    node_feats : Tensor[N, d]     float32  (flat across batch)
    batch      : Tensor[N]        int64    (graph membership vector)

    Returns
    -------
    Tensor[B, d]  float32  -- one vector per graph
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))        # [1, 1, d]
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------ #
    def forward(self, node_feats: Tensor, batch: Tensor) -> Tensor:
        """
        Parameters
        ----------
        node_feats : [N, d]  float32
        batch      : [N]     int64

        Returns
        -------
        [B, d]  float32
        """
        device = node_feats.device
        num_graphs = int(batch.max().item()) + 1

        # Group nodes by graph and pad into a dense batch.
        # Determine max number of nodes in any graph for padding.
        counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.long))
        max_nodes = int(counts.max().item())

        # Build dense tensor  [B, max_nodes, d]  and mask  [B, max_nodes].
        H = torch.zeros(num_graphs, max_nodes, self.d_model,
                         device=device, dtype=node_feats.dtype)
        mask = torch.zeros(num_graphs, max_nodes, device=device, dtype=torch.bool)

        # Scatter node features into the padded batch.
        # Compute per-node offset within its graph.
        offsets = torch.zeros(node_feats.size(0), dtype=torch.long, device=device)
        # For each graph, the nodes should be placed at positions 0..count-1.
        running = torch.zeros(num_graphs, dtype=torch.long, device=device)
        for idx in range(node_feats.size(0)):
            g = batch[idx].item()
            pos = running[g].item()
            offsets[idx] = pos
            running[g] += 1

        H[batch, offsets] = node_feats
        mask[batch, offsets] = True

        # Expand seed to batch size.
        S = self.seed.expand(num_graphs, -1, -1)                     # [B, 1, d]

        # Key padding mask: True = padding position to *ignore*.
        key_pad = ~mask                                              # [B, max_nodes]

        out, _ = self.mha(S, H, H, key_padding_mask=key_pad,
                          need_weights=False)                        # [B, 1, d]
        out = self.ln(out.squeeze(1))                                # [B, d]
        return out


# ════════════════════════════════════════════════════════════════════════════
# 8.  BinaryCFGEncoder  (Module A main class)
# ════════════════════════════════════════════════════════════════════════════

class BinaryCFGEncoder(nn.Module):
    """Structure-Aware Binary CFG Encoder -- the complete Module A.

    Accepts a batch of binary CFGs represented as PyG-style batched tensors
    and produces node-level embeddings, graph-level embeddings, and
    structural pattern logits.

    Expected input dict keys (PyG batch conventions)
    -------------------------------------------------
    opcodes           : Tensor[total_instrs]       int64
    operand_types     : Tensor[total_instrs, 3]    int64
    value_buckets     : Tensor[total_instrs, 3]    int64
    instr_norm_positions : Tensor[total_instrs]   float32  (i/|BB| per block)
    instr_to_block    : Tensor[total_instrs]       int64  (flat block id)
    instr_mask        : Tensor[total_blocks, max_instr]  bool
    block_instr_idx   : Tensor[total_blocks, max_instr]  int64  (indices
                        into the flat instruction arrays, padded with 0)
    struct_feats      : Tensor[total_nodes, 12]    float32
    edge_index        : Tensor[2, E]               int64  (COO)
    edge_type         : Tensor[E]                  int64
    batch             : Tensor[total_nodes]         int64  (graph membership)
    num_instrs_per_block : Tensor[total_blocks]    int64  (# instructions
                           per block, used to build mask if not provided)

    Output dict
    -----------
    node_embeddings  : Tensor[total_nodes, d]      float32
    graph_embeddings : Tensor[batch_size, d]        float32
    pattern_logits   : Tensor[total_nodes, C]       float32

    Parameters
    ----------
    vcfg   : VocabConfig
    ma_cfg : ModuleAConfig
    """

    def __init__(self, vcfg: VocabConfig, ma_cfg: ModuleAConfig) -> None:
        super().__init__()
        d = ma_cfg.d_model
        self.d_model = d
        self.max_instr = vcfg.max_instructions_per_block

        # --- Instruction embedding ---
        self.instr_emb = InstructionEmbedding(vcfg, d)

        # --- Block Transformer ---
        self.block_transformer = BlockTransformer(
            d_model=d,
            n_heads=ma_cfg.instr_transformer_heads,
            ffn_dim=ma_cfg.instr_ffn_dim,
            n_layers=ma_cfg.instr_transformer_layers,
            dropout=ma_cfg.instr_dropout,
        )

        # --- Attentive pooling (instructions -> block) ---
        self.block_pool = AttentivePooling(d)

        # --- Structural feature MLP ---
        self.struct_mlp = StructuralFeatureMLP(
            feat_dim=ma_cfg.struct_feat_dim,
            d_model=d,
        )

        # --- Edge-typed GAT layers ---
        self.gat_layers = nn.ModuleList([
            EdgeTypedGATLayer(
                d_model=d,
                n_heads=ma_cfg.gat_heads,
                edge_emb_dim=ma_cfg.edge_emb_dim,
                num_edge_types=vcfg.num_edge_types,
                dropout=ma_cfg.gat_dropout,
            )
            for _ in range(ma_cfg.gat_layers)
        ])

        # --- Subgraph pattern classifier ---
        self.pattern_clf = SubgraphPatternClassifier(
            d_model=d,
            num_patterns=vcfg.num_patterns,
            topo_feat_dim=ma_cfg.pattern_topo_feat_dim,
            k=ma_cfg.pattern_k_hop,
            num_edge_types=vcfg.num_edge_types,
            max_blocks=vcfg.max_blocks_per_function,
        )

        # --- PMA graph pooling ---
        self.graph_pool = PMAPooling(d_model=d, n_heads=ma_cfg.pma_heads)

    # ------------------------------------------------------------------ #
    def _embed_blocks(
        self,
        opcodes: Tensor,
        operand_types: Tensor,
        value_buckets: Tensor,
        instr_norm_positions: Tensor,
        block_instr_idx: Tensor,
        instr_mask: Tensor,
    ) -> Tensor:
        """Embed all instructions, run per-block Transformer, attentive-pool.

        Parameters
        ----------
        opcodes          : [I]        int64   (I = total instructions in batch)
        operand_types    : [I, 3]     int64
        value_buckets    : [I, 3]     int64
        instr_norm_positions : [I]    float32  i/|BB|
        block_instr_idx  : [B_blk, L] int64   (B_blk = total blocks, L = max instr)
        instr_mask       : [B_blk, L] bool

        Returns
        -------
        [B_blk, d]  float32 -- one vector per basic block
        """
        # 1. Embed all instructions.
        instr_vecs = self.instr_emb(
            opcodes, operand_types, value_buckets, instr_norm_positions,
        )                                                            # [I, d]

        # 2. Gather into block-major layout.
        #    block_instr_idx[b, l] is the index into instr_vecs for block b,
        #    position l.  Padding positions have arbitrary index but are
        #    masked out.
        B_blk, L = block_instr_idx.shape
        flat_idx = block_instr_idx.reshape(-1)                       # [B_blk * L]
        gathered = instr_vecs[flat_idx].reshape(B_blk, L, -1)       # [B_blk, L, d]

        # 3. Run Block Transformer.
        ctx = self.block_transformer(gathered, instr_mask)           # [B_blk, L, d]

        # 4. Attentive pooling.
        block_vecs = self.block_pool(ctx, instr_mask)                # [B_blk, d]
        return block_vecs

    # ------------------------------------------------------------------ #
    def forward(
        self,
        opcodes: Tensor,
        operand_types: Tensor,
        value_buckets: Tensor,
        instr_norm_positions: Tensor,
        block_instr_idx: Tensor,
        instr_mask: Tensor,
        struct_feats: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor,
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        opcodes          : [I]             int64
        operand_types    : [I, 3]          int64
        value_buckets    : [I, 3]          int64
        instr_norm_positions : [I]         float32  i/|BB|
        block_instr_idx  : [total_nodes, L] int64
        instr_mask       : [total_nodes, L] bool
        struct_feats     : [total_nodes, 12] float32
        edge_index       : [2, E]          int64
        edge_type        : [E]             int64
        batch            : [total_nodes]   int64

        Returns
        -------
        dict with keys:
            node_embeddings  : [total_nodes, d]   float32
            graph_embeddings : [batch_size, d]     float32
            pattern_logits   : [total_nodes, C]    float32
        """
        N = struct_feats.size(0)

        # --- Per-block encoding ---
        block_vecs = self._embed_blocks(
            opcodes, operand_types, value_buckets, instr_norm_positions,
            block_instr_idx, instr_mask,
        )                                                            # [N, d]

        # --- Add structural features ---
        struct_vecs = self.struct_mlp(struct_feats)                   # [N, d]
        node_h = block_vecs + struct_vecs                            # [N, d]

        # --- 有向 CFG：GAT / 模式 k-hop 子图均沿控制流方向（§5.3.1 / §5.3.2）---
        # --- Graph attention layers ---
        for gat in self.gat_layers:
            node_h = gat(node_h, edge_index, edge_type)              # [N, d]

        # --- Pattern classification ---
        pattern_logits = self.pattern_clf(
            node_h, edge_index, edge_type, batch,
        )                                                            # [N, C]

        # --- Graph-level pooling ---
        graph_emb = self.graph_pool(node_h, batch)                   # [B, d]

        return {
            "node_embeddings": node_h,
            "graph_embeddings": graph_emb,
            "pattern_logits": pattern_logits,
        }


# ════════════════════════════════════════════════════════════════════════════
# 9.  SourceCFGEncoder
# ════════════════════════════════════════════════════════════════════════════

class _SourceTokenTransformer(nn.Module):
    """Simple token-level Transformer for source code statements.

    Uses Pre-LN, same architecture as BlockTransformer but parameterised
    independently.

    Parameters
    ----------
    vocab_size : int
    d_model    : int  (256)
    n_heads    : int  (4)
    ffn_dim    : int  (1024)
    n_layers   : int  (2)
    max_len    : int  (64)
    dropout    : float (0.1)

    Inputs
    ------
    token_ids : Tensor[B_stmts, L]  int64
    mask      : Tensor[B_stmts, L]  bool

    Returns
    -------
    Tensor[B_stmts, L, d]  float32
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        ffn_dim: int = 1024,
        n_layers: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.transformer = BlockTransformer(
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, token_ids: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        token_ids : [B_s, L]  int64
        mask      : [B_s, L]  bool

        Returns
        -------
        [B_s, L, d]  float32
        """
        B, L = token_ids.shape
        device = token_ids.device
        # §5.4：语句内 token 归一化位置 j/|stmt|（与 §5.2.2 二进制 i/|BB| 对称）+ 正弦编码
        idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        len_e = mask.sum(dim=1).float().clamp(min=1.0).unsqueeze(1)
        norm = (idx / len_e) * mask.float()
        pe = sinusoidal_pe_from_normalized(norm, self.d_model, self.max_len)
        x = self.token_emb(token_ids) + pe
        out = self.transformer(x, mask)                              # [B, L, d]
        return out


class SourceCFGEncoder(nn.Module):
    """源码侧 CFG 编码器（StrucLift_Research_Proposal.md §5.4，训练 Module B 对齐用）。

    与二进制侧对称：
      1. Token Transformer + 语句级池化 + AST 类型嵌入；
      2. 源码 CFG 上边类型感知 GAT（与二进制侧 **不共享参数**）；
      3. 标识符占位化 ``variable→var_#``、``function→func_#`` 应在 **词表/tokenizer 前处理**
         中完成；此处输入为已离散化的 ``token_ids``。

    Expected input dict keys (PyG batch conventions)
    -------------------------------------------------
    token_ids         : Tensor[total_stmts, max_tokens]   int64
    token_mask        : Tensor[total_stmts, max_tokens]   bool
    ast_types         : Tensor[total_nodes]                int64
    edge_index        : Tensor[2, E]                       int64  (COO)
    edge_type         : Tensor[E]                          int64
    batch             : Tensor[total_nodes]                 int64

    Output dict
    -----------
    node_embeddings  : Tensor[total_nodes, d]              float32
    graph_embeddings : Tensor[batch_size, d]                float32

    Parameters
    ----------
    vcfg     : VocabConfig
    ma_cfg   : ModuleAConfig
    src_vocab_size : int  -- source token vocabulary size
    max_tokens     : int  -- max tokens per statement
    """

    def __init__(
        self,
        vcfg: VocabConfig,
        ma_cfg: ModuleAConfig,
        src_vocab_size: int = 32000,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        d = ma_cfg.d_model
        self.d_model = d

        # --- Token Transformer (source) ---
        self.token_transformer = _SourceTokenTransformer(
            vocab_size=src_vocab_size,
            d_model=d,
            n_heads=ma_cfg.instr_transformer_heads,
            ffn_dim=ma_cfg.instr_ffn_dim,
            n_layers=ma_cfg.instr_transformer_layers,
            max_len=max_tokens,
            dropout=ma_cfg.instr_dropout,
        )

        # --- Attentive pooling (tokens -> statement) ---
        self.stmt_pool = AttentivePooling(d)

        # --- AST type embedding ---
        self.ast_type_emb = nn.Embedding(vcfg.num_ast_types, d)

        # --- Edge-typed GAT layers (separate parameters) ---
        self.gat_layers = nn.ModuleList([
            EdgeTypedGATLayer(
                d_model=d,
                n_heads=ma_cfg.gat_heads,
                edge_emb_dim=ma_cfg.edge_emb_dim,
                num_edge_types=vcfg.num_edge_types,
                dropout=ma_cfg.gat_dropout,
            )
            for _ in range(ma_cfg.gat_layers)
        ])

        # --- PMA graph pooling ---
        self.graph_pool = PMAPooling(d_model=d, n_heads=ma_cfg.pma_heads)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        ast_types: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        batch: Tensor,
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        token_ids    : [total_nodes, max_tokens]  int64
        token_mask   : [total_nodes, max_tokens]  bool
        ast_types    : [total_nodes]               int64
        edge_index   : [2, E]                      int64
        edge_type    : [E]                         int64
        batch        : [total_nodes]               int64

        Returns
        -------
        dict with keys:
            node_embeddings  : [total_nodes, d]  float32
            graph_embeddings : [batch_size, d]   float32
        """
        # --- Per-statement encoding ---
        tok_ctx = self.token_transformer(token_ids, token_mask)      # [N, L, d]
        stmt_vecs = self.stmt_pool(tok_ctx, token_mask)              # [N, d]

        # --- Add AST type embeddings ---
        ast_vecs = self.ast_type_emb(ast_types)                      # [N, d]
        node_h = stmt_vecs + ast_vecs                                # [N, d]

        # --- 有向源码 CFG（与二进制侧一致）---
        # --- Graph attention layers ---
        for gat in self.gat_layers:
            node_h = gat(node_h, edge_index, edge_type)            # [N, d]

        # --- Graph-level pooling ---
        graph_emb = self.graph_pool(node_h, batch)                   # [B, d]

        return {
            "node_embeddings": node_h,
            "graph_embeddings": graph_emb,
        }

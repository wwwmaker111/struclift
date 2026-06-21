"""
Module B: Multi-Granularity Cross-Graph Structural Alignment.

Takes node embeddings produced by Module A (binary-side and source-side) and
computes:
  1. A soft alignment matrix  M  [n_b, n_s]  via region-aware SCOT.
  2. Refined (cross-attention enhanced) node representations H_hat_b, H_hat_s.
  3. An optional region-level alignment  M_R  [K_b, K_s].

Classes
-------
AttentivePooling          -- attention-weighted aggregation over a set of vectors
RegionAligner             -- region-level cosine + Sinkhorn alignment with type mask
CrossAttentionLayer       -- bidirectional cross-attention (§6.4.2: Pre-LN + 残差)
CrossAttentionRefinement  -- stacked bidirectional cross-attention with SCOT mask
CrossGraphAlignmentModule -- main Module B entry point
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from struclift.config import ModuleBConfig, VocabConfig
from struclift.data.structures import PATTERN_AST_COMPATIBILITY, PatternType, Skeleton
from struclift.utils.sinkhorn import (
    sinkhorn_from_cost,
    structure_constrained_ot,
    sinkhorn_from_cost_batch,
    structure_constrained_ot_batch,
)
from struclift.utils.graph_utils import compute_region_segmentation
from struclift.utils.skeleton import (
    build_skeleton_for_binary_cfg,
    enrich_skeleton_slots_with_alignment_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════
# Optional fine-grained alignment profiler (off by default, zero overhead)
# ---------------------------------------------------------------------------
# 设 STRUCLIFT_PROFILE_ALIGNMENT=1 后，CrossGraphAlignmentModule.forward 会用
# CUDA Event 把内部各阶段（投影 / 区域对齐 / SCOT / 区域池化 / 区域正样本掩码 /
# 交叉注意力）分别计时，benchmark 脚本可在每步末尾调用 ``report_alignment_profile``
# 取出累计耗时。CPU/无 CUDA 时段时间仍可用 perf_counter 测量。
# ═══════════════════════════════════════════════════════════════════════════

_PROFILE_ALIGNMENT: bool = os.environ.get("STRUCLIFT_PROFILE_ALIGNMENT", "0") == "1"
_PROFILE_EVENTS: List[Tuple[str, "torch.cuda.Event", "torch.cuda.Event"]] = []
_PROFILE_CPU: Dict[str, float] = {}


def _bb_source_order_keys_from_alignment(M: torch.Tensor) -> Optional[List[Tuple[float, float]]]:
    """Build per-BB source-order hints from Module-B alignment matrix.

    Proposal §6.5 says skeleton construction uses both ``M`` and ``P_pat``.
    The deterministic CST builder accepts ``bb_order_key`` for exactly this:
    sibling events can be serialized in source-alignment order instead of raw
    DFS order.  When ``M`` is empty or every row has zero mass, return ``None``
    so inference falls back to CFG order.
    """
    if M is None or M.numel() == 0 or M.dim() != 2:
        return None
    with torch.no_grad():
        mass = M.detach().to(torch.float32)
        row_sum = mass.sum(dim=1)
        if bool((row_sum > 1e-12).sum().item() == 0):
            return None
        n_s = int(mass.size(1))
        if n_s <= 0:
            return None
        positions = torch.arange(n_s, device=mass.device, dtype=torch.float32)
        weighted = (mass * positions.unsqueeze(0)).sum(dim=1)
        argmax = mass.argmax(dim=1).to(torch.float32)
        expected = weighted / row_sum.clamp(min=1e-12)
        keys: List[Tuple[float, float]] = []
        for i in range(int(mass.size(0))):
            if float(row_sum[i].item()) <= 1e-12:
                keys.append((float(i), 0.0))
            else:
                keys.append((float(expected[i].item()), float(argmax[i].item())))
        return keys


def _skeleton_hints_from_alignment(
    M: torch.Tensor,
    pattern_labels: torch.Tensor,
    ast_types: torch.Tensor,
) -> Tuple[Dict[int, int], set[int]]:
    """Infer conservative noop/epilogue hints from ``M``.

    ``M`` is currently reliable enough to order BB siblings and suppress
    obvious empty/prologue/epilogue rows, but not reliable enough to invent
    additional source-level prefix slots.  Prefix slots are therefore left
    disabled here; training-time teacher construction can still fill explicit
    slots if a caller provides them from a stronger source-derived oracle.
    """
    prefix: Dict[int, int] = {}
    skip: set[int] = set()
    if M is None or M.numel() == 0 or M.dim() != 2:
        return prefix, skip
    with torch.no_grad():
        mass = M.detach().to(torch.float32)
        n_b, n_s = int(mass.size(0)), int(mass.size(1))
        if n_b == 0 or n_s == 0:
            return prefix, skip
        labels = [int(x) for x in pattern_labels.detach().cpu().tolist()]
        ast = [int(x) for x in ast_types.detach().cpu().tolist()] if ast_types is not None else []
        row_sum = mass.sum(dim=1)
        argmax = mass.argmax(dim=1)
        for i in range(min(n_b, len(labels))):
            if float(row_sum[i].item()) <= 1e-12:
                skip.add(i)
                continue
            lab = labels[i]
            j = int(argmax[i].item())
            if lab not in {
                int(PatternType.IF_HEADER),
                int(PatternType.LOOP_HEADER),
                int(PatternType.SWITCH_HEADER),
            } and j == 0 and i != 0:
                # A non-entry non-header mapped to the first source node is
                # usually dummy/prologue/epilogue mass.  Do not force it into a
                # statement slot.
                skip.add(i)
    return prefix, skip


class _AlignSection:
    """上下文管理器：在 CUDA 上用 Event 计时，CPU 侧用 perf_counter 计时。

    完全无依赖；当 ``_PROFILE_ALIGNMENT`` 为 False 时退化为无操作。
    """

    __slots__ = ("name", "device", "_use_cuda", "_t0", "_ev_s", "_ev_e")

    def __init__(self, name: str, device: torch.device) -> None:
        self.name = name
        self.device = device
        self._use_cuda = _PROFILE_ALIGNMENT and device.type == "cuda"

    def __enter__(self) -> "_AlignSection":
        if not _PROFILE_ALIGNMENT:
            return self
        if self._use_cuda:
            self._ev_s = torch.cuda.Event(enable_timing=True)
            self._ev_e = torch.cuda.Event(enable_timing=True)
            self._ev_s.record()
        else:
            import time as _t
            self._t0 = _t.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if not _PROFILE_ALIGNMENT:
            return
        if self._use_cuda:
            self._ev_e.record()
            _PROFILE_EVENTS.append((self.name, self._ev_s, self._ev_e))
        else:
            import time as _t
            _PROFILE_CPU[self.name] = _PROFILE_CPU.get(self.name, 0.0) + (
                _t.perf_counter() - self._t0
            )


def report_alignment_profile() -> Dict[str, float]:
    """汇总并清空累计的 alignment profile，返回 ``{section_name: 秒}``。

    调用前会做一次 ``cuda.synchronize`` 以读取 CUDA event 弹性时间。返回的
    字典按时间从大到小排序，便于在日志中观察热点。
    """
    out: Dict[str, float] = {}
    if not _PROFILE_ALIGNMENT:
        return out
    if _PROFILE_EVENTS:
        torch.cuda.synchronize()
        for name, s, e in _PROFILE_EVENTS:
            out[name] = out.get(name, 0.0) + s.elapsed_time(e) / 1000.0
        _PROFILE_EVENTS.clear()
    for name, t in _PROFILE_CPU.items():
        out[name] = out.get(name, 0.0) + t
    _PROFILE_CPU.clear()
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def reset_alignment_profile() -> None:
    """清空累计的计时器（不汇总），用于跨 step 切片测量。"""
    _PROFILE_EVENTS.clear()
    _PROFILE_CPU.clear()


def _extend_scot_with_dummy_sink(
    S: torch.Tensor,
    nu: torch.Tensor,
    ast_types: torch.Tensor,
    source_adj: Optional[torch.Tensor],
    dummy_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """§6.3.7：在 OT 中增加 dummy 目标列，列边际 ν_∅ = scale · n_b / max(n_s,1)。

    相似度在 dummy 列上为 0（代价适中），结构惩罚对 dummy 列置零；源码侧邻接扩维后 dummy 仅自环。
    返回扩维后的 (S_ext, nu_ext, ast_ext, source_adj_ext) 与原始列数 n_s。
    """
    n_b, n_s = S.shape[0], S.shape[1]
    if n_s < 1:
        return S, nu, ast_types, source_adj, n_s
    dummy_frac = float(dummy_scale) * n_b / max(n_s, 1)
    dummy_frac = min(dummy_frac, 0.45)
    if dummy_frac < 1e-8:
        return S, nu, ast_types, source_adj, n_s
    device, dtype = S.device, S.dtype
    nu_real = nu * (1.0 - dummy_frac)
    nu_ext = torch.cat([nu_real, nu.new_tensor([dummy_frac])])
    S_ext = torch.cat([S, torch.zeros(n_b, 1, device=device, dtype=dtype)], dim=1)
    ast_ext = torch.cat([ast_types, ast_types.new_zeros(1)])
    adj_ext: Optional[torch.Tensor] = None
    if source_adj is not None:
        adj_ext = torch.zeros(n_s + 1, n_s + 1, device=source_adj.device, dtype=source_adj.dtype)
        adj_ext[:n_s, :n_s] = source_adj
        adj_ext[n_s, n_s] = 1.0
    return S_ext, nu_ext, ast_ext, adj_ext, n_s


# ═══════════════════════════════════════════════════════════════════════════
# Helper: dataclass returned by the forward pass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModuleBOutput:
    """Container for all outputs of :class:`CrossGraphAlignmentModule`."""

    alignment_matrix: torch.Tensor
    """Soft alignment matrix  M  with shape [n_b, n_s]."""

    binary_node_embs: torch.Tensor
    """Refined binary node embeddings  H_hat_b  with shape [n_b, d]."""

    source_node_embs: torch.Tensor
    """Refined source node embeddings  H_hat_s  with shape [n_s, d]."""

    region_alignment: Optional[torch.Tensor]
    """Region-level soft alignment  M_R  with shape [K_b, K_s], or *None*
    when region-level alignment was not performed."""

    cross_attn_weights: Optional[torch.Tensor] = None
    """Bidirectionally averaged cross-attention  A_cross  [n_b, n_s] for
    :math:`\\mathcal{L}_{\\text{ca\\_align}}`."""

    region_bin_embs: Optional[torch.Tensor] = None
    """Pooled region vectors on binary side  [K_b, d]."""

    region_src_embs: Optional[torch.Tensor] = None
    """Pooled region vectors on source side  [K_s, d]."""

    region_positive_mask: Optional[torch.Tensor] = None
    """Positive pairs for region InfoNCE  [K_b, K_s] (from M_gt mass or M_R)."""

    skeleton: Optional[Skeleton] = None
    """§6.5：控制结构树序列化后的代码骨架（含 ``<FILL_*>``）；无 CFG 边或未启用时为 *None*。"""

    predicted_n_s: Optional[torch.Tensor] = None
    """§6.5.3：推理时预测的源码语句数。"""


def build_region_positive_mask(
    M_gt: torch.Tensor,
    binary_regions: List[Dict],
    source_regions: List[Dict],
    threshold: float = 0.01,
) -> torch.Tensor:
    """Mark region pair (k, l) positive if mass of M_gt over that block exceeds
    *threshold* (doc: ground-truth alignment induces region correspondence).

    向量化实现：构造区域→节点的 0/1 隶属矩阵 ``B_m``、``S_m``，则
    ``mass = B_m @ M_gt @ S_m.T``。原版每个 (k,l) 都要 ``sub.sum() > thr`` 触发
    一次 GPU sync（K_b·K_s 次 / 图，batch 内累计极重），这里改成一次 matmul +
    一次比较，数值与原版等价。
    """
    Kb = len(binary_regions)
    Ks = len(source_regions)
    device = M_gt.device
    dtype = M_gt.dtype
    n_b, n_s = int(M_gt.shape[0]), int(M_gt.shape[1])
    if Kb == 0 or Ks == 0 or n_b == 0 or n_s == 0:
        return torch.zeros(Kb, Ks, device=device, dtype=dtype)

    rows_b: List[int] = []
    cols_b: List[int] = []
    for k, br in enumerate(binary_regions):
        for i in br.get("nodes", []) or []:
            if 0 <= i < n_b:
                rows_b.append(k)
                cols_b.append(i)
    rows_s: List[int] = []
    cols_s: List[int] = []
    for l, sr in enumerate(source_regions):
        for j in sr.get("nodes", []) or []:
            if 0 <= j < n_s:
                rows_s.append(l)
                cols_s.append(j)
    if not rows_b or not rows_s:
        return torch.zeros(Kb, Ks, device=device, dtype=dtype)

    B_m = torch.zeros(Kb, n_b, device=device, dtype=dtype)
    S_m = torch.zeros(Ks, n_s, device=device, dtype=dtype)
    rb = torch.tensor(rows_b, device=device, dtype=torch.long)
    cb = torch.tensor(cols_b, device=device, dtype=torch.long)
    rs = torch.tensor(rows_s, device=device, dtype=torch.long)
    cs = torch.tensor(cols_s, device=device, dtype=torch.long)
    B_m[rb, cb] = 1.0
    S_m[rs, cs] = 1.0

    mass = B_m @ M_gt @ S_m.t()  # [Kb, Ks]
    return (mass > threshold).to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Attentive Pooling
# ═══════════════════════════════════════════════════════════════════════════

class AttentivePooling(nn.Module):
    """Attention-weighted aggregation over a variable-size set of vectors.

    Given a set of vectors  {h_i}  (members of a region), computes:

        a_i = softmax(w^T tanh(W h_i + b))
        r   = sum_i  a_i * h_i

    Parameters
    ----------
    d_model : int
        Dimensionality of input vectors.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor
            Shape [N, d].  The set of member node embeddings.
        mask : Tensor, optional
            Boolean mask of shape [N], *True* = keep, *False* = pad/ignore.

        Returns
        -------
        Tensor
            Shape [d].  The pooled region vector.
        """
        # h: [N, d]
        energy: torch.Tensor = self.score(torch.tanh(self.proj(h)))  # [N, 1]
        energy = energy.squeeze(-1)  # [N]

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        attn: torch.Tensor = F.softmax(energy, dim=0)  # [N]
        pooled: torch.Tensor = (attn.unsqueeze(-1) * h).sum(dim=0)  # [d]
        return pooled


# ═══════════════════════════════════════════════════════════════════════════
# Region Aligner
# ═══════════════════════════════════════════════════════════════════════════

class RegionAligner(nn.Module):
    """Computes region-level soft alignment via cosine similarity + Sinkhorn.

    Each *region* is an attributed subgraph of the CFG (e.g., a loop body,
    an if-header) identified by Module A's pattern classifier.  The aligner:

    1. Pools each region with :class:`AttentivePooling`.
    2. Computes pairwise cosine similarity scaled by temperature.
    3. Masks out incompatible region-type pairs.
    4. Applies Sinkhorn normalisation to obtain a doubly-stochastic-like
       soft alignment matrix  M_R.

    Parameters
    ----------
    cfg : ModuleBConfig
    vocab_cfg : VocabConfig
    """

    # ---- type compatibility (region-level, coarser than node-level) -----
    # Binary region types are PatternType ints; source region types are
    # ASTType ints.  We reuse the canonical compatibility dictionary from
    # ``structures.py``.

    def __init__(self, cfg: ModuleBConfig, vocab_cfg: VocabConfig) -> None:
        super().__init__()
        self.tau: float = cfg.region_tau
        self.sinkhorn_eps: float = cfg.region_sinkhorn_eps
        self.sinkhorn_iter: int = cfg.region_sinkhorn_iter
        self.confidence_threshold: float = cfg.region_confidence_threshold

        self.pool = AttentivePooling(cfg.d_model)

    # -----------------------------------------------------------------
    @staticmethod
    def _build_compatibility_mask(
        binary_types: List[int],
        source_types: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Return a boolean mask [K_b, K_s] where *True* = compatible.

        Uses :data:`PATTERN_AST_COMPATIBILITY` to decide whether a
        (PatternType, ASTType) pair is compatible.

        在 CPU 上构 nested list，再一次 H2D —— 原版每对 (i,j) 都做 ``mask[i,j]=True``
        的标量赋值会触发一次小 H2D，K_b·K_s 累加在 batch 内是大头，但语义无变化。
        """
        K_b = len(binary_types)
        K_s = len(source_types)
        if K_b == 0 or K_s == 0:
            return torch.zeros(K_b, K_s, dtype=torch.bool, device=device)
        rows: List[List[bool]] = [[False] * K_s for _ in range(K_b)]
        for i, btype in enumerate(binary_types):
            compatible = PATTERN_AST_COMPATIBILITY.get(btype, [])
            cset = set(compatible)
            if not cset:
                continue
            row = rows[i]
            for j, stype in enumerate(source_types):
                if stype in cset:
                    row[j] = True
        return torch.tensor(rows, dtype=torch.bool, device=device)

    # -----------------------------------------------------------------
    def _pool_regions(
        self,
        node_embs: torch.Tensor,
        regions: List[Dict],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Pool each region to a single vector via attentive pooling.

        Parameters
        ----------
        node_embs : Tensor
            Shape [N, d].
        regions : list of dict
            Each dict has keys ``'nodes'`` (list of int indices) and
            ``'type'`` (int region type).

        Returns
        -------
        region_vecs : Tensor
            Shape [K, d].
        region_types : list of int
            Length K.

        实现说明
        --------
        把所有 region 的节点索引在 CPU 上拼成 ``flat_idx`` 与 ``flat_rid``，一次
        H2D 后用一次 ``score(tanh(proj(.)))`` 算出全部能量；按 ``rid`` 做 segment
        softmax（``scatter_reduce_amax`` + ``scatter_add``）后做加权求和。原版每
        region 一个 H2D + 一个 attentive pool kernel，K 个 region 串行；这里全
        部融成 4~5 个大 kernel，与原版数值完全等价。
        """
        K = len(regions)
        d = int(node_embs.size(1))
        device = node_embs.device
        dtype = node_embs.dtype

        if K == 0:
            return torch.zeros(0, d, device=device, dtype=dtype), []

        flat_nodes: List[int] = []
        flat_rid: List[int] = []
        types: List[int] = []
        for ri, region in enumerate(regions):
            nodes = region.get("nodes") or []
            types.append(int(region.get("type", 0)))
            for idx in nodes:
                flat_nodes.append(int(idx))
                flat_rid.append(ri)

        if not flat_nodes:
            return torch.zeros(K, d, device=device, dtype=dtype), types

        idx_t = torch.tensor(flat_nodes, dtype=torch.long, device=device)
        rid_t = torch.tensor(flat_rid, dtype=torch.long, device=device)

        h_all = node_embs.index_select(0, idx_t)  # [T, d]
        energy = self.pool.score(torch.tanh(self.pool.proj(h_all))).squeeze(-1)  # [T]

        # segment softmax over rid:
        neg_inf = torch.full((K,), float("-inf"), device=device, dtype=energy.dtype)
        max_per_r = neg_inf.scatter_reduce(0, rid_t, energy, reduce="amax", include_self=True)
        # 空 region 的 max 仍为 -inf：归一化时分母为 0，这里把它替换为 0，确保
        # 后续 sum 给该 region 输出零向量（与原版 ``zeros(d)`` 行为一致）。
        max_safe = torch.where(torch.isfinite(max_per_r), max_per_r, torch.zeros_like(max_per_r))
        e_norm = (energy - max_safe.index_select(0, rid_t)).exp()  # [T]
        sum_per_r = torch.zeros(K, device=device, dtype=energy.dtype)
        sum_per_r = sum_per_r.scatter_add(0, rid_t, e_norm)
        attn = e_norm / sum_per_r.index_select(0, rid_t).clamp(min=1e-30)  # [T]

        weighted = h_all * attn.unsqueeze(-1)  # [T, d]
        region_vecs = torch.zeros(K, d, device=device, dtype=dtype)
        region_vecs = region_vecs.scatter_add(
            0, rid_t.unsqueeze(-1).expand(-1, d), weighted,
        )
        # 空 region 在 flat_rid 里没有对应元素，scatter_add 后就是 0 向量，与原版
        # 显式 ``torch.zeros(d)`` fallback 等价；无需额外 mask 操作（避免 GPU sync）。
        return region_vecs, types

    # -----------------------------------------------------------------
    def forward(
        self,
        binary_node_embs: torch.Tensor,
        source_node_embs: torch.Tensor,
        binary_regions: List[Dict],
        source_regions: List[Dict],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        binary_node_embs : Tensor
            Shape [n_b, d].
        source_node_embs : Tensor
            Shape [n_s, d].
        binary_regions : list of dict
            ``[{'nodes': [int, ...], 'type': int}, ...]`` for the binary
            CFG.
        source_regions : list of dict
            Same schema for the source CFG.

        Returns
        -------
        M_R : Tensor
            Soft region alignment matrix with shape [K_b, K_s].
        aligned_pairs : list of (int, int)
            Greedy-matched (binary_region_idx, source_region_idx) pairs
            extracted from *M_R*.
        bin_vecs : Tensor
            Pooled binary-side region embeddings ``[K_b, d]``.
        src_vecs : Tensor
            Pooled source-side region embeddings ``[K_s, d]``.

        说明：把 pooled vecs 也返回出去，让外层 ``CrossGraphAlignmentModule.forward``
        直接复用，避免在 §6.6 region InfoNCE 之前再跑一遍 ``_pool_regions``
        （原版会跑两次 attentive pooling，每图多一组 ragged kernel）。
        """
        device = binary_node_embs.device

        # 1. Attentive pooling ------------------------------------------------
        bin_vecs, bin_types = self._pool_regions(binary_node_embs, binary_regions)
        src_vecs, src_types = self._pool_regions(source_node_embs, source_regions)
        # bin_vecs: [K_b, d], src_vecs: [K_s, d]

        K_b, K_s = bin_vecs.size(0), src_vecs.size(0)

        # 2. Cosine similarity / tau ------------------------------------------
        bin_norm = F.normalize(bin_vecs, p=2, dim=-1)  # [K_b, d]
        src_norm = F.normalize(src_vecs, p=2, dim=-1)  # [K_s, d]
        S_R: torch.Tensor = (bin_norm @ src_norm.t()) / self.tau  # [K_b, K_s]

        # 3. Type compatibility mask -------------------------------------------
        compat_mask = self._build_compatibility_mask(bin_types, src_types, device)
        S_R = S_R.masked_fill(~compat_mask, float("-inf"))

        # 4. Sinkhorn ----------------------------------------------------------
        # Convert similarity to cost (negate) for the Sinkhorn solver.
        # Entries at -inf stay at +inf cost, naturally excluded.
        cost = -S_R  # [K_b, K_s]
        M_R: torch.Tensor = sinkhorn_from_cost(
            cost,
            eps=self.sinkhorn_eps,
            max_iter=self.sinkhorn_iter,
        )  # [K_b, K_s]

        # 5. §6.3.1 仅保留高置信度区域对 -----------------------------------
        aligned_pairs = self._greedy_pairs(
            M_R, compat_mask,
            confidence_threshold=self.confidence_threshold,
        )

        return M_R, aligned_pairs, bin_vecs, src_vecs

    # -----------------------------------------------------------------
    @staticmethod
    def _greedy_pairs(
        M_R: torch.Tensor,
        compat_mask: torch.Tensor,
        confidence_threshold: float = 0.0,
    ) -> List[Tuple[int, int]]:
        """§6.3.1：从 *M_R* 中贪心取高置信度区域对。

        仅保留 ``M_R[bi, si] >= confidence_threshold`` 的对，
        与文档「高置信度对齐的区域对」一致。

        实现：把 ``M_R`` / ``compat_mask`` 一次性 D2H（仅一个 K_b·K_s 的小矩阵），
        随后在 CPU 上做贪心选择。原版每个循环步两次 ``.item()`` GPU sync，
        在 batch 内放大成几百次阻塞同步，是 alignment forward 的隐性热点。
        """
        K_b, K_s = M_R.shape
        if K_b == 0 or K_s == 0:
            return []
        # 一次同步取小矩阵到 CPU；后续都是纯 Python，无设备同步。
        M_cpu = M_R.detach().to("cpu", copy=True)
        mask_cpu = compat_mask.detach().to("cpu")
        M_cpu[~mask_cpu] = 0.0

        pairs: List[Tuple[int, int]] = []
        num_pairs = min(int(K_b), int(K_s))
        for _ in range(num_pairs):
            flat = int(M_cpu.argmax().item())
            bi = flat // int(K_s)
            si = flat % int(K_s)
            val = float(M_cpu[bi, si].item())
            if val <= 0.0 or val < confidence_threshold:
                break
            pairs.append((bi, si))
            M_cpu[bi, :] = 0.0
            M_cpu[:, si] = 0.0
        return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Attention Layer (single layer, bidirectional)
# ═══════════════════════════════════════════════════════════════════════════

class CrossAttentionLayer(nn.Module):
    """§6.4.2 单层双向交叉注意力（**Pre-LN + 残差**，与提案文字要点 3 一致）。

    每层内先后做 b→s 与 s→b 两个子步；**每个子步**均为
    ``输出 = 残差流 + MHA( LN(残差流) + 类型嵌入 )``，不在残差后再套 LN。

    binary → source::

        nb = LN(H_b),  ns = LN(H_s)
        Q_b = W_q(nb + E_pat),  K_s = W_k(ns + E_ast),  V_s = W_v(ns)
        H_b ← H_b + MHA(Q_b, K_s, V_s, mask)

    source → binary 对称（对更新后的 ``H_b`` 再 LN 后作 K/V）。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Pre-LN：二进制侧 / 源码侧各一个（同一层内两个子步复用）
        self.norm_b = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)

        # binary → source
        self.q_proj_b2s = nn.Linear(d_model, d_model)
        self.k_proj_b2s = nn.Linear(d_model, d_model)
        self.v_proj_b2s = nn.Linear(d_model, d_model)
        self.out_proj_b2s = nn.Linear(d_model, d_model)

        # source → binary
        self.q_proj_s2b = nn.Linear(d_model, d_model)
        self.k_proj_s2b = nn.Linear(d_model, d_model)
        self.v_proj_s2b = nn.Linear(d_model, d_model)
        self.out_proj_s2b = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)

    # -----------------------------------------------------------------
    def _multi_head_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        out_proj: nn.Linear,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard multi-head attention → (out, attn_mean_over_heads)."""
        n_q = Q.size(0)
        n_k = K.size(0)
        Q = Q.view(n_q, self.n_heads, self.d_head).transpose(0, 1)
        K = K.view(n_k, self.n_heads, self.d_head).transpose(0, 1)
        V = V.view(n_k, self.n_heads, self.d_head).transpose(0, 1)
        attn: torch.Tensor = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn_mean = attn.mean(dim=0)
        attn = self.attn_drop(attn)
        out: torch.Tensor = attn @ V
        out = out.transpose(0, 1).contiguous().view(n_q, self.d_model)
        return out_proj(out), attn_mean

    # -----------------------------------------------------------------
    def forward(
        self,
        H_b: torch.Tensor,
        H_s: torch.Tensor,
        Q_bias_b: torch.Tensor,
        K_bias_b: torch.Tensor,
        Q_bias_s: torch.Tensor,
        K_bias_s: torch.Tensor,
        mask_b2s: Optional[torch.Tensor],
        mask_s2b: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (H_b_out, H_s_out, attn_b2s, attn_s2b)."""
        # Pre-LN 子步 1：binary ← attend to source
        nb = self.norm_b(H_b)
        ns = self.norm_s(H_s)
        q_b = self.q_proj_b2s(nb + Q_bias_b)
        k_s = self.k_proj_b2s(ns + K_bias_s)
        v_s = self.v_proj_b2s(ns)
        attn_out_b, attn_b2s = self._multi_head_attention(
            q_b, k_s, v_s, self.out_proj_b2s, mask_b2s,
        )
        H_b = H_b + attn_out_b

        # Pre-LN 子步 2：source ← attend to binary（对更新后的 H_b 再 LN）
        nb = self.norm_b(H_b)
        ns = self.norm_s(H_s)
        q_s = self.q_proj_s2b(ns + Q_bias_s)
        k_b = self.k_proj_s2b(nb + K_bias_b)
        v_b = self.v_proj_s2b(nb)
        attn_out_s, attn_s2b = self._multi_head_attention(
            q_s, k_b, v_b, self.out_proj_s2b, mask_s2b,
        )
        H_s = H_s + attn_out_s

        return H_b, H_s, attn_b2s, attn_s2b


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Attention Refinement (stacked layers)
# ═══════════════════════════════════════════════════════════════════════════

class CrossAttentionRefinement(nn.Module):
    """Stacked bidirectional cross-attention with SCOT-derived attention mask.

    Two layers of :class:`CrossAttentionLayer`, each with separate parameters
    for the two directions.  The alignment matrix *M* is thresholded to produce
    a sparse boolean mask so that each node only attends to its plausible
    counterparts.

    Parameters
    ----------
    cfg : ModuleBConfig
    vocab_cfg : VocabConfig
    """

    def __init__(self, cfg: ModuleBConfig, vocab_cfg: VocabConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.mask_threshold: float = cfg.cross_attn_mask_threshold

        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=d,
                n_heads=cfg.cross_attn_heads,
                dropout=cfg.cross_attn_dropout,
            )
            for _ in range(cfg.cross_attn_layers)
        ])

        # Learnable bias embeddings injected into Q / K.
        self.pattern_emb = nn.Embedding(vocab_cfg.num_patterns, d)
        self.ast_type_emb = nn.Embedding(vocab_cfg.num_ast_types, d)

        # §6.4.2 Pre-LN：每层末为残差流，无后置 LN；堆叠下一层时由下一层的 norm_* 处理

    # -----------------------------------------------------------------
    def forward(
        self,
        H_b: torch.Tensor,
        H_s: torch.Tensor,
        M: torch.Tensor,
        patterns: torch.Tensor,
        ast_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H_b : Tensor  [n_b, d]
            Binary node embeddings from Module A.
        H_s : Tensor  [n_s, d]
            Source node embeddings from Module A.
        M : Tensor  [n_b, n_s]
            Soft alignment matrix from SCOT.
        patterns : Tensor  [n_b]  int64
            Per-node structural pattern labels for binary nodes.
        ast_types : Tensor  [n_s]  int64
            Per-node AST type labels for source nodes.

        Returns
        -------
        H_hat_b : Tensor  [n_b, d]
        H_hat_s : Tensor  [n_s, d]
        A_cross : Tensor  [n_b, n_s]
            Average of bidirectional cross-attention (row-normalised per layer)
            and averaged over layers.
        """
        # Build boolean attention masks from alignment matrix.
        mask_b2s: torch.Tensor = M > self.mask_threshold       # [n_b, n_s]
        mask_s2b: torch.Tensor = M.t() > self.mask_threshold   # [n_s, n_b]

        # Ensure every row has at least one *True* to avoid fully-masked softmax.
        # Fall back to full attention for any such row.
        _ensure_at_least_one(mask_b2s)
        _ensure_at_least_one(mask_s2b)

        # Compute additive bias vectors.
        pat_emb: torch.Tensor = self.pattern_emb(patterns)     # [n_b, d]
        ast_emb: torch.Tensor = self.ast_type_emb(ast_types)   # [n_s, d]

        # Q_bias_b = K_bias_b = pattern_emb (binary queries/keys)
        # Q_bias_s = K_bias_s = ast_type_emb (source queries/keys)
        H_b_cur, H_s_cur = H_b, H_s
        layer_cross: List[torch.Tensor] = []
        for layer in self.layers:
            H_b_cur, H_s_cur, a_b2s, a_s2b = layer(
                H_b=H_b_cur,
                H_s=H_s_cur,
                Q_bias_b=pat_emb,
                K_bias_b=pat_emb,
                Q_bias_s=ast_emb,
                K_bias_s=ast_emb,
                mask_b2s=mask_b2s,
                mask_s2b=mask_s2b,
            )
            # Doc §6.6: bidirectional cross-attention — use average of b↔s
            # (b→s) and transpose of (s→b) to obtain [n_b, n_s].
            A_layer = 0.5 * (a_b2s + a_s2b.t())
            layer_cross.append(A_layer)

        A_cross = torch.stack(layer_cross, dim=0).mean(dim=0)  # [n_b, n_s]

        return H_b_cur, H_s_cur, A_cross


def _ensure_at_least_one(mask: torch.Tensor) -> None:
    """In-place: for any row that is entirely *False*, set the entire row to
    *True* so that softmax attention degrades gracefully to uniform rather
    than producing NaN.

    Parameters
    ----------
    mask : Tensor  [N, M]  bool
    """
    all_false_rows = ~mask.any(dim=1)  # [N]
    if all_false_rows.any():
        mask[all_false_rows] = True


# ═══════════════════════════════════════════════════════════════════════════
# Main Module B: CrossGraphAlignmentModule
# ═══════════════════════════════════════════════════════════════════════════

class CrossGraphAlignmentModule(nn.Module):
    """Module B -- Multi-Granularity Cross-Graph Structural Alignment.

    Orchestrates region-level alignment (:class:`RegionAligner`), node-level
    structure-constrained optimal transport (SCOT), and cross-attention
    refinement (:class:`CrossAttentionRefinement`).

    Depending on whether region information is available (i.e. the function
    is large enough to have meaningful structural regions), the alignment
    operates in one of two modes:

    * **Region-aware mode**: first align regions, then run within-region
      SCOT, and stitch results into a global alignment matrix.
    * **Flat mode**: run SCOT directly on the full similarity matrix (for
      small functions without identifiable regions).

    Parameters
    ----------
    cfg : ModuleBConfig
    vocab_cfg : VocabConfig
    """

    def __init__(
        self,
        cfg: ModuleBConfig | None = None,
        vocab_cfg: VocabConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ModuleBConfig()
        self.vocab_cfg = vocab_cfg or VocabConfig()

        d = self.cfg.d_model
        self.bin_proj = nn.Linear(d, d)
        self.src_proj = nn.Linear(d, d)

        self.region_aligner = RegionAligner(self.cfg, self.vocab_cfg)
        self.cross_attn = CrossAttentionRefinement(self.cfg, self.vocab_cfg)

        # SCOT hyper-parameters
        self.scot_tau: float = self.cfg.scot_tau
        self.scot_eps: float = self.cfg.scot_eps
        self.scot_sinkhorn_iter: int = self.cfg.scot_sinkhorn_iter
        self.scot_outer_iter: int = self.cfg.scot_outer_iter
        self.scot_lambda_struct: float = self.cfg.scot_lambda_struct
        self.scot_gamma_topo: float = self.cfg.scot_gamma_topo

        # §6.5.3 回归头：从二进制图级嵌入预测源码语句数 n_s
        if self.cfg.src_count_regressor:
            self.n_s_regressor = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.ReLU(),
                nn.Linear(d // 2, 1),
            )

    # ── helpers ──────────────────────────────────────────────────────

    def _cosine_similarity_matrix(
        self,
        H_b: torch.Tensor,
        H_s: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temperature-scaled cosine similarity.

        Parameters
        ----------
        H_b : Tensor  [n_b, d]
        H_s : Tensor  [n_s, d]

        Returns
        -------
        S : Tensor  [n_b, n_s]
        """
        H_b_norm = F.normalize(H_b, p=2, dim=-1)  # [n_b, d]
        H_s_norm = F.normalize(H_s, p=2, dim=-1)  # [n_s, d]
        S: torch.Tensor = (H_b_norm @ H_s_norm.t()) / self.scot_tau  # [n_b, n_s]
        return S

    # -----------------------------------------------------------------
    def _run_scot_with_dummy(
        self,
        S: torch.Tensor,
        binary_adj: Optional[torch.Tensor],
        source_adj: Optional[torch.Tensor],
        pattern_probs: torch.Tensor,
        ast_types: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        """§6.3.7：统一 SCOT 入口，含 dummy 列（贯穿所有路径）。

        【单图版，保留作为参考 / fallback】生产路径已切换到 :meth:`_run_scot_batch`，
        将一组子问题 padded 到统一形状一次 batched 求解，避免逐子问题 launch
        overhead。本方法仅在调用方明确不走 batch 路径时使用。
        """
        n_s_orig = S.size(1)
        n_b = S.size(0)
        device, dtype = S.device, S.dtype
        S_ot, nu_ot, ast_ot, src_adj_ot = S, nu, ast_types, source_adj
        if self.cfg.scot_use_dummy_sink and n_s_orig > 0:
            S_ot, nu_ot, ast_ot, src_adj_ot, _ = _extend_scot_with_dummy_sink(
                S, nu, ast_types, source_adj, self.cfg.scot_dummy_mass_scale,
            )
        cost = -S_ot
        type_penalty_mask: Optional[torch.Tensor] = None
        if S_ot.size(1) > n_s_orig:
            type_penalty_mask = torch.ones(n_b, S_ot.size(1), device=device, dtype=dtype)
            type_penalty_mask[:, -1] = 0.0
        if binary_adj is not None and src_adj_ot is not None:
            M_ot: torch.Tensor = structure_constrained_ot(
                cost=cost, D_x=binary_adj, D_y=src_adj_ot,
                pattern_probs=pattern_probs, ast_types=ast_ot,
                mu=mu, nu=nu_ot,
                eps=self.scot_eps, max_iter=self.scot_sinkhorn_iter,
                outer_iter=self.scot_outer_iter,
                lambda_struct=self.scot_lambda_struct,
                gamma_topo=self.scot_gamma_topo,
                type_penalty_mask=type_penalty_mask,
            )
        else:
            M_ot = sinkhorn_from_cost(
                cost, mu=mu, nu=nu_ot,
                eps=self.scot_eps, max_iter=self.scot_sinkhorn_iter,
            )
        if M_ot.size(1) > n_s_orig:
            return M_ot[:, :n_s_orig]
        return M_ot

    # -----------------------------------------------------------------
    def _run_scot_batch(
        self,
        sub_problems: List[Tuple[torch.Tensor, torch.Tensor]],
        S: torch.Tensor,
        binary_adj: Optional[torch.Tensor],
        source_adj: Optional[torch.Tensor],
        pattern_probs: torch.Tensor,
        ast_types: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
    ) -> List[torch.Tensor]:
        """把一组 SCOT 子问题 padded 到统一形状一次 batched 求解。

        ``sub_problems`` 每项是 ``(b_idx [n_b_i], s_idx [n_s_i])`` —— 子问题在
        全图坐标下要参与 SCOT 的节点子集。每个子问题独立做 §6.3.7 dummy 列、
        独立 marginal 归一化、独立 SCOT，逻辑与 :meth:`_run_scot_with_dummy`
        完全一致；返回的 ``M_subs[i]`` 形状 ``[n_b_i, n_s_i]``，与原版相同。

        关键收益：原版每图 ~6 次 SCOT 串行 launch（每次 6 ×
        ``scot_sinkhorn_iter`` 个 logsumexp，矩阵小 → kernel launch 开销远大于
        计算）；批化后一次 SCOT call 处理整图所有子问题，launch 摊薄 6 倍。
        """
        B = len(sub_problems)
        if B == 0:
            return []
        device, dtype = S.device, S.dtype
        P = int(pattern_probs.size(1))

        n_b_list = [int(b.numel()) for b, _ in sub_problems]
        n_s_list = [int(s.numel()) for _, s in sub_problems]
        max_n_b = max(n_b_list)
        max_n_s = max(n_s_list)
        if max_n_b == 0 or max_n_s == 0:
            return [torch.zeros(nb, ns, device=device, dtype=dtype)
                    for nb, ns in zip(n_b_list, n_s_list)]

        use_dummy = bool(self.cfg.scot_use_dummy_sink)
        use_topo = (binary_adj is not None) and (source_adj is not None)

        S_pad = torch.zeros(B, max_n_b, max_n_s, device=device, dtype=dtype)
        pp_pad = torch.zeros(B, max_n_b, P, device=device, dtype=dtype)
        at_pad = torch.zeros(B, max_n_s, device=device, dtype=ast_types.dtype)
        mu_pad = torch.zeros(B, max_n_b, device=device, dtype=dtype)
        nu_pad = torch.zeros(B, max_n_s, device=device, dtype=dtype)
        mu_mask = torch.zeros(B, max_n_b, device=device, dtype=torch.bool)
        nu_mask = torch.zeros(B, max_n_s, device=device, dtype=torch.bool)
        if use_topo:
            adj_b_pad = torch.zeros(
                B, max_n_b, max_n_b, device=device, dtype=binary_adj.dtype,  # type: ignore[union-attr]
            )
            adj_s_pad = torch.zeros(
                B, max_n_s, max_n_s, device=device, dtype=source_adj.dtype,  # type: ignore[union-attr]
            )
        else:
            adj_b_pad = None
            adj_s_pad = None

        # 把每个子问题切片填进 padded buffer。Python 循环这里只跑 B (~6) 次，
        # 每次都是一个 slice/index_select，开销远小于原版 B 次 SCOT。
        for i, (b_idx, s_idx) in enumerate(sub_problems):
            nb_i = n_b_list[i]
            ns_i = n_s_list[i]
            if nb_i == 0 or ns_i == 0:
                continue
            S_pad[i, :nb_i, :ns_i] = S.index_select(0, b_idx).index_select(1, s_idx)
            pp_pad[i, :nb_i, :] = pattern_probs.index_select(0, b_idx)
            at_pad[i, :ns_i] = ast_types.index_select(0, s_idx)
            mu_sub = mu.index_select(0, b_idx)
            mu_pad[i, :nb_i] = mu_sub / mu_sub.sum().clamp(min=1e-30)
            nu_sub = nu.index_select(0, s_idx)
            nu_pad[i, :ns_i] = nu_sub / nu_sub.sum().clamp(min=1e-30)
            mu_mask[i, :nb_i] = True
            nu_mask[i, :ns_i] = True
            if use_topo:
                adj_b_pad[i, :nb_i, :nb_i] = binary_adj.index_select(0, b_idx).index_select(1, b_idx)  # type: ignore[union-attr]
                adj_s_pad[i, :ns_i, :ns_i] = source_adj.index_select(0, s_idx).index_select(1, s_idx)  # type: ignore[union-attr]

        # §6.3.7 dummy 列：每子问题独立 dummy_frac，统一 cat 在 max_n_s 之后。
        if use_dummy and max_n_s > 0:
            n_b_t = mu_mask.sum(-1).to(dtype)
            n_s_t = nu_mask.sum(-1).to(dtype).clamp(min=1.0)
            dummy_scale = float(self.cfg.scot_dummy_mass_scale)
            dummy_frac = (dummy_scale * n_b_t / n_s_t).clamp(max=0.45)  # [B]
            nu_real = nu_pad * (1.0 - dummy_frac.unsqueeze(-1))
            nu_ext = torch.cat([nu_real, dummy_frac.unsqueeze(-1)], dim=-1)
            nu_mask_ext = torch.cat(
                [nu_mask, torch.ones(B, 1, device=device, dtype=torch.bool)], dim=-1,
            )
            S_ext = F.pad(S_pad, (0, 1), value=0.0)
            at_ext = F.pad(at_pad, (0, 1), value=0)
            type_penalty_mask = torch.ones(B, max_n_b, max_n_s + 1, device=device, dtype=dtype)
            type_penalty_mask[:, :, max_n_s] = 0.0
            if use_topo:
                adj_s_ext = F.pad(adj_s_pad, (0, 1, 0, 1), value=0.0)  # type: ignore[arg-type]
                adj_s_ext[:, max_n_s, max_n_s] = 1.0
            else:
                adj_s_ext = None
            n_s_run = max_n_s + 1
        else:
            S_ext = S_pad
            nu_ext = nu_pad
            at_ext = at_pad
            nu_mask_ext = nu_mask
            type_penalty_mask = None
            adj_s_ext = adj_s_pad
            n_s_run = max_n_s

        cost = -S_ext
        if use_topo:
            M_ot = structure_constrained_ot_batch(
                cost=cost, D_x=adj_b_pad, D_y=adj_s_ext,  # type: ignore[arg-type]
                pattern_probs=pp_pad, ast_types=at_ext.long(),
                mu=mu_pad, nu=nu_ext,
                eps=self.scot_eps, max_iter=self.scot_sinkhorn_iter,
                outer_iter=self.scot_outer_iter,
                lambda_struct=self.scot_lambda_struct,
                gamma_topo=self.scot_gamma_topo,
                num_ast=int(self.vocab_cfg.num_ast_types),
                type_penalty_mask=type_penalty_mask,
                mu_mask=mu_mask, nu_mask=nu_mask_ext,
            )
        else:
            M_ot = sinkhorn_from_cost_batch(
                cost, mu_pad, nu_ext,
                epsilon=self.scot_eps, num_iter=self.scot_sinkhorn_iter,
                mu_mask=mu_mask, nu_mask=nu_mask_ext,
            )

        # 切回每个子问题原本的 [n_b_i, n_s_i]，丢掉 dummy 列与 padding。
        out: List[torch.Tensor] = []
        for i in range(B):
            nb_i = n_b_list[i]
            ns_i = n_s_list[i]
            if nb_i == 0 or ns_i == 0:
                out.append(torch.zeros(nb_i, ns_i, device=device, dtype=dtype))
            else:
                out.append(M_ot[i, :nb_i, :ns_i])
        return out

    # -----------------------------------------------------------------
    def _scot_full(
        self,
        S: torch.Tensor,
        binary_adj: Optional[torch.Tensor],
        source_adj: Optional[torch.Tensor],
        pattern_probs: torch.Tensor,
        ast_types: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        """§6.3：整图 SCOT —— 走 batched 路径（B=1）以复用同一套实现。"""
        n_b, n_s = int(S.size(0)), int(S.size(1))
        if n_b == 0 or n_s == 0:
            return torch.zeros(n_b, n_s, device=S.device, dtype=S.dtype)
        device = S.device
        b_idx = torch.arange(n_b, device=device)
        s_idx = torch.arange(n_s, device=device)
        results = self._run_scot_batch(
            [(b_idx, s_idx)],
            S, binary_adj, source_adj, pattern_probs, ast_types, mu, nu,
        )
        return results[0]

    # -----------------------------------------------------------------
    def _scot_region_aware(
        self,
        S: torch.Tensor,
        binary_adj: Optional[torch.Tensor],
        source_adj: Optional[torch.Tensor],
        aligned_pairs: List[Tuple[int, int]],
        binary_regions: List[Dict],
        source_regions: List[Dict],
        n_b: int,
        n_s: int,
        pattern_probs: torch.Tensor,
        ast_types: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        """§6.3.1：区域内 SCOT + 背景 SCOT，全部子块通过 batched SCOT 一次求解。

        与原版语义完全一致（每个 (region pair, background) 子问题独立做 §6.3.7
        dummy + 独立归一化 + 独立 SCOT），但所有子问题 padded 到统一形状后一
        次 batched call，省去 5~6 次 SCOT 串行 launch。
        """
        device = S.device
        dtype = S.dtype
        M = torch.zeros(n_b, n_s, device=device, dtype=dtype)

        # 在 CPU 上把所有子问题（含 background）的节点 idx flat 拼起来，最后**一次** H2D
        # 拷成两个 long tensor，再按各自长度切片成 ``sub_problems``。原版每对 region
        # 单独 ``torch.tensor(...)`` 触发 H2D，每图 ~12 次小 H2D × 64 图 ≈ 768 次 / step，
        # 是 alignment 内除 SCOT 之外最频繁的 host↔device 通信热点。
        all_b_nodes: List[int] = []
        all_s_nodes: List[int] = []
        b_lens: List[int] = []
        s_lens: List[int] = []
        matched_b_nodes: set = set()
        matched_s_nodes: set = set()

        for bi, si in aligned_pairs:
            b_nodes: List[int] = binary_regions[bi]["nodes"]
            s_nodes: List[int] = source_regions[si]["nodes"]
            if len(b_nodes) == 0 or len(s_nodes) == 0:
                continue
            all_b_nodes.extend(b_nodes)
            all_s_nodes.extend(s_nodes)
            b_lens.append(len(b_nodes))
            s_lens.append(len(s_nodes))
            matched_b_nodes.update(b_nodes)
            matched_s_nodes.update(s_nodes)

        remaining_b = sorted(set(range(n_b)) - matched_b_nodes)
        remaining_s = sorted(set(range(n_s)) - matched_s_nodes)
        if len(remaining_b) > 0 and len(remaining_s) > 0:
            all_b_nodes.extend(remaining_b)
            all_s_nodes.extend(remaining_s)
            b_lens.append(len(remaining_b))
            s_lens.append(len(remaining_s))

        if not b_lens:
            return M

        # 一次 H2D：把全部子问题节点 idx 一并送上 GPU
        all_b_t = torch.tensor(all_b_nodes, dtype=torch.long, device=device)
        all_s_t = torch.tensor(all_s_nodes, dtype=torch.long, device=device)

        sub_problems: List[Tuple[torch.Tensor, torch.Tensor]] = []
        b_off = 0
        s_off = 0
        for nb_i, ns_i in zip(b_lens, s_lens):
            sub_problems.append((
                all_b_t[b_off:b_off + nb_i],
                all_s_t[s_off:s_off + ns_i],
            ))
            b_off += nb_i
            s_off += ns_i

        M_subs = self._run_scot_batch(
            sub_problems,
            S, binary_adj, source_adj, pattern_probs, ast_types, mu, nu,
        )

        for (b_idx, s_idx), M_sub in zip(sub_problems, M_subs):
            if b_idx.numel() == 0 or s_idx.numel() == 0:
                continue
            M[b_idx.unsqueeze(1), s_idx.unsqueeze(0)] = M_sub.to(dtype=dtype)

        return M

    # ─────────────────────────────────────────────────────────────────
    # Cross-graph batched API (D)
    # -----------------------------------------------------------------
    # 把 forward 拆成 ``prepare`` / ``finalize`` 两段，外层 caller（``StrucLift.encode``）
    # 可以在中间插入一次跨图 batched SCOT（``_run_scot_batch_cross``），把 batch 内
    # 所有图的 SCOT 子问题（每图 1 background + N region pair 或 1 整图 SCOT）
    # **一次** padded batched 求解，避免「每图 1 次 alignment forward」
    # 在 64 图 batch 上的 launch overhead 累积。
    #
    # 数学上与原版 ``forward`` 完全等价（padding 由 mu_mask/nu_mask 屏蔽），仅浮点
    # 累加顺序不同导致 ~1e-6 量级差异，不影响训练 trajectory。``forward`` 保留为
    # single-graph fallback（Stage 3 SFT / module_c 路径仍可调用）。
    # ─────────────────────────────────────────────────────────────────

    def prepare(
        self,
        binary_node_embs: torch.Tensor,
        source_node_embs: torch.Tensor,
        patterns: torch.Tensor,
        ast_types: torch.Tensor,
        binary_adj: Optional[torch.Tensor] = None,
        source_adj: Optional[torch.Tensor] = None,
        binary_regions: Optional[List[Dict]] = None,
        source_regions: Optional[List[Dict]] = None,
        M_gt: Optional[torch.Tensor] = None,
        pattern_logits: Optional[torch.Tensor] = None,
        binary_edge_index: Optional[torch.Tensor] = None,
        block_instructions_text: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """单图 prepare：算 ``H_b/H_s/S/mu/nu/region_aligner``，收集 SCOT 子问题。

        把每个 SCOT 子问题需要的张量（``S_sub / adj_b_sub / adj_s_sub / pp_sub /
        at_sub / mu_sub / nu_sub``）**在 GPU 上**切好（无 H2D），打包到
        ``sub_problems`` 里，供后续 :meth:`_run_scot_batch_cross` 跨图收集后一次
        batched 求解；其它非 SCOT 输出（``H_b / H_s / M_R / region_*``）一并放进
        返回字典传给 :meth:`finalize`。

        与原版 ``forward`` 0/1/2a 阶段语义完全一致；2b/2c（SCOT）下放到 caller。
        """
        n_b = int(binary_node_embs.size(0))
        n_s = int(source_node_embs.size(0))
        device = binary_node_embs.device
        dtype = binary_node_embs.dtype

        if pattern_logits is not None:
            pattern_probs = F.softmax(pattern_logits, dim=-1)
        else:
            c = self.vocab_cfg.num_patterns
            pattern_probs = F.one_hot(
                patterns.long().clamp(min=0, max=c - 1), num_classes=c,
            ).to(dtype=dtype, device=device)

        # 0. Project + 相似度
        with _AlignSection("proj_sim", device):
            H_b = self.bin_proj(binary_node_embs)
            H_s = self.src_proj(source_node_embs)
            S = self._cosine_similarity_matrix(H_b, H_s)

        # μ 均匀；ν 取自 M_gt 或 S 列和（与 forward 完全一致）
        with _AlignSection("nu_mu", device):
            mu = torch.ones(n_b, device=device, dtype=dtype) / max(n_b, 1)
            if M_gt is not None and M_gt.numel() > 0:
                nu_raw = M_gt.sum(dim=0).clamp(min=1e-30)
                s_sum = nu_raw.sum()
                fallback = torch.full(
                    (n_s,), 1.0 / max(n_s, 1), device=device, dtype=dtype,
                )
                nu_norm = nu_raw / s_sum.clamp(min=1e-30)
                nu = torch.where(s_sum < 1e-20, fallback, nu_norm)
            elif n_s > 0:
                mode = getattr(self.cfg, "nu_no_gt_mode", "soft_similarity")
                if mode == "uniform":
                    nu = torch.ones(n_s, device=device, dtype=dtype) / max(n_s, 1)
                else:
                    col_mass = S.sum(dim=0).clamp(min=1e-30)
                    nu = col_mass / col_mass.sum().clamp(min=1e-30)
            else:
                nu = torch.zeros(0, device=device, dtype=dtype)

        # 是否走 region-aware
        use_regions = (
            binary_regions is not None
            and source_regions is not None
            and len(binary_regions) > 0
            and len(source_regions) > 0
        )

        M_R: Optional[torch.Tensor] = None
        region_bin_embs: Optional[torch.Tensor] = None
        region_src_embs: Optional[torch.Tensor] = None
        sub_problems: List[Dict] = []

        if use_regions and n_b > 0 and n_s > 0:
            assert binary_regions is not None and source_regions is not None
            # 2a. region-level 软对齐（per-graph，region 数较小，未跨图批化）
            with _AlignSection("region_aligner", device):
                M_R, aligned_pairs, region_bin_embs, region_src_embs = self.region_aligner(
                    H_b, H_s, binary_regions, source_regions,
                )

            # 收集每个 region pair / background 的节点 idx，1 次 H2D 拼成两个 long tensor
            all_b_nodes: List[int] = []
            all_s_nodes: List[int] = []
            b_lens: List[int] = []
            s_lens: List[int] = []
            matched_b: set = set()
            matched_s: set = set()
            for bi, si in aligned_pairs:
                bn = binary_regions[bi]["nodes"]
                sn = source_regions[si]["nodes"]
                if len(bn) == 0 or len(sn) == 0:
                    continue
                all_b_nodes.extend(bn)
                all_s_nodes.extend(sn)
                b_lens.append(len(bn))
                s_lens.append(len(sn))
                matched_b.update(bn)
                matched_s.update(sn)
            rb_list = sorted(set(range(n_b)) - matched_b)
            rs_list = sorted(set(range(n_s)) - matched_s)
            if rb_list and rs_list:
                all_b_nodes.extend(rb_list)
                all_s_nodes.extend(rs_list)
                b_lens.append(len(rb_list))
                s_lens.append(len(rs_list))

            if b_lens:
                all_b_t = torch.tensor(all_b_nodes, dtype=torch.long, device=device)
                all_s_t = torch.tensor(all_s_nodes, dtype=torch.long, device=device)
                b_off = 0
                s_off = 0
                for nb_i, ns_i in zip(b_lens, s_lens):
                    b_idx = all_b_t[b_off:b_off + nb_i]
                    s_idx = all_s_t[s_off:s_off + ns_i]
                    S_sub = S.index_select(0, b_idx).index_select(1, s_idx)
                    b_adj_sub = (
                        binary_adj.index_select(0, b_idx).index_select(1, b_idx)
                        if binary_adj is not None else None
                    )
                    s_adj_sub = (
                        source_adj.index_select(0, s_idx).index_select(1, s_idx)
                        if source_adj is not None else None
                    )
                    pp_sub = pattern_probs.index_select(0, b_idx)
                    at_sub = ast_types.index_select(0, s_idx)
                    mu_sub = mu.index_select(0, b_idx)
                    mu_sub = mu_sub / mu_sub.sum().clamp(min=1e-30)
                    nu_sub = nu.index_select(0, s_idx)
                    nu_sub = nu_sub / nu_sub.sum().clamp(min=1e-30)
                    sub_problems.append({
                        "b_idx": b_idx, "s_idx": s_idx,
                        "S": S_sub, "adj_b": b_adj_sub, "adj_s": s_adj_sub,
                        "pp": pp_sub, "at": at_sub,
                        "mu": mu_sub, "nu": nu_sub,
                    })
                    b_off += nb_i
                    s_off += ns_i
        elif n_b > 0 and n_s > 0:
            # 整图 SCOT —— 等价于一个覆盖全节点的子问题
            b_idx = torch.arange(n_b, device=device)
            s_idx = torch.arange(n_s, device=device)
            sub_problems.append({
                "b_idx": b_idx, "s_idx": s_idx,
                "S": S, "adj_b": binary_adj, "adj_s": source_adj,
                "pp": pattern_probs, "at": ast_types,
                "mu": mu, "nu": nu,
            })

        return {
            "n_b": n_b, "n_s": n_s,
            "device": device, "dtype": dtype,
            "H_b": H_b, "H_s": H_s,
            "M_R": M_R,
            "region_bin_embs": region_bin_embs,
            "region_src_embs": region_src_embs,
            "use_regions": use_regions,
            "binary_regions": binary_regions,
            "source_regions": source_regions,
            "M_gt": M_gt,
            "patterns": patterns,
            "ast_types": ast_types,
            "pattern_probs": pattern_probs,
            "binary_edge_index": binary_edge_index,
            "block_instructions_text": block_instructions_text,
            "sub_problems": sub_problems,
        }

    # -----------------------------------------------------------------
    def _run_scot_batch_cross(
        self,
        sub_problems: List[Dict],
    ) -> List[torch.Tensor]:
        """跨图跨子问题 batched SCOT。

        ``sub_problems`` 是来自任意多图 :meth:`prepare` 输出汇总，每项已经在
        GPU 上切好独立 ``S/adj_b/adj_s/pp/at/mu/nu``（没有再次 indexing 也没有
        H2D）。默认按 ``STRUCLIFT_SCOT_CROSS_CHUNK``（默认 16）**分块** padded
        求解：避免单次 ``B × max_n_b²`` / ``B × max_n_s²`` 邻接张量随全局 batch
        爆炸导致 OOM（整 batch 一次合并时 B 可达数百）。

        分块保持 ``sub_problems`` 的原有顺序，输出列表与输入一一对应。
        """
        if not sub_problems:
            return []
        chunk_sz = int(os.environ.get("STRUCLIFT_SCOT_CROSS_CHUNK", "16"))
        if chunk_sz <= 0:
            chunk_sz = len(sub_problems)
        if len(sub_problems) <= chunk_sz:
            return self._run_scot_batch_cross_chunk(sub_problems)
        out: List[torch.Tensor] = []
        for off in range(0, len(sub_problems), chunk_sz):
            out.extend(
                self._run_scot_batch_cross_chunk(sub_problems[off : off + chunk_sz]),
            )
        return out

    def _run_scot_batch_cross_chunk(
        self,
        sub_problems: List[Dict],
    ) -> List[torch.Tensor]:
        """单次 padded batched SCOT（块大小 ≤ :meth:`_run_scot_batch_cross` 的 chunk）。"""
        B = len(sub_problems)
        if B == 0:
            return []
        device = sub_problems[0]["S"].device
        dtype = sub_problems[0]["S"].dtype
        P = int(sub_problems[0]["pp"].size(1))

        n_b_list = [int(sp["S"].size(0)) for sp in sub_problems]
        n_s_list = [int(sp["S"].size(1)) for sp in sub_problems]
        max_n_b = max(n_b_list) if n_b_list else 0
        max_n_s = max(n_s_list) if n_s_list else 0
        if max_n_b == 0 or max_n_s == 0:
            return [torch.zeros(nb, ns, device=device, dtype=dtype)
                    for nb, ns in zip(n_b_list, n_s_list)]

        use_dummy = bool(self.cfg.scot_use_dummy_sink)
        # 全部子问题要么都有 topo（adj），要么都没有；B 内若不一致则按 "any has not" 退化
        use_topo = all(
            sp.get("adj_b") is not None and sp.get("adj_s") is not None
            for sp in sub_problems
        )

        S_pad = torch.zeros(B, max_n_b, max_n_s, device=device, dtype=dtype)
        pp_pad = torch.zeros(B, max_n_b, P, device=device, dtype=dtype)
        at_dtype = sub_problems[0]["at"].dtype
        at_pad = torch.zeros(B, max_n_s, device=device, dtype=at_dtype)
        mu_pad = torch.zeros(B, max_n_b, device=device, dtype=dtype)
        nu_pad = torch.zeros(B, max_n_s, device=device, dtype=dtype)
        mu_mask = torch.zeros(B, max_n_b, device=device, dtype=torch.bool)
        nu_mask = torch.zeros(B, max_n_s, device=device, dtype=torch.bool)
        if use_topo:
            adj_b_pad = torch.zeros(
                B, max_n_b, max_n_b, device=device,
                dtype=sub_problems[0]["adj_b"].dtype,
            )
            adj_s_pad = torch.zeros(
                B, max_n_s, max_n_s, device=device,
                dtype=sub_problems[0]["adj_s"].dtype,
            )
        else:
            adj_b_pad = None
            adj_s_pad = None

        # Python 循环 B 次填充（此处 B ≤ chunk_sz，典型 16）。
        for i, sp in enumerate(sub_problems):
            nb_i = n_b_list[i]
            ns_i = n_s_list[i]
            if nb_i == 0 or ns_i == 0:
                continue
            S_pad[i, :nb_i, :ns_i] = sp["S"]
            pp_pad[i, :nb_i, :] = sp["pp"]
            at_pad[i, :ns_i] = sp["at"]
            mu_pad[i, :nb_i] = sp["mu"]
            nu_pad[i, :ns_i] = sp["nu"]
            mu_mask[i, :nb_i] = True
            nu_mask[i, :ns_i] = True
            if use_topo:
                adj_b_pad[i, :nb_i, :nb_i] = sp["adj_b"]
                adj_s_pad[i, :ns_i, :ns_i] = sp["adj_s"]

        # §6.3.7 dummy 列处理（每子问题独立 dummy_frac，跟 _run_scot_batch 一致）
        if use_dummy and max_n_s > 0:
            n_b_t = mu_mask.sum(-1).to(dtype)
            n_s_t = nu_mask.sum(-1).to(dtype).clamp(min=1.0)
            dummy_scale = float(self.cfg.scot_dummy_mass_scale)
            dummy_frac = (dummy_scale * n_b_t / n_s_t).clamp(max=0.45)
            nu_real = nu_pad * (1.0 - dummy_frac.unsqueeze(-1))
            nu_ext = torch.cat([nu_real, dummy_frac.unsqueeze(-1)], dim=-1)
            nu_mask_ext = torch.cat(
                [nu_mask, torch.ones(B, 1, device=device, dtype=torch.bool)], dim=-1,
            )
            S_ext = F.pad(S_pad, (0, 1), value=0.0)
            at_ext = F.pad(at_pad, (0, 1), value=0)
            type_penalty_mask = torch.ones(B, max_n_b, max_n_s + 1, device=device, dtype=dtype)
            type_penalty_mask[:, :, max_n_s] = 0.0
            if use_topo:
                adj_s_ext = F.pad(adj_s_pad, (0, 1, 0, 1), value=0.0)
                adj_s_ext[:, max_n_s, max_n_s] = 1.0
            else:
                adj_s_ext = None
        else:
            S_ext = S_pad
            nu_ext = nu_pad
            at_ext = at_pad
            nu_mask_ext = nu_mask
            type_penalty_mask = None
            adj_s_ext = adj_s_pad

        cost = -S_ext
        if use_topo:
            M_ot = structure_constrained_ot_batch(
                cost=cost, D_x=adj_b_pad, D_y=adj_s_ext,
                pattern_probs=pp_pad, ast_types=at_ext.long(),
                mu=mu_pad, nu=nu_ext,
                eps=self.scot_eps, max_iter=self.scot_sinkhorn_iter,
                outer_iter=self.scot_outer_iter,
                lambda_struct=self.scot_lambda_struct,
                gamma_topo=self.scot_gamma_topo,
                num_ast=int(self.vocab_cfg.num_ast_types),
                type_penalty_mask=type_penalty_mask,
                mu_mask=mu_mask, nu_mask=nu_mask_ext,
            )
        else:
            M_ot = sinkhorn_from_cost_batch(
                cost, mu_pad, nu_ext,
                epsilon=self.scot_eps, num_iter=self.scot_sinkhorn_iter,
                mu_mask=mu_mask, nu_mask=nu_mask_ext,
            )

        out: List[torch.Tensor] = []
        for i in range(B):
            nb_i = n_b_list[i]
            ns_i = n_s_list[i]
            if nb_i == 0 or ns_i == 0:
                out.append(torch.zeros(nb_i, ns_i, device=device, dtype=dtype))
            else:
                out.append(M_ot[i, :nb_i, :ns_i])
        return out

    # -----------------------------------------------------------------
    def finalize(
        self,
        prepared: Dict,
        M_subs: List[torch.Tensor],
    ) -> ModuleBOutput:
        """单图 finalize：scatter ``M_subs`` 回 ``M``，然后 cross_attn / region_pos_mask /
        n_s 回归 / skeleton。与原版 ``forward`` 第 3 阶段及之后语义一致。
        """
        n_b = prepared["n_b"]
        n_s = prepared["n_s"]
        device = prepared["device"]
        dtype = prepared["dtype"]
        H_b: torch.Tensor = prepared["H_b"]
        H_s: torch.Tensor = prepared["H_s"]
        patterns: torch.Tensor = prepared["patterns"]
        ast_types: torch.Tensor = prepared["ast_types"]

        M = torch.zeros(n_b, n_s, device=device, dtype=dtype)
        for sp, M_sub in zip(prepared["sub_problems"], M_subs):
            b_idx = sp["b_idx"]
            s_idx = sp["s_idx"]
            if b_idx.numel() == 0 or s_idx.numel() == 0:
                continue
            M[b_idx.unsqueeze(1), s_idx.unsqueeze(0)] = M_sub.to(dtype=dtype)

        # region positive mask（与原 forward 一致）
        region_pos_mask: Optional[torch.Tensor] = None
        if prepared["use_regions"]:
            br = prepared["binary_regions"]
            sr = prepared["source_regions"]
            assert br is not None and sr is not None
            with _AlignSection("region_pos_mask", device):
                if prepared["M_gt"] is not None:
                    region_pos_mask = build_region_positive_mask(
                        prepared["M_gt"], br, sr,
                    )
                elif prepared["M_R"] is not None:
                    region_pos_mask = (prepared["M_R"] > 0.05).float()

        with _AlignSection("cross_attn", device):
            H_hat_b, H_hat_s, A_cross = self.cross_attn(
                H_b=H_b, H_s=H_s, M=M,
                patterns=patterns, ast_types=ast_types,
            )

        pred_n_s: Optional[torch.Tensor] = None
        if self.cfg.src_count_regressor and hasattr(self, "n_s_regressor"):
            graph_emb = H_b.mean(dim=0, keepdim=True)
            pred_n_s = self.n_s_regressor(graph_emb).squeeze()

        skel: Optional[Skeleton] = None
        if self.cfg.emit_skeleton and n_b > 0:
            be = prepared["binary_edge_index"]
            if be is not None and be.numel() > 0:
                ei = be
            else:
                ei = torch.zeros(2, 0, dtype=torch.long, device=patterns.device)
            prefix_hints, skip_bbs = _skeleton_hints_from_alignment(M.detach(), patterns, ast_types)
            skel = build_skeleton_for_binary_cfg(
                patterns.long(), ei, n_b,
                prepared["block_instructions_text"],
                pattern_probs=prepared.get("pattern_probs"),
                bb_order_key=_bb_source_order_keys_from_alignment(M.detach()),
                alignment_matrix=M.detach(),
                predicted_source_count=(
                    float(pred_n_s.detach().cpu().item())
                    if isinstance(pred_n_s, torch.Tensor) and pred_n_s.numel() == 1
                    else None
                ),
                header_prefix_slots=prefix_hints,
                skip_statement_bbs=skip_bbs,
            )
            enrich_skeleton_slots_with_alignment_matrix(skel, M.detach())

        return ModuleBOutput(
            alignment_matrix=M,
            binary_node_embs=H_hat_b,
            source_node_embs=H_hat_s,
            region_alignment=prepared["M_R"],
            cross_attn_weights=A_cross,
            region_bin_embs=prepared["region_bin_embs"],
            region_src_embs=prepared["region_src_embs"],
            region_positive_mask=region_pos_mask,
            skeleton=skel,
            predicted_n_s=pred_n_s,
        )

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        binary_node_embs: torch.Tensor,
        source_node_embs: torch.Tensor,
        patterns: torch.Tensor,
        ast_types: torch.Tensor,
        binary_adj: Optional[torch.Tensor] = None,
        source_adj: Optional[torch.Tensor] = None,
        binary_regions: Optional[List[Dict]] = None,
        source_regions: Optional[List[Dict]] = None,
        M_gt: Optional[torch.Tensor] = None,
        pattern_logits: Optional[torch.Tensor] = None,
        binary_edge_index: Optional[torch.Tensor] = None,
        block_instructions_text: Optional[Dict[int, str]] = None,
    ) -> ModuleBOutput:
        """Run the full Module B pipeline.

        Parameters
        ----------
        binary_node_embs : Tensor  [n_b, d]
            Node embeddings from Module A's binary CFG encoder.
        source_node_embs : Tensor  [n_s, d]
            Node embeddings from Module A's source CFG encoder.
        patterns : Tensor  [n_b]  int64
            Structural pattern label (argmax) for cross-attention embeddings.
        pattern_logits : Tensor, optional  [n_b, C]
            Module A pattern logits; softmax gives :math:`p^{\\text{pat}}` for §6.3.3.
        ast_types : Tensor  [n_s]  int64
            AST statement type for each source node.
        binary_adj : Tensor, optional  [n_b, n_b]
            Shortest-path distance (or adjacency) matrix for binary CFG.
            When provided, enables SCOT's structural penalty.
        source_adj : Tensor, optional  [n_s, n_s]
            Same for source CFG.
        binary_regions : list of dict, optional
            Region segmentation for the binary CFG.  Each element has keys
            ``'nodes'`` (list[int]) and ``'type'`` (int, a PatternType value).
            When *None* or empty, flat (region-free) SCOT is used.
        source_regions : list of dict, optional
            Same schema for the source CFG (type values are ASTType ints).
        M_gt : Tensor, optional  [n_b, n_s]
            Ground-truth alignment for building region positive mask
            (Stage 2 region InfoNCE).  When *None*, mask falls back to
            thresholding ``M_R`` if available.
        binary_edge_index : Tensor, optional  [2, E]
            当前函数内 **局部** BB 下标的 CFG 边；提供且 ``emit_skeleton`` 时生成 §6.5 骨架。
        block_instructions_text : dict, optional
            ``bb_idx ->`` 反汇编摘要；缺省为 ``BB_i`` / 由调用方填入 opcode 摘要。

        Returns
        -------
        ModuleBOutput
            含 ``alignment_matrix``、精炼节点向量、可选 ``region_alignment``、
            以及 §6.5 ``skeleton``（边为空或未启用时为 *None*）。
        """
        n_b = binary_node_embs.size(0)
        n_s = source_node_embs.size(0)
        device = binary_node_embs.device
        dtype = binary_node_embs.dtype

        if pattern_logits is not None:
            pattern_probs = F.softmax(pattern_logits, dim=-1)
        else:
            c = self.vocab_cfg.num_patterns
            pattern_probs = F.one_hot(
                patterns.long().clamp(min=0, max=c - 1), num_classes=c,
            ).to(dtype=dtype, device=device)

        # 0. Project + 相似度（§6.3.7 ν 无 M_gt 时需先算 S）-------------------
        with _AlignSection("proj_sim", device):
            H_b = self.bin_proj(binary_node_embs)
            H_s = self.src_proj(source_node_embs)
            S = self._cosine_similarity_matrix(H_b, H_s)  # [n_b, n_s]

        # §6.3.7: μ 均匀；ν_j — 有 M_gt 时为 DWARF/GT 对齐列和；无 M_gt 时用相似度
        # 列和近似「每语句对应 BB 规模」。原版 ``float(s.item()) < 1e-20`` 的 fallback
        # 分支会触发一次 GPU sync —— 由于上一行已 ``clamp(min=1e-30)``，s 至少为
        # ``n_s * 1e-30``（n_s>=1 时 >= 1e-30），换 ``torch.where`` 走纯 GPU 路径，
        # 数值与原版完全一致（fallback 触发条件极罕见，且分支输出仍为均匀分布）。
        with _AlignSection("nu_mu", device):
            mu = torch.ones(n_b, device=device, dtype=dtype) / max(n_b, 1)
            if M_gt is not None and M_gt.numel() > 0:
                nu_raw = M_gt.sum(dim=0).clamp(min=1e-30)
                s = nu_raw.sum()
                fallback = torch.full(
                    (n_s,), 1.0 / max(n_s, 1), device=device, dtype=dtype,
                )
                nu_norm = nu_raw / s.clamp(min=1e-30)
                nu = torch.where(s < 1e-20, fallback, nu_norm)
            elif n_s > 0:
                mode = getattr(self.cfg, "nu_no_gt_mode", "soft_similarity")
                if mode == "uniform":
                    nu = torch.ones(n_s, device=device, dtype=dtype) / max(n_s, 1)
                else:
                    col_mass = S.sum(dim=0).clamp(min=1e-30)
                    nu = col_mass / col_mass.sum().clamp(min=1e-30)
            else:
                nu = torch.zeros(0, device=device, dtype=dtype)

        # 2. Alignment matrix M  -------------------------------------------
        M_R: Optional[torch.Tensor] = None
        region_bin_embs: Optional[torch.Tensor] = None
        region_src_embs: Optional[torch.Tensor] = None
        use_regions = (
            binary_regions is not None
            and source_regions is not None
            and len(binary_regions) > 0
            and len(source_regions) > 0
        )

        if use_regions:
            assert binary_regions is not None  # for type checker
            assert source_regions is not None

            # 2a. Region-level alignment -----------------------------------
            with _AlignSection("region_aligner", device):
                M_R, aligned_pairs, region_bin_embs, region_src_embs = self.region_aligner(
                    H_b, H_s,
                    binary_regions, source_regions,
                )

            # 2b. Within-region SCOT + background SCOT ---------------------
            with _AlignSection("scot_region", device):
                M = self._scot_region_aware(
                    S=S,
                    binary_adj=binary_adj,
                    source_adj=source_adj,
                    aligned_pairs=aligned_pairs,
                    binary_regions=binary_regions,
                    source_regions=source_regions,
                    n_b=n_b,
                    n_s=n_s,
                    pattern_probs=pattern_probs,
                    ast_types=ast_types,
                    mu=mu,
                    nu=nu,
                )
        else:
            # 2c. Flat SCOT on the entire function -------------------------
            with _AlignSection("scot_full", device):
                M = self._scot_full(
                    S, binary_adj, source_adj,
                    pattern_probs, ast_types, mu, nu,
                )

        # Region vectors + positive mask for L_region (doc §6.2 / §6.6)
        # 直接复用 region_aligner 已经 pool 出来的 vecs（与原版 ``_pool_regions(H_b/H_s, ...)``
        # 输入完全一致），节省一组 ragged attentive-pool 的 H2D + kernel。
        region_pos_mask: Optional[torch.Tensor] = None
        if use_regions:
            assert binary_regions is not None and source_regions is not None
            with _AlignSection("region_pos_mask", device):
                if M_gt is not None:
                    region_pos_mask = build_region_positive_mask(
                        M_gt, binary_regions, source_regions,
                    )
                elif M_R is not None:
                    region_pos_mask = (M_R > 0.05).float()

        # 3. Cross-attention refinement ------------------------------------
        with _AlignSection("cross_attn", device):
            H_hat_b, H_hat_s, A_cross = self.cross_attn(
                H_b=H_b,
                H_s=H_s,
                M=M,
                patterns=patterns,
                ast_types=ast_types,
            )

        # §6.5.3 回归头预测 n_s（用于推理模式）
        pred_n_s: Optional[torch.Tensor] = None
        if self.cfg.src_count_regressor and hasattr(self, "n_s_regressor"):
            graph_emb = H_b.mean(dim=0, keepdim=True)  # [1, d]
            pred_n_s = self.n_s_regressor(graph_emb).squeeze()  # scalar

        # §6.5 骨架生成
        skel: Optional[Skeleton] = None
        if self.cfg.emit_skeleton and n_b > 0:
            if binary_edge_index is not None and binary_edge_index.numel() > 0:
                ei = binary_edge_index
            else:
                ei = torch.zeros(2, 0, dtype=torch.long, device=patterns.device)
            prefix_hints, skip_bbs = _skeleton_hints_from_alignment(M.detach(), patterns, ast_types)
            skel = build_skeleton_for_binary_cfg(
                patterns.long(),
                ei,
                n_b,
                block_instructions_text,
                pattern_probs=pattern_probs,
                bb_order_key=_bb_source_order_keys_from_alignment(M.detach()),
                alignment_matrix=M.detach(),
                predicted_source_count=(
                    float(pred_n_s.detach().cpu().item())
                    if isinstance(pred_n_s, torch.Tensor) and pred_n_s.numel() == 1
                    else None
                ),
                header_prefix_slots=prefix_hints,
                skip_statement_bbs=skip_bbs,
            )
            enrich_skeleton_slots_with_alignment_matrix(skel, M.detach())

        return ModuleBOutput(
            alignment_matrix=M,          # [n_b, n_s]
            binary_node_embs=H_hat_b,    # [n_b, d]
            source_node_embs=H_hat_s,    # [n_s, d]
            region_alignment=M_R,        # [K_b, K_s] or None
            cross_attn_weights=A_cross,
            region_bin_embs=region_bin_embs,
            region_src_embs=region_src_embs,
            region_positive_mask=region_pos_mask,
            skeleton=skel,
            predicted_n_s=pred_n_s,
        )

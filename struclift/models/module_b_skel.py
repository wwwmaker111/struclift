"""Optimised binary-only CST/slot predictor for the repaired Module B path.

This module implements the first practical version of the proposed
``CFG-aware Neural CST Parser``:

1. CFG Structural Transformer with additive structural attention bias.
2. Control region proposal: header BB -> region membership matrix.
3. Biaffine parent parser with Matrix-Tree-Theorem loss.
4. Role and slot heads.
5. Same-slot pair grouping as a lightweight, trainable precursor to
   semi-Markov slot CRF / Sinkhorn BB-to-slot assignment.

The module consumes only inference-time binary-side information:

    H_b, P_pat, binary CFG edges, edge types, and structural features.

Source AST / M_gt are used only by teacher export scripts to create targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


CTRL_NONE = 0
CTRL_IF = 1
CTRL_LOOP = 2
CTRL_SWITCH = 3
NUM_CTRL_TYPES = 4
OUTLINE_DEPTH_IGNORE = -100

ROLE_IGNORE = -100
ROLE_NONE = 0
ROLE_COND = 1
ROLE_THEN = 2
ROLE_ELSE = 3
ROLE_LOOP_BODY = 4
ROLE_SWITCH_BODY = 5
ROLE_STMT = 6
ROLE_BODY = 7
NUM_ROLE_TYPES = 8

SLOT_IGNORE = -100
SLOT_NONE = 0
SLOT_COND = 1
SLOT_BLOCK = 2
SLOT_BRANCH = 3
NUM_SLOT_TYPES = 4

PARENT_IGNORE = -100

CTRL_NAME_TO_ID = {
    "none": CTRL_NONE,
    "if": CTRL_IF,
    "loop": CTRL_LOOP,
    "switch": CTRL_SWITCH,
}
CTRL_ID_TO_NAME = {v: k for k, v in CTRL_NAME_TO_ID.items()}

ROLE_NAME_TO_ID = {
    "none": ROLE_NONE,
    "cond": ROLE_COND,
    "then": ROLE_THEN,
    "else": ROLE_ELSE,
    "loop_body": ROLE_LOOP_BODY,
    "switch_body": ROLE_SWITCH_BODY,
    "stmt": ROLE_STMT,
    "body": ROLE_BODY,
}
ROLE_ID_TO_NAME = {v: k for k, v in ROLE_NAME_TO_ID.items()}

SLOT_NAME_TO_ID = {
    "none": SLOT_NONE,
    "FILL_COND": SLOT_COND,
    "FILL_BLOCK": SLOT_BLOCK,
    "FILL_BRANCH": SLOT_BRANCH,
}
SLOT_ID_TO_NAME = {v: k for k, v in SLOT_NAME_TO_ID.items()}


@dataclass
class BinaryCSTParserOutput:
    node_states: Tensor
    ctrl_logits: Tensor
    role_logits: Tensor
    role_pair_logits: Tensor
    root_role_logits: Tensor
    slot_logits: Tensor
    parent_scores: Tensor
    root_scores: Tensor
    region_logits: Tensor
    same_slot_logits: Tensor
    slot_anchor_scores: Tensor
    outline_count_logits: Tensor
    outline_kind_logits: Tensor
    outline_depth_logits: Tensor
    slot_layout_count_logits: Tensor
    slot_layout_kind_logits: Tensor
    slot_layout_depth_logits: Tensor
    slot_primary_logits: Tensor
    slot_context_logits: Tensor
    structural_bias: Tensor


def _safe_edge_index(edge_index: Tensor, edge_type: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    if edge_index.numel() == 0 or n <= 0:
        dev = edge_index.device
        return torch.zeros(2, 0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    mask = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    src = src[mask]
    dst = dst[mask]
    et = edge_type.long()
    et = et[mask] if et.numel() == mask.numel() else torch.zeros_like(src)
    return torch.stack([src, dst], dim=0), et


def _transitive_closure(adj: Tensor) -> Tensor:
    """Boolean reachability with small-N Floyd-Warshall style updates."""
    reach = adj.bool().clone()
    n = int(reach.size(0))
    eye = torch.eye(n, dtype=torch.bool, device=reach.device)
    reach = reach | eye
    for k in range(n):
        reach = reach | (reach[:, k : k + 1] & reach[k : k + 1, :])
    return reach


def _all_pairs_shortest(adj: Tensor, max_dist: int = 8) -> Tensor:
    n = int(adj.size(0))
    dist = torch.full((n, n), int(max_dist), dtype=torch.long, device=adj.device)
    dist.fill_diagonal_(0)
    dist[adj.bool()] = 1
    for k in range(n):
        cand = dist[:, k : k + 1] + dist[k : k + 1, :]
        dist = torch.minimum(dist, cand)
    return dist.clamp(max=max_dist)


def _simple_dominators(adj: Tensor, entry: int = 0) -> Tensor:
    """Return bool dom[a,b] meaning a dominates b.

    This is CPU-like iterative logic in tensors.  It is used only for structural
    attention bias and smoke-scale training, not as a differentiable operation.
    """
    n = int(adj.size(0))
    if n <= 0:
        return adj.new_zeros(0, 0, dtype=torch.bool)
    preds = [torch.where(adj[:, j].bool())[0].tolist() for j in range(n)]
    all_nodes = set(range(n))
    dom: List[set[int]] = []
    for i in range(n):
        dom.append({i} if i == entry else set(all_nodes))
    changed = True
    while changed:
        changed = False
        for b in range(n):
            if b == entry:
                continue
            if preds[b]:
                new = set(all_nodes)
                for p in preds[b]:
                    new &= dom[int(p)]
                new.add(b)
            else:
                new = {b}
            if new != dom[b]:
                dom[b] = new
                changed = True
    out = torch.zeros(n, n, dtype=torch.bool, device=adj.device)
    for b in range(n):
        for a in dom[b]:
            out[int(a), b] = True
    return out


def build_cfg_structural_features(
    edge_index: Tensor,
    edge_type: Tensor,
    n: int,
    *,
    max_dist: int = 8,
    bin_batch: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Build non-learned CFG relations for structural attention bias."""
    device = edge_index.device
    edge_index, edge_type = _safe_edge_index(edge_index, edge_type, n)
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True
    undirected = adj | adj.t()
    reach = _transitive_closure(adj) if n > 0 else adj
    dist = _all_pairs_shortest(undirected, max_dist=max_dist) if n > 0 else torch.zeros(0, 0, dtype=torch.long, device=device)
    dom = _simple_dominators(adj, entry=0) if n > 0 else adj
    # postdom via reversed graph from synthetic exit approximation.
    exits = torch.where(adj.sum(dim=1) == 0)[0]
    if exits.numel() == 1:
        postdom = _simple_dominators(adj.t(), entry=int(exits.item()))
    else:
        postdom = torch.zeros_like(dom)
        for e in exits.tolist():
            postdom = postdom | _simple_dominators(adj.t(), entry=int(e))
    back = torch.zeros_like(adj)
    if edge_index.numel() > 0:
        src, dst = edge_index
        back[src, dst] = dst <= src
        for k in torch.where(edge_type == 2)[0].tolist():
            back[int(src[k]), int(dst[k])] = True
    if bin_batch is not None and bin_batch.numel() == n:
        gb = bin_batch.to(device=device, dtype=torch.long)
        same_graph = gb.view(-1, 1) == gb.view(1, -1)
    else:
        same_graph = torch.ones(n, n, dtype=torch.bool, device=device)
    return {
        "adj": adj,
        "undirected": undirected,
        "reach": reach,
        "dist": dist,
        "dom": dom,
        "postdom": postdom,
        "back": back,
        "same_graph": same_graph,
    }


class CFGStructuralTransformerLayer(nn.Module):
    """Multi-head self-attention with additive CFG structural bias."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_dist: int = 8,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.max_dist = int(max_dist)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dist_bias = nn.Embedding(self.max_dist + 1, n_heads)
        self.edge_bias = nn.Parameter(torch.zeros(n_heads))
        self.dom_bias = nn.Parameter(torch.zeros(n_heads))
        self.postdom_bias = nn.Parameter(torch.zeros(n_heads))
        self.back_bias = nn.Parameter(torch.zeros(n_heads))
        self.reach_bias = nn.Parameter(torch.zeros(n_heads))
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, feats: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        n = int(x.size(0))
        if n == 0:
            return x, x.new_zeros(self.n_heads, 0, 0)
        q = self.q(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head ** 0.5)

        bias = self.dist_bias(feats["dist"].clamp(max=self.max_dist)).permute(2, 0, 1)
        bias = bias + feats["adj"].float().unsqueeze(0) * self.edge_bias.view(-1, 1, 1)
        bias = bias + feats["dom"].float().unsqueeze(0) * self.dom_bias.view(-1, 1, 1)
        bias = bias + feats["postdom"].float().unsqueeze(0) * self.postdom_bias.view(-1, 1, 1)
        bias = bias + feats["back"].float().unsqueeze(0) * self.back_bias.view(-1, 1, 1)
        bias = bias + feats["reach"].float().unsqueeze(0) * self.reach_bias.view(-1, 1, 1)

        masked_logits = logits + bias
        same_graph = feats.get("same_graph")
        if same_graph is not None:
            masked_logits = masked_logits.masked_fill(~same_graph.bool().unsqueeze(0), -1e4)
        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(self.dropout(attn), v).transpose(0, 1).reshape(n, self.d_model)
        x = self.ln1(x + self.dropout(self.out(out)))
        x = self.ln2(x + self.ff(x))
        return x, bias


class BinaryCSTParser(nn.Module):
    """Optimised binary-only CST parser."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        num_patterns: int = 10,
        struct_feat_dim: int = 12,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_dist: int = 8,
        max_outline_controls: int = 12,
        max_outline_depth: int = 8,
        max_slot_layout_items: int = 45,
        max_slot_layout_depth: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_patterns = int(num_patterns)
        self.struct_feat_dim = int(struct_feat_dim)
        self.max_dist = int(max_dist)
        self.max_outline_controls = int(max_outline_controls)
        self.max_outline_depth = int(max_outline_depth)
        self.max_slot_layout_items = int(max_slot_layout_items)
        self.max_slot_layout_depth = int(max_slot_layout_depth)
        in_dim = self.d_model + self.num_patterns + self.struct_feat_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                CFGStructuralTransformerLayer(
                    self.d_model,
                    n_heads=num_heads,
                    dropout=dropout,
                    max_dist=max_dist,
                )
                for _ in range(num_layers)
            ],
        )

        self.ctrl_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, NUM_CTRL_TYPES),
        )
        self.role_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, NUM_ROLE_TYPES),
        )
        self.role_pair = nn.Sequential(
            nn.Linear(self.d_model * 2 + 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, NUM_ROLE_TYPES),
        )
        self.root_role_head = nn.Linear(self.d_model, NUM_ROLE_TYPES)
        self.slot_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, NUM_SLOT_TYPES),
        )
        self.region_pair = nn.Sequential(
            nn.Linear(self.d_model * 2 + 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.slot_anchor_pair = nn.Sequential(
            nn.Linear(self.d_model * 2 + 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.same_slot_pair = nn.Sequential(
            nn.Linear(self.d_model * 2 + 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

        self.parent_child = nn.Linear(self.d_model, self.d_model, bias=False)
        self.parent_parent = nn.Linear(self.d_model, self.d_model, bias=False)
        self.parent_pair = nn.Sequential(
            nn.Linear(self.d_model * 2 + 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.root_head = nn.Linear(self.d_model, 1)
        self.outline_pos = nn.Embedding(self.max_outline_controls, self.d_model)
        self.outline_query = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.outline_attn_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.outline_attn_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.outline_attn_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.outline_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        self.outline_seq = nn.GRU(
            self.d_model,
            self.d_model,
            num_layers=2,
            batch_first=True,
            dropout=float(dropout),
        )
        self.outline_seq_norm = nn.LayerNorm(self.d_model)
        self.outline_kind_head = nn.Linear(self.d_model, NUM_CTRL_TYPES)
        self.outline_depth_head = nn.Linear(self.d_model, self.max_outline_depth)
        self.outline_count_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.max_outline_controls + 1),
        )
        self.slot_layout_pos = nn.Embedding(self.max_slot_layout_items, self.d_model)
        self.slot_layout_query = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.slot_layout_attn_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.slot_layout_attn_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.slot_layout_attn_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.slot_layout_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        self.slot_layout_seq = nn.GRU(
            self.d_model,
            self.d_model,
            num_layers=2,
            batch_first=True,
            dropout=float(dropout),
        )
        self.slot_layout_seq_norm = nn.LayerNorm(self.d_model)
        self.slot_layout_kind_head = nn.Linear(self.d_model, NUM_SLOT_TYPES)
        self.slot_layout_depth_head = nn.Linear(self.d_model, self.max_slot_layout_depth)
        self.slot_layout_count_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.max_slot_layout_items + 1),
        )
        self.slot_primary_head = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.slot_context_head = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    def _outline_logits(self, z: Tensor, bin_batch: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        n = int(z.size(0))
        if n == 0:
            return (
                z.new_zeros(0, self.max_outline_controls + 1),
                z.new_zeros(0, self.max_outline_controls, NUM_CTRL_TYPES),
                z.new_zeros(0, self.max_outline_controls, self.max_outline_depth),
            )
        if bin_batch is None or bin_batch.numel() != n:
            bin_batch = torch.zeros(n, dtype=torch.long, device=z.device)
        else:
            bin_batch = bin_batch.to(device=z.device, dtype=torch.long)
        n_graphs = int(bin_batch.max().item()) + 1 if bin_batch.numel() else 0
        pos = torch.arange(self.max_outline_controls, device=z.device, dtype=torch.long)
        pos_emb = self.outline_pos(pos)
        count_rows: List[Tensor] = []
        kind_rows: List[Tensor] = []
        depth_rows: List[Tensor] = []
        for gi in range(n_graphs):
            nodes = (bin_batch == gi).nonzero(as_tuple=True)[0]
            if nodes.numel() == 0:
                graph = z.new_zeros(1, self.d_model)
                h = z.new_zeros(self.max_outline_controls, self.d_model)
            else:
                zn = z[nodes]
                graph = zn.mean(dim=0, keepdim=True)
                q0 = self.outline_query(graph + pos_emb)
                q = self.outline_attn_q(q0)
                k = self.outline_attn_k(zn)
                v = self.outline_attn_v(zn)
                attn = torch.softmax((q @ k.t()) / (self.d_model ** 0.5), dim=-1)
                ctx = attn @ v
                h = self.outline_fuse(torch.cat([q0, ctx], dim=-1))
                h_seq, _ = self.outline_seq(h.unsqueeze(0))
                h = self.outline_seq_norm(h + h_seq.squeeze(0))
            count_rows.append(self.outline_count_head(graph).squeeze(0))
            kind_rows.append(self.outline_kind_head(h))
            depth_rows.append(self.outline_depth_head(h))
        return torch.stack(count_rows, dim=0), torch.stack(kind_rows, dim=0), torch.stack(depth_rows, dim=0)

    def _slot_layout_logits(self, z: Tensor, bin_batch: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        n = int(z.size(0))
        if n == 0:
            return (
                z.new_zeros(0, self.max_slot_layout_items + 1),
                z.new_zeros(0, self.max_slot_layout_items, NUM_SLOT_TYPES),
                z.new_zeros(0, self.max_slot_layout_items, self.max_slot_layout_depth),
                z.new_zeros(0, self.max_slot_layout_items, self.d_model),
                z.new_zeros(0, self.max_slot_layout_items, 0),
                z.new_zeros(0, self.max_slot_layout_items, 0),
            )
        if bin_batch is None or bin_batch.numel() != n:
            bin_batch = torch.zeros(n, dtype=torch.long, device=z.device)
        else:
            bin_batch = bin_batch.to(device=z.device, dtype=torch.long)
        n_graphs = int(bin_batch.max().item()) + 1 if bin_batch.numel() else 0
        pos = torch.arange(self.max_slot_layout_items, device=z.device, dtype=torch.long)
        pos_emb = self.slot_layout_pos(pos)
        count_rows: List[Tensor] = []
        kind_rows: List[Tensor] = []
        depth_rows: List[Tensor] = []
        state_rows: List[Tensor] = []
        primary_rows: List[Tensor] = []
        context_rows: List[Tensor] = []
        for gi in range(n_graphs):
            nodes = (bin_batch == gi).nonzero(as_tuple=True)[0]
            if nodes.numel() == 0:
                graph = z.new_zeros(1, self.d_model)
                h = z.new_zeros(self.max_slot_layout_items, self.d_model)
                primary = z.new_zeros(self.max_slot_layout_items, n)
                context = z.new_zeros(self.max_slot_layout_items, n)
            else:
                zn = z[nodes]
                graph = zn.mean(dim=0, keepdim=True)
                q0 = self.slot_layout_query(graph + pos_emb)
                q = self.slot_layout_attn_q(q0)
                k = self.slot_layout_attn_k(zn)
                v = self.slot_layout_attn_v(zn)
                attn = torch.softmax((q @ k.t()) / (self.d_model ** 0.5), dim=-1)
                ctx = attn @ v
                h = self.slot_layout_fuse(torch.cat([q0, ctx], dim=-1))
                h_seq, _ = self.slot_layout_seq(h.unsqueeze(0))
                h = self.slot_layout_seq_norm(h + h_seq.squeeze(0))
                pair = torch.cat(
                    [
                        h.unsqueeze(1).expand(self.max_slot_layout_items, int(nodes.numel()), self.d_model),
                        zn.unsqueeze(0).expand(self.max_slot_layout_items, int(nodes.numel()), self.d_model),
                        (h.unsqueeze(1) * zn.unsqueeze(0)).expand(
                            self.max_slot_layout_items,
                            int(nodes.numel()),
                            self.d_model,
                        ),
                    ],
                    dim=-1,
                )
                local_primary = self.slot_primary_head(pair).squeeze(-1)
                local_context = self.slot_context_head(pair).squeeze(-1)
                primary = z.new_full((self.max_slot_layout_items, n), -1e4)
                context = z.new_full((self.max_slot_layout_items, n), -1e4)
                primary[:, nodes] = local_primary
                context[:, nodes] = local_context
            count_rows.append(self.slot_layout_count_head(graph).squeeze(0))
            kind_rows.append(self.slot_layout_kind_head(h))
            depth_rows.append(self.slot_layout_depth_head(h))
            state_rows.append(h)
            primary_rows.append(primary)
            context_rows.append(context)
        return (
            torch.stack(count_rows, dim=0),
            torch.stack(kind_rows, dim=0),
            torch.stack(depth_rows, dim=0),
            torch.stack(state_rows, dim=0),
            torch.stack(primary_rows, dim=0),
            torch.stack(context_rows, dim=0),
        )

    def _pair_features(self, z: Tensor, feats: Dict[str, Tensor]) -> Tensor:
        n = int(z.size(0))
        dist = feats["dist"].to(z.device).float() / float(max(1, self.max_dist))
        pair = torch.stack(
            [
                feats["adj"].float(),
                feats["dom"].float(),
                feats["postdom"].float(),
                dist,
            ],
            dim=-1,
        )
        zi = z.unsqueeze(1).expand(n, n, -1)
        zj = z.unsqueeze(0).expand(n, n, -1)
        return torch.cat([zi, zj, pair], dim=-1)

    def forward(
        self,
        *,
        binary_node_embs: Tensor,
        pattern_logits: Tensor,
        bin_struct_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        bin_batch: Optional[Tensor] = None,
        cfg_features: Optional[Dict[str, Tensor]] = None,
    ) -> BinaryCSTParserOutput:
        n = int(binary_node_embs.size(0))
        if n == 0:
            empty = binary_node_embs.new_zeros(0, self.d_model)
            return BinaryCSTParserOutput(
                node_states=empty,
                ctrl_logits=binary_node_embs.new_zeros(0, NUM_CTRL_TYPES),
                role_logits=binary_node_embs.new_zeros(0, NUM_ROLE_TYPES),
                role_pair_logits=binary_node_embs.new_zeros(0, 0, NUM_ROLE_TYPES),
                root_role_logits=binary_node_embs.new_zeros(0, NUM_ROLE_TYPES),
                slot_logits=binary_node_embs.new_zeros(0, NUM_SLOT_TYPES),
                parent_scores=binary_node_embs.new_zeros(0, 0),
                root_scores=binary_node_embs.new_zeros(0),
                region_logits=binary_node_embs.new_zeros(0, 0),
                same_slot_logits=binary_node_embs.new_zeros(0, 0),
                slot_anchor_scores=binary_node_embs.new_zeros(0, 0),
                outline_count_logits=binary_node_embs.new_zeros(0, self.max_outline_controls + 1),
                outline_kind_logits=binary_node_embs.new_zeros(0, self.max_outline_controls, NUM_CTRL_TYPES),
                outline_depth_logits=binary_node_embs.new_zeros(0, self.max_outline_controls, self.max_outline_depth),
                slot_layout_count_logits=binary_node_embs.new_zeros(0, self.max_slot_layout_items + 1),
                slot_layout_kind_logits=binary_node_embs.new_zeros(0, self.max_slot_layout_items, NUM_SLOT_TYPES),
                slot_layout_depth_logits=binary_node_embs.new_zeros(0, self.max_slot_layout_items, self.max_slot_layout_depth),
                slot_primary_logits=binary_node_embs.new_zeros(0, self.max_slot_layout_items, 0),
                slot_context_logits=binary_node_embs.new_zeros(0, self.max_slot_layout_items, 0),
                structural_bias=binary_node_embs.new_zeros(1, 0, 0),
            )

        pat = F.softmax(pattern_logits.float(), dim=-1)
        struct = bin_struct_features.to(device=binary_node_embs.device, dtype=torch.float32)
        if struct.size(-1) != self.struct_feat_dim:
            if struct.size(-1) > self.struct_feat_dim:
                struct = struct[:, : self.struct_feat_dim]
            else:
                pad = struct.new_zeros(struct.size(0), self.struct_feat_dim - struct.size(-1))
                struct = torch.cat([struct, pad], dim=-1)
        x = self.input_proj(torch.cat([binary_node_embs.float(), pat, struct], dim=-1))
        if cfg_features is None:
            feats = build_cfg_structural_features(
                edge_index.to(device=x.device),
                edge_type.to(device=x.device),
                n,
                max_dist=self.max_dist,
                bin_batch=bin_batch.to(device=x.device) if bin_batch is not None else None,
            )
        else:
            feats = {k: v.to(device=x.device) for k, v in cfg_features.items()}
        last_bias = x.new_zeros(1, n, n)
        z = x
        for layer in self.layers:
            z, last_bias = layer(z, feats)

        pair_in = self._pair_features(z, feats)
        region_logits = self.region_pair(pair_in).squeeze(-1)
        same_slot_logits = self.same_slot_pair(pair_in).squeeze(-1)
        slot_anchor_scores = self.slot_anchor_pair(pair_in).squeeze(-1)

        child = self.parent_child(z)
        parent = self.parent_parent(z)
        bilinear = child @ parent.t() / (self.d_model ** 0.5)
        parent_scores = bilinear + self.parent_pair(pair_in).squeeze(-1)
        root_scores = self.root_head(z).squeeze(-1)
        outline_count_logits, outline_kind_logits, outline_depth_logits = self._outline_logits(z, bin_batch)
        (
            slot_layout_count_logits,
            slot_layout_kind_logits,
            slot_layout_depth_logits,
            _slot_layout_states,
            slot_primary_logits,
            slot_context_logits,
        ) = self._slot_layout_logits(z, bin_batch)

        return BinaryCSTParserOutput(
            node_states=z,
            ctrl_logits=self.ctrl_head(z),
            role_logits=self.role_head(z),
            role_pair_logits=self.role_pair(pair_in),
            root_role_logits=self.root_role_head(z),
            slot_logits=self.slot_head(z),
            parent_scores=parent_scores,
            root_scores=root_scores,
            region_logits=region_logits,
            same_slot_logits=same_slot_logits,
            slot_anchor_scores=slot_anchor_scores,
            outline_count_logits=outline_count_logits,
            outline_kind_logits=outline_kind_logits,
            outline_depth_logits=outline_depth_logits,
            slot_layout_count_logits=slot_layout_count_logits,
            slot_layout_kind_logits=slot_layout_kind_logits,
            slot_layout_depth_logits=slot_layout_depth_logits,
            slot_primary_logits=slot_primary_logits,
            slot_context_logits=slot_context_logits,
            structural_bias=last_bias,
        )


def _sinkhorn_with_col_marginals(scores: Tensor, col_mass: Tensor, *, iters: int = 30) -> Tensor:
    """Doubly-normalise exp(scores) with row mass 1/n and supplied col mass.

    This is used as a differentiable BB-to-slot-candidate OT assignment.  Slot
    candidates are BB anchors; ``col_mass`` comes from teacher slot sizes.
    """
    n = int(scores.size(0))
    if n == 0:
        return scores
    log_q = scores.float().clamp(min=-30, max=30)
    row_mass = scores.new_full((n,), 1.0 / float(n)).float()
    col_mass = col_mass.float().to(scores.device)
    col_mass = col_mass.clamp(min=1e-8)
    col_mass = col_mass / col_mass.sum().clamp(min=1e-8)
    for _ in range(int(iters)):
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True) + row_mass.log().view(-1, 1)
        log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True) + col_mass.log().view(1, -1)
    return log_q.exp()


@torch.no_grad()
def decode_parent_tree(scores: Tensor, root_scores: Tensor) -> List[int]:
    """Greedy acyclic parent decode with a virtual root encoded as ``n``.

    This is a lightweight constrained decode for smoke/audit.  It enforces a
    forest rooted at the virtual root and breaks cycles by attaching the weakest
    cycle node to root.  The training loss still uses Matrix-Tree NLL.
    """
    n = int(scores.size(0))
    if n == 0:
        return []
    parent: List[int] = []
    for i in range(n):
        cand = torch.cat([scores[i], root_scores[i].view(1)], dim=0).detach().cpu()
        cand[i] = -1e9
        parent.append(int(cand.argmax().item()))

    def find_cycle() -> List[int]:
        for start in range(n):
            seen: Dict[int, int] = {}
            cur = start
            while 0 <= cur < n:
                if cur in seen:
                    path = list(seen.keys())
                    return path[seen[cur] :]
                seen[cur] = len(seen)
                nxt = parent[cur]
                if nxt == n:
                    break
                cur = nxt
        return []

    while True:
        cyc = find_cycle()
        if not cyc:
            break
        weakest = min(cyc, key=lambda x: float(root_scores[x].detach().cpu().item()))
        parent[weakest] = n
    if all(p != n for p in parent):
        best_root = int(root_scores.detach().cpu().argmax().item())
        parent[best_root] = n
    return parent


def _directed_tree_nll(scores: Tensor, root_scores: Tensor, parent_targets: Sequence[int]) -> Tensor:
    """Matrix-Tree-Theorem NLL for one graph.

    ``scores[child, parent]`` scores parent -> child arcs.  The virtual root is
    represented by ``root_scores[child]``.  Targets use local parent indices or
    ``n`` for root.
    """
    n = int(scores.size(0))
    device = scores.device
    valid = [i for i, p in enumerate(parent_targets) if int(p) != PARENT_IGNORE and int(p) != i]
    if not valid:
        return scores.sum() * 0.0
    s = scores.float().clone()
    s.fill_diagonal_(-1e9)
    weights = torch.exp(s.clamp(min=-30, max=30))
    root_w = torch.exp(root_scores.float().clamp(min=-30, max=30))
    lap = torch.zeros(n, n, device=device, dtype=torch.float32)
    incoming_sum = weights.sum(dim=1) + root_w
    lap.diagonal().copy_(incoming_sum)
    lap = lap - weights.t()
    # Directed rooted tree partition with root attachments folded into diagonal.
    log_z = torch.logdet(lap + torch.eye(n, device=device) * 1e-4)
    gold = scores.new_tensor(0.0)
    for child, parent in enumerate(parent_targets):
        p = int(parent)
        if p == PARENT_IGNORE or p == child:
            continue
        if p == n:
            gold = gold + root_scores[child]
        elif 0 <= p < n and p != child:
            gold = gold + scores[child, p]
    return log_z - gold


def compute_bskel_loss(
    out: BinaryCSTParserOutput,
    *,
    ctrl_targets: Tensor,
    role_targets: Tensor,
    slot_targets: Tensor,
    outline_count_targets: Optional[Tensor] = None,
    outline_kind_targets: Optional[Tensor] = None,
    outline_depth_targets: Optional[Tensor] = None,
    slot_layout_count_targets: Optional[Tensor] = None,
    slot_layout_kind_targets: Optional[Tensor] = None,
    slot_layout_depth_targets: Optional[Tensor] = None,
    slot_primary_targets_by_graph: Optional[Sequence[Tensor]] = None,
    slot_context_targets_by_graph: Optional[Sequence[Tensor]] = None,
    slot_grounding_masks_by_graph: Optional[Sequence[Tensor]] = None,
    parent_targets_by_graph: Sequence[Sequence[int]],
    slot_anchor_targets_by_graph: Sequence[Sequence[int]],
    slot_col_mass_by_graph: Sequence[Tensor],
    region_targets_by_graph: Sequence[Tensor],
    same_slot_targets_by_graph: Sequence[Tensor],
    bin_batch: Tensor,
    ctrl_weight: float = 1.0,
    role_weight: float = 1.0,
    slot_weight: float = 0.5,
    parent_weight: float = 1.0,
    tree_weight: float = 0.2,
    region_weight: float = 1.0,
    same_slot_weight: float = 0.5,
    slot_ot_weight: float = 0.5,
    role_pair_weight: float = 0.5,
    outline_count_weight: float = 1.0,
    outline_kind_weight: float = 1.0,
    outline_depth_weight: float = 1.0,
    slot_layout_count_weight: float = 1.0,
    slot_layout_kind_weight: float = 1.0,
    slot_layout_depth_weight: float = 1.0,
    slot_primary_weight: float = 1.0,
    slot_context_weight: float = 0.5,
    loss_profile: str = "legacy",
    cst_weight: float = 1.0,
    region_struct_weight: float = 1.0,
    slot_struct_weight: float = 1.0,
    ot_weight: float = 1.0,
    skel_weight: float = 1.0,
) -> Dict[str, Tensor]:
    device = out.ctrl_logits.device
    zero = out.ctrl_logits.new_tensor(0.0)
    losses: Dict[str, Tensor] = {}

    ctrl_t = ctrl_targets.to(device=device, dtype=torch.long)
    ctrl_weight_t = out.ctrl_logits.new_tensor([0.35, 2.0, 3.0, 8.0])
    losses["loss_ctrl"] = F.cross_entropy(out.ctrl_logits, ctrl_t, weight=ctrl_weight_t) if ctrl_t.numel() else zero

    role_t = role_targets.to(device=device, dtype=torch.long)
    if bool((role_t != ROLE_IGNORE).any().item()):
        losses["loss_role"] = F.cross_entropy(out.role_logits, role_t, ignore_index=ROLE_IGNORE)
    else:
        losses["loss_role"] = zero

    slot_t = slot_targets.to(device=device, dtype=torch.long)
    if bool((slot_t != SLOT_IGNORE).any().item()):
        losses["loss_slot"] = F.cross_entropy(out.slot_logits, slot_t, ignore_index=SLOT_IGNORE)
    else:
        losses["loss_slot"] = zero

    if outline_count_targets is not None and out.outline_count_logits.numel() > 0:
        oct_t = outline_count_targets.to(device=device, dtype=torch.long)
        losses["loss_outline_count"] = F.cross_entropy(out.outline_count_logits, oct_t)
    else:
        losses["loss_outline_count"] = zero

    if outline_kind_targets is not None and out.outline_kind_logits.numel() > 0:
        okt = outline_kind_targets.to(device=device, dtype=torch.long)
        kind_weight = out.outline_kind_logits.new_tensor([0.25, 2.0, 3.0, 8.0])
        losses["loss_outline_kind"] = F.cross_entropy(
            out.outline_kind_logits.reshape(-1, NUM_CTRL_TYPES),
            okt.reshape(-1),
            weight=kind_weight,
        )
    else:
        losses["loss_outline_kind"] = zero

    if outline_depth_targets is not None and out.outline_depth_logits.numel() > 0:
        odt = outline_depth_targets.to(device=device, dtype=torch.long)
        losses["loss_outline_depth"] = F.cross_entropy(
            out.outline_depth_logits.reshape(-1, out.outline_depth_logits.size(-1)),
            odt.reshape(-1),
            ignore_index=OUTLINE_DEPTH_IGNORE,
        )
    else:
        losses["loss_outline_depth"] = zero

    if slot_layout_count_targets is not None and out.slot_layout_count_logits.numel() > 0:
        slct = slot_layout_count_targets.to(device=device, dtype=torch.long)
        losses["loss_slot_layout_count"] = F.cross_entropy(out.slot_layout_count_logits, slct)
    else:
        losses["loss_slot_layout_count"] = zero

    if slot_layout_kind_targets is not None and out.slot_layout_kind_logits.numel() > 0:
        slkt = slot_layout_kind_targets.to(device=device, dtype=torch.long)
        slot_kind_weight = out.slot_layout_kind_logits.new_tensor([0.20, 2.0, 2.0, 3.0])
        losses["loss_slot_layout_kind"] = F.cross_entropy(
            out.slot_layout_kind_logits.reshape(-1, NUM_SLOT_TYPES),
            slkt.reshape(-1),
            weight=slot_kind_weight,
        )
    else:
        losses["loss_slot_layout_kind"] = zero

    if slot_layout_depth_targets is not None and out.slot_layout_depth_logits.numel() > 0:
        sldt = slot_layout_depth_targets.to(device=device, dtype=torch.long)
        losses["loss_slot_layout_depth"] = F.cross_entropy(
            out.slot_layout_depth_logits.reshape(-1, out.slot_layout_depth_logits.size(-1)),
            sldt.reshape(-1),
            ignore_index=OUTLINE_DEPTH_IGNORE,
        )
    else:
        losses["loss_slot_layout_depth"] = zero

    parent_terms: List[Tensor] = []
    tree_terms: List[Tensor] = []
    region_terms: List[Tensor] = []
    same_terms: List[Tensor] = []
    slot_ot_terms: List[Tensor] = []
    role_pair_terms: List[Tensor] = []
    slot_primary_terms: List[Tensor] = []
    slot_context_terms: List[Tensor] = []
    bin_batch = bin_batch.to(device=device, dtype=torch.long)
    n_graphs = int(bin_batch.max().item()) + 1 if bin_batch.numel() else 0
    for gi in range(n_graphs):
        nodes = (bin_batch == gi).nonzero(as_tuple=True)[0]
        if nodes.numel() == 0:
            continue
        n_local = int(nodes.numel())
        targets = list(parent_targets_by_graph[gi]) if gi < len(parent_targets_by_graph) else []
        local_scores = out.parent_scores[nodes][:, nodes]
        local_roots = out.root_scores[nodes]
        if len(targets) == n_local and (float(parent_weight) > 0.0 or float(tree_weight) > 0.0 or float(role_pair_weight) > 0.0):
            for local_i, tgt in enumerate(targets):
                tgt_i = int(tgt)
                if tgt_i == PARENT_IGNORE or tgt_i == local_i:
                    continue
                row_scores = local_scores[local_i].clone()
                row_scores[local_i] = -1e9
                cand = torch.cat([row_scores, local_roots[local_i].view(1)], dim=0)
                target_idx = n_local if tgt_i == n_local else tgt_i
                if 0 <= target_idx <= n_local and float(parent_weight) > 0.0:
                    parent_terms.append(F.cross_entropy(cand.view(1, -1), torch.tensor([target_idx], device=device)))
                if 0 <= target_idx <= n_local and float(role_pair_weight) > 0.0:
                    if tgt_i == n_local:
                        role_pair_terms.append(
                            F.cross_entropy(
                                out.root_role_logits[nodes[local_i]].view(1, -1),
                                role_t[nodes[local_i]].view(1),
                                ignore_index=ROLE_IGNORE,
                            ),
                        )
                    elif 0 <= tgt_i < n_local:
                        role_pair_terms.append(
                            F.cross_entropy(
                                out.role_pair_logits[nodes[local_i], nodes[tgt_i]].view(1, -1),
                                role_t[nodes[local_i]].view(1),
                                ignore_index=ROLE_IGNORE,
                            ),
                        )
            if float(tree_weight) > 0.0:
                tree_targets = [
                    (PARENT_IGNORE if int(tgt) == local_i else int(tgt))
                    for local_i, tgt in enumerate(targets)
                ]
                tree_terms.append(_directed_tree_nll(local_scores, local_roots, tree_targets))
        if float(slot_ot_weight) > 0.0 and gi < len(slot_anchor_targets_by_graph) and gi < len(slot_col_mass_by_graph):
            anchors = list(slot_anchor_targets_by_graph[gi])
            if len(anchors) == n_local:
                col_mass = slot_col_mass_by_graph[gi].to(device=device, dtype=torch.float32)
                if col_mass.numel() == n_local:
                    q = _sinkhorn_with_col_marginals(out.slot_anchor_scores[nodes][:, nodes], col_mass)
                    gold = torch.zeros_like(q)
                    for i, a in enumerate(anchors):
                        ai = int(a)
                        if 0 <= ai < n_local:
                            gold[i, ai] = 1.0 / float(n_local)
                    if bool((gold > 0).any().item()):
                        slot_ot_terms.append(F.kl_div(q.clamp(min=1e-8).log(), gold, reduction="batchmean"))
        if float(region_weight) > 0.0 and gi < len(region_targets_by_graph):
            rt = region_targets_by_graph[gi].to(device=device, dtype=torch.float32)
            if rt.shape == (n_local, n_local):
                region_terms.append(F.binary_cross_entropy_with_logits(out.region_logits[nodes][:, nodes], rt))
        if float(same_slot_weight) > 0.0 and gi < len(same_slot_targets_by_graph):
            st = same_slot_targets_by_graph[gi].to(device=device, dtype=torch.float32)
            if st.shape == (n_local, n_local):
                same_terms.append(F.binary_cross_entropy_with_logits(out.same_slot_logits[nodes][:, nodes], st))
        if (
            slot_primary_targets_by_graph is not None
            and slot_grounding_masks_by_graph is not None
            and float(slot_primary_weight) > 0.0
            and gi < len(slot_primary_targets_by_graph)
            and gi < len(slot_grounding_masks_by_graph)
            and out.slot_primary_logits.numel() > 0
        ):
            pt = slot_primary_targets_by_graph[gi].to(device=device, dtype=torch.float32)
            gm = slot_grounding_masks_by_graph[gi].to(device=device, dtype=torch.float32)
            logits = out.slot_primary_logits[gi, :, nodes]
            if pt.shape == logits.shape and gm.shape == logits.shape and bool((gm > 0).any().item()):
                pos = (pt * gm).sum()
                neg = ((1.0 - pt) * gm).sum()
                pos_weight = (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=20.0)
                raw = F.binary_cross_entropy_with_logits(logits, pt, reduction="none")
                raw = raw * (1.0 + (pos_weight - 1.0) * pt)
                slot_primary_terms.append((raw * gm).sum() / gm.sum().clamp_min(1.0))
        if (
            slot_context_targets_by_graph is not None
            and slot_grounding_masks_by_graph is not None
            and float(slot_context_weight) > 0.0
            and gi < len(slot_context_targets_by_graph)
            and gi < len(slot_grounding_masks_by_graph)
            and out.slot_context_logits.numel() > 0
        ):
            ct = slot_context_targets_by_graph[gi].to(device=device, dtype=torch.float32)
            gm = slot_grounding_masks_by_graph[gi].to(device=device, dtype=torch.float32)
            logits = out.slot_context_logits[gi, :, nodes]
            if ct.shape == logits.shape and gm.shape == logits.shape and bool((gm > 0).any().item()):
                pos = (ct * gm).sum()
                neg = ((1.0 - ct) * gm).sum()
                pos_weight = (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=10.0)
                raw = F.binary_cross_entropy_with_logits(logits, ct, reduction="none")
                raw = raw * (1.0 + (pos_weight - 1.0) * ct)
                slot_context_terms.append((raw * gm).sum() / gm.sum().clamp_min(1.0))

    losses["loss_parent"] = torch.stack(parent_terms).mean() if parent_terms else zero
    losses["loss_tree"] = torch.stack(tree_terms).mean() if tree_terms else zero
    losses["loss_region"] = torch.stack(region_terms).mean() if region_terms else zero
    losses["loss_same_slot"] = torch.stack(same_terms).mean() if same_terms else zero
    losses["loss_slot_ot"] = torch.stack(slot_ot_terms).mean() if slot_ot_terms else zero
    losses["loss_role_pair"] = torch.stack(role_pair_terms).mean() if role_pair_terms else zero
    losses["loss_slot_primary"] = torch.stack(slot_primary_terms).mean() if slot_primary_terms else zero
    losses["loss_slot_context"] = torch.stack(slot_context_terms).mean() if slot_context_terms else zero

    profile = str(loss_profile or "legacy").lower()
    if profile == "grouped5":
        losses["loss_cst"] = (
            float(outline_count_weight) * losses["loss_outline_count"]
            + float(outline_kind_weight) * losses["loss_outline_kind"]
            + float(outline_depth_weight) * losses["loss_outline_depth"]
            + float(slot_layout_count_weight) * losses["loss_slot_layout_count"]
            + float(slot_layout_kind_weight) * losses["loss_slot_layout_kind"]
            + float(slot_layout_depth_weight) * losses["loss_slot_layout_depth"]
            + float(parent_weight) * losses["loss_parent"]
            + float(tree_weight) * losses["loss_tree"]
        )
        losses["loss_region_struct"] = (
            float(ctrl_weight) * losses["loss_ctrl"]
            + float(region_weight) * losses["loss_region"]
        )
        losses["loss_slot_struct"] = (
            float(role_weight) * losses["loss_role"]
            + float(slot_weight) * losses["loss_slot"]
            + float(same_slot_weight) * losses["loss_same_slot"]
            + float(role_pair_weight) * losses["loss_role_pair"]
            + float(slot_primary_weight) * losses["loss_slot_primary"]
            + float(slot_context_weight) * losses["loss_slot_context"]
        )
        losses["loss_ot"] = float(slot_ot_weight) * losses["loss_slot_ot"]
        losses["loss_skel"] = (
            0.5 * losses["loss_cst"]
            + 0.3 * losses["loss_slot_struct"]
            + 0.2 * losses["loss_ot"]
        )
        total = (
            float(cst_weight) * losses["loss_cst"]
            + float(region_struct_weight) * losses["loss_region_struct"]
            + float(slot_struct_weight) * losses["loss_slot_struct"]
            + float(ot_weight) * losses["loss_ot"]
            + float(skel_weight) * losses["loss_skel"]
        )
    else:
        losses["loss_cst"] = (
            float(outline_count_weight) * losses["loss_outline_count"]
            + float(outline_kind_weight) * losses["loss_outline_kind"]
            + float(outline_depth_weight) * losses["loss_outline_depth"]
            + float(slot_layout_count_weight) * losses["loss_slot_layout_count"]
            + float(slot_layout_kind_weight) * losses["loss_slot_layout_kind"]
            + float(slot_layout_depth_weight) * losses["loss_slot_layout_depth"]
            + float(parent_weight) * losses["loss_parent"]
            + float(tree_weight) * losses["loss_tree"]
        )
        losses["loss_region_struct"] = float(ctrl_weight) * losses["loss_ctrl"] + float(region_weight) * losses["loss_region"]
        losses["loss_slot_struct"] = (
            float(role_weight) * losses["loss_role"]
            + float(slot_weight) * losses["loss_slot"]
            + float(same_slot_weight) * losses["loss_same_slot"]
            + float(role_pair_weight) * losses["loss_role_pair"]
            + float(slot_primary_weight) * losses["loss_slot_primary"]
            + float(slot_context_weight) * losses["loss_slot_context"]
        )
        losses["loss_ot"] = float(slot_ot_weight) * losses["loss_slot_ot"]
        losses["loss_skel"] = zero
        total = (
            float(ctrl_weight) * losses["loss_ctrl"]
            + float(role_weight) * losses["loss_role"]
            + float(slot_weight) * losses["loss_slot"]
            + float(outline_count_weight) * losses["loss_outline_count"]
            + float(outline_kind_weight) * losses["loss_outline_kind"]
            + float(outline_depth_weight) * losses["loss_outline_depth"]
            + float(slot_layout_count_weight) * losses["loss_slot_layout_count"]
            + float(slot_layout_kind_weight) * losses["loss_slot_layout_kind"]
            + float(slot_layout_depth_weight) * losses["loss_slot_layout_depth"]
            + float(parent_weight) * losses["loss_parent"]
            + float(tree_weight) * losses["loss_tree"]
            + float(region_weight) * losses["loss_region"]
            + float(same_slot_weight) * losses["loss_same_slot"]
            + float(slot_ot_weight) * losses["loss_slot_ot"]
            + float(role_pair_weight) * losses["loss_role_pair"]
            + float(slot_primary_weight) * losses["loss_slot_primary"]
            + float(slot_context_weight) * losses["loss_slot_context"]
        )
    losses["loss_total"] = total
    return losses


@torch.no_grad()
def bskel_metrics(
    out: BinaryCSTParserOutput,
    *,
    ctrl_targets: Tensor,
    role_targets: Tensor,
    slot_targets: Tensor,
    outline_count_targets: Optional[Tensor] = None,
    outline_kind_targets: Optional[Tensor] = None,
    outline_depth_targets: Optional[Tensor] = None,
    slot_layout_count_targets: Optional[Tensor] = None,
    slot_layout_kind_targets: Optional[Tensor] = None,
    slot_layout_depth_targets: Optional[Tensor] = None,
    slot_primary_targets_by_graph: Optional[Sequence[Tensor]] = None,
    slot_context_targets_by_graph: Optional[Sequence[Tensor]] = None,
    slot_grounding_masks_by_graph: Optional[Sequence[Tensor]] = None,
    parent_targets_by_graph: Sequence[Sequence[int]],
    slot_anchor_targets_by_graph: Sequence[Sequence[int]],
    slot_col_mass_by_graph: Sequence[Tensor],
    region_targets_by_graph: Sequence[Tensor],
    same_slot_targets_by_graph: Sequence[Tensor],
    bin_batch: Tensor,
) -> Dict[str, float]:
    device = out.ctrl_logits.device
    metrics: Dict[str, float] = {}
    ctrl_t = ctrl_targets.to(device=device, dtype=torch.long)
    ctrl_p = out.ctrl_logits.argmax(dim=-1)
    metrics["ctrl_acc"] = float((ctrl_p == ctrl_t).float().mean().item()) if ctrl_t.numel() else 0.0
    for cid, name in [(CTRL_IF, "if"), (CTRL_LOOP, "loop"), (CTRL_SWITCH, "switch")]:
        m = ctrl_t == cid
        metrics[f"ctrl_{name}_recall"] = float((ctrl_p[m] == cid).float().mean().item()) if bool(m.any().item()) else 0.0

    role_t = role_targets.to(device=device, dtype=torch.long)
    role_mask = role_t != ROLE_IGNORE
    metrics["role_acc"] = (
        float((out.role_logits.argmax(dim=-1)[role_mask] == role_t[role_mask]).float().mean().item())
        if bool(role_mask.any().item()) else 0.0
    )
    slot_t = slot_targets.to(device=device, dtype=torch.long)
    slot_mask = slot_t != SLOT_IGNORE
    metrics["slot_acc"] = (
        float((out.slot_logits.argmax(dim=-1)[slot_mask] == slot_t[slot_mask]).float().mean().item())
        if bool(slot_mask.any().item()) else 0.0
    )
    if outline_kind_targets is not None and out.outline_kind_logits.numel() > 0:
        okt = outline_kind_targets.to(device=device, dtype=torch.long)
        okp = out.outline_kind_logits.argmax(dim=-1)
        if outline_count_targets is not None and out.outline_count_logits.numel() > 0:
            oct_t = outline_count_targets.to(device=device, dtype=torch.long)
            count_p = out.outline_count_logits.argmax(dim=-1)
            gold_len = oct_t.clamp(min=0, max=okp.size(1))
            pred_len = count_p.clamp(min=0, max=okp.size(1))
            metrics["outline_count_acc"] = float((pred_len == gold_len).float().mean().item()) if gold_len.numel() else 0.0
        else:
            pred_len = (okp != CTRL_NONE).sum(dim=1)
            gold_len = (okt != CTRL_NONE).sum(dim=1)
            metrics["outline_count_acc"] = float((pred_len == gold_len).float().mean().item()) if gold_len.numel() else 0.0
        valid = okt != CTRL_NONE
        metrics["outline_kind_acc"] = (
            float((okp[valid] == okt[valid]).float().mean().item()) if bool(valid.any().item()) else 0.0
        )
        seq_ok = []
        for i in range(okt.size(0)):
            gl = int(gold_len[i].item())
            pl = int(pred_len[i].item())
            seq_ok.append(float(gl == pl and torch.equal(okp[i, :gl], okt[i, :gl])))
        metrics["outline_kindseq_acc"] = float(sum(seq_ok) / len(seq_ok)) if seq_ok else 0.0
    else:
        metrics["outline_kind_acc"] = 0.0
        metrics["outline_count_acc"] = 0.0
        metrics["outline_kindseq_acc"] = 0.0
    if (
        outline_kind_targets is not None
        and outline_depth_targets is not None
        and out.outline_kind_logits.numel() > 0
        and out.outline_depth_logits.numel() > 0
    ):
        okt = outline_kind_targets.to(device=device, dtype=torch.long)
        odt = outline_depth_targets.to(device=device, dtype=torch.long)
        okp = out.outline_kind_logits.argmax(dim=-1)
        odp = out.outline_depth_logits.argmax(dim=-1)
        if outline_count_targets is not None and out.outline_count_logits.numel() > 0:
            oct_t = outline_count_targets.to(device=device, dtype=torch.long)
            count_p = out.outline_count_logits.argmax(dim=-1)
            gold_len = oct_t.clamp(min=0, max=okp.size(1))
            pred_len = count_p.clamp(min=0, max=okp.size(1))
        else:
            gold_len = (okt != CTRL_NONE).sum(dim=1)
            pred_len = (okp != CTRL_NONE).sum(dim=1)
        valid_depth = odt != OUTLINE_DEPTH_IGNORE
        metrics["outline_depth_acc"] = (
            float((odp[valid_depth] == odt[valid_depth]).float().mean().item())
            if bool(valid_depth.any().item()) else 0.0
        )
        outline_ok = []
        for i in range(okt.size(0)):
            gl = int(gold_len[i].item())
            pl = int(pred_len[i].item())
            outline_ok.append(
                float(
                    gl == pl
                    and torch.equal(okp[i, :gl], okt[i, :gl])
                    and torch.equal(odp[i, :gl], odt[i, :gl])
                ),
            )
        metrics["outline_exact_acc"] = float(sum(outline_ok) / len(outline_ok)) if outline_ok else 0.0
    else:
        metrics["outline_depth_acc"] = 0.0
        metrics["outline_exact_acc"] = 0.0

    if slot_layout_kind_targets is not None and out.slot_layout_kind_logits.numel() > 0:
        slkt = slot_layout_kind_targets.to(device=device, dtype=torch.long)
        slkp = out.slot_layout_kind_logits.argmax(dim=-1)
        if slot_layout_count_targets is not None and out.slot_layout_count_logits.numel() > 0:
            slct = slot_layout_count_targets.to(device=device, dtype=torch.long)
            sl_count_p = out.slot_layout_count_logits.argmax(dim=-1)
            sl_gold_len = slct.clamp(min=0, max=slkp.size(1))
            sl_pred_len = sl_count_p.clamp(min=0, max=slkp.size(1))
            metrics["slot_layout_count_acc"] = (
                float((sl_pred_len == sl_gold_len).float().mean().item()) if sl_gold_len.numel() else 0.0
            )
        else:
            sl_pred_len = (slkp != SLOT_NONE).sum(dim=1)
            sl_gold_len = (slkt != SLOT_NONE).sum(dim=1)
            metrics["slot_layout_count_acc"] = (
                float((sl_pred_len == sl_gold_len).float().mean().item()) if sl_gold_len.numel() else 0.0
            )
        sl_valid = slkt != SLOT_NONE
        metrics["slot_layout_kind_acc"] = (
            float((slkp[sl_valid] == slkt[sl_valid]).float().mean().item())
            if bool(sl_valid.any().item()) else 0.0
        )
        sl_seq_ok = []
        for i in range(slkt.size(0)):
            gl = int(sl_gold_len[i].item())
            pl = int(sl_pred_len[i].item())
            sl_seq_ok.append(float(gl == pl and torch.equal(slkp[i, :gl], slkt[i, :gl])))
        metrics["slot_layout_kindseq_acc"] = float(sum(sl_seq_ok) / len(sl_seq_ok)) if sl_seq_ok else 0.0
    else:
        metrics["slot_layout_count_acc"] = 0.0
        metrics["slot_layout_kind_acc"] = 0.0
        metrics["slot_layout_kindseq_acc"] = 0.0

    if (
        slot_layout_kind_targets is not None
        and slot_layout_depth_targets is not None
        and out.slot_layout_kind_logits.numel() > 0
        and out.slot_layout_depth_logits.numel() > 0
    ):
        slkt = slot_layout_kind_targets.to(device=device, dtype=torch.long)
        sldt = slot_layout_depth_targets.to(device=device, dtype=torch.long)
        slkp = out.slot_layout_kind_logits.argmax(dim=-1)
        sldp = out.slot_layout_depth_logits.argmax(dim=-1)
        if slot_layout_count_targets is not None and out.slot_layout_count_logits.numel() > 0:
            slct = slot_layout_count_targets.to(device=device, dtype=torch.long)
            sl_count_p = out.slot_layout_count_logits.argmax(dim=-1)
            sl_gold_len = slct.clamp(min=0, max=slkp.size(1))
            sl_pred_len = sl_count_p.clamp(min=0, max=slkp.size(1))
        else:
            sl_gold_len = (slkt != SLOT_NONE).sum(dim=1)
            sl_pred_len = (slkp != SLOT_NONE).sum(dim=1)
        sl_valid_depth = sldt != OUTLINE_DEPTH_IGNORE
        metrics["slot_layout_depth_acc"] = (
            float((sldp[sl_valid_depth] == sldt[sl_valid_depth]).float().mean().item())
            if bool(sl_valid_depth.any().item()) else 0.0
        )
        sl_exact_ok = []
        for i in range(slkt.size(0)):
            gl = int(sl_gold_len[i].item())
            pl = int(sl_pred_len[i].item())
            sl_exact_ok.append(
                float(
                    gl == pl
                    and torch.equal(slkp[i, :gl], slkt[i, :gl])
                    and torch.equal(sldp[i, :gl], sldt[i, :gl])
                ),
            )
        metrics["slot_layout_exact_acc"] = float(sum(sl_exact_ok) / len(sl_exact_ok)) if sl_exact_ok else 0.0
    else:
        metrics["slot_layout_depth_acc"] = 0.0
        metrics["slot_layout_exact_acc"] = 0.0

    total_parent = 0
    correct_parent = 0
    region_rows: List[float] = []
    same_rows: List[float] = []
    slot_anchor_total = 0
    slot_anchor_correct = 0
    slot_primary_rows: List[float] = []
    slot_context_rows: List[float] = []
    slot_primary_exact_rows: List[float] = []
    slot_context_exact_rows: List[float] = []
    bin_batch = bin_batch.to(device=device, dtype=torch.long)
    n_graphs = int(bin_batch.max().item()) + 1 if bin_batch.numel() else 0
    for gi in range(n_graphs):
        nodes = (bin_batch == gi).nonzero(as_tuple=True)[0]
        if nodes.numel() == 0:
            continue
        n_local = int(nodes.numel())
        if gi < len(parent_targets_by_graph):
            targets = list(parent_targets_by_graph[gi])
            if len(targets) == n_local:
                local_scores = out.parent_scores[nodes][:, nodes]
                local_roots = out.root_scores[nodes]
                for local_i, tgt in enumerate(targets):
                    tgt_i = int(tgt)
                    if tgt_i == PARENT_IGNORE or tgt_i == local_i:
                        continue
                    pred = int(torch.cat([local_scores[local_i], local_roots[local_i].view(1)]).argmax().item())
                    target_idx = n_local if tgt_i == n_local else tgt_i
                    correct_parent += int(pred == target_idx)
                    total_parent += 1
        if gi < len(slot_anchor_targets_by_graph) and gi < len(slot_col_mass_by_graph):
            anchors = list(slot_anchor_targets_by_graph[gi])
            if len(anchors) == n_local:
                col_mass = slot_col_mass_by_graph[gi].to(device=device, dtype=torch.float32)
                if col_mass.numel() == n_local:
                    q = _sinkhorn_with_col_marginals(out.slot_anchor_scores[nodes][:, nodes], col_mass)
                    pred_anchor = q.argmax(dim=1).detach().cpu().tolist()
                    for pa, ga in zip(pred_anchor, anchors):
                        if int(ga) >= 0:
                            slot_anchor_correct += int(int(pa) == int(ga))
                            slot_anchor_total += 1
        if gi < len(region_targets_by_graph):
            rt = region_targets_by_graph[gi].to(device=device)
            if rt.shape == (n_local, n_local):
                pred = (torch.sigmoid(out.region_logits[nodes][:, nodes]) >= 0.5).float()
                region_rows.append(float((pred == rt.float()).float().mean().item()))
        if gi < len(same_slot_targets_by_graph):
            st = same_slot_targets_by_graph[gi].to(device=device)
            if st.shape == (n_local, n_local):
                pred = (torch.sigmoid(out.same_slot_logits[nodes][:, nodes]) >= 0.5).float()
                same_rows.append(float((pred == st.float()).float().mean().item()))
        if (
            slot_primary_targets_by_graph is not None
            and slot_grounding_masks_by_graph is not None
            and gi < len(slot_primary_targets_by_graph)
            and gi < len(slot_grounding_masks_by_graph)
            and out.slot_primary_logits.numel() > 0
        ):
            pt = slot_primary_targets_by_graph[gi].to(device=device, dtype=torch.float32)
            gm = slot_grounding_masks_by_graph[gi].to(device=device, dtype=torch.float32)
            logits = out.slot_primary_logits[gi, :, nodes]
            if pt.shape == logits.shape and gm.shape == logits.shape and bool((gm > 0).any().item()):
                pred = (torch.sigmoid(logits) >= 0.5).float()
                slot_primary_rows.append(float(((pred == pt).float() * gm).sum().item() / gm.sum().clamp_min(1.0).item()))
                active = gm.sum(dim=1) > 0
                if bool(active.any().item()):
                    exact = ((pred == pt) | (gm <= 0)).all(dim=1)
                    slot_primary_exact_rows.append(float(exact[active].float().mean().item()))
        if (
            slot_context_targets_by_graph is not None
            and slot_grounding_masks_by_graph is not None
            and gi < len(slot_context_targets_by_graph)
            and gi < len(slot_grounding_masks_by_graph)
            and out.slot_context_logits.numel() > 0
        ):
            ct = slot_context_targets_by_graph[gi].to(device=device, dtype=torch.float32)
            gm = slot_grounding_masks_by_graph[gi].to(device=device, dtype=torch.float32)
            logits = out.slot_context_logits[gi, :, nodes]
            if ct.shape == logits.shape and gm.shape == logits.shape and bool((gm > 0).any().item()):
                pred = (torch.sigmoid(logits) >= 0.5).float()
                slot_context_rows.append(float(((pred == ct).float() * gm).sum().item() / gm.sum().clamp_min(1.0).item()))
                active = gm.sum(dim=1) > 0
                if bool(active.any().item()):
                    exact = ((pred == ct) | (gm <= 0)).all(dim=1)
                    slot_context_exact_rows.append(float(exact[active].float().mean().item()))
    metrics["parent_acc"] = float(correct_parent / total_parent) if total_parent else 0.0
    metrics["parent_n"] = float(total_parent)
    metrics["slot_anchor_acc"] = float(slot_anchor_correct / slot_anchor_total) if slot_anchor_total else 0.0
    metrics["region_pair_acc"] = float(sum(region_rows) / len(region_rows)) if region_rows else 0.0
    metrics["same_slot_pair_acc"] = float(sum(same_rows) / len(same_rows)) if same_rows else 0.0
    metrics["slot_primary_pair_acc"] = float(sum(slot_primary_rows) / len(slot_primary_rows)) if slot_primary_rows else 0.0
    metrics["slot_context_pair_acc"] = float(sum(slot_context_rows) / len(slot_context_rows)) if slot_context_rows else 0.0
    metrics["slot_primary_exact_acc"] = float(sum(slot_primary_exact_rows) / len(slot_primary_exact_rows)) if slot_primary_exact_rows else 0.0
    metrics["slot_context_exact_acc"] = float(sum(slot_context_exact_rows) / len(slot_context_exact_rows)) if slot_context_exact_rows else 0.0
    return metrics


@torch.no_grad()
def decode_slot_groups(out: BinaryCSTParserOutput, *, threshold: float = 0.5) -> List[List[int]]:
    n = int(out.same_slot_logits.size(0))
    if n == 0:
        return []
    same = torch.sigmoid(out.same_slot_logits[:n, :n]).detach().cpu() >= float(threshold)
    used = [False] * n
    groups: List[List[int]] = []
    for i in range(n):
        if used[i]:
            continue
        group = [j for j in range(n) if bool(same[i, j]) or bool(same[j, i])]
        if not group:
            group = [i]
        for j in group:
            used[j] = True
        groups.append(sorted(set(group)))
    return groups


@torch.no_grad()
def render_predicted_canonical_skeleton(
    out: BinaryCSTParserOutput,
    *,
    max_nodes: int = 256,
) -> str:
    """Render a simple semantic skeleton from predicted ctrl/slot/parent heads.

    This renderer is for smoke/audit only.  It intentionally produces a
    canonical semantic skeleton, not source-AST-exact code.
    """
    n = int(out.ctrl_logits.size(0))
    if n == 0:
        return "int func(...) {\n}"
    n = min(n, int(max_nodes))
    ctrl = out.ctrl_logits[:n].argmax(dim=-1).detach().cpu().tolist()
    slot = out.slot_logits[:n].argmax(dim=-1).detach().cpu().tolist()
    parents = decode_parent_tree(out.parent_scores[:n, :n], out.root_scores[:n])
    groups = decode_slot_groups(out, threshold=0.55)
    bb_to_group: Dict[int, int] = {}
    for gi, group in enumerate(groups):
        for bb in group:
            bb_to_group[int(bb)] = gi
    lines: List[str] = ["int func(...) {"]
    open_stack: List[str] = []
    sid = 0
    emitted_groups: set[int] = set()
    for i in range(n):
        c = int(ctrl[i])
        pad = "    " * (1 + len(open_stack))
        if c == CTRL_LOOP:
            lines.append(f"{pad}while (<FILL_COND:slot_{sid}>) {{  /* BB_{i} parent={parents[i]} */")
            sid += 1
            open_stack.append("loop")
            continue
        if c == CTRL_IF:
            lines.append(f"{pad}if (<FILL_COND:slot_{sid}>) {{  /* BB_{i} parent={parents[i]} */")
            sid += 1
            open_stack.append("if")
            continue
        if c == CTRL_SWITCH:
            lines.append(f"{pad}switch (<FILL_COND:slot_{sid}>) {{  /* BB_{i} parent={parents[i]} */")
            sid += 1
            open_stack.append("switch")
            continue
        s = int(slot[i])
        if s in {SLOT_BLOCK, SLOT_BRANCH, SLOT_NONE}:
            gid = bb_to_group.get(i, i)
            if gid in emitted_groups:
                continue
            emitted_groups.add(gid)
            bbs = groups[gid] if 0 <= gid < len(groups) else [i]
            lines.append(f"{pad}<FILL_BLOCK:slot_{sid}>;  /* BBs {bbs} */")
            sid += 1
        if len(open_stack) > 0 and (i + 1 == n or int(ctrl[i + 1]) in {CTRL_IF, CTRL_LOOP, CTRL_SWITCH}):
            lines.append("    " * len(open_stack) + "}")
            open_stack.pop()
    while open_stack:
        lines.append("    " * len(open_stack) + "}")
        open_stack.pop()
    lines.append("}")
    return "\n".join(lines)

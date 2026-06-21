"""
Module C -- Structure-Conditioned LLM Decoder.

This module wraps a pre-trained causal LLM with *cross-attention adapters*
that inject structural information from the binary CFG (produced by Modules A
& B) into every transformer layer.  The decoder's job is to fill in the
``<FILL_*>`` expression / statement / condition slots that appear in the
structural skeleton emitted by the skeleton builder.

Key design decisions
--------------------
* **No HuggingFace imports** in the core classes -- the adapter logic is pure
  PyTorch and can be unit-tested independently of any specific LLM.
* The heavy LLM (~6.7 B params) is loaded *outside* this file via
  :class:`AdapterInjector`, which monkey-patches each transformer layer at
  runtime to route hidden states through the cross-attention adapters.
* A separate :class:`SlotWeightBuilder` assigns per-token loss weights so that
  the training signal is concentrated on the ``FILL`` positions.

Classes
-------
CrossAttentionAdapter
    A single cross-attention block (~4 M params) inserted after one LLM layer.
StructureConditionedDecoder
    Abstract / modular wrapper that owns the adapters and orchestrates
    inference.
SlotWeightBuilder
    Utility for building per-token loss weight tensors.
AdapterInjector
    Monkey-patch helper that splices adapters into an existing HuggingFace
    model at runtime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────────────
from struclift.config import ModuleCConfig
from struclift.data.structures import SlotInfo, Skeleton

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. CrossAttentionAdapter
# ═════════════════════════════════════════════════════════════════════════════

class CrossAttentionAdapter(nn.Module):
    """Lightweight cross-attention block injected after an LLM transformer layer.

    The adapter lets the LLM's hidden states *query* graph-level node
    representations produced by Modules A/B, gated by a learned scalar so that
    the adapter can be initialised as an identity (gate = 0) and warmed up
    during fine-tuning.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the host LLM (default 4096).
    d_graph : int
        Dimension of the graph node embeddings (default 256).
    d_bottle : int
        Bottleneck dimension inside the adapter (default 256).
    n_heads : int
        Number of attention heads (default 4).

    Approximate parameter count::

        W_q: d_model * d_bottle      = 4096 * 256 = 1 048 576
        W_k: d_graph * d_bottle      =  256 * 256 =    65 536
        W_v: d_graph * d_bottle      =  256 * 256 =    65 536
        W_o: d_bottle * d_model      =  256 * 4096= 1 048 576
        LayerNorm: 2 * d_model       =     2*4096 =     8 192
        gate: 1                                         1
        ─────────────────────────────────────────────────────
        total ≈ 2.24 M  (four of these per layer ~ 4 M budgeted)

    Shape contract
    --------------
    forward(
        hidden_states : ``[B, L, d_model]``
        graph_kv      : ``[B, N, d_graph]``
        slot_mask     : ``[B, L, N]`` or ``None``
    ) → ``[B, L, d_model]``
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_graph: int = 256,
        d_bottle: int = 256,
        n_heads: int = 4,
    ) -> None:
        super().__init__()

        assert d_bottle % n_heads == 0, (
            f"d_bottle ({d_bottle}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.d_graph = d_graph
        self.d_bottle = d_bottle
        self.n_heads = n_heads
        self.d_head = d_bottle // n_heads

        # ── projections ──────────────────────────────────────────────────
        self.W_q = nn.Linear(d_model, d_bottle, bias=False)
        self.W_k = nn.Linear(d_graph, d_bottle, bias=False)
        self.W_v = nn.Linear(d_graph, d_bottle, bias=False)
        self.W_o = nn.Linear(d_bottle, d_model, bias=False)

        # ── pre-LayerNorm on the hidden states ───────────────────────────
        self.layer_norm = nn.LayerNorm(d_model)

        # ── gating scalar (initialised to 0 → identity at init time) ────
        self.gate = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    # ------------------------------------------------------------------
    def _reset_parameters(self) -> None:
        """Xavier-uniform for Q/K/V/O, zero-init gate."""
        for proj in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(proj.weight)
        # gate is already zero from ``torch.zeros``

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_kv: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply gated cross-attention from LLM hidden states to graph KV.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``[B, L, d_model]`` -- output of the preceding LLM layer.
        graph_kv : torch.Tensor
            ``[B, N, d_graph]`` -- per-node graph representations from
            Modules A/B.
        slot_mask : torch.Tensor or None
            ``[B, L, N]`` -- binary mask that controls which graph nodes each
            sequence position may attend to.  ``1.0`` = allowed, ``0.0`` =
            blocked.  If ``None`` every position attends to every node.

        Returns
        -------
        torch.Tensor
            ``[B, L, d_model]`` -- gated-residual output.
        """
        # graph_kv usually lives on embedding/encoder GPU; adapters may reside on
        # other CUDA devices when the LLM is split via accelerate device_map — move
        # activations cheaply vs duplicating LM weights onto one card.
        if graph_kv.device != hidden_states.device:
            graph_kv = graph_kv.to(hidden_states.device, non_blocking=True)
        if slot_mask is not None and slot_mask.device != hidden_states.device:
            slot_mask = slot_mask.to(hidden_states.device, non_blocking=True)

        B, L, _ = hidden_states.shape
        _, N, _ = graph_kv.shape

        residual = hidden_states
        residual_dtype = hidden_states.dtype

        # Keep adapter math in module parameter dtype (e.g. bf16) to avoid
        # mixed bf16/fp32 runtime errors during generation/checkpointing.
        ln_dtype = self.layer_norm.weight.dtype
        if hidden_states.dtype != ln_dtype:
            hidden_states = hidden_states.to(ln_dtype)
        proj_dtype = self.W_k.weight.dtype
        if graph_kv.dtype != proj_dtype:
            graph_kv = graph_kv.to(proj_dtype)

        # ── pre-LayerNorm ────────────────────────────────────────────────
        normed = self.layer_norm(hidden_states)       # [B, L, d_model]

        # ── Q, K, V projections ──────────────────────────────────────────
        Q = self.W_q(normed)                          # [B, L, d_bottle]
        K = self.W_k(graph_kv)                        # [B, N, d_bottle]
        V = self.W_v(graph_kv)                        # [B, N, d_bottle]

        # ── reshape for multi-head attention ─────────────────────────────
        #   Q : [B, L, d_bottle] → [B, H, L, d_head]
        #   K : [B, N, d_bottle] → [B, H, N, d_head]
        #   V : [B, N, d_bottle] → [B, H, N, d_head]
        Q = Q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # ── scaled dot-product attention ─────────────────────────────────
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # attn_weights : [B, H, L, N]

        if slot_mask is not None:
            # slot_mask : [B, L, N] → [B, 1, L, N] (broadcast over heads)
            mask_expanded = slot_mask.unsqueeze(1)    # [B, 1, L, N]
            attn_weights = attn_weights.masked_fill(
                mask_expanded == 0.0,
                float("-inf"),
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        # Guard against all-masked rows producing NaN after softmax
        attn_weights = attn_weights.nan_to_num(0.0)

        # ── weighted sum ─────────────────────────────────────────────────
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, d_head]
        attn_output = (
            attn_output.transpose(1, 2)               # [B, L, H, d_head]
            .contiguous()
            .view(B, L, self.d_bottle)                # [B, L, d_bottle]
        )

        # ── output projection ────────────────────────────────────────────
        attn_output = self.W_o(attn_output)           # [B, L, d_model]

        # ── gated residual ───────────────────────────────────────────────
        output = residual + torch.tanh(self.gate) * attn_output.to(residual.dtype)
        if output.dtype != residual_dtype:
            output = output.to(residual_dtype)
        return output


# ═════════════════════════════════════════════════════════════════════════════
# 2. StructureConditionedDecoder
# ═════════════════════════════════════════════════════════════════════════════

class StructureConditionedDecoder(nn.Module):
    """Abstract / modular wrapper that owns cross-attention adapters and the
    projection layer for graph embeddings.

    .. note::

        The actual LLM (~6.7 B parameters) is **not** instantiated here.  Call
        :meth:`inject_into_llm` (or use :class:`AdapterInjector`) to splice
        the adapters into an existing HuggingFace model after loading it in
        the training script.

    Parameters
    ----------
    config : ModuleCConfig
        Decoder hyper-parameters (LLM hidden dim, adapter bottleneck, etc.).

    Attributes
    ----------
    adapters : nn.ModuleList
        One :class:`CrossAttentionAdapter` per LLM transformer layer.
    graph_proj : nn.Linear
        Projects graph embeddings (potentially refined by Module B) into the
        adapter key/value space.

    Shape contract (conceptual forward)
    ------------------------------------
    forward(
        input_ids        : ``[B, L]``      -- tokenised skeleton + fill text
        graph_kv         : ``[B, N, d_graph]``
        slot_mask        : ``[B, L, N]``
        labels           : ``[B, L]`` or ``None``
        slot_weight_mask : ``[B, L]`` or ``None``
    ) → (logits ``[B, L, V]``, loss ``scalar`` or ``None``)
    """

    # Special token sentinel -- the tokeniser must map ``<FILL_*>`` variants
    # to ids whose range starts here.  The exact ids are set during data
    # preprocessing; the constant below is used in :meth:`build_slot_mask`
    # as a lower-bound to identify FILL token positions.
    FILL_TOKEN_ID_RANGE_START: int = 32_000

    def __init__(self, config: ModuleCConfig) -> None:
        super().__init__()
        self.config = config

        # ── one adapter per LLM layer ────────────────────────────────────
        self.adapters = nn.ModuleList([
            CrossAttentionAdapter(
                d_model=config.d_llm,
                d_graph=config.d_graph,
                d_bottle=config.d_bottle,
                n_heads=config.adapter_heads,
            )
            for _ in range(config.num_llm_layers)
        ])

        # ── graph embedding projection (identity-dimensioned, but lets
        #    the model learn a task-specific transform) ───────────────────
        self.graph_proj = nn.Linear(config.d_graph, config.d_graph, bias=True)

        # Placeholder: the LLM is loaded externally and injected via
        # ``AdapterInjector.inject`` or ``self.inject_into_llm``.
        self._llm: Optional[nn.Module] = None
        self._graph_kv_holder: Optional[_GraphKVHolder] = None

    # ------------------------------------------------------------------
    # LLM integration helpers
    # ------------------------------------------------------------------

    def inject_into_llm(self, llm_model: nn.Module) -> None:
        """Splice adapters into *llm_model* via :class:`AdapterInjector`.

        After calling this the LLM's ``forward`` will automatically route
        hidden states through the cross-attention adapters at each layer.

        Parameters
        ----------
        llm_model : nn.Module
            A HuggingFace-style causal LM whose transformer body exposes
            layers as an iterable (e.g. ``model.model.layers``).
        """
        self._llm = llm_model
        self._graph_kv_holder = _GraphKVHolder()
        AdapterInjector.inject(llm_model, self.adapters, self._graph_kv_holder)

    # ------------------------------------------------------------------
    # Slot mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_slot_mask(
        skeleton_token_ids: torch.Tensor,
        slot_to_bb_mapping: List[Dict[int, List[int]]],
        num_bbs: int,
        fill_token_range_start: int = 32_000,
        fill_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a ``[B, L, N]`` slot mask for cross-attention.

        Rules
        -----
        * **FILL tokens** -- the position may only attend to the basic blocks
          that the corresponding slot was aligned to (via Module B).
        * **Non-FILL tokens** (structural keywords like ``if``, ``{``, etc.)
          attend to *all* basic blocks so the LLM can ground its structural
          predictions globally.

        Parameters
        ----------
        skeleton_token_ids : torch.Tensor
            ``[B, L]`` int64 -- tokenised skeleton (FILL tokens have ids
            >= *fill_token_range_start*).
        slot_to_bb_mapping : list of dict
            Length-B list.  Each dict maps **token position** (int) to
            a list of aligned basic-block indices (``List[int]``).
        num_bbs : int
            Number of basic blocks ``N`` in the graph.
        fill_token_range_start : int
            Token-id threshold: ids >= this are considered FILL tokens when
            *fill_token_ids* is ``None``.
        fill_token_ids : torch.Tensor, optional
            Shape ``[K]`` long — explicit ``<|STRUCLIFT_FILL_*|>`` ids (与 augment /
            Stage3 tokenizer add_tokens 一致). 若给出则优先 ``torch.isin`` 判定 FILL，
            避免 DeepSeek 等词表在 32000+ 仍有大量普通 token 时的误判。

        Returns
        -------
        torch.Tensor
            ``[B, L, N]`` float32 mask.  ``1.0`` = attend, ``0.0`` = block.
        """
        B, L = skeleton_token_ids.shape
        device = skeleton_token_ids.device

        # Start with all-ones (non-FILL default: attend to every BB).
        mask = torch.ones(B, L, num_bbs, dtype=torch.float32, device=device)

        if fill_token_ids is not None and fill_token_ids.numel() > 0:
            fid = fill_token_ids.to(device=device, dtype=torch.long).flatten()
            is_fill = torch.isin(skeleton_token_ids, fid)
        else:
            is_fill = skeleton_token_ids >= fill_token_range_start  # [B, L]

        for b in range(B):
            mapping_b = slot_to_bb_mapping[b] if b < len(slot_to_bb_mapping) else {}
            for pos in range(L):
                if is_fill[b, pos].item():
                    # Restrict this position to its aligned BBs only.
                    aligned_bbs: List[int] = mapping_b.get(pos, [])
                    if aligned_bbs:
                        mask[b, pos, :] = 0.0
                        for bb_idx in aligned_bbs:
                            if 0 <= bb_idx < num_bbs:
                                mask[b, pos, bb_idx] = 1.0
                    # If no mapping exists for this FILL token, fall back
                    # to attending to all BBs (keep the ones-row).

        return mask

    # ------------------------------------------------------------------
    # Forward pass (conceptual -- requires an injected LLM)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        graph_kv: torch.Tensor,
        slot_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        slot_weight_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Conceptual forward pass through the adapter-augmented LLM.

        The method projects graph embeddings, stores them in the
        ``_graph_kv_holder`` so that the monkey-patched transformer layers can
        access them, then runs the LLM's own forward.

        Parameters
        ----------
        input_ids : torch.Tensor
            ``[B, L]`` -- tokenised skeleton + filled text.
        graph_kv : torch.Tensor
            ``[B, N, d_graph]`` -- per-node graph embeddings from
            Modules A/B.
        slot_mask : torch.Tensor
            ``[B, L, N]`` -- cross-attention mask (see
            :meth:`build_slot_mask`).
        labels : torch.Tensor or None
            ``[B, L]`` -- target token ids for CE loss.  Positions where loss
            should be ignored must be set to ``-100`` (PyTorch convention).
        slot_weight_mask : torch.Tensor or None
            ``[B, L]`` -- per-token loss weights produced by
            :class:`SlotWeightBuilder`.  If ``None`` all FILL positions are
            weighted equally at 1.0.

        Returns
        -------
        logits : torch.Tensor
            ``[B, L, V]`` -- vocabulary logits.
        loss : torch.Tensor or None
            Scalar weighted cross-entropy loss if *labels* is given, else
            ``None``.

        Raises
        ------
        RuntimeError
            If no LLM has been injected via :meth:`inject_into_llm`.
        """
        if self._llm is None:
            raise RuntimeError(
                "No LLM has been injected.  Call "
                "`decoder.inject_into_llm(llm_model)` before forward."
            )

        # ── project graph embeddings ─────────────────────────────────────
        projected_graph_kv = self.graph_proj(graph_kv)  # [B, N, d_graph]

        # ── store context for the monkey-patched layers ──────────────────
        assert self._graph_kv_holder is not None
        self._graph_kv_holder.set(projected_graph_kv, slot_mask)

        # ── run the LLM forward ──────────────────────────────────────────
        # The adapters are called inside each transformer layer's patched
        # forward (see AdapterInjector).  We pass ``labels=None`` to the
        # LLM itself so we can compute a custom weighted loss below.
        llm_outputs = self._llm(input_ids=input_ids)

        # HuggingFace CausalLM models return an object with ``logits``.
        logits: torch.Tensor = llm_outputs.logits     # [B, L, V]

        # ── compute weighted loss on FILL positions only ─────────────────
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = self._compute_weighted_loss(logits, labels, slot_weight_mask)

        return logits, loss

    # ------------------------------------------------------------------
    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        slot_weight_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute token-level cross-entropy weighted by *slot_weight_mask*.

        Parameters
        ----------
        logits : torch.Tensor
            ``[B, L, V]``
        labels : torch.Tensor
            ``[B, L]`` -- ``-100`` at positions that should be ignored.
        slot_weight_mask : torch.Tensor or None
            ``[B, L]`` -- per-token weight.  ``0.0`` for non-FILL positions.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        B, L, V = logits.shape
        # device_map 多卡：lm_head 在末卡；loss 须在 logits 同卡上算，backward 才合法。
        loss_dev = logits.device
        shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
        shift_labels = labels[:, 1:].to(loss_dev, non_blocking=True)  # [B, L-1]

        # Unreduced CE loss per token.
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, L - 1)                                # [B, L-1]

        if slot_weight_mask is not None:
            # Align weight mask with the shifted positions.
            shift_weights = slot_weight_mask[:, 1:].contiguous().to(loss_dev)  # [B, L-1]
            weighted = loss_per_token * shift_weights
            sw = shift_weights.sum()
            # augment 使用 ``--mask-non-fill-labels`` 时：监督落在槽位真值 token 上，而
            # :meth:`SlotWeightBuilder.build_slot_weights` 只在 FILL 占位符 id 上给非零
            # 权重，二者错位会导致加权项全为 0、训练 loss 恒为 0。此时回退为按 ``labels``
            # 有效位置平均（与 slot_weight_mask is None 分支一致）。
            if sw < 1e-8 or weighted.sum() == 0:
                valid = shift_labels != -100
                denom = valid.float().sum().clamp(min=1.0)
                loss = loss_per_token.sum() / denom
            else:
                denom = sw.clamp(min=1.0)
                loss = weighted.sum() / denom
        else:
            # Plain mean over non-ignored positions.
            valid = shift_labels != -100
            denom = valid.float().sum().clamp(min=1.0)
            loss = loss_per_token.sum() / denom

        return loss


# ═════════════════════════════════════════════════════════════════════════════
# 3. SlotWeightBuilder
# ═════════════════════════════════════════════════════════════════════════════

class SlotWeightBuilder:
    """Utility for constructing per-token loss weight tensors.

    Different ``FILL`` slot types receive different weights so the training
    signal emphasises semantically harder slots (e.g. branch conditions).

    Usage::

        builder = SlotWeightBuilder()
        weights = builder.build_slot_weights(
            skeleton_tokens=token_ids,   # [L] int64
            slot_types=["FILL_COND", "FILL_STMT", ...],  # per-FILL-token type
            config=module_c_config,
        )  # → [L] float32
    """

    # Canonical slot type names that can appear in a skeleton.
    KNOWN_SLOT_TYPES = frozenset({
        "FILL_COND",
        "FILL_STMT",
        "FILL_EXPR",
        "FILL_TYPE",
        "FILL_PARAM",
    })

    @staticmethod
    def build_slot_weights(
        skeleton_tokens: torch.Tensor,
        slot_types: List[str],
        config: ModuleCConfig,
        fill_token_range_start: int = 32_000,
        fill_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a per-token weight vector.

        Parameters
        ----------
        skeleton_tokens : torch.Tensor
            ``[L]`` int64 -- the full tokenised skeleton sequence (including
            both structural tokens and FILL placeholders).
        slot_types : list of str
            Ordered list of slot type names, one per FILL token found in
            *skeleton_tokens*.  Must have length equal to the number of FILL
            tokens in the sequence.
        config : ModuleCConfig
            Provides the ``slot_weights`` dict mapping slot type names to
            scalar weights.
        fill_token_range_start : int
            仅当 *fill_token_ids* 为 ``None`` 时使用：``id >=`` 此值视为 FILL（大词表上会误判）。
        fill_token_ids : torch.Tensor, optional
            形状 ``[K]`` 的 long，为 ``<|STRUCLIFT_FILL_xxxx|>`` 等**真实**占位符 id（与 augment 一致）。
            若提供则优先用 ``torch.isin`` 判定 FILL，避免 DeepSeek 等词表在 32000+ 仍有大量普通 token。

        Returns
        -------
        torch.Tensor
            ``[L]`` float32 weight vector.  Non-FILL positions are ``0.0``.

        Raises
        ------
        ValueError
            If the number of FILL tokens does not match ``len(slot_types)``.
        """
        L = skeleton_tokens.shape[0]
        device = skeleton_tokens.device

        if fill_token_ids is not None and fill_token_ids.numel() > 0:
            fid = fill_token_ids.to(device=device, dtype=torch.long).flatten()
            uniq = torch.unique(fid)
            if uniq.numel() <= 8 and fid.numel() >= 64:
                logger.warning(
                    "fill_token_ids 几乎全为重复 id（可能未对 tokenizer add_tokens 注册 "
                    "<|STRUCLIFT_FILL_xxxx|>）；请使用含 add_tokens 的 main._load_hf_llm 或 "
                    "--tokenizer-dir 指向 augment --save-tokenizer 目录。",
                )
            is_fill = torch.isin(skeleton_tokens, fid)
        else:
            is_fill = skeleton_tokens >= fill_token_range_start  # [L] bool

        num_fill_raw = int(is_fill.sum().item())
        k = len(slot_types)

        if num_fill_raw > k:
            fill_pos = torch.where(is_fill)[0]
            keep = fill_pos[:k]
            is_fill = torch.zeros(L, dtype=torch.bool, device=device)
            is_fill[keep] = True
            logger.warning(
                "FILL 位置数=%d 与 slot_types=%d 不一致，已按序列顺序保留前 %d 个；"
                "建议确认已注册占位符并与 augment 词表一致。",
                num_fill_raw,
                k,
                k,
            )

        num_fill = int(is_fill.sum().item())

        if num_fill != k:
            raise ValueError(
                f"Number of FILL tokens ({num_fill}) does not match "
                f"len(slot_types) ({len(slot_types)})."
            )

        weights = torch.zeros(L, dtype=torch.float32, device=device)
        fill_idx = 0
        weight_map: Dict[str, float] = config.slot_weights

        for pos in range(L):
            if is_fill[pos].item():
                stype = slot_types[fill_idx]
                weights[pos] = weight_map.get(stype, 1.0)
                fill_idx += 1

        return weights

    # ------------------------------------------------------------------
    @staticmethod
    def build_slot_weights_batched(
        skeleton_tokens: torch.Tensor,
        slot_types_batch: List[List[str]],
        config: ModuleCConfig,
        fill_token_range_start: int = 32_000,
        fill_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched version of :meth:`build_slot_weights`.

        Parameters
        ----------
        skeleton_tokens : torch.Tensor
            ``[B, L]`` int64.
        slot_types_batch : list of list of str
            Length-B outer list; each inner list has one entry per FILL token
            in that batch element.
        config : ModuleCConfig
            Configuration with ``slot_weights``.
        fill_token_range_start : int
            回退阈值（无 *fill_token_ids* 时）。
        fill_token_ids : torch.Tensor, optional
            与 :meth:`build_slot_weights` 相同。

        Returns
        -------
        torch.Tensor
            ``[B, L]`` float32 weight tensor.
        """
        B, L = skeleton_tokens.shape
        weights = torch.zeros(B, L, dtype=torch.float32,
                              device=skeleton_tokens.device)
        for b in range(B):
            weights[b] = SlotWeightBuilder.build_slot_weights(
                skeleton_tokens[b],
                slot_types_batch[b],
                config,
                fill_token_range_start=fill_token_range_start,
                fill_token_ids=fill_token_ids,
            )
        return weights

    @staticmethod
    def build_slot_weights_batched_for_supervised_labels(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        slot_types_batch: List[List[str]],
        config: ModuleCConfig,
        fill_token_ids: Optional[torch.Tensor] = None,
        fill_token_range_start: int = 32_000,
    ) -> torch.Tensor:
        """与 ``augment_binskel_sft --mask-non-fill-labels`` 对齐的槽位权重。

        原 :meth:`build_slot_weights` 只在 FILL 占位符 id 上给非零权重；而监督
        ``labels`` 落在紧随其后的真值 token 上。本方法按序列中 FILL 出现顺序，把
        每个槽的类型权重赋给该槽对应 span 内所有 ``labels != -100`` 的位置，
        使加权 CE 与 mask 监督一致。
        """
        B, L = input_ids.shape
        device = input_ids.device
        weights = torch.zeros(B, L, dtype=torch.float32, device=device)
        weight_map: Dict[str, float] = config.slot_weights

        if fill_token_ids is not None and fill_token_ids.numel() > 0:
            fid = fill_token_ids.to(device=device, dtype=torch.long).flatten()
            is_fill = torch.isin(input_ids, fid)
        else:
            is_fill = input_ids >= fill_token_range_start

        for b in range(B):
            slot_types = slot_types_batch[b]
            fill_pos = torch.where(is_fill[b])[0].tolist()
            if len(fill_pos) != len(slot_types):
                for t in range(L):
                    if labels[b, t] != -100:
                        weights[b, t] = 1.0
                logger.warning(
                    "FILL 数=%d 与 slot_types=%d 不一致，样本 batch=%d 槽位权重退化为全 1（仅监督位）",
                    len(fill_pos),
                    len(slot_types),
                    b,
                )
                continue

            for i, fp in enumerate(fill_pos):
                stype = slot_types[i]
                w = float(weight_map.get(stype, 1.0))
                fp_next = fill_pos[i + 1] if i + 1 < len(fill_pos) else L
                for pos in range(fp + 1, fp_next):
                    if labels[b, pos] != -100:
                        weights[b, pos] = w
        return weights


# ═════════════════════════════════════════════════════════════════════════════
# 4. _GraphKVHolder  (internal transport object)
# ═════════════════════════════════════════════════════════════════════════════

class _GraphKVHolder:
    """Thread-local-ish container that shuttles graph KV tensors and the slot
    mask from :class:`StructureConditionedDecoder` into the monkey-patched
    LLM layers installed by :class:`AdapterInjector`.

    This avoids modifying the HuggingFace model's ``forward`` signature.

    Attributes
    ----------
    graph_kv : torch.Tensor or None
        ``[B, N, d_graph]``
    slot_mask : torch.Tensor or None
        ``[B, L, N]``
    """

    def __init__(self) -> None:
        self.graph_kv: Optional[torch.Tensor] = None
        self.slot_mask: Optional[torch.Tensor] = None

    def set(
        self,
        graph_kv: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> None:
        """Store the graph context for the current forward pass.

        Parameters
        ----------
        graph_kv : torch.Tensor
            ``[B, N, d_graph]``
        slot_mask : torch.Tensor
            ``[B, L, N]``
        """
        self.graph_kv = graph_kv
        self.slot_mask = slot_mask

    def clear(self) -> None:
        """Release references after the forward pass."""
        self.graph_kv = None
        self.slot_mask = None


# ═════════════════════════════════════════════════════════════════════════════
# 5. AdapterInjector
# ═════════════════════════════════════════════════════════════════════════════

class AdapterInjector:
    """Monkey-patch helper that splices :class:`CrossAttentionAdapter` blocks
    into an existing HuggingFace causal LM **without** modifying its source.

    The injector locates the iterable of transformer layers inside the model
    (following common HuggingFace conventions), then wraps each layer's
    ``forward`` method so that the adapter's cross-attention is executed
    immediately after the layer's normal forward.

    Usage
    -----
    ::

        from transformers import AutoModelForCausalLM

        llm = AutoModelForCausalLM.from_pretrained(config.llm_name)
        holder = _GraphKVHolder()
        AdapterInjector.inject(llm, adapters, holder)

    After injection, every call to ``llm(input_ids=...)`` will automatically
    route hidden states through the adapters, provided the holder has been
    populated (see :class:`StructureConditionedDecoder.forward`).
    """

    # Common attribute paths for HuggingFace model internals.
    _LAYER_ATTR_PATHS: List[List[str]] = [
        # PEFT 包裹后：PeftModel → LoraModel.base → LlamaForCausalLM → LlamaModel.layers
        ["base_model", "model", "model", "layers"],
        ["model", "layers"],          # LLaMA / DeepSeek / Mistral（裸 *ForCausalLM）
        ["transformer", "h"],         # GPT-2 / GPT-Neo
        ["gpt_neox", "layers"],       # GPT-NeoX / Pythia
        ["model", "decoder", "layers"],  # OPT
    ]

    # ------------------------------------------------------------------
    @staticmethod
    def _find_layers(model: nn.Module) -> nn.ModuleList:
        """Walk known attribute paths to locate the transformer layer list.

        Parameters
        ----------
        model : nn.Module
            A HuggingFace ``*ForCausalLM`` model.

        Returns
        -------
        nn.ModuleList
            The list of transformer layer modules.

        Raises
        ------
        ValueError
            If no known attribute path is found.
        """
        for path in AdapterInjector._LAYER_ATTR_PATHS:
            obj: Any = model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    return obj  # type: ignore[return-value]
            except AttributeError:
                continue
        raise ValueError(
            "Could not locate transformer layers in the model.  "
            "Searched paths: "
            + ", ".join(".".join(p) for p in AdapterInjector._LAYER_ATTR_PATHS)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def inject(
        model: nn.Module,
        adapters: nn.ModuleList,
        graph_kv_holder: _GraphKVHolder,
    ) -> None:
        """Monkey-patch each transformer layer to include cross-attention.

        For layer *i*, the original ``forward`` is replaced by a wrapper that:

        1. Calls the original ``forward`` to obtain the layer output.
        2. Extracts the hidden state tensor from the output (tuple element 0).
        3. Passes it through ``adapters[i]`` with the graph KV and slot mask
           stored in *graph_kv_holder*.
        4. Re-packs the modified hidden state back into the output tuple.

        Parameters
        ----------
        model : nn.Module
            HuggingFace ``*ForCausalLM`` model.
        adapters : nn.ModuleList
            One :class:`CrossAttentionAdapter` per transformer layer.
        graph_kv_holder : _GraphKVHolder
            Shared container populated by
            :class:`StructureConditionedDecoder` before each forward pass.

        Raises
        ------
        ValueError
            If ``len(adapters)`` does not equal the number of transformer
            layers found in *model*.
        """
        layers = AdapterInjector._find_layers(model)
        if len(adapters) != len(layers):
            raise ValueError(
                f"Number of adapters ({len(adapters)}) must match the number "
                f"of transformer layers ({len(layers)})."
            )

        hook_handles: List[torch.utils.hooks.RemovableHook] = []
        for layer_idx, layer in enumerate(layers):
            handles = AdapterInjector._wrap_layer(
                layer, adapters[layer_idx], graph_kv_holder,
            )
            hook_handles.extend(handles)

        model._structlift_hook_handles = hook_handles  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Known attribute names for the pre-FFN layer-norm in common architectures.
    _PRE_FFN_LN_ATTRS: List[str] = [
        "post_attention_layernorm",  # LLaMA / DeepSeek / Mistral / Qwen
        "ln_2",                      # GPT-2 / GPT-Neo
        "final_layer_norm",          # OPT
    ]

    @staticmethod
    def _find_pre_ffn_ln(layer: nn.Module) -> Optional[nn.Module]:
        """Locate the layer-norm that sits *between* self-attention and FFN."""
        for attr in AdapterInjector._PRE_FFN_LN_ATTRS:
            ln = getattr(layer, attr, None)
            if ln is not None:
                return ln
        return None

    @staticmethod
    def _adapt_slot_mask(
        slot_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        """Truncate or pad *slot_mask* to match the current hidden-state
        sequence length (handles KV-cache generation)."""
        if slot_mask is None:
            return None
        mask_seq_len = slot_mask.shape[1]
        if mask_seq_len > seq_len:
            return slot_mask[:, -seq_len:, :]
        if mask_seq_len < seq_len:
            pad = torch.ones(
                slot_mask.shape[0],
                seq_len - mask_seq_len,
                slot_mask.shape[2],
                dtype=slot_mask.dtype,
                device=slot_mask.device,
            )
            return torch.cat([slot_mask, pad], dim=1)
        return slot_mask

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_layer(
        layer: nn.Module,
        adapter: CrossAttentionAdapter,
        holder: _GraphKVHolder,
    ) -> List[Any]:
        """Inject the cross-attention adapter **between** self-attention and
        FFN (§7.2.2: Self-Attn → [Adapter] → FFN).

        For architectures with a recognisable pre-FFN layer-norm (LLaMA,
        DeepSeek, GPT-2, OPT, …), a ``register_forward_pre_hook`` on that
        layer-norm inserts the adapter at exactly the right point.

        Fallback: if the pre-FFN layer-norm is not found, the adapter is
        appended after the full layer output.

        Parameters
        ----------
        layer : nn.Module
            A single transformer decoder layer.
        adapter : CrossAttentionAdapter
            The adapter to inject.
        holder : _GraphKVHolder
            Shared graph-KV container.

        Returns
        -------
        list
            Hook handles (for removal). Empty list when fallback is used.
        """
        pre_ffn_ln = AdapterInjector._find_pre_ffn_ln(layer)

        if pre_ffn_ln is not None:
            # ── Mid-layer injection (proposal §7.2.2) ────────────────────
            # The pre-FFN layer-norm receives hidden states that have
            # already been through self-attention + residual.  By hooking
            # *before* this layer-norm, we insert the adapter at exactly
            # the Self-Attn → [Adapter] → FFN boundary.
            def _pre_hook(
                _module: nn.Module,
                args: Tuple[Any, ...],
                _adapter: CrossAttentionAdapter = adapter,
                _holder: _GraphKVHolder = holder,
            ) -> Tuple[Any, ...]:
                hidden_states = args[0]
                if _holder.graph_kv is not None:
                    sm = AdapterInjector._adapt_slot_mask(
                        _holder.slot_mask, hidden_states.shape[1],
                    )
                    hidden_states = _adapter(
                        hidden_states, _holder.graph_kv, sm,
                    )
                return (hidden_states,) + args[1:]

            handle = pre_ffn_ln.register_forward_pre_hook(_pre_hook)
            return [handle]
        else:
            # ── Fallback: post-layer injection ───────────────────────────
            original_forward: Callable[..., Any] = layer.forward

            def _patched_forward(
                *args: Any,
                _orig_fn: Callable[..., Any] = original_forward,
                _adapter: CrossAttentionAdapter = adapter,
                _holder: _GraphKVHolder = holder,
                **kwargs: Any,
            ) -> Any:
                output = _orig_fn(*args, **kwargs)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if _holder.graph_kv is not None:
                    sm = AdapterInjector._adapt_slot_mask(
                        _holder.slot_mask, hidden_states.shape[1],
                    )
                    hidden_states = _adapter(
                        hidden_states, _holder.graph_kv, sm,
                    )

                if isinstance(output, tuple):
                    output = (hidden_states,) + output[1:]
                else:
                    output = hidden_states
                return output

            layer.forward = _patched_forward  # type: ignore[assignment]
            return []

    # ------------------------------------------------------------------
    @staticmethod
    def remove(model: nn.Module) -> None:
        """Undo injection by removing hooks and restoring ``forward`` methods.

        Parameters
        ----------
        model : nn.Module
            The previously patched model.
        """
        # 1. Remove registered hooks (mid-layer injection path).
        handles = getattr(model, "_structlift_hook_handles", None)
        if handles:
            for h in handles:
                h.remove()
            model._structlift_hook_handles = []  # type: ignore[attr-defined]

        # 2. Restore monkey-patched layer forwards (fallback path).
        try:
            layers = AdapterInjector._find_layers(model)
        except ValueError:
            return

        for layer in layers:
            fwd = getattr(layer, "forward", None)
            if fwd is None:
                continue
            closure = getattr(fwd, "__closure__", None)
            if closure is None:
                continue
            for cell in closure:
                try:
                    val = cell.cell_contents
                    if callable(val) and val is not fwd:
                        if hasattr(val, "__self__") and val.__self__ is layer:
                            layer.forward = val  # type: ignore[assignment]
                            break
                except ValueError:
                    continue


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    "CrossAttentionAdapter",
    "StructureConditionedDecoder",
    "SlotWeightBuilder",
    "AdapterInjector",
]

"""
Stage 4 — Structural Consistency Reinforcement Learning.

Uses GRPO (Group Relative Policy Optimisation) to fine-tune Module C
with three reward signals:
  1. Compilation reward   (gcc -fsyntax-only)
  2. Structural reward    (pattern distribution + tree edit distance + nesting depth)
  3. Semantic reward      (test-case execution or edit similarity)

与 StrucLift_Research_Proposal §8.3 对齐的**数据流**说明：

- **Prompt（含骨架）**：来自训练 jsonl 中预计算的 ``sft_input_ids``（由
  ``scripts/augment_binskel_sft.py`` 根据二进制 CFG + ``pattern_labels`` 生成骨架，
  与 DWARF/对齐标签一起构造；**不是**在 RL 循环内从零现场拼骨架字符串）。
- **图条件 KV**：每个 batch 用 ``encode(..., run_alignment=True)`` 跑 **Module A+B**，
  将 ``binary_node_embs`` 拼成 ``graph_kv`` 供 cross-attn（对应提案「增强二进制表示」）。
- **采样与 GRPO**：``Module C`` 对同一条 prompt 采样 G 个续写，按 §8.2 奖励更新策略。

Frozen: Module A；Module B 默认冻结，若 ``module_d.finetune_module_b_cross_attn`` 则仅
``cross_attn`` 以 ``module_b_lr_ratio`` 倍学习率更新。
Trained: Module C adapters + LoRA（主）；可选 B 的 cross-attn。
Reference model: frozen copy of the Stage 3 SFT checkpoint（常放 CPU 省显存）。
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.models.struclift import StrucLift, batched_graph_kv_from_nodes
from struclift.models.module_d import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)


def extend_slot_mask(slot_mask: torch.Tensor, new_seq_len: int) -> torch.Tensor:
    """Pad with ones so new token positions may attend to all graph nodes."""
    B, L, N = slot_mask.shape
    if new_seq_len <= L:
        return slot_mask[:, :new_seq_len, :]
    pad = torch.ones(
        B, new_seq_len - L, N,
        dtype=slot_mask.dtype,
        device=slot_mask.device,
    )
    return torch.cat([slot_mask, pad], dim=1)


def decoder_completion_log_prob(
    decoder: nn.Module,
    prompt_ids: torch.Tensor,
    graph_kv: torch.Tensor,
    slot_mask: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """Sum of log π(token | prefix) over generated continuation tokens only."""
    d_dev = next(decoder.parameters()).device
    if completion_ids.numel() == 0:
        return torch.zeros((), device=d_dev, dtype=torch.float32)
    prompt_ids = prompt_ids.to(d_dev)
    graph_kv = graph_kv.to(d_dev)
    slot_mask = slot_mask.to(d_dev)
    completion_ids = completion_ids.to(d_dev)
    full = torch.cat([prompt_ids, completion_ids], dim=1)
    sm = extend_slot_mask(slot_mask, full.size(1))
    projected = decoder.graph_proj(graph_kv)
    holder = decoder._graph_kv_holder
    if holder is None:
        raise RuntimeError("decoder._graph_kv_holder 未初始化（须先 inject_into_llm）")
    holder.set(projected, sm)
    try:
        logits, _ = decoder(
            input_ids=full,
            graph_kv=graph_kv,
            slot_mask=sm,
            labels=None,
            slot_weight_mask=None,
        )
    finally:
        holder.clear()
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = full[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    p_len = prompt_ids.size(1)
    start = max(0, p_len - 1)
    return gathered[:, start:].sum()


class Stage4Trainer:
    """RL trainer for structural consistency reinforcement.

    Parameters
    ----------
    model : StrucLift
        Policy model (Module C with LLM, from Stage 3 SFT checkpoint).
    config : StrucLiftConfig
    device : torch.device
    tokenizer
        HuggingFace tokenizer，用于 ``decode`` 与 ``pad_token_id``；可为 None。
    max_new_tokens
        每个候选的最大续写长度。
    """

    def __init__(
        self,
        model: StrucLift,
        config: StrucLiftConfig,
        device: torch.device = torch.device("cpu"),
        tokenizer: Optional[Any] = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        dc = config.module_d

        # Freeze Module A；B 整模块先冻结，§8.3.2 步 7 可选仅解冻 cross-attn
        model.freeze_encoders()
        model.freeze_alignment()

        self._finetune_b = dc.finetune_module_b_cross_attn
        if self._finetune_b:
            for p in model.alignment_module.cross_attn.parameters():
                p.requires_grad = True

        # ── create reference model (frozen copy of the SFT checkpoint) ───
        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()
        # 与 policy 各放一份 6.7B 级权重会占满 GPU；ref 仅用于无梯度 log π_ref，放 CPU。
        if device.type == "cuda":
            self.ref_model = self.ref_model.cpu()
            logger.info(
                "Stage4: reference model on CPU (saves ~1× LLM VRAM; ref forward is slower)"
            )

        # ── GRPO config ─────────────────────────────────────────────────
        grpo_config = GRPOConfig(
            num_candidates=dc.num_candidates,
            temperature=dc.temperature,
            top_p=dc.top_p,
            kl_beta=dc.kl_beta,
            lr=dc.rl_lr,
            max_grad_norm=dc.max_grad_norm,
            alpha_compile=dc.alpha_compile,
            beta_struct=dc.beta_struct,
            gamma_semantic=dc.gamma_semantic,
        )

        # ── GRPO trainer wraps policy and reference ──────────────────────
        param_groups = [
            {"params": [p for p in model.decoder.parameters()
                        if p.requires_grad],
             "lr": dc.rl_lr},
        ]
        if self._finetune_b:
            param_groups.append({
                "params": list(model.alignment_module.cross_attn.parameters()),
                "lr": dc.rl_lr * dc.module_b_lr_ratio,
            })
        optimizer = torch.optim.AdamW(param_groups, lr=dc.rl_lr)
        self.grpo = GRPOTrainer(
            policy_model=model,
            ref_model=self.ref_model,
            config=grpo_config,
            optimizer=optimizer,
        )

    # ------------------------------------------------------------------
    def _ensure_graph_kv_slot_mask(
        self,
        batch: Dict[str, Any],
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """若 batch 未带 ``graph_kv``，则用冻结的 A+B 编码构造 ``[B,N,d]`` KV。"""
        B, L = input_ids.shape
        if batch.get("graph_kv") is not None:
            graph_kv = batch["graph_kv"]
            slot_mask = batch.get("slot_mask")
            if slot_mask is None:
                slot_mask = torch.ones(
                    B, L, graph_kv.size(1),
                    device=input_ids.device,
                    dtype=torch.float32,
                )
            return graph_kv, slot_mask

        with torch.no_grad():
            enc = self.model.encode(
                bin_opcodes=batch["bin_opcodes"],
                bin_operand_types=batch["bin_operand_types"],
                bin_operand_values=batch["bin_operand_values"],
                bin_block_lengths=batch["bin_block_lengths"],
                bin_edge_index=batch["bin_edge_index"],
                bin_edge_type=batch["bin_edge_type"],
                bin_struct_features=batch["bin_struct_features"],
                bin_batch=batch.get("bin_batch"),
                src_token_ids=batch.get("src_token_ids"),
                src_stmt_lengths=batch.get("src_stmt_lengths"),
                src_edge_index=batch.get("src_edge_index"),
                src_edge_type=batch.get("src_edge_type"),
                src_ast_types=batch.get("src_ast_types"),
                src_compound_id=batch.get("src_compound_id"),
                src_batch=batch.get("src_batch"),
                bin_adj=batch.get("adj_bin"),
                src_adj=batch.get("adj_src_gt"),
                M_gt=batch.get("M_gt"),
                run_alignment=True,
            )
        bb = batch.get("bin_batch")
        if bb is None:
            n = enc.binary_node_embs.size(0)
            bb = torch.zeros(
                n, dtype=torch.long, device=input_ids.device,
            )
        graph_kv = batched_graph_kv_from_nodes(
            enc.binary_node_embs, bb, batch_size=B,
        )
        slot_mask = torch.ones(
            B, L, graph_kv.size(1),
            device=input_ids.device,
            dtype=torch.float32,
        )
        return graph_kv, slot_mask

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_candidates(
        self,
        input_ids: torch.Tensor,
        graph_kv: torch.Tensor,
        slot_mask: torch.Tensor,
        num_candidates: int,
    ) -> List[Dict[str, Any]]:
        """使用 ``llm.generate``（temperature / top_p）采样 G 个候选；含 cross-attention 上下文。

        Returns
        -------
        list of dict with keys ``text`` (str) and ``completion_ids`` (``[1, Lc]`` long).
        """
        llm = self.model.decoder._llm
        dec = self.model.decoder
        out_list: List[Dict[str, Any]] = []
        if llm is None:
            raise RuntimeError(
                "Stage4 RL requires an injected LLM for candidate generation. "
                "Call model.decoder.inject_into_llm(llm) before training."
            )

        dc = self.config.module_d
        projected = dec.graph_proj(graph_kv)
        holder = dec._graph_kv_holder
        if holder is None:
            raise RuntimeError("decoder 未 inject LLM，无法 generate")

        pad_id = 0
        if self.tokenizer is not None:
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            if pad_id is None:
                pad_id = 0

        for _ in range(num_candidates):
            holder.set(projected, slot_mask)
            try:
                try:
                    from transformers import GenerationConfig

                    gcfg = GenerationConfig(
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=dc.temperature,
                        top_p=dc.top_p,
                        pad_token_id=pad_id,
                    )
                    out = llm.generate(
                        input_ids=input_ids,
                        generation_config=gcfg,
                    )
                except ImportError:
                    out = llm.generate(
                        input_ids=input_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=dc.temperature,
                        top_p=dc.top_p,
                        pad_token_id=pad_id,
                    )
            finally:
                holder.clear()

            gen_only = out[:, input_ids.size(1):]
            if self.tokenizer is not None:
                text = self.tokenizer.decode(
                    gen_only[0], skip_special_tokens=True,
                )
            else:
                text = " ".join(str(int(x)) for x in gen_only[0].tolist())
            out_list.append({"text": text, "completion_ids": gen_only})

        return out_list

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one RL epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Each batch yields a dict with:
            - all encoder fields
            - ``input_ids`` : [1, L]  skeleton prompt (one sample at a time)
            - ``graph_kv`` : [1, N, d_graph]
            - ``slot_mask`` : [1, L, N]
            - ``bin_pattern_dist`` : [C]  pattern distribution of binary
            - ``bin_cst_serialised`` : List[str]  CST pre-order
            - ``bin_nesting_depth`` : int
            - ``test_cases`` : List[Tuple[str, str]], optional
            - ``reference_code`` : str, optional

        Returns
        -------
        dict with loss/reward metrics
        """
        self.model.train()
        self.model.binary_encoder.eval()
        self.model.source_encoder.eval()
        self.model.alignment_module.eval()
        if self._finetune_b:
            self.model.alignment_module.cross_attn.train()

        accum = {"loss": 0.0, "mean_reward": 0.0, "kl": 0.0}
        n_batches = 0
        dc = self.config.module_d

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            input_ids = batch["input_ids"]
            graph_kv, slot_mask = self._ensure_graph_kv_slot_mask(batch, input_ids)
            B = input_ids.size(0)

            raw_cst = batch.get("bin_cst_serialised_batch")
            if raw_cst:
                cst_batch = raw_cst
            else:
                cst_batch = [[] for _ in range(B)]
            pat_dist = batch["bin_pattern_dist"]
            depths = batch["bin_nesting_depth"]

            for b in range(B):
                ids_b = input_ids[b : b + 1]
                gkv_b = graph_kv[b : b + 1]
                sm_b = slot_mask[b : b + 1]

                cand_entries = self._generate_candidates(
                    ids_b, gkv_b, sm_b,
                    num_candidates=dc.num_candidates,
                )
                texts = [c["text"] for c in cand_entries]

                rewards = self.grpo.compute_rewards_for_candidates(
                    candidates=texts,
                    bin_pattern_dist=pat_dist[b],
                    bin_cst_serialised=(
                        cst_batch[b] if b < len(cst_batch) else []
                    ),
                    bin_nesting_depth=int(depths[b].item()),
                    test_cases=batch.get("test_cases"),
                    reference_code=batch.get("reference_code"),
                    num_opcodes=self.config.vocab.num_opcodes,
                ).to(self.device)

                G = dc.num_candidates
                self.ref_model.eval()
                ref_lp_list: List[torch.Tensor] = []
                for g in range(G):
                    comp = cand_entries[g]["completion_ids"]
                    with torch.no_grad():
                        rlp = decoder_completion_log_prob(
                            self.ref_model.decoder,
                            ids_b, gkv_b, sm_b, comp,
                        )
                    ref_lp_list.append(rlp)
                ref_log_probs = torch.stack(ref_lp_list).detach().to(self.device)

                def _policy_lp(g: int) -> torch.Tensor:
                    comp = cand_entries[g]["completion_ids"]
                    return decoder_completion_log_prob(
                        self.model.decoder,
                        ids_b, gkv_b, sm_b, comp,
                    )

                step_metrics = self.grpo.step_with_policy_forward(
                    rewards=rewards,
                    ref_log_probs=ref_log_probs,
                    policy_log_prob_fn=_policy_lp,
                )

                for k in accum:
                    if k in step_metrics:
                        accum[k] += step_metrics[k]
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = "checkpoints/stage4",
    ) -> None:
        """Full Stage 4 RL training loop."""
        os.makedirs(save_dir, exist_ok=True)
        best_reward = -float("inf")
        dc = self.config.module_d

        for epoch in range(1, dc.rl_epochs + 1):
            metrics = self.train_epoch(train_loader)
            logger.info(
                f"[Stage4] Epoch {epoch}/{dc.rl_epochs}  "
                f"loss={metrics['loss']:.4f}  "
                f"reward={metrics['mean_reward']:.4f}  "
                f"kl={metrics['kl']:.4f}"
            )

            if metrics["mean_reward"] > best_reward:
                best_reward = metrics["mean_reward"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, "best_stage4.pt"),
                )

        torch.save(
            self.model.state_dict(),
            os.path.join(save_dir, "final_stage4.pt"),
        )
        logger.info(f"[Stage4] RL complete. Best reward: {best_reward:.4f}")

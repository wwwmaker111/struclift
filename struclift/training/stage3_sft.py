"""
Stage 3 — Supervised Fine-Tuning (SFT).

Trains Module C (Structure-Conditioned LLM Decoder):
  - Cross-attention adapters injected into each LLM layer
  - LoRA on Q/K/V/O projections
  - Slot-weighted cross-entropy loss on FILL positions
  - §7.5 / §9.4 可微的 L_compile 辅助损失（``s3_compile_loss_mode``）

Frozen: Module A (encoders), Module B (alignment).
Trained: Module C adapters + LoRA weights.

Scheduled sampling（§7.5.3）：**每个 epoch 内**前 ``s3_scheduled_sampling_start`` 比例的
batch 100% teacher forcing；后 ``1-start`` 比例内 ``p_sample`` 从 0 线性增至
``s3_scheduled_sampling_max_p``（按 epoch 内 batch 下标，避免早停永远碰不到 ss）。

默认每个 train/val epoch 结束只打一行总墙钟（``wall_s`` / ``avg_s_per_batch``）。
调试逐 batch 耗时：``export STRUCLIFT_STAGE3_LOG_STEP_MS=1``（勿在正式长跑中开启）。

L_compile（提案 §7.5 / §9.4）的三种 mode：

* ``reward_weighted_ce``（默认）— ``mean_b (1 - R_b) * L_fill_b``，可微，编译失败的样本拿更大权重。
* ``reinforce`` — 在 FILL 位置从 logits 采样得到 ``y_sample``，整段解码后跑 ``compile_reward``，
  用 ``-(R - baseline) * log π(y_sample)`` 作为 surrogate（baseline 用 EMA）。
* ``off`` — 兼容旧版，仅累加一个无梯度常量到总 loss 上，不反传。
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.models.struclift import StrucLift
from struclift.models.module_c import SlotWeightBuilder

logger = logging.getLogger(__name__)

# 改 optimizer 逻辑时递增，便于确认服务器已同步本文件（启动日志会打印）
STAGE3_OPTIMIZER_REV = "2025-05-19-per-param-adamw"


def _env_truthy(key: str) -> bool:
    v = os.environ.get(key, "").strip().lower()
    return v in ("1", "true", "yes", "on")


class Stage3Trainer:
    """Supervised fine-tuning trainer for Module C.

    Parameters
    ----------
    model : StrucLift
        Must have LLM injected via ``model.decoder.inject_into_llm(llm)``.
    config : StrucLiftConfig
    device : torch.device
    """

    def __init__(
        self,
        model: StrucLift,
        config: StrucLiftConfig,
        device: torch.device = torch.device("cpu"),
        fill_token_ids: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
        *,
        decoder_device_map_auto: bool = False,
    ) -> None:
        self.decoder_device_map_auto = bool(decoder_device_map_auto)
        if decoder_device_map_auto:
            # LLM 已按 hf device_map 分片：`model.to(cuda)` 会破坏占位；A/B encoder 已由调用方放 cuda:0。
            self.model = model
        else:
            self.model = model.to(device)
        self.config = config
        self.device = device
        tc = config.training

        # device_map 分片 + GradScaler fp16 常在 backward 触发 CUDA invalid argument
        self.fp16 = bool(tc.fp16) and device.type == "cuda"
        if decoder_device_map_auto and self.fp16:
            logger.warning(
                "decoder_device_map_auto 已自动关闭 AMP fp16/GradScaler（多卡分片 backward 不稳定）；"
                "基座权重仍为 bf16。",
            )
            self.fp16 = False

        # ``<|STRUCLIFT_FILL_xxxx|>`` 的 id 表（与 augment 一致）；None 时回退为 id>=32000（大词表易误判）
        self.fill_token_ids = fill_token_ids
        # §9.4 compile 辅助 loss 解码用；与 HF tokenizer 一致
        self.tokenizer = tokenizer

        # Freeze Modules A and B
        model.freeze_encoders()
        model.freeze_alignment()

        # ``device_map`` 时每 (device, dtype) 一个 AdamW，避免 PyTorch foreach 跨卡/跨 dtype 批量 step
        self._optimizers: List[torch.optim.Optimizer] = []
        self._init_optimizer()

        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 else None

        self.slot_weight_builder = SlotWeightBuilder()

        # §9.4: L_fill + λ₆·L_compile
        self.lambda_compile = tc.s3_lambda_compile
        self.compile_loss_mode = str(tc.s3_compile_loss_mode).lower()
        self.compile_reinforce_entropy_coef = float(
            tc.s3_compile_reinforce_entropy_coef
        )
        self._compile_baseline: Optional[float] = None
        self._compile_baseline_momentum = float(
            tc.s3_compile_reinforce_baseline_momentum
        )

        # scheduled sampling config (§7.5.3 — by global train step)
        self.ss_start = float(tc.s3_scheduled_sampling_start)
        self.ss_max_p = float(tc.s3_scheduled_sampling_max_p)
        self.total_epochs = tc.s3_epochs
        self.global_step = 0
        self.total_train_steps: Optional[int] = None
        self.early_stop_patience = int(getattr(tc, "s3_early_stop_patience", 0) or 0)
        self.early_stop_min_epochs = int(getattr(tc, "s3_early_stop_min_epochs", 2))
        self.early_stop_min_delta = float(getattr(tc, "s3_early_stop_min_delta", 1e-4))

    _CKPT_VERSION = 2

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _init_optimizer(self) -> None:
        """新建 AdamW。``device_map_auto`` 时按 (device, dtype) 拆多个 optimizer（彻底避开 foreach）。"""
        tc = self.config.training
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("Stage3: 无可训练参数（检查 LoRA / adapter 是否 requires_grad）")

        if getattr(self, "decoder_device_map_auto", False):
            # 每个 Parameter 单独一个 AdamW，强制走 _single_tensor_adamw（避免 foreach 跨卡/跨 dtype）
            self._optimizers = [
                torch.optim.AdamW(
                    [p],
                    lr=tc.s3_lr,
                    weight_decay=tc.s3_weight_decay,
                    foreach=False,
                    fused=False,
                )
                for p in trainable
            ]
            self.optimizer = self._optimizers[0]
            logger.info(
                "[Stage3] %s | device_map: %d×单参数 AdamW（foreach=False）",
                STAGE3_OPTIMIZER_REV,
                len(self._optimizers),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=tc.s3_lr,
                weight_decay=tc.s3_weight_decay,
            )
            self._optimizers = [self.optimizer]
            n_tensors = len(trainable)
            n_params = sum(int(p.numel()) for p in trainable)
            logger.info(
                "[Stage3] %s | 单 AdamW trainable_tensors=%d trainable_params=%d",
                STAGE3_OPTIMIZER_REV,
                n_tensors,
                n_params,
            )
        if not self._optimizers:
            raise RuntimeError("Stage3: optimizer list is empty after initialization")

    def _prepare_grads_for_optimizer(self) -> None:
        """``device_map`` 下反传常为 fp32 grad + bf16 param，须在 step 前对齐。"""
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if p.grad.device != p.device:
                p.grad = p.grad.to(device=p.device, non_blocking=True)
            if p.grad.dtype != p.dtype:
                p.grad = p.grad.to(dtype=p.dtype, non_blocking=True)

    def _optimizer_zero_grad(self) -> None:
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=True)

    def _optimizer_step(self) -> None:
        if not self._optimizers:
            raise RuntimeError("Stage3: optimizer list is empty before optimizer step")
        if getattr(self, "decoder_device_map_auto", False):
            self._prepare_grads_for_optimizer()
        if _env_truthy("STRUCLIFT_STAGE3_MANUAL_SGD"):
            lr = float(self.config.training.s3_lr)
            for p in self.model.parameters():
                if p.grad is not None and p.requires_grad:
                    p.data.add_(p.grad, alpha=-lr)
            return
        for opt in self._optimizers:
            opt.step()

    def save_checkpoint(
        self,
        path: str,
        *,
        epoch: int,
        best_loss: float,
        epochs_without_improve: int,
        global_step: Optional[int] = None,
        total_train_steps: Optional[int] = None,
    ) -> None:
        """完整 Stage3 状态（模型 + 优化器 + scaler + epoch），供 --resume 续训。"""
        gs = int(self.global_step if global_step is None else global_step)
        tts = self.total_train_steps if total_train_steps is None else total_train_steps
        obj = {
            "version": self._CKPT_VERSION,
            "model": self._unwrap_model().state_dict(),
            "optimizer": (
                [o.state_dict() for o in self._optimizers]
                if len(self._optimizers) > 1
                else self.optimizer.state_dict()
            ),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "epochs_without_improve": int(epochs_without_improve),
            "global_step": gs,
            "total_train_steps": int(tts) if tts is not None else None,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        save_dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(save_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".",
            suffix=".tmp",
            dir=save_dir,
        )
        os.close(fd)
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def load_full_checkpoint(self, path: str) -> Dict[str, Any]:
        """恢复完整训练状态。兼容仅含 ``state_dict`` 的旧 ``best_stage3.pt``。"""
        try:
            obj = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location=self.device)

        result: Dict[str, Any] = {
            "start_epoch": 1,
            "best_loss": float("inf"),
            "epochs_without_improve": 0,
            "global_step": 0,
            "total_train_steps": None,
            "is_legacy": False,
        }
        is_new = (
            isinstance(obj, dict)
            and "model" in obj
            and "optimizer" in obj
            and "epoch" in obj
        )
        if is_new:
            missing, unexpected = self._unwrap_model().load_state_dict(
                obj["model"], strict=False,
            )
            skip_opt = bool(getattr(self, "decoder_device_map_auto", False))
            if skip_opt:
                logger.warning(
                    "[Stage3] decoder_device_map_auto：跳过 optimizer/scaler 恢复，"
                    "仅载入 model 权重并重建 AdamW（避免跨卡 optimizer state 设备不一致）",
                )
                self._init_optimizer()
            else:
                try:
                    self.optimizer.load_state_dict(obj["optimizer"])
                except Exception as e:
                    logger.warning(
                        "[Stage3] optimizer state load 失败: %s；将重建 AdamW",
                        e,
                    )
                    self._init_optimizer()
                if self.scaler is not None and obj.get("scaler") is not None:
                    try:
                        self.scaler.load_state_dict(obj["scaler"])
                    except Exception as e:
                        logger.warning("[Stage3] scaler state load 失败: %s", e)
            result["start_epoch"] = int(obj["epoch"]) + 1
            result["best_loss"] = float(obj.get("best_loss", float("inf")))
            result["epochs_without_improve"] = int(
                obj.get("epochs_without_improve", 0),
            )
            if "global_step" in obj:
                result["global_step"] = int(obj["global_step"])
            if obj.get("total_train_steps") is not None:
                result["total_train_steps"] = int(obj["total_train_steps"])
            rng = obj.get("rng")
            if isinstance(rng, dict):
                try:
                    torch.set_rng_state(rng["torch"])
                    if torch.cuda.is_available() and rng.get("cuda") is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception as e:
                    logger.warning("[Stage3] RNG 恢复失败: %s", e)
            logger.info(
                "[Stage3] 已恢复完整 ckpt → 下轮 epoch=%d global_step=%d "
                "best_loss=%.6f no_improve=%d missing=%d unexpected=%d",
                result["start_epoch"],
                result.get("global_step", 0),
                result["best_loss"],
                result["epochs_without_improve"],
                len(missing),
                len(unexpected),
            )
            return result

        if isinstance(obj, dict) and "model" in obj:
            state = obj["model"]
        else:
            state = obj
        self._unwrap_model().load_state_dict(state, strict=False)
        result["is_legacy"] = True
        logger.warning(
            "[Stage3] legacy 权重 ckpt（无 optimizer/epoch），从 epoch=1 重新训",
        )
        return result

    # ------------------------------------------------------------------
    def _scheduled_sampling_prob_for_step(self, step: int, total_steps: int) -> float:
        """§7.5.3：按**当前 epoch 内** batch 下标计算 ``p_sample``。

        - ``step / total_steps < ss_start`` → 0（100% teacher forcing）
        - 否则在剩余 ``(1 - ss_start)`` 步内从 0 线性增至 ``ss_max_p``

        *step* 为 epoch 内 0-based batch 索引；*total_steps* = ``len(train_loader)``。
        """
        if self.ss_max_p <= 0.0 or total_steps <= 0:
            return 0.0
        step = max(0, min(int(step), total_steps - 1))
        frac = float(step) / float(total_steps)
        if frac < self.ss_start:
            return 0.0
        tail = max(1e-8, 1.0 - self.ss_start)
        progress = (frac - self.ss_start) / tail
        return min(self.ss_max_p * progress, self.ss_max_p)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_scheduled_sampling(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch: Dict,
        slot_weight_mask: Optional[torch.Tensor],
        ss_p: float,
    ) -> torch.Tensor:
        """Replace teacher-forced tokens with model predictions at rate *ss_p*.

        For each position ``t`` where ``labels[t] != -100`` (supervised),
        with probability *ss_p* the input at position ``t+1`` is replaced
        by the model's argmax prediction at position ``t``.  The original
        *labels* tensor is unchanged so the loss still supervises against
        the ground truth.

        Returns the (possibly mixed) ``input_ids`` tensor.
        """
        self.model.eval()
        with torch.amp.autocast("cuda", enabled=self.fp16):
            out = self.model(
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
                input_ids=input_ids,
                graph_kv=batch.get("graph_kv"),
                slot_mask=batch.get("slot_mask"),
                labels=None,
                slot_weight_mask=None,
            )
        self.model.train()
        self.model.binary_encoder.eval()
        self.model.source_encoder.eval()
        self.model.alignment_module.eval()

        if out.logits is None:
            return input_ids

        pred_ids = out.logits.argmax(dim=-1)  # [B, L]

        B, L = input_ids.shape
        mixed = input_ids.clone()
        supervised = labels != -100  # [B, L]
        coin = torch.rand(B, L, device=input_ids.device)
        replace_mask = (coin < ss_p) & supervised

        # Shift: prediction at position t fills input at position t+1
        if L > 1:
            mixed[:, 1:] = torch.where(
                replace_mask[:, :-1],
                pred_ids[:, :-1],
                mixed[:, 1:],
            )

        return mixed

    # ------------------------------------------------------------------
    # §7.5 / §9.4 — L_compile 辅助损失（多模式实现）
    # ------------------------------------------------------------------
    def _per_sample_fill_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        slot_weight_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """返回 ``[B]`` 的样本级加权 fill CE，与 :meth:`StructureConditionedDecoder._compute_weighted_loss` 数值口径一致。"""
        B, L, V = logits.shape
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, L - 1)

        valid_b = (shift_labels != -100).float().sum(dim=1).clamp(min=1.0)
        unweighted_per_sample = loss_per_token.sum(dim=1) / valid_b

        if slot_weight_mask is None:
            return unweighted_per_sample

        shift_weights = slot_weight_mask[:, 1:].contiguous()
        sw_b = shift_weights.sum(dim=1)
        weighted_per_sample = (
            (loss_per_token * shift_weights).sum(dim=1) / sw_b.clamp(min=1e-8)
        )
        return torch.where(sw_b >= 1e-8, weighted_per_sample, unweighted_per_sample)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _greedy_rewards(
        self,
        logits: torch.Tensor,
        tokenizer: Any,
    ) -> torch.Tensor:
        """对 batch 内每个样本 greedy 解码并跑 ``compile_reward`` → ``[B]`` 张量。"""
        from struclift.models.module_d import compile_reward

        pred_ids = logits.argmax(dim=-1)
        rewards: List[float] = []
        for b in range(pred_ids.size(0)):
            text = tokenizer.decode(pred_ids[b], skip_special_tokens=True)
            rewards.append(float(compile_reward(text)))
        return torch.tensor(
            rewards, device=logits.device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    def _compile_loss_reward_weighted_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        slot_weight_mask: Optional[torch.Tensor],
        tokenizer: Any,
    ) -> Optional[torch.Tensor]:
        """``L_compile`` = ``mean_b (1 - R_b) * L_fill_b``。

        - ``R_b`` 来自 *无梯度* 的 greedy 解码 + gcc，作为标量权重。
        - ``L_fill_b`` 是可微的，乘 ``(1 - R_b)`` 后整体可反传到 adapter+LoRA。
        - 等价于「编译越差的样本，CE 权重越大」，与提案 §7.5/§9.4 的辅助 loss 精神一致。
        """
        R = self._greedy_rewards(logits, tokenizer).to(logits.device)
        per_sample = self._per_sample_fill_loss(logits, labels, slot_weight_mask)
        weight = (1.0 - R).detach()
        return (weight * per_sample).mean()

    # ------------------------------------------------------------------
    def _compile_loss_reinforce(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        slot_weight_mask: Optional[torch.Tensor],
        tokenizer: Any,
    ) -> Optional[torch.Tensor]:
        """REINFORCE 风格的可微 ``L_compile``（与 Stage4 GRPO 同源、单候选简化）。

        步骤
        ----
        1. 在 FILL 位置（``labels != -100``）从 ``softmax(logits)`` **采样** ``y_sample``；
        2. 用 ``input_ids`` 模板替换 FILL 段为 ``y_sample`` 并解码 → ``compile_reward`` → ``R_b``；
        3. surrogate loss = ``- mean_b (R_b - baseline) * sum_t log π(y_sample_{b,t})``，
           其中 ``baseline`` 用 EMA 平滑；可选 ``+ entropy_coef * H(π)``。
        """
        from struclift.models.module_d import compile_reward

        B, L, V = logits.shape
        if L <= 1:
            return None

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        supervised = shift_labels != -100  # [B, L-1]
        if supervised.sum() == 0:
            return None

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        with torch.no_grad():
            probs = log_probs.exp()
            sampled = torch.distributions.Categorical(probs=probs).sample()  # [B, L-1]

        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
        )

        with torch.no_grad():
            y_text = input_ids.clone()
            y_text[:, 1:] = torch.where(
                supervised,
                sampled,
                y_text[:, 1:],
            )
            rewards: List[float] = []
            for b in range(B):
                seq = y_text[b].tolist()
                if pad_id in seq:
                    seq = [t for t in seq if t != pad_id]
                text = tokenizer.decode(seq, skip_special_tokens=True)
                rewards.append(float(compile_reward(text)))
            R = torch.tensor(rewards, device=logits.device, dtype=torch.float32)

        # EMA baseline
        batch_mean = float(R.mean().item())
        if self._compile_baseline is None:
            self._compile_baseline = batch_mean
        else:
            m = self._compile_baseline_momentum
            self._compile_baseline = m * self._compile_baseline + (1.0 - m) * batch_mean
        advantage = (R - self._compile_baseline).detach()  # [B]

        if slot_weight_mask is not None:
            shift_weights = slot_weight_mask[:, 1:].contiguous()
        else:
            shift_weights = supervised.float()
        mask = supervised.float() * (shift_weights > 0).float()  # [B, L-1]

        sampled_logp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        sum_logp = (sampled_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        rl_loss = -(advantage * sum_logp).mean()

        if self.compile_reinforce_entropy_coef > 0.0:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # [B, L-1]
            ent_term = (entropy * mask).sum() / mask.sum().clamp(min=1.0)
            rl_loss = rl_loss - self.compile_reinforce_entropy_coef * ent_term

        return rl_loss

    # ------------------------------------------------------------------
    def _compute_compile_loss(
        self,
        out: Any,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        slot_weight_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """根据 ``s3_compile_loss_mode`` 选择 ``L_compile`` 的实现方式。"""
        if self.lambda_compile <= 0.0 or out.logits is None:
            return None
        if self.tokenizer is None:
            return None
        mode = self.compile_loss_mode
        logits = out.logits
        if mode in ("off", "none", "disabled"):
            with torch.no_grad():
                R = self._greedy_rewards(logits, self.tokenizer)
            return torch.tensor(
                1.0 - float(R.mean().item()),
                device=logits.device,
                dtype=torch.float32,
            ).requires_grad_(False)
        if mode == "reward_weighted_ce":
            return self._compile_loss_reward_weighted_ce(
                logits, labels, slot_weight_mask, self.tokenizer,
            )
        if mode == "reinforce":
            return self._compile_loss_reinforce(
                logits, input_ids, labels, slot_weight_mask, self.tokenizer,
            )
        logger.warning(
            "未知 s3_compile_loss_mode=%s，已回退到 reward_weighted_ce",
            mode,
        )
        return self._compile_loss_reward_weighted_ce(
            logits, labels, slot_weight_mask, self.tokenizer,
        )

    # ------------------------------------------------------------------
    def _forward_batch(self, batch: Dict, input_ids: torch.Tensor,
                       labels: torch.Tensor,
                       slot_weight_mask: Optional[torch.Tensor]):
        """Single forward pass through the full model returning the output."""
        return self.model(
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
            input_ids=input_ids,
            graph_kv=batch.get("graph_kv"),
            slot_mask=batch.get("slot_mask"),
            labels=labels,
            slot_weight_mask=slot_weight_mask,
        )

    # ------------------------------------------------------------------
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one SFT epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Each batch yields a dict with:
            - all encoder fields (for ``StrucLift.encode``)
            - ``input_ids`` : [B, L]  tokenised skeleton + ground-truth fills
            - ``labels`` : [B, L]  target tokens (-100 for non-FILL)
            - ``slot_types`` : List[List[str]]  per-batch, per-slot type names
            - ``graph_kv`` : [B, N, d_graph]  (optional, built from encoder if absent)
            - ``slot_mask`` : [B, L, N]  (optional)
        epoch : int
            Current epoch number (1-indexed).

        Returns
        -------
        dict with loss components
        """
        self.model.train()
        self.model.binary_encoder.eval()
        self.model.source_encoder.eval()
        self.model.alignment_module.eval()

        accum: Dict[str, float] = {"total": 0.0, "compile": 0.0}
        n_batches = 0
        n_compile_batches = 0
        log_steps = _env_truthy("STRUCLIFT_STAGE3_LOG_STEP_MS")
        t_epoch_wall = time.perf_counter()

        steps_per_epoch = len(dataloader)
        for bi, batch in enumerate(dataloader):
            t_batch = time.perf_counter() if log_steps else 0.0
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if self.tokenizer is not None:
                batch["_tokenizer"] = self.tokenizer

            input_ids = batch["input_ids"]       # [B, L]
            labels = batch["labels"]             # [B, L]

            # build slot weight mask（与 augment mask_non_fill 的监督位置对齐）。
            # V10 repack 数据可直接提供 span-aware slot_weight_mask；优先使用，
            # 避免按 FILL token 区间猜 output span。
            slot_weight_mask = batch.get("slot_weight_mask")
            if slot_weight_mask is None and "slot_types" in batch:
                slot_weight_mask = (
                    self.slot_weight_builder.build_slot_weights_batched_for_supervised_labels(
                        input_ids,
                        labels,
                        batch["slot_types"],
                        self.config.module_c,
                        fill_token_ids=self.fill_token_ids,
                    )
                )

            ss_p = self._scheduled_sampling_prob_for_step(bi, steps_per_epoch)

            # ── scheduled sampling: mix teacher tokens with model predictions ──
            if ss_p > 0.0 and self.model.decoder._llm is not None:
                input_ids = self._apply_scheduled_sampling(
                    input_ids, labels, batch, slot_weight_mask, ss_p,
                )

            with torch.amp.autocast("cuda", enabled=self.fp16):
                out = self._forward_batch(batch, input_ids, labels,
                                          slot_weight_mask)
                loss = out.lm_loss

            if loss is None:
                continue

            # §7.5 / §9.4: 可微的 compile 辅助 loss（mode 由 config 控制）
            if self.lambda_compile > 0:
                compile_loss = self._compute_compile_loss(
                    out, input_ids, labels, slot_weight_mask,
                )
                if compile_loss is not None:
                    loss = loss + self.lambda_compile * compile_loss
                    accum["compile"] += float(compile_loss.detach().item())
                    n_compile_batches += 1

            self._optimizer_zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                if len(self._optimizers) > 1:
                    raise RuntimeError(
                        "device_map_auto 下不支持 GradScaler；请保持 decoder_device_map_auto 时 fp16 关闭",
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self._optimizer_step()

            accum["total"] += loss.item()
            n_batches += 1
            self.global_step += 1
            if log_steps:
                dt_ms = (time.perf_counter() - t_batch) * 1000.0
                logger.info(
                    "[Stage3] train global_step=%d/%d epoch_batch=%d/%d ss_p=%.4f "
                    "wall_ms=%.1f batch_loss=%.4f",
                    self.global_step,
                    int(self.total_train_steps or 0),
                    bi + 1,
                    steps_per_epoch,
                    ss_p,
                    dt_ms,
                    loss.item(),
                )

        if n_batches > 0:
            wall_s = time.perf_counter() - t_epoch_wall
            logger.info(
                "[Stage3] train epoch %d done: wall_s=%.1f (%.1f min) batches=%d "
                "avg_s_per_batch=%.2f train_loss_avg=%.4f",
                epoch,
                wall_s,
                wall_s / 60.0,
                n_batches,
                wall_s / n_batches,
                accum["total"] / n_batches,
            )
        denom_total = max(n_batches, 1)
        denom_compile = max(n_compile_batches, 1)
        return {
            k: v / (denom_compile if k == "compile" else denom_total)
            for k, v in accum.items()
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one evaluation epoch (no gradient, no scheduled sampling)."""
        self.model.eval()
        accum = {"total": 0.0}
        n_batches = 0
        log_steps = _env_truthy("STRUCLIFT_STAGE3_LOG_STEP_MS")
        t_epoch_wall = time.perf_counter()

        for bi, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            t_batch = time.perf_counter() if log_steps else 0.0
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            slot_weight_mask = batch.get("slot_weight_mask")
            if slot_weight_mask is None and "slot_types" in batch:
                slot_weight_mask = (
                    self.slot_weight_builder.build_slot_weights_batched_for_supervised_labels(
                        input_ids,
                        labels,
                        batch["slot_types"],
                        self.config.module_c,
                        fill_token_ids=self.fill_token_ids,
                    )
                )
            with torch.amp.autocast("cuda", enabled=self.fp16):
                out = self.model(
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
                    input_ids=input_ids,
                    graph_kv=batch.get("graph_kv"),
                    slot_mask=batch.get("slot_mask"),
                    labels=labels,
                    slot_weight_mask=slot_weight_mask,
                )
                loss = out.lm_loss
            if loss is not None:
                accum["total"] += loss.item()
                n_batches += 1
                if log_steps:
                    dt_ms = (time.perf_counter() - t_batch) * 1000.0
                    logger.info(
                        "[Stage3] val   step idx=%d wall_ms=%.1f batch_loss=%.4f",
                        bi,
                        dt_ms,
                        loss.item(),
                    )

        if n_batches > 0:
            wall_s = time.perf_counter() - t_epoch_wall
            logger.info(
                "[Stage3] val done: wall_s=%.1f (%.1f min) batches=%d avg_s_per_batch=%.2f "
                "val_loss_avg=%.4f",
                wall_s,
                wall_s / 60.0,
                n_batches,
                wall_s / n_batches,
                accum["total"] / n_batches,
            )

        self.model.train()
        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = "checkpoints/stage3",
        *,
        start_epoch: int = 1,
        initial_best_loss: float = float("inf"),
        initial_epochs_without_improve: int = 0,
        initial_global_step: int = 0,
        initial_total_train_steps: Optional[int] = None,
    ) -> None:
        """Full Stage 3 SFT training loop."""
        os.makedirs(save_dir, exist_ok=True)
        best_loss = float(initial_best_loss)
        epochs_without_improve = int(initial_epochs_without_improve)
        patience = self.early_stop_patience
        min_epochs = self.early_stop_min_epochs
        min_delta = self.early_stop_min_delta
        max_epochs = int(self.config.training.s3_epochs)
        steps_per_epoch = len(train_loader)
        self.global_step = int(initial_global_step)
        if initial_total_train_steps is not None and initial_total_train_steps > 0:
            self.total_train_steps = int(initial_total_train_steps)
        else:
            self.total_train_steps = steps_per_epoch * max_epochs
        if self.global_step <= 0 and start_epoch > 1:
            self.global_step = (start_epoch - 1) * steps_per_epoch

        logger.info(
            "[Stage3] scheduled sampling §7.5.3: 每 epoch 内前 %.0f%% batch "
            "teacher forcing，后 %.0f%% batch p_sample 0→%.2f | steps/epoch=%d "
            "max_epochs=%d total_train_steps=%d global_step=%d",
            self.ss_start * 100.0,
            (1.0 - self.ss_start) * 100.0,
            self.ss_max_p,
            steps_per_epoch,
            max_epochs,
            self.total_train_steps,
            self.global_step,
        )

        if patience > 0 and val_loader is None:
            logger.warning(
                "[Stage3] 已设早停 patience=%d 但未提供 val_loader，早停不会生效",
                patience,
            )

        if patience > 0 and val_loader is not None:
            logger.info(
                "[Stage3] 早停: patience=%d min_epochs=%d min_delta=%g max_epochs=%d",
                patience,
                min_epochs,
                min_delta,
                max_epochs,
            )

        stopped_early = False
        last_epoch = start_epoch - 1

        for epoch in range(start_epoch, max_epochs + 1):
            last_epoch = epoch
            metrics = self.train_epoch(train_loader, epoch)
            ss_p_end = self._scheduled_sampling_prob_for_step(
                max(0, steps_per_epoch - 1),
                max(1, steps_per_epoch),
            )

            msg = (
                f"[Stage3] Epoch {epoch}/{max_epochs}  "
                f"train_loss={metrics['total']:.4f}  "
                f"global_step={self.global_step}/{self.total_train_steps}  "
                f"scheduled_sampling_p={ss_p_end:.3f}"
            )

            val_loss = None
            if val_loader is not None:
                val_metrics = self.eval_epoch(val_loader)
                val_loss = val_metrics["total"]
                msg += f"  val_loss={val_loss:.4f}"

            logger.info(msg)

            save_metric = val_loss if val_loss is not None else metrics["total"]
            improved = save_metric < (best_loss - min_delta)
            if improved:
                best_loss = save_metric
                epochs_without_improve = 0
                self.save_checkpoint(
                    os.path.join(save_dir, "best_stage3.pt"),
                    epoch=epoch,
                    best_loss=best_loss,
                    epochs_without_improve=epochs_without_improve,
                )
                logger.info(
                    "[Stage3] 新 best（%s=%.6f）→ best_stage3.pt",
                    "val_loss" if val_loss is not None else "train_loss",
                    best_loss,
                )
            elif val_loader is not None:
                epochs_without_improve += 1

            self.save_checkpoint(
                os.path.join(save_dir, "latest_stage3.pt"),
                epoch=epoch,
                best_loss=best_loss,
                epochs_without_improve=epochs_without_improve,
            )

            if (
                patience > 0
                and val_loader is not None
                and epoch >= min_epochs
                and epochs_without_improve >= patience
            ):
                logger.info(
                    "[Stage3] 早停: epoch=%d val 连续 %d 轮未改善（≥ patience=%d）",
                    epoch,
                    epochs_without_improve,
                    patience,
                )
                stopped_early = True
                break

        if last_epoch >= start_epoch:
            src = os.path.join(save_dir, "best_stage3.pt")
            dst = os.path.join(save_dir, "final_stage3.pt")
            if os.path.isfile(src):
                import shutil
                shutil.copy2(src, dst)
            else:
                self.save_checkpoint(
                    dst,
                    epoch=last_epoch,
                    best_loss=best_loss,
                    epochs_without_improve=epochs_without_improve,
                )

        tag = "早停结束" if stopped_early else "跑满 epoch"
        logger.info(
            "[Stage3] SFT complete (%s). best_loss=%.6f  last_epoch=%d  "
            "续训: --resume %s",
            tag,
            best_loss,
            last_epoch,
            os.path.join(save_dir, "latest_stage3.pt"),
        )

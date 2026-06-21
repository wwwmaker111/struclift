"""
Stage 2 — Alignment Training.

Trains Module B — 与提案 §6.6 一致，并可选 §6.5.3 回归项：
  λ₁ L_SCOT + λ₂ L_edge + λ₃ L_region + λ₄ L_ca_align + λ₅ L_src_count

可选 §9.3：
  - Curriculum（O0→O1→O2→O3→Os）：由 :class:`Stage2CurriculumDataset` + 每阶段重建 DataLoader
  - 节点级硬负：每 ``s2_node_hard_neg_interval`` 个 epoch 启用
    :class:`struclift.losses.alignment.NodeHardNegativeContrastiveLoss`（其余 epoch 权重为 0）；
    可选 ``s2_node_hard_neg_warmup_epochs`` 在首次启用后线性升权，减轻冷启动。

调试（环境变量）：

  - ``STAGE2_LOG_NONFINITE=1``：某分项 nan/inf 时逐 batch 打标签（首轮总会打一次告警）。
  - ``STAGE2_LOG_GRAD=1``：每 epoch 第 0 个 batch 打印 ``alignment_module`` 梯度 L2 范数（近 0 则无有效反传）。
  - ``STAGE2_LOG_ALIGN_NORM=1``：每 epoch 结束打印 ``alignment_module`` 权重 L2；多轮几乎不变则参数几乎未更新。

Frozen: Module A encoders (pretrained in Stage 1).
Trained: Module B (alignment module).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import Stage2CurriculumDataset, build_stage2_train_loader
from struclift.models.struclift import StrucLift
from struclift.losses import (
    SCOTAlignmentLoss,
    EdgeRecoveryLoss,
    CrossAttentionAlignmentLoss,
    RegionInfoNCELoss,
    NodeHardNegativeContrastiveLoss,
)
from struclift.losses.alignment import SourceCountRegressionLoss

logger = logging.getLogger(__name__)


def _module_param_l2_norm(module: nn.Module) -> float:
    """Flat L2 of all parameters (用于 STAGE2_LOG_ALIGN_NORM 检查 Module B 是否在更新)。"""
    s = torch.tensor(0.0)
    dev = next(module.parameters()).device
    s = s.to(dev)
    with torch.no_grad():
        for p in module.parameters():
            s = s + p.detach().float().pow(2).sum()
    return float(torch.sqrt(s).item())


class Stage2Trainer:
    """Alignment training for Module B.

    Parameters
    ----------
    model : StrucLift
    config : StrucLiftConfig
    device : torch.device
    curriculum_dataset
        若给定，:meth:`train` 内按 epoch 切换 ``set_stage`` 并重建 DataLoader。
    curriculum_epochs_per_stage
        每多少个 epoch 推进一档（例如 10 → 1–10 仅 O0，11–20 含 O0+O1）。
    stage2_loader_kwargs
        传给 :func:`build_stage2_train_loader` 的 ``batch_size / num_workers / pin_memory / skip_invalid``。
    """

    def __init__(
        self,
        model: StrucLift,
        config: StrucLiftConfig,
        device: torch.device = torch.device("cpu"),
        *,
        curriculum_dataset: Optional[Stage2CurriculumDataset] = None,
        curriculum_epochs_per_stage: Optional[int] = None,
        stage2_loader_kwargs: Optional[Dict[str, Any]] = None,
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._use_ddp = bool(use_ddp) and self._world_size > 1
        tc = config.training

        self._curriculum_dataset = curriculum_dataset
        self._curriculum_epochs_per_stage = int(
            curriculum_epochs_per_stage
            if curriculum_epochs_per_stage is not None
            else getattr(tc, "s2_curriculum_epochs_per_stage", 10),
        )
        self._loader_kw: Dict[str, Any] = dict(stage2_loader_kwargs or {})
        self._curriculum_stage_cached: int = -999
        self._warned_nonfinite_component: bool = False

        # Freeze Module A (pretrained encoders)
        self._unwrap_model().freeze_encoders()
        # Stage2 仅跑 encode+对齐，不参与 Module C；冻结避免 DDP 把 decoder 参数纳入未用桶
        self._unwrap_model().freeze_decoder_adapters()

        # ── losses ───────────────────────────────────────────────────────
        _s1_tau = float(tc.s1_infonce_tau)
        _reg_t = getattr(tc, "s2_region_infonce_tau", None)
        _nhn_t = getattr(tc, "s2_node_hn_infonce_tau", None)
        _tau_reg = _s1_tau if _reg_t is None else float(_reg_t)
        _tau_nhn = _s1_tau if _nhn_t is None else float(_nhn_t)

        self.scot_loss_fn = SCOTAlignmentLoss()
        self.edge_loss_fn = EdgeRecoveryLoss()
        self.ca_align_loss_fn = CrossAttentionAlignmentLoss()
        self.region_nce_fn = RegionInfoNCELoss(tau=_tau_reg)
        self.src_count_loss_fn = SourceCountRegressionLoss()
        self.node_hn_fn = NodeHardNegativeContrastiveLoss(tau=_tau_nhn)

        # ── loss weights（§6.6）──────────────────────────────────────────
        self.lambda_scot = tc.s2_lambda_scot
        self.lambda_edge = tc.s2_lambda_edge
        self.lambda_region = tc.s2_lambda_region
        self.lambda_ca_align = tc.s2_lambda_ca_align
        self.lambda_src_count = getattr(tc, "s2_lambda_src_count", 0.1)
        self.lambda_node_hn = float(getattr(tc, "s2_lambda_node_hard_neg", 0.2))
        self.node_hard_neg_topk = int(getattr(tc, "s2_node_hard_neg_topk", 32))
        self.node_hard_neg_pos_thresh = float(getattr(tc, "s2_node_hard_neg_pos_thresh", 0.05))
        self.node_hard_neg_enabled = bool(getattr(tc, "s2_node_hard_neg_enabled", True))
        self.node_hard_neg_interval = int(getattr(tc, "s2_node_hard_neg_interval", 5))
        self.node_hard_neg_warmup_epochs = int(getattr(tc, "s2_node_hard_neg_warmup_epochs", 0))
        self._first_node_hn_epoch = self._compute_first_node_hn_epoch()

        # ── optimizer (only Module B params) ─────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self._unwrap_model().alignment_module.parameters(),
            lr=tc.s2_lr,
            weight_decay=tc.s2_weight_decay,
        )

        # 与全局 fp16 解耦：Stage2 默认 s2_fp16=False（见 TrainingConfig），避免 AMP 下梯度 nan 导致不更新
        _s2fp = bool(getattr(tc, "s2_fp16", False))
        self.fp16 = _s2fp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.fp16 else None

    def _sync_alignment_gradients(self) -> None:
        """多卡时同步 Module B 梯度（跨 rank 求平均），替代 nn.parallel.DDP。

        :meth:`struclift.models.struclift.StrucLift.encode` 在 batch 多图时会在**一次**
        ``forward`` 里多次调用 ``alignment_module``。任何 ``DistributedDataParallel`` 包装
        都会与这种「同迭代内多路使用同一子模块」冲突（marked ready twice）。故 Stage2
        使用 DataLoader 分片 + 本函数手工 ``all_reduce`` 规约，与 DDP 的梯度平均一致。
        """
        if not self._use_ddp or not dist.is_initialized():
            return
        ws = float(self._world_size)
        am = self._unwrap_model().alignment_module
        for p in am.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                # 一单卡出现 inf/nan，all_reduce SUM 会把两卡梯度全污染，后续 scaler 跳过 step，
                # 分项 loss（经 nan_to_num）看起来像「突然全 0」且 val 完全不变。
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(ws)

    def _unwrap_model(self) -> StrucLift:
        """单卡 / 多卡（无 DDP 包装，见 :meth:`_sync_alignment_gradients`）下均为 ``StrucLift``。"""
        m = self.model
        if isinstance(m, DDP):
            return m.module
        return m

    def _forward_stage2_encoder_out(self, batch: Dict[str, Any]) -> Any:
        """经 ``StrucLift.forward``（与 ``__call__`` 一致）以兼容 DDP 的梯度规约。"""
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
        )
        assert out.encoder_out is not None, "Stage2 需要 run_alignment 的 encode 输出"
        return out.encoder_out

    # -------------------------------------------------------------------------
    # §完整 checkpoint（含优化器 / scaler / epoch / best_loss / RNG）
    # -------------------------------------------------------------------------
    _CKPT_VERSION = 1  # bump 当 schema 变化

    def save_checkpoint(
        self,
        path: str,
        *,
        epoch: int,
        best_loss: float,
    ) -> None:
        """保存完整 Stage 2 训练状态（含优化器 / scaler / epoch / best_loss / RNG）。

        原子写：先写 ``<path>.tmp`` 再 ``os.replace`` 到目标路径，避免中途 Ctrl+C
        产生损坏的 ckpt。
        """
        obj = {
            "version": self._CKPT_VERSION,
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": int(epoch),
            "best_loss": float(best_loss),
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
        """恢复完整训练状态（含优化器）。兼容 legacy（仅权重）格式。

        Returns
        -------
        dict
            ``{start_epoch, best_loss, missing, unexpected, is_legacy}``
        """
        try:
            obj = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location=self.device)

        result: Dict[str, Any] = {
            "start_epoch": 1,
            "best_loss": float("inf"),
            "missing": 0,
            "unexpected": 0,
            "is_legacy": False,
        }
        is_new = (
            isinstance(obj, dict)
            and "model" in obj
            and "optimizer" in obj
            and "epoch" in obj
        )

        if is_new:
            state = obj["model"]
            missing, unexpected = self._unwrap_model().load_state_dict(state, strict=False)
            try:
                self.optimizer.load_state_dict(obj["optimizer"])
            except Exception as e:  # pragma: no cover
                logger.warning("[Stage2] optimizer state load 失败（将从零重启）: %s", e)
            if self.scaler is not None and obj.get("scaler") is not None:
                try:
                    self.scaler.load_state_dict(obj["scaler"])
                except Exception as e:  # pragma: no cover
                    logger.warning("[Stage2] scaler state load 失败: %s", e)
            rng = obj.get("rng")
            if rng is not None:
                try:
                    torch.set_rng_state(rng["torch"])
                    if torch.cuda.is_available() and rng.get("cuda") is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception:
                    pass
            result["start_epoch"] = int(obj.get("epoch", 0)) + 1
            result["best_loss"] = float(obj.get("best_loss", float("inf")))
        else:
            state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            missing, unexpected = self._unwrap_model().load_state_dict(state, strict=False)
            result["is_legacy"] = True

        result["missing"] = len(missing)
        result["unexpected"] = len(unexpected)
        logger.info(
            "[Stage2] checkpoint 已加载 format=%s start_epoch=%d best_loss=%.6f "
            "missing=%d unexpected=%d",
            "legacy" if result["is_legacy"] else "full",
            result["start_epoch"],
            result["best_loss"],
            result["missing"],
            result["unexpected"],
        )
        return result

    def _node_hn_active(self, epoch: int) -> bool:
        if not self.node_hard_neg_enabled:
            return False
        if self.node_hard_neg_interval <= 0:
            return True
        return (epoch % self.node_hard_neg_interval) == 0

    def _compute_first_node_hn_epoch(self) -> int:
        """满足 ``_node_hn_active`` 的最小正历元（或极大值当该项整训关闭时）。"""
        if not self.node_hard_neg_enabled:
            return 1_000_000_000
        if self.node_hard_neg_interval <= 0:
            return 1
        return int(self.node_hard_neg_interval)

    def _node_hn_warmup_scale(self, epoch: int) -> float:
        """在 ``_node_hn_active`` 为真时，将 node_hn 的 λ 再乘 0→1 的系数；关 warmup 时恒为 1。"""
        w = self.node_hard_neg_warmup_epochs
        if w <= 0:
            return 1.0
        if not self._node_hn_active(epoch):
            return 1.0
        s = (epoch - self._first_node_hn_epoch) / float(w)
        return float(min(1.0, max(0.0, s)))

    def _rebuild_curriculum_loader(self) -> DataLoader:
        assert self._curriculum_dataset is not None
        tc = self.config.training
        return build_stage2_train_loader(
            self._curriculum_dataset,
            batch_size=int(self._loader_kw.get("batch_size", tc.s2_batch_size)),
            shuffle=True,
            num_workers=int(self._loader_kw.get("num_workers", tc.num_workers)),
            skip_invalid=bool(self._loader_kw.get("skip_invalid", True)),
            pin_memory=bool(self._loader_kw.get("pin_memory", False)),
            prefetch_factor=int(self._loader_kw.get("prefetch_factor", 4)),
            distributed=bool(self._loader_kw.get("distributed", False)),
            rank=int(self._loader_kw.get("rank", 0)),
            world_size=int(self._loader_kw.get("world_size", 1)),
        )

    def _loss_tensors_from_enc_out(
        self,
        enc_out: Any,
        batch: Dict[str, Any],
        *,
        hn_on: bool,
        epoch: int = 1,
        log_nonfinite_warnings: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """从 ``encode`` 输出与 ``batch`` 计算总 loss 与各分量（与 train/val 口径一致）。"""
        assert enc_out.alignment is not None, "Stage2 需要 run_alignment=True 的 encode 输出"
        M_pred_list = enc_out.alignment.alignment_matrix
        M_gt_batch = batch["M_gt"]
        conf_batch = batch.get("alignment_confidence")
        adj_bin_batch = batch.get("adj_bin")
        adj_src_batch = batch.get("adj_src_gt")

        if not isinstance(M_pred_list, list):
            M_pred_list = [M_pred_list]

        loss_scot = torch.tensor(0.0, device=self.device)
        loss_edge = torch.tensor(0.0, device=self.device)
        loss_ca = torch.tensor(0.0, device=self.device)
        B_align = len(M_pred_list)

        for gi, M_pred_i in enumerate(M_pred_list):
            nb_i, ns_i = M_pred_i.shape
            M_gt_i = M_gt_batch[gi, :nb_i, :ns_i]
            conf_i = conf_batch[gi, :nb_i, :ns_i] if conf_batch is not None else None
            ab_i = adj_bin_batch[gi, :nb_i, :nb_i] if adj_bin_batch is not None else None
            as_i = adj_src_batch[gi, :ns_i, :ns_i] if adj_src_batch is not None else None

            loss_scot = loss_scot + self.scot_loss_fn(M_pred_i, M_gt_i, confidence=conf_i)
            if ab_i is not None and as_i is not None:
                loss_edge = loss_edge + self.edge_loss_fn(M_pred_i, ab_i, as_i)
            if (
                enc_out.cross_attn_weights_list is not None
                and gi < len(enc_out.cross_attn_weights_list)
            ):
                A_cross_i = enc_out.cross_attn_weights_list[gi]
                loss_ca = loss_ca + self.ca_align_loss_fn(A_cross_i, M_gt_i)
            else:
                loss_ca = loss_ca + self.ca_align_loss_fn(M_pred_i, M_gt_i)

        loss_scot = loss_scot / max(B_align, 1)
        loss_edge = loss_edge / max(B_align, 1)
        loss_ca = loss_ca / max(B_align, 1)

        loss_region = torch.tensor(0.0, device=self.device)
        if enc_out.region_bin_embs_list is not None and enc_out.region_positive_mask_list is not None:
            for gi in range(B_align):
                r_b = enc_out.region_bin_embs_list[gi]
                r_s = enc_out.region_src_embs_list[gi] if enc_out.region_src_embs_list else None
                mask = enc_out.region_positive_mask_list[gi]
                if r_b is None or r_s is None or mask is None:
                    continue
                if not mask.any():
                    continue
                r_bn = nn.functional.normalize(r_b, dim=-1)
                r_sn = nn.functional.normalize(r_s, dim=-1)
                loss_region = loss_region + self.region_nce_fn(r_bn, r_sn, mask)
            loss_region = loss_region / max(B_align, 1)

        loss_src_count = torch.tensor(0.0, device=self.device)
        src_batch_t = batch.get("src_batch")
        if enc_out.predicted_n_s_list is not None:
            for gi in range(B_align):
                pred_ns = enc_out.predicted_n_s_list[gi]
                if pred_ns is None:
                    continue
                if src_batch_t is not None:
                    actual_ns = float((src_batch_t == gi).sum().item())
                else:
                    actual_ns = float(M_pred_list[gi].shape[1])
                loss_src_count = loss_src_count + self.src_count_loss_fn(
                    pred_ns,
                    torch.tensor(actual_ns, device=self.device),
                )
            loss_src_count = loss_src_count / max(B_align, 1)

        loss_node_hn = torch.tensor(0.0, device=self.device)
        bin_batch_t = batch.get("bin_batch")
        if (
            hn_on
            and bin_batch_t is not None
            and src_batch_t is not None
            and enc_out.binary_node_embs is not None
            and enc_out.source_node_embs is not None
        ):
            H_all_b = enc_out.binary_node_embs
            H_all_s = enc_out.source_node_embs
            acc_hn = torch.tensor(0.0, device=self.device)
            n_hn = 0
            for gi in range(B_align):
                b_mask = bin_batch_t == gi
                s_mask = src_batch_t == gi
                Hb = H_all_b[b_mask]
                Hs = H_all_s[s_mask]
                nb_i, ns_i = M_pred_list[gi].shape
                if Hb.size(0) != nb_i or Hs.size(0) != ns_i:
                    continue
                M_gt_i = M_gt_batch[gi, :nb_i, :ns_i]
                acc_hn = acc_hn + self.node_hn_fn(
                    Hb,
                    Hs,
                    M_gt_i,
                    topk=self.node_hard_neg_topk,
                    pos_thresh=self.node_hard_neg_pos_thresh,
                )
                n_hn += 1
            if n_hn > 0:
                loss_node_hn = acc_hn / float(n_hn)

        def _nf(t: torch.Tensor, *, tag: str) -> torch.Tensor:
            if (
                log_nonfinite_warnings
                and not torch.isfinite(t.detach()).all()
                and self._rank == 0
            ):
                if not self._warned_nonfinite_component:
                    self._warned_nonfinite_component = True
                    logger.warning(
                        "[Stage2] 某 loss 分项含 nan/inf（先出现于「%s」）；已 nan_to_num 后才反传。"
                        "日志里 edge/region 突然变 0 多为此处被抹平；设 STAGE2_LOG_NONFINITE=1 "
                        "可每笔 batch 打标签。常与 EdgeRecovery/KL、lr 或过大多卡梯度 inf 有关。",
                        tag,
                    )
                elif os.environ.get("STAGE2_LOG_NONFINITE", "").strip() == "1":
                    logger.warning("[Stage2] 分项「%s」当前 batch 仍非 finite。", tag)
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        loss_scot = _nf(loss_scot, tag="scot")
        loss_edge = _nf(loss_edge, tag="edge")
        loss_ca = _nf(loss_ca, tag="ca_align")
        loss_region = _nf(loss_region, tag="region")
        loss_src_count = _nf(loss_src_count, tag="src_count")
        loss_node_hn = _nf(loss_node_hn, tag="node_hn")

        hn_warm = self._node_hn_warmup_scale(epoch) if hn_on else 0.0

        loss = (
            self.lambda_scot * loss_scot
            + self.lambda_edge * loss_edge
            + self.lambda_ca_align * loss_ca
            + self.lambda_region * loss_region
            + self.lambda_src_count * loss_src_count
            + (self.lambda_node_hn * hn_warm) * loss_node_hn
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, {
            "scot": loss_scot,
            "edge": loss_edge,
            "ca_align": loss_ca,
            "region": loss_region,
            "src_count": loss_src_count,
            "node_hn": loss_node_hn,
        }

    @torch.no_grad()
    def eval_epoch(self, dataloader: DataLoader, *, epoch: int = 1) -> Dict[str, float]:
        """验证一 round：无反传。节点硬负的开关与 :meth:`train_epoch` 同 ``epoch`` 规则一致。"""
        self.model.eval()
        self._unwrap_model().binary_encoder.eval()
        self._unwrap_model().source_encoder.eval()

        accum = {
            "total": 0.0, "scot": 0.0, "edge": 0.0,
            "ca_align": 0.0, "region": 0.0, "src_count": 0.0, "node_hn": 0.0,
        }
        n_batches = 0
        hn_on = self._node_hn_active(epoch)

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self.fp16):
                enc_out = self._forward_stage2_encoder_out(batch)
            with torch.amp.autocast("cuda", enabled=self.fp16):
                loss, comp = self._loss_tensors_from_enc_out(
                    enc_out,
                    batch,
                    hn_on=hn_on,
                    epoch=epoch,
                    log_nonfinite_warnings=False,
                )

            accum["total"] += loss.item()
            accum["scot"] += comp["scot"].item()
            accum["edge"] += comp["edge"].item()
            accum["ca_align"] += comp["ca_align"].item()
            accum["region"] += comp["region"].item()
            accum["src_count"] += comp["src_count"].item()
            accum["node_hn"] += comp["node_hn"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader, *, epoch: int = 1) -> Dict[str, float]:
        """Run one training epoch。"""
        self.model.train()
        # keep encoders in eval mode (frozen)
        self._unwrap_model().binary_encoder.eval()
        self._unwrap_model().source_encoder.eval()

        accum = {
            "total": 0.0, "scot": 0.0, "edge": 0.0,
            "ca_align": 0.0, "region": 0.0, "src_count": 0.0, "node_hn": 0.0,
        }
        n_batches = 0
        hn_on = self._node_hn_active(epoch)

        for batch in dataloader:
            # 仅第一步 + CUDA：细分耗时（h2d / encode / 各项 loss / backward+opt）
            _prof = (
                os.environ.get("STAGE2_PROFILE_STEP", "") == "1"
                and n_batches == 0
                and self.device.type == "cuda"
            )
            t_prof0: float | None = None
            if _prof:
                t_prof0 = time.perf_counter()

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if _prof:
                torch.cuda.synchronize()
                t_h2d_end = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=self.fp16):
                enc_out = self._forward_stage2_encoder_out(batch)

            if _prof:
                torch.cuda.synchronize()
                t_enc_end = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=self.fp16):
                loss, comp = self._loss_tensors_from_enc_out(
                    enc_out, batch, hn_on=hn_on, epoch=epoch
                )

            if _prof:
                torch.cuda.synchronize()
                t_loss_end = time.perf_counter()

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._sync_alignment_gradients()
                gn_align = torch.nn.utils.clip_grad_norm_(
                    self._unwrap_model().alignment_module.parameters(),
                    max_norm=float("inf"),
                )
                if (
                    os.environ.get("STAGE2_LOG_GRAD", "").strip() == "1"
                    and n_batches == 0
                ):
                    logger.info(
                        "[Stage2] debug STAGE2_LOG_GRAD: batch0 |grad|_2(alignment)=%.6e",
                        float(gn_align),
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._sync_alignment_gradients()
                gn_align = torch.nn.utils.clip_grad_norm_(
                    self._unwrap_model().alignment_module.parameters(),
                    max_norm=float("inf"),
                )
                if (
                    os.environ.get("STAGE2_LOG_GRAD", "").strip() == "1"
                    and n_batches == 0
                ):
                    logger.info(
                        "[Stage2] debug STAGE2_LOG_GRAD: batch0 |grad|_2(alignment)=%.6e",
                        float(gn_align),
                    )
                self.optimizer.step()

            if _prof and t_prof0 is not None:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                assert t_h2d_end is not None and t_enc_end is not None and t_loss_end is not None
                dt_h2d = t_h2d_end - t_prof0
                dt_enc = t_enc_end - t_h2d_end
                dt_loss = t_loss_end - t_enc_end
                dt_bwd = t_end - t_loss_end
                dt_tot = t_end - t_prof0
                logger.info(
                    "[Stage2][profile 仅 batch0] 秒:  "
                    "host→GPU=%.3f  encode(含冻结Encoder前向)=%.3f  各项loss(含scot/edge/…/autocast)=%.3f  "
                    "bwd+optimizer=%.3f  合计=%.3f  (DataLoader 取本 batch 的阻塞在上面的 for 之前，可另用脚本测)",
                    dt_h2d,
                    dt_enc,
                    dt_loss,
                    dt_bwd,
                    dt_tot,
                )

            accum["total"] += loss.item()
            accum["scot"] += comp["scot"].item()
            accum["edge"] += comp["edge"].item()
            accum["ca_align"] += comp["ca_align"].item()
            accum["region"] += comp["region"].item()
            accum["src_count"] += comp["src_count"].item()
            accum["node_hn"] += comp["node_hn"].item()
            n_batches += 1

        if self._use_ddp and dist.is_initialized():
            # 必须**每个 rank 都**参加 all_reduce；若某 rank 本 epoch 无 batch（n_batches=0）而
            # 另一 rank 有 batch，原先 `n_batches>0` 才 reduce 会少一路参与，导致 NCCL 报错或死锁。
            mvec = torch.tensor(
                [
                    accum["total"],
                    accum["scot"],
                    accum["edge"],
                    accum["ca_align"],
                    accum["region"],
                    accum["src_count"],
                    accum["node_hn"],
                    float(n_batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(mvec, op=dist.ReduceOp.SUM)
            tot_b = mvec[7].item()
            d = max(tot_b, 1.0)
            return {
                "total": float(mvec[0].item() / d),
                "scot": float(mvec[1].item() / d),
                "edge": float(mvec[2].item() / d),
                "ca_align": float(mvec[3].item() / d),
                "region": float(mvec[4].item() / d),
                "src_count": float(mvec[5].item() / d),
                "node_hn": float(mvec[6].item() / d),
            }
        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        val_loaders_by_curriculum_stage: Optional[List[DataLoader]] = None,
        save_dir: str = "checkpoints/stage2",
        start_epoch: int = 1,
        best_loss: float = float("inf"),
    ) -> None:
        """Full Stage 2 training loop。

        每轮结束保存 ``latest_stage2.pt``（含优化器状态），若 **选用** 的 total loss
        创新低则再保存 ``best_stage2.pt``。训练完毕后另存 ``final_stage2.pt``。

        若 ``val_loader`` 非空，每轮在验证集上计算与训练同口径的 loss，并 **以验证
        total 作为选优与写入 checkpoint 的 best_loss**；否则与旧行为一致，使用训练
        total。从「仅训集」改为「带验证集」续训时，若需重新以 val 为基线，请在
        ``scripts/train_stage2_binskel.py`` 使用 ``--reset-best-baseline``。

        若提供 ``val_loaders_by_curriculum_stage``（与 ``--curriculum`` 同用），则按
        当前课程 ``stage`` 选用对应下标的验证集；**阶段切换时** 将 ``best_loss`` 重置
        为 ``+inf``，以便**只在同一并集 val 内**比较选优（不同档的 total 数值不可横向
        对比）。此时除 ``best_stage2.pt``（当前档内滚动最优）外，每次刷新 best 会**同时**
        写入 ``best_stage2_curriculum_stage{k}.pt``，供各档各留一份「该档 val 上最优」
        权重，避免只认一个全局文件名时不知道哪一档最好。

        配合 :meth:`load_full_checkpoint` + ``--resume`` 可保留动量/epoch/best_loss。
        """
        os.makedirs(save_dir, exist_ok=True)
        total_epochs = int(self.config.training.s2_epochs)
        loader = train_loader
        self._curriculum_stage_cached = -999

        if start_epoch > 1 and self._rank == 0:
            logger.info(
                "[Stage2] 续训：start_epoch=%d best_loss=%.6f",
                start_epoch,
                best_loss,
            )

        if val_loaders_by_curriculum_stage is not None:
            if self._curriculum_dataset is None:
                raise ValueError("val_loaders_by_curriculum_stage 需要 curriculum_dataset")
            n_s = self._curriculum_dataset.max_stage() + 1
            if len(val_loaders_by_curriculum_stage) != n_s:
                raise ValueError(
                    f"val_loaders 数量({len(val_loaders_by_curriculum_stage)})须等于课程阶数 {n_s}"
                )

        last_epoch = start_epoch - 1
        prev_val_stage: Optional[int] = None
        for epoch in range(start_epoch, total_epochs + 1):
            stage = 0
            if self._curriculum_dataset is not None:
                stage = min(
                    (epoch - 1) // max(self._curriculum_epochs_per_stage, 1),
                    self._curriculum_dataset.max_stage(),
                )
                self._curriculum_dataset.set_stage(stage)
                if stage != self._curriculum_stage_cached:
                    self._curriculum_stage_cached = stage
                    loader = self._rebuild_curriculum_loader()
                    if self._rank == 0:
                        logger.info(
                            "[Stage2] Curriculum 阶段=%d/%d 包含优化档=[%s] 样本数=%d",
                            stage,
                            self._curriculum_dataset.max_stage(),
                            ",".join(self._curriculum_dataset.active_opt_tags()),
                            len(self._curriculum_dataset),
                        )

            if val_loaders_by_curriculum_stage is not None:
                if prev_val_stage is not None and stage != prev_val_stage:
                    best_loss = float("inf")
                    if self._rank == 0:
                        logger.info(
                            "[Stage2] 验证已换为 curriculum stage=%d 并集（别于 stage=%d），"
                            "best_loss 基线重置",
                            stage,
                            prev_val_stage,
                        )
                prev_val_stage = stage

            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

            v_loader_this: Optional[DataLoader] = val_loader
            if val_loaders_by_curriculum_stage is not None:
                v_loader_this = val_loaders_by_curriculum_stage[stage]

            metrics = self.train_epoch(loader, epoch=epoch)
            hn_on_e = self._node_hn_active(epoch)
            hn_tag = "on" if hn_on_e else "off"
            hn_w = self._node_hn_warmup_scale(epoch) if hn_on_e else 0.0
            hn_w_s = f" w={hn_w:.2f}" if (hn_on_e and self.node_hard_neg_warmup_epochs > 0) else ""
            if self._use_ddp and dist.is_initialized():
                dist.barrier()

            val_metrics: Optional[Dict[str, float]] = None
            if (
                v_loader_this is not None
                and len(v_loader_this) > 0
                and self._rank == 0
            ):
                val_metrics = self.eval_epoch(v_loader_this, epoch=epoch)

            if self._use_ddp and dist.is_initialized():
                dist.barrier()

            if self._rank == 0:
                logger.info(
                    f"[Stage2] Epoch {epoch}/{total_epochs}  train  "
                    f"loss={metrics['total']:.4f}  "
                    f"scot={metrics['scot']:.4f}  "
                    f"edge={metrics['edge']:.4f}  "
                    f"ca={metrics['ca_align']:.4f}  "
                    f"region={metrics['region']:.4f}  "
                    f"src_count={metrics['src_count']:.4f}  "
                    f"node_hn={metrics['node_hn']:.4f}({hn_tag}{hn_w_s})"
                )
            if self._rank == 0 and val_metrics is not None:
                _stg = f"stage{stage} " if val_loaders_by_curriculum_stage else ""
                logger.info(
                    f"[Stage2] Epoch {epoch}/{total_epochs}  val    {_stg}"
                    f"loss={val_metrics['total']:.4f}  "
                    f"scot={val_metrics['scot']:.4f}  "
                    f"edge={val_metrics['edge']:.4f}  "
                    f"ca={val_metrics['ca_align']:.4f}  "
                    f"region={val_metrics['region']:.4f}  "
                    f"src_count={val_metrics['src_count']:.4f}  "
                    f"node_hn={val_metrics['node_hn']:.4f}({hn_tag}{hn_w_s})"
                )

            if (
                self._rank == 0
                and os.environ.get("STAGE2_LOG_ALIGN_NORM", "").strip() == "1"
            ):
                pl2 = _module_param_l2_norm(self._unwrap_model().alignment_module)
                logger.info(
                    "[Stage2] debug STAGE2_LOG_ALIGN_NORM: alignment_module param L2=%.8f",
                    pl2,
                )

            select = val_metrics["total"] if val_metrics is not None else metrics["total"]
            tag = "val" if val_metrics is not None else "train"

            if self._rank == 0:
                if select < best_loss:
                    best_loss = select
                    self.save_checkpoint(
                        os.path.join(save_dir, "best_stage2.pt"),
                        epoch=epoch,
                        best_loss=best_loss,
                    )
                    _log_path = "best_stage2.pt"
                    if val_loaders_by_curriculum_stage is not None:
                        _st_path = os.path.join(
                            save_dir,
                            f"best_stage2_curriculum_stage{stage}.pt",
                        )
                        self.save_checkpoint(
                            _st_path,
                            epoch=epoch,
                            best_loss=best_loss,
                        )
                        _log_path = f"best_stage2.pt + {os.path.basename(_st_path)}"
                    logger.info(
                        "[Stage2] 已更新 %s（best_loss=%.6f，按 %s total）",
                        _log_path,
                        best_loss,
                        tag,
                    )
                # 每轮结束保存 latest，用于 --resume 无缝续训
                self.save_checkpoint(
                    os.path.join(save_dir, "latest_stage2.pt"),
                    epoch=epoch,
                    best_loss=best_loss,
                )
            last_epoch = epoch

        if self._rank == 0:
            self.save_checkpoint(
                os.path.join(save_dir, "final_stage2.pt"),
                epoch=last_epoch,
                best_loss=best_loss,
            )
            if val_loaders_by_curriculum_stage is not None:
                logger.info(
                    "[Stage2] Training complete. 当前 in-memory best_loss=%.4f 仅对**最后一段**"
                    " val 有意义；分档最优见 save_dir 下 best_stage2_curriculum_stage*.pt。",
                    best_loss,
                )
            else:
                logger.info(f"[Stage2] Training complete. Best loss: {best_loss:.4f}")

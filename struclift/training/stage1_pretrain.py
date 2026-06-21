"""
Stage 1 — Module A 预训练（StrucLift_Research_Proposal.md §5.5）。

损失（与原文一致）::

    L_A = L_pattern_cls + λ_recon · L_neighbor_recon

可选（config.training.s1_lambda_graph_align>0 时）::

    + λ_graph · L_graph_bin_src   # 训练源码侧编码器；§5.5 正文未单独写出

Trained: BinaryCFGEncoder, SourceCFGEncoder（§5.3.3 / §5.4 PMA 图级向量）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import move_binskel_batch_to_device
from struclift.models.struclift import StrucLift
from struclift.losses import (
    GraphBinarySourceContrastiveLoss,
    NeighborReconLoss,
    PatternClassificationLoss,
    RegionInfoNCELoss,
)
from struclift.training.hard_negative_bank import HardNegativeBank

logger = logging.getLogger(__name__)


def _atomic_torch_save(obj: Any, path: str) -> None:
    """写入临时文件再 ``replace``，降低断电/断网时写到一半的损坏概率。"""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".pt", dir=d)
    try:
        os.close(fd)
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


class Stage1Trainer:
    """Module A 预训练（提案 §5.5）。"""

    def __init__(
        self,
        model: StrucLift,
        config: StrucLiftConfig,
        device: torch.device = torch.device("cpu"),
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_distributed = self.world_size > 1
        tc = config.training

        self.pattern_loss_fn = PatternClassificationLoss(
            num_classes=config.vocab.num_patterns,
        )
        self.neighbor_loss_fn = NeighborReconLoss(tau=tc.s1_infonce_tau)
        # §9.2 graph-level 对比：若启用 cross_opt_pairs 则用 multi-positive（同函数
        # 多编译作为正对），否则默认 mask_neg（同函数不作为负样本，但不当正对）。
        graph_mode = "multi_positive" if getattr(
            tc, "s1_cross_opt_pairs", False,
        ) else "mask_neg"
        self.graph_align_loss_fn = GraphBinarySourceContrastiveLoss(
            tau=tc.s1_infonce_tau, mode=graph_mode,
        )
        self.region_nce_fn = RegionInfoNCELoss(tau=tc.s1_infonce_tau)

        self.lambda_recon = tc.s1_lambda_recon
        self.lambda_graph_align = tc.s1_lambda_graph_align
        self.lambda_region = tc.s1_lambda_region

        model_for_optim = self._unwrap_model()
        params = (
            list(model_for_optim.binary_encoder.parameters())
            + list(model_for_optim.source_encoder.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params, lr=tc.s1_lr, weight_decay=tc.s1_weight_decay,
        )

        self.scaler = torch.amp.GradScaler("cuda") if tc.fp16 and device.type == "cuda" else None
        self.fp16 = tc.fp16

        # §9.2 硬负挖掘（可选）；真正创建 bank 的时机延后到 ``attach_hard_negative_bank``，
        # 因为需要训练脚本提供 Dataset 长度和 refresh loader 工厂
        self.hard_neg_bank: Optional[HardNegativeBank] = None
        self.refresh_loader_factory: Optional[Callable[[], DataLoader]] = None
        self.hard_neg_refresh_every: int = int(
            getattr(tc, "s1_hard_neg_refresh_every", 0) or 0,
        )
        self.hard_neg_topk: int = int(getattr(tc, "s1_hard_neg_topk", 32) or 32)

    def attach_hard_negative_bank(
        self,
        num_samples: int,
        d_model: int,
        refresh_loader_factory: Callable[[], DataLoader],
    ) -> None:
        """训练脚本调用：注入硬负 bank + 一个无参工厂（返回全量顺序 DataLoader）。

        工厂每次调用应返回**新**的 DataLoader（``shuffle=False`` +
        ``cross_opt_pairs=False`` + ``DistributedSampler`` 若 DDP），用于 bank refresh。
        """
        if self.hard_neg_refresh_every <= 0:
            return
        self.hard_neg_bank = HardNegativeBank(
            num_samples=num_samples,
            d_model=d_model,
            topk=self.hard_neg_topk,
            device=self.device,
        )
        self.refresh_loader_factory = refresh_loader_factory
        if self.rank == 0:
            logger.info(
                "[Stage1] 已挂载 HardNegativeBank: N=%d K=%d refresh_every=%d",
                num_samples, self.hard_neg_topk, self.hard_neg_refresh_every,
            )

    @torch.no_grad()
    def _maybe_refresh_hard_negatives(self, epoch: int) -> None:
        """按 ``s1_hard_neg_refresh_every`` 周期触发 bank 全量刷新 + Top-K 计算。

        触发条件（满足任一即刷新）：

        - 第 1 个 epoch（或 start_epoch，resume 后第一轮）——bank 初次填充
        - 到达 refresh_every 周期
        - bank 尚未 ready（``hard_neg_bin_idx is None``，典型场景：resume 进入
          非周期点的中间 epoch，此时 bank 为空，必须刷新一次才能产出硬负）
        """
        if self.hard_neg_bank is None or self.refresh_loader_factory is None:
            return
        if self.hard_neg_refresh_every <= 0:
            return
        is_scheduled = (
            epoch == getattr(self, "_train_start_epoch", 1)
            or ((epoch - 1) % self.hard_neg_refresh_every == 0)
        )
        is_bank_empty = self.hard_neg_bank.hard_neg_bin_idx is None
        if not is_scheduled and not is_bank_empty:
            return
        if self.rank == 0:
            logger.info(
                "[Stage1] Epoch %d: 开始刷新硬负 bank（cost ≈ 半个训练 epoch）", epoch,
            )
        loader = self.refresh_loader_factory()
        model_for_enc = self._unwrap_model()
        self.hard_neg_bank.refresh(
            model_for_enc, loader, self.device,
            is_distributed=self.is_distributed,
            rank=self.rank, world_size=self.world_size, fp16=self.fp16,
        )
        self.hard_neg_bank.compute_hard_negs(self.device)
        self.model.train()
        if self.rank == 0:
            logger.info("[Stage1] Epoch %d: 硬负 bank Top-K 已就绪", epoch)

    def _unwrap_model(self) -> StrucLift:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _state_dict_for_save(self) -> Dict[str, Any]:
        """仅返回模型权重（兼容旧调用方；新代码请用 :meth:`save_checkpoint`）。"""
        return self._unwrap_model().state_dict()

    # -------------------------------------------------------------------------
    # §完整 checkpoint（含优化器 / scaler / epoch / best_loss / stalled / RNG）
    # -------------------------------------------------------------------------
    _CKPT_VERSION = 2  # bump 当 schema 变化

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        best_loss: float,
        stalled: int,
    ) -> None:
        """保存完整训练状态，便于中断后无缝续训（动量/学习率/早停计数不丢失）。

        产出结构::

            {
                "version": 2,
                "model":     <unwrapped model.state_dict()>,
                "optimizer": <AdamW.state_dict()>,
                "scaler":    <GradScaler.state_dict() 或 None>,
                "epoch":     int（本次刚训完的 epoch 号，resume 从 epoch+1 继续）,
                "best_loss": float,
                "stalled":   int（早停计数器）,
                "rng": {"torch": ..., "cuda": ...},
            }

        同时保存一个仅含 ``model`` 的浅层 snapshot 作 legacy 兼容
        （``{path}.weights_only.pt``）——任何下游只需权重的脚本都能继续用。
        """
        obj = {
            "version": self._CKPT_VERSION,
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "stalled": int(stalled),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }
        _atomic_torch_save(obj, path)

    def load_full_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载 checkpoint，**向后兼容**两种格式：

        1. **新格式（version>=2）**：完整状态，模型/优化器/scaler/进度全部还原。
        2. **旧格式（裸 state_dict）**：只能还原模型权重，优化器/scaler/进度丢失
           （首次升级时不可避免，下次 crash 后 resume 就能无缝）。

        返回值（dict，供训练主循环使用）::

            {
                "start_epoch": int,      # 从该 epoch 开始跑（= saved_epoch + 1）
                "best_loss":   float,    # 早停基线
                "stalled":     int,      # 早停计数器
                "missing":     int,      # load_state_dict 缺失 key 数量
                "unexpected":  int,      # load_state_dict 多余 key 数量
                "is_legacy":   bool,     # 是否是旧格式（只有权重）
            }
        """
        obj = torch.load(path, map_location="cpu", weights_only=False)
        is_new = (
            isinstance(obj, dict)
            and obj.get("version", 0) >= 2
            and "model" in obj
        )

        model_state = obj["model"] if is_new else obj
        missing, unexpected = self._unwrap_model().load_state_dict(
            model_state, strict=False,
        )
        result: Dict[str, Any] = {
            "start_epoch": 1,
            "best_loss": float("inf"),
            "stalled": 0,
            "missing": len(missing),
            "unexpected": len(unexpected),
            "is_legacy": not is_new,
        }
        if not is_new:
            if self.rank == 0:
                logger.info(
                    "[Stage1] 加载 legacy checkpoint（仅权重）：%s；"
                    "优化器/scaler/进度重置",
                    path,
                )
            return result

        # 新格式：继续恢复优化器状态
        try:
            self.optimizer.load_state_dict(obj["optimizer"])
            # 把 optimizer state 里的 tensor 搬到正确 device（AdamW 动量）
            for state in self.optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        except Exception as e:  # noqa: BLE001
            if self.rank == 0:
                logger.warning(
                    "[Stage1] 优化器状态加载失败（%s）；动量将从 0 重新累积", e,
                )

        if self.scaler is not None and obj.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(obj["scaler"])
            except Exception as e:  # noqa: BLE001
                if self.rank == 0:
                    logger.warning("[Stage1] GradScaler 加载失败: %s", e)

        rng = obj.get("rng") or {}
        try:
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        except Exception as e:  # noqa: BLE001
            if self.rank == 0:
                logger.warning("[Stage1] RNG 状态加载失败: %s", e)

        result["start_epoch"] = int(obj.get("epoch", 0)) + 1
        result["best_loss"] = float(obj.get("best_loss", float("inf")))
        result["stalled"] = int(obj.get("stalled", 0))
        if self.rank == 0:
            logger.info(
                "[Stage1] 完整 checkpoint 已恢复：start_epoch=%d best_loss=%.6f "
                "stalled=%d（missing=%d unexpected=%d）",
                result["start_epoch"],
                result["best_loss"],
                result["stalled"],
                result["missing"],
                result["unexpected"],
            )
        return result

    def _compute_region_infonce(
        self,
        bin_node_embs: torch.Tensor,
        src_node_embs: torch.Tensor,
        bin_batch: torch.Tensor,
        src_batch: torch.Tensor,
        binary_regions_list: list,
        source_regions_list: list,
        region_pos_mask_list: list,
    ) -> Optional[torch.Tensor]:
        """§6.2 + §9.2：在每个 batch 内，按样本把区域内节点向量 **mean-pool** 成区域向量，
        然后拼成 `[ΣK_b, d]` 与 `[ΣK_s, d]`，对齐 mask 做 **块对角** 拼接送 `RegionInfoNCELoss`。

        - 跨样本的区域对自动成为负样本（in-batch negatives）。
        - 若 batch 内没有任何样本满足 `K_b>=2 且 K_s>=2 且 alignment_matrix 可用`，返回 None。
        """
        r_bin_parts: list = []
        r_src_parts: list = []
        pos_blocks: list = []

        n_samples = len(binary_regions_list)
        for gi in range(n_samples):
            bin_regs = binary_regions_list[gi]
            src_regs = source_regions_list[gi]
            pos_mask = region_pos_mask_list[gi]
            if (
                not bin_regs
                or not src_regs
                or pos_mask is None
                or len(bin_regs) < 1
                or len(src_regs) < 1
            ):
                continue

            bin_idx_sample = (bin_batch == gi).nonzero(as_tuple=False).flatten()
            src_idx_sample = (src_batch == gi).nonzero(as_tuple=False).flatten()
            if bin_idx_sample.numel() == 0 or src_idx_sample.numel() == 0:
                continue
            bin_embs_i = bin_node_embs.index_select(0, bin_idx_sample)  # [n_b_i, d]
            src_embs_i = src_node_embs.index_select(0, src_idx_sample)  # [n_s_i, d]
            n_b_i = bin_embs_i.size(0)
            n_s_i = src_embs_i.size(0)

            r_b_i: list = []
            for reg in bin_regs:
                nodes = [n for n in reg.get("nodes", []) if 0 <= n < n_b_i]
                if not nodes:
                    continue
                idx = torch.tensor(nodes, device=bin_embs_i.device, dtype=torch.long)
                r_b_i.append(bin_embs_i.index_select(0, idx).mean(dim=0))
            r_s_i: list = []
            for reg in src_regs:
                nodes = [n for n in reg.get("nodes", []) if 0 <= n < n_s_i]
                if not nodes:
                    continue
                idx = torch.tensor(nodes, device=src_embs_i.device, dtype=torch.long)
                r_s_i.append(src_embs_i.index_select(0, idx).mean(dim=0))
            if len(r_b_i) == 0 or len(r_s_i) == 0:
                continue

            r_b_sample = torch.stack(r_b_i, dim=0)  # [Kb_i, d]
            r_s_sample = torch.stack(r_s_i, dim=0)  # [Ks_i, d]
            # pos_mask 的行/列数对应 collate 时的 bin_regs / src_regs，
            # 与这里因越界节点剔除可能导致的 0-节点区域同步裁剪
            pos_mask_t = pos_mask.to(device=self.device, dtype=torch.float32)
            keep_b = [k for k, reg in enumerate(bin_regs)
                      if any(0 <= n < n_b_i for n in reg.get("nodes", []))]
            keep_s = [k for k, reg in enumerate(src_regs)
                      if any(0 <= n < n_s_i for n in reg.get("nodes", []))]
            if pos_mask_t.shape != (len(bin_regs), len(src_regs)):
                continue
            pos_mask_t = pos_mask_t[keep_b][:, keep_s]
            if pos_mask_t.sum() <= 0:
                continue

            r_bin_parts.append(r_b_sample)
            r_src_parts.append(r_s_sample)
            pos_blocks.append(pos_mask_t)

        if not r_bin_parts:
            return None

        r_bin = nn.functional.normalize(torch.cat(r_bin_parts, dim=0), dim=-1)
        r_src = nn.functional.normalize(torch.cat(r_src_parts, dim=0), dim=-1)

        total_Kb = sum(x.size(0) for x in pos_blocks)
        total_Ks = sum(x.size(1) for x in pos_blocks)
        pos_full = torch.zeros(total_Kb, total_Ks, device=self.device, dtype=torch.float32)
        rb, cs = 0, 0
        for blk in pos_blocks:
            pos_full[rb:rb + blk.size(0), cs:cs + blk.size(1)] = blk
            rb += blk.size(0)
            cs += blk.size(1)

        if pos_full.sum() <= 0:
            return None
        return self.region_nce_fn(r_bin, r_src, pos_full)

    def _loss_from_encoder_out(self, enc_out, batch) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """与 train/eval 共享：由 encoder 输出和 batch 组同样的 Stage1 total loss。"""

                loss_pattern = self.pattern_loss_fn(
                    enc_out.pattern_logits,
                    batch["pattern_labels"],
                    weights=batch.get("pattern_weights"),
                )

                h_norm = nn.functional.normalize(enc_out.binary_node_embs, dim=-1)
                loss_neighbor = self.neighbor_loss_fn(
                    h_norm,
                    batch["bin_edge_index"],
                    batch.get("bin_batch"),
                )

                loss = loss_pattern + self.lambda_recon * loss_neighbor

                loss_graph_align = torch.tensor(
                    0.0, device=self.device, requires_grad=False,
                )
                if (
                    self.lambda_graph_align > 0
                    and enc_out.source_graph_emb is not None
                    and enc_out.source_graph_emb.numel() > 0
                ):
            hn_src = None
            hn_bin = None
            if (
                self.hard_neg_bank is not None
                and self.hard_neg_bank.hard_neg_bin_idx is not None
                and batch.get("global_sample_idx") is not None
            ):
                hn_src, hn_bin = self.hard_neg_bank.lookup(
                    batch["global_sample_idx"], self.device,
                )
                    loss_graph_align = self.graph_align_loss_fn(
                        enc_out.binary_graph_emb,
                        enc_out.source_graph_emb,
                func_uid=batch.get("graph_func_uid"),
                hard_neg_src=hn_src,
                hard_neg_bin=hn_bin,
                    )
                    loss = loss + self.lambda_graph_align * loss_graph_align

                loss_region = torch.tensor(
                    0.0, device=self.device, requires_grad=False,
                )
                if (
                    self.lambda_region > 0
                    and enc_out.source_node_embs is not None
            and batch.get("binary_regions_list") is not None
            and batch.get("source_regions_list") is not None
            and batch.get("region_pos_mask_list") is not None
            and batch.get("bin_batch") is not None
            and batch.get("src_batch") is not None
        ):
            region_loss_val = self._compute_region_infonce(
                enc_out.binary_node_embs,
                enc_out.source_node_embs,
                batch["bin_batch"],
                batch["src_batch"],
                batch["binary_regions_list"],
                batch["source_regions_list"],
                batch["region_pos_mask_list"],
            )
            if region_loss_val is not None:
                loss_region = region_loss_val
                    loss = loss + self.lambda_region * loss_region

        return loss, loss_pattern, loss_neighbor, loss_graph_align, loss_region

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        accum = {
            "total": 0.0, "pattern": 0.0, "neighbor": 0.0,
            "graph_align": 0.0, "region": 0.0,
        }
        n_batches = 0

        for batch in dataloader:
            batch = move_binskel_batch_to_device(batch, self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

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
                    run_alignment=False,
                )
                enc_out = out.encoder_out
                (
                    loss,
                    loss_pattern,
                    loss_neighbor,
                    loss_graph_align,
                    loss_region,
                ) = self._loss_from_encoder_out(enc_out, batch)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            accum["total"] += loss.item()
            accum["pattern"] += loss_pattern.item()
            accum["neighbor"] += loss_neighbor.item()
            accum["graph_align"] += loss_graph_align.item()
            accum["region"] += loss_region.item()
            n_batches += 1

        if self.is_distributed and torch.distributed.is_initialized():
            metric_vec = torch.tensor(
                [
                    accum["total"],
                    accum["pattern"],
                    accum["neighbor"],
                    accum["graph_align"],
                    accum["region"],
                    float(n_batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(metric_vec, op=torch.distributed.ReduceOp.SUM)
            accum["total"] = float(metric_vec[0].item())
            accum["pattern"] = float(metric_vec[1].item())
            accum["neighbor"] = float(metric_vec[2].item())
            accum["graph_align"] = float(metric_vec[3].item())
            accum["region"] = float(metric_vec[4].item())
            n_batches = int(metric_vec[5].item())

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    def eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证/测试：不反传、不 step；损失定义与 :meth:`train_epoch` 一致（含 graph/region）。"""
        self.model.eval()
        accum = {
            "total": 0.0, "pattern": 0.0, "neighbor": 0.0,
            "graph_align": 0.0, "region": 0.0,
        }
        n_batches = 0

        for batch in dataloader:
            batch = move_binskel_batch_to_device(batch, self.device, non_blocking=True)
            with torch.inference_mode():
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
                        run_alignment=False,
                    )
                    enc_out = out.encoder_out
                    (
                        loss,
                        loss_pattern,
                        loss_neighbor,
                        loss_graph_align,
                        loss_region,
                    ) = self._loss_from_encoder_out(enc_out, batch)

            accum["total"] += float(loss.item())
            accum["pattern"] += float(loss_pattern.item())
            accum["neighbor"] += float(loss_neighbor.item())
            accum["graph_align"] += float(loss_graph_align.item())
            accum["region"] += float(loss_region.item())
            n_batches += 1

        if self.is_distributed and torch.distributed.is_initialized():
            metric_vec = torch.tensor(
                [
                    accum["total"],
                    accum["pattern"],
                    accum["neighbor"],
                    accum["graph_align"],
                    accum["region"],
                    float(n_batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(metric_vec, op=torch.distributed.ReduceOp.SUM)
            accum["total"] = float(metric_vec[0].item())
            accum["pattern"] = float(metric_vec[1].item())
            accum["neighbor"] = float(metric_vec[2].item())
            accum["graph_align"] = float(metric_vec[3].item())
            accum["region"] = float(metric_vec[4].item())
            n_batches = int(metric_vec[5].item())

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = "checkpoints/stage1",
        start_epoch: int = 1,
        best_loss: float = float("inf"),
        stalled: int = 0,
    ) -> None:
        """主训练循环。

        支持续训：调用者（通常是 ``scripts/train_stage1_binskel.py``）先调
        :meth:`load_full_checkpoint` 拿到 ``start_epoch / best_loss / stalled``，再
        把这三者传进来即可无缝续训（含优化器动量、早停计数）。

        Parameters
        ----------
        start_epoch
            从哪个 epoch 开始训练（1-based，resume 时 = saved_epoch + 1）。
        best_loss
            当前已知最佳 **选择用** loss（用于早停与 ``best_stage1.pt``）。
            若提供了 ``val_loader``，则为 **验证集** ``total``；否则为 **训练集** ``total``。
            从「仅训集早停」改为「验证集早停」续训时，请用脚本 ``--reset-selection-baseline``
            将基线置 ``inf``，避免与旧训集 best 混用。
        stalled
            早停计数器（连续多少个 epoch 未改善）。
        """
        os.makedirs(save_dir, exist_ok=True)
        tc = self.config.training
        patience = int(getattr(tc, "s1_early_stop_patience", 0) or 0)
        min_delta = float(getattr(tc, "s1_early_stop_min_delta", 1e-4))
        min_epochs = int(getattr(tc, "s1_early_stop_min_epochs", 5))
        total_epochs = int(self.config.training.s1_epochs)

        # 供 _maybe_refresh_hard_negatives 判断"首轮 refresh"时使用（resume 后
        # 首轮可能不是 epoch 1，但仍需刷新 bank）
        self._train_start_epoch = int(start_epoch)

        if self.rank == 0 and start_epoch > 1:
            logger.info(
                "[Stage1] 续训：start_epoch=%d best_loss=%.6f stalled=%d",
                start_epoch,
                best_loss,
                stalled,
            )

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, total_epochs + 1):
            last_epoch = epoch
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            self._maybe_refresh_hard_negatives(epoch)
            metrics = self.train_epoch(train_loader)
            val_metrics: Optional[Dict[str, float]] = None
            if val_loader is not None and len(val_loader) > 0:
                val_metrics = self.eval_epoch(val_loader)

            if self.rank == 0:
            logger.info(
                    f"[Stage1] Epoch {epoch}/{total_epochs}  train "
                f"loss={metrics['total']:.4f}  "
                f"pattern={metrics['pattern']:.4f}  "
                f"neighbor={metrics['neighbor']:.4f}  "
                f"graph_align={metrics['graph_align']:.4f}  "
                f"region={metrics['region']:.4f}"
                )
                if val_metrics is not None:
                    logger.info(
                        f"[Stage1] Epoch {epoch}/{total_epochs}  val   "
                        f"loss={val_metrics['total']:.4f}  "
                        f"pattern={val_metrics['pattern']:.4f}  "
                        f"neighbor={val_metrics['neighbor']:.4f}  "
                        f"graph_align={val_metrics['graph_align']:.4f}  "
                        f"region={val_metrics['region']:.4f}"
                    )

            cur = (
                val_metrics["total"]
                if val_metrics is not None
                else metrics["total"]
            )
            if cur < best_loss - min_delta:
                best_loss = cur
                stalled = 0
                if self.rank == 0:
                    self.save_checkpoint(
                    os.path.join(save_dir, "best_stage1.pt"),
                        epoch=epoch,
                        best_loss=best_loss,
                        stalled=stalled,
                    )
                    sel = "val" if val_metrics is not None else "train"
                    logger.info(
                        "[Stage1] 已更新 best_stage1.pt（按 %s total, best_loss=%.6f）",
                        sel,
                        best_loss,
                    )
            else:
                stalled += 1

            # 每 epoch 结束保存 latest：中断后 --resume 可从最近一次完整 epoch 续训
            # 完整状态（权重 + 优化器 + 早停计数）由 save_checkpoint 写入
            if self.rank == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, "latest_stage1.pt"),
                    epoch=epoch,
                    best_loss=best_loss,
                    stalled=stalled,
                )
                meta_path = os.path.join(save_dir, "latest_stage1_meta.json")
                meta = {
                    "epoch": epoch,
                    "train_loss": float(metrics["total"]),
                    "val_loss": (
                        float(val_metrics["total"]) if val_metrics is not None else None
                    ),
                    "selection_metric": (
                        "val_total" if val_metrics is not None else "train_total"
                    ),
                    "best_loss_so_far": float(best_loss),
                    "stalled": int(stalled),
                    "ckpt_version": self._CKPT_VERSION,
                }
                fd, tmp = tempfile.mkstemp(suffix=".json", dir=save_dir)
                try:
                    os.close(fd)
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    os.replace(tmp, meta_path)
                except Exception:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                    raise

            if (
                patience > 0
                and epoch >= min_epochs
                and stalled >= patience
            ):
                if self.rank == 0:
                    logger.info(
                        "[Stage1] Early stopping: no improvement > %.1e for %d epochs "
                        "(after epoch %d).",
                        min_delta,
                        patience,
                        epoch,
                    )
                break

        if self.rank == 0:
            self.save_checkpoint(
            os.path.join(save_dir, "final_stage1.pt"),
                epoch=last_epoch,
                best_loss=best_loss,
                stalled=stalled,
            )
            sel = "val total" if val_loader is not None else "train total"
            logger.info(
                f"[Stage1] Training complete. Best {sel} loss: {best_loss:.4f}",
            )

"""
§9.2 硬负样本挖掘：Memory Bank + 每 N epoch 全量刷新。

设计要点
--------
1. **Bank**：同时维护 ``bank_bin [N, d]``、``bank_src [N, d]``（L2 归一化、fp16）
   以及 ``func_uids [N]``、``valid [N]``。
2. **Refresh**：使用 ``DistributedSampler`` 对全数据集切片，每个 rank forward 自己
   那一份，随后 ``all_gather`` 拼成完整 bank。整个过程 ``torch.no_grad()``。
3. **Top-K**：分块计算 ``sim = bank_bin @ bank_bin.T``（或 bin↔src），用
   ``func_uid`` 的对称关系把同函数位置 mask 成 ``-inf``，再 ``topk``。
4. **Lookup**：训练 step 里按 ``global_sample_idx`` 索引出每个 anchor 的 Top-K
   硬负 embedding（[B, K, d]），送进 ``GraphBinarySourceContrastiveLoss``。

DDP 语义
--------
* Bank 在每个 rank 上**复制**一份（fp16 下 36w × 256 × 2B ≈ 180MB，可接受）。
* ``all_gather`` 只在 refresh 时发生，后续 lookup 全在本地 GPU/CPU 完成。
* 若 ``world_size == 1``，跳过 all_gather 直接填 bank。

限制
----
* **仅支持单 JSONL 文件**（``BinSkelJsonlDataset``，非 ``ConcatDataset``）——
  因为 Top-K 要求全局样本 idx 是连续 ``[0, N)``。多文件数据集需要额外的 idx
  平移逻辑，目前不支持；训练脚本会在启用硬负时检查并报错。
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HardNegativeBank:
    """StrucLift §9.2 硬负样本 memory bank。

    Parameters
    ----------
    num_samples : int
        数据集总样本数（即 :func:`BinSkelJsonlDataset.__len__`）。
    d_model : int
        graph-level embedding 维度（与 :class:`StrucLift` 的 ``d_model`` 一致）。
    topk : int
        每个 anchor 保留的 Top-K 硬负数量（建议 16-64）。
    device : torch.device
        Top-K 计算时临时上 GPU 的目标设备；bank 本体长期存 CPU（fp16）。
    """

    def __init__(
        self,
        num_samples: int,
        d_model: int,
        topk: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.N = int(num_samples)
        self.d = int(d_model)
        self.K = int(topk)
        self.device = device or torch.device("cpu")

        # 常驻 CPU：避免训练显存被占；lookup 时按需搬 GPU
        self.bank_bin = torch.zeros(self.N, self.d, dtype=torch.float16)
        self.bank_src = torch.zeros(self.N, self.d, dtype=torch.float16)
        self.func_uids = torch.full((self.N,), -1, dtype=torch.long)
        self.valid = torch.zeros(self.N, dtype=torch.bool)

        # Top-K 结果（每 refresh 更新一次）
        self.hard_neg_bin_idx: Optional[torch.Tensor] = None  # [N, K]
        self.hard_neg_src_idx: Optional[torch.Tensor] = None  # [N, K]

    # ────────────────────────────── refresh ──────────────────────────────
    @torch.no_grad()
    def refresh(
        self,
        model: torch.nn.Module,
        refresh_loader: "torch.utils.data.DataLoader",
        device: torch.device,
        *,
        is_distributed: bool,
        rank: int,
        world_size: int,
        fp16: bool = True,
    ) -> None:
        """对全数据集 forward 一遍，收集 graph-level embedding 填充 bank。

        ``refresh_loader`` 要求：
          - 使用 :class:`BinSkelJsonlDataset`（单文件）
          - ``shuffle=False``、**非 pair 模式**
          - 带 :class:`DistributedSampler`（多卡 DDP）或普通顺序（单卡）
          - collate 输出含 ``global_sample_idx``、``graph_func_uid``
        """
        model.eval()
        # §DDP 同步：两个 rank 在进入 refresh 主循环前先 barrier 一次，避免一个 rank
        # 因模型加载 / cudnn autotune / 第一批数据 IO 慢导致另一个 rank 先跑完循环
        # 直接撞到后面的 all_reduce，进而在默认 10min watchdog 下死锁。
        if is_distributed and dist.is_initialized() and world_size > 1:
            dist.barrier()
        local_bin_list: list = []
        local_src_list: list = []
        local_idx_list: list = []
        local_uid_list: list = []
        n_batches = 0
        n_samples_seen = 0

        for batch in refresh_loader:
            # 只移动编码必需张量，避免无谓 IO
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=(fp16 and device.type == "cuda")):
                enc = model.encode(
                    bin_opcodes=batch_dev["bin_opcodes"],
                    bin_operand_types=batch_dev["bin_operand_types"],
                    bin_operand_values=batch_dev["bin_operand_values"],
                    bin_block_lengths=batch_dev["bin_block_lengths"],
                    bin_edge_index=batch_dev["bin_edge_index"],
                    bin_edge_type=batch_dev["bin_edge_type"],
                    bin_struct_features=batch_dev["bin_struct_features"],
                    bin_batch=batch_dev.get("bin_batch"),
                    src_token_ids=batch_dev.get("src_token_ids"),
                    src_stmt_lengths=batch_dev.get("src_stmt_lengths"),
                    src_edge_index=batch_dev.get("src_edge_index"),
                    src_edge_type=batch_dev.get("src_edge_type"),
                    src_ast_types=batch_dev.get("src_ast_types"),
                    src_compound_id=batch_dev.get("src_compound_id"),
                    src_batch=batch_dev.get("src_batch"),
                    run_alignment=False,
                )
            if enc.binary_graph_emb is None or enc.source_graph_emb is None:
                raise RuntimeError(
                    "HardNegativeBank.refresh 需要 encode() 返回 binary/source "
                    "graph_emb；请确保数据集含源码侧（require_source=True）",
                )
            bin_emb = F.normalize(enc.binary_graph_emb.float(), dim=-1).half().cpu()
            src_emb = F.normalize(enc.source_graph_emb.float(), dim=-1).half().cpu()
            gidx = batch_dev["global_sample_idx"].cpu().long()
            uid = batch_dev["graph_func_uid"].cpu().long()
            local_bin_list.append(bin_emb)
            local_src_list.append(src_emb)
            local_idx_list.append(gidx)
            local_uid_list.append(uid)
            n_batches += 1
            n_samples_seen += bin_emb.size(0)

        # §DDP：所有 rank 协商"是否全体都为空"；若只有部分 rank 为空，用空张量继续
        # 走 all_reduce / all_gather 流程，避免"空 rank 提前 return → 另 rank 死等"
        # 的死锁模式。
        if n_batches == 0:
            logger.warning(
                "[HardNeg] refresh 收到 0 个 batch（rank=%d）；将以空切片参与集合通信",
                rank,
            )
            local_bin = torch.zeros(0, self.d, dtype=torch.float16)
            local_src = torch.zeros(0, self.d, dtype=torch.float16)
            local_idx = torch.zeros(0, dtype=torch.long)
            local_uid = torch.zeros(0, dtype=torch.long)
        else:
            local_bin = torch.cat(local_bin_list, dim=0)
            local_src = torch.cat(local_src_list, dim=0)
            local_idx = torch.cat(local_idx_list, dim=0)
            local_uid = torch.cat(local_uid_list, dim=0)

        if is_distributed and dist.is_initialized() and world_size > 1:
            total_batches = torch.tensor([n_batches], device=device, dtype=torch.long)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
            if int(total_batches.item()) == 0:
                if rank == 0:
                    logger.warning("[HardNeg] 所有 rank 都未收到 batch，refresh 跳过")
                return
        elif n_batches == 0:
            return

        if is_distributed and dist.is_initialized() and world_size > 1:
            # 各 rank 的 local batch 数可能不同，先填 NaN pad 到 max
            max_n = torch.tensor([local_bin.size(0)], device=device, dtype=torch.long)
            dist.all_reduce(max_n, op=dist.ReduceOp.MAX)
            max_n_int = int(max_n.item())
            pad = max_n_int - local_bin.size(0)
            if pad > 0:
                local_bin = torch.cat(
                    [local_bin, torch.zeros(pad, self.d, dtype=torch.float16)], dim=0,
                )
                local_src = torch.cat(
                    [local_src, torch.zeros(pad, self.d, dtype=torch.float16)], dim=0,
                )
                local_idx = torch.cat(
                    [local_idx, torch.full((pad,), -1, dtype=torch.long)], dim=0,
                )
                local_uid = torch.cat(
                    [local_uid, torch.full((pad,), -1, dtype=torch.long)], dim=0,
                )
            # all_gather（只能 gather GPU 张量）
            gathered_bin = [
                torch.zeros_like(local_bin, device=device) for _ in range(world_size)
            ]
            gathered_src = [
                torch.zeros_like(local_src, device=device) for _ in range(world_size)
            ]
            gathered_idx = [
                torch.zeros_like(local_idx, device=device) for _ in range(world_size)
            ]
            gathered_uid = [
                torch.zeros_like(local_uid, device=device) for _ in range(world_size)
            ]
            dist.all_gather(gathered_bin, local_bin.to(device))
            dist.all_gather(gathered_src, local_src.to(device))
            dist.all_gather(gathered_idx, local_idx.to(device))
            dist.all_gather(gathered_uid, local_uid.to(device))
            all_bin = torch.cat([t.cpu() for t in gathered_bin], dim=0)
            all_src = torch.cat([t.cpu() for t in gathered_src], dim=0)
            all_idx = torch.cat([t.cpu() for t in gathered_idx], dim=0)
            all_uid = torch.cat([t.cpu() for t in gathered_uid], dim=0)
        else:
            all_bin, all_src, all_idx, all_uid = local_bin, local_src, local_idx, local_uid

        # 按 global_sample_idx 回填 bank；-1 表示 padding，跳过
        valid_mask = (all_idx >= 0) & (all_idx < self.N)
        gi = all_idx[valid_mask]
        self.bank_bin.index_copy_(0, gi, all_bin[valid_mask])
        self.bank_src.index_copy_(0, gi, all_src[valid_mask])
        self.func_uids.index_copy_(0, gi, all_uid[valid_mask])
        self.valid.index_fill_(0, gi, True)

        if rank == 0:
            filled = int(self.valid.sum().item())
            logger.info(
                "[HardNeg] refresh 完成：bank 已填 %d/%d (%.1f%%) 样本",
                filled, self.N, 100.0 * filled / max(self.N, 1),
            )

    # ──────────────────────────── compute Top-K ───────────────────────────
    @torch.no_grad()
    def compute_hard_negs(
        self,
        device: torch.device,
        *,
        chunk_size: int = 2048,
    ) -> None:
        """分块计算 Top-K 硬负 idx，结果存入 ``self.hard_neg_{bin,src}_idx``。

        对每个 anchor i，硬负定义为：
          * 与 i 不同 ``func_uid``（排除同一函数不同编译产物）
          * 相似度（归一化内积）最高的 K 个样本
        """
        if not bool(self.valid.any()):
            raise RuntimeError("bank 为空，请先 refresh 后再 compute_hard_negs")

        # 把 bank 整体上 GPU 以加速 matmul（fp16，180MB 可承受）
        bank_bin_gpu = self.bank_bin.to(device, dtype=torch.float16)
        bank_src_gpu = self.bank_src.to(device, dtype=torch.float16)
        uids_gpu = self.func_uids.to(device, dtype=torch.long)
        valid_gpu = self.valid.to(device, dtype=torch.bool)

        hard_bin = torch.zeros(self.N, self.K, dtype=torch.long)
        hard_src = torch.zeros(self.N, self.K, dtype=torch.long)
        neg_inf = torch.tensor(-1e4, dtype=torch.float16, device=device)

        n_chunks = math.ceil(self.N / chunk_size)
        for c in range(n_chunks):
            a = c * chunk_size
            b = min(a + chunk_size, self.N)
            # bin ↔ bin  → 每个 bin anchor 的硬负 bin 候选；但最终我们要的是
            # 每个 **bin_i** 最像的 **src_j**（bin 要区分哪些 src 最容易混淆）。
            # 按 §9.2 写作：硬负是 "最容易被误认为正样本的样本"，所以 anchor 是
            # bin_i，候选是 src_j；同理反方向 anchor 是 src_i，候选是 bin_j。
            sim_b2s = bank_bin_gpu[a:b] @ bank_src_gpu.t()          # [chunk, N]
            sim_s2b = bank_src_gpu[a:b] @ bank_bin_gpu.t()          # [chunk, N]
            # mask 同函数 + 无效样本
            chunk_uids = uids_gpu[a:b].unsqueeze(1)                 # [chunk, 1]
            same_func = chunk_uids == uids_gpu.unsqueeze(0)         # [chunk, N]
            invalid = ~valid_gpu.unsqueeze(0).expand_as(sim_b2s)    # [chunk, N]
            bad = same_func | invalid
            sim_b2s = sim_b2s.masked_fill(bad, neg_inf)
            sim_s2b = sim_s2b.masked_fill(bad, neg_inf)
            _, topk_b = sim_b2s.topk(self.K, dim=1)                 # [chunk, K]
            _, topk_s = sim_s2b.topk(self.K, dim=1)
            hard_bin[a:b] = topk_b.cpu()
            hard_src[a:b] = topk_s.cpu()

        self.hard_neg_bin_idx = hard_bin   # [N, K]: 每个 i 最像自己的 src_j 下标
        self.hard_neg_src_idx = hard_src   # [N, K]: 每个 i 最像自己的 bin_j 下标
        # 释放 GPU
        del bank_bin_gpu, bank_src_gpu, uids_gpu, valid_gpu

    # ─────────────────────────────── lookup ───────────────────────────────
    @torch.no_grad()
    def lookup(
        self,
        global_sample_idx: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """取 batch 内每个 anchor 的 Top-K 硬负 embedding。

        Returns
        -------
        hard_neg_src : [B, K, d]  —— 给 bin anchor 用的 src 硬负
        hard_neg_bin : [B, K, d]  —— 给 src anchor 用的 bin 硬负

        若尚未 ``compute_hard_negs``，返回空 tensor（调用方应检查 ``numel()``）。
        """
        if self.hard_neg_bin_idx is None or self.hard_neg_src_idx is None:
            return (
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            )
        gidx = global_sample_idx.cpu().long().clamp(min=0, max=self.N - 1)
        neg_src_idx = self.hard_neg_bin_idx.index_select(0, gidx)   # [B, K]
        neg_bin_idx = self.hard_neg_src_idx.index_select(0, gidx)   # [B, K]
        # gather embedding
        neg_src_emb = self.bank_src.index_select(0, neg_src_idx.view(-1))
        neg_bin_emb = self.bank_bin.index_select(0, neg_bin_idx.view(-1))
        neg_src_emb = neg_src_emb.view(gidx.size(0), self.K, self.d).to(device)
        neg_bin_emb = neg_bin_emb.view(gidx.size(0), self.K, self.d).to(device)
        return neg_src_emb.float(), neg_bin_emb.float()

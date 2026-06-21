"""
Structural losses for StrucLift — Stage 1（StrucLift_Research_Proposal.md §5.5）.

* PatternClassificationLoss — §5.5 ``L_pattern_cls``
* NeighborReconLoss — §5.5 ``L_neighbor_recon``（邻居相似自监督；实现为行 softmax / InfoNCE 形式）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatternClassificationLoss(nn.Module):
    """§5.5 模式分类损失（与原文符号一致）。

    .. math::
        \\mathcal{L}_{\\text{pattern\\_cls}}
        = -\\frac{1}{n}\\sum_i w_i \\sum_c y_{ic} \\log p_{ic}

    其中 :math:`w_i` 为 DWARF 标签置信度（实现为可选 ``weights``）。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [N, C]  raw pattern logits
        targets : [N]     int64 class labels (0 .. C-1)
        weights : [N]     per-node confidence weights (optional)

        Returns
        -------
        scalar loss
        """
        ce = F.cross_entropy(logits, targets, reduction="none")  # [N]
        if weights is not None:
            ce = ce * weights
        return ce.mean()


class NeighborReconLoss(nn.Module):
    """§5.5 ``L_neighbor_recon``（原文：CFG 邻居节点应有相似表示）。

    实现：对 ``h`` 做 L2 归一化后，沿 **有向** 边 ``src→dst`` 将 ``(src,dst)`` 视为正样本对，
    最大化 log-softmax；分母在**同一函数图**内归一化（``bin_batch``）。
    不再添加反向边：仅控制流箭头上的后继关系参与对齐。
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        node_embs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_embs  : [N, d]  L2-normalised node embeddings
        edge_index : [2, E]  COO edge list
        batch      : [N]     optional graph id per node（与 §5.1 中 batched CFG 一致）

        Returns
        -------
        scalar loss
        """
        N = node_embs.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=node_embs.device, requires_grad=True)

        src, dst = edge_index[0], edge_index[1]
        # 自环在 sim 上被对角 mask 掉，若仍当作正边会得到 log p ≈ -1e9、损失爆炸；跨图边同理。
        ok = src != dst
        if batch is not None and batch.numel() == N:
            ok = ok & (batch[src] == batch[dst])
        edge_index = edge_index[:, ok]
        if edge_index.numel() == 0:
            return torch.tensor(0.0, device=node_embs.device, requires_grad=True)

        sim = torch.mm(node_embs, node_embs.t()) / self.tau

        # softmax 掩码：须与 sim.dtype 一致；float16 无法表示 -1e9（会 overflow）
        neg_inf = torch.finfo(sim.dtype).min

        if batch is not None and batch.numel() == N:
            same_g = batch.unsqueeze(0) == batch.unsqueeze(1)
            sim = sim.masked_fill(~same_g, neg_inf)

        diag_mask = torch.eye(N, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(diag_mask, neg_inf)

        log_softmax = sim - sim.logsumexp(dim=1, keepdim=True)
        log_softmax = torch.nan_to_num(log_softmax, nan=0.0)

        src, dst = edge_index[0], edge_index[1]
        pos_logprobs = log_softmax[src.long(), dst.long()]

        if pos_logprobs.numel() == 0:
            return torch.tensor(0.0, device=node_embs.device, requires_grad=True)

        return -pos_logprobs.mean()


class GraphBinarySourceContrastiveLoss(nn.Module):
    """可选图级 bin–src 对比（原文 §5.5 公式仅两项；用于训练源码编码器时开启）。

    同一 batch 内第 *i* 个样本的二进制图向量与第 *i* 个样本的源码图向量为正对；
    其余为负对（对称 InfoNCE）。

    * ``B >= 2``：对称 InfoNCE。
    * ``B == 1``：退化为 ``1 - cos(z_bin, z_src)``。

    **§9.2 同函数处理**（需传 ``func_uid`` [B]）：

    * ``mode="mask_neg"``（默认）：**伪负过滤**——同 ``func_uid`` 的非对角位置
      mask 成 ``-inf``，不作为负样本，对角线（真正对）保持。
    * ``mode="multi_positive"``：**跨优化正对**——同 ``func_uid`` 的**所有**位置
      （含对角）作为正样本；使用 multi-positive InfoNCE：
      :math:`\\mathcal{L}_i = -\\frac{1}{|P(i)|}\\sum_{j\\in P(i)} \\log
      \\frac{\\exp(s_{ij}/\\tau)}{\\sum_k \\exp(s_{ik}/\\tau)}`。

    Parameters
    ----------
    tau : float
        Temperature（与 ``s1_infonce_tau`` 一致即可）。
    mode : str
        ``"mask_neg"`` or ``"multi_positive"``（默认 ``mask_neg``）。
    """

    def __init__(self, tau: float = 0.07, *, mode: str = "mask_neg") -> None:
        super().__init__()
        if mode not in ("mask_neg", "multi_positive"):
            raise ValueError(f"unknown mode: {mode}")
        self.tau = tau
        self.mode = mode

    def forward(
        self,
        z_bin: torch.Tensor,
        z_src: torch.Tensor,
        func_uid: torch.Tensor | None = None,
        *,
        hard_neg_src: torch.Tensor | None = None,
        hard_neg_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_bin : [B, d]
        z_src : [B, d]
        func_uid : [B] optional，同 batch 内样本的函数唯一 id（来自
            ``collate_binskel_batch`` 的 ``graph_func_uid``）。
        hard_neg_src : [B, K, d] optional，§9.2 硬负——给每个 bin anchor 的
            Top-K 最难 src 负样本 embedding（来自 :class:`HardNegativeBank`）。
            需已 L2 归一化。
        hard_neg_bin : [B, K, d] optional，给每个 src anchor 的 Top-K 最难 bin
            负样本（同上）。

        Returns
        -------
        scalar
        """
        z_bin = F.normalize(z_bin, dim=-1)
        z_src = F.normalize(z_src, dim=-1)
        b = z_bin.size(0)
        if b == 1:
            return (1.0 - (z_bin * z_src).sum(dim=-1)).mean()
        logits = torch.mm(z_bin, z_src.t()) / self.tau  # [B, B]

        # §9.2 硬负：把 bank 里的 top-K 硬负拼到 logits 右侧形成 [B, B+K]
        use_hn_src = (
            hard_neg_src is not None
            and hard_neg_src.numel() > 0
            and hard_neg_src.size(0) == b
        )
        use_hn_bin = (
            hard_neg_bin is not None
            and hard_neg_bin.numel() > 0
            and hard_neg_bin.size(0) == b
        )
        hn_cols_b2s = None
        hn_cols_s2b = None
        if use_hn_src:
            hn_src = F.normalize(hard_neg_src, dim=-1)  # [B, K, d]
            # 每个 anchor 的 K 个硬负：[B, K]
            hn_cols_b2s = torch.einsum("bd,bkd->bk", z_bin, hn_src) / self.tau
        if use_hn_bin:
            hn_bin = F.normalize(hard_neg_bin, dim=-1)
            hn_cols_s2b = torch.einsum("bd,bkd->bk", z_src, hn_bin) / self.tau

        has_uid = func_uid is not None and func_uid.numel() == b
        if has_uid and self.mode == "multi_positive":
            uid = func_uid.to(device=logits.device).view(-1)  # type: ignore[union-attr]
            same_func = uid.unsqueeze(0) == uid.unsqueeze(1)  # [B, B], bool
            pos = same_func.float()  # [B, B]
            # 拼接硬负列：硬负不是正对（pos=0）
            if hn_cols_b2s is not None:
                logits_b2s = torch.cat([logits, hn_cols_b2s], dim=1)          # [B, B+K]
                pos_b2s = torch.cat([pos, torch.zeros_like(hn_cols_b2s)], dim=1)
            else:
                logits_b2s = logits
                pos_b2s = pos
            if hn_cols_s2b is not None:
                logits_s2b = torch.cat([logits.t(), hn_cols_s2b], dim=1)      # [B, B+K]
                pos_s2b = torch.cat([pos.t(), torch.zeros_like(hn_cols_s2b)], dim=1)
            else:
                logits_s2b = logits.t()
                pos_s2b = pos.t()
            log_sm_row = logits_b2s - logits_b2s.logsumexp(dim=1, keepdim=True)
            log_sm_col = logits_s2b - logits_s2b.logsumexp(dim=1, keepdim=True)
            num_pos_row = pos_b2s.sum(dim=1).clamp(min=1.0)
            num_pos_col = pos_s2b.sum(dim=1).clamp(min=1.0)
            loss_b2s = -(log_sm_row * pos_b2s).sum(dim=1) / num_pos_row
            loss_s2b = -(log_sm_col * pos_s2b).sum(dim=1) / num_pos_col
            return 0.5 * (loss_b2s.mean() + loss_s2b.mean())

        if has_uid:  # mode == "mask_neg"
            uid = func_uid.to(device=logits.device).view(-1)  # type: ignore[union-attr]
            same_func = uid.unsqueeze(0) == uid.unsqueeze(1)
            eye = torch.eye(b, dtype=torch.bool, device=logits.device)
            pseudo_neg = same_func & ~eye
            if bool(pseudo_neg.any()):
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(pseudo_neg, neg_inf)

        # 拼接硬负后走标准 CE（硬负视为额外负样本列）
        if hn_cols_b2s is not None:
            logits_b2s = torch.cat([logits, hn_cols_b2s], dim=1)
        else:
            logits_b2s = logits
        if hn_cols_s2b is not None:
            logits_s2b = torch.cat([logits.t(), hn_cols_s2b], dim=1)
        else:
            logits_s2b = logits.t()
        labels = torch.arange(b, device=z_bin.device, dtype=torch.long)
        loss_b2s = F.cross_entropy(logits_b2s, labels)
        loss_s2b = F.cross_entropy(logits_s2b, labels)
        return 0.5 * (loss_b2s + loss_s2b)


NeighborContrastiveLoss = NeighborReconLoss
NeighborReconstructionLoss = NeighborReconLoss

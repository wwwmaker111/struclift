"""
Alignment losses for StrucLift Stage 2.

* SCOTAlignmentLoss        — weighted KL between predicted and GT alignment
* EdgeRecoveryLoss         — KL on normalised source-edge mass (A_s_hat vs A_s^gt)
* CrossAttentionAlignmentLoss — KL between cross-attn weights and GT alignment
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCOTAlignmentLoss(nn.Module):
    """Weighted KL divergence between the SCOT-predicted alignment matrix
    and the DWARF ground-truth alignment.

    L = Σ_{i,j} w_{ij} · M_gt_{ij} · log( M_gt_{ij} / (M_{ij} + ε) )

    Parameters
    ----------
    eps : float
        Small constant for numerical stability (default 1e-8).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        M_pred: torch.Tensor,
        M_gt: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        M_pred     : [n_b, n_s]  predicted soft alignment (from SCOT).
                     受行边际约束 μ_i=1/n_b，若再开 §6.3.7 dummy sink，截掉 dummy
                     列后实际行和 ≤ 1/n_b 且逐行不等。直接对其取 log 与 M_gt 算 KL
                     会带 log(n_b) + 的常数偏移，损失永远降不到 0。
        M_gt       : [n_b, n_s]  ground-truth alignment (约定每行和 ≈ 1，但容错)
        confidence : [n_b, n_s]  per-element weight (DWARF confidence), optional

        Returns
        -------
        scalar loss
        """
        # FP16 下 1e-8 与极小概率会下溢为 0，log 变 -inf → NaN；KL 在 float32 中算
        Mp = M_pred.float()
        Mg = M_gt.float()
        conf = confidence.float() if confidence is not None else None

        # §6.6 尺度对齐：把两侧都按行归一化到 P(src|bb) 的条件分布语义
        # —— 消掉 μ=1/n_b 的 log(n_b) 偏移和 dummy sink 吸收的额外质量。
        row_b = Mp.sum(dim=-1, keepdim=True)
        row_g = Mg.sum(dim=-1, keepdim=True)
        nb, ns = Mp.shape
        u_b = torch.full((nb, 1), 1.0 / max(ns, 1), device=Mp.device, dtype=Mp.dtype)
        u_g = u_b
        Mp = torch.where(row_b > self.eps, Mp / row_b, u_b.expand_as(Mp))
        Mg = torch.where(row_g > self.eps, Mg / row_g, u_g.expand_as(Mg))

        M_gt_safe = Mg.clamp(min=self.eps)
        M_pred_safe = Mp.clamp(min=self.eps)

        kl = M_gt_safe * (M_gt_safe.log() - M_pred_safe.log())

        if conf is not None:
            kl = kl * conf

        out = kl.sum() / max(M_gt.size(0), 1)
        if not torch.isfinite(out):
            return (Mp * 0.0).sum()
        return out


class EdgeRecoveryLoss(nn.Module):
    """Match predicted vs ground-truth **distribution of source-edge mass**.

    ``A_s_hat = M^T A_b M`` is typically ≪1 entrywise under transport marginals;
    BCE to 0/1 targets is unattainable.  We form two distributions on the
    ``[n_s, n_s]`` grid and minimise ``KL( norm(GT) || norm(pred) )``.

    Predicted:  A_s_hat = M^T · A_b · M
    """

    def __init__(self, mass_eps: float = 1e-12):
        super().__init__()
        self.mass_eps = float(mass_eps)

    def forward(
        self,
        M: torch.Tensor,
        adj_bin: torch.Tensor,
        adj_src_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        M          : [n_b, n_s]  alignment matrix
        adj_bin    : [n_b, n_b]  binary adjacency (0/1)
        adj_src_gt : [n_s, n_s]  ground-truth source adjacency (0/1)

        Returns
        -------
        scalar loss
        """
        # fp32：避免 M^T A M 在 fp16 下乘两次后下溢为全 0。
        M32 = M.float()
        adj_src_pred = M32.t() @ adj_bin.float() @ M32
        gt = adj_src_gt.float()
        s_gt = gt.sum()
        s_pr = adj_src_pred.sum()
        # 任一侧非有限则跳过（否则 KL / 除法 会产生 inf/NaN，把整批 total loss 撑爆）
        if not (torch.isfinite(s_gt) and torch.isfinite(s_pr)):
            return (M32 * 0.0).sum()
        if s_gt < self.mass_eps:
            return (M32 * 0.0).sum()
        if s_pr < self.mass_eps:
            # 预测总质量塌成 0：KL 无定义，用与 GT 总边「质量」的 MSE 拉回（对 M 仍有梯度）
            mse = F.mse_loss(s_pr, s_gt)
            if not torch.isfinite(mse):
                return (M32 * 0.0).sum()
            return mse
        P = adj_src_pred / s_pr
        Q = gt / s_gt
        # 比 mass_eps(1e-12) 略大的下界，避免 log 里出现 -inf * q → inf/NaN
        log_floor = max(self.mass_eps, 1e-8)
        p_safe = P.flatten().clamp(min=log_floor)
        q = Q.flatten()
        kl = torch.where(
            q > 1e-12,
            q * (q.log() - p_safe.log()),
            torch.zeros_like(q),
        ).sum()
        if not torch.isfinite(kl):
            return (M32 * 0.0).sum()
        out = torch.clamp(kl, max=1e3)
        # clamp 不消除 NaN；若仍非有限则跳过该子图对 edge loss 的贡献
        if not torch.isfinite(out):
            return (M32 * 0.0).sum()
        return out


class CrossAttentionAlignmentLoss(nn.Module):
    """Encourage cross-attention weights to be consistent with the SCOT
    alignment matrix.

    L = KL( M_gt || A_cross )

    where A_cross is the average attention weight from the bidirectional
    cross-attention layers.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        attn_weights: torch.Tensor,
        M_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        attn_weights : [n_b, n_s]  averaged cross-attention weights
        M_gt         : [n_b, n_s]  ground-truth alignment (normalised)

        Returns
        -------
        scalar loss
        """
        Mg = M_gt.float()
        Aw = attn_weights.float()
        M_gt_safe = Mg.clamp(min=self.eps)
        attn_safe = Aw.clamp(min=self.eps)

        kl = M_gt_safe * (M_gt_safe.log() - attn_safe.log())
        out = kl.sum() / max(M_gt.size(0), 1)
        if not torch.isfinite(out):
            return (Mg * 0.0).sum()
        return out


class RefinedEmbeddingAlignmentLoss(nn.Module):
    """Supervise the cross-attention refined embeddings so that their
    pairwise cosine similarity matches the ground-truth alignment matrix.

    This ensures that cross-attention parameters receive gradients during
    Stage 2 training.

    L = MSE( softmax(H_hat_b @ H_hat_s^T / tau),  M_gt_row_normalised )
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        H_hat_b: torch.Tensor,
        H_hat_s: torch.Tensor,
        M_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        H_hat_b : [n_b, d]  refined binary node embeddings (from cross-attention)
        H_hat_s : [n_s, d]  refined source node embeddings (from cross-attention)
        M_gt    : [n_b, n_s]  ground-truth alignment

        Returns
        -------
        scalar loss
        """
        H_b_norm = F.normalize(H_hat_b, p=2, dim=-1)
        H_s_norm = F.normalize(H_hat_s, p=2, dim=-1)
        sim = (H_b_norm @ H_s_norm.t()) / self.tau
        pred_align = F.softmax(sim, dim=-1)

        M_gt_norm = M_gt / M_gt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return F.mse_loss(pred_align, M_gt_norm)


class NodeHardNegativeContrastiveLoss(nn.Module):
    """§9.3 节点级硬负：对非匹配 (i,j)（``M_gt`` 低于阈值）按相似度取 Top-K，
    与多正样本（``M_gt`` 高于阈值）一起做 InfoNCE 风格对比。

    对 BB→Stmt 与 Stmt→BB 两个方向各算一遍再平均（对称）。

    **全向量化**：无任何 Python `for` 循环 / `.item()` 同步点，可安全放进
    DDP + AMP 训练流水线（与 Stage 1 `SubgraphPatternClassifier` 向量化同思路）。
    """

    def __init__(self, tau: float = 0.07, eps: float = 1e-8) -> None:
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)

    @staticmethod
    def _direction_loss(
        sim: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """沿 dim=1 做 InfoNCE(多正 + Top-K 硬负)；返回 ``(row_loss [L], valid [L])``。

        sim / pos / neg: [L, M]（L 为 anchor 维度，M 为候选维度）。
        row_loss 在 ``valid=False`` 的行上被填 0；调用方据 valid 计均值。
        """
        L, M = sim.shape
        # log_pos[i] = logsumexp(sim[i, pos_mask])；无正样本的行为 -inf
        sim_pos = sim.masked_fill(~pos, float("-inf"))
        log_pos = torch.logsumexp(sim_pos, dim=1)                 # [L]

        # Top-K 硬负：把非负位置置 -inf，再 topk
        k_eff = int(min(topk, M))
        if k_eff < 1:
            # 候选维度为 0：返回全 0 row_loss 与全 False valid（调用方按 0 处理）
            zero = sim.sum(dim=1) * 0.0  # [L]，保图
            return zero, torch.zeros(L, dtype=torch.bool, device=sim.device)
        sim_neg = sim.masked_fill(~neg, float("-inf"))
        # 某些行可能负样本不足 k_eff 个——多余位置会是 -inf，logsumexp 自动忽略
        neg_topk, _ = torch.topk(sim_neg, k_eff, dim=1)            # [L, k_eff]

        # 分母 = logsumexp([log_pos, neg_topk])
        denom = torch.logsumexp(
            torch.cat([log_pos.unsqueeze(1), neg_topk], dim=1),
            dim=1,
        )                                                          # [L]
        row_loss = -(log_pos - denom)                              # [L]

        # 只保留同时有正样本和至少 1 个负样本的行
        has_pos = pos.any(dim=1)                                   # [L]
        has_neg = neg.any(dim=1)                                   # [L]
        valid = has_pos & has_neg                                  # [L]
        # 无效行用 0 填充（保持梯度连通，避免 masked_select 破图）
        row_loss = torch.where(valid, row_loss, torch.zeros_like(row_loss))
        return row_loss, valid

    def forward(
        self,
        H_b: torch.Tensor,
        H_s: torch.Tensor,
        M_gt: torch.Tensor,
        *,
        topk: int = 32,
        pos_thresh: float = 0.05,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        H_b : [n_b, d]
        H_s : [n_s, d]
        M_gt : [n_b, n_s]  非负，不必行归一化
        topk : 每行 / 每列硬负条数
        pos_thresh : 大于该值视为正位置，不参与硬负候选
        """
        nb = H_b.size(0)
        ns = H_s.size(0)
        if nb == 0 or ns == 0:
            return (H_b.sum() + H_s.sum()) * 0.0

        H_b_n = F.normalize(H_b.float(), dim=-1)
        H_s_n = F.normalize(H_s.float(), dim=-1)
        sim = (H_b_n @ H_s_n.t()) / max(self.tau, self.eps)        # [nb, ns]
        M = M_gt.float()
        pos = M > float(pos_thresh)                                # [nb, ns]
        neg = ~pos                                                 # [nb, ns]

        # BB → Stmt（anchor=BB，候选=Stmt）
        row_loss_b2s, valid_b2s = self._direction_loss(sim, pos, neg, topk)

        # Stmt → BB（anchor=Stmt，候选=BB）：转置复用同一函数
        row_loss_s2b, valid_s2b = self._direction_loss(
            sim.t().contiguous(), pos.t().contiguous(), neg.t().contiguous(), topk,
        )

        total = row_loss_b2s.sum() + row_loss_s2b.sum()
        n_valid = valid_b2s.sum() + valid_s2b.sum()
        # clamp 避免 0 分母；无有效行时返回 0*sum 保持图连通
        denom_t = n_valid.float().clamp(min=1.0)
        loss = total / denom_t
        # n_valid==0 时强制回到 0（乘 0 不破梯度）
        loss = torch.where(
            n_valid > 0,
            loss,
            (H_b_n.sum() + H_s_n.sum()) * 0.0,
        )
        if not torch.isfinite(loss):
            return (H_b_n * 0.0).sum() + (H_s_n * 0.0).sum()
        return loss


class SourceCountRegressionLoss(nn.Module):
    """§6.5.3：回归头预测源码语句数 n_s 的 MSE loss。"""

    def forward(
        self,
        predicted_n_s: torch.Tensor,
        actual_n_s: torch.Tensor,
    ) -> torch.Tensor:
        out = F.mse_loss(predicted_n_s.float(), actual_n_s.float())
        if not torch.isfinite(out):
            return (predicted_n_s.float() * 0.0).sum()
        return out

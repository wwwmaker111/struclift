"""
Contrastive losses for StrucLift Stage 1 pre-training.

* InfoNCELoss        — graph-level contrastive loss
* RegionInfoNCELoss  — region-level contrastive loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) between two sets of L2-normalised
    embeddings.

    Given a batch of N (binary, source) pairs, the positive for binary_i
    is source_i; all other source_j (j ≠ i) are negatives (and vice-versa).

    Parameters
    ----------
    tau : float
        Temperature scaling factor (default 0.07).
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        z_bin: torch.Tensor,
        z_src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_bin : [N, d]  L2-normalised binary graph embeddings
        z_src : [N, d]  L2-normalised source graph embeddings

        Returns
        -------
        scalar loss
        """
        N = z_bin.size(0)
        # similarity matrix  [N, N]
        sim = torch.mm(z_bin, z_src.t()) / self.tau  # [N, N]

        labels = torch.arange(N, device=sim.device)

        # symmetric loss
        loss_b2s = F.cross_entropy(sim, labels)
        loss_s2b = F.cross_entropy(sim.t(), labels)
        return 0.5 * (loss_b2s + loss_s2b)


class RegionInfoNCELoss(nn.Module):
    """InfoNCE at region level.

    Each sample in the batch contributes K_b binary regions and K_s source
    regions.  Positive pairs are regions from the same function that have
    been aligned by the region aligner (provided as labels).
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        r_bin: torch.Tensor,
        r_src: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        r_bin : [K_b, d]   region embeddings (binary side), L2-normalised
        r_src : [K_s, d]   region embeddings (source side), L2-normalised
        positive_mask : [K_b, K_s]  1 where region pair is positive, 0 otherwise

        Returns
        -------
        scalar loss
        """
        sim = torch.mm(r_bin, r_src.t()) / self.tau  # [K_b, K_s]

        # for each binary region row, we have potentially multiple positives
        # use multi-positive cross-entropy (log-sum-exp trick)
        # loss_i = -log( sum_j pos_ij * exp(sim_ij) / sum_k exp(sim_ik) )
        pos_mask = positive_mask.bool()
        if not pos_mask.any():
            return torch.tensor(0.0, device=sim.device, requires_grad=True)

        log_softmax = sim - sim.logsumexp(dim=1, keepdim=True)  # [K_b, K_s]

        # mask and average over positives per row
        pos_log_probs = log_softmax.masked_fill(~pos_mask, 0.0)
        num_pos_per_row = pos_mask.float().sum(dim=1).clamp(min=1)
        loss_per_row = -(pos_log_probs.sum(dim=1) / num_pos_per_row)

        # only rows with at least one positive
        active = num_pos_per_row > 0
        if active.sum() == 0:
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        out = loss_per_row[active].mean()
        if not torch.isfinite(out):
            return (r_bin * 0.0).sum() + (r_src * 0.0).sum()
        return out

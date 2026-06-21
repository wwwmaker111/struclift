"""
Sinkhorn optimal transport solvers for StrucLift alignment.

Provides log-space Sinkhorn iteration, a convenience cost-matrix wrapper,
the Structure-Constrained Optimal Transport (SCOT) solver used in Module B,
and a Hungarian hard-matching baseline for evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from struclift.data.structures import PATTERN_AST_COMPATIBILITY


# ═══════════════════════════ log-space Sinkhorn ══════════════════════════════

def log_sinkhorn(
    log_K: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    num_iter: int = 20,
) -> torch.Tensor:
    """Log-space Sinkhorn algorithm for numerically stable optimal transport.

    Iterates dual variables (u, v) in log-space so that
        M = diag(exp(u)) @ exp(log_K) @ diag(exp(v))
    satisfies the marginal constraints  M 1 = mu,  M^T 1 = nu.

    Args:
        log_K: Log of the Gibbs kernel, i.e. -C / epsilon.
               Shape ``[n_b, n_s]``.
        mu:    Source (row) marginal distribution.
               Shape ``[n_b]``.  Must sum to 1 (or same total as ``nu``).
        nu:    Target (column) marginal distribution.
               Shape ``[n_s]``.  Must sum to 1 (or same total as ``mu``).
        num_iter: Number of Sinkhorn iterations.

    Returns:
        M: Transport plan.  Shape ``[n_b, n_s]``, non-negative, with
           row-sums ~ mu and column-sums ~ nu.
    """
    # fp16 下 logsumexp 有效精度不足，u/v 易震荡、M 易 NaN；内部全程 fp32，输出还原 dtype。
    orig_dtype = log_K.dtype
    log_K = log_K.float()
    mu = mu.float()
    nu = nu.float()

    # log marginals
    log_mu: torch.Tensor = torch.log(mu.clamp(min=1e-30))  # [n_b]
    log_nu: torch.Tensor = torch.log(nu.clamp(min=1e-30))  # [n_s]

    # dual variables initialised to zero
    u: torch.Tensor = torch.zeros_like(mu)  # [n_b]
    v: torch.Tensor = torch.zeros_like(nu)  # [n_s]

    for _ in range(num_iter):
        # row update:  u  <-  log_mu - logsumexp_j(log_K[i,j] + v[j])
        u = log_mu - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)  # [n_b]
        # column update: v  <-  log_nu - logsumexp_i(log_K[i,j] + u[i])
        v = log_nu - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)  # [n_s]

    # assemble transport plan: M[i,j] = exp(u[i] + log_K[i,j] + v[j])
    M: torch.Tensor = torch.exp(u.unsqueeze(1) + log_K + v.unsqueeze(0))  # [n_b, n_s]
    return M.to(orig_dtype)


# ═══════════════════════════ cost-matrix wrapper ═════════════════════════════

def sinkhorn_from_cost(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 0.05,
    num_iter: int = 20,
    *,
    eps: float = None,
    max_iter: int = None,
) -> torch.Tensor:
    """Convenience wrapper: converts a cost matrix to log-space and runs Sinkhorn.

    Args:
        C:       Cost matrix.  Shape ``[n_b, n_s]``.
        mu:      Source marginal.  Shape ``[n_b]``.
        nu:      Target marginal.  Shape ``[n_s]``.
        epsilon: Entropic regularisation strength.
        num_iter: Sinkhorn iterations.

    Returns:
        M: Transport plan.  Shape ``[n_b, n_s]``.
    """
    if eps is not None:
        epsilon = eps
    if max_iter is not None:
        num_iter = max_iter
    n_b, n_s = C.shape
    if mu is None:
        mu = torch.ones(n_b, device=C.device, dtype=C.dtype) / n_b
    if nu is None:
        nu = torch.ones(n_s, device=C.device, dtype=C.dtype) / n_s
    log_K: torch.Tensor = -C / epsilon  # [n_b, n_s]
    return log_sinkhorn(log_K, mu, nu, num_iter=num_iter)


# ═══════════════════════ structure type penalty ══════════════════════════════

def _build_type_distance_matrix(
    pattern_probs: torch.Tensor,
    ast_types: torch.Tensor,
    compatibility_dict: Optional[Dict[int, List[int]]] = None,
) -> torch.Tensor:
    """Build the structure-type incompatibility penalty matrix d_type.

    For each pair (binary block *i*, source statement *j*) we compute a
    penalty based on how unlikely the predicted pattern of block *i* is to
    align with the AST type of statement *j* according to
    ``PATTERN_AST_COMPATIBILITY``.

    Args:
        pattern_probs: Softmax pattern predictions.  Shape ``[n_b, num_patterns]``.
        ast_types:     Integer AST type labels.  Shape ``[n_s]`` (int64).
        compatibility_dict: Override for ``PATTERN_AST_COMPATIBILITY``.

    Returns:
        d_type: Penalty matrix.  Shape ``[n_b, n_s]``, values in [0, 1].
    """
    if compatibility_dict is None:
        compatibility_dict = PATTERN_AST_COMPATIBILITY

    n_b: int = pattern_probs.shape[0]
    n_s: int = ast_types.shape[0]
    num_patterns: int = pattern_probs.shape[1]
    device: torch.device = pattern_probs.device

    # Build a binary compatibility table: compat[p, a] = 1 if compatible
    num_ast: int = int(ast_types.max().item()) + 1
    compat: torch.Tensor = torch.zeros(num_patterns, num_ast, device=device)
    for pat_id, allowed_ast_ids in compatibility_dict.items():
        for ast_id in allowed_ast_ids:
            if pat_id < num_patterns and ast_id < num_ast:
                compat[pat_id, ast_id] = 1.0

    # For each block i, the probability mass on *compatible* patterns w.r.t. each source type
    # compatible_prob[i, j] = sum_p  pattern_probs[i, p] * compat[p, ast_types[j]]
    compat_for_targets: torch.Tensor = compat[:, ast_types.long()]  # [num_patterns, n_s]
    compatible_prob: torch.Tensor = pattern_probs @ compat_for_targets  # [n_b, n_s]

    # Penalty: 1 - compatible_prob (higher penalty when blocks/stmts are incompatible)
    d_type: torch.Tensor = 1.0 - compatible_prob  # [n_b, n_s]
    return d_type


# ═══════════════════════ SCOT solver ═════════════════════════════════════════

def structure_constrained_ot(
    S: torch.Tensor = None,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    adj_b: torch.Tensor = None,
    adj_s: torch.Tensor = None,
    pattern_probs: torch.Tensor = None,
    ast_types: torch.Tensor = None,
    epsilon: float = 0.05,
    gamma_topo: float = 0.1,
    lambda_struct: float = 0.3,
    num_outer: int = 5,
    num_sinkhorn: int = 20,
    compatibility_dict: Optional[Dict[int, List[int]]] = None,
    *,
    cost: torch.Tensor = None,
    D_x: torch.Tensor = None,
    D_y: torch.Tensor = None,
    eps: float = None,
    max_iter: int = None,
    outer_iter: int = None,
    type_penalty_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Structure-Constrained Optimal Transport (SCOT) solver.

    Combines entropic OT with (a) structure type compatibility penalties and
    (b) topology-preserving gradients that encourage neighbouring binary
    blocks to map to neighbouring source statements.

    Algorithm
    ---------
    1.  Compute base cost  C = -S / tau  (here ``S`` is already the similarity;
        the caller may pre-divide by tau, so we use ``C = -S`` directly).
    2.  Add structure-type penalty: ``C_struct = C + lambda_struct * d_type``.
    3.  For ``num_outer`` iterations:
        a.  Compute the topology discrepancy matrix ``Phi`` where
            ``Phi[j1, j2] = 1`` if ``(j1, j2)`` is *not* an edge in the
            source CFG and ``j1 != j2``.
        b.  Compute gradient:
            ``grad_topo = adj_b @ (M @ Phi) + adj_b^T @ (M @ Phi^T)``.
        c.  Update cost:
            ``C_updated = C_struct + gamma_topo * grad_topo``.
        d.  ``M = sinkhorn_from_cost(C_updated, mu, nu, epsilon, num_sinkhorn)``.
    4.  Return final ``M``.

    Args:
        S:               Similarity matrix.  Shape ``[n_b, n_s]``.
        mu:              Source marginal.  Shape ``[n_b]``.
        nu:              Target marginal.  Shape ``[n_s]``.
        adj_b:           Binary CFG adjacency (possibly weighted).
                         Shape ``[n_b, n_b]``.
        adj_s:           Source CFG adjacency.
                         Shape ``[n_s, n_s]``.
        pattern_probs:   Predicted pattern probabilities per binary block.
                         Shape ``[n_b, num_patterns]``.
        ast_types:       Ground-truth AST type per source statement.
                         Shape ``[n_s]`` (int64).
        epsilon:         Sinkhorn regularisation.
        gamma_topo:      Weight of the topology-preserving gradient.
        lambda_struct:   Weight of the structure-type penalty.
        num_outer:       Number of outer Frank-Wolfe-style iterations.
        num_sinkhorn:    Sinkhorn iterations per outer step.
        compatibility_dict: Optional override for ``PATTERN_AST_COMPATIBILITY``.

    Returns:
        M: Transport plan.  Shape ``[n_b, n_s]``.
    """
    if cost is not None and S is None:
        S = -cost
    if D_x is not None and adj_b is None:
        adj_b = D_x
    if D_y is not None and adj_s is None:
        adj_s = D_y
    if eps is not None:
        epsilon = eps
    if max_iter is not None:
        num_sinkhorn = max_iter
    if outer_iter is not None:
        num_outer = outer_iter

    n_b: int = S.shape[0]
    n_s: int = S.shape[1]
    device: torch.device = S.device

    if mu is None:
        mu = torch.ones(n_b, device=device, dtype=S.dtype) / n_b
    if nu is None:
        nu = torch.ones(n_s, device=device, dtype=S.dtype) / n_s

    # Step 1: base cost (negative similarity)
    C: torch.Tensor = -S  # [n_b, n_s]

    # Step 2: structure-type compatibility penalty
    if pattern_probs is not None and ast_types is not None:
        d_type: torch.Tensor = _build_type_distance_matrix(
            pattern_probs, ast_types, compatibility_dict
        )  # [n_b, n_s]
        if type_penalty_mask is not None:
            d_type = d_type * type_penalty_mask
        C_struct: torch.Tensor = C + lambda_struct * d_type  # [n_b, n_s]
    else:
        C_struct = C

    # Phi: topology discrepancy in source side.
    # Phi[j1, j2] = 1 if (j1, j2) is NOT an edge in source CFG and j1 != j2
    Phi: torch.Tensor = 1.0 - adj_s  # [n_s, n_s]
    # zero out the diagonal (j1 == j2 does not count)
    Phi = Phi - torch.diag(Phi.diag())  # [n_s, n_s]

    # Initialise M with plain Sinkhorn (no topo term yet)
    M: torch.Tensor = sinkhorn_from_cost(
        C_struct, mu, nu, epsilon, num_sinkhorn
    )  # [n_b, n_s]

    # Step 3: outer iterations with topology gradient
    for _ in range(num_outer):
        # M @ Phi  -> [n_b, n_s]
        M_Phi: torch.Tensor = M @ Phi  # [n_b, n_s]
        M_PhiT: torch.Tensor = M @ Phi.t()  # [n_b, n_s]

        # grad_topo = adj_b @ M_Phi + adj_b^T @ M_PhiT
        grad_topo: torch.Tensor = adj_b @ M_Phi + adj_b.t() @ M_PhiT  # [n_b, n_s]

        C_updated: torch.Tensor = C_struct + gamma_topo * grad_topo  # [n_b, n_s]
        M = sinkhorn_from_cost(C_updated, mu, nu, epsilon, num_sinkhorn)  # [n_b, n_s]

    return M


# ═══════════════════════ Hungarian hard matching ═════════════════════════════

def hungarian_matching(S: torch.Tensor) -> torch.Tensor:
    """Hard (0/1) matching via the Hungarian algorithm.

    Used at evaluation time to convert a soft similarity matrix into a
    one-to-one assignment.

    Args:
        S: Similarity matrix.  Shape ``[n_b, n_s]``.  Higher = better match.

    Returns:
        M_hard: Binary assignment matrix.  Shape ``[n_b, n_s]`` (float32),
                with exactly ``min(n_b, n_s)`` entries equal to 1.
    """
    # scipy minimises cost, so negate similarity
    cost_np = (-S).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    M_hard: torch.Tensor = torch.zeros_like(S)
    M_hard[row_ind, col_ind] = 1.0
    return M_hard


# ═════════════════════════════════════════════════════════════════════════════
# Batched (vectorised) variants
# -----------------------------------------------------------------------------
# 把 batch 内多个独立的 SCOT/sinkhorn 子问题 padded 到 ``[B, n_b, n_s]`` 一次性
# 求解，等价于原版逐个调用，但 launch overhead 摊薄了 B 倍。SCOT region-aware
# 路径每图 ~6 个子问题 × batch=64 ≈ 几百次 SCOT 串行，是 alignment forward 的
# 主要时间来源；走 batched 后 GPU 利用率显著提高。
# ─────────────────────────────────────────────────────────────────────────────


def log_sinkhorn_batch(
    log_K: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    num_iter: int = 20,
    mu_mask: Optional[torch.Tensor] = None,
    nu_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched log-space Sinkhorn solver.

    Args:
        log_K:   ``[B, n_b, n_s]`` Gibbs log-kernel ``-C / epsilon``.
        mu:      ``[B, n_b]`` row marginals (padded rows = 0).
        nu:      ``[B, n_s]`` column marginals (padded cols = 0).
        num_iter: Sinkhorn iterations.
        mu_mask: ``[B, n_b]`` bool, ``True`` for valid rows. Optional.
        nu_mask: ``[B, n_s]`` bool, ``True`` for valid cols. Optional.

    Returns:
        ``M`` of shape ``[B, n_b, n_s]``; entries at padded rows/cols are 0.
    """
    orig_dtype = log_K.dtype
    log_K = log_K.float()
    mu = mu.float()
    nu = nu.float()

    if mu_mask is not None:
        log_K = log_K.masked_fill(~mu_mask.unsqueeze(-1), float("-inf"))
    if nu_mask is not None:
        log_K = log_K.masked_fill(~nu_mask.unsqueeze(-2), float("-inf"))

    log_mu = torch.log(mu.clamp(min=1e-30))  # padded rows -> ~-69, 之后 mask 掉
    log_nu = torch.log(nu.clamp(min=1e-30))

    u = torch.zeros_like(mu)  # [B, n_b]
    v = torch.zeros_like(nu)  # [B, n_s]
    zero_u = torch.zeros_like(u)
    zero_v = torch.zeros_like(v)

    for _ in range(num_iter):
        u = log_mu - torch.logsumexp(log_K + v.unsqueeze(-2), dim=-1)
        if mu_mask is not None:
            # padded 行的 u 在数学上无意义（log_K 全 -inf → logsumexp = -inf →
            # u = +inf），强制置 0 防止后续 (log_K + u) = -inf + +inf = NaN。
            u = torch.where(mu_mask, u, zero_u)
        v = log_nu - torch.logsumexp(log_K + u.unsqueeze(-1), dim=-2)
        if nu_mask is not None:
            v = torch.where(nu_mask, v, zero_v)

    M = torch.exp(u.unsqueeze(-1) + log_K + v.unsqueeze(-2))
    return M.to(orig_dtype)


def sinkhorn_from_cost_batch(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float = 0.05,
    num_iter: int = 20,
    mu_mask: Optional[torch.Tensor] = None,
    nu_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched cost-matrix wrapper around :func:`log_sinkhorn_batch`.

    Padded positions inside ``C`` should already be set to ``+inf``（代价无穷大，
    log_K 自然 -inf），caller 通常通过 ``mu_mask``/``nu_mask`` 表达 padding 即可。
    """
    log_K = -C / epsilon
    return log_sinkhorn_batch(log_K, mu, nu, num_iter=num_iter, mu_mask=mu_mask, nu_mask=nu_mask)


def _build_type_distance_matrix_batch(
    pattern_probs: torch.Tensor,   # [B, n_b, P]
    ast_types: torch.Tensor,       # [B, n_s]   long
    num_ast: int,
    compatibility_dict: Optional[Dict[int, List[int]]] = None,
) -> torch.Tensor:
    """Batched version of :func:`_build_type_distance_matrix`.

    与单图版语义一致：``d_type[b, i, j] = 1 - sum_p p^pat_b[i, p] * compat[p, ast_b[j]]``。
    单图版每次还要 ``ast_types.max().item()`` 推 ``num_ast``，是 GPU sync 热点；
    批化版让 caller 一次传 ``num_ast``（由词表大小决定，常量），完全免 sync。
    """
    if compatibility_dict is None:
        compatibility_dict = PATTERN_AST_COMPATIBILITY

    B, n_b, P = pattern_probs.shape
    device = pattern_probs.device
    dtype = pattern_probs.dtype

    compat = torch.zeros(P, num_ast, device=device, dtype=dtype)
    for pat_id, allowed in compatibility_dict.items():
        if pat_id >= P:
            continue
        for ast_id in allowed:
            if 0 <= ast_id < num_ast:
                compat[pat_id, ast_id] = 1.0

    ast_types = ast_types.clamp(min=0, max=num_ast - 1).long()
    # gather: compat_for_targets[b, p, j] = compat[p, ast_types[b, j]]
    compat_for_targets = compat[:, ast_types]  # [P, B, n_s]
    compat_for_targets = compat_for_targets.permute(1, 0, 2).contiguous()  # [B, P, n_s]
    compatible_prob = torch.bmm(pattern_probs, compat_for_targets)  # [B, n_b, n_s]
    return 1.0 - compatible_prob


def structure_constrained_ot_batch(
    cost: torch.Tensor,            # [B, n_b, n_s]
    D_x: torch.Tensor,             # [B, n_b, n_b]  binary adj
    D_y: torch.Tensor,             # [B, n_s, n_s]  source adj
    pattern_probs: torch.Tensor,   # [B, n_b, P]
    ast_types: torch.Tensor,       # [B, n_s]
    mu: torch.Tensor,              # [B, n_b]
    nu: torch.Tensor,              # [B, n_s]
    *,
    eps: float = 0.05,
    max_iter: int = 20,
    outer_iter: int = 5,
    lambda_struct: float = 1.0,
    gamma_topo: float = 0.5,
    num_ast: int,
    compatibility_dict: Optional[Dict[int, List[int]]] = None,
    type_penalty_mask: Optional[torch.Tensor] = None,  # [B, n_b, n_s]
    mu_mask: Optional[torch.Tensor] = None,            # [B, n_b]
    nu_mask: Optional[torch.Tensor] = None,            # [B, n_s]
) -> torch.Tensor:
    """Batched SCOT solver. 与 :func:`structure_constrained_ot` 数学等价。

    所有 batched 张量已经 padded 到统一形状，padded 行/列由 mask 标识。
    返回 ``M``：``[B, n_b, n_s]``，padded 位置为 0。
    """
    C = cost
    if pattern_probs is not None and ast_types is not None:
        d_type = _build_type_distance_matrix_batch(
            pattern_probs, ast_types, num_ast=num_ast,
            compatibility_dict=compatibility_dict,
        )
        if type_penalty_mask is not None:
            d_type = d_type * type_penalty_mask
        C_struct = C + lambda_struct * d_type
    else:
        C_struct = C

    # Phi[b, j1, j2] = 1 if (j1, j2) NOT an edge in source CFG and j1 != j2
    Phi = 1.0 - D_y  # [B, n_s, n_s]
    eye_s = torch.eye(D_y.size(-1), device=D_y.device, dtype=D_y.dtype).unsqueeze(0)
    Phi = Phi - Phi * eye_s  # zero diagonal

    # Initial sinkhorn
    M = sinkhorn_from_cost_batch(
        C_struct, mu, nu, epsilon=eps, num_iter=max_iter,
        mu_mask=mu_mask, nu_mask=nu_mask,
    )

    D_x_t = D_x.transpose(-2, -1)
    Phi_t = Phi.transpose(-2, -1)
    for _ in range(outer_iter):
        M_Phi = torch.bmm(M, Phi)              # [B, n_b, n_s]
        M_PhiT = torch.bmm(M, Phi_t)           # [B, n_b, n_s]
        grad_topo = torch.bmm(D_x, M_Phi) + torch.bmm(D_x_t, M_PhiT)
        C_updated = C_struct + gamma_topo * grad_topo
        M = sinkhorn_from_cost_batch(
            C_updated, mu, nu, epsilon=eps, num_iter=max_iter,
            mu_mask=mu_mask, nu_mask=nu_mask,
        )
    return M

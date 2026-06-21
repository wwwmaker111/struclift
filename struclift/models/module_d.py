"""
Module D — Structural Consistency Reinforcement Learning.

Provides reward functions (compile, structural, semantic) and the GRPO
(Group Relative Policy Optimisation) training loop that fine-tunes
Module C's adapters + LoRA weights.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════ reward helpers ═════════════════════════════════

def compile_reward(generated_code: str, timeout: int = 10) -> float:
    """Attempt to compile *generated_code* with gcc and return a [0, 1] score.

    Returns
    -------
    float
        1.0  if compilation succeeds,
        max(0, 1 - 0.1 * error_count)  otherwise.
    """
    boilerplate = (
        '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n'
        '#include <stdint.h>\n#include <math.h>\n'
    )
    code = boilerplate + generated_code
    try:
        # ``-fsyntax-only`` avoids ``-o`` / ``/dev/null`` portability (Windows vs POSIX).
        result = subprocess.run(
            ["gcc", "-fsyntax-only", "-x", "c", "-", "-w"],
            input=code.encode(),
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return 1.0
        error_lines = [
            l for l in result.stderr.decode(errors="replace").splitlines()
            if ": error:" in l
        ]
        return max(0.0, 1.0 - 0.1 * len(error_lines))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """Jensen–Shannon divergence between two discrete distributions."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float(0.5 * (kl_pm + kl_qm))


def _tree_edit_distance_simple(
    tree_a: List[str], tree_b: List[str]
) -> int:
    """Simple tree-edit distance approximation via sequence edit distance
    on the pre-order serialisation of the control-structure trees.

    A full Zhang–Shasha implementation can replace this for better accuracy.
    """
    n, m = len(tree_a), len(tree_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if tree_a[i - 1] == tree_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def structural_reward(
    gen_pattern_dist: torch.Tensor,
    bin_pattern_dist: torch.Tensor,
    gen_cst_serialised: List[str],
    bin_cst_serialised: List[str],
    gen_nesting_depth: int,
    bin_nesting_depth: int,
    w_type: float = 0.4,
    w_ted: float = 0.4,
    w_depth: float = 0.2,
) -> float:
    """Compute the three-component structural consistency reward.

    Parameters
    ----------
    gen_pattern_dist : Tensor [C]
        Pattern type distribution of the generated code's CFG.
    bin_pattern_dist : Tensor [C]
        Pattern type distribution of the original binary CFG.
    gen_cst_serialised / bin_cst_serialised : list of str
        Pre-order serialisation of the control-structure tree.
    gen_nesting_depth / bin_nesting_depth : int
        Maximum nesting depth.

    Returns
    -------
    float   in [0, 1]
    """
    # (a) pattern distribution similarity
    type_sim = 1.0 - _js_divergence(gen_pattern_dist, bin_pattern_dist)

    # (b) 提案 §8.2.3(b)「CST 树编辑距离」：实现上对 **先序 CST 类型标签序列** 做 Levenshtein
    # 近似（``_tree_edit_distance_simple``）。若需与 Zhang–Shasha 完全一致的树编辑距离，
    # 可在此替换为有序树编辑实现（需显式树结构，而非扁平序列）。
    ted = _tree_edit_distance_simple(gen_cst_serialised, bin_cst_serialised)
    max_size = max(len(gen_cst_serialised), len(bin_cst_serialised), 1)
    ted_sim = 1.0 - ted / max_size

    # (c) nesting depth consistency
    d_max = max(gen_nesting_depth, bin_nesting_depth, 1)
    depth_sim = 1.0 - abs(gen_nesting_depth - bin_nesting_depth) / d_max

    return w_type * type_sim + w_ted * ted_sim + w_depth * depth_sim


def semantic_reward(
    generated_code: str,
    test_cases: Optional[List[Tuple[str, str]]] = None,
    reference_code: Optional[str] = None,
) -> float:
    """Semantic equivalence reward.

    If *test_cases* is provided (list of (stdin, expected_stdout)),
    compile + run and count passes.  Otherwise fall back to edit similarity
    with *reference_code*.
    """
    if test_cases:
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(
                '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n'
            )
            f.write(generated_code)
            src_path = f.name
        bin_path = src_path.replace(".c", ".out")
        try:
            subprocess.run(
                ["gcc", src_path, "-o", bin_path, "-w", "-lm"],
                capture_output=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 0.0
        if not Path(bin_path).exists():
            return 0.0
        passed = 0
        for stdin_data, expected in test_cases:
            try:
                res = subprocess.run(
                    [bin_path], input=stdin_data.encode(),
                    capture_output=True, timeout=5,
                )
                if res.stdout.decode(errors="replace").strip() == expected.strip():
                    passed += 1
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        return passed / max(len(test_cases), 1)

    if reference_code is not None:
        return _edit_similarity(generated_code, reference_code)

    return 0.0


def _edit_similarity(a: str, b: str) -> float:
    """Normalised edit-similarity (1 − edit_distance / max_len)."""
    na, nb = len(a), len(b)
    if na == 0 and nb == 0:
        return 1.0
    # use a two-row DP to save memory
    prev = list(range(nb + 1))
    for i in range(1, na + 1):
        curr = [i] + [0] * nb
        for j in range(1, nb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[nb]
    return 1.0 - dist / max(na, nb, 1)


def combined_reward(
    r_compile: float,
    r_struct: float,
    r_semantic: float,
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> float:
    return alpha * r_compile + beta * r_struct + gamma * r_semantic


# ═══════════════════════════ GRPO trainer ═══════════════════════════════════

@dataclass
class GRPOConfig:
    num_candidates: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    kl_beta: float = 0.05
    lr: float = 5e-6
    max_grad_norm: float = 1.0
    alpha_compile: float = 0.2
    beta_struct: float = 0.5
    gamma_semantic: float = 0.3


class GRPOTrainer:
    """Group Relative Policy Optimisation trainer for Module C.

    Requires a *policy_model* (Module C with LLM) and a frozen *ref_model*
    (the SFT checkpoint).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: GRPOConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.cfg = config

        # freeze reference
        for p in self.ref.parameters():
            p.requires_grad = False

        if optimizer is None:
            trainable = [p for p in self.policy.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        else:
            self.optimizer = optimizer

    # ------------------------------------------------------------------ #
    #  public API                                                          #
    # ------------------------------------------------------------------ #

    def step(
        self,
        input_ids: torch.Tensor,
        graph_kv: torch.Tensor,
        slot_mask: torch.Tensor,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """One GRPO gradient step.

        Parameters
        ----------
        input_ids : [B, L]   (B = num_candidates for one sample)
        graph_kv  : [B, N, d_graph]
        slot_mask : [B, L, N]
        rewards   : [B]       raw rewards for each candidate
        log_probs : [B]       log π_θ(y | x)
        ref_log_probs : [B]   log π_ref(y | x)

        Returns
        -------
        dict with 'loss', 'mean_reward', 'mean_advantage', 'kl'
        """
        # group normalisation
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # KL penalty
        kl = log_probs - ref_log_probs  # per-candidate approx KL

        # GRPO objective  (maximise)
        # L = - E[ A_g * log π(y_g) - β * KL ]
        loss = -(advantages * log_probs - self.cfg.kl_beta * kl).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "kl": kl.mean().item(),
        }

    def step_with_policy_forward(
        self,
        rewards: torch.Tensor,
        ref_log_probs: torch.Tensor,
        policy_log_prob_fn: Callable[[int], torch.Tensor],
    ) -> Dict[str, float]:
        """与 :meth:`step` 相同的 GRPO 目标，但按候选依次前向并 ``backward(loss/G)``。

        ``policy_log_prob_fn(g)`` 返回第 ``g`` 个候选的 ``log π_θ``（标量张量）。每次只保留
        一条 policy 计算图，避免 :meth:`step` 中 ``stack`` 后单次 backward 同时保留 G 条图。
        """
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        G = rewards.numel()
        assert ref_log_probs.shape[0] == G

        self.optimizer.zero_grad()
        loss_sum = 0.0
        kl_sum = 0.0
        beta = self.cfg.kl_beta
        for g in range(G):
            lp = policy_log_prob_fn(g)
            kl_g = lp - ref_log_probs[g]
            term = -(advantages[g] * lp - beta * kl_g)
            loss_sum += term.item()
            kl_sum += kl_g.item()
            (term / G).backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": loss_sum / max(G, 1),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "kl": kl_sum / max(G, 1),
        }

    @torch.no_grad()
    def compute_rewards_for_candidates(
        self,
        candidates: List[str],
        bin_pattern_dist: torch.Tensor,
        bin_cst_serialised: List[str],
        bin_nesting_depth: int,
        test_cases: Optional[List[Tuple[str, str]]] = None,
        reference_code: Optional[str] = None,
        num_opcodes: int = 1024,
        use_structural_from_generated_c: bool = True,
    ) -> torch.Tensor:
        """Evaluate all G candidates and return a reward tensor [G].

        结构项：对生成 C 做 ``gcc -c`` + Capstone CFG（复用 ``build_binskel_dataset``），
        再与 ``bin_pattern_dist`` / ``bin_cst_serialised`` / ``bin_nesting_depth`` 比较
        （与 :func:`structural_reward` 一致）。失败时 ``r_s=0``。
        """
        from struclift.utils.generated_c_struct import structural_metrics_from_generated_c

        rewards = []
        c_pat = int(bin_pattern_dist.numel())
        bin_pd_cpu = bin_pattern_dist.detach().cpu().float()
        for code in candidates:
            r_c = compile_reward(code)
            r_s = 0.0
            if use_structural_from_generated_c and c_pat > 0:
                sm = structural_metrics_from_generated_c(
                    code,
                    num_pattern_classes=c_pat,
                    num_opcodes=num_opcodes,
                )
                if sm is not None:
                    gen_dist, gen_cst, gen_depth = sm
                    r_s = structural_reward(
                        gen_dist,
                        bin_pd_cpu,
                        gen_cst,
                        bin_cst_serialised,
                        gen_depth,
                        bin_nesting_depth,
                    )
            r_sem = semantic_reward(code, test_cases, reference_code)
            r = combined_reward(
                r_c, r_s, r_sem,
                self.cfg.alpha_compile,
                self.cfg.beta_struct,
                self.cfg.gamma_semantic,
            )
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)

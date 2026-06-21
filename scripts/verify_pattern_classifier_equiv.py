#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 ``SubgraphPatternClassifier`` 向量化版与 legacy 版（逐节点 Python 循环 + .item()）
在数值上按位等价（浮点尾部差异 < 阈值），并度量各自耗时。

流程
----
1. 构造多种规模随机 CFG（节点 5/20/50/150，边密度变化）。
2. 分别调 ``_topo_vector_for_graph_legacy`` 与 ``_topo_vector_for_graph``，对比 13 维。
3. 分别调 ``_forward_single_graph_legacy`` 与 ``_forward_single_graph``，对比最终
   ``pattern_logits``、前向耗时、``node_feats.grad`` 的等价性。
4. 任何维 diff 超阈值即 FAIL，退出码非 0。

示例::

    python scripts/verify_pattern_classifier_equiv.py --device cuda
    python scripts/verify_pattern_classifier_equiv.py --device cpu --strict
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.models.module_a import SubgraphPatternClassifier


TOPO_NAMES = (
    [f"counts[{t}]" for t in range(5)]
    + ["has_back", "min_b", "size_norm", "mean_n", "max_n", "min_n", "std_n", "diameter_n"]
)


def make_random_graph(
    n: int,
    e: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    d_model: int,
    num_edge_types: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成一张随机有向 CFG（节点特征 + COO 边 + 边类型）。"""
    g = torch.Generator(device=device).manual_seed(seed)
    feats = torch.randn(n, d_model, device=device, dtype=dtype, generator=g)
    src = torch.randint(0, n, (e,), device=device, generator=g)
    dst = torch.randint(0, n, (e,), device=device, generator=g)
    ei = torch.stack([src, dst], dim=0)
    et = torch.randint(0, num_edge_types, (e,), device=device, generator=g)
    return feats, ei, et


def _pretty_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    max_abs = float(d.max().item()) if d.numel() > 0 else 0.0
    mean_abs = float(d.mean().item()) if d.numel() > 0 else 0.0
    return max_abs, mean_abs


def main() -> int:
    ap = argparse.ArgumentParser(description="SubgraphPatternClassifier 等价性验证")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu（默认自动）")
    ap.add_argument(
        "--fp16", action="store_true",
        help="用 fp16 特征（模拟 autocast 场景；阈值放宽至 1e-3）",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="严格模式：阈值收紧到 1e-6（默认 1e-5）",
    )
    ap.add_argument(
        "--skip-backward", action="store_true",
        help="跳过反向验证（只对比 forward）",
    )
    args = ap.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if args.fp16 else torch.float32
    # forward / topo 与 grad 分开设阈值：
    # grad 是 forward 误差沿 chain rule 的累积，n 个节点的 sum().backward() 会放大 ~n 倍 fp32 ε。
    # 这不是算法不等价，是浮点结合律。训练里的 autocast+DDP 噪声比这大好几个数量级。
    if args.fp16:
        atol_fwd, atol_grad = 1e-3, 5e-3
    elif args.strict:
        atol_fwd, atol_grad = 1e-6, 1e-5
    else:
        atol_fwd, atol_grad = 1e-5, 5e-5
    print(
        f"device={device}  dtype={dtype}  "
        f"atol_fwd={atol_fwd:.1e}  atol_grad={atol_grad:.1e}",
    )

    cfg = StrucLiftConfig()
    torch.manual_seed(42)
    clf = SubgraphPatternClassifier(
        d_model=cfg.module_a.d_model,
        num_patterns=cfg.vocab.num_patterns,
        topo_feat_dim=cfg.module_a.pattern_topo_feat_dim,
        k=cfg.module_a.pattern_k_hop,
        num_edge_types=cfg.vocab.num_edge_types,
        max_blocks=cfg.vocab.max_blocks_per_function,
    ).to(device=device, dtype=dtype)
    clf.eval()

    # (名字, n, e)
    configs = [
        ("tiny",       5,   8),
        ("small",      20,  40),
        ("medium",     50,  120),
        ("typical",    100, 250),
        ("large",      200, 500),
    ]

    global_fail = False
    overall_max = 0.0

    for name, n, e in configs:
        for seed in range(3):
            feats, ei, et = make_random_graph(
                n, e, device, dtype, seed,
                cfg.module_a.d_model, cfg.vocab.num_edge_types,
            )

            # === 1) 对比 topo 向量（13 维分别） ===
            with torch.no_grad():
                dmat = clf._floyd_warshall(n, ei, device, dtype)
                k_hop_m = (dmat <= float(clf.k)) & (dmat < 1e5)
                topo_leg = clf._topo_vector_for_graph_legacy(
                    dmat, k_hop_m, ei, et, n, device, dtype,
                )
                topo_new = clf._topo_vector_for_graph(
                    dmat, k_hop_m, ei, et, n, device, dtype,
                )

            any_fail_topo = False
            max_abs_topo = 0.0
            for dim, col_name in enumerate(TOPO_NAMES):
                mx, _ = _pretty_diff(col_name, topo_leg[:, dim], topo_new[:, dim])
                max_abs_topo = max(max_abs_topo, mx)
                if mx > atol_fwd:
                    any_fail_topo = True
                    print(f"  [FAIL] {name} seed={seed} topo[{col_name}]: max_abs={mx:.3e} > {atol_fwd:.1e}")

            # === 2) 对比最终 pattern_logits ===
            with torch.no_grad():
                logits_leg = clf._forward_single_graph_legacy(feats, ei, et)
                logits_new = clf._forward_single_graph(feats, ei, et)
            max_abs_log, mean_abs_log = _pretty_diff("logits", logits_leg, logits_new)

            # === 3) 前向耗时（各跑 3 次取 min） ===
            def _time(fn) -> float:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                best = float("inf")
                for _ in range(3):
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = fn()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    best = min(best, time.perf_counter() - t0)
                return best * 1000  # ms

            t_leg = _time(lambda: clf._forward_single_graph_legacy(feats, ei, et))
            t_new = _time(lambda: clf._forward_single_graph(feats, ei, et))

            # === 4) 反向验证（可选） ===
            grad_max = 0.0
            if not args.skip_backward:
                feats_leg = feats.detach().clone().requires_grad_(True)
                feats_new = feats.detach().clone().requires_grad_(True)
                out_leg = clf._forward_single_graph_legacy(feats_leg, ei, et)
                out_new = clf._forward_single_graph(feats_new, ei, et)
                out_leg.sum().backward()
                out_new.sum().backward()
                grad_max, _ = _pretty_diff(
                    "grad", feats_leg.grad, feats_new.grad,
                )

            overall_max = max(overall_max, max_abs_topo, max_abs_log, grad_max)
            status = "OK" if (
                not any_fail_topo
                and max_abs_log <= atol_fwd
                and grad_max <= atol_grad
            ) else "FAIL"
            if status == "FAIL":
                global_fail = True
            print(
                f"  [{status}] {name} n={n} e={e} seed={seed}  "
                f"topo_max={max_abs_topo:.2e}  logits_max={max_abs_log:.2e}  "
                f"grad_max={grad_max:.2e}  "
                f"t_legacy={t_leg:.1f}ms  t_vec={t_new:.1f}ms  "
                f"speedup={t_leg/max(t_new, 1e-6):.1f}x",
            )

    print()
    if global_fail:
        print(
            f"RESULT: FAIL  overall_max_abs_diff={overall_max:.3e}  "
            f"(atol_fwd={atol_fwd:.1e}, atol_grad={atol_grad:.1e})",
        )
        return 1
    print(
        f"RESULT: PASS  overall_max_abs_diff={overall_max:.3e}  "
        f"(atol_fwd={atol_fwd:.1e}, atol_grad={atol_grad:.1e})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

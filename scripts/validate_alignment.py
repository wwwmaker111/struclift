#!/usr/bin/env python3
"""
按 BB_Source_Alignment_Pipeline.md 第七步：对齐质量 4 项检查

用法: python scripts/validate_alignment.py <binskel.jsonl>
"""

import json
import sys
from pathlib import Path


def validate_sample(d: dict) -> dict:
    """对单条样本做 4 项检查。"""
    M_gt = d.get("alignment_matrix") or []
    W_mat = d.get("alignment_confidence") or []
    block_lines = d.get("block_src_lines") or []
    edge_index = d.get("bin_edge_index") or [[], []]
    edges_src = edge_index[0] if len(edge_index) > 0 else []
    edges_dst = edge_index[1] if len(edge_index) > 1 else []
    edge_types = d.get("bin_edge_type") or []

    n_b = len(block_lines)
    if n_b == 0:
        return {"coverage": 0, "mean_conf": 0, "high_conf_ratio": 0, "low_conf_ratio": 0, "order_consistency": 1, "cf_consistency": 1}

    n_s = len(M_gt[0]) if M_gt and M_gt[0] else 0
    if not M_gt or n_s == 0:
        line_nums = sorted(set(ln for ls in block_lines for ln in ls))
        n_s = len(line_nums)
        line_to_idx = {ln: i for i, ln in enumerate(line_nums)}
        M_gt = [[0.0] * n_s for _ in range(n_b)]
        W_mat = [[0.0] * n_s for _ in range(n_b)]
        for b_id, lines in enumerate(block_lines):
            for ln in lines:
                if ln in line_to_idx:
                    M_gt[b_id][line_to_idx[ln]] = 1.0
            row_sum = sum(M_gt[b_id])
            if row_sum > 0:
                for j in range(n_s):
                    M_gt[b_id][j] /= row_sum
                    if M_gt[b_id][j] > 0:
                        W_mat[b_id][j] = 0.85 / row_sum
        n_s = len(M_gt[0]) if M_gt and M_gt[0] else 0

    if n_s == 0:
        return {"coverage": 0, "mean_conf": 0, "high_conf_ratio": 0, "low_conf_ratio": 0, "order_consistency": 1, "cf_consistency": 1}
    coverage = sum(1 for i in range(n_b) if i < len(M_gt) and sum(M_gt[i]) > 0) / n_b
    valid_w = []
    for i in range(n_b):
        for j in range(n_s):
            w = W_mat[i][j] if i < len(W_mat) and j < len(W_mat[i]) else 0
            if w > 0:
                valid_w.append(w)
    mean_conf = sum(valid_w) / len(valid_w) if valid_w else 0
    high = sum(1 for w in valid_w if w > 0.7) / len(valid_w) if valid_w else 0
    low = sum(1 for w in valid_w if w < 0.3) / len(valid_w) if valid_w else 0

    bb_primary = []
    for i in range(n_b):
        if i < len(M_gt) and sum(M_gt[i]) > 0:
            bb_primary.append(max(range(n_s), key=lambda j, m=M_gt[i]: m[j]))
        else:
            bb_primary.append(-1)
    order_ok = 0
    order_total = 0
    for i in range(n_b - 1):
        if bb_primary[i] >= 0 and bb_primary[i + 1] >= 0:
            order_total += 1
            if bb_primary[i + 1] >= bb_primary[i] - 2:
                order_ok += 1
    order_consistency = order_ok / order_total if order_total > 0 else 1.0

    cf_ok = 0
    cf_total = 0
    for s, d, et in zip(edges_src, edges_dst, edge_types):
        if et == 0:  # COND
            if 0 <= s < n_b and 0 <= d < n_b and bb_primary[s] >= 0 and bb_primary[d] >= 0:
                cf_total += 1
                if bb_primary[s] != bb_primary[d]:
                    cf_ok += 1
    cf_consistency = cf_ok / cf_total if cf_total > 0 else 1.0

    return {
        "coverage": coverage,
        "mean_conf": mean_conf,
        "high_conf_ratio": high,
        "low_conf_ratio": low,
        "order_consistency": order_consistency,
        "cf_consistency": cf_consistency,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not Path(path).exists():
        print("用法: python scripts/validate_alignment.py <binskel.jsonl>")
        sys.exit(1)

    n = 0
    bad_lines = []
    sum_cov = 0.0
    sum_mean = 0.0
    sum_high = 0.0
    sum_low = 0.0
    sum_order = 0.0
    sum_cf = 0.0
    has_matrix = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines.append((i, e.msg, line[:80]))
                continue
            M = d.get("alignment_matrix")
            if isinstance(M, list) and M and isinstance(M[0], list) and len(M[0]) > 0:
                has_matrix += 1
            r = validate_sample(d)
            sum_cov += r.get("coverage", 0)
            sum_mean += r.get("mean_conf", 0)
            sum_high += r.get("high_conf_ratio", 0)
            sum_low += r.get("low_conf_ratio", 0)
            sum_order += r.get("order_consistency", 1)
            sum_cf += r.get("cf_consistency", 1)
            n += 1
            del d

    if n == 0:
        print("错误: 未读到有效样本。")
        if bad_lines:
            print(f"  前几条无效行: {bad_lines[:3]}")
        else:
            print(f"  文件可能为空或只有空行: {path}")
        sys.exit(1)
    if bad_lines:
        print(f"警告: 跳过 {len(bad_lines)} 行无效 JSON (前几条: {bad_lines[:3]})\n")

    print(f"文件: {path}")
    print(f"加载 {n} 条样本\n")
    print("=" * 50)
    print("整体统计 (对每条样本取平均)")
    print("=" * 50)

    # 多保留小数，避免 O0–O3 细微差异被 .1f/.2f 打成同一行
    print(f"1. BB 覆盖率:        {100 * sum_cov / n:.2f}%  (平均每条)")
    print(f"2. 平均置信度:       {sum_mean / n:.4f}")
    print(f"   高置信度(>0.7):   {100 * sum_high / n:.2f}%")
    print(f"   低置信度(<0.3):   {100 * sum_low / n:.2f}%")
    print(f"3. 顺序一致性:       {100 * sum_order / n:.2f}%")
    print(f"4. 控制流一致性:     {100 * sum_cf / n:.2f}%")
    print(f"5. 含 M_gt (非空 alignment_matrix): {100 * has_matrix / n:.2f}%")

    issues = []
    if sum_cov / n < 0.7:
        issues.append("覆盖率过低 (<70%)")
    if sum_order / n < 0.8:
        issues.append("顺序一致性差 (<80%)")

    print()
    if not issues:
        print("OK: 对齐质量检查通过（覆盖率/顺序一致性未触发文档阈值告警）")
    else:
        print("WARN: 发现问题:")
        for x in issues:
            print(f"  - {x}")


if __name__ == "__main__":
    main()

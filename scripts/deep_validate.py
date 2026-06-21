#!/usr/bin/env python3
"""
深度验证 binskel 数据集的对齐质量

检查项:
  1. 置信度分布直方图 — 各区间样本占比
  2. 对齐矩阵内部一致性 — M_gt 与 block_src_lines 是否自洽
  3. 单点/多点对齐比例 — 每个 BB 对齐到几条语句
  4. 顺序一致性 & 跳跃检测 — 相邻 BB 对齐的语句是否顺序合理
  5. 异常函数检测 — 置信度极低或覆盖率极低的函数
  6. 随机抽样展示 — 挑几个函数，打印 BB→语句 的对齐详情供人工核查

用法:
  python scripts/deep_validate.py <binskel_o0.jsonl> [--show N]
"""

import json
import sys
import random
from collections import Counter
from pathlib import Path


def analyze_sample(d: dict) -> dict:
    M_gt = d.get("alignment_matrix") or []
    W_mat = d.get("alignment_confidence") or []
    block_lines = d.get("block_src_lines") or []
    source_lines = d.get("source_lines") or []
    func_name = d.get("func_name", "?")
    edge_index = d.get("bin_edge_index") or [[], []]
    edge_types = d.get("bin_edge_type") or []
    sig_stats = d.get("signal_stats") or {}

    n_b = len(block_lines)
    n_s = len(M_gt[0]) if M_gt and M_gt[0] else 0

    if n_b == 0 or n_s == 0:
        return {
            "func_name": func_name, "n_blocks": n_b, "n_stmts": n_s,
            "coverage": 0, "mean_w": 0, "w_values": [],
            "onehot_ratio": 0, "multi_ratio": 0, "empty_ratio": 1,
            "order_consistency": 1, "jumps": [],
            "matrix_lines_consistent": True,
            "max_jump": 0,
            "sig_stats": sig_stats,
        }

    # --- 置信度统计 ---
    w_values = []
    for i in range(n_b):
        for j in range(n_s):
            w = W_mat[i][j] if i < len(W_mat) and j < len(W_mat[i]) else 0
            if w > 0:
                w_values.append(w)
    mean_w = sum(w_values) / len(w_values) if w_values else 0

    # --- BB 覆盖率 ---
    aligned_count = sum(1 for i in range(n_b) if any(M_gt[i][j] > 0 for j in range(n_s)))
    coverage = aligned_count / n_b

    # --- 单点 / 多点 / 空对齐比例 ---
    n_onehot = 0
    n_multi = 0
    n_empty = 0
    bb_primary = []
    for i in range(n_b):
        nonzero = [j for j in range(n_s) if M_gt[i][j] > 0.01]
        if len(nonzero) == 0:
            n_empty += 1
            bb_primary.append(-1)
        elif len(nonzero) == 1:
            n_onehot += 1
            bb_primary.append(nonzero[0])
        else:
            n_multi += 1
            bb_primary.append(max(nonzero, key=lambda j: M_gt[i][j]))

    # --- 顺序一致性 & 跳跃检测 ---
    order_ok = 0
    order_total = 0
    jumps = []
    for i in range(n_b - 1):
        if bb_primary[i] >= 0 and bb_primary[i + 1] >= 0:
            order_total += 1
            diff = bb_primary[i + 1] - bb_primary[i]
            if diff >= -2:
                order_ok += 1
            else:
                jumps.append((i, bb_primary[i], bb_primary[i + 1], diff))
    order_consistency = order_ok / order_total if order_total > 0 else 1.0
    max_jump = max((abs(j[3]) for j in jumps), default=0)

    # --- M_gt 与 block_src_lines 一致性 ---
    consistent = True
    for i in range(min(n_b, len(block_lines))):
        has_lines = len(block_lines[i]) > 0
        has_alignment = any(M_gt[i][j] > 0 for j in range(n_s))
        if has_lines and not has_alignment:
            consistent = False
            break

    return {
        "func_name": func_name,
        "n_blocks": n_b,
        "n_stmts": n_s,
        "coverage": coverage,
        "mean_w": mean_w,
        "w_values": w_values,
        "onehot_ratio": n_onehot / n_b if n_b else 0,
        "multi_ratio": n_multi / n_b if n_b else 0,
        "empty_ratio": n_empty / n_b if n_b else 0,
        "order_consistency": order_consistency,
        "jumps": jumps,
        "max_jump": max_jump,
        "matrix_lines_consistent": consistent,
        "bb_primary": bb_primary,
        "sig_stats": sig_stats,
    }


def format_spot_check(d: dict, result: dict, max_bb: int = 8) -> str:
    lines = []
    func = d.get("func_name", "?")
    src_file = d.get("source_file", "?")
    block_lines = d.get("block_src_lines") or []
    M_gt = d.get("alignment_matrix") or []
    W_mat = d.get("alignment_confidence") or []
    source_lines = d.get("source_lines") or []
    n_s = len(M_gt[0]) if M_gt and M_gt[0] else 0

    lines.append(f"  函数: {func}")
    lines.append(f"  源文件: {src_file}")
    lines.append(f"  BB数: {result['n_blocks']}, 语句数: {result['n_stmts']}")
    lines.append(f"  覆盖率: {result['coverage']:.1%}, 平均W: {result['mean_w']:.3f}")
    lines.append(f"  单点对齐: {result['onehot_ratio']:.1%}, 多点: {result['multi_ratio']:.1%}, 空: {result['empty_ratio']:.1%}")
    lines.append(f"  顺序一致性: {result['order_consistency']:.1%}")
    lines.append("")

    bb_primary = result.get("bb_primary", [])
    n_show = min(max_bb, result["n_blocks"])
    lines.append(f"  前 {n_show} 个 BB 的对齐详情:")
    lines.append(f"  {'BB':>4} | {'src_lines':>20} | {'→stmt_idx':>10} | {'M_gt值':>8} | {'W值':>8}")
    lines.append(f"  {'-'*4}-+-{'-'*20}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for i in range(n_show):
        bl = block_lines[i] if i < len(block_lines) else []
        bl_str = str(bl[:5]) + ("..." if len(bl) > 5 else "")
        if i < len(bb_primary) and bb_primary[i] >= 0:
            j = bb_primary[i]
            m_val = M_gt[i][j] if i < len(M_gt) and j < len(M_gt[i]) else 0
            w_val = W_mat[i][j] if i < len(W_mat) and j < len(W_mat[i]) else 0
            stmt_ln = source_lines[j] if j < len(source_lines) else "?"
            lines.append(f"  {i:>4} | {bl_str:>20} | {j:>5}(L{stmt_ln}) | {m_val:>8.3f} | {w_val:>8.3f}")
        else:
            lines.append(f"  {i:>4} | {bl_str:>20} | {'(空)':>10} | {'---':>8} | {'---':>8}")

    if result["jumps"]:
        lines.append(f"\n  ⚠ 顺序跳跃 ({len(result['jumps'])} 处):")
        for bb_i, s_from, s_to, diff in result["jumps"][:3]:
            lines.append(f"    BB{bb_i}→BB{bb_i+1}: stmt {s_from}→{s_to} (跳 {diff})")

    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="深度验证 binskel 对齐质量")
    ap.add_argument("jsonl", help="binskel JSONL 文件路径")
    ap.add_argument("--show", type=int, default=5, help="随机抽样展示的函数数 (默认 5)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    random.seed(args.seed)

    all_results = []
    all_w_values = []
    raw_samples = []
    n_inconsistent = 0

    print(f"读取 {path.name} ...", flush=True)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            r = analyze_sample(d)
            all_results.append(r)
            all_w_values.extend(r["w_values"])
            if not r["matrix_lines_consistent"]:
                n_inconsistent += 1

            if random.random() < min(200 / max(line_no, 1), 1.0):
                raw_samples.append((d, r))

            del d

    n = len(all_results)
    if n == 0:
        print("无有效样本")
        sys.exit(1)

    print(f"共 {n} 条样本\n")

    # ═══════════════════════════════════════════════════════
    # 1. 置信度分布直方图
    # ═══════════════════════════════════════════════════════
    print("=" * 60)
    print("1. 置信度 W 分布直方图")
    print("=" * 60)

    buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    total_w = len(all_w_values)
    if total_w > 0:
        for lo, hi in buckets:
            cnt = sum(1 for w in all_w_values if lo <= w < hi)
            pct = cnt / total_w
            bar = "█" * int(pct * 50)
            label = f"[{lo:.1f}, {hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}, 1.0]"
            print(f"  {label:>12}  {pct:>6.1%}  {bar}  ({cnt})")
        print(f"\n  总计 {total_w} 个非零 W 值")
        print(f"  均值: {sum(all_w_values)/total_w:.4f}")
        print(f"  中位数: {sorted(all_w_values)[total_w//2]:.4f}")
    else:
        print("  无 W 值 (可能是 O1-O3 数据集，无 alignment_matrix)")

    # ═══════════════════════════════════════════════════════
    # 2. 对齐类型统计
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("2. 对齐类型统计 (每个 BB 对齐到几条语句)")
    print("=" * 60)

    avg_onehot = sum(r["onehot_ratio"] for r in all_results) / n
    avg_multi = sum(r["multi_ratio"] for r in all_results) / n
    avg_empty = sum(r["empty_ratio"] for r in all_results) / n
    print(f"  单点对齐 (M_gt 只有一个非零):  {avg_onehot:.1%}")
    print(f"  多点对齐 (M_gt 有多个非零):    {avg_multi:.1%}")
    print(f"  空对齐 (BB 无对应语句):        {avg_empty:.1%}")
    print(f"\n  理想情况: 单点对齐越高越好 (说明 BB→语句 是确定性的)")
    if avg_multi > 0.3:
        print(f"  ⚠ 多点对齐比例 {avg_multi:.1%} 偏高，可能存在歧义")

    # ═══════════════════════════════════════════════════════
    # 3. 内部一致性
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("3. 内部一致性检查")
    print("=" * 60)

    avg_cov = sum(r["coverage"] for r in all_results) / n
    avg_order = sum(r["order_consistency"] for r in all_results) / n
    print(f"  BB 覆盖率 (有对齐的 BB 占比):  {avg_cov:.1%}")
    print(f"  顺序一致性 (相邻 BB 语句顺序):  {avg_order:.1%}")
    print(f"  M_gt vs block_src_lines 自洽:    {n - n_inconsistent}/{n} ({(n - n_inconsistent)/n:.1%})")

    jump_funcs = [r for r in all_results if r["max_jump"] > 5]
    if jump_funcs:
        print(f"\n  ⚠ {len(jump_funcs)} 个函数有大跳跃 (>5 语句):")
        jump_funcs.sort(key=lambda r: r["max_jump"], reverse=True)
        for r in jump_funcs[:5]:
            print(f"    {r['func_name'][:40]}: 最大跳跃 {r['max_jump']} 语句")

    # ═══════════════════════════════════════════════════════
    # 4. 异常函数检测
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("4. 异常函数检测")
    print("=" * 60)

    low_conf_funcs = [r for r in all_results if r["mean_w"] < 0.3 and r["n_blocks"] >= 3]
    low_cov_funcs = [r for r in all_results if r["coverage"] < 0.5 and r["n_blocks"] >= 3]
    all_empty_funcs = [r for r in all_results if r["empty_ratio"] == 1.0]

    print(f"  低置信度函数 (W<0.3, ≥3BB):    {len(low_conf_funcs)}/{n}")
    if low_conf_funcs:
        for r in sorted(low_conf_funcs, key=lambda x: x["mean_w"])[:5]:
            print(f"    {r['func_name'][:40]}: W={r['mean_w']:.3f}, {r['n_blocks']}BB, 覆盖{r['coverage']:.0%}")

    print(f"  低覆盖率函数 (cov<50%, ≥3BB):  {len(low_cov_funcs)}/{n}")
    if low_cov_funcs:
        for r in sorted(low_cov_funcs, key=lambda x: x["coverage"])[:5]:
            print(f"    {r['func_name'][:40]}: 覆盖{r['coverage']:.0%}, {r['n_blocks']}BB")

    print(f"  完全无对齐的函数:               {len(all_empty_funcs)}/{n}")

    # ═══════════════════════════════════════════════════════
    # 5. 置信度校准检查
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("5. 置信度校准 — 高 W 样本是否确实更一致")
    print("=" * 60)

    hi_results = [r for r in all_results if r["mean_w"] >= 0.7]
    lo_results = [r for r in all_results if 0.3 <= r["mean_w"] < 0.7]

    if hi_results and lo_results:
        hi_order = sum(r["order_consistency"] for r in hi_results) / len(hi_results)
        lo_order = sum(r["order_consistency"] for r in lo_results) / len(lo_results)
        hi_onehot = sum(r["onehot_ratio"] for r in hi_results) / len(hi_results)
        lo_onehot = sum(r["onehot_ratio"] for r in lo_results) / len(lo_results)
        hi_cov = sum(r["coverage"] for r in hi_results) / len(hi_results)
        lo_cov = sum(r["coverage"] for r in lo_results) / len(lo_results)

        print(f"  {'指标':<20} {'高W (≥0.7)':>12} {'中W (0.3-0.7)':>14} {'差值':>8}")
        print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*8}")
        print(f"  {'分组样本数(非总样本数)':<20} {len(hi_results):>12} {len(lo_results):>14}")
        print(f"  {'顺序一致性':<20} {hi_order:>11.1%} {lo_order:>13.1%} {hi_order-lo_order:>+7.1%}")
        print(f"  {'单点对齐比例':<20} {hi_onehot:>11.1%} {lo_onehot:>13.1%} {hi_onehot-lo_onehot:>+7.1%}")
        print(f"  {'BB覆盖率':<20} {hi_cov:>11.1%} {lo_cov:>13.1%} {hi_cov-lo_cov:>+7.1%}")

        if hi_order > lo_order and hi_onehot > lo_onehot:
            print(f"\n  ✓ 校准合理: 高置信度样本的顺序一致性和单点对齐均优于中置信度")
        elif hi_order <= lo_order:
            print(f"\n  ⚠ 校准异常: 高置信度样本的顺序一致性反而更低，W 值可能虚高")
    else:
        print("  样本量不足，无法做校准检查")

    # ═══════════════════════════════════════════════════════
    # 6. 随机抽样展示
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"6. 随机抽样展示 ({args.show} 个函数)")
    print("=" * 60)

    if raw_samples:
        show_samples = random.sample(raw_samples, min(args.show, len(raw_samples)))
        for idx, (d, r) in enumerate(show_samples, 1):
            print(f"\n--- 样本 {idx} ---")
            print(format_spot_check(d, r))
    else:
        print("  无可展示样本")

    # ═══════════════════════════════════════════════════════
    # 7. DWARF-IR 一致率 (MD 第三项指标)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("7. DWARF-IR 一致率")
    print("=" * 60)

    total_both = 0
    total_agree = 0
    total_near = 0
    total_conflict = 0
    total_dwarf_only = 0
    total_ir_only = 0
    total_none = 0
    has_sig_stats = False

    for r in all_results:
        ss = r.get("sig_stats") or {}
        if ss:
            has_sig_stats = True
            total_both += ss.get("n_both", 0)
            total_agree += ss.get("n_agree", 0)
            total_near += ss.get("n_near", 0)
            total_conflict += ss.get("n_conflict", 0)
            total_dwarf_only += ss.get("n_dwarf_only", 0)
            total_ir_only += ss.get("n_ir_only", 0)
            total_none += ss.get("n_none", 0)

    if has_sig_stats and total_both > 0:
        agree_rate = total_agree / total_both
        near_rate = total_near / total_both
        conflict_rate = total_conflict / total_both
        agree_near_rate = (total_agree + total_near) / total_both
        total_all = total_both + total_dwarf_only + total_ir_only + total_none
        print(f"  双信号 BB 数:    {total_both} / {total_all} ({total_both/total_all:.1%})")
        print(f"  完全一致:        {total_agree} ({agree_rate:.1%})")
        print(f"  近似一致:        {total_near} ({near_rate:.1%})")
        print(f"  一致+近似:       {total_agree + total_near} ({agree_near_rate:.1%})")
        print(f"  冲突:            {total_conflict} ({conflict_rate:.1%})")
        print(f"  仅 DWARF:        {total_dwarf_only}")
        print(f"  仅 IR:           {total_ir_only}")
        print(f"  无信号:          {total_none}")
        dwarf_ir_agree = agree_near_rate
    elif not has_sig_stats:
        print("  (数据集无 signal_stats 字段，需用更新后的脚本重新生成)")
        dwarf_ir_agree = None
    else:
        print("  无双信号 BB (可能 IR 未提供)")
        dwarf_ir_agree = None

    # ═══════════════════════════════════════════════════════
    # 8. MD 文档四项指标对照
    # 与 validate_alignment.py 一致：高置信度 = 对每条样本取平均（非全局比例）
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("8. MD 文档四项指标对照")
    print("=" * 60)

    high_conf_per_sample = [
        sum(1 for w in r["w_values"] if w > 0.7) / len(r["w_values"])
        for r in all_results if r["w_values"]
    ]
    high_conf_ratio = sum(high_conf_per_sample) / len(high_conf_per_sample) if high_conf_per_sample else 0

    print(f"  {'指标':<20} {'实际值':>10} {'预期范围':>12} {'判定':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*6}")

    cov_ok = "✓" if 0.90 <= avg_cov <= 0.95 else ("≈" if avg_cov >= 0.85 else "✗")
    print(f"  {'BB 覆盖率':<20} {avg_cov:>9.1%} {'90-95%':>12} {cov_ok:>6}")

    hc_ok = "✓" if 0.75 <= high_conf_ratio <= 0.85 else ("≈" if high_conf_ratio >= 0.60 else "✗")
    print(f"  {'高置信度(>0.7)':<20} {high_conf_ratio:>9.1%} {'75-85%':>12} {hc_ok:>6}")

    if dwarf_ir_agree is not None:
        di_ok = "✓" if 0.85 <= dwarf_ir_agree <= 0.90 else ("≈" if dwarf_ir_agree >= 0.75 else "✗")
        print(f"  {'DWARF-IR 一致率':<20} {dwarf_ir_agree:>9.1%} {'85-90%':>12} {di_ok:>6}")
    else:
        print(f"  {'DWARF-IR 一致率':<20} {'N/A':>10} {'85-90%':>12} {'--':>6}")

    ord_ok = "✓" if avg_order >= 0.90 else ("≈" if avg_order >= 0.85 else "✗")
    print(f"  {'顺序一致性':<20} {avg_order:>9.1%} {'90%+':>12} {ord_ok:>6}")

    # ═══════════════════════════════════════════════════════
    # 9. 总结
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("总结")
    print("=" * 60)

    issues = []
    if avg_cov < 0.85:
        issues.append(f"BB 覆盖率偏低: {avg_cov:.1%} (预期 90-95%)")
    if high_conf_ratio < 0.60:
        issues.append(f"高置信度比例过低: {high_conf_ratio:.1%} (预期 75-85%)")
    if dwarf_ir_agree is not None and dwarf_ir_agree < 0.75:
        issues.append(f"DWARF-IR 一致率偏低: {dwarf_ir_agree:.1%} (预期 85-90%)")
    if avg_order < 0.85:
        issues.append(f"顺序一致性差: {avg_order:.1%} (预期 90%+)")
    if avg_multi > 0.4:
        issues.append(f"多点对齐比例过高: {avg_multi:.1%}")
    if n_inconsistent > n * 0.1:
        issues.append(f"内部不一致样本过多: {n_inconsistent}/{n}")
    if total_w > 0:
        median_w = sorted(all_w_values)[total_w // 2]
        if median_w > 0.85:
            issues.append(f"W 中位数 {median_w:.3f} 偏高，可能存在置信度膨胀")

    print(f"  样本数: {n}")
    print(f"  BB 覆盖率: {avg_cov:.1%}")
    if total_w:
        print(f"  平均 W: {sum(all_w_values)/total_w:.3f}")
        print(f"  高置信度(>0.7): {high_conf_ratio:.1%}")
    if dwarf_ir_agree is not None:
        print(f"  DWARF-IR 一致率: {dwarf_ir_agree:.1%}")
    print(f"  顺序一致性: {avg_order:.1%}")
    print(f"  单点对齐: {avg_onehot:.1%}")

    if not issues:
        print("\n  ✓ 所有检查通过，对齐质量符合 MD 预期")
    else:
        print(f"\n  ⚠ 发现 {len(issues)} 个问题:")
        for x in issues:
            print(f"    - {x}")


if __name__ == "__main__":
    main()

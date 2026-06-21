#!/usr/bin/env python3
"""
从四个项目 O0 数据集中各随机抽取 5 个函数，展示 BB→源码对齐详情供人工核查。

用法: python scripts/sample_align_check.py [--seed 42]
"""

import json
import random
import argparse
from pathlib import Path

DATASETS = {
    "zlib": "~/structlift_datasets/binskel_zlib_o0.jsonl",
    "sqlite": "~/structlift_datasets/binskel_sqlite_o0.jsonl",
    "busybox": "~/structlift_datasets/binskel_busybox_o0.jsonl",
    "openssl": "~/structlift_datasets/binskel_openssl_o0.jsonl",
}


def read_source_lines(filepath: str, line_nums: list) -> dict:
    """读取源文件中指定行号的内容。"""
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        return {ln: f"[文件不存在: {path}]" for ln in line_nums}
    result = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in sorted(set(line_nums)):
            if 1 <= ln <= len(lines):
                result[ln] = lines[ln - 1].rstrip()
            else:
                result[ln] = "[行号越界]"
    except Exception as e:
        return {ln: f"[读取失败: {e}]" for ln in line_nums}
    return result


def format_sample(proj: str, d: dict, line_to_text: dict, max_bb: int = 10) -> str:
    """格式化单条样本的 BB 对齐详情。"""
    fn = d.get("func_name", "?")
    sf = d.get("source_file", "")
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    n_b = len(bl)
    n_s = len(M[0]) if M and M[0] else 0

    lines = []
    lines.append(f"  函数: {fn}")
    lines.append(f"  源文件: {sf}")
    lines.append(f"  BB 数: {n_b}, 语句列数: {n_s}")
    if n_b > max_bb:
        lines.append(f"  (仅展示前 {max_bb} 个 BB)")
    lines.append("")

    show_indices = list(range(min(n_b, max_bb)))
    for i in show_indices:
        ln_list = bl[i] if i < len(bl) else []
        lines.append(f"  --- BB {i} ---")
        lines.append(f"    对应行号: {ln_list}")
        lines.append("    源码:")
        if not ln_list:
            lines.append("      [无]")
        else:
            for ln in sorted(ln_list):
                src = line_to_text.get(ln, "")
                lines.append(f"      {ln:4d} | {src[:100]}")
        if M and i < len(M):
            nonzero = [(j, M[i][j], W[i][j] if W and j < len(W[i]) else 0) for j in range(n_s) if M[i][j] > 0.01]
            if nonzero:
                lines.append(f"    对齐语句: {[(j, f'M={m:.2f}', f'W={w:.2f}') for j, m, w in nonzero]}")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=5, help="每项目抽样数")
    ap.add_argument("--max-bb", type=int, default=10, help="每函数最多展示 BB 数")
    ap.add_argument("--projects", nargs="*", default=None, help="仅检查指定项目，如 zlib sqlite")
    ap.add_argument("--min-mean-conf", type=float, default=None, help="仅抽样高置信度函数：平均 W >= 此值，如 0.85")
    args = ap.parse_args()
    random.seed(args.seed)

    datasets = DATASETS
    if args.projects:
        datasets = {k: v for k, v in DATASETS.items() if k in args.projects}

    for proj, rel_path in datasets.items():
        path = Path(rel_path).expanduser()
        if not path.exists():
            print(f"[跳过] {proj}: 文件不存在 {path}")
            continue

        samples = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))

        if args.min_mean_conf is not None:
            pool = []
            for d in samples:
                W = d.get("alignment_confidence") or []
                M = d.get("alignment_matrix") or []
                vals = []
                for i in range(len(W)):
                    for j in range(len(W[i]) if W and W[i] else 0):
                        if M and i < len(M) and j < len(M[i]) and M[i][j] > 0.01:
                            vals.append(W[i][j])
                mean_w = sum(vals) / len(vals) if vals else 0
                if mean_w >= args.min_mean_conf:
                    pool.append(d)
            samples = pool
            if not samples:
                print(f"[跳过] {proj}: 无满足 --min-mean-conf>={args.min_mean_conf} 的样本")
                continue

        if len(samples) < args.n:
            chosen = samples
        else:
            chosen = random.sample(samples, args.n)

        print("=" * 70)
        title = f"项目: {proj}  (随机抽样 {len(chosen)} 个函数"
        if args.min_mean_conf is not None:
            title += f", 仅高置信度 mean_W>={args.min_mean_conf}"
        title += ")"
        print(title)
        print("=" * 70)

        for i, d in enumerate(chosen):
            bl = d.get("block_src_lines") or []
            all_lines = sorted(set(ln for ls in bl for ln in ls))
            sf = d.get("source_file", "")
            line_to_text = read_source_lines(sf, all_lines) if sf and all_lines else {}

            print(f"\n--- 样本 {i+1}/{len(chosen)} ---")
            print(format_sample(proj, d, line_to_text, args.max_bb))

        print()


if __name__ == "__main__":
    main()

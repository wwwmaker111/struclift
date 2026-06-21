#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从单个 binskel jsonl 中按策略抽取若干条，打印「BB -> 主对齐语句 + M/W」便于人工判断对齐好坏。

与汇总表的关系：表里的均信/高置%是整体统计；本脚本看具体函数里对齐是否合理。

用法:
  python3 scripts/sample_alignment_inspect.py /mnt/e/structlift_datasets/binskel_ffmpeg_o0.jsonl

  python3 scripts/sample_alignment_inspect.py /path/to.jsonl --strategy random --total 10 --max-bb 16

  python3 scripts/sample_alignment_inspect.py /path/to.jsonl --strategy high --total 6

策略 --strategy:
  mixed (默认): 高/低/随机各三分之一（尽量）
  random: 均匀随机
  high: 按 validate_alignment 的 mean_conf 从高到低取
  low:  按 mean_conf 从低到高取（优先 coverage>0）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deep_validate import analyze_sample, format_spot_check  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402


def _scan_scores(
    path: Path,
    progress_label: str | None = None,
    progress_every: int = 0,
) -> list[tuple[int, float, float, str]]:
    """返回 [(line_idx, mean_conf, coverage, func_name), ...] 仅有效 JSON 行。

    progress_every>0 时每隔若干条向 stderr 打进度（大 jsonl 整文件扫描较慢）。
    """
    out: list[tuple[int, float, float, str]] = []
    idx = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx += 1
            vr = validate_sample(d)
            fn = str(d.get("func_name") or "?")[:64]
            out.append((idx, float(vr.get("mean_conf", 0)), float(vr.get("coverage", 0)), fn))
            if progress_every > 0 and progress_label and (idx + 1) % progress_every == 0:
                print(
                    f"[进度] {progress_label} 已扫 {idx + 1} 条有效样本 ...",
                    file=sys.stderr,
                    flush=True,
                )
    return out


def _load_at_indices(path: Path, want: set[int]) -> dict[int, dict]:
    got: dict[int, dict] = {}
    idx = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx += 1
            if idx not in want:
                continue
            got[idx] = d
            if len(got) == len(want):
                break
    return got


def _pick_indices(
    scores: list[tuple[int, float, float, str]],
    total: int,
    strategy: str,
    seed: int,
) -> list[tuple[int, str]]:
    """返回 [(line_idx, reason), ...]"""
    random.seed(seed)
    if not scores:
        return []

    indexed = list(scores)

    if strategy == "random":
        k = min(total, len(indexed))
        picked = random.sample(indexed, k)
        return [(t[0], "random") for t in picked]

    by_conf = sorted(indexed, key=lambda x: (x[1], x[0]))
    n = len(by_conf)

    if strategy == "high":
        k = min(total, n)
        top = by_conf[-k:]
        return [(t[0], f"high_mean_conf={t[1]:.4f}") for t in reversed(top)]

    if strategy == "low":
        k = min(total, n)
        with_cov = [t for t in by_conf if t[2] > 0.01]
        pool = with_cov if len(with_cov) >= k else by_conf
        bottom = pool[:k]
        return [(t[0], f"low_mean_conf={t[1]:.4f}") for t in bottom]

    k = min(total, n)
    n_high = max(1, k // 3)
    n_low = max(1, k // 3)
    n_rand = max(1, k - n_high - n_low)
    if n_high + n_low + n_rand > k:
        n_rand = max(0, k - n_high - n_low)

    out_idx: set[int] = set()
    reasons: dict[int, str] = {}

    hi_pool = [t for t in by_conf if t[2] > 0.01]
    hi_sorted = sorted(hi_pool, key=lambda x: (x[1], x[0]))
    for t in reversed(hi_sorted[-n_high:]):
        out_idx.add(t[0])
        reasons[t[0]] = f"high_mean_conf={t[1]:.4f}"

    lo_pool = [t for t in by_conf if t[2] > 0.01]
    lo_sorted = sorted(lo_pool, key=lambda x: (x[1], x[0]))
    for t in lo_sorted[:n_low]:
        if t[0] in out_idx:
            continue
        out_idx.add(t[0])
        reasons[t[0]] = f"low_mean_conf={t[1]:.4f}"

    candidates = [t for t in indexed if t[0] not in out_idx]
    need_rand = min(n_rand, len(candidates))
    if need_rand > 0:
        for t in random.sample(candidates, need_rand):
            out_idx.add(t[0])
            reasons[t[0]] = "random"

    guard = 0
    while len(out_idx) < min(k, n) and guard < max(n * 3, 100):
        guard += 1
        t = random.choice(indexed)
        if t[0] not in out_idx:
            out_idx.add(t[0])
            reasons[t[0]] = "fill_random"

    order = sorted(out_idx)
    return [(i, reasons[i]) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取 binskel 样本打印对齐细节（人工看好坏）")
    ap.add_argument("jsonl", type=Path, help="binskel .jsonl 路径")
    ap.add_argument("--strategy", choices=("mixed", "random", "high", "low"), default="mixed")
    ap.add_argument("--total", type=int, default=10, help="抽取条数（默认 10）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bb", type=int, default=12, help="每个函数打印前多少个 BB（默认 12）")
    args = ap.parse_args()

    path = args.jsonl.expanduser().resolve()
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"扫描 {path}（计算每条 mean_conf / coverage）...", flush=True)
    scores = _scan_scores(path)
    if not scores:
        print("无有效样本", file=sys.stderr)
        sys.exit(1)

    picked = _pick_indices(scores, args.total, args.strategy, args.seed)
    want = {i for i, _ in picked}
    print(f"加载 {len(want)} 条并打印...\n", flush=True)
    data = _load_at_indices(path, want)

    for rank, (line_idx, why) in enumerate(picked, 1):
        d = data.get(line_idx)
        print("=" * 72)
        print(f"#{rank} 行下标={line_idx}  策略: {why}")
        if not d:
            print("  (未读到)")
            continue
        ar = analyze_sample(d)
        vr = validate_sample(d)
        print(
            f"  validate: cov={vr['coverage']:.3f} mean_conf={vr['mean_conf']:.4f} "
            f"order={vr['order_consistency']:.3f} cf={vr['cf_consistency']:.3f}"
        )
        print(
            f"  自洽(M vs block_lines): "
            f"{'PASS' if ar['matrix_lines_consistent'] else 'FAIL'}"
        )
        print(format_spot_check(d, ar, max_bb=args.max_bb))
        print()


if __name__ == "__main__":
    main()

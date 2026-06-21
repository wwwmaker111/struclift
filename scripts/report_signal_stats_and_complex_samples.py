#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与 binskel_alignment_batch_stats 相同搜索路径，汇总：
1) 各 jsonl 内 signal_stats 的样本均值（n_conflict / n_agree / …）
2) 各项目 O0：按 len(bin_block_lengths) 取前 K 条「复杂」函数，供人工 spot-check

用法:
  python3 scripts/report_signal_stats_and_complex_samples.py --out-dir /mnt/e/structlift --all-default-paths
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from binskel_alignment_batch_stats import (  # noqa: E402
    DEFAULT_PROJECTS,
    dedupe_paths,
    locate_jsonl,
    default_multi_root_search_list,
)


def aggregate_signals(path: Path | None) -> dict | None:
    if not path or not path.is_file():
        return None
    keys = ("n_both", "n_agree", "n_near", "n_conflict", "n_dwarf_only", "n_ir_only", "n_none")
    sums = {k: 0.0 for k in keys}
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ss = d.get("signal_stats") or {}
            for k in keys:
                sums[k] += float(ss.get(k, 0) or 0)
            n += 1
    if n == 0:
        return {"n": 0, "per_sample": {k: 0.0 for k in keys}}
    per = {k: sums[k] / n for k in keys}
    return {"n": n, "per_sample": per, "total_conflict": sums["n_conflict"]}


def top_complex_samples(path: Path | None, k: int = 3) -> list[dict]:
    if not path or not path.is_file():
        return []
    rows: list[tuple[int, dict]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            bl = d.get("bin_block_lengths") or []
            nb = len(bl) if isinstance(bl, list) else 0
            ei = d.get("bin_edge_index")
            ne = 0
            if isinstance(ei, list) and len(ei) >= 2 and isinstance(ei[0], list):
                ne = len(ei[0])
            rows.append((nb * 1000 + ne, d))  # BB 优先，其次边数
    rows.sort(key=lambda x: -x[0])
    out = []
    for _, d in rows[:k]:
        ss = d.get("signal_stats") or {}
        out.append(
            {
                "func_name": d.get("func_name", "?"),
                "n_bb": len(d.get("bin_block_lengths") or []),
                "n_edges": (
                    len((d.get("bin_edge_index") or [[], []])[0])
                    if isinstance(d.get("bin_edge_index"), list)
                    else 0
                ),
                "n_conflict": ss.get("n_conflict", 0),
                "n_agree": ss.get("n_agree", 0),
                "bb_cov": d.get("bb_alignment_coverage"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help=f"逗号分隔（默认 {','.join(DEFAULT_PROJECTS)}）",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if args.all_default_paths:
        search_dirs = default_multi_root_search_list(out_dir)
        for p in args.search_dir:
            search_dirs.extend([Path(p).expanduser().resolve()])
        search_dirs = dedupe_paths(search_dirs)
    else:
        search_dirs = [out_dir] + [Path(p).expanduser().resolve() for p in args.search_dir]

    projects = [x.strip() for x in args.projects.split(",") if x.strip()]

    print("## 一、signal_stats 按样本平均（全文件聚合）\n")
    print(
        "| 数据集 | 样本数 | n_conflict | n_agree | n_near | n_both | n_dwarf_only | n_ir_only | n_none |"
    )
    print("|--------|--------|------------|---------|--------|--------|--------------|-----------|--------|")

    complex_by_proj: dict[str, list[dict]] = {}

    for proj in projects:
        for opt in range(4):
            name = f"binskel_{proj}_o{opt}.jsonl"
            path_used, _ = locate_jsonl(name, search_dirs)
            stat = aggregate_signals(path_used)
            if stat is None or stat["n"] == 0:
                print(f"| {name} | - | - | - | - | - | - | - | - |")
                continue
            ps = stat["per_sample"]
            print(
                f"| {name} | {stat['n']} | {ps['n_conflict']:.4f} | {ps['n_agree']:.4f} | "
                f"{ps['n_near']:.4f} | {ps['n_both']:.4f} | {ps['n_dwarf_only']:.4f} | "
                f"{ps['n_ir_only']:.4f} | {ps['n_none']:.4f} |"
            )
            if opt == 0:
                complex_by_proj[proj] = top_complex_samples(path_used, k=args.top_k)

    print("\n## 二、各项目 O0：BB 数最多的前 %d 条（便于人工抽查）\n" % args.top_k)
    for proj in projects:
        samples = complex_by_proj.get(proj) or []
        print(f"### {proj} (O0)\n")
        if not samples:
            print("（无数据或文件缺失）\n")
            continue
        for i, s in enumerate(samples, 1):
            print(
                f"{i}. `{s['func_name']}` — BB={s['n_bb']}, edges={s['n_edges']}, "
                f"n_conflict={s['n_conflict']}, n_agree={s['n_agree']}, bb_cov={s['bb_cov']}"
            )
        print()


if __name__ == "__main__":
    main()

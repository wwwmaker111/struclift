#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对默认项目 × 4 档各抽取 N 条样本，做内部一致性检查（与 deep_validate 同源逻辑）。

不访问人工金标准；检查项：
  - M_gt 与 block_src_lines 是否自洽（有行无对齐则标 FAIL）
  - validate_alignment 单条指标
  - analyze_sample：顺序跳跃、空 BB 比例等

用法:
  python3 scripts/audit_two_samples_all_projects.py
  python3 scripts/audit_two_samples_all_projects.py --out-dir ~/structlift_datasets
  python3 scripts/audit_two_samples_all_projects.py --all-default-paths
  python3 scripts/audit_two_samples_all_projects.py --compact   # 每文件一行摘要
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
    default_multi_root_search_list,
    locate_jsonl,
)
from deep_validate import analyze_sample, format_spot_check  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402


def _pick_spread_indices(n_valid: int, want: int) -> list[int]:
    """在有效行下标 0..n-1 内取 want 个尽量均匀分散的下标（确定性，与置信度无关）。"""
    if n_valid <= 0 or want <= 0:
        return []
    m = min(want, n_valid)
    if m == 1:
        return [0]
    raw = [int(round(i * (n_valid - 1) / (m - 1))) for i in range(m)]
    out: list[int] = []
    for x in raw:
        x = min(max(x, 0), n_valid - 1)
        if not out or out[-1] != x:
            out.append(x)
    # 去重后若不足 m（极小 n_valid），补相邻未用下标
    if len(out) < m:
        used = set(out)
        for j in range(n_valid):
            if j not in used:
                out.append(j)
                used.add(j)
                if len(out) >= m:
                    break
        out.sort()
    return out[:m]


def _load_samples_at_indices(path: Path, indices: list[int]) -> list[tuple[int, dict | None]]:
    """只解析需要的若干条「有效 JSON 记录」，下标与 _count_valid_lines 一致。按 indices 顺序返回。"""
    idx_needed = sorted(set(indices))
    got: dict[int, dict | None] = {}
    cur = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cur += 1
            if cur not in idx_needed:
                continue
            got[cur] = d
            if len(got) == len(idx_needed):
                break
    return [(i, got.get(i)) for i in indices]


def _count_valid_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
    return n


def _one_line_summary(name: str, d: dict, ar: dict, vr: dict) -> str:
    fn = d.get("func_name", "?")
    ok = ar.get("matrix_lines_consistent", True)
    cov = vr.get("coverage", 0)
    mc = ar.get("mean_w", 0)
    flag = "OK" if ok and cov > 0 else ("WARN" if cov > 0 else "EMPTY")
    if ar.get("jumps"):
        flag = "JUMP" if flag == "OK" else flag
    return (
        f"{name}: [{flag}] {fn[:40]} cov={cov:.2f} meanW={mc:.3f} "
        f"order={vr.get('order_consistency', 0):.2f} cf={vr.get('cf_consistency', 0):.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="28 档 binskel 各抽 N 条样本做一致性审计")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help="逗号分隔项目名",
    )
    ap.add_argument("--compact", action="store_true", help="每文件仅打印两行摘要")
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if args.all_default_paths:
        search_dirs = default_multi_root_search_list(out_dir)
    else:
        search_dirs = [out_dir]

    projects = [x.strip() for x in args.projects.split(",") if x.strip()]

    print("搜索路径:", search_dirs)
    print()

    n_files = 0
    n_fail_consistency = 0
    n_total_samples = 0

    for proj in projects:
        for k in range(4):
            name = f"binskel_{proj}_o{k}.jsonl"
            path, root = locate_jsonl(name, search_dirs)
            if not path:
                print(f"### 缺失 {name}\n")
                continue
            n_files += 1
            n_valid = _count_valid_lines(path)
            i0, i1 = _pick_two_indices(n_valid)
            if n_valid == 0:
                print(f"### {name} (无有效行)\n")
                continue

            d0, d1 = _load_samples_at_indices(path, i0, i1)
            for tag, d, idx in [("#1", d0, i0), ("#2", d1, i1)]:
                if d is None:
                    continue
                if not d:
                    print(f"  {name} {tag} JSON 解析失败 (line index {idx})")
                    continue
                n_total_samples += 1
                ar = analyze_sample(d)
                vr = validate_sample(d)
                if not ar["matrix_lines_consistent"]:
                    n_fail_consistency += 1

                if args.compact:
                    print(_one_line_summary(f"{name} {tag}", d, ar, vr))
                else:
                    print("=" * 72)
                    print(f"文件: {name}")
                    print(f"来源: {root}")
                    print(f"抽样: 有效行数={n_valid}, 行下标={idx} ({tag})")
                    print(
                        f"一致性(矩阵 vs block_src_lines): "
                        f"{'PASS' if ar['matrix_lines_consistent'] else 'FAIL'}"
                    )
                    print(
                        f"validate: cov={vr['coverage']:.3f} mean_conf={vr['mean_conf']:.3f} "
                        f"high%={vr['high_conf_ratio']:.3f} order={vr['order_consistency']:.3f} "
                        f"cf={vr['cf_consistency']:.3f}"
                    )
                    print(format_spot_check(d, ar))
                    print()

    print()
    print(
        f"结束: 扫描到 {n_files} 个 jsonl, 共审计约 {n_total_samples} 条样本 "
        f"(每文件最多 2 条)。"
    )
    print(f"矩阵与 block_src_lines 不一致样本数: {n_fail_consistency}")
    print(
        "说明: 此为数据集内部自洽检查，不等价于与人工标注一致；"
        "FAIL/JUMP 建议用 deep_validate.py 对单文件深挖。"
    )


if __name__ == "__main__":
    main()

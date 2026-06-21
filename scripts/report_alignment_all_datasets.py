#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总所有 binskel 数据集的对齐效果（与 validate_alignment.py 同一套指标）。

指标（对每条样本取平均）:
  - 样本数
  - BB 覆盖率
  - 平均置信度
  - 高置信度占比 (W>0.7，按非零 W 计数)
  - 低置信度占比 (W<0.3)
  - 顺序一致性
  - 控制流一致性 (COND 边)
  - 带 alignment_matrix 的样本占比（判断是否为 tree-sitter 细对齐）

用法:
  python scripts/report_alignment_all_datasets.py [输出目录]
  python scripts/report_alignment_all_datasets.py ~/structlift_datasets --csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 复用 validate_sample
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from validate_alignment import validate_sample

DEFAULT_PROJECTS = ("zlib", "sqlite", "busybox", "openssl", "coreutils", "curl", "libxml2")
OPTS = ("0", "1", "2", "3")


def analyze_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    n = 0
    has_matrix = 0
    sum_cov = sum_mean = sum_high = sum_low = sum_order = sum_cf = 0.0
    bad = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            M = d.get("alignment_matrix")
            if isinstance(M, list) and M and isinstance(M[0], list) and len(M[0]) > 0:
                has_matrix += 1
            r = validate_sample(d)
            sum_cov += r["coverage"]
            sum_mean += r["mean_conf"]
            sum_high += r["high_conf_ratio"]
            sum_low += r["low_conf_ratio"]
            sum_order += r["order_consistency"]
            sum_cf += r["cf_consistency"]
            n += 1

    if n == 0:
        return {
            "path": str(path),
            "n": 0,
            "bad_lines": bad,
            "coverage_pct": 0.0,
            "mean_conf": 0.0,
            "high_conf_pct": 0.0,
            "low_conf_pct": 0.0,
            "order_pct": 0.0,
            "cf_pct": 0.0,
            "matrix_ratio_pct": 0.0,
        }

    return {
        "path": str(path),
        "n": n,
        "bad_lines": bad,
        "coverage_pct": 100 * sum_cov / n,
        "mean_conf": sum_mean / n,
        "high_conf_pct": 100 * sum_high / n,
        "low_conf_pct": 100 * sum_low / n,
        "order_pct": 100 * sum_order / n,
        "cf_pct": 100 * sum_cf / n,
        "matrix_ratio_pct": 100 * has_matrix / n,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="汇总 binskel 对齐指标")
    ap.add_argument("out_dir", nargs="?", default=str(Path.home() / "structlift_datasets"))
    ap.add_argument("--csv", action="store_true", help="输出 CSV 到标准输出")
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS), help="逗号分隔项目名")
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser()
    projects = tuple(p.strip() for p in args.projects.split(",") if p.strip())

    rows: list[tuple[str, dict | None]] = []
    for proj in projects:
        for o in OPTS:
            name = f"binskel_{proj}_o{o}.jsonl"
            p = out / name
            rows.append((name, analyze_file(p)))

    if args.csv:
        hdr = (
            "dataset,n_samples,bad_json,bb_coverage_pct,mean_conf,"
            "high_conf_pct,low_conf_pct,order_pct,cf_pct,has_matrix_pct"
        )
        print(hdr)
        for name, r in rows:
            if r is None or r["n"] == 0:
                print(f"{name},0,,,,,,,,")
                continue
            print(
                f"{name},{r['n']},{r['bad_lines']},"
                f"{r['coverage_pct']:.2f},{r['mean_conf']:.4f},"
                f"{r['high_conf_pct']:.2f},{r['low_conf_pct']:.2f},"
                f"{r['order_pct']:.2f},{r['cf_pct']:.2f},{r['matrix_ratio_pct']:.2f}"
            )
        return

    w_name = 24
    print("=" * 120)
    print(f" binskel 对齐效果汇总  |  目录: {out}")
    print("=" * 120)
    print(
        f"{'数据集':<{w_name}} {'样本':>8} {'BB覆盖率%':>10} {'平均置信':>10} {'高置信%':>9} {'低置信%':>9} "
        f"{'顺序%':>8} {'控制流%':>8} {'含M_gt%':>8}"
    )
    print("-" * 120)

    total_n = 0
    for name, r in rows:
        if r is None or r["n"] == 0:
            print(f"{name:<{w_name}} {'0':>8} {'-':>10} {'-':>10} {'-':>9} {'-':>9} {'-':>8} {'-':>8} {'-':>8}")
            continue
        total_n += r["n"]
        print(
            f"{name:<{w_name}} {r['n']:>8} {r['coverage_pct']:>10.2f} {r['mean_conf']:>10.3f} "
            f"{r['high_conf_pct']:>9.2f} {r['low_conf_pct']:>9.2f} {r['order_pct']:>8.2f} {r['cf_pct']:>8.2f} "
            f"{r['matrix_ratio_pct']:>8.1f}"
        )

    print("-" * 120)
    print(f"{'合计样本数':<{w_name}} {total_n:>8}")
    print()
    print("说明:")
    print("  - BB覆盖率 / 平均置信 / 高·低置信 / 顺序 / 控制流: 与 scripts/validate_alignment.py 一致（逐样本平均）")
    print("  - 含M_gt%: 该文件中带有非空 alignment_matrix 的样本比例；较低时常为行级回退或未跑 tree-sitter")
    print("  - 若某数据集为 '-' 或样本为 0: 文件不存在或为空")
    print()
    print("一键 CSV: python scripts/report_alignment_all_datasets.py", out, "--csv")


if __name__ == "__main__":
    main()

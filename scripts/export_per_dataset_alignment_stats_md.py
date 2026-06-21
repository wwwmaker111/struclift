#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 9 项目 × 5 档（与 stats_9proj_5opt_prefer_multibin 相同选文件规则），
每个 jsonl 输出一块「整体统计（对每条样本取平均）」Markdown 表（与 validate_alignment 汇总一致）。

用法:
  python scripts/export_per_dataset_alignment_stats_md.py \\
    --datasets E:/structlift_datasets \\
    -o E:/structlift_datasets/alignment_stats_per_dataset.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from binskel_alignment_batch_stats import DEFAULT_PROJECTS, aggregate_jsonl  # noqa: E402

MULTIBIN = frozenset({"coreutils", "ffmpeg", "openssh"})


def pick_path(proj: str, opt: str, base: Path) -> tuple[Path | None, str]:
    if proj in MULTIBIN:
        if opt == "os":
            cand = base / f"binskel_{proj}_os_multibin.jsonl"
        else:
            cand = base / f"binskel_{proj}_o{opt}_multibin.jsonl"
        if cand.is_file():
            return cand, cand.name
    if opt == "os":
        cand = base / f"binskel_{proj}_os.jsonl"
    else:
        cand = base / f"binskel_{proj}_o{opt}.jsonl"
    if cand.is_file():
        return cand, cand.name
    return None, cand.name


def format_table(st: dict) -> str:
    if not st or st["n"] == 0:
        return (
            "| 指标 | 数值 |\n"
            "| :--- | :--- |\n"
            "| （无有效样本） | — |\n"
        )
    return (
        "| 指标 | 数值 |\n"
        "| :--- | :--- |\n"
        f"| BB 覆盖率（平均每条） | {st['cov_pct']:.2f}% |\n"
        f"| 平均置信度 | {st['mean_conf']:.4f} |\n"
        f"| 高置信度（>0.7） | {st['high_pct']:.2f}% |\n"
        f"| 低置信度（<0.3） | {st['low_pct']:.2f}% |\n"
        f"| 顺序一致性 | {st['order_pct']:.2f}% |\n"
        f"| 控制流一致性 | {st['cf_pct']:.2f}% |\n"
        f"| 含 M_gt（非空 alignment_matrix） | {st['m_gt_pct']:.2f}% |\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="每数据集一块 validate_alignment 风格 Markdown 表")
    ap.add_argument("--datasets", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help="逗号分隔，默认 9 项目",
    )
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    base = args.datasets.expanduser().resolve()
    projects = [x.strip() for x in args.projects.split(",") if x.strip()]
    opts = ["0", "1", "2", "3", "os"]

    parts: list[str] = [
        "# 各数据集对齐整体统计\n",
        "\n",
        "口径：`validate_alignment.validate_sample` 对**每条**样本算指标后，在文件内取**算术平均**；"
        "与 `python scripts/validate_alignment.py <jsonl>` 打印的「整体统计」一致。\n",
        "\n",
        "选文件规则：**coreutils / ffmpeg / openssh** 优先 `*_multibin.jsonl`，否则标准文件名。\n",
        "\n",
        "---\n\n",
    ]

    for proj in projects:
        for opt in opts:
            label = f"{proj}_{opt}"
            path, fname = pick_path(proj, opt, base)
            parts.append(f"## `{label}` · `{fname}`\n\n")
            parts.append("**整体统计（对每条样本取平均）**\n\n")
            if path is None or not path.is_file():
                parts.append("*文件缺失，未统计。*\n\n")
                parts.append("| 指标 | 数值 |\n| :--- | :--- |\n| — | — |\n\n")
                parts.append("---\n\n")
                continue
            st = aggregate_jsonl(path)
            parts.append(format_table(st or {}))
            if st and st.get("n"):
                parts.append(f"\n*样本数: {st['n']:,}；无效 JSON 行: {st.get('bad_lines', 0)}*\n")
            parts.append("\n---\n\n")

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

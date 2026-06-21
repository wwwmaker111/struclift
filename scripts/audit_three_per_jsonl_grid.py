#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对「8 项目 × 4 档」binskel jsonl 网格：每个文件抽若干条函数（默认 3 条），
可选 **高置信 + 多基本块** 过滤后再抽样（与 validate_alignment.validate_sample 的 mean_conf 一致）。

复用 audit_35_samples_semantics.semantic_heuristic 做自动化启发式语义检查。

说明: 这是规则/行号/矩阵自洽类启发式，不是人工对照反汇编金标准。
用法:
  # 默认：≥5 个 BB 且 mean_conf ≥ 0.7，每文件 3 条 → 32×3=96
  python scripts/audit_three_per_jsonl_grid.py --datasets ~/structlift_datasets -o audit_grid_hq.tsv

  # 旧行为（仅排除单块）
  python scripts/audit_three_per_jsonl_grid.py --datasets ~/structlift_datasets --min-bb 2 --no-mean-conf-filter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_35_samples_semantics import (  # noqa: E402
    load_by_line_idx,
    semantic_heuristic,
)
from validate_alignment import validate_sample  # noqa: E402

# 与论文/BinSkel 八项一致，按项目×档位顺序
PROJECTS = [
    "zlib",
    "curl",
    "sqlite",
    "busybox",
    "openssl",
    "openssh",
    "libxml2",
    "ffmpeg",
]
OPTS = ["o0", "o1", "o2", "o3"]


def iter_jsonl_names() -> list[str]:
    return [f"binskel_{p}_{o}.jsonl" for p in PROJECTS for o in OPTS]


def count_basic_blocks(d: dict) -> int:
    """与 binskel 样本一致：基本块数优先取 block_src_lines 长度，否则 alignment_matrix 行数。"""
    bl = d.get("block_src_lines")
    if isinstance(bl, list) and len(bl) > 0:
        return len(bl)
    M = d.get("alignment_matrix")
    if isinstance(M, list) and len(M) > 0:
        return len(M)
    return 0


def iter_filtered_line_indices(
    path: Path,
    min_bb: int,
    min_mean_conf: float | None,
) -> list[int]:
    """扫描 jsonl，返回同时满足 BB 数与（可选）mean_conf 的样本行号。"""
    out: list[int] = []
    idx = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if count_basic_blocks(d) < min_bb:
                continue
            if min_mean_conf is not None:
                mc = float(validate_sample(d).get("mean_conf", 0.0))
                if mc < min_mean_conf:
                    continue
            out.append(idx)
    return out


def pick_spread_line_indices(candidates: list[int], k: int) -> list[int]:
    """在候选行号列表上均匀取 k 个（候选已按文件顺序递增）。"""
    m = len(candidates)
    if m == 0 or k <= 0:
        return []
    if m <= k:
        return candidates[:m]
    if k == 1:
        return [candidates[m // 2]]
    positions = sorted({int((m - 1) * i / (k - 1)) for i in range(k)})
    return [candidates[p] for p in positions]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        type=Path,
        required=True,
        help="含 binskel_*_o*.jsonl 的目录（如 ~/structlift_datasets）",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="可选：TSV 报告路径",
    )
    ap.add_argument(
        "--per-file",
        type=int,
        default=3,
        help="每个 jsonl 抽取条数（默认 3）",
    )
    ap.add_argument(
        "--min-bb",
        type=int,
        default=5,
        help="至少多少个基本块才参与抽样（默认 5，略大一点便于分析多 BB 对齐）",
    )
    ap.add_argument(
        "--min-mean-conf",
        type=float,
        default=0.7,
        help="validate_sample 的 mean_conf 下限（默认 0.7，含等于 0.7）",
    )
    ap.add_argument(
        "--no-mean-conf-filter",
        action="store_true",
        help="不按 mean_conf 过滤（仍用 --min-bb）",
    )
    args = ap.parse_args()
    min_mc: float | None = None if args.no_mean_conf_filter else float(args.min_mean_conf)
    root = args.datasets.expanduser().resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 1

    names = iter_jsonl_names()
    rows: list[str] = []
    stat_ok = stat_warn = stat_fail = stat_miss = 0

    for name in names:
        path = root / name
        if not path.is_file():
            rows.append(f"MISS\t{name}\t-\t-\t-\t-\t文件不存在")
            stat_miss += 1
            continue
        n = count_jsonl_lines(path)
        if args.per_file != 3:
            # 简化：仅 per_file==3 用 pick_three；否则均匀切分
            idxs = [
                int((n - 1) * i / max(args.per_file - 1, 1))
                for i in range(args.per_file)
            ] if n > 0 else []
        else:
            idxs = pick_three_line_indices(n)
        for li in idxs:
            d = load_by_line_idx(path, li)
            if not d:
                rows.append(f"FAIL\t{name}\t{li}\t?\t索引越界或空行")
                stat_fail += 1
                continue
            fn = d.get("func_name", "?")
            level, notes = semantic_heuristic(d)
            if level == "OK":
                stat_ok += 1
            elif level == "WARN":
                stat_warn += 1
            else:
                stat_fail += 1
            note_s = "; ".join(notes).replace("\t", " ")
            rows.append(f"{level}\t{name}\t{li}\t{fn}\t{note_s}")

    header = "level\tjsonl\tline_idx\tfunc_name\tnote"
    out_txt = header + "\n" + "\n".join(rows) + "\n"
    print(out_txt)
    total_scored = stat_ok + stat_warn + stat_fail
    print(
        f"# 统计: OK={stat_ok} WARN={stat_warn} FAIL={stat_fail} "
        f"MISS_FILE={stat_miss} SKIP={stat_skip} 已评样本={total_scored}",
        file=sys.stderr,
    )
    mc_desc = "关闭" if min_mc is None else f"mean_conf>={min_mc}"
    print(
        "# 说明: 目标 32 文件 × per-file；候选不足时会少抽。过滤: "
        f"min_bb>={args.min_bb}, {mc_desc}",
        file=sys.stderr,
    )
    if args.output:
        args.output.write_text(out_txt, encoding="utf-8")
        print(f"# 已写 {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

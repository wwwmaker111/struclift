#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出「当前」binskel 数据集的对齐效果汇总表（与 validate_alignment / deep_validate 口径一致）。

默认在常见根目录中查找 binskel_<proj>_o*.jsonl（与 stats_binskel_alignment_effect.py 相同策略）。

用法:
  cd /path/to/structlift && python3 scripts/print_binskel_alignment_table.py

  python3 scripts/print_binskel_alignment_table.py --out-dir E:/structlift_datasets
  python3 scripts/print_binskel_alignment_table.py --format md    # Markdown
  python3 scripts/print_binskel_alignment_table.py --format tsv   # 制表符

  # 不扫多路径，只认一个目录:
  python3 scripts/print_binskel_alignment_table.py --no-all-default-paths --out-dir ~/structlift_datasets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from binskel_alignment_batch_stats import (  # noqa: E402
    DEFAULT_PROJECTS,
    aggregate_jsonl,
    dedupe_paths,
    default_multi_root_search_list,
    locate_jsonl,
)

HEADERS = [
    "数据集",
    "来源目录",
    "样本",
    "BB覆盖%",
    "均信",
    "高置%",
    "低置%",
    "顺序%",
    "CF%",
    "M_gt%",
    "bb_cov均%",
    "stmt%",
    "行档%",
    "M行自洽%",
    "Δ均信",
    "Δ高%",
]


def _baseline(rows: list[dict]) -> dict[str, tuple[float | None, float | None]]:
    out: dict[str, tuple[float | None, float | None]] = {}
    for r in rows:
        if r["opt"] != 0:
            continue
        st = r["stat"]
        if st and st["n"] and st["mean_conf"] is not None:
            out[r["proj"]] = (st["mean_conf"], st["high_pct"])
        else:
            out[r["proj"]] = (None, None)
    return out


def _fmt_delta(
    rows: list[dict],
    baseline: dict[str, tuple[float | None, float | None]],
    proj: str,
    opt: int,
    key: str,
    no_delta: bool,
) -> str:
    if no_delta or opt == 0:
        return "-"
    st = None
    for r in rows:
        if r["proj"] == proj and r["opt"] == opt:
            st = r["stat"]
            break
    b = baseline.get(proj, (None, None))
    if not st or not st["n"] or b[0] is None:
        return "-"
    if key == "mean":
        return f"{st['mean_conf'] - b[0]:+.4f}"
    if key == "high":
        if b[1] is None or st["high_pct"] is None:
            return "-"
        return f"{st['high_pct'] - b[1]:+.2f}"
    return "-"


def _fmt_ext(st: dict | None) -> tuple[str, str, str, str]:
    if not st or st["n"] == 0:
        return "-", "-", "-", "-"
    bb = f"{st['mean_bb_cov_pct']:.2f}" if st.get("mean_bb_cov_pct") is not None else "-"
    sc = f"{st['stmt_tier_pct']:.2f}"
    lf = f"{st['line_tier_pct']:.2f}"
    mc = f"{st['matrix_consistent_pct']:.2f}"
    return bb, sc, lf, mc


def _root_short(root: Path | None) -> str:
    if not root:
        return "-"
    try:
        s = str(root.relative_to(Path.home()))
        return "~/" + s if not s.startswith("~") else s
    except ValueError:
        return str(root)


def _row_cells(
    r: dict,
    rows: list[dict],
    baseline: dict[str, tuple[float | None, float | None]],
    no_delta: bool,
    md_missing: bool,
) -> list[str]:
    name, st = r["name"], r["stat"]
    root_s = _root_short(r["root"])
    bb, sc, lf, mc = _fmt_ext(st) if st else ("-", "-", "-", "-")
    if st is None:
        miss = "**缺失**" if md_missing else "缺失"
        return [name, miss, "-", "-", "-", "-", "-", "-", "-", "-", bb, sc, lf, mc, "-", "-"]
    if st["n"] == 0:
        extra = f" (坏行{st['bad_lines']})" if st["bad_lines"] else ""
        return [
            name,
            root_s,
            f"0{extra}",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            bb,
            sc,
            lf,
            mc,
            "-",
            "-",
        ]
    d_mean = _fmt_delta(rows, baseline, r["proj"], r["opt"], "mean", no_delta)
    d_high = _fmt_delta(rows, baseline, r["proj"], r["opt"], "high", no_delta)
    return [
        name,
        root_s,
        str(st["n"]),
        f"{st['cov_pct']:.2f}",
        f"{st['mean_conf']:.4f}",
        f"{st['high_pct']:.2f}",
        f"{st['low_pct']:.2f}",
        f"{st['order_pct']:.2f}",
        f"{st['cf_pct']:.2f}",
        f"{st['m_gt_pct']:.2f}",
        bb,
        sc,
        lf,
        mc,
        d_mean,
        d_high,
    ]


def _print_ascii(headers: list[str], data: list[list[str]]) -> None:
    rows = [headers] + data
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells: list[str]) -> str:
        parts = []
        for i, c in enumerate(cells):
            parts.append(" " + str(c).ljust(widths[i]) + " ")
        return "|" + "|".join(parts) + "|"

    print(sep)
    print(line(headers))
    print(sep)
    for row in data:
        print(line(row))
    print(sep)


def _print_markdown(headers: list[str], data: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in data:
        print("| " + " | ".join(row) + " |")


def _print_tsv(headers: list[str], data: list[list[str]]) -> None:
    print("\t".join(headers))
    for row in data:
        print("\t".join(row))


def main() -> None:
    ap = argparse.ArgumentParser(description="打印 binskel 对齐效果表（当前磁盘上的 jsonl）")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "structlift_datasets",
        help="jsonl 主目录（默认 ~/structlift_datasets）",
    )
    ap.add_argument(
        "--search-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="额外查找目录，可重复",
    )
    ap.add_argument(
        "--no-all-default-paths",
        action="store_true",
        help="关闭多根目录扫描，仅用 --out-dir 与 --search-dir",
    )
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help=f"逗号分隔项目名（默认 {','.join(DEFAULT_PROJECTS)}）",
    )
    ap.add_argument(
        "--no-delta",
        action="store_true",
        help="不输出相对 O0 的 Δ 列",
    )
    ap.add_argument(
        "--format",
        choices=("ascii", "md", "tsv"),
        default="ascii",
        help="输出格式：终端网格 / Markdown / TSV（默认 ascii）",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    if not args.no_all_default_paths:
        search_dirs = default_multi_root_search_list(out_dir)
        for p in args.search_dir:
            search_dirs.extend([Path(p).expanduser().resolve()])
        search_dirs = dedupe_paths(search_dirs)
    else:
        search_dirs = [out_dir] + [Path(p).expanduser().resolve() for p in args.search_dir]

    projects = [x.strip() for x in args.projects.split(",") if x.strip()]
    rows: list[dict] = []
    for proj in projects:
        for k in range(4):
            name = f"binskel_{proj}_o{k}.jsonl"
            path_used, root_used = locate_jsonl(name, search_dirs)
            stat = aggregate_jsonl(path_used) if path_used else None
            rows.append(
                {
                    "name": name,
                    "proj": proj,
                    "opt": k,
                    "stat": stat,
                    "path": path_used,
                    "root": root_used,
                }
            )

    baseline = _baseline(rows)
    hdr = HEADERS if not args.no_delta else [h for h in HEADERS if h not in ("Δ均信", "Δ高%")]
    data: list[list[str]] = []
    for r in rows:
        full = _row_cells(r, rows, baseline, args.no_delta, args.format == "md")
        if args.no_delta:
            full = full[:-2]
        data.append(full)

    if args.format == "tsv":
        _print_tsv(hdr, data)
    elif args.format == "md":
        _print_markdown(hdr, data)
    else:
        if not args.no_all_default_paths or args.search_dir:
            print("搜索路径（按顺序）:")
            for i, d in enumerate(search_dirs):
                print(f"  [{i}] {d}")
            print()
        _print_ascii(hdr, data)

    n_proj = len(projects)
    ok = sum(1 for r in rows if r["stat"] and r["stat"]["n"] > 0)
    total_lines = sum(r["stat"]["n"] for r in rows if r["stat"] and r["stat"]["n"] > 0)
    print()
    print(f"共 {n_proj} 项目 × 4 档 = {n_proj * 4} 项，有效文件 {ok} 个；样本行合计 {total_lines}。")
    if not args.no_delta:
        print("Δ 列：相对同项目 O0。")
    print(
        "bb_cov均% / stmt% / 行档% / M行自洽%：见 scripts/binskel_alignment_batch_stats.py 顶部说明。"
    )
    inc = sum((r["stat"]["n_inconsistent"] or 0) for r in rows if r["stat"] and r["stat"]["n"])
    if inc:
        print(f"不一致样本条数（可加总）: {inc}")


if __name__ == "__main__":
    main()

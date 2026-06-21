#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对多项目 × 4 优化档 binskel jsonl 批量统计对齐指标（与 validate_alignment.py 口径一致）。

额外汇总（每条 jsonl 内逐样本平均/占比）：
  - bb_cov均%: 字段 bb_alignment_coverage 的样本均值×100（无该字段则为 -）
  - stmt档%/行档%: src_encoding_tier 占比（重生后行档应≈0）
  - 自洽%: M_gt↔block_src_lines 一致样本占比（deep_validate 与 audit 同源）

默认 9 项目 × 4 档（含 openssh、coreutils）；其中 ffmpeg 常与另外项目不在同一目录，请配合 --search-dir。
FFmpeg 常在单独目录（如 ffmpeg_binskel），用 --search-dir 追加查找路径。

用法:
  python3 scripts/binskel_alignment_batch_stats.py
  python3 scripts/binskel_alignment_batch_stats.py --out-dir ~/structlift_datasets
  python3 scripts/binskel_alignment_batch_stats.py \\
      --out-dir ~/structlift_datasets \\
      --search-dir ~/struclift_wsl/datasets/ffmpeg_binskel

  # 一键：常见根目录全扫（含 ffmpeg 子目录 + /mnt/e 备份路径）
  python3 scripts/binskel_alignment_batch_stats.py --all-default-paths
  python3 scripts/report_all_binskel_alignment.py
  python3 scripts/binskel_alignment_batch_stats.py --tsv ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_alignment import validate_sample  # noqa: E402
from deep_validate import analyze_sample as deep_analyze_sample  # noqa: E402

# 默认 9 项目 × 4 档；ffmpeg 常与解压路径不在同一目录，可配合 --search-dir。
DEFAULT_PROJECTS = [
    "zlib",
    "curl",
    "sqlite",
    "busybox",
    "openssl",
    "openssh",
    "libxml2",
    "ffmpeg",
    "coreutils",
]


def aggregate_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    n = 0
    sum_cov = sum_mean = sum_high = sum_low = sum_order = sum_cf = 0.0
    has_matrix = 0
    bad = 0
    n_stmt_tier = n_line_tier = n_tier_unknown = 0
    sum_bb_cov = 0.0
    n_bb_cov = 0
    n_matrix_inconsistent = 0
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
            tier = d.get("src_encoding_tier")
            if tier == "stmt":
                n_stmt_tier += 1
            elif tier == "line_fallback":
                n_line_tier += 1
            else:
                n_tier_unknown += 1
            bc = d.get("bb_alignment_coverage")
            if isinstance(bc, (int, float)):
                sum_bb_cov += float(bc)
                n_bb_cov += 1
            if not deep_analyze_sample(d)["matrix_lines_consistent"]:
                n_matrix_inconsistent += 1
            n += 1
    if n == 0:
        return {
            "n": 0,
            "bad_lines": bad,
            "cov_pct": None,
            "mean_conf": None,
            "high_pct": None,
            "low_pct": None,
            "order_pct": None,
            "cf_pct": None,
            "m_gt_pct": None,
            "mean_bb_cov_pct": None,
            "stmt_tier_pct": None,
            "line_tier_pct": None,
            "tier_unknown_pct": None,
            "matrix_consistent_pct": None,
            "n_inconsistent": 0,
        }
    return {
        "n": n,
        "bad_lines": bad,
        "cov_pct": 100.0 * sum_cov / n,
        "mean_conf": sum_mean / n,
        "high_pct": 100.0 * sum_high / n,
        "low_pct": 100.0 * sum_low / n,
        "order_pct": 100.0 * sum_order / n,
        "cf_pct": 100.0 * sum_cf / n,
        "m_gt_pct": 100.0 * has_matrix / n,
        "mean_bb_cov_pct": (100.0 * sum_bb_cov / n_bb_cov) if n_bb_cov else None,
        "stmt_tier_pct": 100.0 * n_stmt_tier / n,
        "line_tier_pct": 100.0 * n_line_tier / n,
        "tier_unknown_pct": 100.0 * n_tier_unknown / n,
        "matrix_consistent_pct": 100.0 * (n - n_matrix_inconsistent) / n,
        "n_inconsistent": n_matrix_inconsistent,
    }


def locate_jsonl(filename: str, dirs: list[Path]) -> tuple[Path | None, Path | None]:
    """返回 (找到的路径, 所在搜索根目录)。"""
    for d in dirs:
        p = d.expanduser().resolve() / filename
        if p.is_file():
            return p, d.expanduser().resolve()
    return None, None


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        r = p.expanduser().resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def default_multi_root_search_list(out_dir: Path) -> list[Path]:
    """用户本机常见布局：主目录 + struclift_wsl + ffmpeg 子目录 + /mnt/e 备份。"""
    home = Path.home()
    e_drive_datasets = Path("E:/structlift_datasets")
    e_wsl_datasets = Path("/mnt/e/structlift_datasets")
    mangled_win_home_datasets = REPO_ROOT / "C\uf03aUsers15402" / "structlift_datasets"
    raw = [
        out_dir,
        e_drive_datasets,
        e_wsl_datasets,
        home / "structlift_datasets",
        home / "struclift_wsl" / "datasets",
        home / "struclift_wsl" / "datasets" / "ffmpeg_binskel",
        Path("/mnt/e/structlift_src_datasets/ffmpeg_binskel"),
        Path("/mnt/e/structlift_src_datasets"),
        Path("/mnt/e/structlift_datasets"),
        mangled_win_home_datasets,
        REPO_ROOT,
    ]
    return dedupe_paths(raw)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="binskel 数据集对齐效果批量统计（默认含 ffmpeg，可多目录查找）"
    )
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
        help="额外目录：同样查找 binskel_<proj>_o*.jsonl，先匹配 --out-dir 再依次匹配此处。可重复",
    )
    ap.add_argument(
        "--all-default-paths",
        action="store_true",
        help="在 --out-dir 之外自动追加常用根目录（~/structlift_datasets、~/struclift_wsl/datasets、"
        "…/ffmpeg_binskel、/mnt/e 下常见路径），合并去重后按顺序查找",
    )
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS),
        help=f"逗号分隔项目名（默认: {','.join(DEFAULT_PROJECTS)}）",
    )
    ap.add_argument(
        "--no-delta",
        action="store_true",
        help="不打印相对 O0 的 Δ 列（均信/高置信%%）",
    )
    ap.add_argument("--tsv", action="store_true", help="输出制表符分隔，便于粘贴 Excel")
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

    # 各项目 O0 基线（用于 Δ）
    baseline: dict[str, tuple[float | None, float | None]] = {}
    for r in rows:
        if r["opt"] != 0:
            continue
        st = r["stat"]
        if st and st["n"] and st["mean_conf"] is not None:
            baseline[r["proj"]] = (st["mean_conf"], st["high_pct"])
        else:
            baseline[r["proj"]] = (None, None)

    def fmt_delta(proj: str, opt: int, key: str) -> str:
        if args.no_delta or opt == 0:
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
            d = st["mean_conf"] - b[0]
            return f"{d:+.4f}"
        if key == "high":
            if b[1] is None or st["high_pct"] is None:
                return "-"
            d = st["high_pct"] - b[1]
            return f"{d:+.2f}"
        return "-"

    def fmt_ext(st: dict) -> tuple[str, str, str, str]:
        """bb_cov均, stmt%, line%, 自洽%"""
        if not st or st["n"] == 0:
            return "-", "-", "-", "-"
        bb = f"{st['mean_bb_cov_pct']:.2f}" if st.get("mean_bb_cov_pct") is not None else "-"
        sc = f"{st['stmt_tier_pct']:.2f}"
        lf = f"{st['line_tier_pct']:.2f}"
        mc = f"{st['matrix_consistent_pct']:.2f}"
        return bb, sc, lf, mc

    if args.tsv:
        print(
            "数据集\t目录\t样本\tBB覆盖率%\t平均置信\t高置信%\t低置信%\t顺序%\t控制流%\tM_gt%\t"
            "bb_cov均%\tstmt档%\t行档%\tM↔行自洽%\tΔ均信(vsO0)\tΔ高%%(vsO0)"
        )
        for r in rows:
            name, st = r["name"], r["stat"]
            root_s = str(r["root"]) if r["root"] else ""
            bb, sc, lf, mc = fmt_ext(st) if st else ("-", "-", "-", "-")
            if st is None or st["n"] == 0:
                print(
                    f"{name}\t{root_s}\t-\t-\t-\t-\t-\t-\t-\t-\t{bb}\t{sc}\t{lf}\t{mc}\t-\t-"
                )
                continue
            print(
                f"{name}\t{root_s}\t{st['n']}\t{st['cov_pct']:.2f}\t{st['mean_conf']:.4f}\t"
                f"{st['high_pct']:.2f}\t{st['low_pct']:.2f}\t{st['order_pct']:.2f}\t"
                f"{st['cf_pct']:.2f}\t{st['m_gt_pct']:.2f}\t{bb}\t{sc}\t{lf}\t{mc}\t"
                f"{fmt_delta(r['proj'], r['opt'], 'mean')}\t{fmt_delta(r['proj'], r['opt'], 'high')}"
            )
        return

    print("搜索路径（按顺序）:")
    for i, d in enumerate(search_dirs):
        print(f"  [{i}] {d}")
    print()

    print(
        "| 数据集 | 来源目录 | 样本 | BB覆盖率% | 平均置信 | 高置信% | 低置信% | 顺序% | 控制流% | M_gt% | "
        "bb_cov均% | stmt档% | 行档% | M↔行自洽% | Δ均信 | Δ高% |"
    )
    print(
        "|--------|----------|------|-----------|----------|---------|---------|-------|---------|-------|"
        "-----------|---------|-------|------------|-------|------|"
    )
    for r in rows:
        name, st = r["name"], r["stat"]
        root_short = "-"
        if r["root"]:
            try:
                root_short = str(r["root"].relative_to(Path.home()))
                if not root_short.startswith("~"):
                    root_short = "~/" + root_short
            except ValueError:
                root_short = str(r["root"])
        bb, sc, lf, mc = fmt_ext(st) if st else ("-", "-", "-", "-")
        if st is None:
            print(
                f"| {name} | **缺失** | - | - | - | - | - | - | - | - | {bb} | {sc} | {lf} | {mc} | - | - |"
            )
            continue
        if st["n"] == 0:
            extra = f" (无效行 {st['bad_lines']})" if st["bad_lines"] else ""
            print(
                f"| {name} | {root_short} | 0{extra} | - | - | - | - | - | - | - | "
                f"{bb} | {sc} | {lf} | {mc} | - | - |"
            )
            continue
        d_mean = fmt_delta(r["proj"], r["opt"], "mean")
        d_high = fmt_delta(r["proj"], r["opt"], "high")
        cell_root = f"`{root_short}`" if root_short != "-" else "-"
        print(
            f"| {name} | {cell_root} | {st['n']} | {st['cov_pct']:.2f} | {st['mean_conf']:.4f} | "
            f"{st['high_pct']:.2f} | {st['low_pct']:.2f} | {st['order_pct']:.2f} | "
            f"{st['cf_pct']:.2f} | {st['m_gt_pct']:.2f} | {bb} | {sc} | {lf} | {mc} | {d_mean} | {d_high} |"
        )

    ok = sum(1 for r in rows if r["stat"] and r["stat"]["n"] > 0)
    n_proj = len(projects)
    total_lines = sum(r["stat"]["n"] for r in rows if r["stat"] and r["stat"]["n"] > 0)
    print()
    print(f"共 {n_proj} 项目 × 4 档 = {n_proj * 4} 项，有效文件 {ok} 个。")
    print(f"有效 jsonl 内样本行合计（各档相加）: {total_lines}")
    if not args.no_delta:
        print(
            "Δ 列：相对同项目 O0 的差（便于看 FFmpeg 等细微变化；四舍五入少位数时原表易「看起来一样」）。"
        )
    print(
        "bb_cov均% / stmt档% / 行档% / M↔行自洽%：见脚本顶部说明；旧 jsonl 无 tier 字段时「行档+未知档」可能合并到未知。"
    )
    inc = sum((r["stat"]["n_inconsistent"] or 0) for r in rows if r["stat"] and r["stat"]["n"])
    if inc:
        print(f"全表合计不一致样本条数（逐文件可加总）: {inc}（若需按文件看各 st['n_inconsistent']）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从默认项目 × 4 优化档每文件抽取若干条「高置信」样本，
打印成一整段文本，便于复制给协作者做语义对齐人工检查。

每文件策略（默认）：
  1) 先扫全文件，得到每条 validate_sample 的 mean_conf、coverage；
  2) 按 mean_conf 降序，在满足 --min-mean-conf / --min-coverage / --min-n-stmts 的样本里取前 --per-file 条。

用法（WSL，与 print_binskel_alignment_table 相同搜索路径）:
  python3 scripts/export_highconf_audit_pack.py --all-default-paths --per-file 1 > /tmp/highconf_audit.txt

  # 每文件 2 条、语句数至少 5（减少「多 BB 对一句」退化样本）
  python3 scripts/export_highconf_audit_pack.py --all-default-paths --per-file 2 --min-n-stmts 5

  # 输出到文件
  python3 scripts/export_highconf_audit_pack.py --all-default-paths --per-file 1 -o ~/highconf_audit.txt
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
    dedupe_paths,
    default_multi_root_search_list,
    locate_jsonl,
)
from deep_validate import analyze_sample, format_spot_check  # noqa: E402
from sample_alignment_inspect import _load_at_indices, _scan_scores  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402


def _pick_highconf_indices(
    path: Path,
    per_file: int,
    min_mean: float,
    min_cov: float,
    min_n_stmt: int,
    max_scan_pool: int,
    progress_label: str = "",
    progress_every: int = 5000,
) -> list[tuple[int, float]]:
    """
    返回 [(line_idx, mean_conf), ...] 至多 per_file 条。
    max_scan_pool: 从按 conf 排序后的前多少个候选里尝试加载并过滤（避免超大函数全排序进内存）。
    """
    scores = _scan_scores(
        path,
        progress_label=(progress_label or None) if progress_every > 0 else None,
        progress_every=progress_every,
    )
    if not scores:
        return []
    ranked = sorted(scores, key=lambda x: (-x[1], -x[2], x[0]))
    want_try: list[int] = []
    for idx, mc, cov, _fn in ranked:
        if mc < min_mean or cov < min_cov:
            continue
        want_try.append(idx)
        if len(want_try) >= max_scan_pool:
            break
    if not want_try:
        return []

    loaded = _load_at_indices(path, set(want_try))
    picked: list[tuple[int, float]] = []
    for idx in want_try:
        if len(picked) >= per_file:
            break
        d = loaded.get(idx)
        if not d:
            continue
        ar = analyze_sample(d)
        if ar["n_stmts"] < min_n_stmt:
            continue
        vr = validate_sample(d)
        mc = float(vr.get("mean_conf", 0))
        if mc < min_mean or float(vr.get("coverage", 0)) < min_cov:
            continue
        picked.append((idx, mc))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="28 jsonl 高置信样本导出（供人工语义检查）")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS))
    ap.add_argument("--per-file", type=int, default=1, help="每个 jsonl 抽取条数（默认 1，共最多 28 条）")
    ap.add_argument("--min-mean-conf", type=float, default=0.72, metavar="T")
    ap.add_argument("--min-coverage", type=float, default=0.5, metavar="C")
    ap.add_argument("--min-n-stmts", type=int, default=0, metavar="N", help="analyze_sample 的语句数下限（0 表示不限制）")
    ap.add_argument(
        "--pool",
        type=int,
        default=400,
        help="每文件从高到低最多考察多少条候选（加载 JSON 过滤，大文件可调大）",
    )
    ap.add_argument("--max-bb", type=int, default=16, help="每条样本打印前多少个 BB")
    ap.add_argument("-o", "--output", type=Path, default=None, help="写入文件；默认 stdout")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="不向 stderr 打印扫描进度",
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

    lines_out: list[str] = []
    summary_rows: list[str] = []

    def emit(s: str = "") -> None:
        lines_out.append(s)

    if not args.output:
        print(
            "说明: 正文只在全部 28 个文件处理完后一次性输出到 stdout；"
            "大库单文件扫描可达数分钟，进度见下方 stderr。",
            file=sys.stderr,
            flush=True,
        )

    emit("=" * 72)
    emit("高置信样本导出包（供人工检查 BB<->语句 语义是否合理）")
    emit("筛选: min_mean_conf>=%.3f min_coverage>=%.3f min_n_stmts>=%d per_file=%d" % (
        args.min_mean_conf, args.min_coverage, args.min_n_stmts, args.per_file,
    ))
    emit("搜索路径: %s" % search_dirs)
    emit("=" * 72)
    emit()

    sample_no = 0
    for proj in projects:
        for opt in range(4):
            name = f"binskel_{proj}_o{opt}.jsonl"
            path, root = locate_jsonl(name, search_dirs)
            if not path:
                emit(f"### [缺失] {name}")
                emit()
                continue

            if not args.no_progress:
                print(f"[进度] 开始扫描 {name} ...", file=sys.stderr, flush=True)

            pe = 0 if args.no_progress else 5000
            picked = _pick_highconf_indices(
                path,
                args.per_file,
                args.min_mean_conf,
                args.min_coverage,
                args.min_n_stmts,
                max_scan_pool=max(args.pool, args.per_file * 50),
                progress_label=name,
                progress_every=pe,
            )

            if not args.no_progress:
                print(
                    f"[进度] 完成 {name}，选出 {len(picked)} 条候选",
                    file=sys.stderr,
                    flush=True,
                )
            if not picked:
                emit(f"### [无满足条件的样本] {name} @ {root}")
                emit(f"    (可提高 --pool 或放宽 --min-mean-conf / --min-coverage / --min-n-stmts)")
                emit()
                continue

            want = {i for i, _ in picked}
            data = _load_at_indices(path, want)
            for line_idx, mc in picked:
                d = data.get(line_idx)
                sample_no += 1
                if not d:
                    emit(f"### SAMPLE {sample_no} | {name} | line_idx={line_idx} | (未读到)")
                    emit()
                    continue
                ar = analyze_sample(d)
                vr = validate_sample(d)
                fn = d.get("func_name", "?")
                sf = d.get("source_file", "?")
                emit(f"### SAMPLE {sample_no} | {name} | line_idx={line_idx} | mean_conf={mc:.4f}")
                emit(f"jsonl_root: {root}")
                emit(f"func_name: {fn}")
                emit(f"source_file: {sf}")
                emit(
                    f"metrics: n_bb={ar['n_blocks']} n_stmt={ar['n_stmts']} "
                    f"cov={vr['coverage']:.4f} mean_conf={vr['mean_conf']:.4f} "
                    f"order={vr['order_consistency']:.4f} cf={vr['cf_consistency']:.4f}"
                )
                emit(
                    f"matrix_vs_block_lines: "
                    f"{'PASS' if ar['matrix_lines_consistent'] else 'FAIL'}"
                )
                emit()
                emit(format_spot_check(d, ar, max_bb=args.max_bb))
                emit()
                emit("-" * 72)
                emit()
                summary_rows.append(
                    f"{name}\t{line_idx}\t{mc:.4f}\t{ar['n_blocks']}\t{ar['n_stmts']}\t"
                    f"{vr['coverage']:.4f}\t{fn}"
                )

    emit("### 汇总表（制表符分隔，可贴表格）")
    emit("jsonl\tline_idx\tmean_conf\tn_bb\tn_stmt\tcoverage\tfunc_name")
    for row in summary_rows:
        emit(row)
    emit()
    emit(f"共导出 {sample_no} 条（含未读到的占位）。")

    text = "\n".join(lines_out) + "\n"
    if args.output:
        args.output.expanduser().resolve().write_text(text, encoding="utf-8")
        print(f"已写入: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

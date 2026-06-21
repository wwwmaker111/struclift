#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按「项目」（zlib/curl/… 共 7 个）各抽 N 条样本，优先多基本块、便于人工/LLM 检查 BB↔语句语义。

默认每个项目只扫 binskel_<proj>_o0.jsonl（调试信息最好、速度可接受）；
过滤：至少 2 个 BB、至少 min_n_stmts 条语句；最终每项目至多 max_single_bb 条「1 个 BB」的兜底样本。

输出含：jsonl 名、行下标、func_name、source_file、validate 指标、自洽、
      source_lines 列下标->行号（便于对照 L?）、format_spot_check 前若干 BB。

用法（复制整段到 WSL，在 structlift 根目录执行）:

  cd /mnt/e/structlift
  PYTHONUNBUFFERED=1 python3 -u scripts/export_per_project_audit_pack.py \\
      --all-default-paths --per-project 5 \\
      --min-n-bb 2 --min-n-stmts 4 \\
      --min-mean-conf 0.55 --min-coverage 0.55 \\
      --pool 6000 --max-bb 28 \\
      2> /tmp/audit_progress.log

正文在 stdout，进度在 stderr；检查时可把 stdout 全文复制给审查方。
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


def _emit_source_lines_legend(d: dict, max_show: int = 32) -> list[str]:
    """列下标 j -> source_lines[j] 物理行号，供解读 L?。"""
    lines_out: list[str] = []
    sl = d.get("source_lines") or []
    if not isinstance(sl, list) or not sl:
        lines_out.append("source_lines: (无或空，→stmt_idx 旁 L? 需自行打开源文件对照)")
        return lines_out
    n = min(len(sl), max_show)
    lines_out.append(f"source_lines[0..{n-1}] 对应矩阵列 0..{n-1} 的源码行号:")
    for j in range(n):
        lines_out.append(f"  col {j:3d} -> line {sl[j]}")
    if len(sl) > max_show:
        lines_out.append(f"  ... 共 {len(sl)} 列，仅列前 {max_show}")
    return lines_out


def _pick_for_project(
    path: Path,
    name: str,
    per_project: int,
    min_mean: float,
    min_cov: float,
    min_n_bb: int,
    min_n_stmt: int,
    pool: int,
    max_single_bb: int,
    progress_every: int,
) -> list[tuple[int, float, dict, dict, dict]]:
    """
    返回 [(line_idx, mean_conf, d, ar, vr), ...] 至多 per_project 条。
    """
    scores = _scan_scores(
        path,
        progress_label=name,
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
        if len(want_try) >= pool:
            break

    loaded = _load_at_indices(path, set(want_try)) if want_try else {}

    multi_bb: list[tuple[int, float, dict, dict, dict, int]] = []
    single_bb: list[tuple[int, float, dict, dict, dict, int]] = []

    for idx in want_try:
        d = loaded.get(idx)
        if not d:
            continue
        vr = validate_sample(d)
        mc = float(vr.get("mean_conf", 0))
        cov = float(vr.get("coverage", 0))
        if mc < min_mean or cov < min_cov:
            continue
        ar = analyze_sample(d)
        nb = ar["n_blocks"]
        ns = ar["n_stmts"]
        if ns < min_n_stmt:
            continue
        richness = int(nb * max(ns, 1))
        tup = (idx, mc, d, ar, vr, richness)
        if nb >= min_n_bb:
            multi_bb.append(tup)
        elif nb == 1:
            single_bb.append(tup)

    multi_bb.sort(key=lambda t: (-t[1], -t[5], -t[0]))
    single_bb.sort(key=lambda t: (-t[1], -t[5], -t[0]))

    out: list[tuple[int, float, dict, dict, dict]] = []
    for t in multi_bb:
        if len(out) >= per_project:
            break
        out.append((t[0], t[1], t[2], t[3], t[4]))

    n_single = sum(1 for x in out if x[3]["n_blocks"] == 1)
    for t in single_bb:
        if len(out) >= per_project:
            break
        if n_single >= max_single_bb:
            break
        if t[0] in {x[0] for x in out}:
            continue
        out.append((t[0], t[1], t[2], t[3], t[4]))
        n_single += 1

    return out[:per_project]


def main() -> None:
    ap = argparse.ArgumentParser(description="每项目 N 条审计样本（优先多 BB）")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS))
    ap.add_argument("--per-project", type=int, default=5)
    ap.add_argument("--opt", type=int, default=0, help="只用该优化档 jsonl（默认 0）")
    ap.add_argument("--min-mean-conf", type=float, default=0.55)
    ap.add_argument("--min-coverage", type=float, default=0.55)
    ap.add_argument("--min-n-bb", type=int, default=2, help="优先样本最少 BB 数")
    ap.add_argument("--min-n-stmts", type=int, default=4)
    ap.add_argument("--pool", type=int, default=6000)
    ap.add_argument(
        "--max-single-bb",
        type=int,
        default=1,
        help="每项目最终结果里「1 个 BB」的样本最多几条（其余用多 BB 填满）",
    )
    ap.add_argument("--max-bb", type=int, default=28, help="每条打印前多少个 BB 的对齐表")
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("-o", "--output", type=Path, default=None)
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
    pe = 0 if args.no_progress else args.progress_every

    lines_out: list[str] = []
    summary: list[str] = []

    def emit(s: str = "") -> None:
        lines_out.append(s)

    if not args.output:
        print(
            "说明: 完整报告在 stdout（处理完 7 个项目后一次性输出）；进度在 stderr。\n"
            "复制时请选中终端里**正文**（或重定向 stdout 到文件再用编辑器打开）。",
            file=sys.stderr,
            flush=True,
        )

    emit("=" * 72)
    emit("按项目导出审计包（供检查 BB↔语句 语义）")
    emit(
        f"规则: 每项目 {args.per_project} 条 | opt=o{args.opt} | "
        f"优先 n_bb>={args.min_n_bb}, n_stmt>={args.min_n_stmts} | "
        f"每项目至多 {args.max_single_bb} 条「仅 1 BB」兜底"
    )
    emit(
        f"阈值: mean_conf>={args.min_mean_conf} cov>={args.min_coverage} "
        f"pool={args.pool}"
    )
    emit(f"搜索路径: {search_dirs}")
    emit("=" * 72)
    emit()

    global_no = 0
    for proj in projects:
        name = f"binskel_{proj}_o{args.opt}.jsonl"
        path, root = locate_jsonl(name, search_dirs)
        if not path:
            emit(f"### [缺失] 项目 {proj}: {name}")
            emit()
            summary.append(f"{proj}\tMISSING\t{name}")
            continue

        if pe:
            print(f"[进度] 项目 {proj}: 扫描 {name} ...", file=sys.stderr, flush=True)

        picked = _pick_for_project(
            path,
            name,
            args.per_project,
            args.min_mean_conf,
            args.min_coverage,
            args.min_n_bb,
            args.min_n_stmts,
            args.pool,
            args.max_single_bb,
            pe,
        )

        if pe:
            print(f"[进度] 项目 {proj}: 得到 {len(picked)} 条", file=sys.stderr, flush=True)

        if not picked:
            emit(f"### [无满足条件样本] 项目 {proj} | {name} @ {root}")
            emit("    可放宽 --min-n-bb / --min-n-stmts / --min-mean-conf 或增大 --pool")
            emit()
            summary.append(f"{proj}\t0\t{name}")
            continue

        emit(f"## 项目 {proj} （{name} @ {root}） 共 {len(picked)} 条")
        emit()

        for line_idx, mc, d, ar, vr in picked:
            global_no += 1
            fn = d.get("func_name", "?")
            sf = d.get("source_file", "?")
            emit(f"### SAMPLE {global_no} | {name} | line_idx={line_idx}")
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
            emit(f"src_encoding_tier: {d.get('src_encoding_tier', '?')}")
            emit()
            for x in _emit_source_lines_legend(d):
                emit(x)
            emit()
            emit(format_spot_check(d, ar, max_bb=args.max_bb))
            emit()
            emit("-" * 72)
            emit()
            summary.append(
                f"{proj}\t{line_idx}\t{mc:.4f}\t{ar['n_blocks']}\t{ar['n_stmts']}\t{fn}"
            )

    emit("### 汇总（项目\tline_idx\tmean_conf\tn_bb\tn_stmt\tfunc_name）")
    for row in summary:
        emit(row)
    emit()
    emit(f"共 {global_no} 条样本。")

    text = "\n".join(lines_out) + "\n"
    if args.output:
        args.output.expanduser().resolve().write_text(text, encoding="utf-8")
        print(f"已写入: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()

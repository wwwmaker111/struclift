#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从最新 binskel 数据集中：9 项目 × 仅 O0 × 每项目 2 条 = 18 个样本，
过滤「基本块过少」的函数，写出单 txt，并做与 monitor_bb_alignment_dual 同口径的
结构 + 语义对齐监测摘要。

设计要点：
  - 同函数只导出一份（不按 O1/O2 重复抽）。
  - 紧凑导出：默认省略「全矩阵 j↔语句」表、限制 j* 源码行数、默认跳过反汇编（可加 --with-disasm）。
  - 每函数仅均匀抽样若干 BB 写入核对包，避免 sqlite 等巨型 span 刷屏。

用法（仓库根）:
  python3 scripts/export_9proj_o0_18_samples_dual_audit_txt.py \\
      --all-default-paths --out-dir ~/structlift_datasets \\
      --bin-root ~/structlift_datasets -o samples_9proj_o0_18_dual.txt

  python3 scripts/export_9proj_o0_18_samples_dual_audit_txt.py \\
      --datasets E:/structlift_datasets --bin-root E:/structlift_datasets \\
      -o samples_9proj_o0_18_dual.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_four_tier_alignment_45x3 import (  # noqa: E402
    binskel_jsonl_name,
    iter_filtered_line_indices,
    load_by_line_idx,
    pick_spread_line_indices,
)
from export_bb_manual_audit_pack import build_pack  # noqa: E402
from monitor_bb_alignment_dual import process_one  # noqa: E402


def _default_structlift_datasets() -> Path:
    """按优先级尝试：E:\\（Win）、/mnt/e/（WSL）、~/structlift_datasets。"""
    for cand in [
        Path("E:/structlift_datasets"),
        Path("/mnt/e/structlift_datasets"),
    ]:
        if cand.is_dir():
            return cand.resolve()
    return (Path.home() / "structlift_datasets").expanduser().resolve()


try:
    from binskel_alignment_batch_stats import (  # noqa: E402
        DEFAULT_PROJECTS,
        dedupe_paths,
        default_multi_root_search_list,
        locate_jsonl,
    )
except ImportError:
    DEFAULT_PROJECTS = ["zlib"]
    dedupe_paths = None  # type: ignore
    default_multi_root_search_list = None  # type: ignore
    locate_jsonl = None  # type: ignore

try:
    from check_alignment_semantics import resolve_binary_path  # noqa: E402
except ImportError:
    resolve_binary_path = None  # type: ignore


def count_basic_blocks(d: dict) -> int:
    bl = d.get("block_src_lines")
    if isinstance(bl, list) and bl:
        return len(bl)
    M = d.get("alignment_matrix")
    if isinstance(M, list) and M:
        return len(M)
    return 0


def dataset_banner(
    proj: str,
    jp: Optional[Path],
    line_idx: int,
    func_name: str,
    n_bb: int,
    note: str = "",
) -> str:
    path_s = str(jp) if jp else ""
    extra = f"\n# NOTE: {note}" if note else ""
    name = binskel_jsonl_name(proj, "0")
    return (
        "\n"
        + "=" * 88
        + "\n"
        + f"# PROJECT: {proj}  (O0 only)\n"
        + f"# jsonl: {name}\n"
        + f"# path: {path_s}\n"
        + f"# line_idx(0-based): {line_idx}\n"
        + f"# func_name: {func_name}\n"
        + f"# n_bb: {n_bb}\n"
        + "=" * 88
        + extra
        + "\n"
    )


def format_dual_block(po: Dict[str, Any]) -> str:
    lines = [
        "--- 结构 + 语义监测（与 monitor_bb_alignment_dual.process_one 同源）---",
        f"  structural: {po.get('structural')}  |  {po.get('structural_notes', '')}",
        f"  semantic:   {po.get('semantic')}  |  {po.get('semantic_notes', '')}",
        f"  viol_peak={po.get('viol_peak')}  viol_highw={po.get('viol_highw')}  "
        f"n_bb_checked_sem={po.get('n_bb_checked_sem')}",
        f"  matrix_lines_consistent={po.get('matrix_lines_consistent')}  "
        f"coverage={po.get('coverage')}  order_consistency={po.get('order_consistency')}",
    ]
    if po.get("semantic_skip_reason"):
        lines.append(f"  semantic_skip: {po.get('semantic_skip_reason')}")
    if po.get("sem_asm_grounded"):
        lines.append(
            f"  asm_grounded: asm_cfg_ok={po.get('asm_cfg_ok')}  "
            f"n_viol_dwarf_json={po.get('n_viol_dwarf_json')}  "
            f"n_viol_no_pc_line={po.get('n_viol_no_pc_line')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="9 项目 O0 各 2 条样本 → 单 txt + 双通道对齐监测摘要"
    )
    ap.add_argument("--datasets", type=Path, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="jsonl 主目录；默认若存在则用 E:/structlift_datasets，否则 ~/structlift_datasets",
    )
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS) if DEFAULT_PROJECTS else "zlib",
    )
    ap.add_argument("--per-project", type=int, default=2, help="每项目抽取条数（默认 2）")
    ap.add_argument("--min-bb", type=int, default=8, help="最少基本块数（默认 8）")
    ap.add_argument("--min-mean-conf", type=float, default=0.65)
    ap.add_argument("--no-mean-conf-filter", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bin-root", type=Path, default=None)
    ap.add_argument("--src-map", action="append", default=[], metavar="PROJ=PATH")
    ap.add_argument("--high-w", type=float, default=0.85)
    ap.add_argument(
        "--sem-asm-grounded",
        action="store_true",
        help="语义检查使用 ELF+PC 重放（较慢，需 bin_root 等）",
    )
    ap.add_argument(
        "--with-disasm",
        action="store_true",
        help="核对包内包含 Capstone 反汇编（默认跳过以加速）",
    )
    ap.add_argument(
        "--sample-bb",
        type=int,
        default=4,
        help="每函数写入核对包时均匀抽样的 BB 数（默认 4）",
    )
    ap.add_argument(
        "--max-src-lines-jstar",
        type=int,
        default=36,
        help="每个 BB 的 j* 源码片段最多输出行数（默认 36）",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="合并输出的 txt",
    )
    args = ap.parse_args()

    src_map: Dict[str, Path] = {}
    for p in args.src_map:
        if "=" in p:
            k, v = p.split("=", 1)
            src_map[k.strip()] = Path(v).expanduser().resolve()

    preset = {
        "zlib": "/mnt/e/structlift/zlib_build",
        "curl": "/mnt/e/structlift_src_datasets/curl/curl-8.7.1",
        "sqlite": "/mnt/e/structlift_src_datasets/sqlite/sqlite-autoconf-3520000",
        "busybox": "/mnt/e/structlift_src_datasets/busybox/busybox-1.36.1",
        "openssl": "/mnt/e/structlift_src_datasets/openssl/openssl-3.3.1",
        "openssh": "/mnt/e/structlift_src_datasets/openssh/openssh-9.8p1",
        "libxml2": "/mnt/e/structlift_src_datasets/libxml2/libxml2-2.12.7",
        "ffmpeg": "/mnt/e/structlift_src_datasets/ffmpeg/ffmpeg-7.1",
        "coreutils": "/mnt/e/structlift_src_datasets/coreutils/coreutils-9.4",
    }
    win_alt = Path("E:/structlift_src_datasets")
    if win_alt.is_dir():
        preset["curl"] = str(win_alt / "curl/curl-8.7.1")
        preset["sqlite"] = str(win_alt / "sqlite/sqlite-autoconf-3520000")
        preset["busybox"] = str(win_alt / "busybox/busybox-1.36.1")
        preset["openssl"] = str(win_alt / "openssl/openssl-3.3.1")
        preset["openssh"] = str(win_alt / "openssh/openssh-9.8p1")
        preset["libxml2"] = str(win_alt / "libxml2/libxml2-2.12.7")
        preset["ffmpeg"] = str(win_alt / "ffmpeg/ffmpeg-7.1")
        preset["coreutils"] = str(win_alt / "coreutils/coreutils-9.4")
    zb = Path("E:/structlift/zlib_build")
    if zb.is_dir():
        preset["zlib"] = str(zb)

    for k, v in preset.items():
        pp = Path(v)
        if pp.is_dir():
            src_map.setdefault(k, pp.resolve())

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else _default_structlift_datasets()
    )
    if args.all_default_paths and default_multi_root_search_list and locate_jsonl:
        search_dirs = default_multi_root_search_list(out_dir)
        for p in args.search_dir:
            search_dirs.extend([Path(p).expanduser().resolve()])
        search_dirs = dedupe_paths(search_dirs) if dedupe_paths else search_dirs
    else:
        base = args.datasets.expanduser().resolve() if args.datasets else out_dir
        search_dirs = [base] + [Path(p).expanduser().resolve() for p in args.search_dir]

    _extra: list[Path] = []
    for _cand in [
        Path("E:/structlift_datasets"),
        Path("/mnt/e/structlift_datasets"),
        REPO_ROOT / "C\uf03aUsers15402" / "structlift_datasets",
    ]:
        if _cand.is_dir():
            _extra.append(_cand.resolve())
    _extra.append(REPO_ROOT.resolve())
    _roots = _extra + search_dirs
    search_dirs = dedupe_paths(_roots) if dedupe_paths else _roots

    projects = [x.strip() for x in args.projects.split(",") if x.strip()]
    min_mc: Optional[float] = None if args.no_mean_conf_filter else float(args.min_mean_conf)
    bin_root = args.bin_root.expanduser().resolve() if args.bin_root else None
    k_pick = max(1, args.per_project)

    chunks: list[str] = [
        "# 9 项目 × O0 × "
        f"{k_pick} 条/项目 = 至多 {len(projects) * k_pick} 样本\n"
        f"# min_bb={args.min_bb}  mean_conf>={min_mc}  seed={args.seed}\n"
        f"# 紧凑模式: omit_matrix_column_table  max_src_lines_jstar={args.max_src_lines_jstar}  "
        f"skip_disasm={not args.with_disasm}\n\n"
    ]
    n_ok = n_skip = 0

    import random

    rng = random.Random(args.seed)

    for proj in projects:
        name = binskel_jsonl_name(proj, "0")
        jp: Optional[Path] = None
        if locate_jsonl:
            jp, _ = locate_jsonl(name, search_dirs)
        else:
            base = (
                args.datasets.expanduser().resolve()
                if args.datasets
                else out_dir
            )
            cand = base / name
            jp = cand if cand.is_file() else None

        if not jp or not jp.is_file():
            chunks.append(
                dataset_banner(proj, jp, -1, "", 0, note=f"missing {name}")
            )
            n_skip += k_pick
            print(f"# skip project={proj} missing {name}", file=sys.stderr, flush=True)
            continue

        candidates = iter_filtered_line_indices(jp, args.min_bb, min_mc)
        if len(candidates) < k_pick:
            chunks.append(
                dataset_banner(
                    proj,
                    jp,
                    -1,
                    "",
                    0,
                    note=f"候选不足: need {k_pick}, have {len(candidates)} "
                    f"(min_bb={args.min_bb})",
                )
            )
            n_skip += k_pick
            print(
                f"# skip project={proj} only {len(candidates)} candidates",
                file=sys.stderr,
                flush=True,
            )
            continue

        picked = pick_spread_line_indices(candidates, k_pick)
        if len(picked) < k_pick:
            extra = [x for x in candidates if x not in picked]
            rng.shuffle(extra)
            for x in extra:
                if len(picked) >= k_pick:
                    break
                picked.append(x)
            picked = picked[:k_pick]

        src_root = src_map.get(proj)
        sr = src_root if src_root and src_root.is_dir() else None

        for li in picked:
            d = load_by_line_idx(jp, li)
            if not d:
                chunks.append(dataset_banner(proj, jp, li, "?", 0, note="load failed"))
                n_skip += 1
                continue

            fn = d.get("func_name", "?")
            n_bb = count_basic_blocks(d)
            elf = resolve_binary_path(d, bin_root) if resolve_binary_path else None

            chunks.append(dataset_banner(proj, jp, li, fn, n_bb))
            po = process_one(
                d,
                sr,
                args.high_w,
                asm_grounded=args.sem_asm_grounded,
                bin_root=bin_root,
            )
            chunks.append(format_dual_block(po))
            chunks.append(
                build_pack(
                    d,
                    sr,
                    elf,
                    num_bb=min(args.sample_bb, max(1, n_bb)),
                    all_blocks=False,
                    skip_disasm=not args.with_disasm,
                    omit_matrix_column_table=True,
                    max_src_lines_bb_jstar=args.max_src_lines_jstar,
                )
            )
            n_ok += 1
            print(
                f"# ok: {proj} line_idx={li} {fn!r} n_bb={n_bb}",
                file=sys.stderr,
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(chunks), encoding="utf-8")
    print(
        f"# 已写 {args.output} | samples_ok={n_ok} slots_skip≈{n_skip}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

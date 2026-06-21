#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9 项目 × 5 档（O0–O3 + Os）= 45 个 binskel jsonl：
每个文件取 1 条样本（与 audit_four_tier 相同的 min_bb / mean_conf 过滤），
将该函数的**全部基本块**按 export_bb_manual_audit_pack 的版式写入**一个** txt，
便于肉眼核对 BB ↔ j* ↔ DWARF ↔ 反汇编。

用法（仓库根）:
  python3 scripts/export_45datasets_one_func_full_bb_audit.py \\
    --all-default-paths --out-dir ~/structlift_datasets \\
    --bin-root ~/structlift_datasets \\
    -o manual_bb_audit_45x1_full.txt

单根目录:
  python3 scripts/export_45datasets_one_func_full_bb_audit.py \\
    --datasets ~/structlift_datasets --bin-root ~/structlift_datasets \\
    -o manual_bb_audit_45x1_full.txt

说明：输出可能很大（45 × 全 BB × 反汇编）；可加 --min-bb 提高门槛减少小函数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_four_tier_alignment_45x3 import (  # noqa: E402
    binskel_jsonl_name,
    iter_filtered_line_indices,
    load_by_line_idx,
    parse_opt_levels,
)
from export_bb_manual_audit_pack import build_pack  # noqa: E402

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
    label: str,
    name: str,
    jp: Optional[Path],
    line_idx: Any,
    func_name: str,
    n_bb: Any,
    note: str = "",
) -> str:
    path_s = str(jp) if jp else ""
    extra = f"\n# NOTE: {note}" if note else ""
    return (
        "\n"
        + "=" * 88
        + "\n"
        + f"# DATASET: {label}\n"
        + f"# jsonl: {name}\n"
        + f"# path: {path_s}\n"
        + f"# line_idx(0-based 有效JSON行): {line_idx}\n"
        + f"# func_name: {func_name}\n"
        + f"# n_bb: {n_bb}\n"
        + "=" * 88
        + extra
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="45 个 jsonl 各 1 函数、全部 BB → 单 txt"
    )
    ap.add_argument("--datasets", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS) if DEFAULT_PROJECTS else "zlib",
    )
    ap.add_argument("--opt-levels", default="0,1,2,3,os")
    ap.add_argument("--min-bb", type=int, default=5)
    ap.add_argument("--min-mean-conf", type=float, default=0.65)
    ap.add_argument("--no-mean-conf-filter", action="store_true")
    ap.add_argument("--bin-root", type=Path, default=None)
    ap.add_argument("--src-map", action="append", default=[], metavar="PROJ=PATH")
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
    for k, v in preset.items():
        pp = Path(v)
        if pp.is_dir():
            src_map.setdefault(k, pp.resolve())

    out_dir = args.out_dir.expanduser().resolve()
    if args.all_default_paths and default_multi_root_search_list and locate_jsonl:
        search_dirs = default_multi_root_search_list(out_dir)
        for p in args.search_dir:
            search_dirs.extend([Path(p).expanduser().resolve()])
        search_dirs = dedupe_paths(search_dirs) if dedupe_paths else search_dirs
    else:
        base = args.datasets.expanduser().resolve() if args.datasets else out_dir
        search_dirs = [base] + [Path(p).expanduser().resolve() for p in args.search_dir]

    projects = [x.strip() for x in args.projects.split(",") if x.strip()]
    opts = parse_opt_levels(args.opt_levels)
    min_mc: Optional[float] = None if args.no_mean_conf_filter else float(args.min_mean_conf)
    bin_root = args.bin_root.expanduser().resolve() if args.bin_root else None

    if resolve_binary_path is None:
        print("# warn: resolve_binary_path 未导入，ELF 可能仅依赖 json 内绝对路径", file=sys.stderr)

    chunks: list[str] = []
    n_ok = n_skip = 0
    n_jobs = len(projects) * len(opts)
    print(
        f"# export_45x1_full_bb: {n_jobs} slots, min_bb={args.min_bb}, "
        f"mean_conf>={min_mc}, writing {args.output}",
        file=sys.stderr,
        flush=True,
    )

    for proj in projects:
        src_root = src_map.get(proj)
        for ot in opts:
            name = binskel_jsonl_name(proj, ot)
            label = f"{proj}_{ot}"
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
                    dataset_banner(
                        label,
                        name,
                        jp,
                        "-",
                        "",
                        "",
                        note="missing jsonl",
                    )
                )
                n_skip += 1
                print(f"# skip: {label} missing {name}", file=sys.stderr, flush=True)
                continue

            candidates = iter_filtered_line_indices(jp, args.min_bb, min_mc)
            if not candidates:
                chunks.append(
                    dataset_banner(
                        label,
                        name,
                        jp,
                        "-",
                        "",
                        "",
                        note=f"no row with ≥{args.min_bb} BB"
                        + ("" if min_mc is None else f" and mean_conf≥{min_mc}"),
                    )
                )
                n_skip += 1
                print(f"# skip: {label} no candidates", file=sys.stderr, flush=True)
                continue

            li = candidates[0]
            d = load_by_line_idx(jp, li)
            if not d:
                n_skip += 1
                chunks.append(
                    dataset_banner(label, name, jp, li, "", "", note="load_by_line_idx failed")
                )
                continue

            fn = d.get("func_name", "?")
            n_bb = count_basic_blocks(d)
            sr = src_root if src_root and src_root.is_dir() else None
            elf = resolve_binary_path(d, bin_root) if resolve_binary_path else None

            chunks.append(dataset_banner(label, name, jp, li, fn, n_bb))
            chunks.append(
                build_pack(
                    d,
                    sr,
                    elf,
                    num_bb=1,
                    all_blocks=True,
                )
            )
            n_ok += 1
            print(
                f"# ok: {label} line_idx={li} {fn!r} n_bb={n_bb}",
                file=sys.stderr,
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(chunks), encoding="utf-8")
    print(
        f"# 已写 {args.output} | ok={n_ok} skip={n_skip} slots={n_jobs}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

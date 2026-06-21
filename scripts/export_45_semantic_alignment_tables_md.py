#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45 个槽位（9 项目 × O0–O3 + Os）与 export_45datasets_one_func_full_bb_audit 相同抽样规则，
或为 manual_bb_audit_45x1_full.txt 中记录的 jsonl + line_idx，
批量生成「BB | 汇编简述 | j* | M | W | 对不对」Markdown 表（与 export_bb_semantic_alignment_table_md 一致）。

用法 A — 自动 45 槽（与 export_45datasets 同 min_bb / mean_conf、第一条候选）:

  python3 scripts/export_45_semantic_alignment_tables_md.py \\
    --datasets ~/structlift_datasets --bin-root ~/structlift_datasets \\
    -o ~/structlift_datasets/semantic_alignment_45_tables.md \\
    --all-bb --auto-verdict

用法 B — 从 manual_bb_audit_45x1_full.txt 读每槽的 path + line_idx（与当时导出 audit 一致）:

  python3 scripts/export_45_semantic_alignment_tables_md.py \\
    --from-audit-txt ~/structlift_datasets/manual_bb_audit_45x1_full.txt \\
    --bin-root ~/structlift_datasets \\
    -o ~/structlift_datasets/semantic_alignment_45_tables.md \\
    --all-bb --auto-verdict

用法 C — 简易清单（每行: label<TAB>jsonl文件名或绝对路径<TAB>line_idx）:

  python3 scripts/export_45_semantic_alignment_tables_md.py \\
    --manifest ~/structlift_datasets/my_45_funcs.txt \\
    --datasets ~/structlift_datasets --bin-root ~/structlift_datasets \\
    -o ~/out.md --all-bb

可选: --out-dir DIR 同时写出 semantic_table_{label}.md 共 45 份。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from export_bb_semantic_alignment_table_md import build_table_md  # noqa: E402

try:
    from check_alignment_semantics import resolve_binary_path  # noqa: E402
except ImportError:
    resolve_binary_path = None  # type: ignore

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

RE_DATASET = re.compile(r"^# DATASET:\s*(\S+)")
RE_JSONL = re.compile(r"^# jsonl:\s*(.+)$")
RE_PATH = re.compile(r"^# path:\s*(.+)$")
RE_LINE_IDX = re.compile(r"^# line_idx.+?:\s*(-?\d+)")


def _preset_src_map() -> Dict[str, Path]:
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
    out: Dict[str, Path] = {}
    for k, v in preset.items():
        pp = Path(v)
        if pp.is_dir():
            out[k] = pp.resolve()
    return out


def _multibin_alt_name(proj: str, ot: str) -> str:
    t = ot.strip().lower()
    if t == "os":
        return f"binskel_{proj}_os_multibin.jsonl"
    return f"binskel_{proj}_o{t}_multibin.jsonl"


def locate_jsonl_maybe_multibin(
    name: str,
    search_dirs: List[Path],
    proj: str,
    ot: str,
) -> Optional[Path]:
    if locate_jsonl:
        jp, _ = locate_jsonl(name, search_dirs)
        if jp is not None and jp.is_file():
            return jp.resolve()
    bn = Path(name).name
    for sd in search_dirs:
        c = sd / bn
        if c.is_file():
            return c.resolve()
    alt = _multibin_alt_name(proj, ot)
    for sd in search_dirs:
        c = sd / alt
        if c.is_file():
            return c.resolve()
    return None


def _norm_path_cross_platform(p: str, search_dirs: List[Path]) -> Optional[Path]:
    raw = p.strip()
    if not raw:
        return None
    cand = Path(raw)
    if cand.is_file():
        return cand.resolve()
    bn = cand.name
    for sd in search_dirs:
        c = sd / bn
        if c.is_file():
            return c.resolve()
    if raw.startswith("/mnt/e/") or raw.startswith("/mnt/E/"):
        tail = raw.split("/", 3)[-1] if raw.count("/") >= 3 else raw
        e = Path("E:/") / tail.replace("/", "\\")
        if e.is_file():
            return e.resolve()
    return None


def parse_audit_txt(path: Path, search_dirs: List[Path]) -> List[Dict[str, Any]]:
    """从 manual_bb_audit_45x1_full.txt 解析各 DATASET 块。"""
    blocks: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_DATASET.match(line)
            if m:
                if cur:
                    blocks.append(cur)
                cur = {
                    "label": m.group(1).strip(),
                    "jsonl_name": "",
                    "path_s": "",
                    "line_idx": -1,
                    "note": "",
                }
                continue
            if cur is None:
                continue
            if m := RE_JSONL.match(line):
                cur["jsonl_name"] = m.group(1).strip()
            elif m := RE_PATH.match(line):
                cur["path_s"] = m.group(1).strip()
            elif m := RE_LINE_IDX.match(line):
                cur["line_idx"] = int(m.group(1))
    if cur:
        blocks.append(cur)

    for b in blocks:
        jp = None
        if b.get("path_s"):
            jp = _norm_path_cross_platform(b["path_s"], search_dirs)
        if jp is None and b.get("jsonl_name"):
            jp = _norm_path_cross_platform(b["jsonl_name"], search_dirs)
        b["jp"] = jp
    return blocks


def parse_manifest(path: Path, search_dirs: List[Path]) -> List[Dict[str, Any]]:
    """每行: label<TAB>jsonl路径或文件名<TAB>line_idx（0-based）。"""
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\t,]+", line)
            if len(parts) < 3:
                print(f"# skip manifest line (need 3 fields): {line}", file=sys.stderr)
                continue
            label, jspec, li_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
            li = int(li_s)
            jp = Path(jspec).expanduser()
            if not jp.is_file():
                jp2 = _norm_path_cross_platform(jspec, search_dirs)
                jp = jp2 if jp2 else jp
            rows.append({"label": label, "line_idx": li, "jp": jp if jp.is_file() else None, "path_s": jspec})
    return rows


def _label_to_proj(label: str) -> str:
    projs = list(DEFAULT_PROJECTS or [])
    projs.sort(key=len, reverse=True)
    for p in projs:
        if label.startswith(p + "_"):
            return p
    return label.split("_", 1)[0] if "_" in label else label


def stub_md(label: str, reason: str) -> str:
    return f"## {label}\n\n*跳过：{reason}*\n\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="45×1 语义对齐 Markdown 表（与 export_45datasets 抽样一致或来自 audit/manifest）"
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
        "--from-audit-txt",
        type=Path,
        default=None,
        help="manual_bb_audit_45x1_full.txt（含 # DATASET / path / line_idx）",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="每行: label<TAB>jsonl路径或文件名<TAB>line_idx",
    )
    ap.add_argument("-o", "--output", type=Path, required=True, help="合并 Markdown 输出")
    ap.add_argument(
        "--out-dir-tables",
        type=Path,
        default=None,
        metavar="DIR",
        help="若设置，则额外写出 semantic_table_{label}.md",
    )
    ap.add_argument("--all-bb", action="store_true", help="输出全部基本块（默认与单脚本一致建议开启）")
    ap.add_argument("--num-bb", type=int, default=13)
    ap.add_argument("--auto-verdict", action="store_true")
    ap.add_argument("--simple-verdict", action="store_true")
    args = ap.parse_args()

    src_map: Dict[str, Path] = {}
    for p in args.src_map:
        if "=" in p:
            k, v = p.split("=", 1)
            src_map[k.strip()] = Path(v).expanduser().resolve()
    for k, v in _preset_src_map().items():
        src_map.setdefault(k, v)

    out_dir = args.out_dir.expanduser().resolve()
    if args.all_default_paths and default_multi_root_search_list and locate_jsonl:
        search_dirs = default_multi_root_search_list(out_dir)
        for p in args.search_dir:
            search_dirs.extend([Path(p).expanduser().resolve()])
        search_dirs = dedupe_paths(search_dirs) if dedupe_paths else search_dirs
    else:
        base = args.datasets.expanduser().resolve() if args.datasets else out_dir
        search_dirs = [base] + [Path(p).expanduser().resolve() for p in args.search_dir]

    bin_root = args.bin_root.expanduser().resolve() if args.bin_root else None
    min_mc: Optional[float] = None if args.no_mean_conf_filter else float(args.min_mean_conf)

    jobs: List[Tuple[str, Optional[Path], int, str]] = []
    # (label, jsonl_path, line_idx, mode_note)

    if args.manifest:
        mf = args.manifest.expanduser().resolve()
        if not mf.is_file():
            print(f"找不到 manifest: {mf}", file=sys.stderr)
            return 1
        for b in parse_manifest(mf, search_dirs):
            label = b["label"]
            li = int(b["line_idx"])
            jp = b.get("jp")
            jobs.append((label, jp, li, "manifest"))
    elif args.from_audit_txt:
        au = args.from_audit_txt.expanduser().resolve()
        if not au.is_file():
            print(f"找不到 audit txt: {au}", file=sys.stderr)
            return 1
        for b in parse_audit_txt(au, search_dirs):
            label = b["label"]
            li = int(b["line_idx"])
            jp = b.get("jp")
            jobs.append((label, jp, li, "audit"))
    else:
        projects = [x.strip() for x in args.projects.split(",") if x.strip()]
        opts = parse_opt_levels(args.opt_levels)
        for proj in projects:
            for ot in opts:
                label = f"{proj}_{ot}"
                name = binskel_jsonl_name(proj, ot)
                jp = locate_jsonl_maybe_multibin(name, search_dirs, proj, ot)
                if jp is None or not jp.is_file():
                    jobs.append((label, None, -1, "missing_jsonl"))
                    continue
                cands = iter_filtered_line_indices(jp, args.min_bb, min_mc)
                if not cands:
                    jobs.append((label, jp, -1, "no_candidate"))
                    continue
                li = cands[0]
                jobs.append((label, jp, li, "auto45"))

    chunks: List[str] = [
        "# 基本块语义对齐表（45 样本）",
        "",
        "列说明与单文件脚本 `export_bb_semantic_alignment_table_md.py` 相同；"
        "「对不对」在未加 `--auto-verdict` 时为待人工。",
        "",
        "---",
        "",
    ]

    n_ok = n_skip = 0
    for label, jp, li, mode in jobs:
        if jp is None or li < 0:
            reason = mode
            if mode == "missing_jsonl":
                reason = "jsonl 未找到（可检查 multibin 命名或 --datasets）"
            elif mode == "no_candidate":
                reason = f"无满足 min_bb≥{args.min_bb} 且 mean_conf 的样本"
            chunks.append(stub_md(label, reason))
            n_skip += 1
            if args.out_dir_tables:
                args.out_dir_tables.mkdir(parents=True, exist_ok=True)
                safe = re.sub(r"[^\w\-.]+", "_", label)
                (args.out_dir_tables / f"semantic_table_{safe}.md").write_text(
                    stub_md(label, reason), encoding="utf-8"
                )
            continue

        d = load_by_line_idx(jp, li)
        if not d:
            chunks.append(stub_md(label, "load_by_line_idx 失败"))
            n_skip += 1
            continue

        proj = _label_to_proj(label)
        src_root = src_map.get(proj)
        if src_root and not src_root.is_dir():
            src_root = None
        elf_path = resolve_binary_path(d, bin_root) if resolve_binary_path else None

        d["_jsonl_path"] = str(jp)
        title_suffix = f"与你先前 {label} 一致"
        md = build_table_md(
            d,
            src_root,
            elf_path,
            bin_root,
            all_bb=args.all_bb,
            num_bb=args.num_bb,
            auto_verdict=args.auto_verdict,
            simple_verdict=args.simple_verdict,
            dataset_banner=None,
            title_suffix=title_suffix,
        )
        chunks.append(md.rstrip() + "\n\n---\n\n")
        n_ok += 1
        if args.out_dir_tables:
            args.out_dir_tables.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w\-.]+", "_", label)
            (args.out_dir_tables / f"semantic_table_{safe}.md").write_text(md, encoding="utf-8")

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(chunks), encoding="utf-8")
    print(f"# wrote {out} | ok={n_ok} skip={n_skip}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

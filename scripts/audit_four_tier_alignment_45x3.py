#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 9 项目 × 5 档（O0–O3 + Os）= 45 个 binskel jsonl：每文件抽取 K 条「基本块不少于阈值」的样本，
按四档标准输出自动化结论（A/B/C 可机器判；D 仅标记为需人工）。

四档（与文档一致）：
  A — 内部自洽：M、block_src_lines、CFG 指标、矩阵–行号一致性（validate + deep_validate）
  B — 调试信息锚：ELF + Capstone 重切 BB，.debug_line 按 PC 重放行 vs 主峰/高W/ json 行（asm 模式）
  C — 句法/位置合理：主峰 BB 的**粗汇编模式**（call/ret/访存）与对齐到的 tree-sitter 语句文本**启发式**是否冲突
  D — 语义等价：无法可靠自动化 → 固定输出 MANUAL_REQUIRED

用法（在 structlift 仓库根，需能 import build_binskel_dataset_md）:
  python3 scripts/audit_four_tier_alignment_45x3.py \\
    --all-default-paths --out-dir ~/structlift_datasets \\
    --bin-root ~/structlift_datasets \\
    --per-file 3 --min-bb 5 --csv -o audit_four_tier_45x3.csv

  # 仅单根目录（不尝试多路径找 jsonl）:
  python3 scripts/audit_four_tier_alignment_45x3.py \\
    --datasets ~/structlift_datasets --bin-root ~/structlift_datasets \\
    --src-map zlib=/path/to/zlib_src ... # 按需
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deep_validate import analyze_sample  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402

try:
    from check_alignment_semantics import (  # noqa: E402
        check_sample_asm_grounded,
        infer_col_to_stmt,
    )
    from check_alignment_semantics import (  # noqa: E402
        _filter_stmts_for_func,
        _parse_file,
    )
except ImportError:
    check_sample_asm_grounded = None  # type: ignore

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


def binskel_jsonl_name(proj: str, opt_token: str) -> str:
    t = opt_token.strip().lower()
    if t == "os":
        return f"binskel_{proj}_os.jsonl"
    return f"binskel_{proj}_o{t}.jsonl"


def parse_opt_levels(arg: str) -> List[str]:
    out: List[str] = []
    for part in arg.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p.lower() if p.lower() == "os" else p.lstrip("oO") or "0")
    return out or ["0"]


def count_basic_blocks(d: dict) -> int:
    bl = d.get("block_src_lines")
    if isinstance(bl, list) and bl:
        return len(bl)
    M = d.get("alignment_matrix")
    if isinstance(M, list) and M:
        return len(M)
    return 0


def load_by_line_idx(path: Path, target: int) -> Optional[dict]:
    idx = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            idx += 1
            if idx == target:
                return json.loads(line)
    return None


def iter_filtered_line_indices(
    path: Path, min_bb: int, min_mean_conf: Optional[float]
) -> List[int]:
    out: List[int] = []
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


def pick_spread_line_indices(candidates: List[int], k: int) -> List[int]:
    m = len(candidates)
    if m == 0 or k <= 0:
        return []
    if m <= k:
        return candidates[:m]
    if k == 1:
        return [candidates[m // 2]]
    positions = sorted({int((m - 1) * i / (k - 1)) for i in range(k)})
    return [candidates[p] for p in positions]


def grade_tier_a(d: dict) -> Tuple[str, str]:
    """内部自洽：FAIL/WARN/OK + 备注。"""
    v = analyze_sample(d)
    r_val = validate_sample(d)
    notes: List[str] = []
    g = "OK"
    if not v.get("matrix_lines_consistent", True):
        g = "FAIL"
        notes.append("M与block_src_lines不自洽")
    if v.get("empty_ratio", 0) > 0.2:
        g = "WARN" if g != "FAIL" else g
        notes.append(f"空BB比例{v['empty_ratio']:.0%}")
    if v.get("order_consistency", 1) < 0.7:
        g = "WARN" if g != "FAIL" else g
        notes.append(f"顺序一致性{v['order_consistency']:.2f}")
    if r_val.get("coverage", 0) < 0.75:
        g = "WARN" if g != "FAIL" else g
        notes.append(f"coverage={r_val['coverage']:.2f}")
    return g, "; ".join(notes) if notes else "-"


def grade_tier_b(
    d: dict, src_root: Path, bin_root: Optional[Path]
) -> Tuple[str, str]:
    """调试信息锚：依赖 asm-grounded 检查。"""
    if check_sample_asm_grounded is None:
        return "SKIP", "check_alignment_semantics 未导入"
    try:
        r = check_sample_asm_grounded(
            d,
            src_root,
            high_w=0.85,
            viol_detail_limit=None,
            bin_root=bin_root,
        )
    except Exception as e:
        return "SKIP", str(e)
    if r.get("skip"):
        return "SKIP", str(r.get("reason", "skip"))
    if not r.get("asm_cfg_ok", True):
        return "FAIL", r.get("asm_cfg_reason", "asm_cfg_ok=False")
    npk = int(r.get("n_viol_peak") or 0)
    nhw = int(r.get("n_viol_highw") or 0)
    ndj = int(r.get("n_viol_dwarf_json") or 0)
    npc = int(r.get("n_viol_no_pc_line") or 0)
    if npk or nhw:
        return "WARN", f"peak={npk} highw={nhw} dwarf_json={ndj} no_pc_line={npc}"
    if ndj or npc:
        return "WARN", f"dwarf_json={ndj} no_pc_line={npc}"
    return "OK", "-"


def _resolve_src_path(d: dict, src_root: Path) -> Optional[Path]:
    sf = d.get("source_file")
    if not sf:
        return None
    rel = Path(sf).name
    p = src_root / rel
    if p.is_file():
        return p
    for hit in src_root.rglob(rel):
        if hit.is_file():
            return hit
    return None


def grade_tier_c(d: dict, src_root: Path, bin_root: Optional[Path]) -> Tuple[str, str]:
    """
    句法/位置粗判：逐 BB 看 call/ret 与源码片段是否明显矛盾（启发式）。
    SKIP：无 ELF/无源码/无 capstone。
    """
    try:
        from elftools.elf.elffile import ELFFile
        from build_binskel_dataset_md import build_cfg_for_function, iter_functions
    except ImportError as e:
        return "SKIP", str(e)

    from check_alignment_semantics import resolve_binary_path

    sf = d.get("source_file")
    src_path = _resolve_src_path(d, src_root)
    if not src_path:
        return "SKIP", "无源码"
    M = d.get("alignment_matrix") or []
    bl = d.get("block_src_lines") or []
    if not M or not bl:
        return "SKIP", "无矩阵"

    elf_path = resolve_binary_path(d, bin_root)
    if not elf_path or not elf_path.is_file():
        return "SKIP", "无ELF"

    fn = d.get("func_name")
    if not fn:
        return "SKIP", "无函数名"

    try:
        statements, _ = _parse_file(src_path)
        all_lns = [ln for ls in bl for ln in (ls or [])]
        if not all_lns:
            return "SKIP", "无DWARF行"
        fmin, fmax = min(all_lns), max(all_lns)
        filtered = _filter_stmts_for_func(statements, fmin, fmax)
        col_stmts, _ = infer_col_to_stmt(bl, M, filtered, thr=0.005)
        src_lines = src_path.read_text(encoding="utf-8", errors="replace").splitlines()

        with elf_path.open("rb") as ef:
            elffile = ELFFile(ef)
            text_sec = elffile.get_section_by_name(".text")
            if text_sec is None:
                return "SKIP", "无.text"
            text_bytes = text_sec.data()
            base_addr = text_sec["sh_addr"]
            addr = size = None
            for name, a, sz in iter_functions(elffile):
                if name == fn:
                    addr, size = a, sz
                    break
            if addr is None:
                return "SKIP", "符号无"
            start = addr - base_addr
            if start < 0 or start + size > len(text_bytes):
                return "SKIP", "范围越界"
            code_bytes = text_bytes[start : start + size]
            cfg, _ = build_cfg_for_function(code_bytes, addr, fn)
    except Exception as e:
        return "SKIP", str(e)

    n_bb = len(bl)
    suspicions = 0
    checked = 0
    detail: List[str] = []

    for i in range(n_bb):
        D = set(bl[i] or [])
        row = M[i] if i < len(M) else []
        if not D or not row or max(float(x) for x in row) < 1e-6:
            continue
        if i >= len(cfg.blocks):
            break
        j_star = max(range(len(row)), key=lambda j: row[j])
        st = col_stmts[j_star] if j_star < len(col_stmts) else None
        if st is None:
            continue
        checked += 1
        lo, hi = st["start_line"], st["end_line"]
        snippet = ""
        for ln in range(lo, hi + 1):
            if 1 <= ln <= len(src_lines):
                snippet += src_lines[ln - 1] + "\n"
        snippet_l = snippet.lower()

        raw_blob = " ".join(
            (inst.raw_text or "").lower()
            for inst in cfg.blocks[i].instructions
        )
        has_call = " call" in f" {raw_blob}" or raw_blob.startswith("call")
        has_ret = " ret" in f" {raw_blob}" or "retn" in raw_blob

        if has_call and "(" not in snippet and " assert" not in snippet_l:
            suspicions += 1
            detail.append(f"BB{i}:call_vs_no_paren")
        if has_ret and "return" not in snippet_l and hi > lo:
            suspicions += 1
            detail.append(f"BB{i}:ret_vs_no_return")

    if checked == 0:
        return "SKIP", "无可用BB做C档"
    if suspicions == 0:
        return "OK", f"checked_bb={checked}"
    if suspicions <= 2:
        return "WARN", f"checked_bb={checked}; " + ",".join(detail[:6])
    return "FAIL", f"checked_bb={checked}; " + ",".join(detail[:8])


def main() -> int:
    ap = argparse.ArgumentParser(description="45×K 样本四档 BB-语句对齐审计")
    ap.add_argument(
        "--datasets",
        type=Path,
        default=None,
        help="jsonl 主目录（与 --all-default-paths 二选一或同用）",
    )
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument(
        "--projects",
        default=",".join(DEFAULT_PROJECTS) if DEFAULT_PROJECTS else "zlib",
    )
    ap.add_argument("--opt-levels", default="0,1,2,3,os")
    ap.add_argument("--per-file", type=int, default=3)
    ap.add_argument("--min-bb", type=int, default=5)
    ap.add_argument(
        "--min-mean-conf",
        type=float,
        default=0.65,
        help="mean_conf 下限（略低于 audit_three 默认，避免 45 档里候选过少）",
    )
    ap.add_argument("--no-mean-conf-filter", action="store_true")
    ap.add_argument(
        "--bin-root",
        type=Path,
        default=None,
        help="ELF 查找根（建议与数据集目录一致）",
    )
    ap.add_argument(
        "--src-map",
        action="append",
        default=[],
        metavar="PROJ=PATH",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--csv", action="store_true", help="CSV；否则 TSV")
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

    rows: List[Dict[str, Any]] = []
    n_jobs = len(projects) * len(opts)
    print(
        f"# audit_four_tier: {n_jobs} jsonl slots, per_file={args.per_file}, "
        f"min_bb={args.min_bb} (stderr progress; CSV written at end)",
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
                print(f"# skip: {label} (missing {name})", file=sys.stderr, flush=True)
                rows.append(
                    {
                        "dataset": label,
                        "jsonl": name,
                        "line_idx": "",
                        "func_name": "",
                        "n_bb": "",
                        "tier_a": "SKIP",
                        "tier_b": "SKIP",
                        "tier_c": "SKIP",
                        "tier_d": "MANUAL_REQUIRED",
                        "note_a": "missing jsonl",
                        "note_b": "-",
                        "note_c": "-",
                        "note_d": "文件缺失；D 档恒为人工",
                    }
                )
                continue

            candidates = iter_filtered_line_indices(jp, args.min_bb, min_mc)
            if not candidates:
                print(
                    f"# skip: {label} (no candidates min_bb={args.min_bb})",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(
                    {
                        "dataset": label,
                        "jsonl": name,
                        "line_idx": "",
                        "func_name": "",
                        "n_bb": "",
                        "tier_a": "SKIP",
                        "tier_b": "SKIP",
                        "tier_c": "SKIP",
                        "tier_d": "MANUAL_REQUIRED",
                        "note_a": f"无候选: ≥{args.min_bb} BB"
                        + ("" if min_mc is None else f", mean_conf≥{min_mc}"),
                        "note_b": "-",
                        "note_c": "-",
                        "note_d": "候选不足；D 档恒为人工",
                    }
                )
                continue

            idxs = pick_spread_line_indices(candidates, args.per_file)
            for li in idxs:
                d = load_by_line_idx(jp, li)
                if not d:
                    continue
                fn = d.get("func_name", "?")
                n_bb = count_basic_blocks(d)

                ga, na = grade_tier_a(d)

                if not src_root or not src_root.is_dir():
                    gb, nb = "SKIP", "no src_root for project"
                    gc, nc = "SKIP", "no src_root"
                else:
                    gb, nb = grade_tier_b(d, src_root, bin_root)
                    gc, nc = grade_tier_c(d, src_root, bin_root)

                gd = "MANUAL_REQUIRED"
                nd = "语义等价需人工/动态执行；脚本不自动判 PASS/FAIL"

                rows.append(
                    {
                        "dataset": label,
                        "jsonl": str(jp),
                        "line_idx": li,
                        "func_name": fn,
                        "n_bb": n_bb,
                        "tier_a": ga,
                        "tier_b": gb,
                        "tier_c": gc,
                        "tier_d": gd,
                        "note_a": na,
                        "note_b": nb,
                        "note_c": nc,
                        "note_d": nd,
                    }
                )
                print(
                    f"# progress: {label} line_idx={li} {fn[:48]!r} "
                    f"A={ga} B={gb} C={gc} | rows={len(rows)}",
                    file=sys.stderr,
                    flush=True,
                )

    summary = {
        "A": {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0},
        "B": {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0},
        "C": {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0},
        "D": {"MANUAL": 0},
    }
    for r in rows:
        for t in ("A", "B", "C"):
            k = r.get(f"tier_{t.lower()}", "SKIP")
            if k not in summary[t]:
                k = "SKIP"
            summary[t][k] += 1
        if r.get("tier_d") == "MANUAL_REQUIRED":
            summary["D"]["MANUAL"] += 1

    fieldnames = [
        "dataset",
        "jsonl",
        "line_idx",
        "func_name",
        "n_bb",
        "tier_a",
        "tier_b",
        "tier_c",
        "tier_d",
        "note_a",
        "note_b",
        "note_c",
        "note_d",
    ]

    def write_to(stream) -> None:
        if args.csv:
            w = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        else:
            stream.write("\t".join(fieldnames) + "\n")
            for r in rows:
                stream.write(
                    "\t".join(str(r.get(c, "")) for c in fieldnames) + "\n"
                )

    write_to(sys.stdout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="" if args.csv else "") as f:
            write_to(f)

    print("\n# === 汇总（每档行数 = 输出行数）===", file=sys.stderr)
    for t in ("A", "B", "C"):
        print(f"# Tier {t}: {summary[t]}", file=sys.stderr)
    print(f"# Tier D: {summary['D']}（均为 MANUAL）", file=sys.stderr)
    if args.output:
        print(f"# 已写 {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

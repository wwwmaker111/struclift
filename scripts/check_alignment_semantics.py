#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对齐「语义」自检（可自动化，非人工读汇编）

不声称等价于形式化验证；检查的是 **自洽性**：
  1) 用 tree-sitter 重解析源文件，得到语句行号范围；
  2) 用矩阵每列在各 BB 上的非零质量，推断「列 ↔ 语句」；
  3) 对每个有 DWARF 行的 BB：主峰列对应的语句行范围是否与 DWARF 行有交集；
  4) 对高 W 的 (BB,列)：同样要求与对应 BB 的 DWARF 行有交集。

用法:
  python scripts/check_alignment_semantics.py \\
    ~/structlift_datasets/binskel_zlib_o0.jsonl \\
    --src-root ~/struclift_wsl/datasets/zlib/zlib-1.3.1

  python scripts/check_alignment_semantics.py --all-opt \\
    --data-dir ~/structlift_datasets \\
    --src-root ~/struclift_wsl/datasets/zlib/zlib-1.3.1

说明:
  - 列压缩后 jsonl 不保存「列号→语句 id」映射，这里用 **列支撑行集** 与语句行范围
    最大交启发式恢复，小函数上通常与构建时一致；若列映射失败会标为 weak_mapping。
  - 无法替代：逐条对照 objdump 与源码的控制流证明。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_file(path: Path):
    import build_binskel_dataset_md as md

    text = path.read_text(encoding="utf-8", errors="replace")
    return md._parse_snippet_and_offset(text, 0)


def _filter_stmts_for_func(
    statements: List[Dict], func_min: int, func_max: int
) -> List[Dict]:
    out = []
    for s in statements:
        if s["end_line"] >= func_min and s["start_line"] <= func_max:
            out.append({**s, "id": len(out)})
    if not out and statements and func_min > 0:
        lo, hi = max(1, func_min - 15), func_max + 15
        for s in statements:
            if s["end_line"] >= lo and s["start_line"] <= hi:
                out.append({**s, "id": len(out)})
    return out


def _column_support_lines(block_src_lines: List[List[int]], M: List[List[float]], t: int, thr: float) -> set:
    s = set()
    for i, row in enumerate(M):
        if t >= len(row) or row[t] <= thr:
            continue
        for ln in block_src_lines[i] or []:
            s.add(ln)
    return s


def infer_col_to_stmt(
    block_src_lines: List[List[int]],
    M: List[List[float]],
    filtered_stmts: List[Dict],
    thr: float = 0.005,
) -> Tuple[List[Optional[Dict]], List[int]]:
    """返回 (每列对应语句 dict 或 None, 每列与支撑集交大小).

    与构建时 map_lines_to_statement 一致：优先排除 compound_statement，
    取最深最窄的语句；仅当无其他候选时才回退到 compound_statement。
    """
    if not M or not M[0]:
        return [], []
    n_col = len(M[0])
    scores: List[int] = []
    mapped: List[Optional[Dict]] = []
    for t in range(n_col):
        sup = _column_support_lines(block_src_lines, M, t, thr)
        if not sup:
            scores.append(0)
            mapped.append(None)
            continue

        candidates: List[Tuple[int, int, Dict]] = []
        for k, st in enumerate(filtered_stmts):
            S = set(range(st["start_line"], st["end_line"] + 1))
            sc = len(S & sup)
            if sc > 0:
                candidates.append((k, sc, st))

        if not candidates:
            scores.append(0)
            mapped.append(None)
            continue

        non_compound = [(k, sc, st) for k, sc, st in candidates
                        if st.get("type") != "compound_statement"]
        pool = non_compound if non_compound else candidates

        best_k, best_sc, best_st = max(
            pool,
            key=lambda x: (
                x[1],
                x[2].get("depth", 0),
                -(x[2]["end_line"] - x[2]["start_line"]),
            ),
        )
        scores.append(best_sc)
        mapped.append(best_st)
    return mapped, scores


def stmt_overlaps_lines(st: Dict, lines: set) -> bool:
    S = set(range(st["start_line"], st["end_line"] + 1))
    return bool(S & lines)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def resolve_binary_path(d: Dict[str, Any], bin_root: Optional[Path]) -> Optional[Path]:
    """jsonl 里 binary 常为构建机绝对路径；可在本机用 bin_root / rglob 兜底。"""
    raw = d.get("binary")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    if p.is_file():
        return p.resolve()
    name = p.name
    if bin_root:
        br = Path(bin_root).expanduser().resolve()
        cand = br / name
        if cand.is_file():
            return cand.resolve()
        try:
            for hit in br.rglob(name):
                if hit.is_file():
                    return hit.resolve()
        except OSError:
            pass
    return None


def check_sample_asm_grounded(
    d: Dict[str, Any],
    src_root: Path,
    high_w: float = 0.85,
    viol_detail_limit: Optional[int] = 5,
    bin_root: Optional[Path] = None,
    dwarf_json_jaccard_warn: float = 0.85,
) -> Dict[str, Any]:
    """
    **汇编/ELF 锚定**语义检查（强于纯源码启发式，仍非形式化证明）：

    1. 打开 jsonl 记录的 ELF，解析 .debug_line（与 build_binskel_dataset_md 同源）。
    2. 用 Capstone 对 **同一函数** 重切基本块（与数据集构建同一例程 build_cfg_for_function）。
    3. 校验 BB 数、`bin_block_lengths` 与重切结果一致；不一致则整个样本记为 asm_cfg_ok=False。
    4. 对每个 BB，用指令地址区间调用 get_bb_source_lines 得到 **PC 重放行号集合 L_pc**。
    5. 主峰/高 W 列映射到的 tree-sitter 语句行必须与 **L_pc**（非空时）有交集；否则计 violation。
    6. 若 json 内 block_src_lines 与 L_pc 差异大（Jaccard < dwarf_json_jaccard_warn），另记 viol_dwarf_json
       （标注「标注行号」与「PC 重放」不一致，需人审）。
    """
    try:
        from elftools.elf.elffile import ELFFile
        from build_binskel_dataset_md import (
            build_cfg_for_function,
            extract_dwarf_line_map,
            get_bb_source_lines,
            iter_functions,
        )
    except ImportError as e:
        return {"skip": True, "reason": f"asm_grounded_import:{e}"}

    sf = d.get("source_file")
    if not sf:
        return {"skip": True, "reason": "no source_file"}
    rel = Path(sf).name
    src_path = src_root / rel
    if not src_path.is_file():
        for p in src_root.rglob(rel):
            if p.is_file():
                src_path = p
                break
        if not src_path.is_file():
            return {"skip": True, "reason": f"missing {rel} under {src_root}"}

    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    bl = d.get("block_src_lines") or []
    if not M or not bl:
        return {"skip": True, "reason": "no matrix or block_src_lines"}

    elf_path = resolve_binary_path(d, bin_root)
    if not elf_path or not elf_path.is_file():
        return {
            "skip": True,
            "reason": "missing ELF (json binary path invalid; try --bin-root)",
        }

    func_name = d.get("func_name")
    if not func_name:
        return {"skip": True, "reason": "no func_name"}

    try:
        with elf_path.open("rb") as ef:
            elffile = ELFFile(ef)
            text_sec = elffile.get_section_by_name(".text")
            if text_sec is None:
                return {"skip": True, "reason": "ELF has no .text"}
            text_bytes = text_sec.data()
            base_addr = text_sec["sh_addr"]
            dwarf_entries = extract_dwarf_line_map(elffile)
            addr = size = None
            for name, a, sz in iter_functions(elffile):
                if name == func_name:
                    addr, size = a, sz
                    break
            if addr is None or size is None:
                return {"skip": True, "reason": f"symbol {func_name} not in ELF"}
            start = addr - base_addr
            if start < 0 or start + size > len(text_bytes):
                return {"skip": True, "reason": "function range OOB"}
            code_bytes = text_bytes[start : start + size]
            cfg, block_addr_ranges = build_cfg_for_function(code_bytes, addr, func_name)
    except Exception as e:
        return {"skip": True, "reason": f"elf/cfg:{e}"}

    if len(block_addr_ranges) != len(bl) or len(block_addr_ranges) != len(M):
        return {
            "skip": False,
            "asm_grounded": True,
            "asm_cfg_ok": False,
            "asm_cfg_reason": f"bb_count cfg={len(block_addr_ranges)} json={len(bl)}",
            "func_name": func_name,
            "opt_level": d.get("opt_level"),
            "weak_map_cols": 0,
            "n_col": len(M[0]) if M and M[0] else 0,
            "n_stmt_filtered": 0,
            "n_bb_checked": 0,
            "n_viol_peak": 0,
            "n_viol_highw": 0,
            "viol_peak": [],
            "viol_highw": [],
            "n_viol_dwarf_json": 0,
            "viol_dwarf_json": [],
            "n_viol_no_pc_line": 0,
            "bb_pc_line_count": [],
        }

    json_lens = d.get("bin_block_lengths") or []
    asm_bb_lens_ok = True
    if json_lens:
        if len(json_lens) != len(cfg.blocks):
            asm_bb_lens_ok = False
        else:
            for i, bb in enumerate(cfg.blocks):
                if int(json_lens[i]) != len(bb.instructions):
                    asm_bb_lens_ok = False
                    break

    all_lns = [ln for ls in bl for ln in (ls or [])]
    if not all_lns:
        return {"skip": True, "reason": "no DWARF lines in any BB"}

    statements, _ = _parse_file(src_path)
    fmin, fmax = min(all_lns), max(all_lns)
    filtered = _filter_stmts_for_func(statements, fmin, fmax)
    col_stmts, col_scores = infer_col_to_stmt(bl, M, filtered)
    weak_map = sum(1 for sc in col_scores if sc == 0)

    prim_name = Path(sf).name if sf else None
    L_pc_by_bb: List[set] = []
    for sa, ea in block_addr_ranges:
        infos = get_bb_source_lines(sa, ea + 8, dwarf_entries)
        if prim_name:
            fi = [
                x
                for x in infos
                if x.get("file_path") and Path(str(x["file_path"])).name == prim_name
            ]
            use_infos = fi if fi else infos
        else:
            use_infos = infos
        L_pc_by_bb.append({int(x["line"]) for x in use_infos if x.get("line")})

    viol_dwarf_json: List[Dict[str, Any]] = []
    for i in range(len(bl)):
        Lj = {int(x) for x in (bl[i] or [])}
        Lp = L_pc_by_bb[i]
        if not Lj or not Lp:
            continue
        j = _jaccard(Lj, Lp)
        if j + 1e-9 < dwarf_json_jaccard_warn:
            viol_dwarf_json.append(
                {
                    "bb": i,
                    "jaccard": round(j, 4),
                    "json": sorted(Lj)[:16],
                    "pc": sorted(Lp)[:16],
                }
            )

    violations_peak: List[Dict[str, Any]] = []
    violations_highw: List[Dict[str, Any]] = []
    viol_no_pc_line = 0
    n_checked_bb = 0
    n_bb = len(bl)
    n_col = len(M[0])

    for i in range(n_bb):
        D_json = set(bl[i] or [])
        Lpc = L_pc_by_bb[i]
        if not D_json:
            continue
        row = M[i]
        if not row or max(row) < 1e-6:
            continue
        n_checked_bb += 1
        if not Lpc:
            viol_no_pc_line += 1
        D_eff = Lpc if Lpc else D_json

        j_star = max(range(len(row)), key=lambda j: row[j])
        st = col_stmts[j_star] if j_star < len(col_stmts) else None
        if st is None or not stmt_overlaps_lines(st, D_eff):
            violations_peak.append(
                {
                    "bb": i,
                    "j_peak": j_star,
                    "dwarf_lines_json": sorted(D_json)[:12],
                    "dwarf_lines_pc": sorted(Lpc)[:12] if Lpc else [],
                    "stmt": f"L{st['start_line']}-{st['end_line']}" if st else None,
                    "col_support_score": col_scores[j_star] if j_star < len(col_scores) else -1,
                    "used_pc_lines": bool(Lpc),
                }
            )

        row_w = W[i] if i < len(W) else []
        for j in range(min(len(row), len(row_w))):
            if row[j] <= 0.01 or row_w[j] < high_w:
                continue
            stj = col_stmts[j] if j < len(col_stmts) else None
            if stj is None or not stmt_overlaps_lines(stj, D_eff):
                violations_highw.append(
                    {
                        "bb": i,
                        "j": j,
                        "W": round(row_w[j], 4),
                        "dwarf_lines_json": sorted(D_json)[:8],
                        "dwarf_lines_pc": sorted(Lpc)[:8] if Lpc else [],
                        "stmt": f"L{stj['start_line']}-{stj['end_line']}" if stj else None,
                        "used_pc_lines": bool(Lpc),
                    }
                )

    def _lim(xs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if viol_detail_limit is None:
            return xs
        return xs[: viol_detail_limit]

    return {
        "skip": False,
        "asm_grounded": True,
        "asm_cfg_ok": asm_bb_lens_ok,
        "asm_cfg_reason": (
            "" if asm_bb_lens_ok else "bin_block_lengths!=capstone or len mismatch"
        ),
        "func_name": func_name,
        "opt_level": d.get("opt_level"),
        "weak_map_cols": weak_map,
        "n_col": n_col,
        "n_stmt_filtered": len(filtered),
        "n_bb_checked": n_checked_bb,
        "n_viol_peak": len(violations_peak),
        "n_viol_highw": len(violations_highw),
        "viol_peak": _lim(violations_peak),
        "viol_highw": _lim(violations_highw),
        "n_viol_dwarf_json": len(viol_dwarf_json),
        "viol_dwarf_json": _lim(viol_dwarf_json),
        "n_viol_no_pc_line": viol_no_pc_line,
        "bb_pc_line_count": [len(x) for x in L_pc_by_bb],
    }


def check_sample(
    d: Dict[str, Any],
    src_root: Path,
    high_w: float = 0.85,
    viol_detail_limit: Optional[int] = 5,
    *,
    asm_grounded: bool = False,
    bin_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if asm_grounded:
        return check_sample_asm_grounded(
            d, src_root, high_w=high_w, viol_detail_limit=viol_detail_limit, bin_root=bin_root
        )

    sf = d.get("source_file")
    if not sf:
        return {"skip": True, "reason": "no source_file"}
    rel = Path(sf).name
    src_path = src_root / rel
    if not src_path.is_file():
        # jsonl 常为绝对路径；源码树可能在子目录（如 curl 的 lib/strcase.c）
        for p in src_root.rglob(rel):
            if p.is_file():
                src_path = p
                break
        if not src_path.is_file():
            return {"skip": True, "reason": f"missing {rel} under {src_root}"}

    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    bl = d.get("block_src_lines") or []
    if not M or not bl:
        return {"skip": True, "reason": "no matrix or block_src_lines"}

    statements, _ = _parse_file(src_path)
    all_lns = [ln for ls in bl for ln in (ls or [])]
    if not all_lns:
        return {"skip": True, "reason": "no DWARF lines in any BB"}
    fmin, fmax = min(all_lns), max(all_lns)
    filtered = _filter_stmts_for_func(statements, fmin, fmax)

    col_stmts, col_scores = infer_col_to_stmt(bl, M, filtered)
    weak_map = sum(1 for sc in col_scores if sc == 0)

    violations_peak = []
    violations_highw = []
    n_checked_bb = 0

    n_bb = len(bl)
    n_col = len(M[0])
    for i in range(n_bb):
        D = set(bl[i] or [])
        if not D:
            continue
        row = M[i]
        if not row or max(row) < 1e-6:
            continue
        n_checked_bb += 1
        j_star = max(range(len(row)), key=lambda j: row[j])
        st = col_stmts[j_star] if j_star < len(col_stmts) else None
        if st is None or not stmt_overlaps_lines(st, D):
            violations_peak.append(
                {
                    "bb": i,
                    "j_peak": j_star,
                    "dwarf_lines": sorted(D)[:12],
                    "stmt": f"L{st['start_line']}-{st['end_line']}" if st else None,
                    "col_support_score": col_scores[j_star] if j_star < len(col_scores) else -1,
                }
            )

        row_w = W[i] if i < len(W) else []
        for j in range(min(len(row), len(row_w))):
            if row[j] <= 0.01 or row_w[j] < high_w:
                continue
            stj = col_stmts[j] if j < len(col_stmts) else None
            if stj is None or not stmt_overlaps_lines(stj, D):
                violations_highw.append(
                    {
                        "bb": i,
                        "j": j,
                        "W": round(row_w[j], 4),
                        "dwarf_lines": sorted(D)[:8],
                        "stmt": f"L{stj['start_line']}-{stj['end_line']}" if stj else None,
                    }
                )

    if viol_detail_limit is None:
        vp_out = violations_peak
        vh_out = violations_highw
    else:
        vp_out = violations_peak[:viol_detail_limit]
        vh_out = violations_highw[:viol_detail_limit]

    return {
        "skip": False,
        "func_name": d.get("func_name"),
        "opt_level": d.get("opt_level"),
        "weak_map_cols": weak_map,
        "n_col": n_col,
        "n_stmt_filtered": len(filtered),
        "n_bb_checked": n_checked_bb,
        "n_viol_peak": len(violations_peak),
        "n_viol_highw": len(violations_highw),
        "viol_peak": vp_out,
        "viol_highw": vh_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="binskel 对齐语义自洽性检查")
    ap.add_argument("jsonl", nargs="?", type=Path, help="单个 binskel jsonl")
    ap.add_argument("--src-root", type=Path, required=True, help="源码根目录（与构建时 --src 一致）")
    ap.add_argument("--limit", type=int, default=0, help="最多检查前 N 条（0=全部）")
    ap.add_argument("--high-w", type=float, default=0.85)
    ap.add_argument("--all-opt", action="store_true", help="检查 data-dir 下 binskel_zlib_o0..o3")
    ap.add_argument("--data-dir", type=Path, help="与 audit 脚本相同")
    ap.add_argument(
        "--asm-grounded",
        action="store_true",
        help="用 ELF+DWARF 按指令地址重放行号，再与主峰/高W 核对（需 pyelftools/capstone，且 json binary 路径可用或配合 --bin-root）",
    )
    ap.add_argument(
        "--bin-root",
        type=Path,
        default=None,
        help="json 里 binary 路径失效时，在此目录下按文件名/rglob 查找 ELF",
    )
    args = ap.parse_args()

    paths: List[Path] = []
    if args.all_opt:
        if not args.data_dir:
            ap.error("--all-opt 需要 --data-dir")
        dd = args.data_dir.expanduser()
        for k in range(4):
            p = dd / f"binskel_zlib_o{k}.jsonl"
            if p.is_file():
                paths.append(p)
    else:
        if not args.jsonl:
            ap.error("请指定 jsonl 或使用 --all-opt")
        paths = [args.jsonl.expanduser()]

    src_root = args.src_root.expanduser()

    for jp in paths:
        print("=" * 72)
        print(f"文件: {jp}")
        print("=" * 72)
        tot = peak_bad = highw_bad = skipped = checked = 0
        weak_maps = []

        with open(jp, encoding="utf-8", errors="replace") as f:
            for line_i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                tot += 1
                if args.limit and checked >= args.limit:
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r = check_sample(
                    d,
                    src_root,
                    high_w=args.high_w,
                    asm_grounded=args.asm_grounded,
                    bin_root=args.bin_root,
                )
                if r.get("skip"):
                    skipped += 1
                    continue
                checked += 1
                if r.get("weak_map_cols", 0) > 0:
                    weak_maps.append((r["func_name"], r["weak_map_cols"]))
                if r["n_viol_peak"] > 0:
                    peak_bad += 1
                if r["n_viol_highw"] > 0:
                    highw_bad += 1
                if r["n_viol_peak"] or r["n_viol_highw"]:
                    print(f"\n[{r['opt_level']}] {r['func_name']}")
                    print(
                        f"  weak_map_cols={r['weak_map_cols']}/{r['n_col']}  "
                        f"n_stmt={r['n_stmt_filtered']}  bb_checked={r['n_bb_checked']}"
                    )
                    if r["viol_peak"]:
                        print(f"  主峰与 DWARF 行无交 ({r['n_viol_peak']} 个 BB)，示例:")
                        for v in r["viol_peak"]:
                            print(f"    BB{v['bb']} j={v['j_peak']} stmt={v['stmt']} dwarf={v['dwarf_lines']} col_sup={v['col_support_score']}")
                    if r["viol_highw"]:
                        print(f"  高W(>={args.high_w})但与 DWARF 无交: {r['n_viol_highw']} 处，示例:")
                        for v in r["viol_highw"]:
                            print(f"    BB{v['bb']} j={v['j']} W={v['W']} stmt={v['stmt']} dwarf={v['dwarf_lines']}")

        print(f"\n汇总: 读入 {tot} 条, 实查 {checked} 条, 跳过 {skipped} 条")
        print(f"  主峰语义冲突样本数: {peak_bad}")
        print(f"  高W语义冲突样本数: {highw_bad}")
        if weak_maps:
            print(f"  列映射支撑为空(weak)的列出现过 {len(weak_maps)} 个样本 (func, n_weak_cols 前5):")
            for x in weak_maps[:5]:
                print(f"    {x}")
        print()


if __name__ == "__main__":
    main()

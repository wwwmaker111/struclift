#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zlib：每个优化等级抽取 5 条样本，打印对齐摘要（DWARF 行、M_gt 主峰列、置信度），供人工核对。

用法（WSL/Linux 示例）:
  python scripts/audit_zlib_alignment_five_samples.py --data-dir ~/structlift_datasets

或显式指定四个 jsonl:
  python scripts/audit_zlib_alignment_five_samples.py \\
    --o0 /path/binskel_zlib_o0.jsonl --o1 ... --o2 ... --o3 ...

可选：--zlib-src 指向 zlib 源码根目录（如 .../zlib-1.3.1），用于在本地打印对应行源码。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_five_indices(n: int) -> list[int]:
    if n == 0:
        return []
    if n <= 5:
        return list(range(n))
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})


def read_source_lines(src_root: Path | None, source_file: str, line_nums: list[int]) -> dict[int, str]:
    """按 basename 在 src_root 下找文件，读取指定行。"""
    if not src_root or not source_file:
        return {}
    base = Path(source_file).name
    cand = src_root / base
    if not cand.is_file():
        return {}
    text = cand.read_text(encoding="utf-8", errors="replace").splitlines()
    out = {}
    for ln in line_nums:
        if 1 <= ln <= len(text):
            out[ln] = text[ln - 1].rstrip()[:120]
    return out


def _bb_row_summary(
    i: int,
    src_ln: list,
    row_m: list[float],
    row_w: list[float],
) -> str:
    if not row_m:
        return f"    BB{i}: DWARF行={src_ln}  [无 alignment_matrix 行]"
    j_star = max(range(len(row_m)), key=lambda j: row_m[j])
    m_val = row_m[j_star]
    w_val = row_w[j_star] if j_star < len(row_w) else 0.0
    s_m = sum(row_m)
    return (
        f"    BB{i}: DWARF行={src_ln}  -> 主峰 stmt列 j={j_star}  "
        f"M={m_val:.4f} W={w_val:.4f}  row_sum(M)={s_m:.4f}"
    )


def summarize_one(
    d: dict,
    src_root: Path | None,
    verbose_source: bool,
    max_bb_lines: int = 24,
) -> str:
    fn = d.get("func_name", "?")
    opt = d.get("opt_level", "?")
    sf = d.get("source_file") or ""
    binary = d.get("binary") or ""
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    n_b = len(bl)
    n_s = len(M[0]) if M and M[0] else 0

    nonempty = sum(1 for ls in bl if ls)
    pos_m = sum(
        1
        for i in range(min(n_b, len(M)))
        if M[i] and sum(M[i]) > 1e-9
    )

    all_lines = sorted({ln for ls in bl for ln in (ls or [])})
    snippets = read_source_lines(src_root, sf, all_lines[:20]) if verbose_source else {}

    parts = []
    parts.append(f"  函数: {fn}  |  opt_level={opt}")
    parts.append(f"  binary: {Path(binary).name}  |  source_file: {Path(sf).name if sf else '?'}")
    parts.append(
        f"  BB数={n_b}  语句列={n_s}  "
        f"有DWARF行的BB={nonempty}/{n_b}  M非零行={pos_m}/{n_b}  "
        f"signal_stats={d.get('signal_stats', {})}"
    )
    if nonempty == 0 and n_b > 8:
        parts.append("  ⚠ 该函数几乎所有 BB 无 DWARF 行号，对齐不可信（常见于静态库符号/错误源文件）。")

    shown = 0
    for i in range(n_b):
        src_ln = bl[i] if i < len(bl) else []
        row_m = M[i] if i < len(M) else []
        row_w = W[i] if i < len(W) else []
        # 长函数：只打印「有 DWARF 行」或「M 非零」的 BB，最多 max_bb_lines 行
        interesting = bool(src_ln) or (row_m and sum(row_m) > 1e-9)
        if n_b > 32:
            if not interesting:
                continue
        if shown >= max_bb_lines:
            parts.append(f"    ... 其余 BB 省略（已显示 {shown} 行，共 {n_b} 个 BB）")
            break
        parts.append(_bb_row_summary(i, src_ln, row_m, row_w))
        shown += 1
        if verbose_source and src_ln and snippets:
            for ln in src_ln[:3]:
                if ln in snippets:
                    parts.append(f"         L{ln}: {snippets[ln]}")

    if n_b > 32 and shown == 0:
        parts.append("  (无「有行号或非零M」的 BB，未逐行展开)")

    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="zlib 每档 5 样本对齐审计")
    ap.add_argument("--data-dir", type=Path, help="目录下含 binskel_zlib_o0.jsonl … o3")
    ap.add_argument("--o0", type=Path)
    ap.add_argument("--o1", type=Path)
    ap.add_argument("--o2", type=Path)
    ap.add_argument("--o3", type=Path)
    ap.add_argument(
        "--zlib-src",
        type=Path,
        help="zlib 源码根目录（用于打印 DWARF 行对应源码，可选）",
    )
    ap.add_argument("--no-source", action="store_true", help="不尝试打印源码行")
    ap.add_argument(
        "--funcs",
        type=str,
        default="",
        help="逗号分隔函数名；若指定则每档按函数名各取一条（便于 O0–O3 横向对比），忽略默认 5 索引抽样",
    )
    args = ap.parse_args()

    paths: dict[str, Path | None] = {"O0": None, "O1": None, "O2": None, "O3": None}
    if args.data_dir:
        dd = args.data_dir.expanduser()
        for k, name in [(0, "O0"), (1, "O1"), (2, "O2"), (3, "O3")]:
            p = dd / f"binskel_zlib_o{k}.jsonl"
            paths[name] = p if p.is_file() else None
    for k, p in [("O0", args.o0), ("O1", args.o1), ("O2", args.o2), ("O3", args.o3)]:
        if p is not None:
            paths[k] = p.expanduser()

    src_root = args.zlib_src.expanduser() if args.zlib_src else None
    verbose_src = bool(src_root) and not args.no_source

    print("=" * 72)
    print("zlib 对齐审计：每优化等级 5 条样本（索引: 0, n//4, n//2, 3n//4, n-1）")
    print("=" * 72)

    any_ok = False
    for label in ("O0", "O1", "O2", "O3"):
        p = paths[label]
        if not p or not p.is_file():
            print(f"\n### {label}: 跳过（未找到 jsonl: {p}）\n")
            continue
        any_ok = True
        rows = load_jsonl(p)
        n = len(rows)
        func_names = [x.strip() for x in args.funcs.split(",") if x.strip()]
        if func_names:
            idxs = []
            name_hits = []
            for name in func_names:
                hit = None
                for ii, r in enumerate(rows):
                    if r.get("func_name") == name:
                        hit = ii
                        break
                if hit is not None:
                    idxs.append(hit)
                    name_hits.append(name)
                else:
                    print(f"  [警告] {label} 中未找到函数: {name}")
            if not idxs:
                print(f"  {label}: 无匹配函数，跳过本档")
                continue
        else:
            idxs = pick_five_indices(n)
            name_hits = None

        print(f"\n{'#' * 72}")
        print(f"## {label}  |  文件: {p}")
        print(f"## 总样本数: {n}  |  抽查: {name_hits or idxs}")
        print(f"{'#' * 72}")

        for j, ii in enumerate(idxs, 1):
            tag = f"函数 {name_hits[j-1]}" if name_hits else f"文件内第 {ii+1} 行"
            print(f"\n--- 样本 {j}/{len(idxs)} ({tag}) ---")
            print(summarize_one(rows[ii], src_root, verbose_src))

    if not any_ok:
        print("错误: 没有找到任何 binskel_zlib_o*.jsonl。", file=sys.stderr)
        print("请使用 --data-dir 或 --o0/--o1/--o2/--o3 指定路径。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印「audit_bundle_bb_heuristic_check 高置信冲突」那几条 BB 在 jsonl 里的主对齐置信度。

主对齐列：对 alignment_matrix[bb] 取 argmax 得到 stmt 列 j*，
置信度取 alignment_confidence[bb][j*]（若缺该矩阵则尝试从 block_src_confidence 聚合或标 N/A）。

用法（WSL，jsonl 与导出 bundle 时一致）:
  python3 scripts/show_hi_mismatch_bb_confidence.py --jsonl-dir /mnt/e/structlift_datasets

若 jsonl 不在同一目录，可多次指定:
  python3 scripts/show_hi_mismatch_bb_confidence.py \\
    --jsonl /path/binskel_zlib_o0.jsonl --jsonl /path/binskel_sqlite_o0.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 与 audit_bundle 高置信明细一致（jsonl 名, 0-based 行号, BB 下标）
HI_MISMATCH: list[tuple[str, int, int, str]] = [
    ("binskel_zlib_o0.jsonl", 110, 1, "gzputs BB1"),
    ("binskel_zlib_o0.jsonl", 110, 5, "gzputs BB5"),
    ("binskel_sqlite_o0.jsonl", 495, 14, "openDirectory BB14"),
    ("binskel_busybox_o0.jsonl", 2023, 0, "get_geometry BB0"),
    ("binskel_busybox_o0.jsonl", 561, 5, "handle_port BB5"),
]


def load_by_line_idx(path: Path, target: int) -> dict | None:
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


def primary_stmt_and_conf(d: dict, bb: int) -> tuple[int | None, float | None, float | None, str]:
    """
    返回 (stmt_idx, M[bb][j], W[bb][j] 或 None, 备注)
    """
    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    bl = d.get("block_src_lines") or []
    bc = d.get("block_src_confidence") or []

    if bb < 0 or bb >= len(M) or not M[bb]:
        return None, None, None, "无 alignment_matrix 或 BB 越界"
    row = M[bb]
    j = max(range(len(row)), key=lambda k: row[k])
    mval = row[j]
    if mval <= 0:
        return j, mval, None, "该行矩阵全零或主列为0"

    w_ij = None
    if bb < len(W) and j < len(W[bb]):
        w_ij = W[bb][j]

    note = ""
    if w_ij is None and bb < len(bc) and isinstance(bc[bb], list) and bc[bb]:
        # 无 W 时：用 block 行级置信度平均作参考（口径不同于 W）
        vals = [float(x) for x in bc[bb] if x is not None]
        if vals:
            w_ij = sum(vals) / len(vals)
            note = "（无 alignment_confidence，用 block_src_confidence 均值近似）"

    return j, float(mval), float(w_ij) if w_ij is not None else None, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="在该目录下按文件名查找 jsonl",
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        default=[],
        help="直接指定 jsonl 路径，可重复",
    )
    args = ap.parse_args()

    dirs: list[Path] = []
    if args.jsonl_dir:
        dirs.append(args.jsonl_dir.expanduser().resolve())

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("主对齐置信度：alignment_confidence[BB][主 stmt 列]，主 stmt 列 = argmax(alignment_matrix[BB])")
    print()

    for jname, line_idx, bb, tag in HI_MISMATCH:
        path = None
        for p in args.jsonl:
            if p.name == jname or str(p).endswith(jname):
                path = p.expanduser().resolve()
                if path.is_file():
                    break
        if path is None or not path.is_file():
            for d in dirs:
                cand = d / jname
                if cand.is_file():
                    path = cand
                    break
        if path is None or not path.is_file():
            print(f"[跳过] {tag}  未找到 {jname}")
            continue

        d = load_by_line_idx(path, line_idx)
        if d is None:
            print(f"[跳过] {tag}  {path} 无 line_idx={line_idx}")
            continue

        fn = d.get("func_name", "?")
        j, mval, wval, note = primary_stmt_and_conf(d, bb)
        w_s = f"{wval:.6g}" if wval is not None else "N/A"
        print(f"{tag}  func={fn}  file={path.name} line_idx={line_idx}")
        print(f"  BB{bb}  primary_stmt_idx={j}  matrix_value={mval}  confidence={w_s}  {note}")
        bl = d.get("block_src_lines") or []
        if bb < len(bl):
            ln = bl[bb]
            print(f"  block_src_lines[{bb}]={ln[:8]}{'...' if len(ln) > 8 else ''}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对固定 35 条审计样本做自动化「语义对齐」启发式判定（无需打开 C 文件）。

依据 jsonl 内字段：
  - block_src_lines vs alignment_matrix（与 deep_validate 一致）
  - source_lines 列号对应的物理行号是否落在合理区间
  - 主对齐列 j 对应的 source_lines[j] 是否与 block_src_lines 行块接近（DWARF 常多行）

用法:
  python scripts/audit_35_samples_semantics.py [--datasets DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deep_validate import analyze_sample  # noqa: E402

# 与用户导出汇总一致: (jsonl 文件名, line_idx)
SAMPLES: list[tuple[str, int]] = []
for idx in [110, 135, 50, 119, 95]:
    SAMPLES.append(("binskel_zlib_o0.jsonl", idx))
for idx in [105, 266, 185, 186, 209]:
    SAMPLES.append(("binskel_curl_o0.jsonl", idx))
for idx in [1609, 1834, 495, 425, 2356]:
    SAMPLES.append(("binskel_sqlite_o0.jsonl", idx))
for idx in [2528, 4057, 2023, 561, 2574]:
    SAMPLES.append(("binskel_busybox_o0.jsonl", idx))
for idx in [10632, 7832, 7292, 864, 1382]:
    SAMPLES.append(("binskel_openssl_o0.jsonl", idx))
for idx in [6, 58, 60, 52, 47]:
    SAMPLES.append(("binskel_libxml2_o0.jsonl", idx))
for idx in [840, 311, 12646, 12645, 7061]:
    SAMPLES.append(("binskel_ffmpeg_o0.jsonl", idx))


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


def semantic_heuristic(d: dict) -> tuple[str, list[str]]:
    """
    返回 (等级, 说明列表) 等级: OK | WARN | FAIL
    """
    notes: list[str] = []
    ar = analyze_sample(d)
    M = d.get("alignment_matrix") or []
    block_lines = d.get("block_src_lines") or []
    source_lines = d.get("source_lines") or []
    n_b = ar["n_blocks"]
    n_s = ar["n_stmts"]
    bb_primary: list[int] = ar.get("bb_primary") or []

    if not M or not M[0]:
        return "FAIL", ["无 alignment_matrix"]

    if not ar["matrix_lines_consistent"]:
        return "FAIL", ["matrix_vs_block_lines: 有 DWARF 行但矩阵全 0 的行"]

    sl_nums = [int(x) for x in source_lines if isinstance(x, (int, float))]
    if not sl_nums:
        return "WARN", ["source_lines 为空，无法做行号范围检查"]

    lo, hi = min(sl_nums), max(sl_nums)
    span = hi - lo + 1

    # --- FAIL: block_src_lines 出现远离语句列行号的「飞线」---
    slack = max(80, span // 4)
    bad_lines: list[tuple[int, int]] = []
    for bi, lines in enumerate(block_lines):
        if not isinstance(lines, list):
            continue
        for L in lines:
            if not isinstance(L, (int, float)):
                continue
            li = int(L)
            if li < lo - slack or li > hi + slack:
                bad_lines.append((bi, li))
    if bad_lines:
        notes.append(f"飞线: {len(bad_lines)} 个 block 行号落在语句行区间外(>{slack}): 例 {bad_lines[:5]}")
        return "FAIL", notes

    # --- 主对齐列对应行号 vs 该 BB 的 DWARF 行块（软匹配）---
    margin = 25
    mism = 0
    checked = 0
    for bi in range(min(n_b, len(block_lines), len(bb_primary))):
        j = bb_primary[bi]
        if j < 0:
            continue
        bl = block_lines[bi] if bi < len(block_lines) else []
        if not bl:
            continue
        checked += 1
        if j >= len(source_lines):
            continue
        Lp = source_lines[j]
        if not isinstance(Lp, (int, float)):
            continue
        lp = int(Lp)
        bmin, bmax = min(int(x) for x in bl if isinstance(x, (int, float))), max(
            int(x) for x in bl if isinstance(x, (int, float))
        )
        if not (bmin - margin <= lp <= bmax + margin):
            mism += 1
    if checked > 0 and mism / checked > 0.35:
        notes.append(f"主对齐行与 DWARF 行块偏离: {mism}/{checked} BB (>35%)")

    # --- WARN 条件 ---
    if ar["empty_ratio"] > 0.2:
        notes.append(f"空对齐 BB 比例高: {ar['empty_ratio']:.1%}")
    if n_s > 0 and n_b / n_s > 12:
        notes.append(f"BB/语句比极大: {n_b}/{n_s}={n_b/n_s:.1f}（监督很粗）")
    if ar["order_consistency"] < 0.82 and n_b > 8:
        notes.append(f"顺序一致性偏低: {ar['order_consistency']:.1%}")

    mean_conf = 0.0
    try:
        from validate_alignment import validate_sample

        vr = validate_sample(d)
        mean_conf = float(vr.get("mean_conf", 0))
    except Exception:
        pass
    if mean_conf > 0 and mean_conf < 0.74:
        notes.append(f"validate mean_conf={mean_conf:.3f} 偏低")

    if notes:
        return "WARN", notes
    return "OK", ["矩阵自洽; 行号在语句区间内; 主对齐与 DWARF 行块多数接近"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "structlift_datasets",
        help="含 binskel_*_o0.jsonl 的目录",
    )
    args = ap.parse_args()
    root: Path = args.datasets.expanduser().resolve()
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    ok = warn = fail = 0
    rows: list[str] = []
    for name, li in SAMPLES:
        path = root / name
        if not path.is_file():
            rows.append(f"FAIL\t{name}\tline_idx={li}\t缺失文件")
            fail += 1
            continue
        d = load_by_line_idx(path, li)
        if not d:
            rows.append(f"FAIL\t{name}\tline_idx={li}\t索引不存在")
            fail += 1
            continue
        fn = d.get("func_name", "?")
        level, notes = semantic_heuristic(d)
        if level == "OK":
            ok += 1
        elif level == "WARN":
            warn += 1
        else:
            fail += 1
        note_s = "; ".join(notes)
        rows.append(f"{level}\t{name}\tline_idx={li}\t{fn}\t{note_s}")

    print("等级\tjsonl\t索引\tfunc_name\t说明")
    for r in rows:
        print(r)
    print()
    print(f"统计: OK={ok}  WARN={warn}  FAIL={fail}  计={len(SAMPLES)}")
    print()
    print(
        "说明: FAIL=行号飞出语句区间或矩阵/DWARF 自洽失败; "
        "WARN=启发式风险(空行多、BB/stmt 极大、顺序或 conf 偏低等); "
        "OK=未触发上述规则（仍非人工逐条反汇编证明）。"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
默认项目 × 4 优化档每个文件均匀抽 N 条，检查是否满足 StrucLift 数据集大致要求。

检查项（单条样本）：
  - Module A/B 核心字段是否存在且非空、形状与数量关系是否合理
  - alignment_matrix / alignment_confidence 与 BB、语句数一致
  - deep_validate: 矩阵与 block_src_lines 自洽
  - validate_alignment: coverage / mean_conf（仅作参考，标 WARN）

用法:
  python3 scripts/spotcheck_binskel_dataset_requirements.py --all-default-paths
  python3 scripts/spotcheck_binskel_dataset_requirements.py --out-dir ~/structlift_datasets --num-samples 5
  python3 scripts/spotcheck_binskel_dataset_requirements.py --projects zlib,curl --num-samples 3
"""
from __future__ import annotations

import argparse
import json
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
from deep_validate import analyze_sample  # noqa: E402
from validate_alignment import validate_sample  # noqa: E402


def _pick_spread_indices(n_valid: int, want: int) -> list[int]:
    if n_valid <= 0 or want <= 0:
        return []
    m = min(want, n_valid)
    if m == 1:
        return [0]
    raw = [int(round(i * (n_valid - 1) / (m - 1))) for i in range(m)]
    out: list[int] = []
    for x in raw:
        x = min(max(x, 0), n_valid - 1)
        if not out or out[-1] != x:
            out.append(x)
    if len(out) < m:
        used = set(out)
        for j in range(n_valid):
            if j not in used:
                out.append(j)
                used.add(j)
                if len(out) >= m:
                    break
        out.sort()
    return out[:m]


def _count_valid_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
    return n


def _load_samples_at_indices(path: Path, indices: list[int]) -> list[tuple[int, dict | None]]:
    idx_needed = sorted(set(indices))
    got: dict[int, dict | None] = {}
    cur = -1
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cur += 1
            if cur not in idx_needed:
                continue
            got[cur] = d
            if len(got) == len(idx_needed):
                break
    return [(i, got.get(i)) for i in indices]


def _check_sample(d: dict, line_idx: int) -> tuple[str, list[str]]:
    """返回 (PASS|WARN|FAIL, 说明列表)。"""
    fail: list[str] = []
    warn: list[str] = []

    def req_list(name: str, min_len: int = 1) -> list | None:
        v = d.get(name)
        if not isinstance(v, list) or len(v) < min_len:
            fail.append(f"缺或空列表: {name}")
            return None
        return v

    req_list("bin_opcodes")
    req_list("bin_operand_types")
    req_list("bin_operand_values")
    bl = req_list("bin_block_lengths")
    bei = d.get("bin_edge_index")
    if not (isinstance(bei, list) and len(bei) == 2):
        fail.append("bin_edge_index 应为 [src_list, dst_list]")
    elif len(bei[0]) == 0:
        warn.append("bin_edge_index 无边(单块函数可接受)")

    sf = req_list("bin_struct_features")
    if sf and bl:
        if len(sf) != len(bl):
            fail.append(f"bin_struct_features 行数 {len(sf)} != n_bb {len(bl)}")
        elif sf and isinstance(sf[0], list) and len(sf[0]) != 12:
            fail.append("bin_struct_features 每行应为 12 维")

    op = d.get("bin_opcodes")
    if bl and op and isinstance(op, list) and sum(bl) != len(op):
        fail.append(f"sum(bin_block_lengths)={sum(bl)} != len(bin_opcodes)={len(op)}")

    tok = req_list("src_token_ids")
    sl = req_list("src_stmt_lengths")
    if tok and sl:
        if sum(sl) != len(tok):
            fail.append(f"sum(src_stmt_lengths)={sum(sl)} != len(src_token_ids)={len(tok)}")
    ast = req_list("src_ast_types", min_len=0)
    if sl and ast is not None and len(ast) != len(sl):
        fail.append(f"len(src_ast_types)={len(ast)} != n_stmt={len(sl)}")
    sci = d.get("src_compound_id")
    if sci is None:
        warn.append("缺少 src_compound_id（旧版 jsonl；重建数据集可消除）")
    elif sl and isinstance(sci, list) and len(sci) != len(sl):
        fail.append(f"len(src_compound_id)={len(sci)} != n_stmt={len(sl)}")

    sei = d.get("src_edge_index")
    if isinstance(sei, list) and len(sei) == 2 and len(sei[0]) == 0:
        warn.append("src_edge_index 为空(部分构建允许)")

    bsl = d.get("block_src_lines")
    if not isinstance(bsl, list):
        fail.append("block_src_lines 缺失或非列表")
    elif bl and len(bsl) != len(bl):
        fail.append(f"len(block_src_lines)={len(bsl)} != n_bb={len(bl)}")

    M = d.get("alignment_matrix")
    W = d.get("alignment_confidence")
    if not M or not isinstance(M, list) or not M:
        fail.append("alignment_matrix 缺失或空")
    elif not M[0]:
        fail.append("alignment_matrix 宽为 0")
    else:
        n_s = len(M[0])
        if bl and len(M) != len(bl):
            fail.append(f"alignment_matrix 行数 {len(M)} != n_bb {len(bl)}")
        if sl and n_s != len(sl):
            fail.append(f"alignment_matrix 列数 {n_s} != n_stmt {len(sl)}")
        if not any(any(abs(float(x)) > 1e-12 for x in row) for row in M):
            warn.append("alignment_matrix 全零(对齐监督无效)")

    if W is not None and isinstance(W, list) and M:
        if len(W) != len(M) or (W and len(W[0]) != len(M[0])):
            fail.append("alignment_confidence 形状与 alignment_matrix 不一致")

    pl = d.get("pattern_labels")
    if not isinstance(pl, list):
        fail.append("pattern_labels 缺失或非列表")
    elif bl and len(pl) != len(bl):
        fail.append(f"len(pattern_labels)={len(pl)} != n_bb={len(bl)}")

    ar = analyze_sample(d)
    if not ar.get("matrix_lines_consistent", True):
        fail.append("矩阵与 block_src_lines 不自洽(有行无对齐)")

    vr = validate_sample(d)
    if vr.get("coverage", 0) <= 0:
        warn.append(f"coverage=0 无可对齐BB (line_idx={line_idx})")
    if vr.get("mean_conf", 0) <= 0 and vr.get("coverage", 0) > 0:
        warn.append("mean_conf=0 但 coverage>0 需核对置信矩阵")

    if fail:
        return "FAIL", fail
    if warn:
        return "WARN", warn
    return "PASS", []


def main() -> None:
    ap = argparse.ArgumentParser(description="28 档 binskel 抽样检查数据集字段与自洽性")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "structlift_datasets")
    ap.add_argument("--search-dir", action="append", default=[], metavar="DIR")
    ap.add_argument("--all-default-paths", action="store_true")
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS))
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="仅打印汇总与 FAIL 明细")
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

    print("搜索路径:", search_dirs)
    print(f"每文件抽样数: {args.num_samples}")
    print()

    total_files = 0
    missing_files = 0
    total_samples = 0
    n_pass = n_warn = n_fail = 0
    fail_lines: list[str] = []

    for proj in projects:
        for k in range(4):
            name = f"binskel_{proj}_o{k}.jsonl"
            path, root = locate_jsonl(name, search_dirs)
            if not path:
                missing_files += 1
                if not args.quiet:
                    print(f"[缺失] {name}")
                continue
            total_files += 1
            n_valid = _count_valid_lines(path)
            if n_valid == 0:
                if not args.quiet:
                    print(f"[空文件] {name} @ {root}")
                continue
            idxs = _pick_spread_indices(n_valid, args.num_samples)
            pairs = _load_samples_at_indices(path, idxs)
            if not args.quiet:
                print(f"--- {name} (有效行 {n_valid}) @ {root} ---")
            for idx, d in pairs:
                total_samples += 1
                if d is None:
                    st = "FAIL"
                    msgs = [f"行下标 {idx} 未读到"]
                    n_fail += 1
                else:
                    st, msgs = _check_sample(d, idx)
                    if st == "PASS":
                        n_pass += 1
                    elif st == "WARN":
                        n_warn += 1
                    else:
                        n_fail += 1
                fn = d.get("func_name", "?")[:48] if d else "?"
                line = f"  [{st}] idx={idx} func={fn!r}"
                if msgs:
                    line += " | " + "; ".join(msgs)
                if st == "FAIL" or (st == "WARN" and not args.quiet):
                    print(line)
                elif not args.quiet and st == "PASS":
                    print(line)
                if st == "FAIL":
                    fail_lines.append(f"{name} idx={idx} {fn}: {'; '.join(msgs)}")

            if not args.quiet:
                print()

    print("========== 汇总 ==========")
    print(f"jsonl 存在: {total_files} / {len(projects) * 4}  (缺失 {missing_files} 个文件名)")
    print(f"抽样条数合计: {total_samples}")
    print(f"PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}")
    if fail_lines:
        print("\nFAIL 列表:")
        for x in fail_lines:
            print(" ", x)
    print(
        "\n说明: PASS=字段与形状满足大致要求且矩阵行自洽; WARN=仍可用于训练但建议加权或排查;"
        " FAIL=不满足基本要求或自洽失败。"
    )


if __name__ == "__main__":
    main()

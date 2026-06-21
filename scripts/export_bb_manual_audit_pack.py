#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 binskel 一条样本中导出若干基本块的「人工语义对齐」核对包：
  DWARF 行号、矩阵主峰列、该列对应的 tree-sitter 语句（行范围+源码片段）、该 BB 的反汇编。

用法:
  python3 scripts/export_bb_manual_audit_pack.py \\
    binskel_zlib_o0.jsonl --line-idx 1 --num-bb 10 \\
    --src-root E:/zlib-1.3.1 \\
    -o manual_bb_audit_pack.txt

若 json 中 binary 路径在本机无效，可:
  --elf-path /path/to/libz.so  （与构建时一致）

不写 -o 则打印到 stdout。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deep_validate import analyze_sample  # noqa: E402


def _spread_indices(n_bb: int, k: int) -> list[int]:
    if n_bb <= 0 or k <= 0:
        return []
    if n_bb <= k:
        return list(range(n_bb))
    return sorted({int((n_bb - 1) * i / (k - 1)) for i in range(k)})


def _disasm_blocks(elf_path: Path, func_name: str):
    sys.path.insert(0, str(REPO_ROOT))
    from elftools.elf.elffile import ELFFile
    from export_sample_for_manual_check import disasm_function

    if not elf_path.is_file():
        return []
    return disasm_function(str(elf_path), func_name)


def _peak_cols(row: list, topk: int = 3) -> list[tuple[int, float]]:
    pairs = [(j, float(row[j])) for j in range(len(row)) if float(row[j]) > 1e-9]
    pairs.sort(key=lambda x: -x[1])
    return pairs[:topk]


# 与 build_binskel_dataset_md.ast_type_map 一致（jsonl 里 src_ast_types）
_AST_TYPE_ID_NAME: dict[int, str] = {
    2: "if_statement",
    3: "while_statement",
    4: "for_statement",
    5: "do_statement",
    6: "switch_statement",
    7: "case_statement",
    8: "return_statement",
    9: "break_statement",
    10: "continue_statement",
    11: "goto_statement",
    12: "compound_statement",
    15: "other_or_lexical",
}


def _ast_type_caption(tid: int | str) -> str:
    if tid == "?":
        return "?"
    try:
        i = int(tid)
    except (TypeError, ValueError):
        return str(tid)
    return _AST_TYPE_ID_NAME.get(i, f"type_id={i}")


def _emit_source_lines_capped(
    src_text_lines: list[str], lo: int, hi: int, max_lines: Optional[int]
) -> list[str]:
    """按行号区间输出源码；max_lines 为 None 时不截断。"""
    out: list[str] = []
    if lo > hi or not src_text_lines:
        return out
    n = hi - lo + 1
    if max_lines is None or n <= max_lines:
        for ln in range(lo, hi + 1):
            if 1 <= ln <= len(src_text_lines):
                out.append(f"  {ln:5d} | {src_text_lines[ln - 1].rstrip()}")
        return out
    head = max_lines // 2
    tail = max_lines - head
    for ln in range(lo, lo + head):
        if 1 <= ln <= len(src_text_lines):
            out.append(f"  {ln:5d} | {src_text_lines[ln - 1].rstrip()}")
    out.append(f"  ... ({n - max_lines} 行省略) ...")
    for ln in range(hi - tail + 1, hi + 1):
        if 1 <= ln <= len(src_text_lines):
            out.append(f"  {ln:5d} | {src_text_lines[ln - 1].rstrip()}")
    return out


def _stmt_one_liner(src_text_lines: list[str], lo: int, hi: int, cap: int = 200) -> str:
    parts: list[str] = []
    for ln in range(lo, hi + 1):
        if 1 <= ln <= len(src_text_lines):
            parts.append(src_text_lines[ln - 1].strip())
    s = " ".join(parts).replace("\t", " ")
    if len(s) > cap:
        return s[: cap - 3] + "..."
    return s


def build_pack(
    d: dict,
    src_root: Path | None,
    elf_path: Path | None,
    num_bb: int,
    *,
    all_blocks: bool = False,
    skip_disasm: bool = False,
    omit_matrix_column_table: bool = False,
    max_src_lines_bb_jstar: Optional[int] = None,
    max_matrix_cols: Optional[int] = None,
) -> str:
    fn = d.get("func_name", "?")
    sf = d.get("source_file") or ""
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    ast_types = d.get("src_ast_types") or []

    from check_alignment_semantics import (  # noqa: E402
        _filter_stmts_for_func,
        _parse_file,
        infer_col_to_stmt,
    )

    rel = Path(sf).name if sf else ""
    src_path = None
    if src_root and rel:
        cand = src_root / rel
        if cand.is_file():
            src_path = cand
        else:
            for hit in src_root.rglob(rel):
                if hit.is_file():
                    src_path = hit
                    break

    filtered: list = []
    col_stmts: list = []
    col_scores: list = []
    stmt_line_hint = ""
    if src_path and src_path.is_file() and M and bl:
        try:
            statements, _ = _parse_file(src_path)
            all_lns = [ln for ls in bl for ln in (ls or [])]
            if all_lns:
                fmin, fmax = min(all_lns), max(all_lns)
                filtered = _filter_stmts_for_func(statements, fmin, fmax)
                col_stmts, col_scores = infer_col_to_stmt(
                    bl, M, filtered, thr=0.005
                )
                stmt_line_hint = str(src_path)
        except Exception as e:
            stmt_line_hint = f"(解析失败: {e})"
    else:
        stmt_line_hint = f"(无源码: src_root={src_root}, file={rel})"

    bin_raw = d.get("binary") or ""
    ep = elf_path
    if ep is None and bin_raw:
        ep = Path(str(bin_raw)).expanduser()
    bb_asm: list = []
    if not skip_disasm and ep and ep.is_file():
        bb_asm = _disasm_blocks(ep, fn)
    asm_by_id = {b_id: insns for b_id, insns in bb_asm}

    ar = analyze_sample(d)
    bb_primary: list = list(ar.get("bb_primary") or [])

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("人工核对：基本块 ↔ 源码语句（单条样本）")
    lines.append("=" * 72)
    lines.append("")
    lines.append("【你怎么判「对齐对不对」——建议顺序】")
    lines.append("1) 读该 BB 的「反汇编」：主要在干什么（call / load-store / 分支 / ret）？")
    lines.append("2) 看「DWARF 行号」(block_src_lines)：编译器标了哪些源码行落在这个 BB？")
    lines.append("3) 看「主峰列 j*」对应的「语句」行范围与源码片段：数据里说「主对齐」到 tree-sitter 的第 j 条语句。")
    lines.append("4) 结论（自己打勾）：")
    lines.append("   · 合理：DWARF 行与 j* 语句在位置/语义上能解释；汇编与这句 C 无明显矛盾。")
    lines.append("   · 可疑：汇编像在干别的，或 j* 跑到离 DWARF 很远、无关的语句。")
    lines.append("   · 错：主峰语句与 DWARF 或汇编明显不一致。")
    lines.append("")
    lines.append("注意：优化(O1+)内联/重排会让「一句」对应多块；以「能否用源码解释这块码」为准。")
    lines.append("")
    lines.append(f"函数: {fn}")
    lines.append(f"源文件(记录): {sf}")
    lines.append(f"源码解析: {stmt_line_hint}")
    lines.append(f"ELF: {ep if ep else bin_raw} (exists={ep.is_file() if ep else False})")
    if all_blocks:
        idxs = list(range(len(bl)))
        bb_desc = f"全部 {len(idxs)} 个基本块（无抽样）"
    else:
        idxs = _spread_indices(len(bl), min(num_bb, len(bl)))
        bb_desc = f"{len(idxs)} 个（从共 {len(bl)} 里均匀抽样）"
    lines.append(f"本包导出 BB: {bb_desc}")
    lines.append("")

    src_text_lines: list[str] = []
    if src_path and src_path.is_file():
        src_text_lines = src_path.read_text(encoding="utf-8", errors="replace").splitlines()

    n_col = len(M[0]) if M and M[0] else 0
    if n_col > 0 and omit_matrix_column_table:
        lines.append("")
        lines.append(
            f"（已省略「矩阵列 j↔语句」全表，共 {n_col} 列；见下方各 BB 的 j* 行范围与截断源码）"
        )

    col_limit = n_col if max_matrix_cols is None else min(n_col, max_matrix_cols)
    if n_col > 0 and not omit_matrix_column_table:
        lines.append("")
        lines.append("#" * 72)
        lines.append(
            "## 矩阵列 j ↔ tree-sitter 语句（列下标 = alignment_matrix 列 = jsonl 中第 j 条语句）"
        )
        lines.append("#" * 72)
        lines.append(
            "说明：下列「ts_type」为 tree-sitter 结点类型；ast_type_id 为数据集中的类别编号。"
        )
        lines.append(
            "「列支撑映射」来自 infer_col_to_stmt：列 j 与 DWARF 行交集最大的那条语句（可能与第 j 条略有差别）。"
        )
        if max_matrix_cols is not None and col_limit < n_col:
            lines.append(f"（仅列出前 {col_limit}/{n_col} 列，其余省略以控制体积）")
        lines.append("")
        for j in range(col_limit):
            st_j = filtered[j] if j < len(filtered) else None
            st_inf = col_stmts[j] if j < len(col_stmts) else None
            tid = ast_types[j] if j < len(ast_types) else "?"
            cap = _ast_type_caption(tid)
            sc = col_scores[j] if j < len(col_scores) else -1
            if st_j:
                lo, hi = st_j["start_line"], st_j["end_line"]
                tst = st_j.get("type", "?")
                snippet = _stmt_one_liner(src_text_lines, lo, hi)
                line0 = (
                    f"  j={j:3d}  ast_type_id={tid} ({cap})  ts_type={tst}  "
                    f"L{lo}–L{hi}  col_support_score={sc}"
                )
            elif st_inf:
                lo, hi = st_inf["start_line"], st_inf["end_line"]
                tst = st_inf.get("type", "?")
                snippet = _stmt_one_liner(src_text_lines, lo, hi)
                line0 = (
                    f"  j={j:3d}  ast_type_id={tid} ({cap})  ts_type={tst}  "
                    f"L{lo}–L{hi}  (仅映射) col_support_score={sc}"
                )
            else:
                line0 = f"  j={j:3d}  ast_type_id={tid} ({cap})  (无语句结点)"
                snippet = ""
            lines.append(line0)
            if snippet:
                lines.append(f"       摘录: {snippet}")
            if st_j and st_inf and (
                st_j["start_line"] != st_inf["start_line"]
                or st_j["end_line"] != st_inf["end_line"]
            ):
                ilo, ihi = st_inf["start_line"], st_inf["end_line"]
                lines.append(
                    f"       infer_col_to_stmt 映射到: L{ilo}–L{ihi} ({st_inf.get('type', '?')})"
                )
        if max_matrix_cols is not None and col_limit < n_col:
            lines.append(f"  ... 省略列 j={col_limit}..{n_col - 1}（共 {n_col - col_limit} 列）")
        lines.append("")

    for rank, i in enumerate(idxs):
        lines.append("")
        lines.append("#" * 72)
        lines.append(f"## BB {i}  （第 {rank+1}/{len(idxs)} 个）")
        lines.append("#" * 72)
        D = bl[i] if i < len(bl) else []
        row = M[i] if i < len(M) else []
        j_from_analyze = bb_primary[i] if i < len(bb_primary) else -1
        j_star = max(range(len(row)), key=lambda j: row[j]) if row else -1
        tops = _peak_cols(row, 5)

        lines.append(f"block_src_lines (DWARF): {D}")
        lines.append(f"bin_block_lengths[i]: {(d.get('bin_block_lengths') or [])[i] if i < len(d.get('bin_block_lengths') or []) else '-'}")
        lines.append(f"主峰列 j* (argmax M): {j_star}  (analyze_sample bb_primary: {j_from_analyze})")
        lines.append(f"M 列 top 权重: {tops}")

        if j_star >= 0 and j_star < len(col_stmts) and col_stmts[j_star]:
            st = col_stmts[j_star]
            lo, hi = st["start_line"], st["end_line"]
            ast_t = ast_types[j_star] if j_star < len(ast_types) else "?"
            tst = st.get("type", "?")
            lines.append(
                f"语句 j* (=矩阵第{j_star}列): ts_type={tst}  "
                f"ast_type_id={ast_t} ({_ast_type_caption(ast_t)})"
            )
            ref = (
                f"详见上文「矩阵列 j ↔ tree-sitter」表中 j={j_star}"
                if not omit_matrix_column_table
                else f"j*={j_star}（全表已省略）"
            )
            lines.append(f"行范围: L{lo}–L{hi}（{ref}）")
            lines.append("源码片段:")
            lines.extend(
                _emit_source_lines_capped(
                    src_text_lines, lo, hi, max_src_lines_bb_jstar
                )
            )
        elif j_star >= 0:
            lines.append(f"语句 j*={j_star}: (未能映射到带行号的语句；检查源码路径/解析)")

        lines.append("反汇编:")
        if skip_disasm:
            lines.append("  (已跳过反汇编以加速导出)")
            insns = None
        else:
            insns = asm_by_id.get(i)
        if insns:
            for addr, mnem, op in insns:
                lines.append(f"  {addr}  {mnem:8s} {op}")
        else:
            lines.append("  (无：ELF 打不开或符号不对；请用 --elf-path 指向构建产物)")

        lines.append("")
        lines.append("—— 留白：你的结论 [ 合理 / 可疑 / 错 ]  备注: _______________")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--line-idx", type=int, required=True, help="jsonl 内「有效 JSON 行」的 0-based 索引")
    ap.add_argument("--num-bb", type=int, default=10)
    ap.add_argument(
        "--all-bb",
        action="store_true",
        help="导出该函数全部基本块（忽略均匀抽样）",
    )
    ap.add_argument("--src-root", type=Path, default=None)
    ap.add_argument("--elf-path", type=Path, default=None)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    jp = args.jsonl.expanduser()
    d = None
    idx = -1
    with jp.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            idx += 1
            if idx == args.line_idx:
                d = json.loads(line)
                break

    if not d:
        print("未找到 line_idx", args.line_idx, file=sys.stderr)
        return 1

    src_root = args.src_root.expanduser().resolve() if args.src_root else None
    elf = args.elf_path.expanduser().resolve() if args.elf_path else None

    text = build_pack(
        d,
        src_root,
        elf,
        args.num_bb,
        all_blocks=args.all_bb,
    )
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"已写 {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

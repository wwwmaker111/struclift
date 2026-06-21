#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「基本块 ↔ tree-sitter 语句槽」语义对齐 Markdown 对照表（论文/附录用）

列：BB | 汇编在干什么（启发式中文） | j*（树 sitter 槽） | M[j*] | W[j*] | 对不对（启发式）

说明：
  - M/W 来自 jsonl；j* 为 argmax M[i]。
  - 「汇编在干什么」由反汇编指令关键词规则生成，**需人工扫一眼**。
  - 「对不对」为 **启发式**（DWARF 行与主峰语句行是否相交、是否双峰等），
    **不能**替代你逐块读汇编后的判断；可整列替换为「待人工」。

用法（仓库根）:
  python scripts/export_bb_semantic_alignment_table_md.py \\
    E:/structlift_datasets/binskel_zlib_o0.jsonl --line-idx 1 \\
    --src-root E:/zlib-1.3.1 \\
    --bin-root E:/structlift_src_datasets/zlib \\
    -o gz_uncompress_table.md

  # 默认最后一列「对不对」为 **待人工**；加 --auto-verdict 启用 DWARF∩语句行 启发式
  python scripts/export_bb_semantic_alignment_table_md.py ... --all-bb --auto-verdict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_bb_manual_audit_pack import (  # noqa: E402
    _ast_type_caption,
    _disasm_blocks,
    _peak_cols,
    _spread_indices,
)
from check_alignment_semantics import (  # noqa: E402
    _filter_stmts_for_func,
    _parse_file,
    infer_col_to_stmt,
    resolve_binary_path,
)


def _summarize_asm_cn(insns: list | None) -> str:
    """反汇编 → 一行中文简述（启发式）。"""
    if not insns:
        return "（无反汇编）"
    mnems = [str(x[1]).lower() for x in insns[:24]]
    txt = " ".join(mnems)
    calls = [x[2] for x in insns if str(x[1]).lower() == "call"]
    call_hint = ""
    if calls:
        op0 = calls[0].strip()
        for key, lab in (
            ("gzread", "gzread"),
            ("gzclose", "gzclose"),
            ("fwrite", "fwrite"),
            ("fclose", "fclose"),
            ("error", "error"),
        ):
            if key in op0.lower():
                call_hint = key
                break
        if not call_hint and op0:
            call_hint = op0[:40]

    if "ret" in mnems[-3:]:
        tail = "返回"
    else:
        tail = ""
    if mnems[:3] in (["push", "mov", "sub"], ["push", "push", "mov"]):
        head = "序言/栈帧"
    elif "call" in mnems:
        head = f"call（{call_hint or '外部'})"
    elif any(m.startswith("j") for m in mnems):
        jm = next((m for m in mnems if m.startswith("j")), "jmp")
        head = f"分支/{jm}"
    elif "cmp" in mnems or "test" in mnems:
        head = "比较/测试"
    else:
        head = "数据/算术"

    parts = [head]
    if tail:
        parts.append(tail)
    return "，".join(parts) if len(parts) > 1 else parts[0]


def _heuristic_ok(
    dw_lines: list,
    row: list[float],
    col_stmts: list,
    filtered: list,
    *,
    auto: bool,
    simple: bool,
) -> str:
    if not auto:
        return "待人工"
    tops = _peak_cols(row, 3)
    if not dw_lines:
        if any(float(x) > 1e-12 for x in row):
            return "可疑（无 DWARF 行）"
        return "可疑（无 DWARF）"
    if len(tops) >= 2 and tops[1][1] >= 0.32 and tops[0][1] < 0.98:
        return "基本对（双峰）"
    if simple:
        return "对（启发式：DWARF 非空且矩阵有监督）" if any(float(x) > 1e-12 for x in row) else "可疑"
    ls = set(int(x) for x in dw_lines)
    for j, val in enumerate(row):
        if float(val) <= 1e-12:
            continue
        st = None
        if j < len(col_stmts) and col_stmts[j]:
            st = col_stmts[j]
        elif j < len(filtered) and filtered[j]:
            st = filtered[j]
        if st and isinstance(st, dict):
            st_rng = set(range(int(st["start_line"]), int(st["end_line"]) + 1))
            if ls & st_rng:
                return "对"
    return "略勉强（DWARF 与矩阵非零列对应语句行均不相交）"


def build_table_md(
    d: dict,
    src_root: Path | None,
    elf_path: Path | None,
    bin_root: Path | None,
    *,
    all_bb: bool,
    num_bb: int,
    auto_verdict: bool,
    simple_verdict: bool,
    dataset_banner: str | None = None,
    title_suffix: str | None = None,
) -> str:
    fn = d.get("func_name", "?")
    sf = d.get("source_file") or ""
    bl = d.get("block_src_lines") or []
    M = d.get("alignment_matrix") or []
    W = d.get("alignment_confidence") or []
    ast_types = d.get("src_ast_types") or []

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
    if src_path and src_path.is_file() and M and bl:
        try:
            statements, _ = _parse_file(src_path)
            all_lns = [ln for ls in bl for ln in (ls or [])]
            if all_lns:
                fmin, fmax = min(all_lns), max(all_lns)
                filtered = _filter_stmts_for_func(statements, fmin, fmax)
                col_stmts, _ = infer_col_to_stmt(bl, M, filtered, thr=0.005)
        except Exception:
            col_stmts = []

    bin_raw = d.get("binary") or ""
    ep = elf_path
    if ep is None and bin_raw:
        ep = Path(str(bin_raw)).expanduser()
    if (ep is None or not ep.is_file()) and bin_root:
        ep = resolve_binary_path(d, bin_root)
    bb_asm: list = []
    if ep and ep.is_file():
        bb_asm = _disasm_blocks(ep, fn)
    asm_by_id = {b_id: insns for b_id, insns in bb_asm}

    n_bb = len(bl)
    if all_bb:
        idxs = list(range(n_bb))
    else:
        idxs = _spread_indices(n_bb, min(num_bb, n_bb))

    jsonl_name = Path(d.get("_jsonl_path", "sample.jsonl")).name

    lines: list[str] = []
    if dataset_banner:
        lines.append(dataset_banner.rstrip())
        lines.append("")
    sec = "###" if dataset_banner else "##"
    suf = f"，{title_suffix}" if title_suffix else ""
    title = f"1. {fn}（{jsonl_name}{suf}）"
    lines.append(f"{sec} {title}")
    lines.append("")
    vcol = "对不对（启发式）" if auto_verdict else "对不对（待人工填）"
    lines.append(f"| BB | 汇编在干什么（启发式） | j*（树 sitter 槽） | M[j*] | W[j*] | {vcol} |")
    lines.append("|----|--------------------------|---------------------|-------|-------|-------------------|")

    for i in idxs:
        row = M[i] if i < len(M) else []
        wrow = W[i] if i < len(W) else []
        j_star = max(range(len(row)), key=lambda j: row[j]) if row else -1
        m_peak = float(row[j_star]) if j_star >= 0 and j_star < len(row) else 0.0
        w_peak = (
            float(wrow[j_star])
            if j_star >= 0 and j_star < len(wrow)
            else 0.0
        )

        tid = ast_types[j_star] if j_star < len(ast_types) else "?"
        cap = _ast_type_caption(tid)
        st = col_stmts[j_star] if j_star < len(col_stmts) else None
        if st and isinstance(st, dict):
            tst = st.get("type", "?")
            j_cell = f"{j_star}（≈ `{tst}` / {_ast_type_caption(tid)}）"
        else:
            j_cell = f"{j_star}（ast_id={tid} {cap}）"

        insns = asm_by_id.get(i)
        asm_cn = _summarize_asm_cn(insns)
        verdict = _heuristic_ok(
            bl[i] if i < len(bl) else [],
            row,
            col_stmts,
            filtered,
            auto=auto_verdict,
            simple=simple_verdict,
        )

        def esc(s: str) -> str:
            return s.replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {i} | {esc(asm_cn)} | {esc(j_cell)} | {m_peak:.2f} | {w_peak:.3f} | {esc(verdict)} |"
        )

    lines.append("")
    if auto_verdict:
        lines.append(
            "*自动列：无 DWARF 可疑；双峰→基本对；默认相交模式需本机可解析 tree-sitter 源码；"
            "`--simple-verdict` 则仅看 DWARF+矩阵；仍请对照反汇编终判。*"
        )
    else:
        lines.append(
            "*最后一列「待人工」：请对照反汇编与源码，按「对 / 基本对 / 略勉强 / 可疑」填写。*"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--line-idx", type=int, required=True)
    ap.add_argument("--num-bb", type=int, default=13)
    ap.add_argument("--all-bb", action="store_true")
    ap.add_argument("--src-root", type=Path, default=None)
    ap.add_argument("--elf-path", type=Path, default=None)
    ap.add_argument(
        "--bin-root",
        type=Path,
        default=None,
        help="jsonl 内 binary 路径失效时，在此目录按文件名查找 ELF",
    )
    ap.add_argument(
        "--auto-verdict",
        action="store_true",
        help="自动填「对不对」列（否则为「待人工」）",
    )
    ap.add_argument(
        "--simple-verdict",
        action="store_true",
        help="与 --auto-verdict 合用：仅用 DWARF+矩阵，不做语句行相交",
    )
    ap.add_argument(
        "--title-suffix",
        default=None,
        help="标题括号内追加说明（如：与你先前 zlib_0 一致）",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    jp = args.jsonl.expanduser().resolve()
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

    d["_jsonl_path"] = str(jp)
    src_root = args.src_root.expanduser().resolve() if args.src_root else None
    elf_path = args.elf_path.expanduser().resolve() if args.elf_path else None
    bin_root = args.bin_root.expanduser().resolve() if args.bin_root else None

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
        title_suffix=args.title_suffix,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

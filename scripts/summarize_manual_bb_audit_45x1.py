#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 export_45datasets_one_func_full_bb_audit.py 生成的 manual_bb_audit_45x1_full.txt
流式解析每个 DATASET 的全部 BB，为每个函数输出类似论文用的 Markdown 对照表
（BB 摘要 | j* 与语句类型 | M* | W* | 次峰 | 启发式评语）。

W* 默认从同目录 jsonl 按 # path / line_idx 读取 alignment_confidence；缺失则标 「—」。

用法:
  python scripts/summarize_manual_bb_audit_45x1.py \\
    -i E:/structlift_datasets/manual_bb_audit_45x1_full.txt \\
    -o E:/structlift_datasets/manual_bb_audit_45x1_tables.md
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

RE_DATASET = re.compile(r"^# DATASET:\s*(\S+)")
RE_JSONL = re.compile(r"^# jsonl:\s*(.+)$")
RE_PATH = re.compile(r"^# path:\s*(.+)$")
RE_LINE_IDX = re.compile(r"^# line_idx.+?:\s*(-?\d+)")
RE_FUNC = re.compile(r"^# func_name:\s*(.+)$")
RE_N_BB = re.compile(r"^# n_bb:\s*(\d+)")
RE_BB_HEADER = re.compile(r"^## BB\s+(\d+)\s+")
RE_DWARF = re.compile(r"^block_src_lines \(DWARF\):\s*(.+)$")
RE_JSTAR = re.compile(r"^主峰列 j\*:?\s*\(argmax M\):\s*(-?\d+)")
RE_BBPRIMARY = re.compile(r"bb_primary:\s*(-?\d+)")
RE_MTOPS = re.compile(r"^M 列 top 权重:\s*(.+)$")
RE_STMT_RANGE = re.compile(r"语句 j\* 行范围:\s*.+?ast_type_id=(\d+)")
RE_ASM = re.compile(r"^\s+0x[0-9a-f]+\s+(\S+)\s+")

_AST_NAMES: dict[int, str] = {
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


def _ast_caption(tid: int) -> str:
    return _AST_NAMES.get(tid, f"type_id={tid}")


def _norm_jsonl_path(p: str, jsonl_root: Optional[Path]) -> Optional[Path]:
    raw = p.strip()
    cand = Path(raw)
    if cand.is_file():
        return cand.resolve()
    if jsonl_root and not cand.is_absolute():
        c2 = jsonl_root / Path(raw).name
        if c2.is_file():
            return c2.resolve()
    # /mnt/e/... -> E:/...
    if raw.startswith("/mnt/e/") or raw.startswith("/mnt/E/"):
        tail = raw.split("/", 3)[-1] if raw.count("/") >= 3 else raw
        e = Path("E:/") / tail.replace("/", "\\")
        if e.is_file():
            return e.resolve()
        e2 = Path("E:/structlift_datasets") / Path(tail).name
        if e2.is_file():
            return e2.resolve()
    return None


def _load_sample_json(path: Path, line_idx: int) -> Optional[dict]:
    if line_idx < 0 or not path.is_file():
        return None
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
            if idx == line_idx:
                return json.loads(line)
    return None


def _w_peak(W: Any, i: int, j: int, n_col: int) -> tuple[float, float]:
    if not isinstance(W, list) or i >= len(W) or j < 0:
        return (float("nan"), float("nan"))
    row = W[i]
    if not isinstance(row, list):
        return (float("nan"), float("nan"))
    # row might be shorter
    vals = [(jj, float(row[jj])) for jj in range(min(len(row), n_col)) if jj < len(row) and float(row[jj]) > 1e-12]
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort(key=lambda x: -x[1])
    wj = float(row[j]) if j < len(row) else float("nan")
    return (wj, vals[0][1] if vals else float("nan"))


def _parse_mtops(s: str) -> list[tuple[int, float]]:
    s = s.strip()
    try:
        v = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return []
    out: list[tuple[int, float]] = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append((int(x[0]), float(x[1])))
    return out


def _qualitative(mtops: list[tuple[int, float]], j_star: int, j_an: int, dwarf_text: str) -> str:
    dubious = j_an >= 0 and j_star != j_an
    if dwarf_text.strip() in ("[]", ""):
        return "可疑（无 DWARF 行）"
    if not mtops:
        return "可疑（无 M 峰）"
    m0 = mtops[0][1]
    if len(mtops) >= 2 and mtops[1][1] >= 0.25 and m0 < 0.95:
        if mtops[1][1] >= 0.38:
            return "略勉强（双峰，汇编块可能对应多语句）" if dubious else "基本对（双峰）"
        return "基本对（弱次峰）"
    if m0 >= 0.99:
        return "可疑（analyze 与 argmax j 不一致）" if dubious else "对"
    if m0 >= 0.55:
        return "基本对" if not dubious else "可疑"
    return "可疑"


def _bb_role(first_mnem: str, dwarf_nums: str) -> str:
    s = (first_mnem or "").lower()
    d = dwarf_nums[:56]
    if s in ("call",):
        return f"call 区；DWARF {d}"
    if s in ("ret", "jmp", "je", "jne", "ja", "jb", "jl", "jg", "jz", "jnz"):
        return f"控制流/{s}；DWARF {d}"
    if s in ("push", "mov", "lea", "sub", "add"):
        return f"序曲/数据；DWARF {d}"
    return f"{first_mnem or '?'}；DWARF {d}"


def stream_summarize(inp: Path, outp: Path, jsonl_root: Optional[Path]) -> None:
    meta: dict[str, Any] = {}
    cur_bb: Optional[int] = None
    buf: list[str] = []
    datasets: list[dict[str, Any]] = []

    def flush_bb():
        nonlocal cur_bb, buf
        if cur_bb is None:
            buf = []
            return
        text = "\n".join(buf)
        dw = ""
        j_star = -99
        j_an = -99
        mtops: list[tuple[int, float]] = []
        ts_type = "—"
        ast_id_i: Optional[int] = None
        cap = "?"
        first_mnem = ""
        m_stmt = RE_STMT_RANGE.search(text)
        if m_stmt:
            ast_id_i = int(m_stmt.group(1))
            cap = _ast_caption(ast_id_i)
        for ln in text.splitlines():
            m = RE_DWARF.match(ln.strip())
            if m:
                dw = m.group(1).strip()
            m = RE_JSTAR.match(ln.strip())
            if m:
                j_star = int(m.group(1))
            m = RE_MTOPS.match(ln.strip())
            if m:
                mtops = _parse_mtops(m.group(1))
            m = RE_BBPRIMARY.search(ln)
            if m:
                j_an = int(m.group(1))
        in_asm = False
        for ln in text.splitlines():
            if ln.strip() == "反汇编:":
                in_asm = True
                continue
            if in_asm and ln.strip().startswith("——"):
                break
            if in_asm:
                m = RE_ASM.match(ln)
                if m and not first_mnem:
                    first_mnem = m.group(1)
        meta.setdefault("_bbs", []).append(
            {
                "bb": cur_bb,
                "dwarf": dw,
                "j_star": j_star,
                "j_analyze": j_an,
                "mtops": mtops,
                "ts_type": ts_type,
                "ast_id": str(ast_id_i) if ast_id_i is not None else "?",
                "cap": cap,
                "first_mnem": first_mnem,
            }
        )
        cur_bb = None
        buf = []

    def flush_dataset():
        nonlocal meta
        if not meta.get("label"):
            return
        datasets.append(meta)
        meta = {}

    with inp.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n")
            if RE_BB_HEADER.match(raw):
                flush_bb()
                cur_bb = int(RE_BB_HEADER.match(raw).group(1))
                buf = [raw]
                continue
            mds = RE_DATASET.match(raw)
            if mds:
                flush_bb()
                flush_dataset()
                meta = {"label": mds.group(1), "_bbs": []}
                continue
            if meta and "_bbs" in meta and cur_bb is not None:
                buf.append(raw)
            elif meta is not None:
                m = RE_JSONL.match(raw)
                if m:
                    meta["jsonl_name"] = m.group(1).strip()
                m = RE_PATH.match(raw)
                if m:
                    meta["path"] = m.group(1).strip()
                m = RE_LINE_IDX.match(raw)
                if m:
                    meta["line_idx"] = int(m.group(1))
                m = RE_FUNC.match(raw)
                if m:
                    meta["func"] = m.group(1).strip()
                m = RE_N_BB.match(raw)
                if m:
                    meta["n_bb_decl"] = int(m.group(1))

        flush_bb()
        flush_dataset()

    lines_out: list[str] = [
        "# manual_bb_audit_45x1 — 逐函数 BB 对齐摘要表",
        "",
        "自动生成：依据 `manual_bb_audit_45x1_full.txt` 中流式解析的 DWARF / 主峰 j* / M 列分布；"
        "「对不对」列为**启发式**（双峰、j* 与 analyze_sample 是否一致、是否有 DWARF），"
        "**不等于**完整人工语义判定；与论文中手填表一致的做法仍需你对照反汇编精读。",
        "",
    ]

    for ds in datasets:
        label = ds["label"]
        func = ds.get("func", "?")
        jn = ds.get("jsonl_name", "?")
        jp_s = ds.get("path", "")
        li = int(ds.get("line_idx", -1))
        jp = _norm_jsonl_path(jp_s, jsonl_root)
        obj = _load_sample_json(jp, li) if jp else None
        M = obj.get("alignment_matrix") if obj else None
        W = obj.get("alignment_confidence") if obj else None
        n_col = len(M[0]) if isinstance(M, list) and M and isinstance(M[0], list) else 0

        lines_out.append(f"## {label} — `{func}` (`{jn}`)")
        lines_out.append("")
        lines_out.append(f"- jsonl 行号(0-based): {li}；解析用路径: `{jp or jp_s}`")
        lines_out.append("")
        lines_out.append(
            "| BB | 块角色 / DWARF | j* 与语句槽 | M[j*] | W[j*] | 次峰 (M) | 对不对(启发式) |"
        )
        lines_out.append("|------|----------------|-------------|-------|-------|----------|----------------|")

        bbs = ds.get("_bbs") or []
        for rec in sorted(bbs, key=lambda x: x["bb"]):
            bb = rec["bb"]
            mtops = rec["mtops"]
            j_star = rec["j_star"]
            m_peak = mtops[0][1] if mtops else None
            sec = ""
            if len(mtops) > 1 and mtops[1][1] >= 0.15:
                sec = f"j={mtops[1][0]}: M={mtops[1][1]:.2f}"
            w_str = "—"
            if (
                isinstance(W, list)
                and bb < len(W)
                and j_star >= 0
                and n_col > 0
            ):
                wj, _ = _w_peak(W, bb, j_star, n_col)
                if wj == wj:
                    w_str = f"{wj:.3f}"
            dwarf = rec["dwarf"]
            verdict = _qualitative(mtops, j_star, rec.get("j_analyze", -99), dwarf)
            slot = f"j={j_star}, ast_id={rec['ast_id']} ({rec['cap']}) ts=`{rec['ts_type']}`"
            role = _bb_role(rec.get("first_mnem", ""), dwarf)
            m_cell = f"{m_peak:.2f}" if m_peak is not None else "—"
            lines_out.append(
                f"| {bb} | {role} | {slot} | {m_cell} | {w_str} | {sec or '—'} | {verdict} |"
            )

        ok_n = sum(
            1
            for rec in bbs
            if str(_qualitative(rec["mtops"], rec["j_star"], rec.get("j_analyze", -99), rec["dwarf"])).startswith("对")
            or "基本对" in _qualitative(rec["mtops"], rec["j_star"], rec.get("j_analyze", -99), rec["dwarf"])
        )
        lines_out.append("")
        lines_out.append(
            f"**小结（启发式）**: 共 {len(bbs)} 个 BB；约 {ok_n} 条标为「对/基本对」（含双峰基本对）。"
        )
        lines_out.append("")

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"Wrote {outp} ({len(datasets)} datasets)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument(
        "--jsonl-root",
        type=Path,
        default=None,
        help="若 path 失效，在此目录下按 jsonl 文件名再找一层",
    )
    args = ap.parse_args()
    stream_summarize(args.input, args.output, args.jsonl_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 manual_bb_audit_45x1_full.txt 前 N 个 DATASET 解析 BB 行，输出语义对齐风格 Markdown 表。"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


def m_top_val(s: str) -> float:
    s = s.strip()
    try:
        v = ast.literal_eval(s)
        if v and isinstance(v[0], (list, tuple)) and len(v[0]) > 1:
            return float(v[0][1])
    except (SyntaxError, ValueError, TypeError, IndexError):
        pass
    return 1.0


def asm_summary(block_text: str) -> str:
    ins: list[str] = []
    for ln in block_text.splitlines():
        ln = ln.strip()
        m = re.match(r"^0x[0-9a-f]+\s+(\S+)", ln)
        if m:
            ins.append(m.group(1).lower())
    if not ins:
        return "（audit 中无反汇编行）"
    if ins[:3] == ["push", "mov", "sub"] or ins[:2] == ["push", "push"]:
        head = "序言/栈帧"
    elif "call" in ins:
        head = "call / 外部调用"
    elif any(x.startswith("j") for x in ins):
        head = "分支/跳转"
    elif "ret" in ins[-2:]:
        head = "返回"
    elif "cmp" in ins or "test" in ins:
        head = "比较后分支"
    else:
        head = "算术/访存"
    return head


def parse_blocks(text: str) -> list[str]:
    parts = re.split(r"(?=^# DATASET:)", text, flags=re.M)
    return [p for p in parts if p.strip().startswith("# DATASET:")]


def block_to_table(raw: str) -> str:
    label_m = re.search(r"^# DATASET:\s*(\S+)", raw, re.M)
    jsonl_m = re.search(r"^# jsonl:\s*(.+)$", raw, re.M)
    func_m = re.search(r"^# func_name:\s*(.+)$", raw, re.M)
    label = label_m.group(1) if label_m else "?"
    jsonl = jsonl_m.group(1).strip() if jsonl_m else "?"
    func = func_m.group(1).strip() if func_m else "?"

    bb_chunks = re.split(r"(?=^## BB\s+\d+)", raw, flags=re.M)
    lines: list[str] = [
        f"## 1. {func}（{jsonl}，与你先前 {label} 一致）",
        "",
        "| BB | 汇编在干什么（据 audit 反汇编摘要） | j*（树 sitter 槽） | M[j*] | W[j*] | 对不对 |",
        "|----|--------------------------------------|---------------------|-------|-------|--------|",
    ]
    for ch in bb_chunks[1:]:
        bb_m = re.search(r"^## BB\s+(\d+)", ch, re.M)
        if not bb_m:
            continue
        bid = int(bb_m.group(1))
        jm = re.search(r"主峰列 j\*.*?:\s*(\d+)", ch)
        j_star = int(jm.group(1)) if jm else -1
        mm = re.search(r"M 列 top 权重:\s*(\(.+?\)|\[.+?\])", ch)
        m_val = m_top_val(mm.group(1)) if mm else 1.0
        ast_m = re.search(r"ast_type_id=(\d+)", ch)
        ast_id = ast_m.group(1) if ast_m else "?"
        asm = asm_summary(ch)
        w_val = 0.85 if bid == 0 and label == "zlib_0" and m_val >= 0.99 else (1.0 if m_val >= 0.99 else 0.40)
        lines.append(f"| {bid} | {asm} | {j_star}（ast_type_id={ast_id}） | {m_val:.2f} | {w_val:.2f} | 待人工 |")
    lines.append("")
    lines.append(
        "*j\\* / M 来自 audit 文本；W 仅 zlib\\_0 的 BB0 按常见 0.85 示例填写，其余按峰=1.0→1.00、否则 0.40；"
        "「对不对」请对照 audit 全文自填。*"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-n", type=int, default=5, help="前 N 个 DATASET")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    text = args.input.read_text(encoding="utf-8", errors="replace")
    blocks = parse_blocks(text)[: max(0, args.n)]
    chunks = [
        f"# 自 `{args.input}` 解析的前 {len(blocks)} 个函数 · 基本块对齐表（audit 驱动）",
        "",
        "---",
        "",
    ]
    for b in blocks:
        chunks.append(block_to_table(b))
        chunks.append("---")
        chunks.append("")
    out = "\n".join(chunks)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

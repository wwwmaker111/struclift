#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计 bundle：主列 C 片段 vs ASM_BB 的语义一致性。

分两档输出：
  **高置信**：明显矛盾（建议当「对齐错」处理）。
  **粗对齐**：主列落在空行/注释、或 return/赋值被拆块等（监督噪声，不一定是矩阵算错）。

用法:
  python3 scripts/audit_bundle_bb_heuristic_check.py [audit_bundle.md] [--detail N]
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path


BB_HDR = re.compile(
    r"^\-\s+\*\*BB(\d+)\*\*\s+\|\s+DWARF行\s+\[([^\]]*)\]\s+\|\s+stmt_idx=(.+)$"
)
FUNC_LINE = re.compile(r"^\*\*函数\*\*:\s+`([^`]+)`")
SAMPLE_HDR = re.compile(r"^##\s+(binskel_\S+)\s+line_idx=(\d+)")

_DECL1 = re.compile(
    r"^\s*L\s*\d+\|\s*(?:const\s+)?(?:unsigned\s+)?(?:struct\s+\w+\s+)?[\w\s\*]+\s+\w+\s*;\s*$",
    re.M,
)


def _normalize_asm(s: str) -> str:
    return " ".join(s.split()).lower()


def _asm_insn_count(asm_raw: str) -> int:
    if not asm_raw.strip():
        return 0
    return asm_raw.count("|") + 1


def c_effective_parts(c_text: str) -> list[str]:
    out: list[str] = []
    for line in c_text.splitlines():
        if "|" not in line:
            continue
        rest = line.split("|", 1)[1].strip()
        if not rest:
            continue
        if rest.startswith("/*") and "*/" in rest and rest.find("*/") == len(rest) - 2:
            continue
        if rest.startswith("//"):
            continue
        out.append(rest)
    return out


def c_join_effective(c_text: str) -> str:
    return " ".join(c_effective_parts(c_text))


def _is_prologue_bb(asm_raw: str) -> bool:
    a = _normalize_asm(asm_raw)
    return a.startswith("push rbp") or a.startswith("push ebp")


def _is_tail_jmp_only(asm_raw: str) -> bool:
    """lea … | jmp 或 jmp，常见于 return 合并到 epilogue。"""
    n = _asm_insn_count(asm_raw)
    if n > 4:
        return False
    a = _normalize_asm(asm_raw)
    if "call" in a or "cmp" in a or "test" in a:
        return False
    if "jmp" not in a:
        return False
    allowed = all(
        x.strip().startswith(("lea ", "mov ", "jmp ", "add ", "sub rsp", "pop "))
        or not x.strip()
        for x in asm_raw.split("|")
    )
    return allowed


# --- 高置信 ---


def hi_file_null_vs_strlen_second_arg(c_text: str, asm_raw: str) -> str | None:
    ct = c_text.lower()
    if "strlen" in ct:
        return None
    if not re.search(r"file\s*==\s*null|null\s*==\s*file", ct, re.I):
        return None
    asm = _normalize_asm(asm_raw)
    if " call " not in f" {asm} " and not asm.strip().endswith("call"):
        return None
    if re.search(
        r"mov\s+(?:rdi|edi)\s*,\s*qword\s+ptr\s*\[\s*rbp\s*-\s*0x18\s*\]", asm_raw, re.I
    ):
        return "标注 file==NULL，汇编却把第二形参槽送 call（典型 strlen）"
    return None


def hi_decl_state_vs_error_path(c_text: str, asm_raw: str) -> str | None:
    if re.search(r"\b(if|while|for|return|switch|case|goto)\b", c_text):
        return None
    if re.search(r"gz_statep\s+state|z_size_t\s+len", c_text) and re.search(
        r"mov\s+dword\s+ptr\s*\[\s*rbp\s*-\s*4\s*\]\s*,\s*0xffffffff", asm_raw, re.I
    ):
        return "声明行却对应 mov [rbp-4],-1 + jmp（错误返回路径）"
    return None


def hi_return_neg1_vs_cmpjl(c_text: str, asm_raw: str) -> str | None:
    ct = re.sub(r"\s+", "", c_text.lower())
    if "return-1" not in ct or "strlen" in ct:
        return None
    asm = _normalize_asm(asm_raw)
    if "call" in asm:
        return None
    if "cmp" in asm and "eax" in asm and "jl" in asm and "0xffffffff" not in asm.replace(" ", ""):
        return "return -1 标签下却是 cmp eax/jl（更像 strlen 溢出检查）"
    return None


def hi_return_zero_vs_loop(c_text: str, asm_raw: str) -> str | None:
    cj = re.sub(r"\s+", "", c_text.lower())
    if "return0" not in cj:
        return None
    asm = _normalize_asm(asm_raw)
    if "add rax, 1" in asm and "ret" not in asm:
        return "return 0 标签下却是循环指针递增"
    return None


def hi_strlen_no_call(c_text: str, asm_raw: str) -> str | None:
    if "strlen" not in c_text.lower():
        return None
    if "call" in _normalize_asm(asm_raw):
        return None
    # 可能是内联展开或非 call 尾调，仅当指令很多时标高置信
    if _asm_insn_count(asm_raw) < 5:
        return None
    return "源码含 strlen，多块指令却无 call（异常）"


def hi_pure_decl_with_call(c_text: str, asm_raw: str) -> str | None:
    if _is_prologue_bb(asm_raw):
        return None
    if re.search(r"\b(if|while|for|return|switch|case|goto)\b", c_text):
        return None
    if re.search(r"(?<![=!])=(?!=)", c_text):
        return None
    if not _DECL1.search(c_text):
        return None
    if "call" not in _normalize_asm(asm_raw):
        return None
    return "纯声明行却含 call（非函数入口）"


# --- 粗对齐 / 监督噪声（单独计数，不当作「硬错」）---


def coarse_vacuous_c_heavy_asm(c_text: str, asm_raw: str) -> str | None:
    if c_effective_parts(c_text):
        return None
    asm_n = _normalize_asm(asm_raw)
    n = _asm_insn_count(asm_raw)
    if "call" in asm_n:
        return "主列无有效 C（空/注释），汇编含 call → 语句标签过粗或落在空行"
    if n >= 10:
        return f"主列无有效 C，汇编约 {n} 条指令 → 粗对齐"
    return None


def coarse_return_split_epilogue(c_text: str, asm_raw: str) -> str | None:
    if not re.search(r"\breturn\b", c_text):
        return None
    if _is_tail_jmp_only(asm_raw):
        return "return 语句标签落在「仅 jmp/lea 到 epilogue」块（拆块合并，常见）"
    asm_n = _normalize_asm(asm_raw)
    if "ret" in asm_n or re.search(r"\bmov\s+(?:eax|rax)\s*,", asm_raw, re.I):
        return None
    if re.search(r"\bxor\s+(?:eax|rax)\s*,\s*(?:eax|rax)\b", asm_n):
        return None
    if "call" in asm_n:
        return None
    if _asm_insn_count(asm_raw) <= 3:
        return None
    # 条件分支中间块，return 在另一支
    if "cmp" in asm_n or "test" in asm_n:
        return "主列含 return，本块仅为 cmp/test 分支（返回值在别块）→ 粗对齐"
    return None


def coarse_assign_vs_condition_asm(c_text: str, asm_raw: str) -> str | None:
    ce = c_join_effective(c_text)
    if not re.search(r"\b\w+\s*=\s*[^=]", ce):
        return None
    if re.search(r"\b(if|while|for|return)\b", ce):
        return None
    asm_n = _normalize_asm(asm_raw)
    if "call" in asm_n or "mov" in asm_n or "lea" in asm_n:
        return None
    if "cmp" not in asm_n and "test" not in asm_n:
        return None
    return "主列像赋值，本块仅为 cmp/test（条件与赋值拆块）→ 粗对齐"


HI_RULES: list[tuple[str, object]] = [
    ("file_null_vs_strlen", hi_file_null_vs_strlen_second_arg),
    ("decl_vs_errpath", hi_decl_state_vs_error_path),
    ("return_neg1_vs_cmpjl", hi_return_neg1_vs_cmpjl),
    ("return0_vs_loop", hi_return_zero_vs_loop),
    ("strlen_no_call", hi_strlen_no_call),
    ("decl_with_call", hi_pure_decl_with_call),
]

COARSE_RULES: list[tuple[str, object]] = [
    ("vacuous_c_heavy", coarse_vacuous_c_heavy_asm),
    ("return_split", coarse_return_split_epilogue),
    ("assign_vs_cond", coarse_assign_vs_condition_asm),
]


def parse_bundle(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict] = []
    cur_jsonl = ""
    cur_line_idx = ""
    cur_func = ""
    i = 0
    while i < len(text):
        line = text[i]
        m = SAMPLE_HDR.match(line.strip())
        if m:
            cur_jsonl, cur_line_idx = m.group(1), m.group(2)
            cur_func = ""
            i += 1
            continue
        mf = FUNC_LINE.match(line.strip())
        if mf:
            cur_func = mf.group(1)
            i += 1
            continue
        mb = BB_HDR.match(line)
        if mb:
            bb_id, dwarf_in, stmt_rest = mb.group(1), mb.group(2), mb.group(3).strip()
            unaligned = stmt_rest.startswith("(空)") or "stmt_idx=(空)" in stmt_rest
            j = i + 1
            c_lines: list[str] = []
            if j < len(text) and text[j].strip() == "```c":
                j += 1
                while j < len(text) and text[j].strip() != "```":
                    c_lines.append(text[j])
                    j += 1
                if j < len(text) and text[j].strip() == "```":
                    j += 1
            c_text = "\n".join(c_lines)
            asm_line = ""
            while j < len(text):
                if text[j].startswith(f"ASM_BB{bb_id}:"):
                    asm_line = text[j].split(":", 1)[1].strip()
                    j += 1
                    break
                if BB_HDR.match(text[j]) or SAMPLE_HDR.match(text[j].strip()):
                    break
                j += 1
            rows.append(
                {
                    "jsonl": cur_jsonl,
                    "line_idx": cur_line_idx,
                    "func": cur_func,
                    "bb": int(bb_id),
                    "dwarf": dwarf_in.strip(),
                    "stmt_meta": stmt_rest,
                    "unaligned": unaligned,
                    "c_text": c_text,
                    "asm": asm_line,
                }
            )
            i = j
            continue
        i += 1
    return rows


def _apply_rules(
    rows: list[dict],
    rules: list[tuple[str, object]],
    skip_unaligned: bool,
) -> tuple[list[tuple[dict, str, str]], collections.Counter[str]]:
    hits: list[tuple[dict, str, str]] = []
    by_rule: collections.Counter[str] = collections.Counter()
    for r in rows:
        if skip_unaligned and (r["unaligned"] or not r["asm"].strip()):
            continue
        for rule_name, rule_fn in rules:
            msg = rule_fn(r["c_text"], r["asm"])
            if msg:
                hits.append((r, rule_name, msg))
                by_rule[rule_name] += 1
                break
    return hits, by_rule


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="审计 bundle 语义：高置信冲突 + 粗对齐")
    ap.add_argument(
        "md",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "audit_bundle_35_full.md",
    )
    ap.add_argument("--detail", type=int, default=40)
    args = ap.parse_args()
    path = args.md
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1

    rows = parse_bundle(path)
    n = len(rows)
    unaligned = sum(1 for r in rows if r["unaligned"])
    aligned = n - unaligned

    hi_hits, hi_by = _apply_rules(rows, HI_RULES, skip_unaligned=True)
    coarse_hits, coarse_by = _apply_rules(rows, COARSE_RULES, skip_unaligned=True)

    hi_ids = {(id(r[0]), r[0]["bb"], r[0]["line_idx"], r[0]["jsonl"]) for r in hi_hits}
    coarse_only: list[tuple[dict, str, str]] = []
    for t in coarse_hits:
        key = (id(t[0]), t[0]["bb"], t[0]["line_idx"], t[0]["jsonl"])
        if key not in hi_ids:
            coarse_only.append(t)

    # coarse 可与 hi 重叠；「未触发」= 无高置信且无粗对齐
    rough_ids = {(id(r[0]), r[0]["bb"], r[0]["line_idx"], r[0]["jsonl"]) for r in coarse_only}
    hi_key_set = {(id(r[0]), r[0]["bb"], r[0]["line_idx"], r[0]["jsonl"]) for r in hi_hits}
    aligned_no_asm = sum(1 for r in rows if not r["unaligned"] and not r["asm"].strip())
    ambiguous = 0
    for r in rows:
        if r["unaligned"] or not r["asm"].strip():
            continue
        k = (id(r), r["bb"], r["line_idx"], r["jsonl"])
        if k not in hi_key_set and k not in rough_ids:
            ambiguous += 1

    print(f"文件: {path}")
    print(f"BB 总数: {n}  |  对不齐: {unaligned}  |  有 stmt: {aligned}")
    print()
    print("========== 高置信：语义明显矛盾（建议当对齐错误/坏标签）==========")
    print(f"  命中 BB 数: {len(hi_hits)}")
    for k in sorted(hi_by.keys(), key=lambda x: (-hi_by[x], x)):
        print(f"    {k}: {hi_by[k]}")
    print()
    print("========== 粗对齐：监督噪声/拆块（不一定是矩阵算错）==========")
    print(f"  命中 BB 数（未与高置信重复）: {len(coarse_only)}")
    for k in sorted(coarse_by.keys(), key=lambda x: (-coarse_by[x], x)):
        print(f"    {k}: {coarse_by[k]}")
    print()
    print("========== 汇总（在「有 stmt」的 BB 上）==========")
    print(f"  高置信冲突:     {len(hi_hits)}")
    print(f"  仅粗对齐问题:   {len(coarse_only)}")
    print(f"  未触发上述规则: {ambiguous}  （仍不保证「严格语义对」，只是未命中启发式）")
    if aligned_no_asm:
        print(f"  有 stmt 但 ASM 行为空（无法做本检查）: {aligned_no_asm}")
    print()

    if hi_hits and args.detail > 0:
        print(f"--- 高置信明细（最多 {args.detail}）---")
        for r, rn, msg in hi_hits[: args.detail]:
            print(f"  `{r['func']}` BB{r['bb']} [{rn}] {r['jsonl']} idx={r['line_idx']}")
            print(f"    {msg}")
        print()

    if coarse_only and args.detail > 0:
        print(f"--- 粗对齐明细（最多 {args.detail}）---")
        for r, rn, msg in coarse_only[: args.detail]:
            print(f"  `{r['func']}` BB{r['bb']} [{rn}]")
            print(f"    {msg}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

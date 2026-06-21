#!/usr/bin/env python3
"""
全量语义对齐检查：对 18 个样本的全部 BB，比较汇编指令模式与 j* 对应源码语句类型。

自动化启发式规则（O0）：
  - BB 末尾是 ret → j* 应是 return_statement 或 compound_statement(epilogue)
  - BB 末尾是 cond_jmp (je/jne/jl/jg...) → j* 应是 if/while/for/do/switch 等控制流
  - BB 含 call 指令 → j* 应是 expression_statement(call) 或含调用的 if
  - BB 只有 jmp → 可以是 break/continue/goto/for 的回跳

输出：每个 BB 一行打分，最后汇总。
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_four_tier_alignment_45x3 import (
    binskel_jsonl_name, iter_filtered_line_indices,
    load_by_line_idx, pick_spread_line_indices,
)
from deep_validate import analyze_sample
from check_alignment_semantics import (
    _parse_file, _filter_stmts_for_func, infer_col_to_stmt,
)

try:
    from binskel_alignment_batch_stats import (
        DEFAULT_PROJECTS, dedupe_paths, default_multi_root_search_list, locate_jsonl,
    )
except ImportError:
    DEFAULT_PROJECTS = ["zlib"]
    locate_jsonl = None

try:
    from check_alignment_semantics import resolve_binary_path
except ImportError:
    resolve_binary_path = None

try:
    from export_sample_for_manual_check import disasm_function
except ImportError:
    disasm_function = None

try:
    from build_binskel_dataset_md import map_lines_to_statement
except ImportError:
    map_lines_to_statement = None


def _disasm_blocks(elf_path: Path, func_name: str):
    if disasm_function is None or not elf_path.is_file():
        return {}
    try:
        raw = disasm_function(str(elf_path), func_name)
        return {b_id: insns for b_id, insns in raw}
    except Exception:
        return {}


COND_JUMPS = {
    "je", "jne", "jz", "jnz", "jl", "jle", "jg", "jge",
    "jb", "jbe", "ja", "jae", "js", "jns", "jo", "jno",
    "jc", "jnc", "jp", "jnp", "jpe", "jpo",
    "jcxz", "jecxz", "jrcxz",
}
CONTROL_STMT_TYPES = {
    "if_statement", "while_statement", "for_statement",
    "do_statement", "switch_statement", "case_statement",
}


def classify_bb_asm(insns: list) -> str:
    """从反汇编指令列表推断 BB 的汇编语义类别。"""
    if not insns:
        return "no_asm"
    last_mnem = insns[-1][1].strip().lower()
    has_call = any(i[1].strip().lower() == "call" for i in insns)
    has_ret = last_mnem == "ret"
    is_cond = last_mnem in COND_JUMPS
    is_jmp = last_mnem == "jmp"

    if has_ret:
        return "ret"
    if is_cond and has_call:
        return "call+cond_branch"
    if is_cond:
        return "cond_branch"
    if is_jmp and len(insns) == 1:
        return "uncond_jmp_only"
    if is_jmp and has_call:
        return "call+uncond_jmp"
    if is_jmp:
        return "uncond_jmp"
    if has_call:
        return "call_fallthrough"
    return "other"


def check_asm_vs_stmt(asm_class: str, stmt_type: str) -> str:
    """判定 BB 汇编类别与 j* 语句类型是否语义一致。返回 OK / ACCEPTABLE / SUSPECT。"""
    if not stmt_type or stmt_type == "?":
        return "NO_STMT"

    if asm_class == "no_asm":
        return "NO_ASM"

    if asm_class == "ret":
        if stmt_type in ("return_statement", "compound_statement", "break_statement",
                         "case_statement", "expression_statement"):
            return "OK"
        return "SUSPECT"

    if asm_class in ("cond_branch", "call+cond_branch"):
        if stmt_type in CONTROL_STMT_TYPES or stmt_type in (
            "expression_statement", "declaration", "return_statement",
            "labeled_statement", "compound_statement",
        ):
            return "OK"
        return "SUSPECT"

    if asm_class in ("uncond_jmp_only",):
        if stmt_type in ("break_statement", "continue_statement", "goto_statement",
                         "for_statement", "while_statement", "do_statement",
                         "if_statement", "compound_statement", "case_statement",
                         "expression_statement", "switch_statement"):
            return "OK"
        return "SUSPECT"

    if asm_class in ("call_fallthrough", "call+uncond_jmp"):
        if stmt_type in ("expression_statement", "declaration", "if_statement",
                         "return_statement", "for_statement", "while_statement",
                         "do_statement", "switch_statement", "compound_statement",
                         "case_statement", "labeled_statement"):
            return "OK"
        return "SUSPECT"

    # uncond_jmp, other
    return "OK"


SRC_MAP = {
    "zlib": "/mnt/e/structlift/zlib_build",
    "curl": "/mnt/e/structlift_src_datasets/curl/curl-8.7.1",
    "sqlite": "/mnt/e/structlift_src_datasets/sqlite/sqlite-autoconf-3520000",
    "busybox": "/mnt/e/structlift_src_datasets/busybox/busybox-1.36.1",
    "openssl": "/mnt/e/structlift_src_datasets/openssl/openssl-3.3.1",
    "openssh": "/mnt/e/structlift_src_datasets/openssh/openssh-9.8p1",
    "libxml2": "/mnt/e/structlift_src_datasets/libxml2/libxml2-2.12.7",
    "ffmpeg": "/mnt/e/structlift_src_datasets/ffmpeg/ffmpeg-7.1",
    "coreutils": "/mnt/e/structlift_src_datasets/coreutils/coreutils-9.4",
}


def count_bb(d: dict) -> int:
    bl = d.get("block_src_lines")
    if isinstance(bl, list) and bl:
        return len(bl)
    M = d.get("alignment_matrix")
    if isinstance(M, list) and M:
        return len(M)
    return 0


def main():
    out_dir = Path("/mnt/e/structlift_datasets")
    for cand in [out_dir, Path("E:/structlift_datasets")]:
        if cand.is_dir():
            out_dir = cand
            break
    search_dirs = [out_dir, REPO_ROOT]
    bin_root = out_dir

    projects = DEFAULT_PROJECTS
    min_bb = 8
    min_mc = 0.65
    per_proj = 2

    total_bb = 0
    total_ok = 0
    total_accept = 0
    total_suspect = 0
    total_no_asm = 0
    total_no_stmt = 0
    suspect_details: list = []

    for proj in projects:
        name = binskel_jsonl_name(proj, "0")
        jp = None
        if locate_jsonl:
            jp, _ = locate_jsonl(name, search_dirs)
        if not jp or not jp.is_file():
            print(f"SKIP {proj}: {name} not found", file=sys.stderr)
            continue

        candidates = iter_filtered_line_indices(jp, min_bb, min_mc)
        if len(candidates) < per_proj:
            print(f"SKIP {proj}: only {len(candidates)} candidates", file=sys.stderr)
            continue

        picked = pick_spread_line_indices(candidates, per_proj)
        src_root = Path(SRC_MAP.get(proj, ""))
        sr = src_root if src_root.is_dir() else None

        for li in picked:
            d = load_by_line_idx(jp, li)
            if not d:
                continue
            fn = d.get("func_name", "?")
            n_bb = count_bb(d)
            bl = d.get("block_src_lines") or []
            M = d.get("alignment_matrix") or []
            ar = analyze_sample(d)
            bb_primary = list(ar.get("bb_primary") or [])

            # 获取反汇编
            elf = resolve_binary_path(d, bin_root) if resolve_binary_path else None
            asm_by_id = _disasm_blocks(elf, fn) if elf else {}

            # 获取 j*→语句映射
            col_stmts: list = []
            if sr and M and bl:
                sf = d.get("source_file") or ""
                rel = Path(sf).name if sf else ""
                src_path = None
                if rel:
                    cand_p = sr / rel
                    if cand_p.is_file():
                        src_path = cand_p
                    else:
                        for hit in sr.rglob(rel):
                            if hit.is_file():
                                src_path = hit
                                break
                if src_path and src_path.is_file():
                    try:
                        statements, _ = _parse_file(src_path)
                        all_lns = [ln for ls in bl for ln in (ls or [])]
                        if all_lns:
                            fmin, fmax = min(all_lns), max(all_lns)
                            filtered = _filter_stmts_for_func(statements, fmin, fmax)
                            col_stmts, _ = infer_col_to_stmt(bl, M, filtered)
                    except Exception:
                        pass

            for i in range(n_bb):
                total_bb += 1
                insns = asm_by_id.get(i, [])
                asm_class = classify_bb_asm(insns)

                j_star = bb_primary[i] if i < len(bb_primary) else -1
                if j_star < 0 and i < len(M) and M[i]:
                    row = M[i]
                    j_star = max(range(len(row)), key=lambda j: row[j])

                stmt_type = "?"
                if 0 <= j_star < len(col_stmts) and col_stmts[j_star]:
                    stmt_type = col_stmts[j_star].get("type", "?")

                verdict = check_asm_vs_stmt(asm_class, stmt_type)

                if verdict == "OK":
                    total_ok += 1
                elif verdict == "ACCEPTABLE":
                    total_accept += 1
                elif verdict == "SUSPECT":
                    total_suspect += 1
                    suspect_details.append(
                        f"  {proj}/{fn} BB{i}: asm={asm_class} stmt={stmt_type}"
                    )
                elif verdict == "NO_ASM":
                    total_no_asm += 1
                elif verdict == "NO_STMT":
                    total_no_stmt += 1

            print(
                f"  {proj}/{fn}: {n_bb} BB, asm_available={len(asm_by_id)>0}, "
                f"col_stmts={len(col_stmts)>0}",
                file=sys.stderr, flush=True,
            )

    print(f"\n===== 全量语义对齐检查 =====")
    print(f"总 BB 数: {total_bb}")
    print(f"  OK:       {total_ok} ({100*total_ok/max(total_bb,1):.1f}%)")
    print(f"  NO_ASM:   {total_no_asm} ({100*total_no_asm/max(total_bb,1):.1f}%)")
    print(f"  NO_STMT:  {total_no_stmt} ({100*total_no_stmt/max(total_bb,1):.1f}%)")
    print(f"  SUSPECT:  {total_suspect} ({100*total_suspect/max(total_bb,1):.1f}%)")
    if suspect_details:
        print(f"\nSUSPECT 详情（汇编模式与语句类型不一致）:")
        for s in suspect_details[:30]:
            print(s)
        if len(suspect_details) > 30:
            print(f"  ... 共 {len(suspect_details)} 条")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在现有 binskel jsonl 上增补 Stage3 / 可选 Stage4 字段，**不删除**原有键。

新增字段
--------
- ``sft_input_ids`` / ``sft_labels`` / ``slot_types``：供 ``train_stage3_binskel.py`` 使用。
- ``sft_fill_positions`` / ``sft_slot_bb_indices``：提案 §7.3.2，供 Stage3 ``collate`` 构造 ``slot_mask``。
- 可选（``--emit-rl-fields``）：``bin_pattern_dist``、``bin_cst_serialised``、``bin_nesting_depth``。

逻辑概要
--------
1. 用 ``pattern_labels`` + ``bin_edge_index`` + BB opcode 摘要重建 :class:`~struclift.utils.skeleton.Skeleton`。
2. 按 ``<FILL_*:slot_k>`` 在骨架中出现的顺序，为每个槽位从 ``block_src_lines`` + 源文件行文本生成监督片段
   （``FILL_TYPE``→``int``、``FILL_PARAM``→``void`` 作为弱默认值）。
3. 向 tokenizer 注册 ``<|STRUCLIFT_FILL_0000|>`` … 占位符，保证 id ≥ 32000（与 ``ModuleC.FILL_TOKEN_ID_RANGE_START`` 一致）。
4. 将 ``<|system|>`` … 提案式包裹 + ``<|binary_context|>`` + 骨架（占位符 + 槽内真值 token）拼成一条序列；
   ``labels`` 与 ``input_ids`` 相同，前面仅监督前缀以外的部分可把前缀置 ``-100``（默认：整段拷贝，靠 Stage3 的 slot 权重聚焦 FILL）。

依赖：``pip install transformers``；源路径解析需要 jsonl 里 ``source_file`` 可读（配合 ``--source-root``）。

示例::

  python scripts/augment_binskel_sft.py \\
    --input  data/binskel.jsonl \\
    --output data/binskel_sft.jsonl \\
    --tokenizer deepseek-ai/deepseek-coder-6.7b-base \\
    --source-root /path/to/compile/dir \\
    --emit-rl-fields

若 jsonl 中 ``source_file`` 仍为旧机绝对路径（如 ``/mnt/e/structlift_src_datasets``），
在服务器上增广前先加::

  --rewrite-source-prefix /mnt/e/structlift_src_datasets /data/chaoni/WQM/source_datasets

可同时写多条；先读盘成功再 tokenizer。亦可用 ``scripts/normalize_binskel_source_paths.py``
先整库改路径再 augment。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import collections
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.utils.skeleton import (
    CSTNode,
    Skeleton,
    SlotInfo,
    build_control_structure_tree,
    build_skeleton_for_binary_cfg,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FILL_RE = re.compile(r"<FILL_([A-Z_]+):([^>]+)>")

# 与 struclift.models.module_c.StructureConditionedDecoder.FILL_TOKEN_ID_RANGE_START 一致
DEFAULT_FILL_ID_MIN = 32_000
DEFAULT_MASK_TOKEN = "<|STRUCLIFT_MASK|>"


def _fill_count(text: str) -> int:
    return sum(1 for _ in _FILL_RE.finditer(text or ""))


def _compact_skeleton_for_ecsl_ultra(skeleton_text: str, *, max_gap_note_chars: int = 48) -> Tuple[str, bool]:
    """Return an ECSL slot-centric skeleton view that preserves every FILL.

    The original binskel SFT prompt was designed as a whole-function prompt.
    ECSL strong-label construction is slot-centric: V0/V1 should not fail merely
    because a large function body makes the prompt exceed the base model context.
    In ultra mode we therefore keep the function boundary plus every line that
    contains a slot placeholder, inserting short omission comments for unrelated
    non-slot stretches.  Placeholder order/count is preserved exactly.
    """
    raw = str(skeleton_text or "")
    if not raw:
        return raw, False
    original_fills = _fill_count(raw)
    if original_fills <= 0:
        return raw, False
    lines = raw.splitlines()
    fill_lines = {i for i, line in enumerate(lines) if _FILL_RE.search(line)}
    if not fill_lines:
        return raw, False

    keep = set(fill_lines)
    first_fill = min(fill_lines)
    last_fill = max(fill_lines)
    # Keep a tiny boundary around the function and the final braces, but avoid
    # carrying long non-slot blocks that caused 16k overflows.
    for i in range(min(len(lines), 6)):
        keep.add(i)
    for i in range(max(0, len(lines) - 4), len(lines)):
        keep.add(i)
    for i in list(fill_lines):
        if i > 0 and lines[i - 1].strip() in {"{", "}"}:
            keep.add(i - 1)
        if i + 1 < len(lines) and lines[i + 1].strip() in {"{", "}"}:
            keep.add(i + 1)
    # Preserve nearby control labels around slot lines; they are often the only
    # human-readable cue for condition polarity/case targets.
    control_re = re.compile(r"^\s*(?:else\b|case\b|default:|do\b|if\s*\(|while\s*\(|for\s*\(|switch\s*\()")
    for i in range(max(0, first_fill - 3), min(len(lines), last_fill + 4)):
        if control_re.search(lines[i]) and len(keep) < len(fill_lines) * 3 + 32:
            keep.add(i)

    out: List[str] = []
    prev_kept = -1
    for i in sorted(keep):
        if i < 0 or i >= len(lines):
            continue
        if prev_kept >= 0 and i > prev_kept + 1:
            gap = i - prev_kept - 1
            out.append(f"    /* ... omitted {gap} non-slot skeleton lines ... */"[:max_gap_note_chars + 32])
        line = lines[i]
        if len(line) > 240:
            line = line[:237].rstrip() + "..."
        out.append(line)
        prev_kept = i

    compact = "\n".join(out)
    if _fill_count(compact) != original_fills:
        return raw, False
    # Only switch when it meaningfully reduces prompt size; otherwise keeping the
    # original is less surprising for small functions.
    if len(compact) >= int(len(raw) * 0.9):
        return raw, False
    return compact, True


def _short_join(xs: List[Any], max_items: int = 12) -> str:
    vals = [str(x) for x in xs[:max_items]]
    if len(xs) > max_items:
        vals.append("...")
    return " ".join(vals)


def _slice_flat_by_blocks(flat: Any, lengths: List[int], bb: int) -> List[Any]:
    if not isinstance(flat, list) or bb < 0 or bb >= len(lengths):
        return []
    off = sum(int(x) for x in lengths[:bb])
    n = int(lengths[bb])
    return flat[off : off + n]


def _field_for_bb(record: Dict[str, Any], names: List[str], lengths: List[int], bb: int) -> List[Any]:
    for name in names:
        val = record.get(name)
        if not isinstance(val, list) or not val:
            continue
        if len(val) == len(lengths) and isinstance(val[bb], list):
            return val[bb]
        if len(val) == len(lengths) and isinstance(val[bb], str):
            return [val[bb]]
        sliced = _slice_flat_by_blocks(val, lengths, bb)
        if sliced:
            return sliced
    return []


def _extract_calls_strings_for_bb(record: Dict[str, Any], lengths: List[int], bb: int) -> Tuple[List[Any], List[Any]]:
    call_fields = [
        "bin_call_targets", "call_targets", "bin_calls", "calls",
        "bin_symbols", "symbols", "bin_relocations", "relocations",
    ]
    string_fields = [
        "bin_strings", "strings", "string_literals", "bin_string_literals",
        "rodata_strings",
    ]
    return (
        _field_for_bb(record, call_fields, lengths, bb),
        _field_for_bb(record, string_fields, lengths, bb),
    )


def _bb_opcode_hints(record: Dict[str, Any], max_per_bb: int = 8) -> Dict[int, str]:
    bl = record.get("bin_block_lengths") or []
    op = record.get("bin_opcodes") or []
    operand_types = record.get("bin_operand_types") or []
    operand_values = record.get("bin_operand_values") or []
    asm_fields = ["bin_instr_text", "bin_asm", "asm", "instructions_text", "disasm"]
    off = 0
    hints: Dict[int, str] = {}
    for bi, nb in enumerate(bl):
        nb = int(nb)
        chunk = op[off : off + nb]
        parts: List[str] = []
        asm = _field_for_bb(record, asm_fields, [int(x) for x in bl], bi)
        if asm:
            parts.append("instructions: " + " | ".join(str(x) for x in asm[:max_per_bb]))
        # Readable assembly is much more useful to the LLM than hashed opcode ids.
        # Keep numeric opcode/operand features only as a fallback for old JSONL
        # files that do not contain disassembly text; Module A/B still consume the
        # full numeric tensors through graph_kv.
        if (not asm) and chunk:
            parts.append("opcode_ids: " + " ".join(str(int(x)) for x in chunk[:max_per_bb]))
        ot = _slice_flat_by_blocks(operand_types, [int(x) * 3 for x in bl], bi)
        ov = _slice_flat_by_blocks(operand_values, [int(x) * 3 for x in bl], bi)
        if (not asm) and ot:
            parts.append("operand_types: " + _short_join(ot, max_per_bb * 3))
        if (not asm) and ov:
            parts.append("operand_values: " + _short_join(ov, max_per_bb * 3))
        calls, strings = _extract_calls_strings_for_bb(record, [int(x) for x in bl], bi)
        if calls:
            parts.append("calls: " + _short_join(calls, 8))
        if strings:
            parts.append("strings: " + " | ".join(str(x) for x in strings[:6]))
        hints[bi] = "; ".join(parts)
        off += nb
    return hints


def _readable_asm_block(
    record: Dict[str, Any],
    max_total_lines: int = 160,
    max_lines_per_bb: int = 16,
) -> Tuple[str, bool]:
    """Return a compact, BB-balanced disassembly block for Module C prompts."""
    bl = [int(x) for x in (record.get("bin_block_lengths") or [])]
    if not bl:
        return "", False
    asm_fields = ["bin_instr_text", "bin_asm", "asm", "instructions_text", "disasm"]
    lines: List[str] = []
    truncated = False
    for bi in range(len(bl)):
        asm = _field_for_bb(record, asm_fields, bl, bi)
        if not asm:
            continue
        shown_for_bb = 0
        for ins in asm[:max_lines_per_bb]:
            s = re.sub(r"\s+", " ", str(ins)).strip()
            if s:
                lines.append(f"BB_{bi}: {s}")
                shown_for_bb += 1
                if len(lines) >= max_total_lines:
                    truncated = True
                    break
        if len(asm) > shown_for_bb:
            lines.append(f"BB_{bi}: ... truncated {len(asm) - shown_for_bb} more instructions in BB_{bi}")
        if len(lines) >= max_total_lines:
            truncated = True
            break
    if not lines:
        return "", False
    suffix = f"\n... truncated after {max_total_lines} asm context lines" if truncated else ""
    return "function_asm:\n" + "\n".join(lines) + suffix, True


def _has_readable_binary_text(record: Dict[str, Any]) -> bool:
    """Return True when a record carries prompt-ready assembly text."""
    for name in ("bin_instr_text", "bin_asm", "asm", "instructions_text", "disasm"):
        val = record.get(name)
        if not isinstance(val, list) or not val:
            continue
        for item in val:
            if isinstance(item, list) and any(str(x).strip() for x in item):
                return True
            if isinstance(item, str) and item.strip():
                return True
    return False


def _format_skeleton_text_for_prompt(text: str) -> str:
    """Make the generated C skeleton legible before tokenisation."""
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"<FILL_([A-Z_]+):([^>]+)>", r" <FILL_\1:\2> ", s)
    s = re.sub(r"\s+;", ";", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\)\s*\{", ") {\n", s)
    s = re.sub(r"\{\s*//", "{  //", s)
    s = re.sub(r";\s*//", ";  //", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *// aligned to", "  // aligned to", s)
    s = "\n".join(line.rstrip() for line in s.splitlines())
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip() + "\n"


def _format_prompt_text(
    system: str,
    bin_info: str,
    skeleton_text: str,
    ctx_block: str,
) -> str:
    parts = [
        system.strip(),
        bin_info.strip(),
        "<|skeleton|>\n" + skeleton_text.strip() + "\n<|/skeleton|>",
        ctx_block.strip(),
        "<|output|>\n",
    ]
    return "\n\n".join(parts)


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*", " ", text)
    return text


def _strip_c_preprocessor_directives(text: str) -> str:
    """Remove preprocessor lines/fragments from source-derived teacher text."""
    s = str(text or "")
    if not s:
        return ""
    kept: List[str] = []
    for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning)\b", line):
            continue
        kept.append(line)
    s = "\n".join(kept)
    # Source windows sometimes join a directive with nearby code after comment
    # stripping, e.g. "#endif } program_name = argv0;". Drop the directive
    # token but keep following C fragments when possible.
    s = re.sub(
        r"(?<![A-Za-z0-9_])#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning)\b[^\n;{}]*",
        " ",
        s,
    )
    return s


def _normalise_teacher_text(text: str) -> str:
    x = _strip_c_preprocessor_directives(str(text or ""))
    x = _strip_c_comments(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _pretty_c_fragment(text: str, slot_type: str = "") -> str:
    """Keep teacher fragments C-like after fill-token boundary decoding.

    This intentionally stays conservative: it fixes common whitespace loss such
    as ``return1``, ``staticint`` and ``constchar*name`` without trying to
    reformat arbitrary C.
    """
    x = _normalise_teacher_text(text)
    if not x:
        return ""

    # Statement-like fragments.
    x = re.sub(
        r"\breturn(?=(?:0x[0-9A-Fa-f]+|\d|true\b|false\b|NULL\b|nullptr\b|sizeof\b|[(!*&~+\-]|[A-Za-z_]))",
        "return ",
        x,
    )
    x = re.sub(r"\b(if|while|for|switch)(?=\()", r"\1 ", x)

    # Collapsed C type/specifier pairs.
    type_words = (
        "void|char|short|int|long|float|double|bool|_Bool|size_t|ssize_t|"
        "idx_t|ptrdiff_t|intptr_t|uintptr_t|int8_t|int16_t|int32_t|int64_t|"
        "uint8_t|uint16_t|uint32_t|uint64_t|uintmax_t|intmax_t|FILE"
    )
    spec_words = "static|extern|inline|const|volatile|signed|unsigned|restrict"
    x = re.sub(rf"\b({type_words})(?=const\b)", r"\1 ", x)
    x = re.sub(rf"\b({spec_words})(?=({type_words})\b)", r"\1 ", x)
    x = re.sub(r"\blong(?=long\b)", "long ", x)
    x = re.sub(r"\bstruct(?=[A-Za-z_][A-Za-z0-9_]*\b)", "struct ", x)
    x = re.sub(r"\benum(?=[A-Za-z_][A-Za-z0-9_]*\b)", "enum ", x)
    x = re.sub(r"\bunion(?=[A-Za-z_][A-Za-z0-9_]*\b)", "union ", x)

    # Pointer and parameter punctuation.
    x = re.sub(rf"\b({type_words})\s*(\*+)\s*([A-Za-z_])", r"\1 \2\3", x)
    x = re.sub(r"\s*,\s*", ", ", x)
    x = re.sub(r"\s+([;\)\]\}])", r"\1", x)
    x = re.sub(r"([\(\[\{])\s+", r"\1", x)

    # Very common argv-style collapses seen in decoded/eval views.
    if slot_type == "FILL_PARAM":
        x = re.sub(r"\bint(argc|fd|rc|n|i|j|k|style)\b", r"int \1", x)
        x = re.sub(r"\bsize_t([A-Za-z_][A-Za-z0-9_]*)\b", r"size_t \1", x)
        x = re.sub(r"\bidx_t([A-Za-z_][A-Za-z0-9_]*)\b", r"idx_t \1", x)
        x = re.sub(r"\bssize_t([A-Za-z_][A-Za-z0-9_]*)\b", r"ssize_t \1", x)
        x = re.sub(r"\bptrdiff_t([A-Za-z_][A-Za-z0-9_]*)\b", r"ptrdiff_t \1", x)
        x = re.sub(r"\bchar(\*+)(argv|envp|name|path|buf|str)\b", r"char \1\2", x)
        x = re.sub(r"\bchar\s+const\s*(\*+)\s*([A-Za-z_])", r"char const \1\2", x)
        x = re.sub(r"\bvoid\s+const\s*(\*+)\s*([A-Za-z_])", r"void const \1\2", x)
        x = re.sub(r"\bconst char(\*+)(name|path|buf|str)\b", r"const char \1\2", x)

    x = re.sub(r"\s+", " ", x).strip()
    return x


def _replace_identifier(text: str, name: str, replacement: str = "func") -> str:
    name = str(name or "").strip()
    if not name or name == replacement or len(name) < 2:
        return str(text or "")
    return re.sub(
        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
        replacement,
        str(text or ""),
    )


def _looks_like_function_definition_fragment(text: str) -> bool:
    x = _normalise_teacher_text(text)
    if not x or "{" not in x or "(" not in x or ")" not in x:
        return False
    if x.lstrip().startswith(("if ", "while ", "for ", "switch ", "do ")):
        return False
    ident = r"[A-Za-z_][A-Za-z0-9_]*"
    typeish = (
        rf"(?:static|extern|inline|const|volatile|signed|unsigned|restrict|"
        rf"struct\s+{ident}|enum\s+{ident}|union\s+{ident}|{ident}|\*)"
    )
    return re.search(
        rf"(?<![A-Za-z0-9_])(?:{typeish})(?:\s+|\s*\*)+{ident}\s*\([^;{{}}]*\)\s*\{{",
        x,
    ) is not None


def _is_bad_teacher_target(slot_type: str, text: str) -> bool:
    x = _pretty_c_fragment(text, slot_type)
    if not x:
        return True
    if x in {"{", "}", ";", "};", "(", ")", "0"}:
        return True
    if re.fullmatch(r"[{};()\s]+", x):
        return True
    if slot_type == "FILL_COND":
        if x.startswith(("return ", "case ", "break", "continue", "goto ")):
            return True
        if x.endswith(";") or "{" in x or "}" in x:
            return True
        if x.endswith(","):
            return True
    elif slot_type in {"FILL_STMT", "FILL_EXPR"}:
        if x.startswith("else ") or x == "else":
            return True
        if x.endswith(","):
            return True
        if x in {"0", "1"} and slot_type == "FILL_STMT":
            return True
        if x in {"{", "}"}:
            return True
        if re.search(r",\s*;+", x):
            return True
        par, brk, brc = _c_delimiter_balance(x)
        if par > 0 or brk > 0 or brc > 0:
            return True
        if _looks_like_function_definition_fragment(x):
            return True
    return False


def _target_token_len(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def _count_semicolons_outside_strings(text: str) -> int:
    n = 0
    in_s = False
    in_c = False
    esc = False
    for ch in str(text or ""):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == ";":
            n += 1
    return n


def _c_delimiter_balance(text: str) -> Tuple[int, int, int]:
    """Return rough ``(paren, bracket, brace)`` balance outside strings."""
    par = brk = brc = 0
    in_s = False
    in_c = False
    esc = False
    for ch in str(text or ""):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "(":
            par += 1
        elif ch == ")":
            par -= 1
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk -= 1
        elif ch == "{":
            brc += 1
        elif ch == "}":
            brc -= 1
    return par, brk, brc


def _line_numbers_for_slot(slot: SlotInfo, block_src_lines: List[List[int]]) -> List[int]:
    lines: List[int] = []
    for bb in slot.aligned_bb_indices:
        if isinstance(bb, int) and 0 <= bb < len(block_src_lines):
            for ln in block_src_lines[bb]:
                try:
                    lines.append(int(ln))
                except Exception:
                    pass
    return sorted(set(lines))


def _source_lines_have_statement_content(
    line_nums: List[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]] = None,
) -> bool:
    """Whether the aligned source lines contain a real statement fragment.

    DWARF line tables often map a synthetic return/epilogue BB to the function
    closing brace.  Looking around such a line makes the teacher steal a nearby
    source statement and attach it to the epilogue slot.  For statement slots,
    require the aligned lines themselves to contain something other than braces
    or control separators before doing any nearby-statement search.
    """
    for ln in _clamp_lines_to_bounds(line_nums, bounds):
        raw = _normalise_teacher_text(lt.get(int(ln), ""))
        if not raw:
            continue
        stripped = re.sub(r"[{}\s;]+", "", raw)
        if not stripped:
            continue
        if stripped == "else":
            continue
        return True
    return False


def _join_source_lines(line_nums: List[int], lt: Dict[int, str]) -> str:
    parts: List[str] = []
    for ln in sorted(set(int(x) for x in line_nums)):
        if ln not in lt:
            continue
        x = _normalise_teacher_text(lt[ln])
        if x:
            parts.append(x)
    return " ".join(parts)


def _statement_looks_complete(text: str) -> bool:
    x = str(text or "").strip()
    if not x:
        return True
    par, brk, _brc = _c_delimiter_balance(x)
    if par > 0 or brk > 0:
        return False
    if x.endswith((';', '}', ':')):
        return True
    # A control header without its body is complete enough for condition
    # extraction; statement slots will reject/control-clean it later.
    if any(_extract_balanced_paren_after_keyword(x, kw) for kw in ("if", "while", "for", "switch")):
        return True
    return False


def _extend_source_span_forward(
    line_nums: List[int],
    lt: Dict[int, str],
    *,
    max_extra: int = 16,
    bounds: Optional[Tuple[int, int]] = None,
) -> List[int]:
    if not line_nums:
        return []
    if bounds is not None:
        lo_b, hi_b = bounds
        line_nums = [ln for ln in line_nums if lo_b <= int(ln) <= hi_b]
        if not line_nums:
            return []
    out = sorted(set(line_nums))
    max_line = max(lt) if lt else max(out)
    if bounds is not None:
        max_line = min(max_line, bounds[1])
    cur = max(out)
    while cur < max_line and cur < max(out) + max_extra:
        raw = _join_source_lines(out, lt)
        if _statement_looks_complete(raw):
            break
        cur += 1
        if cur in lt:
            out.append(cur)
    return sorted(set(out))


def _build_char_line_index(line_nums: List[int], lt: Dict[int, str]) -> Tuple[str, List[Tuple[int, int, int]]]:
    text_parts: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    pos = 0
    for ln in line_nums:
        line = _normalise_teacher_text(lt.get(ln, ""))
        if not line:
            continue
        if text_parts:
            text_parts.append("\n")
            pos += 1
        start = pos
        text_parts.append(line)
        pos += len(line)
        spans.append((ln, start, pos))
    return "".join(text_parts), spans


def _line_for_char_pos(spans: List[Tuple[int, int, int]], pos: int) -> Optional[int]:
    for ln, start, end in spans:
        if start <= pos < end:
            return ln
    if spans and pos >= spans[-1][2]:
        return spans[-1][0]
    return None


def _balanced_brace_end(text: str, open_brace: int) -> int:
    depth = 0
    in_s = False
    in_c = False
    esc = False
    for i in range(open_brace, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _function_body_line_range(record: Dict[str, Any], lt: Dict[int, str]) -> Optional[Tuple[int, int]]:
    """Best-effort current function body bounds, used to keep teacher slices local."""
    fn = str(record.get("func_name") or "").strip()
    if not fn or not lt:
        return None
    line_nums = sorted(lt)
    text, spans = _build_char_line_index(line_nums, lt)
    if not text:
        return None
    name_re = re.escape(fn)
    candidates: List[Tuple[int, int, int]] = []
    for m in re.finditer(rf"(?<![A-Za-z0-9_]){name_re}(?![A-Za-z0-9_])\s*\(", text):
        open_paren = text.find("(", m.end() - 1)
        close_paren = _balanced_param_end(text, open_paren) if open_paren >= 0 else -1
        if close_paren < 0:
            continue
        after = text[close_paren + 1 :]
        brace_rel = after.find("{")
        semi_rel = after.find(";")
        if brace_rel < 0 or (semi_rel >= 0 and semi_rel < brace_rel):
            continue
        open_brace = close_paren + 1 + brace_rel
        close_brace = _balanced_brace_end(text, open_brace)
        if close_brace < 0:
            continue
        start_ln = _line_for_char_pos(spans, open_brace)
        end_ln = _line_for_char_pos(spans, close_brace)
        if start_ln is not None and end_ln is not None:
            candidates.append((start_ln, end_ln, m.start()))
    if not candidates:
        return None
    body_lines = sorted(
        {
            int(ln)
            for group in (record.get("block_src_lines") or [])
            if isinstance(group, list)
            for ln in group
            if isinstance(ln, (int, float)) or str(ln).isdigit()
        }
    )
    if body_lines:
        anchor = min(body_lines)
        containing = [c for c in candidates if c[0] <= anchor <= c[1]]
        if containing:
            start, end, _ = min(containing, key=lambda c: (c[1] - c[0], abs(c[0] - anchor)))
            return start, end
        start, end, _ = min(candidates, key=lambda c: min(abs(c[0] - anchor), abs(c[1] - anchor)))
        return start, end
    start, end, _ = candidates[0]
    return start, end


def _clamp_lines_to_bounds(
    line_nums: List[int],
    bounds: Optional[Tuple[int, int]],
) -> List[int]:
    if bounds is None:
        return sorted(set(int(x) for x in line_nums))
    lo, hi = bounds
    return sorted({int(x) for x in line_nums if lo <= int(x) <= hi})


def _split_semicolon_statements(text: str) -> List[str]:
    """Split a C-ish fragment into semicolon-terminated top-level statements."""
    s = str(text or "")
    out: List[str] = []
    start = 0
    par = brk = 0
    in_s = False
    in_c = False
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "(":
            par += 1
        elif ch == ")":
            par -= 1
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk -= 1
        elif ch == ";" and par <= 0 and brk <= 0:
            frag = s[start : i + 1].strip()
            frag = re.sub(r"^[{}\s]+", "", frag)
            frag = re.sub(r"[{}\s]+$", "", frag)
            if frag:
                out.append(frag)
            start = i + 1
    tail = s[start:].strip()
    tail = re.sub(r"^[{}\s]+", "", tail)
    tail = re.sub(r"[{}\s]+$", "", tail)
    if tail:
        out.append(tail)
    return out


def _split_top_level_semicolon_parts(text: str) -> List[str]:
    """Split a C-ish expression list on top-level semicolons without keeping them."""
    s = str(text or "")
    parts: List[str] = []
    start = 0
    par = brk = 0
    in_s = False
    in_c = False
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "(":
            par += 1
        elif ch == ")":
            par -= 1
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk -= 1
        elif ch == ";" and par <= 0 and brk <= 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def _first_statement_fragment(text: str) -> str:
    stmts = _split_semicolon_statements(text)
    if stmts:
        return stmts[0]
    return str(text or "").strip()


def _source_statement_candidates(
    line_nums: List[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]],
) -> List[Tuple[str, int]]:
    wanted = set(_clamp_lines_to_bounds(line_nums, bounds))
    if not wanted:
        return []
    lo = max(min(lt) if lt else 1, min(wanted) - 8)
    hi = min(max(lt) if lt else max(wanted), max(wanted) + 18)
    if bounds is not None:
        lo = max(lo, bounds[0])
        hi = min(hi, bounds[1])
    window_lines = [ln for ln in range(lo, hi + 1) if ln in lt]
    text, spans = _build_char_line_index(window_lines, lt)
    candidates: List[Tuple[str, int]] = []
    start = 0
    par = brk = 0
    in_s = False
    in_c = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "(":
            par += 1
        elif ch == ")":
            par -= 1
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk -= 1
        elif ch == ";" and par <= 0 and brk <= 0:
            frag = text[start : i + 1].strip()
            frag = re.sub(r"^[{}\s]+", "", frag)
            frag = re.sub(r"[{}\s]+$", "", frag)
            if frag:
                frag = _clean_statement_candidate_fragment(frag)
                if not frag:
                    start = i + 1
                    continue
                overlap_lines = [ln for ln, a, b in spans if ln in wanted and not (b <= start or a >= i + 1)]
                if overlap_lines:
                    dist = 0
                else:
                    frag_ln = _line_for_char_pos(spans, start) or lo
                    dist = min(abs(frag_ln - ln) for ln in wanted)
                candidates.append((frag, dist))
            start = i + 1
    return candidates


def _statement_declared_names(stmt: str) -> List[str]:
    s = _pretty_c_fragment(stmt, "FILL_STMT")
    if not s or not s.endswith(";"):
        return []
    if re.match(r"^(?:return|if|while|for|switch|break|continue|goto)\b", s):
        return []
    if "=" not in s and not re.search(r"\b(?:struct|enum|union|char|short|int|long|float|double|bool|_Bool|size_t|ssize_t|idx_t|ptrdiff_t|FILE)\b", s):
        return []
    before = s.split("=", 1)[0].rstrip(";").strip()
    before = before.split(",", 1)[0].strip()
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$", before)
    if not m:
        return []
    name = m.group(1)
    if name in {"return", "sizeof", "if", "while", "for", "switch"}:
        return []
    type_prefix = before[: m.start()].strip()
    if not type_prefix:
        return []
    if not re.search(
        r"(?:\*|\b(?:struct|enum|union|char|short|int|long|float|double|bool|_Bool|size_t|ssize_t|idx_t|ptrdiff_t|FILE|const|unsigned|signed|static|extern)\b|[A-Za-z_][A-Za-z0-9_]*_t\b)",
        type_prefix,
    ):
        return []
    return [name]


def _dedupe_statements_fragment(text: str, declared: Optional[set] = None) -> str:
    stmts = _split_semicolon_statements(text)
    if not stmts:
        return str(text or "").strip()
    seen: set = set()
    declared_names = declared if declared is not None else set()
    out: List[str] = []
    for stmt in stmts:
        norm = re.sub(r"\s+", " ", _pretty_c_fragment(stmt, "FILL_STMT")).strip()
        if not norm or norm in seen:
            continue
        names = _statement_declared_names(norm)
        if names and any(name in declared_names for name in names):
            continue
        seen.add(norm)
        for name in names:
            declared_names.add(name)
        out.append(norm)
    return " ".join(out)


def _control_span_end(text: str, match: re.Match[str]) -> int:
    start = text.find("(", match.end() - 1)
    if start < 0:
        return -1
    depth = 0
    in_s = False
    in_c = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _span_overlaps_wanted_lines(
    spans: List[Tuple[int, int, int]],
    wanted: set,
    start: int,
    end: int,
) -> bool:
    """Whether a char span touches any wanted source line."""
    if end <= start:
        return False
    for ln, a, b in spans:
        if ln in wanted and not (b <= start or a >= end):
            return True
    return False


def _simple_statement_span_from_pos(text: str, pos: int) -> Optional[Tuple[int, int]]:
    """Return the next simple statement/body span starting at or after ``pos``."""
    s = str(text or "")
    i = int(pos)
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    if i >= n:
        return None
    if s.startswith("else", i) and (i + 4 == n or not s[i + 4].isalnum()):
        return None
    if s[i] == "{":
        end = _balanced_brace_end(s, i)
        if end >= 0:
            return i, end + 1
        return None

    start = i
    par = brk = 0
    in_s = False
    in_c = False
    esc = False
    for j in range(i, n):
        ch = s[j]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "'":
            in_c = True
        elif ch == "(":
            par += 1
        elif ch == ")":
            par -= 1
        elif ch == "[":
            brk += 1
        elif ch == "]":
            brk -= 1
        elif ch == ";" and par <= 0 and brk <= 0:
            return start, j + 1
        elif ch == "}" and par <= 0 and brk <= 0:
            return start, j
    return None


def _first_control_body_statement(text: str) -> str:
    """Extract ``stmt;`` from fragments like ``if (cond) stmt;``."""
    s = str(text or "").strip()
    m = re.search(r"\b(?:if|while|for|switch)\s*\(", s)
    if not m:
        return ""
    end = _control_span_end(s, m)
    if end < 0:
        return ""
    span = _simple_statement_span_from_pos(s, end)
    if span is None:
        return ""
    start, stop = span
    body = s[start:stop].strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1].strip()
    body = _first_statement_fragment(body)
    body = re.sub(r"^[{}\s]+", "", body)
    body = re.sub(r"[{}\s]+$", "", body).strip()
    return body


def _clean_statement_candidate_fragment(text: str) -> str:
    """Trim source-window spillover from a candidate statement fragment."""
    frag = str(text or "").strip()
    if not frag:
        return ""
    body = _first_control_body_statement(frag)
    if body:
        return body
    last_brace = frag.rfind("{")
    if last_brace >= 0:
        tail = frag[last_brace + 1 :].strip()
        body = _first_control_body_statement(tail)
        if body:
            return body
        stmts = _split_semicolon_statements(tail)
        if stmts:
            return stmts[0]
    return frag


def _control_spans_near_lines(
    line_nums: List[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]] = None,
    *,
    lookback: int = 16,
    lookahead: int = 16,
) -> List[Tuple[int, int, int, Optional[Tuple[int, int]], str]]:
    """Return control header spans near aligned source lines.

    Each tuple is ``(header_start, header_end, distance, body_span, text)``.
    ``distance`` is 0 when the wanted lines touch the header or body.
    """
    wanted = set(_clamp_lines_to_bounds(line_nums, bounds))
    if not wanted:
        return []
    lo = max(1, min(wanted) - int(lookback))
    hi = min(max(lt) if lt else max(wanted), max(wanted) + int(lookahead))
    if bounds is not None:
        lo = max(lo, bounds[0])
        hi = min(hi, bounds[1])
    window_lines = [ln for ln in range(lo, hi + 1) if ln in lt]
    text, spans = _build_char_line_index(window_lines, lt)
    out: List[Tuple[int, int, int, Optional[Tuple[int, int]], str]] = []
    for m in re.finditer(r"\b(?:if|while|for|switch)\s*\(", text):
        end = _control_span_end(text, m)
        if end < 0:
            continue
        body_span = _simple_statement_span_from_pos(text, end)
        header_hit = _span_overlaps_wanted_lines(spans, wanted, m.start(), end)
        body_hit = (
            body_span is not None
            and _span_overlaps_wanted_lines(spans, wanted, body_span[0], body_span[1])
        )
        if header_hit or body_hit:
            dist = 0
        else:
            header_ln = _line_for_char_pos(spans, m.start()) or lo
            dist = min(abs(header_ln - ln) for ln in wanted)
        out.append((m.start(), end, dist, body_span, text))
    return out


def _source_slice_for_condition(
    line_nums: List[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]] = None,
) -> str:
    if not line_nums:
        return ""
    line_nums = _clamp_lines_to_bounds(line_nums, bounds)
    if not line_nums:
        return ""
    exact = _extend_source_span_forward(line_nums, lt, max_extra=12, bounds=bounds)
    raw = _join_source_lines(exact, lt)
    if any(_extract_balanced_paren_after_keyword(raw, kw) for kw in ("if", "while", "for", "switch")):
        return raw

    # If the BB line is a condition continuation or a one-line control body
    # such as "fseeko (...);", search nearby source for the owning header.
    control_spans = _control_spans_near_lines(line_nums, lt, bounds, lookback=18, lookahead=12)
    if not control_spans:
        return raw

    start, end, _dist, _body_span, text = sorted(
        control_spans,
        key=lambda x: (
            x[2],
            abs(x[0] - x[4].find(raw[:20])) if raw[:20] and x[4].find(raw[:20]) >= 0 else x[0],
        ),
    )[0]
    # Include simple prefix statements immediately before the control header
    # when they are on the same aligned BB slice; avoid pulling unrelated prior
    # source context from the search window.
    prefix = ""
    exact_raw = _join_source_lines(line_nums, lt)
    first_kw = re.search(r"\b(?:if|while|for|switch)\s*\(", exact_raw)
    if first_kw:
        prefix = exact_raw[: first_kw.start()].strip()
    return (prefix + " " + text[start:end]).strip()


def _source_slice_for_statement(
    line_nums: List[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]] = None,
) -> str:
    if not line_nums:
        return ""
    line_nums = _clamp_lines_to_bounds(line_nums, bounds)
    if not line_nums:
        return ""
    if not _source_lines_have_statement_content(line_nums, lt, bounds):
        return ""
    cands = _source_statement_candidates(line_nums, lt, bounds)
    if cands:
        ranked = sorted(cands, key=lambda x: (x[1], _target_token_len(x[0])))
        best = ranked[0][0]
        if re.match(r"\s*(?:if|while|for|switch)\s*\(", _pretty_c_fragment(best, "FILL_STMT")):
            body = _first_control_body_statement(best)
            if body:
                return body
        return best
    expanded = _extend_source_span_forward(line_nums, lt, max_extra=18, bounds=bounds)
    raw = _join_source_lines(expanded, lt)
    if re.match(r"\s*(?:if|while|for|switch)\s*\(", _pretty_c_fragment(raw, "FILL_STMT")):
        body = _first_control_body_statement(raw)
        if body:
            return body
    stmts = _split_semicolon_statements(raw)
    if stmts:
        return stmts[0]
    return raw


def _source_slice_for_header_prefix(
    line_nums: List[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]] = None,
) -> str:
    """Return contiguous setup statements immediately before a control header.

    Header BBs often map to sparse source lines like ``decl; if (cond) body;``.
    DWARF may omit nearby local declarations from ``block_src_lines`` even
    though they are part of the same BB setup.  For explicit header-prefix
    slots, recover contiguous declaration/assignment/call statements that
    appear just before the matched control header, while stopping at previous
    control-flow statements or distant/comment-separated code.
    """
    lines = _clamp_lines_to_bounds(line_nums, bounds)
    if not lines or not lt:
        return ""
    control_spans = _control_spans_near_lines(lines, lt, bounds, lookback=12, lookahead=8)
    if control_spans:
        start, _end, _dist, _body_span, text = sorted(control_spans, key=lambda x: (x[2], x[0]))[0]
        prefix_text = text[:start].strip()
        stmts = _split_semicolon_statements(prefix_text)
        if stmts:
            kept: List[str] = []
            for stmt in reversed(stmts):
                frag = _pretty_c_fragment(stmt, "FILL_STMT").strip()
                if not frag:
                    continue
                if re.match(r"^(?:if|while|for|switch|else|return|break|continue|goto|case)\b", frag):
                    break
                if re.search(r"\b(?:if|while|for|switch)\s*\(", frag):
                    break
                if not frag.endswith(";"):
                    frag += ";"
                kept.append(frag)
            if kept:
                return _dedupe_statements_fragment(" ".join(reversed(kept)))

    # Fallback: walk directly upward from the first control-looking aligned line.
    header_line = None
    for ln in sorted(lines):
        raw = _normalise_teacher_text(lt.get(int(ln), ""))
        if re.search(r"\b(?:if|while|for|switch)\s*\(", raw):
            header_line = int(ln)
            break
    if header_line is None:
        return ""

    lo = bounds[0] if bounds is not None else min(lt)
    kept_lines: List[int] = []
    prev = header_line
    cur = header_line - 1
    while cur >= lo:
        raw = _normalise_teacher_text(lt.get(cur, ""))
        if not raw:
            cur -= 1
            continue
        if prev - cur > 3:
            break
        frag = _pretty_c_fragment(raw, "FILL_STMT")
        if (
            not frag
            or frag in {"{", "}", ";", "};"}
            or re.match(r"^(?:if|while|for|switch|else|return|break|continue|goto|case)\b", frag)
            or re.search(r"\b(?:if|while|for|switch)\s*\(", frag)
        ):
            break
        kept_lines.append(cur)
        prev = cur
        cur -= 1
    if not kept_lines:
        return ""
    raw = _join_source_lines(sorted(kept_lines), lt)
    return _dedupe_statements_fragment(raw)


def _strict_reject_teacher_target(slot_type: str, text: str, raw_text: str) -> str:
    """Return a short reject reason, or empty string if strict target is usable."""
    x = _pretty_c_fragment(text, slot_type)
    raw = _pretty_c_fragment(raw_text, slot_type)
    if not x:
        return "empty"
    if slot_type == "FILL_COND":
        if raw.endswith(";") or _count_semicolons_outside_strings(raw) > 0:
            if not re.search(r"\b(?:if|while|for|switch)\s*\(", raw):
                return "cond_statement_fragment"
        # In strict mode, conditions must really come from a control predicate.
        # This prevents assignment lines from being taught as conditions.
        control_conds = [
            _extract_balanced_paren_after_keyword(raw, kw)
            for kw in ("if", "while", "for", "switch")
        ]
        if not any(control_conds):
            # Some alignments point at a condition continuation/expression
            # rather than the line containing "if (". Accept expression-like
            # fragments, but reject obvious statements accidentally mapped to a
            # condition slot.
            if (
                raw.endswith(";")
                or raw.startswith(("return ", "case ", "break", "continue", "goto "))
                or re.search(r"\b(?:if|while|for|switch)\b", raw)
                or _count_semicolons_outside_strings(raw) > 0
            ):
                return "cond_not_from_control"
        first_control = re.search(r"\b(?:if|while|for|switch)\s*\(", raw)
        if first_control:
            prefix = raw[: first_control.start()].strip()
            # A BB source slice can be "decl/init; if (cond)". Those prefix
            # statements are now injected back into the skeleton before the
            # control statement, so they are no longer a strict rejection by
            # themselves.
        if re.match(r"^(?:if|while|for|switch)\s*\(", x):
            # The skeleton already renders "if (<FILL_COND>)"; a target that
            # still starts with "if (" becomes "if (if (...))".
            return "cond_contains_control_header"
        if _target_token_len(x) > 80:
            return "cond_too_long"
        if _count_semicolons_outside_strings(x) > 0:
            return "cond_has_semicolon"
    elif slot_type == "FILL_EXPR":
        if _looks_like_function_definition_fragment(x):
            return "body_contains_function_definition"
        if _target_token_len(x) > 80:
            return "expr_too_long"
        if _count_semicolons_outside_strings(x) > 0:
            return "expr_has_semicolon"
    elif slot_type == "FILL_STMT":
        if _looks_like_function_definition_fragment(x):
            return "body_contains_function_definition"
        semi = _count_semicolons_outside_strings(x)
        if _target_token_len(x) > 120:
            return "stmt_too_long"
        if re.search(r"\b(if|while|for|switch)\s*\(", x):
            return "stmt_contains_control"
    return ""


def _extract_balanced_paren_after_keyword(text: str, keyword: str) -> str:
    m = re.search(rf"\b{re.escape(keyword)}\s*\(", text)
    if not m:
        return ""
    start = text.find("(", m.end() - 1)
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i].strip()
    return ""


def _clean_teacher_target_for_slot(slot_type: str, text: str) -> Tuple[str, bool]:
    """Return (clean_target, usable). Empty unusable slots are left unsupervised."""
    x = _pretty_c_fragment(text, slot_type)
    if slot_type in {"FILL_TYPE", "FILL_PARAM"}:
        return x, bool(x)
    if not x:
        return "", False

    if slot_type == "FILL_COND":
        for kw in ("if", "while", "for", "switch"):
            cond = _extract_balanced_paren_after_keyword(x, kw)
            if cond:
                if kw == "for":
                    parts = _split_top_level_semicolon_parts(cond)
                    if len(parts) >= 2:
                        cond = parts[1] or "1"
                return cond, True
        y = x.strip()
        y = re.sub(r"^[{;\s]+", "", y)
        y = re.sub(r"[{;]\s*$", "", y).strip()
        if _is_bad_teacher_target(slot_type, y):
            return "", False
        return y, True

    if slot_type in {"FILL_STMT", "FILL_EXPR"}:
        y = x.strip()
        y = re.sub(r"^[{}\s]+", "", y)
        y = re.sub(r"[{}\s]+$", "", y)
        if not y:
            return "", False
        y = _first_statement_fragment(y)
        # A slice like "else error(...);" means the source branch boundary
        # does not match the predicted skeleton. Rendering it inside the
        # current block produces invalid C ("; else ..."), so leave the slot
        # unsupervised instead of teaching a misplaced branch body.
        if y.startswith("else ") or y == "else":
            return "", False
        # If a statement slice contains prefix code followed by a control
        # header, keep only the prefix statements. The control itself is
        # represented by a FILL_COND slot in the skeleton.
        m_ctrl = re.search(r"\b(?:if|while|for|switch)\s*\(", y)
        if m_ctrl:
            prefix = y[: m_ctrl.start()].strip()
            if prefix and _count_semicolons_outside_strings(prefix) > 0:
                y = prefix
            else:
                return "", False
        if slot_type == "FILL_EXPR":
            y = re.sub(r"^return\b", "", y).strip()
            y = y.rstrip(";").strip()
        elif not y.endswith(";") and not y.endswith("}") and not y.startswith(("if ", "while ", "for ", "switch ")):
            y += ";"
        if _is_bad_teacher_target(slot_type, y):
            return "", False
        return y, True

    return x, not _is_bad_teacher_target(slot_type, x)


def _cond_prefix_statements(raw_text: str) -> str:
    """Return statements before the first control header in a raw condition slice.

    Stage2 block-line alignment sometimes maps a BB to source text like
    ``int fd = fileno(fp); if (fd < 0)``. The condition target should remain
    ``fd < 0``, but the prefix statements must be present in the skeleton or the
    teacher C has undeclared variables. This helper extracts the prefix so it
    can be inserted as non-supervised skeleton text before the control.
    """
    raw = _pretty_c_fragment(raw_text, "FILL_STMT")
    if not raw:
        return ""
    m = re.search(r"\b(?:if|while|for|switch)\s*\(", raw)
    if not m:
        return ""
    prefix = raw[: m.start()].strip()
    prefix = re.sub(r"^[{;\s]+", "", prefix).strip()
    prefix = re.sub(r"[{}\s]+$", "", prefix).strip()
    if not prefix or _count_semicolons_outside_strings(prefix) <= 0:
        return ""
    if re.search(r"\b(?:if|while|for|switch|else)\b", prefix):
        return ""
    prefix = _dedupe_statements_fragment(prefix)
    if not prefix:
        return ""
    if not prefix.endswith(";"):
        prefix += ";"
    return prefix


def _slot_index_by_id(slots: List[SlotInfo]) -> Dict[str, int]:
    return {str(getattr(slot, "slot_id", "")): i for i, slot in enumerate(slots)}


def _normalise_cond_placeholder_punctuation(skeleton_text: str) -> str:
    """Remove punctuation that Module-B may leave inside condition parens."""
    s = str(skeleton_text or "")
    # Broken predicted skeletons can contain ``if (<FILL_COND:slot_2>;)``.
    # The semicolon belongs to statement syntax, not to the condition target.
    s = re.sub(r"(<FILL_COND:[^>]+>)\s*;+(\s*\))", r"\1\2", s)
    return s


def _slot_indices_in_skeleton_order(skeleton_text: str, slots: List[SlotInfo]) -> List[int]:
    """Return slot indices ordered by placeholder occurrence in skeleton text."""
    slot_by_id = _slot_index_by_id(slots)
    out: List[int] = []
    for pos, m in enumerate(_FILL_RE.finditer(skeleton_text or "")):
        idx = slot_by_id.get(m.group(2), pos)
        if 0 <= idx < len(slots):
            out.append(idx)
    return out


def _split_condition_prefix_and_cond(raw_text: str) -> Tuple[str, str]:
    """Return ``(prefix_statements, control_condition)`` from a raw header slice."""
    raw = _pretty_c_fragment(raw_text, "FILL_STMT")
    if not raw:
        return "", ""
    m = re.search(r"\b(if|while|for|switch)\s*\(", raw)
    if not m:
        return "", ""
    prefix = raw[: m.start()].strip()
    prefix = re.sub(r"^[{;\s]+", "", prefix).strip()
    prefix = re.sub(r"[{}\s]+$", "", prefix).strip()
    if prefix and not re.search(r"\b(?:if|while|for|switch|else)\b", prefix):
        prefix = _dedupe_statements_fragment(prefix)
        if prefix and not prefix.endswith(";"):
            prefix += ";"
    else:
        prefix = ""
    cond = _extract_balanced_paren_after_keyword(raw[m.start():], m.group(1))
    if m.group(1) == "for" and cond:
        parts = _split_top_level_semicolon_parts(cond)
        if len(parts) >= 2:
            cond = parts[1] or "1"
    return prefix, _pretty_c_fragment(cond, "FILL_COND") if cond else ""


def _slot_primary_bb(slot: SlotInfo) -> Optional[int]:
    bbs = [int(x) for x in getattr(slot, "aligned_bb_indices", []) or []]
    return bbs[0] if bbs else None


def _slot_is_header_prefix(slot: SlotInfo) -> bool:
    return "role=header_prefix" in str(getattr(slot, "binary_context_text", ""))


def _inject_cond_prefix_statements(
    skeleton_text: str,
    slots: List[SlotInfo],
    raw_gt_texts: List[str],
) -> Tuple[str, Dict[int, str], Dict[int, int]]:
    """Assign header prefix targets to real prefix FILL_STMT slots.

    Older data construction inserted source prefix statements as literal text
    before ``if (<FILL_COND>)``.  That fixed rendered GT but leaked teacher text
    into the prompt.  The skeleton builder now emits explicit ``FILL_STMT``
    slots aligned to the same header BB; this helper moves the prefix target to
    those slots and leaves the condition slot with just the control predicate.
    """
    inserted: Dict[int, str] = {}
    prefix_slot_for_cond: Dict[int, int] = {}
    slot_by_id = _slot_index_by_id(slots)
    ordered = [
        slot_by_id.get(m.group(2), pos)
        for pos, m in enumerate(_FILL_RE.finditer(skeleton_text or ""))
    ]
    for ord_pos, idx in enumerate(ordered):
        if not (0 <= idx < len(slots)) or slots[idx].slot_type != "FILL_COND":
            continue
        prefix, cond = _split_condition_prefix_and_cond(raw_gt_texts[idx] if idx < len(raw_gt_texts) else "")
        if cond:
            raw_gt_texts[idx] = cond
        if not prefix:
            continue
        bb = _slot_primary_bb(slots[idx])
        if bb is None:
            continue
        best: Optional[int] = None
        for prev_idx in reversed(ordered[:ord_pos]):
            if not (0 <= prev_idx < len(slots)):
                continue
            prev_slot = slots[prev_idx]
            if prev_slot.slot_type != "FILL_STMT":
                continue
            if _slot_primary_bb(prev_slot) != bb:
                continue
            if str(raw_gt_texts[prev_idx] if prev_idx < len(raw_gt_texts) else "").strip():
                continue
            best = prev_idx
            break
        if best is None:
            continue
        raw_gt_texts[best] = prefix
        inserted[best] = prefix
        prefix_slot_for_cond[idx] = best
    return skeleton_text, inserted, prefix_slot_for_cond


def _render_skeleton_with_targets(
    skeleton_text: str,
    targets: List[str],
    slots: Optional[List[SlotInfo]] = None,
) -> str:
    i = 0
    slot_by_id = _slot_index_by_id(slots or [])

    def repl(m: re.Match[str]) -> str:
        nonlocal i
        idx = slot_by_id.get(m.group(2), i)
        val = targets[idx] if idx < len(targets) else ""
        i += 1
        return str(val or "")

    return _FILL_RE.sub(repl, skeleton_text or "")


def _iter_control_conditions(text: str) -> Iterable[Tuple[str, str]]:
    s = str(text or "")
    for m in re.finditer(r"\b(if|while|for|switch)\s*\(", s):
        kw = m.group(1)
        start = s.find("(", m.end() - 1)
        if start < 0:
            continue
        depth = 0
        in_s = False
        in_c = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if in_s:
                if ch == '"':
                    in_s = False
                continue
            if in_c:
                if ch == "'":
                    in_c = False
                continue
            if ch == '"':
                in_s = True
            elif ch == "'":
                in_c = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    yield kw, s[start + 1 : i].strip()
                    break


def _strict_reject_rendered_teacher(text: str) -> str:
    """Return a reason if the rendered teacher C has obvious broken syntax."""
    s = str(text or "")
    if re.search(r"\b(?:if|while|for|switch)\s*\(\s*\)", s):
        return "rendered_empty_control_condition"
    if re.search(r"\b(?:if|while|for|switch)\s*\(\s*(?:if|while|for|switch)\s*\(", s):
        return "rendered_nested_control_header"
    if re.search(r",\s*;+", s):
        return "rendered_comma_before_semicolon"
    for kw, cond in _iter_control_conditions(s):
        if kw != "for" and _count_semicolons_outside_strings(cond) > 0:
            return f"rendered_{kw}_condition_has_semicolon"
    return ""


def _repair_condition_targets_from_source(
    slots: List[SlotInfo],
    gt_texts: List[str],
    raw_gt_texts: List[str],
    slot_supervise: List[bool],
    block_src_lines: List[List[int]],
    lt: Dict[int, str],
    source_bounds: Optional[Tuple[int, int]],
    record: Dict[str, Any],
) -> List[int]:
    """Best-effort repair for rendered empty/semicolon condition slots."""
    repaired: List[int] = []
    for i, slot in enumerate(slots):
        if slot.slot_type != "FILL_COND":
            continue
        raw = raw_gt_texts[i] if i < len(raw_gt_texts) else ""
        tgt = gt_texts[i] if i < len(gt_texts) else ""
        needs_repair = (
            not slot_supervise[i]
            or not str(tgt or "").strip()
            or _count_semicolons_outside_strings(tgt) > 0
            or _strict_reject_teacher_target("FILL_COND", tgt, raw)
        )
        if not needs_repair:
            continue
        lines = _clamp_lines_to_bounds(_line_numbers_for_slot(slot, block_src_lines), source_bounds)
        if not lines:
            continue
        cand_raw = _source_slice_for_condition(lines, lt, source_bounds)
        cand_raw = _sanitize_teacher_text(cand_raw, record)
        cand, usable = _clean_teacher_target_for_slot("FILL_COND", cand_raw)
        if not usable:
            continue
        if _strict_reject_teacher_target("FILL_COND", cand, cand_raw):
            continue
        gt_texts[i] = cand
        raw_gt_texts[i] = cand_raw
        slot_supervise[i] = True
        repaired.append(i)
    return repaired


def _strip_inserted_cond_prefix_duplicates(
    gt_texts: List[str],
    raw_gt_texts: List[str],
    slot_supervise: List[bool],
    slots: List[SlotInfo],
    cond_prefix_insertions: Dict[int, str],
) -> List[int]:
    """Remove statement targets that duplicate prefix statements inserted before conditions."""
    if not cond_prefix_insertions:
        return []
    removed: List[int] = []
    prefixes = [
        re.sub(r"\s+", " ", _pretty_c_fragment(v, "FILL_STMT")).strip()
        for v in cond_prefix_insertions.values()
        if str(v or "").strip()
    ]
    if not prefixes:
        return []
    for i, slot in enumerate(slots):
        if slot.slot_type not in {"FILL_STMT", "FILL_EXPR"}:
            continue
        if i >= len(gt_texts) or not slot_supervise[i]:
            continue
        txt = re.sub(r"\s+", " ", _pretty_c_fragment(gt_texts[i], slot.slot_type)).strip()
        raw = re.sub(r"\s+", " ", _pretty_c_fragment(raw_gt_texts[i] if i < len(raw_gt_texts) else "", slot.slot_type)).strip()
        for pref in prefixes:
            if txt == pref or raw == pref or (txt.startswith(pref + " ") and _count_semicolons_outside_strings(pref) > 0):
                gt_texts[i] = ""
                slot_supervise[i] = False
                removed.append(i)
                break
    return removed


def _dedupe_rendered_skeleton_declarations(skeleton_text: str) -> str:
    """Drop repeated plain declaration statements introduced by prefix recovery."""
    declared: set = set()

    def repl(m: re.Match[str]) -> str:
        stmt = m.group(0)
        names = _statement_declared_names(stmt)
        if names and any(name in declared for name in names):
            return ""
        for name in names:
            declared.add(name)
        return stmt

    return re.sub(r"[^;\n{}]*;", repl, skeleton_text or "")


def _line_to_text(source_path: Path) -> Dict[int, str]:
    if not source_path.is_file():
        return {}
    try:
        text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: Dict[int, str] = {}
    for i, line in enumerate(text.splitlines(), 1):
        out[i] = line
    return out


def _strip_preprocessor_lines_from_lt(lt: Dict[int, str]) -> Dict[int, str]:
    if not lt:
        return {}
    out: Dict[int, str] = {}
    for k, line in lt.items():
        if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning)\b", str(line or "")):
            out[k] = ""
        else:
            out[k] = line
    return out


def _strip_block_comments_from_lt(lt: Dict[int, str]) -> Dict[int, str]:
    """Remove C block comments while preserving original line numbers."""
    if not lt:
        return {}
    keys = sorted(lt)
    text = "\n".join(lt[k] for k in keys)

    def repl(m: re.Match[str]) -> str:
        return "\n" * m.group(0).count("\n")

    cleaned = re.sub(r"/\*.*?\*/", repl, text, flags=re.S)
    lines = cleaned.splitlines()
    out: Dict[int, str] = {}
    for i, k in enumerate(keys):
        out[k] = lines[i] if i < len(lines) else ""
    return out


def _rewrite_source_path_string(
    s: str,
    rewrites: List[Tuple[str, str]],
) -> str:
    """将绝对/相对路径里的前缀按顺序替换（用于 /mnt/e/... → 服务器根）。"""
    if not isinstance(s, str) or not s or not rewrites:
        return s
    x = s.replace("\\", "/")
    for frm, to in rewrites:
        ff = str(frm).replace("\\", "/").rstrip("/")
        tt = str(to).replace("\\", "/").rstrip("/")
        if ff and x.startswith(ff):
            x = tt + x[len(ff) :]
    return x


def _resolve_source_path(record: Dict[str, Any], source_root: Optional[Path]) -> Optional[Path]:
    raw = record.get("source_file")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p.resolve()
    if source_root is not None:
        q = (source_root / p).resolve()
        if q.is_file():
            return q
        q2 = (source_root / p.name).resolve()
        if q2.is_file():
            return q2
    return None


def _infer_signature_hint(record: Dict[str, Any]) -> str:
    """提案 §7.3.1 ``Function signature hint``；jsonl 可提供 ``func_sig_hint`` / ``signature_hint``。"""
    hint = record.get("func_sig_hint") or record.get("signature_hint")
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return "unknown (use binary_context per slot)"


def _binary_info_block(
    record: Dict[str, Any],
    signature: Optional[Tuple[str, str, str]] = None,
) -> str:
    """与 StrucLift_Research_Proposal §7.3.1 ``<|binary_info|>`` 块对齐（含 Architecture / Compiler / hint）。"""
    arch = record.get("arch") or record.get("architecture") or "x86_64"
    comp = record.get("compiler") or "unknown (stripped)"
    if signature is not None:
        sig = (
            f"{signature[0]} func({signature[1]})"
            if _signature_is_confident(signature)
            else "unknown func signature; infer from binary context"
        )
    else:
        sig = _infer_signature_hint(record)
        fn = str(record.get("func_name") or "").strip()
        if fn and isinstance(sig, str):
            sig = re.sub(rf"\b{re.escape(fn)}\b", "func", sig)
    return (
        f"<|binary_info|>\n"
        f"Architecture: {arch}\n"
        f"Compiler: {comp}\n"
        f"Function signature hint: {sig}\n"
        f"Function: func\n"
        f"<|/binary_info|>\n"
    )


def _compact_json_value(value: Any, *, max_items: int = 8, max_text: int = 160) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key in sorted(value)[:max_items]:
            out[str(key)] = _compact_json_value(value[key], max_items=max_items, max_text=max_text)
        if len(value) > max_items:
            out["..."] = f"{len(value) - max_items} more"
        return out
    if isinstance(value, list):
        vals = [_compact_json_value(x, max_items=max_items, max_text=max_text) for x in value[:max_items]]
        if len(value) > max_items:
            vals.append(f"... {len(value) - max_items} more")
        return vals
    text = str(value)
    return text[:max_text] + ("..." if len(text) > max_text else "")


def _try_parse_context_json(text: str) -> Optional[Dict[str, Any]]:
    x = str(text or "").strip()
    if not x or not x.startswith("{"):
        return None
    try:
        obj = json.loads(x)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _slot_binary_context_line(
    slot: Any,
    *,
    compact_mode: str = "full",
    max_context_chars: int = 4096,
) -> str:
    bbs = [int(x) for x in getattr(slot, "aligned_bb_indices", []) or []]
    bb_text = ", ".join(f"BB_{b}" for b in bbs) if bbs else "signature/global"
    ctx = str(getattr(slot, "binary_context_text", "") or "").strip()
    if not ctx:
        ctx = "no local binary evidence; use graph adapter and function signature"
    mode = str(compact_mode or "full").lower()
    if mode not in {"full", "compact", "ultra"}:
        mode = "full"
    if mode in {"compact", "ultra"}:
        obj = _try_parse_context_json(ctx)
        if obj is not None:
            if mode == "ultra":
                parts = []
                for key in ("role", "semantic_slot_type", "slot_type"):
                    if obj.get(key):
                        parts.append(f"{key}={obj.get(key)}")
                region = obj.get("region") if isinstance(obj.get("region"), dict) else {}
                for key in ("primary_bbs", "region_bbs", "context_bbs"):
                    if region.get(key):
                        parts.append(f"{key}={_short_join(list(region.get(key) or []), 8)}")
                control = obj.get("control") if isinstance(obj.get("control"), dict) else {}
                if control.get("edge"):
                    parts.append(f"edge={control.get('edge')}")
                data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
                for key in ("live_in", "live_out", "defs", "uses"):
                    if data.get(key):
                        parts.append(f"{key}={_short_join(list(data.get(key) or []), 8)}")
                ctx = "; ".join(parts) if parts else "compact slot contract available"
            else:
                keep: Dict[str, Any] = {}
                for key in ("slot_type", "semantic_slot_type", "role", "anchor", "region", "control", "data", "confidence"):
                    if key in obj:
                        keep[key] = _compact_json_value(obj[key], max_items=8, max_text=120)
                ctx = json.dumps(keep, ensure_ascii=False, sort_keys=True)
        else:
            ctx = re.sub(r"\s+", " ", ctx)
    if max_context_chars > 0 and len(ctx) > int(max_context_chars):
        ctx = ctx[: int(max_context_chars)].rstrip() + "...[truncated]"
    return f"{slot.slot_id} [{getattr(slot, 'slot_type', 'FILL')}; {bb_text}]: {ctx}"


def _sanitize_prompt_text(text: str, record: Dict[str, Any]) -> str:
    """Remove the current stripped function name from prompt-only fields."""
    return _replace_identifier(text, str(record.get("func_name") or ""), "func")


def _sanitize_teacher_text(text: str, record: Dict[str, Any]) -> str:
    """Remove the current stripped function name from supervised targets."""
    return _replace_identifier(text, str(record.get("func_name") or ""), "func")


def _strip_comments(line: str) -> str:
    return re.sub(r"/\*.*?\*/|//.*$", "", line).strip()


UNKNOWN_RET = "__STRUCLIFT_UNKNOWN_RET__"
UNKNOWN_PARAMS = "__STRUCLIFT_UNKNOWN_PARAMS__"


def _signature_is_confident(signature: Tuple[str, str, str]) -> bool:
    return signature[0] != UNKNOWN_RET and signature[1] != UNKNOWN_PARAMS


def _clean_signature_piece(text: str) -> str:
    x = _strip_c_comments(str(text or ""))
    x = re.sub(r"\bATTRIBUTE_[A-Z0-9_]+\s*(\([^)]*\))?", " ", x)
    x = re.sub(r"\b_GL_ATTRIBUTE_[A-Z0-9_]+\s*(\([^)]*\))?", " ", x)
    x = re.sub(r"\b_[A-Z0-9_]*ATTRIBUTE[A-Z0-9_]*\s*(\([^)]*\))?", " ", x)
    x = re.sub(r"\b__attribute__\s*\(\(.*?\)\)", " ", x)
    x = re.sub(r"\b(__THROW|__nonnull\s*\([^)]*\)|__wur|__pure|__const)\b", " ", x)
    x = re.sub(r"^\s*#.*$", " ", x, flags=re.M)
    x = re.sub(r"\s+", " ", x).strip()
    return _pretty_c_fragment(x)


def _looks_like_ret_type(text: str) -> bool:
    x = _clean_signature_piece(text)
    if not x or len(x) > 120:
        return False
    bad_words = {
        "if", "while", "for", "switch", "return", "break", "continue",
        "endif", "else", "see", "messages", "locale", "then",
    }
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", x)
    if not toks or toks[0] in bad_words or toks[-1] in bad_words:
        return False
    if any(op in x for op in ("&&", "||", "==", "!=", "<=", ">=", "=")):
        return False
    type_words = {
        "void", "char", "short", "int", "long", "float", "double", "bool",
        "_Bool", "size_t", "ssize_t", "ptrdiff_t", "uintmax_t", "intmax_t",
        "FILE", "struct", "enum", "union", "static", "extern", "inline",
        "const", "volatile", "signed", "unsigned",
    }
    if any(t in type_words or t.endswith("_t") for t in toks):
        return True
    # Project typedefs such as CURLcode, xmlChar, size aliases, and enum-like
    # return types are valid C signatures even though they are not built-ins.
    if len(toks) <= 6 and all(t not in bad_words for t in toks):
        return True
    return False


def _looks_like_params(text: str) -> bool:
    x = _clean_signature_piece(text)
    if not x:
        return True
    if len(x) > 400:
        return False
    if any(op in x for op in ("&&", "||", "==", "!=", "<=", ">=")):
        return False
    if re.search(r"\b(if|while|for|switch|return|break|continue|else)\b", x):
        return False
    if re.search(r"[{};=]", x):
        return False
    return True


def _balanced_param_end(text: str, open_paren: int) -> int:
    depth = 0
    in_s = False
    in_c = False
    esc = False
    for i in range(open_paren, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_s:
            if ch == '"':
                in_s = False
            continue
        if in_c:
            if ch == "'":
                in_c = False
            continue
        if ch == '"':
            in_s = True
            continue
        if ch == "'":
            in_c = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _signature_candidate_from_match(text: str, fn: str, start: int, end: int) -> Optional[Tuple[int, str, str]]:
    """Extract a source signature around one function-name occurrence."""
    open_paren = text.find("(", end)
    if open_paren < 0:
        return None
    close_paren = _balanced_param_end(text, open_paren)
    if close_paren < 0:
        return None
    after = text[close_paren + 1 :]
    if not re.match(
        r"\s*(?:(?:[A-Za-z_][A-Za-z0-9_]*\s*(?:\([^{};]*\))?\s*)|(?:#.*\n\s*))*\{",
        after,
    ):
        return None

    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    # GNU style often puts the function name on a fresh line after a prototype
    # line, e.g. ``static int\nfoo (...)``.  Join a few previous declaration
    # lines while stopping at statement/declaration boundaries.
    scan = line_start
    for _ in range(12):
        prev_end = scan - 1
        if prev_end <= 0:
            break
        prev_start = text.rfind("\n", 0, prev_end) + 1
        prev = text[prev_start:prev_end].strip()
        if not prev:
            scan = prev_start
            continue
        if prev.startswith("#"):
            scan = prev_start
            continue
        if prev.endswith("\\"):
            scan = prev_start
            continue
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*(?:\s*\([^)]*\))?", prev):
            scan = prev_start
            continue
        if re.search(r"\b(?:ATTRIBUTE|_GL_ATTRIBUTE|__attribute__)\b", prev):
            scan = prev_start
            continue
        if prev in {"extern \"C\""}:
            scan = prev_start
            continue
        if re.search(r"\b(if|while|for|switch|return|else)\b", prev):
            break
        if any(ch in prev for ch in "{};="):
            break
        prefix = prev + " " + prefix
        scan = prev_start

    ret = _clean_signature_piece(prefix)
    params = _clean_signature_piece(text[open_paren + 1 : close_paren])
    if not params:
        params = "void"
    if not ret or ret.split()[-1] in ("return", "if", "while", "for", "switch"):
        return None
    if not _looks_like_ret_type(ret) or not _looks_like_params(params):
        return None
    return start, ret, params


def _line_is_signature_noise(line: str) -> bool:
    x = str(line or "").strip()
    if not x:
        return True
    if x.startswith("#"):
        return True
    if x.startswith(("/*", "*", "*/", "//")):
        return True
    if re.fullmatch(r"[A-Z_][A-Z0-9_]*(?:\s*\([^)]*\))?", x):
        return True
    if re.search(r"\b(?:ATTRIBUTE|_GL_ATTRIBUTE|__attribute__)\b", x):
        return True
    return False


def _line_based_signature_candidate(
    line_nums: List[int],
    lines: List[str],
    idx: int,
    fn: str,
) -> Optional[Tuple[int, str, str]]:
    line = lines[idx]
    m = re.search(rf"(?<![A-Za-z0-9_]){re.escape(fn)}(?![A-Za-z0-9_])\s*\(", line)
    if not m:
        return None

    prefix_parts: List[str] = []
    same_line_prefix = line[: m.start()].strip()
    if same_line_prefix:
        prefix_parts.append(same_line_prefix)

    for k in range(idx - 1, max(-1, idx - 16), -1):
        prev = lines[k].strip()
        if not prev:
            if prefix_parts:
                break
            continue
        if _line_is_signature_noise(prev):
            continue
        if re.search(r"\b(if|while|for|switch|return|else|case|do)\b", prev):
            break
        if any(ch in prev for ch in "{};="):
            break
        prefix_parts.insert(0, prev)

    ret = _clean_signature_piece(" ".join(prefix_parts))
    if not ret:
        return None

    sig_chunks = [line[m.start() :]]
    depth = sig_chunks[0].count("(") - sig_chunks[0].count(")")
    end_idx = idx
    while depth > 0 and end_idx + 1 < len(lines) and end_idx < idx + 20:
        end_idx += 1
        sig_chunks.append(lines[end_idx])
        depth += lines[end_idx].count("(") - lines[end_idx].count(")")
    sig = "\n".join(sig_chunks)
    open_paren = sig.find("(")
    close_paren = _balanced_param_end(sig, open_paren) if open_paren >= 0 else -1
    if open_paren < 0 or close_paren < 0:
        return None
    params = _clean_signature_piece(sig[open_paren + 1 : close_paren])
    if not params:
        params = "void"

    after = sig[close_paren + 1 :]
    has_body = "{" in after
    if not has_body:
        for k in range(end_idx + 1, min(len(lines), end_idx + 10)):
            nxt = lines[k].strip()
            if not nxt or _line_is_signature_noise(nxt):
                continue
            if nxt.startswith("{"):
                has_body = True
            break
    if not has_body:
        return None

    if not _looks_like_ret_type(ret) or not _looks_like_params(params):
        return None
    return int(line_nums[idx]), ret, params


def _extract_source_signature_near_body(record: Dict[str, Any], lt: Dict[int, str]) -> Optional[Tuple[str, str, str]]:
    fn = str(record.get("func_name") or "func")
    block_src_lines = record.get("block_src_lines") or []
    body_lines = sorted(
        {
            int(ln)
            for group in block_src_lines
            if isinstance(group, list)
            for ln in group
            if isinstance(ln, (int, float)) or str(ln).isdigit()
        }
    )
    if not body_lines:
        return None
    lo = max(1, min(body_lines) - 80)
    hi = min(max(lt) if lt else 0, max(body_lines) + 10)
    line_nums = [ln for ln in sorted(lt) if lo <= ln <= hi]
    lines = [lt[ln] for ln in line_nums]
    candidates: List[Tuple[int, str, str]] = []
    for idx, _line in enumerate(lines):
        cand = _line_based_signature_candidate(line_nums, lines, idx, fn)
        if cand is not None:
            candidates.append(cand)
    if not candidates:
        return None
    # Prefer the nearest definition above the first aligned body line.
    first_body = min(body_lines)
    before = [c for c in candidates if c[0] <= first_body]
    chosen = max(before or candidates, key=lambda x: x[0])
    _, ret, params = chosen
    return _pretty_c_fragment(ret, "FILL_TYPE"), _pretty_c_fragment(params, "FILL_PARAM"), fn


def _extract_source_signature(record: Dict[str, Any], lt: Dict[int, str]) -> Tuple[str, str, str]:
    """Best-effort source definition extraction: return (ret_type, params, func_name)."""
    fn = str(record.get("func_name") or "func")
    if not lt or not fn:
        return UNKNOWN_RET, UNKNOWN_PARAMS, fn or "func"

    near_body = _extract_source_signature_near_body(record, lt)
    if near_body is not None and _signature_is_confident(near_body):
        return near_body

    # Prefer full-text function definitions. This handles multi-line params and
    # avoids mistaking calls such as ``return foo(...)`` for a signature.
    text = "\n".join(lt[k] for k in sorted(lt))
    text_no_comments = _strip_c_comments(text)
    name_re = re.escape(fn)
    ident = r"[A-Za-z_][A-Za-z0-9_]*"
    sig_re = re.compile(
        rf"(?P<ret>(?:(?:static|extern|inline|const|volatile|signed|unsigned|struct\s+{ident}|enum\s+{ident}|union\s+{ident}|{ident})\s+|[*\s])+?)"
        rf"\b{name_re}\s*\((?P<params>[^;{{}}]*?)\)\s*(?:\n\s*)?\{{",
        re.S,
    )
    candidates: List[Tuple[int, str, str]] = []
    for m in sig_re.finditer(text_no_comments):
        ret = _clean_signature_piece(m.group("ret"))
        params = _clean_signature_piece(m.group("params"))
        prefix_start = max(0, m.start() - 80)
        prefix = text_no_comments[prefix_start:m.start()]
        # Reject matches that are clearly expressions/calls rather than
        # top-level function definitions.
        if any(ch in prefix.splitlines()[-1] for ch in "=;{}"):
            continue
        if not ret or ret.split()[-1] in ("return", "if", "while", "for", "switch"):
            continue
        if not _looks_like_ret_type(ret) or not _looks_like_params(params):
            continue
        if not params:
            params = "void"
        candidates.append((m.start(), ret, params))
    # Fallback for GNU/coreutils-style definitions whose return type and
    # function name are split across lines or have macro attributes between the
    # parameter list and opening brace.
    for m in re.finditer(rf"(?<![A-Za-z0-9_]){name_re}(?![A-Za-z0-9_])\s*\(", text_no_comments):
        cand = _signature_candidate_from_match(text_no_comments, fn, m.start(), m.end())
        if cand is not None:
            candidates.append(cand)
    if candidates:
        _, ret, params = candidates[0]
        return _pretty_c_fragment(ret, "FILL_TYPE"), _pretty_c_fragment(params, "FILL_PARAM"), fn

    lines = [_strip_comments(lt[k]) for k in sorted(lt)]
    joined_candidates: List[str] = []
    for i, line in enumerate(lines):
        if fn not in line or "(" not in line:
            continue
        prefix_lines: List[str] = []
        for k in range(max(0, i - 12), i):
            prev = lines[k].strip()
            if not prev or prev.startswith("#"):
                continue
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*(?:\s*\([^)]*\))?", prev):
                continue
            if re.search(r"\b(?:ATTRIBUTE|_GL_ATTRIBUTE|__attribute__)\b", prev):
                continue
            if any(ch in prev for ch in "{};="):
                prefix_lines = []
                continue
            if re.search(r"\b(if|while|for|switch|return|else)\b", prev):
                prefix_lines = []
                continue
            prefix_lines.append(prev)
        acc = " ".join(prefix_lines + [line.strip()])
        j = i + 1
        while ")" not in acc and j < len(lines) and j <= i + 16:
            acc += " " + lines[j]
            j += 1
        joined_candidates.append(re.sub(r"\s+", " ", acc).strip())
    for cand in joined_candidates:
        cand = cand.split("{", 1)[0].rstrip(";").strip()
        m = re.search(rf"(.+?)\b{name_re}\s*\((.*)\)", cand)
        if not m:
            continue
        prefix = m.group(1).strip()
        if any(ch in prefix for ch in "=;{}"):
            continue
        ret = re.sub(r"\s+", " ", m.group(1)).strip()
        params = re.sub(r"\s+", " ", m.group(2)).strip()
        if not ret or ret in ("if", "while", "for", "switch", "return"):
            continue
        if ret.split()[-1] in ("return", "if", "while", "for", "switch"):
            continue
        if not _looks_like_ret_type(ret) or not _looks_like_params(params):
            continue
        if not params:
            params = "void"
        return _pretty_c_fragment(ret, "FILL_TYPE"), _pretty_c_fragment(params, "FILL_PARAM"), fn
    return UNKNOWN_RET, UNKNOWN_PARAMS, fn


def _gt_for_slot(
    slot: SlotInfo,
    block_src_lines: List[List[int]],
    lt: Dict[int, str],
    signature: Optional[Tuple[str, str, str]] = None,
    source_bounds: Optional[Tuple[int, int]] = None,
) -> str:
    if slot.slot_type == "FILL_TYPE":
        if signature and signature[0] != UNKNOWN_RET:
            return _pretty_c_fragment(signature[0], slot.slot_type)
        return ""
    if slot.slot_type == "FILL_PARAM":
        if signature and signature[1] != UNKNOWN_PARAMS:
            return _pretty_c_fragment(signature[1], slot.slot_type)
        return ""
    uniq = _clamp_lines_to_bounds(_line_numbers_for_slot(slot, block_src_lines), source_bounds)
    if not uniq:
        return ""
    if slot.slot_type == "FILL_COND":
        return _pretty_c_fragment(_source_slice_for_condition(uniq, lt, source_bounds), slot.slot_type)
    if slot.slot_type in {"FILL_STMT", "FILL_EXPR"}:
        if _slot_is_header_prefix(slot):
            prefix_raw = _source_slice_for_header_prefix(uniq, lt, source_bounds)
            if prefix_raw:
                return _pretty_c_fragment(prefix_raw, slot.slot_type)
            raw_cond = _source_slice_for_condition(uniq, lt, source_bounds)
            prefix, _cond = _split_condition_prefix_and_cond(raw_cond)
            return _pretty_c_fragment(prefix, slot.slot_type)
        return _pretty_c_fragment(_source_slice_for_statement(uniq, lt, source_bounds), slot.slot_type)
    return _pretty_c_fragment(_join_source_lines(uniq, lt), slot.slot_type)


def _interleave_skeleton_token_ids(
    skeleton_text: str,
    slots: List[SlotInfo],
    gt_texts: List[str],
    fill_ids: List[int],
    tokenizer: Any,
    slot_supervise: Optional[List[bool]] = None,
) -> Tuple[List[int], List[bool], List[int]]:
    """返回 ``(body_ids, supervised, fill_token_positions_in_body)``。

    ``fill_token_positions_in_body`` 为每个 ``<|STRUCLIFT_FILL_*|>`` 在 *body* 段中的下标，
    与 ``head_ids`` 拼接后得到 ``sft_fill_positions``（提案 §7.3.2 slot_mask 用）。
    """
    ids: List[int] = []
    supervised: List[bool] = []
    fill_pos_in_body: List[int] = []
    pos = 0
    si = 0
    slot_by_id = _slot_index_by_id(slots)
    for m in _FILL_RE.finditer(skeleton_text):
        chunk_before = skeleton_text[pos : m.start()]
        if chunk_before:
            tok = tokenizer.encode(chunk_before, add_special_tokens=False)
            ids.extend(tok)
            supervised.extend([False] * len(tok))
        slot_idx = slot_by_id.get(m.group(2), si)
        if slot_idx >= len(slots):
            raise ValueError("FILL 占位符数量与 slots 不一致")
        fid = fill_ids[slot_idx] if slot_idx < len(fill_ids) else fill_ids[-1]
        fill_pos_in_body.append(len(ids))
        ids.append(int(fid))
        # 占位符本身不参与监督；仅监督其后对应的槽位真值 token。
        supervised.append(False)
        keep_slot = True if slot_supervise is None or slot_idx >= len(slot_supervise) else bool(slot_supervise[slot_idx])
        gt = gt_texts[slot_idx] if keep_slot and slot_idx < len(gt_texts) else ""
        if gt:
            gt_tok = tokenizer.encode(gt, add_special_tokens=False)
            ids.extend(gt_tok)
            supervised.extend([bool(keep_slot)] * len(gt_tok))
        si += 1
        pos = m.end()
    tail = skeleton_text[pos:]
    if tail:
        tok = tokenizer.encode(tail, add_special_tokens=False)
        ids.extend(tok)
        supervised.extend([False] * len(tok))
    if si != len(slots):
        raise ValueError(f"slots={len(slots)} 与占位符匹配数={si} 不一致")
    if len(ids) != len(supervised):
        raise ValueError("内部错误：ids 与 supervised 长度不一致")
    return ids, supervised, fill_pos_in_body


def _cst_preorder_types(node: CSTNode) -> List[str]:
    out = [node.type]
    for ch in node.children:
        out.extend(_cst_preorder_types(ch))
    return out


def _cst_max_depth(node: CSTNode) -> int:
    def walk(n: CSTNode, d: int) -> int:
        best = d
        for ch in n.children:
            best = max(best, walk(ch, d + 1))
        return best

    return walk(node, 0)


def _emit_rl_fields(
    record: Dict[str, Any],
    pattern_labels: List[int],
    num_patterns: int,
) -> Dict[str, Any]:
    n_bb = len(pattern_labels)
    src_e = (record.get("bin_edge_index") or [[], []])[0]
    dst_e = (record.get("bin_edge_index") or [[], []])[1]
    edge_index = torch.tensor([src_e, dst_e], dtype=torch.int64)
    pl = torch.tensor(pattern_labels, dtype=torch.int64)
    cst = build_control_structure_tree(pl, edge_index, n_bb)
    hist = [0.0] * num_patterns
    for x in pattern_labels:
        xi = int(x)
        if 0 <= xi < num_patterns:
            hist[xi] += 1.0
    s = sum(hist)
    if s <= 0:
        dist = [1.0 / num_patterns] * num_patterns
    else:
        dist = [x / s for x in hist]
    return {
        "bin_pattern_dist": dist,
        "bin_cst_serialised": _cst_preorder_types(cst),
        "bin_nesting_depth": int(_cst_max_depth(cst)),
    }


def _effective_sft_max_len(tokenizer: Any, max_seq_len: Optional[int]) -> int:
    """与基座模型 context 对齐；未指定时用 tokenizer.model_max_length（过大则退回 16384）。"""
    if max_seq_len is not None and max_seq_len > 0:
        return int(max_seq_len)
    mx = getattr(tokenizer, "model_max_length", None)
    if mx is None or mx >= 1_000_000:
        return 16384
    return int(mx)


def process_record(
    record: Dict[str, Any],
    tokenizer: Any,
    fill_ids: List[int],
    source_root: Optional[Path],
    *,
    emit_rl_fields: bool,
    num_patterns: int,
    mask_prompt_labels: bool,
    mask_non_fill_labels: bool,
    mask_supervised_inputs: bool = False,
    supervised_input_mask_prob: float = 1.0,
    supervised_input_mask_id: Optional[int] = None,
    mask_token: str = DEFAULT_MASK_TOKEN,
    max_seq_len: Optional[int] = None,
    prefix_rewrites: Optional[List[Tuple[str, str]]] = None,
    external_skeleton: Optional[Skeleton] = None,
    skeleton_source: str = "jsonl_pattern_labels",
    strict_teacher: bool = False,
    strict_teacher_drop_record: bool = False,
    min_supervised_semantic_slots: int = 1,
    asm_max_total_lines: int = 160,
    asm_max_lines_per_bb: int = 16,
    require_readable_asm: bool = False,
    prompt_compact_mode: str = "auto",
    max_slot_context_chars: int = 1024,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """返回 (新 record 或 None, 错误信息)。"""
    work = dict(record)
    if prefix_rewrites:
        sf0 = work.get("source_file")
        if isinstance(sf0, str) and sf0.strip():
            work["source_file"] = _rewrite_source_path_string(sf0, prefix_rewrites)
    if require_readable_asm and not _has_readable_binary_text(work):
        return None, "missing readable binary asm text (expected bin_instr_text/bin_asm/asm/instructions_text/disasm)"

    pl = work.get("pattern_labels")
    bei = work.get("bin_edge_index")
    if external_skeleton is None:
        if not isinstance(pl, list) or not pl:
            return None, "缺少 pattern_labels"
        if not isinstance(bei, list) or len(bei) != 2:
            return None, "缺少 bin_edge_index"
        n_bb = len(pl)
        if n_bb != len(work.get("bin_block_lengths") or []):
            return None, "pattern_labels 与 bin_block_lengths 长度不一致"

        hints = _bb_opcode_hints(work)
        pl_t = torch.tensor([int(x) for x in pl], dtype=torch.int64)
        edge_index = torch.tensor(
            [[int(x) for x in bei[0]], [int(x) for x in bei[1]]],
            dtype=torch.int64,
        )
        try:
            skel: Skeleton = build_skeleton_for_binary_cfg(
                pl_t, edge_index, n_bb, hints,
            )
        except Exception as e:
            return None, f"骨架失败: {e}"
    else:
        skel = external_skeleton

    src_path = _resolve_source_path(work, source_root)
    block_src_lines = work.get("block_src_lines") or []
    if not isinstance(block_src_lines, list):
        return None, "block_src_lines 非法"
    lt: Dict[int, str] = _line_to_text(src_path) if src_path else {}
    if lt:
        lt = _strip_preprocessor_lines_from_lt(lt)
        lt = _strip_block_comments_from_lt(lt)
    if not lt and block_src_lines and any(block_src_lines):
        logger.debug(
            "%s: source_file 不可读（%s），GT 将用弱默认值",
            work.get("func_name", "?"),
            work.get("source_file"),
        )

    teacher_signature = _extract_source_signature(work, lt)
    source_bounds = _function_body_line_range(work, lt)
    prompt_signature = (
        teacher_signature
        if _signature_is_confident(teacher_signature)
        else (UNKNOWN_RET, UNKNOWN_PARAMS, teacher_signature[2])
    )
    raw_gt_texts = [
        _sanitize_teacher_text(
            _gt_for_slot(s, block_src_lines, lt, teacher_signature, source_bounds),
            work,
        )
        for s in skel.slots
    ]
    gt_texts: List[str] = []
    slot_supervise: List[bool] = []
    dropped_targets: List[Tuple[int, str, str]] = []
    strict_drop_reasons: List[Tuple[int, str, str, str]] = []
    record_reject_reasons: List[Tuple[int, str, str, str]] = []
    strict_repair_reasons: List[Tuple[int, str, str, str]] = []
    for i, (s, raw) in enumerate(zip(skel.slots, raw_gt_texts)):
        clean, usable = _clean_teacher_target_for_slot(s.slot_type, raw)
        if usable and strict_teacher:
            reason = _strict_reject_teacher_target(s.slot_type, clean, raw)
            if reason:
                usable = False
                strict_drop_reasons.append((i, s.slot_type, reason, raw))
                clean = ""
        gt_texts.append(clean)
        slot_supervise.append(bool(usable))
        if not usable:
            dropped_targets.append((i, s.slot_type, raw))
    if strict_teacher:
        repaired_initial = _repair_condition_targets_from_source(
            skel.slots,
            gt_texts,
            raw_gt_texts,
            slot_supervise,
            block_src_lines,
            lt,
            source_bounds,
            work,
        )
        if repaired_initial:
            repaired_set = set(repaired_initial)
            dropped_targets = [x for x in dropped_targets if x[0] not in repaired_set]
            strict_drop_reasons = [x for x in strict_drop_reasons if x[0] not in repaired_set]
            for idx in repaired_initial:
                strict_repair_reasons.append(
                    (idx, skel.slots[idx].slot_type, "repair_initial_condition_target", gt_texts[idx])
                )
        if strict_teacher_drop_record:
            for i, typ, raw in dropped_targets:
                # Empty condition slots render as "if ()" / "while ()" in the
                # skeleton, so they poison the teacher even when unsupervised.
                if typ == "FILL_COND":
                    reason = (i, typ, "cond_unusable_or_empty", raw)
                    strict_drop_reasons.append(reason)
                    record_reject_reasons.append(reason)
                # A statement slot that is just a brace or semicolon is usually
                # harmless skeleton noise; keep those records. Other unusable
                # semantic slots indicate broken source-to-slot alignment.
                elif typ in {"FILL_STMT", "FILL_EXPR"}:
                    norm = _pretty_c_fragment(raw, typ)
                    branch_boundary = norm.startswith("else ") or norm == "else"
                    if (
                        not branch_boundary
                        and norm not in {"", "{", "}", ";", "};", "(", ")"}
                        and not re.fullmatch(r"[{};()\s]+", norm)
                    ):
                        reason = (i, typ, "semantic_slot_unusable", raw)
                        strict_drop_reasons.append(reason)
                        record_reject_reasons.append(reason)
        # If several slots in one function get exactly the same target, the
        # alignment is too coarse. Keep the first occurrence and stop teaching
        # the model that identical code belongs in several unrelated slots.
        seen: Dict[str, int] = {}
        for i, (s, txt) in enumerate(zip(skel.slots, gt_texts)):
            if not slot_supervise[i]:
                continue
            if s.slot_type in {"FILL_TYPE", "FILL_PARAM"}:
                continue
            if s.slot_type == "FILL_COND":
                continue
            key = re.sub(r"\s+", " ", txt).strip()
            if not key:
                continue
            if key in seen:
                slot_supervise[i] = False
                gt_texts[i] = ""
                strict_drop_reasons.append((i, s.slot_type, f"duplicate_of_slot_{seen[key]}", txt))
                dropped_targets.append((i, s.slot_type, txt))
            else:
                seen[key] = i
    system = (
        "<|system|>\n"
        "You are a decompiler. Fill in the placeholders in the code skeleton\n"
        "based on the binary context provided for each slot.\n"
        "<|/system|>\n"
    )
    bin_info = _binary_info_block(work, prompt_signature)
    skeleton_text_raw = _normalise_cond_placeholder_punctuation(skel.skeleton_text)
    cond_prefix_insertions: Dict[int, str] = {}
    cond_prefix_slot_for_cond: Dict[int, int] = {}
    if strict_teacher:
        skeleton_text_raw, cond_prefix_insertions, cond_prefix_slot_for_cond = _inject_cond_prefix_statements(
            skeleton_text_raw,
            skel.slots,
            raw_gt_texts,
        )
        for idx, text in cond_prefix_insertions.items():
            if 0 <= idx < len(gt_texts):
                clean, usable = _clean_teacher_target_for_slot(skel.slots[idx].slot_type, text)
                gt_texts[idx] = clean
                slot_supervise[idx] = bool(usable)
        prefix_target_slots = set(cond_prefix_insertions)
        removed_prefix_dupes = _strip_inserted_cond_prefix_duplicates(
            gt_texts,
            raw_gt_texts,
            slot_supervise,
            skel.slots,
            {},
        )
        if cond_prefix_insertions:
            skeleton_text_raw = _dedupe_rendered_skeleton_declarations(skeleton_text_raw)
        for i in removed_prefix_dupes:
            raw = raw_gt_texts[i] if i < len(raw_gt_texts) else ""
            dropped_targets.append((i, skel.slots[i].slot_type, raw))
            strict_drop_reasons.append((i, skel.slots[i].slot_type, "duplicate_inserted_cond_prefix", raw))
        if prefix_target_slots:
            strict_drop_reasons = [
                x for x in strict_drop_reasons
                if not (0 <= int(x[0]) < len(skel.slots) and int(x[0]) in prefix_target_slots and str(x[2]).startswith("duplicate_of_slot_"))
            ]
        if strict_teacher_drop_record:
            rendered = _render_skeleton_with_targets(skeleton_text_raw, gt_texts, skel.slots)
            rendered_reason = _strict_reject_rendered_teacher(rendered)
            if rendered_reason in {
                "rendered_empty_control_condition",
                "rendered_if_condition_has_semicolon",
                "rendered_while_condition_has_semicolon",
                "rendered_switch_condition_has_semicolon",
            }:
                repaired = _repair_condition_targets_from_source(
                    skel.slots,
                    gt_texts,
                    raw_gt_texts,
                    slot_supervise,
                    block_src_lines,
                    lt,
                    source_bounds,
                    work,
                )
                if repaired:
                    for idx in repaired:
                        strict_repair_reasons.append(
                            (idx, skel.slots[idx].slot_type, f"repair_after_{rendered_reason}", gt_texts[idx])
                        )
                    rendered = _render_skeleton_with_targets(skeleton_text_raw, gt_texts, skel.slots)
                    rendered_reason = _strict_reject_rendered_teacher(rendered)
            if rendered_reason:
                reason = (-1, "RENDERED", rendered_reason, "")
                strict_drop_reasons.append(reason)
                record_reject_reasons.append(reason)
        if strict_teacher_drop_record and record_reject_reasons:
            shown = "; ".join(
                (
                    f"{reason}"
                    if i < 0
                    else f"slot_{i}:{typ}:{reason} raw={raw!r}"
                )
                for i, typ, reason, raw in record_reject_reasons[:3]
            )
            return None, f"strict teacher rejected record: {shown}"
    n_semantic_slots = sum(
        1 for s in skel.slots if s.slot_type not in {"FILL_TYPE", "FILL_PARAM"}
    )
    n_supervised_semantic = sum(
        1
        for s, keep in zip(skel.slots, slot_supervise)
        if keep and s.slot_type not in {"FILL_TYPE", "FILL_PARAM"}
    )
    min_semantic_required = max(0, int(min_supervised_semantic_slots))
    if n_semantic_slots > 0 and min_semantic_required > 0 and n_supervised_semantic < min_semantic_required:
        shown = "; ".join(
            f"slot_{i}:{typ}={txt!r}" for i, typ, txt in dropped_targets[:3]
        )
        return None, f"not enough usable teacher targets ({n_supervised_semantic}): {shown}"
    skeleton_text_for_prompt = _format_skeleton_text_for_prompt(skeleton_text_raw)
    skeleton_text_for_prompt = _sanitize_prompt_text(skeleton_text_for_prompt, work)
    skeleton_text_compacted_for_prompt = False
    slot_order = _slot_indices_in_skeleton_order(skeleton_text_for_prompt, skel.slots)
    slot_types = [skel.slots[i].slot_type for i in slot_order]

    eff_max = _effective_sft_max_len(tokenizer, max_seq_len)
    # 编码过程中临时放宽 model_max_length，避免按「分片」长度误报 16384 警告；最终仍按 eff_max 截断判定
    _old_mx = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = 10**9
        head_text = system.strip() + "\n\n" + bin_info.strip() + "\n\n<|skeleton|>\n"
        head_ids = tokenizer.encode(head_text, add_special_tokens=False)
        requested_prompt_mode = str(prompt_compact_mode or "auto").lower()
        if requested_prompt_mode == "auto":
            context_modes = ["full", "compact", "ultra"]
        elif requested_prompt_mode in {"full", "compact", "ultra"}:
            context_modes = [requested_prompt_mode]
        else:
            context_modes = ["auto", "full", "compact", "ultra"]
        context_modes = ["full", "compact", "ultra"] if "auto" in context_modes else context_modes

        prompt_retry_trace: List[Dict[str, Any]] = []
        ctx_lines: List[str] = []
        ctx_block = ""
        tail_text = ""
        tail_ids: List[int] = []
        body_ids: List[int] = []
        body_sup: List[bool] = []
        fill_pos_body: List[int] = []
        has_readable_asm = False
        used_asm_total_lines = 0
        used_asm_lines_per_bb = 0
        used_prompt_compact_mode = ""
        best_over_len: Optional[int] = None
        prompt_skeleton_variants: List[Tuple[str, str, bool]] = [
            ("full_skeleton", skeleton_text_for_prompt, False),
        ]
        if "ultra" in context_modes:
            compact_raw, did_compact = _compact_skeleton_for_ecsl_ultra(skeleton_text_raw)
            if did_compact:
                compact_prompt = _format_skeleton_text_for_prompt(compact_raw)
                compact_prompt = _sanitize_prompt_text(compact_prompt, work)
                prompt_skeleton_variants.append(("ecsl_slot_skeleton", compact_prompt, True))

        selected_skeleton_text_for_prompt = skeleton_text_for_prompt
        selected_body_ids: List[int] = []
        selected_body_sup: List[bool] = []
        selected_fill_pos_body: List[int] = []

        for skel_variant_name, skel_prompt_text, skel_compacted in prompt_skeleton_variants:
            try:
                cand_body_ids, cand_body_sup, cand_fill_pos_body = _interleave_skeleton_token_ids(
                    skel_prompt_text,
                    skel.slots,
                    gt_texts,
                    fill_ids,
                    tokenizer,
                    slot_supervise=slot_supervise,
                )
            except Exception as e:
                prompt_retry_trace.append(
                    {
                        "skeleton_variant": skel_variant_name,
                        "context_mode": "n/a",
                        "seq_len": -1,
                        "fits": False,
                        "error": f"body_interleave_failed:{e}",
                    }
                )
                continue

            for mode in context_modes:
                # Compact skeleton is an ultra fallback; trying it with full
                # context would defeat the purpose and inflate logs.
                if skel_compacted and mode != "ultra":
                    continue

                slot_ctx_lines = [
                    _slot_binary_context_line(
                        s,
                        compact_mode=mode,
                        max_context_chars=int(max_slot_context_chars),
                    )
                    for s in skel.slots
                ]
                base_ctx_lines: List[str] = list(slot_ctx_lines)
                base_ctx = "<|binary_context|>\n" + "\n".join(base_ctx_lines) + "\n<|/binary_context|>"
                base_ctx = _sanitize_prompt_text(base_ctx, work)
                base_tail = "\n<|/skeleton|>\n\n" + base_ctx + "\n\n<|output|>\n"
                base_tail_ids = tokenizer.encode(base_tail, add_special_tokens=False)
                base_total = len(head_ids) + len(cand_body_ids) + len(base_tail_ids)
                prompt_retry_trace.append(
                    {
                        "skeleton_variant": skel_variant_name,
                        "context_mode": mode,
                        "asm_max_total_lines": 0,
                        "asm_max_lines_per_bb": 0,
                        "seq_len": int(base_total),
                        "fits": bool(base_total <= eff_max),
                    }
                )
                if best_over_len is None or base_total < best_over_len:
                    best_over_len = int(base_total)
                if base_total > eff_max:
                    continue

                ctx_lines = base_ctx_lines
                ctx_block = base_ctx
                tail_text = base_tail
                tail_ids = base_tail_ids
                selected_skeleton_text_for_prompt = skel_prompt_text
                selected_body_ids = cand_body_ids
                selected_body_sup = cand_body_sup
                selected_fill_pos_body = cand_fill_pos_body
                used_prompt_compact_mode = mode
                skeleton_text_compacted_for_prompt = bool(skel_compacted)
                has_readable_asm = False
                used_asm_total_lines = 0
                used_asm_lines_per_bb = 0

                if int(asm_max_total_lines) > 0 and int(asm_max_lines_per_bb) > 0:
                    budgets: List[Tuple[int, int]] = []
                    t = max(1, int(asm_max_total_lines))
                    b = max(1, int(asm_max_lines_per_bb))
                    while True:
                        budgets.append((t, b))
                        if t <= 4 and b <= 1:
                            break
                        t = max(4, t // 2)
                        b = max(1, b // 2)
                    for t, b in budgets:
                        cand_asm, cand_has_asm = _readable_asm_block(
                            work,
                            max_total_lines=t,
                            max_lines_per_bb=b,
                        )
                        if not cand_asm:
                            break
                        cand_ctx_lines = [cand_asm] + slot_ctx_lines
                        cand_ctx = "<|binary_context|>\n" + "\n".join(cand_ctx_lines) + "\n<|/binary_context|>"
                        cand_ctx = _sanitize_prompt_text(cand_ctx, work)
                        cand_tail = "\n<|/skeleton|>\n\n" + cand_ctx + "\n\n<|output|>\n"
                        cand_tail_ids = tokenizer.encode(cand_tail, add_special_tokens=False)
                        cand_total = len(head_ids) + len(cand_body_ids) + len(cand_tail_ids)
                        prompt_retry_trace.append(
                            {
                                "skeleton_variant": skel_variant_name,
                                "context_mode": mode,
                                "asm_max_total_lines": int(t),
                                "asm_max_lines_per_bb": int(b),
                                "seq_len": int(cand_total),
                                "fits": bool(cand_total <= eff_max),
                            }
                        )
                        if best_over_len is None or cand_total < best_over_len:
                            best_over_len = int(cand_total)
                        if cand_total <= eff_max:
                            has_readable_asm = bool(cand_has_asm)
                            used_asm_total_lines = int(t)
                            used_asm_lines_per_bb = int(b)
                            ctx_lines = cand_ctx_lines
                            ctx_block = cand_ctx
                            tail_text = cand_tail
                            tail_ids = cand_tail_ids
                            selected_skeleton_text_for_prompt = skel_prompt_text
                            selected_body_ids = cand_body_ids
                            selected_body_sup = cand_body_sup
                            selected_fill_pos_body = cand_fill_pos_body
                            skeleton_text_compacted_for_prompt = bool(skel_compacted)
                            break
                break
            if tail_ids:
                break
        if not tail_ids:
            over_len = int(best_over_len or len(head_ids))
            return None, f"SFT 序列过长 ({over_len} > {eff_max})，跳过（与基座 max length 不一致将无法训练）"
        skeleton_text_for_prompt = selected_skeleton_text_for_prompt
        body_ids = selected_body_ids
        body_sup = selected_body_sup
        fill_pos_body = selected_fill_pos_body
        prompt_text = _format_prompt_text(system, bin_info, skeleton_text_for_prompt, ctx_block)
    except Exception as e:
        return None, f"分词失败: {e}"
    finally:
        if _old_mx is not None:
            tokenizer.model_max_length = _old_mx

    input_ids = head_ids + body_ids + tail_ids
    if len(input_ids) > eff_max:
        return None, f"SFT 序列过长 ({len(input_ids)} > {eff_max})，跳过（与基座 max length 不一致将无法训练）"
    labels = list(input_ids)
    # 与 body_ids 对齐的监督布尔掩码：仅 FILL 槽位真值 token 为 True。
    supervised_mask: List[bool] = (
        [False] * len(head_ids) + body_sup + [False] * len(tail_ids)
    )
    if len(supervised_mask) != len(labels):
        return None, "内部错误：supervised_mask 与 labels 长度不一致"
    if mask_non_fill_labels:
        for i, keep in enumerate(supervised_mask):
            if not keep:
                labels[i] = -100
    if mask_prompt_labels:
        for i in range(len(head_ids)):
            labels[i] = -100
    if mask_supervised_inputs:
        mask_prob = max(0.0, min(1.0, float(supervised_input_mask_prob)))
        if supervised_input_mask_id is None:
            supervised_input_mask_id = tokenizer.convert_tokens_to_ids(mask_token)
            if supervised_input_mask_id is None or int(supervised_input_mask_id) < 0:
                supervised_input_mask_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                )
        mask_id = int(supervised_input_mask_id or 0)
        for i, y in enumerate(labels):
            if y != -100 and (mask_prob >= 1.0 or random.random() < mask_prob):
                input_ids[i] = mask_id

    out = dict(work)
    out["sft_input_ids"] = input_ids
    out["sft_labels"] = labels
    out["slot_types"] = slot_types
    out["sft_masked_supervised_inputs"] = bool(mask_supervised_inputs)
    if mask_supervised_inputs:
        out["sft_supervised_input_mask_id"] = int(supervised_input_mask_id or 0)
        out["sft_supervised_input_mask_token"] = str(mask_token)
        out["sft_supervised_input_mask_prob"] = float(supervised_input_mask_prob)
    out["sft_signature_ret_type"] = teacher_signature[0]
    out["sft_signature_params"] = teacher_signature[1]
    out["sft_signature_func_name"] = teacher_signature[2]
    out["sft_signature_teacher_ret_type"] = teacher_signature[0]
    out["sft_signature_teacher_params"] = teacher_signature[1]
    out["sft_signature_prompt_ret_type"] = prompt_signature[0]
    out["sft_signature_prompt_params"] = prompt_signature[1]
    out["sft_teacher_pretty_version"] = 2
    out["sft_teacher_targets_raw"] = raw_gt_texts
    out["sft_teacher_targets"] = gt_texts
    out["sft_prompt_slot_ids"] = [
        str(skel.slots[i].slot_id) for i in slot_order
    ]
    out["sft_prompt_slot_original_indices"] = [int(i) for i in slot_order]
    out["sft_teacher_targets_raw_prompt_order"] = [
        raw_gt_texts[i] if i < len(raw_gt_texts) else "" for i in slot_order
    ]
    out["sft_teacher_targets_prompt_order"] = [
        gt_texts[i] if i < len(gt_texts) else "" for i in slot_order
    ]
    out["sft_teacher_slot_supervise"] = slot_supervise
    out["sft_teacher_slot_supervise_prompt_order"] = [
        bool(slot_supervise[i]) if i < len(slot_supervise) else False for i in slot_order
    ]
    out["sft_teacher_dropped_targets"] = [
        {"slot": int(i), "slot_type": typ, "raw": raw}
        for i, typ, raw in dropped_targets
    ]
    out["sft_teacher_strict"] = bool(strict_teacher)
    out["sft_teacher_strict_drop_reasons"] = [
        {"slot": int(i), "slot_type": typ, "reason": reason, "raw": raw}
        for i, typ, reason, raw in strict_drop_reasons
    ]
    out["sft_teacher_strict_repair_reasons"] = [
        {"slot": int(i), "slot_type": typ, "reason": reason, "target": target}
        for i, typ, reason, target in strict_repair_reasons
    ]
    out["sft_teacher_strict_drop_reason_counts"] = dict(
        collections.Counter(reason for _, _, reason, _ in strict_drop_reasons)
    )
    out["sft_teacher_cond_prefix_insertions"] = {
        str(i): text for i, text in cond_prefix_insertions.items()
    }
    out["sft_teacher_cond_prefix_slot_for_cond"] = {
        str(cond_i): int(prefix_i)
        for cond_i, prefix_i in cond_prefix_slot_for_cond.items()
    }
    out["sft_skeleton_text_pretty"] = skeleton_text_for_prompt
    out["sft_output_text_pretty"] = _render_skeleton_with_targets(
        skeleton_text_for_prompt,
        gt_texts,
        skel.slots,
    )
    out["sft_prompt_text_pretty"] = prompt_text
    out["sft_skeleton_source"] = str(skeleton_source)
    out["sft_prompt_layout"] = "proposal_pretty_skeleton_then_binary_context_v2"
    out["sft_prompt_skeleton_compacted"] = bool(skeleton_text_compacted_for_prompt)
    out["sft_prompt_compact_mode"] = str(used_prompt_compact_mode or "unknown")
    out["sft_prompt_requested_compact_mode"] = str(prompt_compact_mode or "auto")
    out["sft_prompt_retry_trace"] = prompt_retry_trace
    out["sft_prompt_token_len"] = len(input_ids)
    out["sft_prompt_head_token_len"] = len(head_ids)
    out["sft_prompt_body_token_len"] = len(body_ids)
    out["sft_prompt_tail_token_len"] = len(tail_ids)
    out["sft_prompt_max_seq_len"] = int(eff_max)
    out["sft_binary_context_max_slot_context_chars"] = int(max_slot_context_chars)
    out["sft_binary_context_lines"] = len(ctx_lines)
    out["sft_binary_context_readable_asm_in_prompt"] = bool(has_readable_asm)
    out["sft_binary_context_asm_max_total_lines"] = int(used_asm_total_lines)
    out["sft_binary_context_asm_max_lines_per_bb"] = int(used_asm_lines_per_bb)
    # §7.3.2：Stage3 collate 构造 slot_mask（FILL token → 仅 attend 对齐 BB）
    hlen = len(head_ids)
    out["sft_fill_positions"] = [hlen + int(p) for p in fill_pos_body]
    out["sft_slot_bb_indices"] = [
        [int(x) for x in skel.slots[i].aligned_bb_indices] for i in slot_order
    ]
    if emit_rl_fields:
        out.update(_emit_rl_fields(work, [int(x) for x in pl], num_patterns))
    return out, ""


def main() -> None:
    ap = argparse.ArgumentParser(description="binskel jsonl 增补 SFT / RL 字段")
    ap.add_argument("--input", type=Path, required=True, help="输入 .jsonl")
    ap.add_argument("--output", type=Path, required=True, help="输出 .jsonl")
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/deepseek-coder-6.7b-base",
        help="HuggingFace tokenizer 名或本地目录",
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="解析 source_file 相对路径时的根目录（与 build_binskel 编译目录一致）",
    )
    ap.add_argument(
        "--rewrite-source-prefix",
        nargs=2,
        metavar=("FROM_PREFIX", "TO_PREFIX"),
        action="append",
        dest="prefix_pairs",
        help="改写每条 source_file 字符串前缀后再读盘（可多组顺序套用；例: "
        "/mnt/e/structlift_src_datasets /data/chaoni/WQM/source_datasets）",
    )
    ap.add_argument(
        "--max-fill-tokens",
        type=int,
        default=512,
        help="每文件预留的占位符 special token 个数（≥ 单条样本最大槽位数）",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="SFT 序列长度上限（默认 0=自动：取 tokenizer.model_max_length，异常大则 16384）；超长样本跳过",
    )
    ap.add_argument(
        "--asm-max-total-lines",
        type=int,
        default=160,
        help="每条 prompt 最多加入多少行 readable asm；若超长会自动减半重试，0=不加入全函数 asm 块",
    )
    ap.add_argument(
        "--asm-max-lines-per-bb",
        type=int,
        default=16,
        help="每个 BB 最多加入多少行 readable asm；若超长会自动减半重试",
    )
    ap.add_argument(
        "--prompt-compact-mode",
        choices=["auto", "full", "compact", "ultra"],
        default="auto",
        help="slot binary_context 压缩策略；auto 会按 full→compact→ultra 重试以适配 max_seq_len",
    )
    ap.add_argument(
        "--max-slot-context-chars",
        type=int,
        default=1024,
        help="每个 slot 的 binary_context 文本最大字符数，避免 JSON contract 堆叠导致 prompt 过长",
    )
    ap.add_argument(
        "--require-readable-asm",
        action="store_true",
        help="要求输入记录已含 bin_instr_text/bin_asm/asm 等可读汇编文本；缺失则跳过该记录",
    )
    ap.add_argument(
        "--emit-rl-fields",
        action="store_true",
        help="写入 bin_pattern_dist / bin_cst_serialised / bin_nesting_depth",
    )
    ap.add_argument(
        "--num-patterns",
        type=int,
        default=10,
        help="与 VocabConfig.num_patterns 一致",
    )
    ap.add_argument(
        "--mask-prompt-labels",
        action="store_true",
        help="将 system/binary_info/skeleton 开头在 labels 中置为 -100",
    )
    ap.add_argument(
        "--mask-non-fill-labels",
        action="store_true",
        help="骨架中 非 FILL 占位符位 的 labels 置 -100（仅在 FILL token 及其真值处保留监督）",
    )
    ap.add_argument(
        "--fill-id-min",
        type=int,
        default=DEFAULT_FILL_ID_MIN,
        help="占位符 token id 应 ≥ 此值（与 Module C 一致）",
    )
    ap.add_argument("--max-lines", type=int, default=None, help="仅处理前 N 行（调试）")
    ap.add_argument(
        "--mask-supervised-inputs",
        action="store_true",
        help="Replace supervised answer tokens in sft_input_ids with pad/eos for masked slot filling training.",
    )
    ap.add_argument(
        "--supervised-input-mask-prob",
        type=float,
        default=1.0,
        help="Probability of masking each supervised input token when --mask-supervised-inputs is set.",
    )
    ap.add_argument(
        "--supervised-input-mask-id",
        type=int,
        default=None,
        help="Token id used by --mask-supervised-inputs; overrides --mask-token.",
    )
    ap.add_argument(
        "--mask-token",
        type=str,
        default=DEFAULT_MASK_TOKEN,
        help="Special token used by --mask-supervised-inputs when --supervised-input-mask-id is not set.",
    )
    ap.add_argument(
        "--strict-teacher",
        action="store_true",
        help="Drop noisy teacher slots: duplicate body targets, non-control conditions, and multi-statement targets.",
    )
    ap.add_argument(
        "--strict-teacher-drop-record",
        action="store_true",
        help="With --strict-teacher, skip the whole record when any dangerous teacher slot is rejected.",
    )
    ap.add_argument(
        "--min-supervised-semantic-slots",
        type=int,
        default=1,
        help="Drop records with fewer supervised non-signature slots after teacher cleaning.",
    )
    ap.add_argument(
        "--save-tokenizer",
        type=Path,
        default=None,
        help="保存已 add_tokens 的 tokenizer，训练 Stage3 时必须加载此目录而非原始模型名",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("需要 transformers: pip install transformers")
        sys.exit(1)

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    if not inp.is_file():
        logger.error("输入不存在: %s", inp)
        sys.exit(1)
    outp.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    extra = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(args.max_fill_tokens)]
    if args.mask_supervised_inputs and args.supervised_input_mask_id is None:
        extra.append(str(args.mask_token))
    tokenizer.add_tokens(extra)
    fill_ids = [tokenizer.convert_tokens_to_ids(t) for t in extra]
    low = min(fill_ids)
    if low < args.fill_id_min:
        logger.warning(
            "占位符最小 id=%d < %d；训练时请设置 module_c 或 SlotWeightBuilder 的 "
            "fill_token_range_start，否则槽位权重无法匹配。",
            low,
            args.fill_id_min,
        )

    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    prefix_pairs: Optional[List[Tuple[str, str]]] = None
    if args.prefix_pairs:
        prefix_pairs = [(str(a), str(b)) for a, b in args.prefix_pairs]
        logger.info("source_file 前缀改写: %s", prefix_pairs)

    n_ok, n_skip, n_err = 0, 0, 0
    err_examples: List[str] = []

    with inp.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
        for li, line in enumerate(fin):
            if args.max_lines is not None and li >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue
            _msl = args.max_seq_len if args.max_seq_len > 0 else None
            new_r, err = process_record(
                record,
                tokenizer,
                fill_ids,
                source_root,
                emit_rl_fields=args.emit_rl_fields,
                num_patterns=args.num_patterns,
                mask_prompt_labels=args.mask_prompt_labels,
                mask_non_fill_labels=args.mask_non_fill_labels,
                mask_supervised_inputs=args.mask_supervised_inputs,
                supervised_input_mask_prob=args.supervised_input_mask_prob,
                supervised_input_mask_id=args.supervised_input_mask_id,
                mask_token=args.mask_token,
                max_seq_len=_msl,
                prefix_rewrites=prefix_pairs,
                strict_teacher=bool(args.strict_teacher),
                strict_teacher_drop_record=bool(args.strict_teacher_drop_record),
                min_supervised_semantic_slots=int(args.min_supervised_semantic_slots),
                asm_max_total_lines=int(args.asm_max_total_lines),
                asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                require_readable_asm=bool(args.require_readable_asm),
                prompt_compact_mode=str(args.prompt_compact_mode),
                max_slot_context_chars=int(args.max_slot_context_chars),
            )
            if new_r is None:
                n_err += 1
                if len(err_examples) < 8:
                    fn = record.get("func_name", "?")
                    err_examples.append(f"line {li+1} {fn}: {err}")
                continue
            fout.write(json.dumps(new_r, ensure_ascii=False) + "\n")
            n_ok += 1

    if args.save_tokenizer is not None:
        save_dir = args.save_tokenizer.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        logger.info("已保存 tokenizer（含占位符词元）→ %s", save_dir)

    logger.info(
        "完成: 写出 %d 条, 跳过(坏 JSON) %d, 失败 %d → %s",
        n_ok,
        n_skip,
        n_err,
        outp,
    )
    if err_examples:
        logger.info("失败示例:\n  %s", "\n  ".join(err_examples))


if __name__ == "__main__":
    main()

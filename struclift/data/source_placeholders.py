"""
源码标识符占位化（与提案 §5.4 一致）：在写入 jsonl / stable_embedding_id 之前完成。

网络侧只消费 ``src_token_ids``，不在模型里做字符串替换。
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

# C/C++ 常见关键字（不参与 var_/func_ 替换）
_C_KEYWORDS = frozenset(
    """
    auto break case char const continue default do double else enum extern
    float for goto if inline int long register restrict return short signed
    sizeof static struct switch typedef union unsigned void volatile while
    _Bool _Complex _Imaginary
    bool complex imaginary
    int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t
    size_t ssize_t intptr_t uintptr_t ptrdiff_t
    NULL true false
    static_assert alignas alignof
    concept requires
    and and_eq bitand bitor compl not or or_eq xor xor_eq
    new delete this namespace using class public private protected virtual
    explicit operator template typename try catch throw
    static_cast dynamic_cast reinterpret_cast const_cast
    nullptr decltype constexpr thread_local noexcept
    """.split(),
)

_ID_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")


def apply_src_identifier_placeholders(
    text: str,
    var_map: Dict[str, str],
    func_map: Dict[str, str],
) -> Tuple[str, int, int]:
    """将标识符替换为 ``var_0, ...`` / ``func_0, ...``。

    同一 **函数** 内多次调用应传入**同一对** ``var_map`` / ``func_map``。

    * **func_**：该处标识符后（可隔空白）紧跟 ``(`` → 视为函数名/调用点。
    * **var_**：其余非关键字标识符。

    Returns
    -------
    replaced_text, n_vars, n_funcs
        替换后文本；当前映射大小（便于日志）。
    """
    if not text:
        return text, len(var_map), len(func_map)

    out: list[str] = []
    pos = 0
    for m in _ID_RE.finditer(text):
        out.append(text[pos : m.start()])
        w = m.group()
        pos = m.end()
        if w in _C_KEYWORDS:
            out.append(w)
            continue
        rest = text[m.end() : m.end() + 48].lstrip()
        is_call = rest.startswith("(")
        if is_call:
            if w not in func_map:
                func_map[w] = f"func_{len(func_map)}"
            out.append(func_map[w])
        else:
            if w not in var_map:
                var_map[w] = f"var_{len(var_map)}"
            out.append(var_map[w])
    out.append(text[pos:])
    return "".join(out), len(var_map), len(func_map)

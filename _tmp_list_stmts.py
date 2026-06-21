"""Standalone: list tree-sitter statement order (no elftools)."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

STATEMENT_TYPES = {
    "if_statement",
    "while_statement",
    "for_statement",
    "do_statement",
    "switch_statement",
    "case_statement",
    "return_statement",
    "expression_statement",
    "declaration",
    "break_statement",
    "continue_statement",
    "goto_statement",
    "compound_statement",
    "labeled_statement",
    "attributed_statement",
}
CONTROL_TYPES = {
    "if_statement",
    "while_statement",
    "for_statement",
    "do_statement",
    "switch_statement",
}

try:
    from tree_sitter_languages import get_parser
except ImportError:
    print("需要: pip install tree_sitter_languages", file=sys.stderr)
    sys.exit(2)

_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        _parser = get_parser("c")
    return _parser


def _parse(snippet: str, line_offset: int) -> Tuple[List[Dict], dict]:
    parser = _get_parser()
    tree = parser.parse(bytes(snippet, "utf-8"))

    def ntype(node) -> str:
        t = getattr(node, "type", None) or getattr(node, "symbol", str(node))
        return t.decode("utf-8") if isinstance(t, bytes) else str(t)

    statements: List[Dict] = []
    stack: List[Tuple[Any, int]] = [(tree.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        nt = ntype(node)
        if nt in STATEMENT_TYPES:
            statements.append({
                "id": len(statements),
                "type": nt,
                "start_line": node.start_point[0] + 1 + line_offset,
                "end_line": node.end_point[0] + 1 + line_offset,
            })
        ch = getattr(node, "children", None) or []
        for c in reversed(ch):
            stack.append((c, depth + 1))
    line_to: dict = defaultdict(list)
    for s in statements:
        for ln in range(s["start_line"], s["end_line"] + 1):
            line_to[ln].append(s)
    return statements, dict(line_to)


def filter_stmts(statements: List[Dict], func_min: int, func_max: int) -> List[Dict]:
    out = []
    for s in statements:
        if s["end_line"] >= func_min and s["start_line"] <= func_max:
            out.append({**s, "id": len(out)})
    if not out and statements and func_min > 0:
        lo, hi = max(1, func_min - 15), func_max + 15
        for s in statements:
            if s["end_line"] >= lo and s["start_line"] <= hi:
                out.append({**s, "id": len(out)})
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    p = root / "zlib_build/test/minigzip.c"
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    stmts, _ = _parse(text, 0)

    ast_map = {
        "if_statement": 2,
        "while_statement": 3,
        "for_statement": 4,
        "do_statement": 5,
        "switch_statement": 6,
        "case_statement": 7,
        "return_statement": 8,
        "break_statement": 9,
        "continue_statement": 10,
        "goto_statement": 11,
        "compound_statement": 12,
    }

    def dump(name: str, lo: int, hi: int) -> None:
        filt = filter_stmts(stmts, lo, hi)
        print(f"\n{'='*60}\n{name}  (DWARF 范围约 L{lo}–L{hi}, 过滤后 n_stmt={len(filt)})\n{'='*60}")
        for s in filt:
            tid = ast_map.get(s["type"], 15)
            sl = s["start_line"] - 1
            snip = lines[sl].strip()[:72] if 0 <= sl < len(lines) else ""
            print(f"  j={s['id']:2d}  ast_id={tid:2d}  {s['type']:22s}  L{s['start_line']}-{s['end_line']}  {snip}")

    dump("gz_uncompress", 369, 386)
    dump("file_uncompress", 429, 474)


if __name__ == "__main__":
    main()

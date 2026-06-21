#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-pack Module-C V9 SFT JSONL into V10 body-slot training format.

V10 keeps the expensive V9 artifacts (Module-B skeleton, slot-BB alignment,
teacher targets, binary graph fields) and rewrites only the SFT surface:

* input prompt follows the proposal order: system, binary_info, skeleton,
  binary_context, output.
* the prompt skeleton contains explicit function signature text and body FILL
  placeholders only.
* the target after ``<|output|>`` is the complete filled function.
* labels supervise only tokens contributed by body slot targets.
* ``sft_slot_output_spans`` records the supervised output spans so collate can
  route those predictor states to the aligned BBs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

try:
    import torch  # type: ignore
    from struclift.utils.skeleton import build_skeleton_for_binary_cfg  # type: ignore
except Exception:
    torch = None
    build_skeleton_for_binary_cfg = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("repack_modulec_v10")

FILL_RE = re.compile(r"<FILL_([A-Z_]+):([^>]+)>")
ALIGNED_BB_RE = re.compile(r"aligned\s+to\s+((?:BB_\d+\s*(?:,\s*)?)*)")
BODY_SLOT_TYPES = {"FILL_STMT", "FILL_COND", "FILL_EXPR"}
HEADER_STMT_SLOT_RE = re.compile(r"\((?:if_header|loop_header|switch_header)\)")
SLOT_KIND_RE = re.compile(r"\(([^()]+)\)")
STATEMENT_TYPES = {
    "if_statement", "while_statement", "for_statement",
    "do_statement", "switch_statement", "case_statement",
    "return_statement", "expression_statement", "declaration",
    "break_statement", "continue_statement", "goto_statement",
    "compound_statement", "labeled_statement", "attributed_statement",
}
AST_TYPE_TO_ID = {
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
DEFAULT_SYSTEM = (
    "<|system|>\n"
    "You are a decompiler. Fill in the placeholders in the code skeleton\n"
    "based on the binary context provided for each slot.\n"
    "<|/system|>"
)

try:
    from tree_sitter_languages import get_parser as _ts_get_parser  # type: ignore
    _TS_PARSER_SRC = "tree_sitter_languages"
except Exception:
    _ts_get_parser = None
    _TS_PARSER_SRC = ""

try:
    import tree_sitter_c as _tsc  # type: ignore
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser  # type: ignore
    if _ts_get_parser is None:
        _TS_PARSER_SRC = "tree_sitter_c"
except Exception:
    _tsc = None
    _TSLanguage = None
    _TSParser = None

_ts_parser: Any = None
_source_stmt_cache: Dict[str, List[Dict[str, Any]]] = {}
_source_stmt_all_cache: Dict[str, List[Dict[str, Any]]] = {}
_source_tree_cache: Dict[str, Tuple[str, bytes, Any]] = {}


def _load_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _tok(tokenizer: Any, text: str) -> List[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def _fill_token(slot_idx: int) -> str:
    return f"<|STRUCLIFT_FILL_{slot_idx:04d}|>"


def _normalise_signature_part(text: Any, fallback: str) -> str:
    s = str(text or "").strip()
    if not s or s.startswith("__STRUCLIFT_UNKNOWN_"):
        return fallback
    return re.sub(r"\s+", " ", s)


def _signature(record: Dict[str, Any]) -> Tuple[str, str, str]:
    fn = str(
        record.get("sft_signature_func_name")
        or record.get("sft_signature_prompt_func_name")
        or record.get("func_name")
        or "func"
    ).strip() or "func"
    ret = _normalise_signature_part(
        record.get("sft_signature_prompt_ret_type")
        or record.get("sft_signature_ret_type")
        or record.get("sft_signature_teacher_ret_type"),
        "int",
    )
    params = _normalise_signature_part(
        record.get("sft_signature_prompt_params")
        or record.get("sft_signature_params")
        or record.get("sft_signature_teacher_params"),
        "void",
    )
    return ret, params, fn


def _function_decl(record: Dict[str, Any], name_for_prompt: str) -> Tuple[str, str]:
    ret, params, fn = _signature(record)
    prompt_decl = f"{ret} {name_for_prompt}({params})"
    output_decl = f"{ret} {name_for_prompt}({params})"
    return prompt_decl, output_decl


def _clean_target_for_output(slot_type: str, target: str) -> str:
    s = str(target or "").strip()
    if not s:
        return ""
    # Keep statement semicolons in the target and do not add an extra one in
    # the skeleton. This avoids V9's common ``;;`` presentation.
    if slot_type == "FILL_STMT" and not s.endswith(";"):
        s += ";"
    if slot_type == "FILL_EXPR":
        s = s.rstrip(";").strip()
    return s


def _strip_c_preprocessor_directives(text: str) -> str:
    s = str(text or "")
    kept: List[str] = []
    for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning)\b", line):
            continue
        kept.append(line)
    s = "\n".join(kept)
    return re.sub(
        r"(?<![A-Za-z0-9_])#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning)\b[^\n;{}]*",
        " ",
        s,
    )


def _protect_c_literals(text: str) -> Tuple[str, List[str]]:
    """Replace string/char literals with sentinels so formatting does not edit them."""
    s = str(text or "")
    literals: List[str] = []
    out: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch not in {"'", '"'}:
            out.append(ch)
            i += 1
            continue
        quote = ch
        j = i + 1
        esc = False
        while j < len(s):
            c = s[j]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                j += 1
                break
            j += 1
        lit = s[i:j]
        token = f"__STRUCLIFT_C_LITERAL_{len(literals)}__"
        literals.append(lit)
        out.append(token)
        i = j
    return "".join(out), literals


def _restore_c_literals(text: str, literals: Sequence[str]) -> str:
    s = str(text or "")
    for i, lit in enumerate(literals):
        s = s.replace(f"__STRUCLIFT_C_LITERAL_{i}__", str(lit))
    return s


def _strip_c_comments(text: str) -> str:
    s = str(text or "")
    protected, literals = _protect_c_literals(s)
    protected = re.sub(r"/\*.*?\*/", " ", protected, flags=re.S)
    protected = re.sub(r"//.*", " ", protected)
    return _restore_c_literals(protected, literals)


def _collapse_ws_outside_literals(text: str) -> str:
    protected, literals = _protect_c_literals(text)
    protected = re.sub(r"\s+", " ", protected).strip()
    return _restore_c_literals(protected, literals)


def _normalise_source_text(text: str) -> str:
    x = _strip_c_preprocessor_directives(str(text or ""))
    x = _strip_c_comments(x)
    return _collapse_ws_outside_literals(x)


def _pretty_c_fragment(text: str, slot_type: str = "") -> str:
    x = _normalise_source_text(text)
    if not x:
        return ""
    protected, literals = _protect_c_literals(x)
    x = protected
    x = re.sub(
        r"\breturn(?=(?:0x[0-9A-Fa-f]+|\d|true\b|false\b|NULL\b|nullptr\b|sizeof\b|[(!*&~+\-]|[A-Za-z_]))",
        "return ",
        x,
    )
    x = re.sub(r"\b(if|while|for|switch)(?=\()", r"\1 ", x)
    x = re.sub(r"\s*,\s*", ", ", x)
    x = re.sub(r"\s+([;\)\]\}])", r"\1", x)
    x = re.sub(r"([\(\[\{])\s+", r"\1", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = _restore_c_literals(x, literals)
    if slot_type == "FILL_STMT" and x and not x.endswith((";", "}")):
        x += ";"
    return x


def _line_looks_like_control_only(line: str) -> bool:
    s = _normalise_source_text(line)
    if not s:
        return True
    if s in {"{", "}", "{;", "};"}:
        return True
    if re.match(r"^(?:else\s+)?(?:if|while|for|switch)\s*\(", s):
        return True
    if re.match(r"^(?:else|do|case\b|default\s*:)", s):
        return True
    return False


def _target_is_usable(slot_type: str, target: str) -> bool:
    x = _pretty_c_fragment(target, slot_type)
    if not x:
        return False
    if slot_type == "FILL_COND":
        if any(ch in x for ch in ";{}"):
            return False
        if x.startswith(("return ", "case ", "break", "continue", "goto ", "else ")):
            return False
        return True
    if slot_type == "FILL_EXPR":
        if any(ch in x for ch in ";{}"):
            return False
        if re.match(r"^\s*(?:if|while|for|switch|return|else|case|break|continue|goto)\b", x):
            return False
        return True
    if slot_type == "FILL_STMT":
        if x.startswith(("else ", "case ")):
            return False
        if re.match(r"^\s*(?:if|while|for|switch)\s*\(", x):
            return False
        if re.fullmatch(r"[{};()\s]+", x):
            return False
        return True
    return True


def _line_to_text(source_file: Any) -> Dict[int, str]:
    p = Path(str(source_file or ""))
    if not p.is_file():
        return {}
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    return {i: line for i, line in enumerate(lines, 1)}


def _record_bb_opcode_hints(record: Dict[str, Any], max_per_bb: int = 8) -> Dict[int, str]:
    lengths = record.get("bin_block_lengths") or []
    instr = record.get("bin_instr_text")
    opcodes = record.get("bin_opcodes") or []
    out: Dict[int, str] = {}
    offset = 0
    for bb, raw_len in enumerate(lengths if isinstance(lengths, list) else []):
        try:
            n = int(raw_len)
        except Exception:
            n = 0
        parts: List[str] = []
        if isinstance(instr, list) and 0 <= bb < len(instr):
            raw = instr[bb]
            if isinstance(raw, list):
                asm = [str(x) for x in raw if str(x).strip()]
            else:
                asm = [p.strip() for p in str(raw or "").split("|") if p.strip()]
            if asm:
                parts.append("instructions: " + " | ".join(asm[:max_per_bb]))
        if not parts and isinstance(opcodes, list) and n > 0:
            chunk = opcodes[offset : offset + n]
            if chunk:
                parts.append("opcode_ids: " + " ".join(str(int(x)) for x in chunk[:max_per_bb]))
        if parts:
            out[bb] = "; ".join(parts)
        offset += max(n, 0)
    return out


def _regenerate_skeleton_text_from_cfg(record: Dict[str, Any]) -> str:
    if torch is None or build_skeleton_for_binary_cfg is None:
        return ""
    labels = record.get("pattern_labels")
    edges = record.get("bin_edge_index")
    if not isinstance(labels, list) or not labels:
        return ""
    if not isinstance(edges, list) or len(edges) != 2:
        return ""
    try:
        n_bb = len(labels)
        pl_t = torch.tensor([int(x) for x in labels], dtype=torch.long)
        edge_index = torch.tensor(
            [[int(x) for x in edges[0]], [int(x) for x in edges[1]]],
            dtype=torch.long,
        )
        skel = build_skeleton_for_binary_cfg(
            pl_t,
            edge_index,
            n_bb,
            _record_bb_opcode_hints(record),
            bb_order_key=_bb_source_order_keys(record, n_bb),
        )
        return str(getattr(skel, "skeleton_text", "") or "").strip()
    except Exception:
        return ""


def _skeleton_text_for_repack(record: Dict[str, Any], source: str) -> Tuple[str, str]:
    requested = str(source or "regenerate").strip().lower()
    old = str(record.get("sft_skeleton_text_pretty") or "").strip()
    if requested in {"v9", "old", "record"}:
        return _repair_predicted_skeleton_text(old), "v9"
    if requested in {"source_ast", "source", "ast", "hybrid_source_ast"}:
        source_ast = _source_ast_skeleton_text_from_source(record)
        if source_ast:
            return source_ast, "source_ast"
        return _repair_predicted_skeleton_text(old), "v9_fallback"
    regenerated = _regenerate_skeleton_text_from_cfg(record)
    if regenerated:
        repaired = _repair_predicted_skeleton_text(regenerated)
        repaired = _repair_source_guided_if_skeleton(repaired, record)
        repaired = _inject_condition_prefix_stmt_slots(repaired, record)
        return repaired, "regenerated"
    return _repair_predicted_skeleton_text(old), "v9_fallback"


def _uses_regenerated_skeleton(skeleton_src_name: str) -> bool:
    return str(skeleton_src_name or "") in {"regenerated", "source_ast"}


def _uses_source_ast_skeleton(skeleton_src_name: str) -> bool:
    return str(skeleton_src_name or "") == "source_ast"


def _bb_source_order_keys(record: Dict[str, Any], n_bb: int) -> List[Tuple[float, float]]:
    """Build per-BB source-order keys from Module-B alignment, then lines.

    This keeps the CFG/CST builder deterministic, but lets training-time
    repacking use the alignment matrix exactly as proposal §6.5 describes.
    Real no-source inference can omit the keys and gets the original CFG order.
    """
    keys: List[Tuple[float, float]] = [(float(i), 0.0) for i in range(max(0, n_bb))]
    stmt_rows = _source_statement_table(record)
    M = record.get("alignment_matrix")
    if (
        isinstance(M, list)
        and len(M) >= n_bb
        and stmt_rows
        and all(isinstance(row, list) and len(row) == len(stmt_rows) for row in M[:n_bb])
    ):
        for bi in range(n_bb):
            row = M[bi]
            best_j = -1
            best_mass = 0.0
            for j, val in enumerate(row):
                try:
                    mass = float(val or 0.0)
                except Exception:
                    mass = 0.0
                if mass > best_mass:
                    best_mass = mass
                    best_j = j
            if best_j >= 0 and best_mass > 1e-8 and best_j < len(stmt_rows):
                stmt = stmt_rows[best_j]
                try:
                    keys[bi] = (
                        float(int(stmt.get("start_line", 0))),
                        float(int(stmt.get("start_col", 0))),
                    )
                except Exception:
                    pass

    # Source-line fallback fills only BBs not covered by alignment mass.
    block_src_lines = record.get("block_src_lines") or []
    if isinstance(block_src_lines, list):
        for bi in range(min(n_bb, len(block_src_lines))):
            if keys[bi][0] != float(bi):
                continue
            lines = _coerce_int_lines(block_src_lines[bi])
            if lines:
                keys[bi] = (float(min(lines)), 0.0)
    return keys


def _bbs_for_source_lines(record: Dict[str, Any], lines: Sequence[int]) -> List[int]:
    wanted = {int(x) for x in lines if isinstance(x, (int, float)) or str(x).isdigit()}
    if not wanted:
        return []
    block_src_lines = record.get("block_src_lines") or []
    if not isinstance(block_src_lines, list):
        return []
    overlaps: List[int] = []
    nearest: List[Tuple[int, int]] = []
    for bb, group in enumerate(block_src_lines):
        if not isinstance(group, list):
            continue
        bb_lines = {
            int(x) for x in group
            if isinstance(x, (int, float)) or str(x).isdigit()
        }
        if not bb_lines:
            continue
        if bb_lines & wanted:
            overlaps.append(int(bb))
        else:
            dist = min(abs(a - b) for a in bb_lines for b in wanted)
            nearest.append((dist, int(bb)))
    if overlaps:
        return sorted(set(overlaps))
    if nearest:
        best = min(d for d, _bb in nearest)
        if best <= 3:
            return sorted({bb for d, bb in nearest if d == best})
    return []


def _source_ast_node_lines(node: Any) -> List[int]:
    try:
        lo = int(node.start_point[0]) + 1
        hi = int(node.end_point[0]) + 1
    except Exception:
        return []
    return list(range(lo, hi + 1))


def _source_ast_branch_statement_nodes(branch: Any) -> List[Any]:
    if branch is None:
        return []
    ntype = _node_type(branch)
    if ntype == "compound_statement":
        return _direct_statement_children(branch)
    if ntype == "else_clause":
        out: List[Any] = []
        for child in getattr(branch, "children", None) or []:
            if _node_type(child) in STATEMENT_TYPES:
                out.append(child)
        return out
    if ntype in STATEMENT_TYPES:
        return [branch]
    children = _direct_statement_children(branch)
    return children or ([branch] if ntype in STATEMENT_TYPES else [])


def _source_ast_condition_text(record: Dict[str, Any], node: Any) -> str:
    ntype = _node_type(node)
    cond = _field_child(node, "condition")
    if cond is None:
        return "1"
    _, source_b, _ = _source_tree_for_record(record)
    raw = _node_text(source_b, cond).strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    if ntype == "for_statement":
        parts = _split_top_level_semicolon_parts(raw)
        if len(parts) >= 2:
            raw = parts[1].strip() or "1"
    return _pretty_c_fragment(raw, "FILL_COND") or "1"


def _source_ast_slot_plan(record: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return "", []
    bounds = _function_body_line_range(record, lt)
    if bounds is None:
        return "", []
    _, source_b, tree = _source_tree_for_record(record)
    if tree is None:
        return "", []
    lo, hi = bounds
    root = tree.root_node
    func_node = None
    stack: List[Any] = [root]
    while stack:
        node = stack.pop()
        if _node_type(node) == "function_definition":
            n_lo = int(node.start_point[0]) + 1
            n_hi = int(node.end_point[0]) + 1
            if n_lo <= lo <= n_hi and n_lo <= hi <= n_hi:
                if func_node is None:
                    func_node = node
                else:
                    old_span = int(func_node.end_point[0]) - int(func_node.start_point[0])
                    new_span = int(node.end_point[0]) - int(node.start_point[0])
                    if new_span < old_span:
                        func_node = node
        for child in reversed(getattr(node, "children", None) or []):
            stack.append(child)
    if func_node is None:
        return "", []
    body = _field_child(func_node, "body")
    if body is None:
        return "", []

    slots: List[Dict[str, Any]] = []

    def add_slot(stype: str, target: str, bbs: Sequence[int], source: str) -> int:
        idx = len(slots)
        clean = _clean_target_for_output(stype, target) if stype in BODY_SLOT_TYPES else str(target or "")
        slots.append({
            "type": stype,
            "target": clean,
            "bbs": [int(x) for x in bbs],
            "source": source,
        })
        return idx

    add_slot("FILL_TYPE", _signature(record)[0], [], "source_ast:signature")
    add_slot("FILL_PARAM", _signature(record)[1], [], "source_ast:signature")

    def comment(bbs: Sequence[int], kind: str) -> str:
        bb_text = ", ".join(f"BB_{int(x)}" for x in bbs) if bbs else "signature/global"
        return f"/* aligned to {bb_text} ({kind}) */"

    def node_bbs(node: Any) -> List[int]:
        return _bbs_for_source_lines(record, _source_ast_node_lines(node))

    def node_stmt_target(node: Any) -> str:
        raw = _node_text(source_b, node)
        return _pretty_c_fragment(raw, "FILL_STMT")

    def serialize_stmt(node: Any, indent: int) -> str:
        pad = "    " * indent
        ntype = _node_type(node)
        if ntype == "compound_statement":
            return "\n".join(serialize_stmt(c, indent) for c in _direct_statement_children(node))

        if ntype == "if_statement":
            bbs = node_bbs(node)
            cond = _source_ast_condition_text(record, node)
            sid = add_slot("FILL_COND", cond, bbs, "source_ast:condition")
            consequence = _field_child(node, "consequence")
            then_lines = [serialize_stmt(c, indent + 1) for c in _source_ast_branch_statement_nodes(consequence)]
            out = [
                f"{pad}if (<FILL_COND:slot_{sid}>) {{{comment(bbs, 'source_ast_if')}",
                "\n".join(x for x in then_lines if x),
                f"{pad}}}",
            ]
            alternative = _field_child(node, "alternative")
            else_nodes = _source_ast_branch_statement_nodes(alternative)
            if else_nodes:
                else_lines = [serialize_stmt(c, indent + 1) for c in else_nodes]
                out[-1] += " else {"
                out.append("\n".join(x for x in else_lines if x))
                out.append(f"{pad}}}")
            return "\n".join(out)

        if ntype in {"while_statement", "for_statement"}:
            prefix: List[str] = []
            if ntype == "for_statement":
                init = _field_child(node, "initializer")
                if init is not None:
                    target = _pretty_c_fragment(_node_text(source_b, init), "FILL_STMT")
                    if target and _target_is_usable("FILL_STMT", target):
                        bbs_init = node_bbs(init)
                        sid_init = add_slot("FILL_STMT", target, bbs_init, "source_ast:for_init")
                        prefix.append(f"{pad}<FILL_STMT:slot_{sid_init}>;  {comment(bbs_init, 'source_ast_for_init')}")
            bbs = node_bbs(node)
            cond = _source_ast_condition_text(record, node)
            sid = add_slot("FILL_COND", cond, bbs, "source_ast:condition")
            body_node = _field_child(node, "body")
            body_lines = [serialize_stmt(c, indent + 1) for c in _source_ast_branch_statement_nodes(body_node)]
            lines = prefix + [
                f"{pad}while (<FILL_COND:slot_{sid}>) {{{comment(bbs, 'source_ast_loop')}",
                "\n".join(x for x in body_lines if x),
            ]
            if ntype == "for_statement":
                update = _field_child(node, "update")
                if update is not None:
                    target = _pretty_c_fragment(_node_text(source_b, update), "FILL_STMT")
                    if target and _target_is_usable("FILL_STMT", target):
                        bbs_upd = node_bbs(update) or bbs
                        sid_upd = add_slot("FILL_STMT", target, bbs_upd, "source_ast:for_update")
                        lines.append(f"{'    ' * (indent + 1)}<FILL_STMT:slot_{sid_upd}>;  {comment(bbs_upd, 'source_ast_for_update')}")
            lines.append(f"{pad}}}")
            return "\n".join(lines)

        if ntype == "do_statement":
            body_node = _field_child(node, "body")
            body_lines = [serialize_stmt(c, indent + 1) for c in _source_ast_branch_statement_nodes(body_node)]
            bbs = node_bbs(node)
            cond = _source_ast_condition_text(record, node)
            sid = add_slot("FILL_COND", cond, bbs, "source_ast:condition")
            return "\n".join([
                f"{pad}do {{",
                "\n".join(x for x in body_lines if x),
                f"{pad}}} while (<FILL_COND:slot_{sid}>);  {comment(bbs, 'source_ast_do')}",
            ])

        if ntype == "switch_statement":
            bbs = node_bbs(node)
            cond = _source_ast_condition_text(record, node)
            sid = add_slot("FILL_COND", cond, bbs, "source_ast:condition")
            body_node = _field_child(node, "body")
            body_lines = [serialize_stmt(c, indent + 1) for c in _source_ast_branch_statement_nodes(body_node)]
            return "\n".join([
                f"{pad}switch (<FILL_COND:slot_{sid}>) {{{comment(bbs, 'source_ast_switch')}",
                "\n".join(x for x in body_lines if x),
                f"{pad}}}",
            ])

        target = node_stmt_target(node)
        if not target or not _target_is_usable("FILL_STMT", target):
            return ""
        bbs = node_bbs(node)
        sid = add_slot("FILL_STMT", target, bbs, "source_ast:stmt")
        return f"{pad}<FILL_STMT:slot_{sid}>;  {comment(bbs, 'source_ast_stmt')}"

    body_lines = [serialize_stmt(c, 1) for c in _direct_statement_children(body)]
    skeleton = (
        "<FILL_TYPE:slot_0> func(<FILL_PARAM:slot_1>) {\n"
        + "\n".join(x for x in body_lines if x)
        + "\n}"
    )
    return skeleton, slots


def _source_ast_skeleton_text_from_source(record: Dict[str, Any]) -> str:
    skeleton, _slots = _source_ast_slot_plan(record)
    return skeleton


def _source_ast_target_for_slot(
    record: Dict[str, Any],
    slot_type: str,
    old_idx: int,
) -> Tuple[str, str]:
    _skeleton, slots = _source_ast_slot_plan(record)
    if not (0 <= int(old_idx) < len(slots)):
        return "", "source_ast_slot_out_of_range"
    item = slots[int(old_idx)]
    if str(item.get("type") or "") != slot_type:
        return "", "source_ast_slot_type_mismatch"
    target = str(item.get("target") or "")
    if slot_type in BODY_SLOT_TYPES and target and _target_is_usable(slot_type, target):
        return target, str(item.get("source") or "source_ast")
    return "", "source_ast_empty_or_unusable"


def _repair_predicted_skeleton_text(text: str) -> str:
    """Fix obvious invalid skeleton punctuation before V10 repacking."""
    s = str(text or "")
    s = re.sub(r"(<FILL_COND:[^>]+>)\s*;+(\s*\))", r"\1\2", s)
    s = re.sub(r"(<FILL_COND:[^>]+>)\s*;+(\s*\])", r"\1\2", s)
    s = _collapse_duplicate_nested_if_placeholders(s)
    return s


def _normalise_placeholder_condition_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _collapse_duplicate_nested_if_placeholders(text: str) -> str:
    """Collapse AB artifacts like if (X) { if (X) { body } } to one layer."""
    s = str(text or "")
    changed = True
    while changed:
        changed = False
        out: List[str] = []
        i = 0
        for m in re.finditer(r"\bif\s*\(", s):
            if m.start() < i:
                continue
            outer_open = s.find("(", m.end() - 1)
            outer_close = _balanced_param_end(s, outer_open)
            if outer_close < 0:
                continue
            j = outer_close + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j >= len(s) or s[j] != "{":
                continue
            outer_body_open = j
            k = outer_body_open + 1
            while k < len(s) and s[k].isspace():
                k += 1
            inner = re.match(r"if\s*\(", s[k:])
            if not inner:
                continue
            inner_open = s.find("(", k + inner.end() - 1)
            inner_close = _balanced_param_end(s, inner_open)
            if inner_close < 0:
                continue
            if (
                _normalise_placeholder_condition_for_compare(s[outer_open + 1 : outer_close])
                != _normalise_placeholder_condition_for_compare(s[inner_open + 1 : inner_close])
            ):
                continue
            inner_body_start = inner_close + 1
            while inner_body_start < len(s) and s[inner_body_start].isspace():
                inner_body_start += 1
            if inner_body_start >= len(s) or s[inner_body_start] != "{":
                continue
            inner_body_end = _balanced_brace_end(s, inner_body_start)
            outer_body_end = _balanced_brace_end(s, outer_body_open)
            if inner_body_end < 0 or outer_body_end < 0:
                continue
            tail = s[inner_body_end + 1 : outer_body_end].strip()
            # Only collapse the unambiguous case: the outer body consists of
            # exactly the duplicate inner if, with no sibling else/body code.
            if tail:
                continue
            out.append(s[i:k])
            out.append(s[inner_body_start + 1 : inner_body_end])
            i = inner_body_end + 1
            changed = True
            break
        if changed:
            out.append(s[i:])
            s = "".join(out)
    return s


def _skeleton_noncomment_text(text: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", str(text or ""), flags=re.S)
    s = re.sub(r"//.*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _first_code_offset(text: str) -> int:
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                return n
            i = j + 2
            continue
        if text.startswith("//", i):
            j = text.find("\n", i + 2)
            if j < 0:
                return n
            i = j + 1
            continue
        break
    return i


def _parse_skeleton_if_at(text: str, if_pos: int) -> Optional[Dict[str, int]]:
    s = str(text or "")
    end = len(s)
    if not _keyword_at(s, int(if_pos), "if"):
        return None
    open_paren = s.find("(", int(if_pos) + 2, end)
    close_paren = _balanced_param_end(s, open_paren)
    if open_paren < 0 or close_paren < 0:
        return None
    body_open = _skip_ws(s, close_paren + 1, end)
    if body_open >= end or s[body_open] != "{":
        return None
    body_close = _balanced_brace_end(s, body_open)
    if body_close < 0:
        return None
    out: Dict[str, int] = {
        "start": int(if_pos),
        "open_paren": open_paren,
        "close_paren": close_paren,
        "body_open": body_open,
        "body_close": body_close,
        "full_end": body_close + 1,
        "else_kw": -1,
        "else_open": -1,
        "else_close": -1,
    }
    j = _skip_ws(s, body_close + 1, end)
    if _keyword_at(s, j, "else"):
        else_open = _skip_ws(s, j + 4, end)
        if else_open < end and s[else_open] == "{":
            else_close = _balanced_brace_end(s, else_open)
            if else_close >= 0:
                out.update({
                    "else_kw": j,
                    "else_open": else_open,
                    "else_close": else_close,
                    "full_end": else_close + 1,
                })
    return out


def _placeholder_id_from_condition_text(cond: str) -> str:
    m = FILL_RE.search(str(cond or ""))
    if not m or f"FILL_{m.group(1)}" != "FILL_COND":
        return ""
    return str(m.group(2))


def _source_if_has_else_for_slot(
    record: Dict[str, Any],
    bbs: Sequence[int],
    cond_target: str,
) -> Optional[bool]:
    row = _source_control_row_for_slot(record, bbs, cond_target, {"if_statement"})
    if row is None:
        return None
    node = _find_tree_sitter_node_for_row(record, row)
    if node is not None:
        return _field_child(node, "alternative") is not None
    raw = _statement_text_from_lt(_line_to_text(record.get("source_file")), row)
    return bool(re.search(r"\belse\b", _normalise_source_text(raw)))


def _source_control_type_for_slot(
    record: Dict[str, Any],
    bbs: Sequence[int],
    cond_target: str,
) -> str:
    row = _source_control_row_for_slot(record, bbs, cond_target)
    return str(row.get("type") or "") if row is not None else ""


def _source_control_row_for_slot(
    record: Dict[str, Any],
    bbs: Sequence[int],
    cond_target: str,
    control_types: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    key = _normalise_condition_for_compare(cond_target)
    if not key:
        return None
    allowed = set(control_types or {"if_statement", "while_statement", "for_statement", "switch_statement"})
    stmt_rows = _source_statement_table(record)
    row_candidates: List[Dict[str, Any]] = []
    if stmt_rows:
        candidates, _masses = _candidate_alignment_columns(record, bbs, len(stmt_rows))
        row_candidates = [
            stmt_rows[j] for j in candidates
            if 0 <= int(j) < len(stmt_rows)
        ]
    if not row_candidates:
        bb_lines = set(_line_numbers_for_bbs(record, bbs))
        row_candidates = [
            row for row in _source_all_statement_rows(record)
            if str(row.get("type") or "") in allowed
            and (_row_line_set(row) & bb_lines)
        ]

    for row in sorted(
        row_candidates,
        key=lambda r: (
            int(r.get("start_line", 0)),
            int(r.get("end_line", 0)),
            int(r.get("depth", 0)),
        ),
    ):
        if str(row.get("type") or "") not in allowed:
            continue
        cond = _condition_from_control_row(record, row)
        if _normalise_condition_for_compare(cond) != key:
            continue
        return row
    return None


def _condition_meta_for_skeleton(record: Dict[str, Any], skeleton_src: str) -> Dict[str, Dict[str, Any]]:
    matches = list(FILL_RE.finditer(skeleton_src or ""))
    out: Dict[str, Dict[str, Any]] = {}
    for mi, m in enumerate(matches):
        if f"FILL_{m.group(1)}" != "FILL_COND":
            continue
        slot_id = str(m.group(2))
        next_start = matches[mi + 1].start() if mi + 1 < len(matches) else len(skeleton_src)
        slot_tail = skeleton_src[m.end():next_start]
        bbs = _aligned_bbs_from_slot_tail(slot_tail)
        expected = _expected_control_types_from_slot_tail(slot_tail)
        target, reason = _source_alignment_target_for_slot(
            record,
            "FILL_COND",
            bbs,
            expected_control_types=expected,
        )
        if not target:
            target, reason2 = _source_multistmt_target_for_slot(
                record,
                "FILL_COND",
                bbs,
                body_slot_count=0,
                old_target="",
            )
            reason = reason2 or reason
        key = _normalise_condition_for_compare(target)
        out[slot_id] = {
            "bbs": bbs,
            "target": target,
            "key": key,
            "source": reason,
            "source_if_has_else": _source_if_has_else_for_slot(record, bbs, target),
            "source_control_type": _source_control_type_for_slot(record, bbs, target),
        }
    return out


def _repair_single_branch_else_polarity(text: str, cond_meta: Dict[str, Dict[str, Any]]) -> Tuple[str, bool]:
    s = str(text or "")
    for m in reversed(list(re.finditer(r"\bif\s*\(", s))):
        info = _parse_skeleton_if_at(s, m.start())
        if info is None or info["else_open"] < 0:
            continue
        cond = s[info["open_paren"] + 1: info["close_paren"]]
        slot_id = _placeholder_id_from_condition_text(cond)
        meta = cond_meta.get(slot_id)
        if not meta or meta.get("source_if_has_else") is not False:
            continue
        then_body = s[info["body_open"] + 1: info["body_close"]]
        else_body = s[info["else_open"] + 1: info["else_close"]]
        if _skeleton_noncomment_text(then_body):
            continue
        if not _skeleton_noncomment_text(else_body):
            continue
        replacement = (
            s[info["start"]: info["body_open"] + 1]
            + then_body
            + else_body
            + s[info["body_close"]: info["body_close"] + 1]
        )
        return s[:info["start"]] + replacement + s[info["full_end"]:], True
    return s, False


def _repair_duplicate_nested_source_if(text: str, cond_meta: Dict[str, Dict[str, Any]]) -> Tuple[str, bool]:
    s = str(text or "")
    for m in reversed(list(re.finditer(r"\bif\s*\(", s))):
        outer = _parse_skeleton_if_at(s, m.start())
        if outer is None or outer["else_open"] >= 0:
            continue
        outer_cond = s[outer["open_paren"] + 1: outer["close_paren"]]
        outer_slot = _placeholder_id_from_condition_text(outer_cond)
        outer_key = str(cond_meta.get(outer_slot, {}).get("key") or "")
        if not outer_key:
            continue
        body_abs_start = outer["body_open"] + 1
        body_abs_end = outer["body_close"]
        body = s[body_abs_start:body_abs_end]
        inner_rel = _first_code_offset(body)
        if inner_rel >= len(body):
            continue
        inner_abs = body_abs_start + inner_rel
        inner = _parse_skeleton_if_at(s, inner_abs)
        if inner is None or inner["else_open"] >= 0:
            continue
        tail = s[inner["full_end"]:body_abs_end]
        if _skeleton_noncomment_text(tail):
            continue
        inner_cond = s[inner["open_paren"] + 1: inner["close_paren"]]
        inner_slot = _placeholder_id_from_condition_text(inner_cond)
        inner_key = str(cond_meta.get(inner_slot, {}).get("key") or "")
        if outer_key != inner_key:
            continue
        prefix = body[:inner_rel]
        inner_body = s[inner["body_open"] + 1: inner["body_close"]]
        replacement = (
            s[outer["start"]: outer["body_open"] + 1]
            + prefix
            + inner_body
            + s[outer["body_close"]: outer["body_close"] + 1]
        )
        return s[:outer["start"]] + replacement + s[outer["full_end"]:], True
    return s, False


def _repair_redundant_nested_source_if_anywhere(
    text: str,
    cond_meta: Dict[str, Dict[str, Any]],
    source_cond_counts: Dict[str, int],
) -> Tuple[str, bool]:
    """Unwrap duplicate source-if shells that survived the narrow repair.

    CFG short-circuit lowering can produce a duplicate condition inside a
    larger if body, with other statement slots before/after it.  When the real
    source contains that condition only once, the inner skeleton ``if`` is just
    a routing artifact.  Keep its non-empty branch body and remove only the
    redundant condition shell.
    """
    s = str(text or "")
    infos: List[Dict[str, Any]] = []
    for m in re.finditer(r"\bif\s*\(", s):
        info = _parse_skeleton_if_at(s, m.start())
        if info is None:
            continue
        cond = s[info["open_paren"] + 1: info["close_paren"]]
        slot = _placeholder_id_from_condition_text(cond)
        key = str(cond_meta.get(slot, {}).get("key") or "")
        if not key:
            continue
        info = dict(info)
        info["key"] = key
        infos.append(info)

    for inner in reversed(infos):
        key = str(inner.get("key") or "")
        if not key or int(source_cond_counts.get(key, 0)) > 1:
            continue
        ancestors = [
            outer for outer in infos
            if int(outer["start"]) < int(inner["start"])
            and outer.get("key") == key
            and (
                int(outer["body_open"]) < int(inner["start"]) < int(outer["body_close"])
                or (
                    int(outer.get("else_open", -1)) >= 0
                    and int(outer["else_open"]) < int(inner["start"]) < int(outer["else_close"])
                )
            )
        ]
        if not ancestors:
            continue
        then_body = s[int(inner["body_open"]) + 1: int(inner["body_close"])]
        else_body = (
            s[int(inner["else_open"]) + 1: int(inner["else_close"])]
            if int(inner.get("else_open", -1)) >= 0
            else ""
        )
        then_has = bool(_skeleton_noncomment_text(then_body))
        else_has = bool(_skeleton_noncomment_text(else_body))
        if int(inner.get("else_open", -1)) >= 0:
            if then_has and not else_has:
                replacement = then_body
            elif else_has and not then_has:
                replacement = else_body
            elif not then_has and not else_has:
                replacement = ""
            else:
                continue
        else:
            replacement = then_body
        return s[: int(inner["start"])] + replacement + s[int(inner["full_end"]):], True
    return s, False


def _repair_redundant_sibling_source_if(
    text: str,
    cond_meta: Dict[str, Dict[str, Any]],
    source_cond_counts: Dict[str, int],
) -> Tuple[str, bool]:
    """Merge repeated same-condition sibling if shells into the first shell."""
    s = str(text or "")
    infos: List[Dict[str, Any]] = []
    for m in re.finditer(r"\bif\s*\(", s):
        info = _parse_skeleton_if_at(s, m.start())
        if info is None:
            continue
        cond = s[int(info["open_paren"]) + 1:int(info["close_paren"])]
        slot = _placeholder_id_from_condition_text(cond)
        key = str(cond_meta.get(slot, {}).get("key") or "")
        if not key or int(source_cond_counts.get(key, 0)) > 1:
            continue
        item = dict(info)
        item["key"] = key
        infos.append(item)

    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for info in infos:
        by_key.setdefault(str(info["key"]), []).append(info)

    for key, group in by_key.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: int(x["start"]))
        keeper = None
        for cand in group:
            if int(cand.get("else_open", -1)) < 0:
                keeper = cand
                break
        if keeper is None:
            continue
        for victim in group:
            if victim is keeper:
                continue
            if int(victim["start"]) < int(keeper["full_end"]):
                continue
            then_body = s[int(victim["body_open"]) + 1:int(victim["body_close"])]
            else_body = (
                s[int(victim["else_open"]) + 1:int(victim["else_close"])]
                if int(victim.get("else_open", -1)) >= 0
                else ""
            )
            then_has = bool(_skeleton_noncomment_text(then_body))
            else_has = bool(_skeleton_noncomment_text(else_body))
            if int(victim.get("else_open", -1)) >= 0:
                if then_has and not else_has:
                    moving = then_body
                elif else_has and not then_has:
                    moving = else_body
                elif not then_has and not else_has:
                    moving = ""
                else:
                    continue
            else:
                moving = then_body
            moving = moving.strip("\n")
            insertion = ("\n" + moving + "\n") if moving.strip() else "\n"
            return (
                s[: int(keeper["body_close"])]
                + insertion
                + s[int(keeper["body_close"]): int(victim["start"])]
                + s[int(victim["full_end"]):]
            ), True
    return s, False


def _repair_else_if_chain_shells(text: str, cond_meta: Dict[str, Dict[str, Any]]) -> Tuple[str, bool]:
    """Remove empty outer if shells around real ``else if`` headers.

    The source AST represents ``else if`` as an if_statement inside the
    alternative branch.  CFG labels can serialize this as ``else { if (...) }``;
    when the outer then branch is empty and the inner condition is the real
    source if, the outer shell is only a routing artifact.
    """
    s = str(text or "")
    for m in reversed(list(re.finditer(r"\bif\s*\(", s))):
        outer = _parse_skeleton_if_at(s, m.start())
        if outer is None or int(outer.get("else_open", -1)) < 0:
            continue
        then_body = s[int(outer["body_open"]) + 1:int(outer["body_close"])]
        else_body = s[int(outer["else_open"]) + 1:int(outer["else_close"])]
        if _skeleton_noncomment_text(then_body):
            continue
        inner_rel = _first_code_offset(else_body)
        if inner_rel >= len(else_body):
            continue
        inner_abs = int(outer["else_open"]) + 1 + inner_rel
        inner = _parse_skeleton_if_at(s, inner_abs)
        if inner is None:
            continue
        tail = s[int(inner["full_end"]):int(outer["else_close"])]
        if _skeleton_noncomment_text(tail):
            continue
        inner_cond = s[int(inner["open_paren"]) + 1:int(inner["close_paren"])]
        inner_slot = _placeholder_id_from_condition_text(inner_cond)
        inner_meta = cond_meta.get(inner_slot) or {}
        if inner_meta.get("source_control_type") != "if_statement":
            continue
        replacement = else_body[:inner_rel] + s[int(inner["start"]):int(inner["full_end"])]
        return s[:int(outer["start"])] + replacement + s[int(outer["full_end"]):], True
    return s, False


def _repair_source_guided_if_skeleton(text: str, record: Dict[str, Any]) -> str:
    """Use source alignment only to repair CFG skeleton shape, not to fill code.

    Short-circuit conditions often produce several binary IF_HEADER BBs for one
    C ``if``.  The pure CFG builder can serialize those as nested duplicate
    ``if`` statements, sometimes with the single real branch under ``else``.
    We keep the input as empty slots, but source-alignment tells us that those
    slots correspond to the same source predicate and whether the source if has
    an ``else`` branch.  This lets us collapse the skeleton before targets are
    extracted, instead of training on duplicated conditions.
    """
    s = str(text or "")
    if not s or "<FILL_COND:" not in s:
        return s
    cond_meta = _condition_meta_for_skeleton(record, s)
    if not cond_meta:
        return s
    source_cond_counts: Dict[str, int] = {}
    for cond_key in _control_conditions_in_text(_source_function_text(record)):
        source_cond_counts[cond_key] = source_cond_counts.get(cond_key, 0) + 1
    for _ in range(16):
        changed = False
        s, did = _repair_single_branch_else_polarity(s, cond_meta)
        changed = changed or did
        s, did = _repair_duplicate_nested_source_if(s, cond_meta)
        changed = changed or did
        s, did = _repair_redundant_nested_source_if_anywhere(s, cond_meta, source_cond_counts)
        changed = changed or did
        s, did = _repair_redundant_sibling_source_if(s, cond_meta, source_cond_counts)
        changed = changed or did
        s, did = _repair_else_if_chain_shells(s, cond_meta)
        changed = changed or did
        if not changed:
            break
    return s


def _source_condition_prefix_target_for_slot(
    record: Dict[str, Any],
    bbs: Sequence[int],
    expected_control_types: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    cond_target, _reason = _source_alignment_target_for_slot(
        record,
        "FILL_COND",
        bbs,
        expected_control_types=expected_control_types,
    )
    if not cond_target:
        cond_target, _reason = _source_multistmt_target_for_slot(
            record,
            "FILL_COND",
            bbs,
            body_slot_count=0,
            old_target="",
        )
    row = _source_control_row_for_slot(record, bbs, cond_target)
    if row is None:
        return "", ""
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return "", ""
    lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    bounds = _function_body_line_range(record, lt_clean)
    bb_lines = _clamp_lines_to_bounds(_line_numbers_for_bbs(record, bbs), bounds)
    if not bb_lines:
        return "", ""
    try:
        control_line = int(row.get("start_line", 0))
        control_col = int(row.get("start_col", 0))
    except Exception:
        return "", ""
    pieces: List[str] = []
    prefix_lines = [ln for ln in bb_lines if ln < control_line]
    if prefix_lines:
        pieces.append(_source_lines_text(lt_clean, prefix_lines))
    if control_line in bb_lines and control_col > 0 and control_line in lt_clean:
        pieces.append(str(lt_clean.get(control_line, ""))[:control_col])
    raw = "\n".join(x for x in pieces if str(x or "").strip())
    if not raw:
        return "", ""
    if re.fullmatch(r"\s*else\s*;?\s*", _normalise_source_text(raw)):
        return "", ""
    stmts: List[str] = []
    seen: Set[str] = set()
    for frag in _split_semicolon_statements(raw):
        stmt = _pretty_c_fragment(frag, "FILL_STMT")
        if not stmt or not _target_is_usable("FILL_STMT", stmt):
            continue
        if re.search(r"\b(?:if|while|for|switch)\s*\(", stmt):
            continue
        key = _normalise_source_text(stmt)
        if key in seen:
            continue
        seen.add(key)
        stmts.append(stmt)
    if not stmts:
        return "", ""
    return _clean_target_for_output("FILL_STMT", " ".join(stmts)), "source_block_lines:stmt:cond_prefix"


def _insert_prefix_slot_before_trailing_control(chunk: str, slot_text: str) -> Tuple[str, bool]:
    m = re.search(r"\b(?:if|while|for|switch)\s*\(\s*$", chunk)
    if not m:
        return chunk, False
    line_start = chunk.rfind("\n", 0, m.start()) + 1
    indent = re.match(r"[ \t]*", chunk[line_start:m.start()])
    pad = indent.group(0) if indent else ""
    return chunk[:m.start()] + f"{pad}{slot_text}\n" + chunk[m.start():], True


def _inject_condition_prefix_stmt_slots(text: str, record: Dict[str, Any]) -> str:
    s = str(text or "")
    if "<FILL_COND:" not in s:
        return s
    matches = list(FILL_RE.finditer(s))
    max_slot = -1
    for m in matches:
        sm = re.search(r"(\d+)$", str(m.group(2)))
        if sm:
            max_slot = max(max_slot, int(sm.group(1)))
    next_slot = max_slot + 1
    out: List[str] = []
    pos = 0
    inserted_for_bbs: Set[Tuple[int, ...]] = set()
    for mi, m in enumerate(matches):
        chunk = s[pos:m.start()]
        if f"FILL_{m.group(1)}" == "FILL_COND":
            next_start = matches[mi + 1].start() if mi + 1 < len(matches) else len(s)
            slot_tail = s[m.end():next_start]
            bbs = _aligned_bbs_from_slot_tail(slot_tail)
            expected = _expected_control_types_from_slot_tail(slot_tail)
            key = tuple(int(x) for x in bbs)
            prefix_target, _reason = _source_condition_prefix_target_for_slot(
                record,
                bbs,
                expected_control_types=expected,
            )
            if prefix_target and key not in inserted_for_bbs:
                bb_text = ", ".join(f"BB_{int(x)}" for x in bbs) if bbs else "signature/global"
                slot_text = (
                    f"<FILL_STMT:slot_{next_slot}>;  "
                    f"/* aligned to {bb_text} (cond_prefix) */"
                )
                chunk2, ok = _insert_prefix_slot_before_trailing_control(chunk, slot_text)
                if ok:
                    chunk = chunk2
                    inserted_for_bbs.add(key)
                    next_slot += 1
        out.append(chunk)
        out.append(m.group(0))
        pos = m.end()
    out.append(s[pos:])
    return "".join(out)


def _get_c_parser() -> Any:
    global _ts_parser
    if _ts_parser is not None:
        return _ts_parser
    if _ts_get_parser is not None:
        try:
            _ts_parser = _ts_get_parser("c")
            return _ts_parser
        except Exception:
            _ts_parser = None
    if _tsc is not None and _TSParser is not None:
        try:
            _ts_parser = _TSParser(_tsc.language())
            return _ts_parser
        except TypeError:
            try:
                lang = _TSLanguage(_tsc.language()) if _TSLanguage is not None else _tsc.language()
                _ts_parser = _TSParser(lang)
                return _ts_parser
            except Exception:
                try:
                    _ts_parser = _TSParser()
                    _ts_parser.language = _TSLanguage(_tsc.language())
                    return _ts_parser
                except Exception:
                    _ts_parser = None
    return None


def _node_type(node: Any) -> str:
    t = getattr(node, "type", None) or getattr(node, "symbol", str(node))
    return t.decode("utf-8") if isinstance(t, bytes) else str(t)


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _keyword_at(text: str, pos: int, keyword: str) -> bool:
    n = len(keyword)
    if text[pos : pos + n] != keyword:
        return False
    if pos > 0 and _is_ident_char(text[pos - 1]):
        return False
    if pos + n < len(text) and _is_ident_char(text[pos + n]):
        return False
    return True


def _skip_ws(text: str, pos: int, end: int) -> int:
    while pos < end and text[pos].isspace():
        pos += 1
    return pos


def _statement_span_from_pos(text: str, pos: int, end: int) -> Optional[Tuple[int, int]]:
    i = _skip_ws(text, pos, end)
    if i >= end:
        return None
    if text[i] == "{":
        j = _balanced_brace_end(text, i)
        if j >= 0:
            return i, min(j + 1, end)
        return None
    start = i
    par = brk = 0
    in_s = in_c = esc = False
    for j in range(i, end):
        ch = text[j]
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
    return start, end


def _control_statement_full_span(text: str, pos: int, end: int, keyword: str) -> Optional[Tuple[int, int]]:
    if keyword == "do":
        body = _statement_span_from_pos(text, pos + 2, end)
        if body is None:
            return None
        j = _skip_ws(text, body[1], end)
        if not _keyword_at(text, j, "while"):
            return pos, body[1]
        open_paren = text.find("(", j, end)
        close_paren = _balanced_param_end(text, open_paren)
        if close_paren < 0:
            return pos, body[1]
        semi = text.find(";", close_paren, end)
        return pos, semi + 1 if semi >= 0 else close_paren + 1

    open_paren = text.find("(", pos + len(keyword), end)
    close_paren = _balanced_param_end(text, open_paren)
    if close_paren < 0:
        return None
    body = _statement_span_from_pos(text, close_paren + 1, end)
    if body is None:
        return pos, close_paren + 1
    full_end = body[1]
    if keyword == "if":
        j = _skip_ws(text, full_end, end)
        if _keyword_at(text, j, "else"):
            else_body = _statement_span_from_pos(text, j + 4, end)
            if else_body is not None:
                full_end = else_body[1]
    return pos, full_end


def _scan_simple_c_statements(
    text: str,
    spans: List[Tuple[int, int, int]],
    start: int,
    end: int,
    depth: int,
    out: List[Dict[str, int]],
) -> None:
    i = start
    while i < end:
        i = _skip_ws(text, i, end)
        if i >= end:
            break
        if text[i] == "}":
            return
        if text[i] == "{":
            j = _balanced_brace_end(text, i)
            if j < 0 or j >= end:
                return
            out.append({
                "id": len(out),
                "type": "compound_statement",
                "start_line": _line_for_char_pos(spans, i) or 1,
                "end_line": _line_for_char_pos(spans, j) or (_line_for_char_pos(spans, i) or 1),
                "depth": depth,
            })
            _scan_simple_c_statements(text, spans, i + 1, j, depth + 1, out)
            i = j + 1
            continue

        matched_control = False
        for kw, typ in (
            ("if", "if_statement"),
            ("while", "while_statement"),
            ("for", "for_statement"),
            ("switch", "switch_statement"),
            ("do", "do_statement"),
        ):
            if not _keyword_at(text, i, kw):
                continue
            span = _control_statement_full_span(text, i, end, kw)
            if span is None:
                continue
            s, e = span
            out.append({
                "id": len(out),
                "type": typ,
                "start_line": _line_for_char_pos(spans, s) or 1,
                "end_line": _line_for_char_pos(spans, max(s, e - 1)) or (_line_for_char_pos(spans, s) or 1),
                "depth": depth,
            })
            # Pre-order approximation: parent control first, then statements in
            # its body. This matches the tree-sitter traversal used by the
            # original dataset closely enough for source-derived GT recovery.
            open_paren = text.find("(", i + len(kw), e) if kw != "do" else -1
            close_paren = _balanced_param_end(text, open_paren) if open_paren >= 0 else -1
            body_start = close_paren + 1 if close_paren >= 0 else i + len(kw)
            body = _statement_span_from_pos(text, body_start, e)
            if body is not None:
                bs, be = body
                if bs < be and text[bs] == "{" and be > bs + 1:
                    _scan_simple_c_statements(text, spans, bs + 1, be - 1, depth + 1, out)
                else:
                    _scan_simple_c_statements(text, spans, bs, be, depth + 1, out)
            if kw == "if":
                j = _skip_ws(text, body[1] if body is not None else body_start, e)
                if _keyword_at(text, j, "else"):
                    else_body = _statement_span_from_pos(text, j + 4, e)
                    if else_body is not None:
                        es, ee = else_body
                        if es < ee and text[es] == "{" and ee > es + 1:
                            _scan_simple_c_statements(text, spans, es + 1, ee - 1, depth + 1, out)
                        else:
                            _scan_simple_c_statements(text, spans, es, ee, depth + 1, out)
            i = e
            matched_control = True
            break
        if matched_control:
            continue

        if _keyword_at(text, i, "case") or _keyword_at(text, i, "default"):
            colon = text.find(":", i, end)
            if colon >= 0:
                out.append({
                    "id": len(out),
                    "type": "case_statement",
                    "start_line": _line_for_char_pos(spans, i) or 1,
                    "end_line": _line_for_char_pos(spans, colon) or (_line_for_char_pos(spans, i) or 1),
                    "depth": depth,
                })
                i = colon + 1
                continue

        span = _statement_span_from_pos(text, i, end)
        if span is None:
            break
        s, e = span
        frag = text[s:e].strip()
        if frag:
            typ = "expression_statement"
            for kw, ktyp in (
                ("return", "return_statement"),
                ("break", "break_statement"),
                ("continue", "continue_statement"),
                ("goto", "goto_statement"),
            ):
                if _keyword_at(text, s, kw):
                    typ = ktyp
                    break
            out.append({
                "id": len(out),
                "type": typ,
                "start_line": _line_for_char_pos(spans, s) or 1,
                "end_line": _line_for_char_pos(spans, max(s, e - 1)) or (_line_for_char_pos(spans, s) or 1),
                "depth": depth,
            })
        i = max(e, i + 1)


def _fallback_statement_table_from_source(
    lt: Dict[int, str],
    bounds: Tuple[int, int],
) -> List[Dict[str, Any]]:
    lo, hi = bounds
    line_nums = [ln for ln in range(lo, hi + 1) if ln in lt]
    if not line_nums:
        return []
    text, spans = _build_char_line_index(line_nums, lt)
    open_brace = text.find("{")
    close_brace = _balanced_brace_end(text, open_brace)
    if open_brace < 0 or close_brace < 0:
        body_start, body_end = 0, len(text)
    else:
        body_start, body_end = open_brace + 1, close_brace
    out: List[Dict[str, int]] = []
    _scan_simple_c_statements(text, spans, body_start, body_end, 1, out)
    return [{**row, "id": i} for i, row in enumerate(out)]


def _statement_text_from_lt(lt: Dict[int, str], row: Dict[str, Any]) -> str:
    try:
        lo = int(row["start_line"])
        hi = int(row["end_line"])
    except Exception:
        return ""
    if lo > hi:
        return ""
    lines = [str(lt.get(ln, "") or "") for ln in range(lo, hi + 1)]
    if not lines:
        return ""
    start_col = row.get("start_col")
    end_col = row.get("end_col")
    try:
        sc = int(start_col)
    except Exception:
        sc = 0
    try:
        ec = int(end_col)
    except Exception:
        ec = -1
    if len(lines) == 1:
        line = lines[0]
        return line[max(sc, 0): ec if ec >= 0 else len(line)]
    lines[0] = lines[0][max(sc, 0):]
    if ec >= 0:
        lines[-1] = lines[-1][:ec]
    return "\n".join(lines)


def _stmt_type_id(row: Dict[str, Any]) -> int:
    return AST_TYPE_TO_ID.get(str(row.get("type")), 15)


def _row_line_set(row: Dict[str, Any]) -> Set[int]:
    try:
        lo = int(row["start_line"])
        hi = int(row["end_line"])
    except Exception:
        return set()
    return set(range(lo, hi + 1))


def _coerce_int_lines(xs: Any) -> Set[int]:
    out: Set[int] = set()
    if not isinstance(xs, list):
        return out
    for x in xs:
        if isinstance(x, (int, float)) or str(x).isdigit():
            out.add(int(x))
    return out


def _source_statement_columns_from_alignment(
    record: Dict[str, Any],
    filtered: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map compressed alignment_matrix columns back to source statement rows.

    The MD builder stores ``alignment_matrix`` columns after dropping unused
    source statement columns, but it does not serialize the original statement
    ranges. Reconstruct them by using each column's aligned BB source lines
    plus ``src_ast_types``. This keeps Module-C GT tied to BB->source alignment
    instead of to old V9 teacher strings.
    """
    M = record.get("alignment_matrix")
    if not isinstance(M, list) or not M or not isinstance(M[0], list):
        return filtered
    n_cols = len(M[0])
    if n_cols <= 0:
        return []
    for row in M:
        if not isinstance(row, list) or len(row) != n_cols:
            return []

    ast_types = record.get("src_ast_types")
    block_src_lines = record.get("block_src_lines") or []
    source_lines = record.get("source_lines")
    source_line_cols = [
        int(x) for x in source_lines
        if isinstance(x, (int, float)) or str(x).isdigit()
    ] if isinstance(source_lines, list) else []
    ast_type_cols = []
    if isinstance(ast_types, list):
        for x in ast_types:
            try:
                ast_type_cols.append(int(x))
            except Exception:
                ast_type_cols.append(15)
    source_line_column_like = len(source_line_cols) == n_cols
    is_line_fallback = (
        str(record.get("src_encoding_tier") or "") == "line_fallback"
        or (len(source_line_cols) == n_cols and ast_type_cols and all(x == 15 for x in ast_type_cols))
    )
    if is_line_fallback and len(source_line_cols) == n_cols:
        return [
            {
                "id": i,
                "alignment_col": i,
                "type": "source_line",
                "start_line": int(ln),
                "end_line": int(ln),
                "start_col": 0,
                "end_col": -1,
                "depth": 0,
            }
            for i, ln in enumerate(source_line_cols)
        ]
    if len(filtered) == n_cols:
        out: List[Dict[str, Any]] = []
        for i, row in enumerate(filtered):
            if ast_type_cols and i < len(ast_type_cols) and ast_type_cols[i] != _stmt_type_id(row):
                if source_line_column_like:
                    return _line_column_rows(source_line_cols)
                return []
            out.append({**row, "id": i, "alignment_col": i})
        return out

    selected: List[Dict[str, Any]] = []
    used: Set[int] = set()
    prev_idx = -1
    for col in range(n_cols):
        wanted_type: Optional[int] = None
        if isinstance(ast_types, list) and col < len(ast_types):
            try:
                wanted_type = int(ast_types[col])
            except Exception:
                wanted_type = None

        col_bbs = [
            bi for bi, row in enumerate(M)
            if isinstance(row, list)
            and col < len(row)
            and float(row[col] or 0.0) > 1e-8
        ]
        col_lines: Set[int] = set()
        if isinstance(block_src_lines, list):
            for bi in col_bbs:
                if 0 <= bi < len(block_src_lines):
                    col_lines.update(_coerce_int_lines(block_src_lines[bi]))

        best_i: Optional[int] = None
        best_score: Optional[Tuple[float, int, int, int]] = None
        for i, stmt in enumerate(filtered):
            if i in used:
                continue
            stmt_lines = _row_line_set(stmt)
            overlap = len(stmt_lines & col_lines) if col_lines else 0
            if col_lines and overlap <= 0:
                continue
            type_id = _stmt_type_id(stmt)
            type_match = wanted_type is None or type_id == wanted_type
            if wanted_type is not None and not type_match:
                # OTHER columns in line fallback are loose; real AST columns are not.
                if wanted_type != 15:
                    continue
            span = max(1, int(stmt.get("end_line", 0)) - int(stmt.get("start_line", 0)) + 1)
            # Prefer matching type, line overlap, source order, deeper/narrower AST nodes.
            order_penalty = abs(i - max(prev_idx + 1, 0))
            score = (
                100.0 if type_match else 0.0,
                float(overlap) / float(span),
                int(stmt.get("depth", 0)),
                -order_penalty,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            # Last resort for line-fallback rows: take the next compatible source row.
            if not col_lines:
                for i, stmt in enumerate(filtered):
                    if i in used or i <= prev_idx:
                        continue
                    if wanted_type is not None and wanted_type != 15 and _stmt_type_id(stmt) != wanted_type:
                        continue
                    best_i = i
                    break
        if best_i is None:
            if source_line_column_like:
                return _line_column_rows(source_line_cols)
            return []
        used.add(best_i)
        prev_idx = best_i
        selected.append({**filtered[best_i], "id": col, "alignment_col": col})

    return selected


def _line_column_rows(source_line_cols: Sequence[int]) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "alignment_col": i,
            "type": "source_line",
            "start_line": int(ln),
            "end_line": int(ln),
            "start_col": 0,
            "end_col": -1,
            "depth": 0,
        }
        for i, ln in enumerate(source_line_cols)
    ]


def _source_statement_table(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild the source-statement columns used by alignment_matrix."""
    p = Path(str(record.get("source_file") or ""))
    if not p.is_file():
        return []
    lt = _line_to_text(p)
    bounds = _function_body_line_range(record, lt) if lt else None
    if bounds is None:
        return []
    parser = _get_c_parser()
    cache_key = str(p) if parser is not None else f"{p}:{bounds[0]}:{bounds[1]}:fallback"
    cached = _source_stmt_cache.get(cache_key)
    if cached is not None:
        statements_full = cached
    else:
        if parser is None:
            statements_full = _fallback_statement_table_from_source(
                _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt)),
                bounds,
            )
        try:
            if parser is not None:
                source = p.read_text(encoding="utf-8", errors="replace")
                tree = parser.parse(source.encode("utf-8"))
                statements_full: List[Dict[str, Any]] = []
                stack: List[Tuple[Any, int]] = [(tree.root_node, 0)]
                while stack:
                    node, depth = stack.pop()
                    ntype = _node_type(node)
                    if ntype in STATEMENT_TYPES:
                        start_line = int(node.start_point[0]) + 1
                        end_line = int(node.end_point[0]) + 1
                        statements_full.append({
                            "id": len(statements_full),
                            "type": ntype,
                            "start_line": start_line,
                            "end_line": end_line,
                            "start_col": int(node.start_point[1]),
                            "end_col": int(node.end_point[1]),
                            "depth": int(depth),
                        })
                    children = getattr(node, "children", None) or []
                    for child in reversed(children):
                        stack.append((child, depth + 1))
        except Exception:
            cache_key = f"{p}:{bounds[0]}:{bounds[1]}:fallback"
            statements_full = _fallback_statement_table_from_source(
                _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt)),
                bounds,
            )
        _source_stmt_cache[cache_key] = statements_full

    lo, hi = bounds
    filtered: List[Dict[str, Any]] = []
    for stmt in statements_full:
        if int(stmt["end_line"]) >= lo and int(stmt["start_line"]) <= hi:
            filtered.append({**stmt, "id": len(filtered)})
    if not filtered:
        return []
    M = record.get("alignment_matrix")
    n_cols = 0
    if isinstance(M, list) and M and isinstance(M[0], list):
        n_cols = len(M[0])
    if n_cols > 0:
        mapped = _source_statement_columns_from_alignment(record, filtered)
        if mapped:
            return mapped
        return []

    return [{**stmt, "id": i} for i, stmt in enumerate(filtered)]


def _source_all_statement_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all tree-sitter statement rows in the source function body.

    Unlike ``_source_statement_table``, this is not compressed to alignment
    columns. It is used to recover extra same-BB source statements such as local
    declarations that are present in block_src_lines but dropped from the
    compressed alignment columns.
    """
    p = Path(str(record.get("source_file") or ""))
    if not p.is_file():
        return []
    lt = _line_to_text(p)
    bounds = _function_body_line_range(record, lt) if lt else None
    if bounds is None:
        return []
    parser = _get_c_parser()
    cache_key = f"{p}:all"
    cached = _source_stmt_all_cache.get(cache_key)
    if cached is not None:
        statements_full = cached
    else:
        if parser is None:
            statements_full = _fallback_statement_table_from_source(
                _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt)),
                bounds,
            )
        else:
            try:
                _, _, tree = _source_tree_for_record(record)
                statements_full = []
                stack: List[Tuple[Any, int]] = [(tree.root_node, 0)] if tree is not None else []
                while stack:
                    node, depth = stack.pop()
                    ntype = _node_type(node)
                    if ntype in STATEMENT_TYPES:
                        statements_full.append({
                            "id": len(statements_full),
                            "type": ntype,
                            "start_line": int(node.start_point[0]) + 1,
                            "end_line": int(node.end_point[0]) + 1,
                            "start_col": int(node.start_point[1]),
                            "end_col": int(node.end_point[1]),
                            "depth": int(depth),
                        })
                    children = getattr(node, "children", None) or []
                    for child in reversed(children):
                        stack.append((child, depth + 1))
            except Exception:
                statements_full = _fallback_statement_table_from_source(
                    _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt)),
                    bounds,
                )
        _source_stmt_all_cache[cache_key] = statements_full
    lo, hi = bounds
    return [
        {**stmt, "id": i}
        for i, stmt in enumerate(statements_full)
        if int(stmt["end_line"]) >= lo and int(stmt["start_line"]) <= hi
    ]


def _same_bb_extra_statement_rows(record: Dict[str, Any], existing_rows: Sequence[Dict[str, Any]], bbs: Sequence[int]) -> List[Dict[str, Any]]:
    bb_lines = set(_line_numbers_for_bbs(record, bbs))
    if not bb_lines:
        return []
    existing_keys = {
        (
            str(r.get("type")),
            int(r.get("start_line", -1)),
            int(r.get("end_line", -1)),
            int(r.get("start_col", -1)),
            int(r.get("end_col", -1)),
        )
        for r in existing_rows
    }
    extras: List[Dict[str, Any]] = []
    for row in _source_all_statement_rows(record):
        key = (
            str(row.get("type")),
            int(row.get("start_line", -1)),
            int(row.get("end_line", -1)),
            int(row.get("start_col", -1)),
            int(row.get("end_col", -1)),
        )
        if key in existing_keys:
            continue
        if not (_row_line_set(row) & bb_lines):
            continue
        rtype = str(row.get("type") or "")
        if rtype in {"if_statement", "while_statement", "for_statement", "do_statement", "switch_statement", "compound_statement"}:
            continue
        extras.append(row)
    return sorted(extras, key=lambda r: (int(r.get("start_line", 0)), int(r.get("start_col", 0))))


def _strip_preprocessor_lines_from_lt(lt: Dict[int, str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k, line in lt.items():
        if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning)\b", str(line or "")):
            out[k] = ""
        else:
            out[k] = line
    return out


def _strip_block_comments_from_lt(lt: Dict[int, str]) -> Dict[int, str]:
    if not lt:
        return {}
    keys = sorted(lt)
    text = "\n".join(str(lt[k] or "") for k in keys)

    def repl(m: re.Match[str]) -> str:
        return "\n" * m.group(0).count("\n")

    cleaned = re.sub(r"/\*.*?\*/", repl, text, flags=re.S)
    lines = cleaned.splitlines()
    return {k: (lines[i] if i < len(lines) else "") for i, k in enumerate(keys)}


def _build_char_line_index(line_nums: List[int], lt: Dict[int, str]) -> Tuple[str, List[Tuple[int, int, int]]]:
    text_parts: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    pos = 0
    for ln in line_nums:
        line = str(lt.get(int(ln), "") or "")
        if text_parts:
            text_parts.append("\n")
            pos += 1
        start = pos
        text_parts.append(line)
        pos += len(line)
        spans.append((int(ln), start, pos))
    return "".join(text_parts), spans


def _line_for_char_pos(spans: List[Tuple[int, int, int]], pos: int) -> Optional[int]:
    for ln, start, end in spans:
        if start <= pos < end:
            return ln
    if spans and pos >= spans[-1][2]:
        return spans[-1][0]
    return None


def _balanced_param_end(text: str, open_paren: int) -> int:
    if open_paren < 0:
        return -1
    depth = 0
    in_s = in_c = esc = False
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
        elif ch == "'":
            in_c = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _balanced_brace_end(text: str, open_brace: int) -> int:
    if open_brace < 0:
        return -1
    depth = 0
    in_s = in_c = esc = False
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
    fn = str(record.get("func_name") or "").strip()
    if not fn or not lt:
        return None
    line_nums = sorted(lt)
    text, spans = _build_char_line_index(line_nums, lt)
    if not text:
        return None
    candidates: List[Tuple[int, int, int]] = []
    for m in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(fn)}(?![A-Za-z0-9_])\s*\(", text):
        open_paren = text.find("(", m.end() - 1)
        close_paren = _balanced_param_end(text, open_paren)
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
    start, end, _ = candidates[0]
    return start, end


def _split_semicolon_statements(text: str) -> List[str]:
    s = str(text or "")
    out: List[str] = []
    start = 0
    par = brk = brc = 0
    in_s = in_c = esc = False
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
        elif ch == "{":
            brc += 1
        elif ch == "}":
            brc -= 1
        elif ch == ";" and par <= 0 and brk <= 0 and brc <= 0:
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


def _line_numbers_for_bbs(record: Dict[str, Any], bbs: Sequence[int]) -> List[int]:
    block_src_lines = record.get("block_src_lines") or []
    out: List[int] = []
    if not isinstance(block_src_lines, list):
        return []
    for bb in bbs:
        if not isinstance(bb, (int, float)):
            continue
        bi = int(bb)
        if not (0 <= bi < len(block_src_lines)):
            continue
        group = block_src_lines[bi]
        if not isinstance(group, list):
            continue
        for ln in group:
            if isinstance(ln, (int, float)) or str(ln).isdigit():
                out.append(int(ln))
    return sorted(set(out))


def _source_lines_text(lt: Dict[int, str], line_nums: Sequence[int]) -> str:
    return "\n".join(str(lt.get(int(ln), "") or "") for ln in sorted(set(int(x) for x in line_nums)))


def _clamp_lines_to_bounds(
    line_nums: Sequence[int],
    bounds: Optional[Tuple[int, int]],
) -> List[int]:
    vals = sorted({int(x) for x in line_nums if isinstance(x, (int, float)) or str(x).isdigit()})
    if bounds is None:
        return vals
    lo, hi = bounds
    return [ln for ln in vals if int(lo) <= ln <= int(hi)]


def _source_text_for_statement_rows(lt: Dict[int, str], rows: Sequence[Dict[str, int]]) -> str:
    line_nums: List[int] = []
    for row in rows:
        try:
            lo = int(row["start_line"])
            hi = int(row["end_line"])
        except Exception:
            continue
        line_nums.extend(ln for ln in range(lo, hi + 1) if ln in lt)
    return _source_lines_text(lt, line_nums)


def _c_delimiter_balance(text: str) -> Tuple[int, int, int]:
    par = brk = brc = 0
    in_s = in_c = esc = False
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


def _statement_looks_complete(text: str) -> bool:
    x = str(text or "").strip()
    if not x:
        return True
    par, brk, _ = _c_delimiter_balance(x)
    if par > 0 or brk > 0:
        return False
    if x.endswith((";", "}", ":")):
        return True
    if _extract_source_condition(x):
        return True
    return False


def _extend_source_span_forward(
    line_nums: Sequence[int],
    lt: Dict[int, str],
    *,
    bounds: Optional[Tuple[int, int]],
    max_extra: int = 18,
) -> List[int]:
    lines = _clamp_lines_to_bounds(line_nums, bounds)
    if not lines:
        return []
    out = list(lines)
    max_line = max(lt) if lt else max(out)
    if bounds is not None:
        max_line = min(max_line, int(bounds[1]))
    cur = max(out)
    limit = max(out) + int(max_extra)
    while cur < max_line and cur < limit:
        raw = _source_lines_text(lt, out)
        if _statement_looks_complete(raw):
            break
        cur += 1
        if cur in lt:
            out.append(cur)
    return sorted(set(out))


def _control_span_end(text: str, match: re.Match[str]) -> int:
    start = text.find("(", match.end() - 1)
    if start < 0:
        return -1
    return _balanced_param_end(text, start) + 1


def _span_overlaps_wanted_lines(
    spans: List[Tuple[int, int, int]],
    wanted: Set[int],
    start: int,
    end: int,
) -> bool:
    if end <= start:
        return False
    for ln, a, b in spans:
        if ln in wanted and not (b <= start or a >= end):
            return True
    return False


def _first_statement_fragment(text: str) -> str:
    stmts = _split_semicolon_statements(text)
    if stmts:
        return stmts[0]
    return str(text or "").strip()


def _clean_statement_candidate_fragment(text: str) -> str:
    frag = str(text or "").strip()
    if not frag:
        return ""
    body = _extract_simple_control_body_statement(frag)
    if body:
        return body
    last_brace = frag.rfind("{")
    if last_brace >= 0:
        tail = frag[last_brace + 1 :].strip()
        body = _extract_simple_control_body_statement(tail)
        if body:
            return body
        stmts = _split_semicolon_statements(tail)
        if stmts:
            return stmts[0]
    return frag


def _source_statement_candidates(
    line_nums: Sequence[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]],
) -> List[Tuple[str, int, int]]:
    """Return source statement fragments touching the BB source lines.

    The alignment often marks a continuation line or one statement inside a
    source window. This searches a local source window but only prefers
    statements whose character span intersects the BB-derived line set.
    """
    wanted = set(_clamp_lines_to_bounds(line_nums, bounds))
    if not wanted:
        return []
    lo = max(min(lt) if lt else 1, min(wanted) - 8)
    hi = min(max(lt) if lt else max(wanted), max(wanted) + 18)
    if bounds is not None:
        lo = max(lo, int(bounds[0]))
        hi = min(hi, int(bounds[1]))
    window_lines = [ln for ln in range(lo, hi + 1) if ln in lt]
    text, spans = _build_char_line_index(window_lines, lt)
    candidates: List[Tuple[str, int, int]] = []
    start = 0
    par = brk = 0
    in_s = in_c = esc = False
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
                if frag:
                    if _span_overlaps_wanted_lines(spans, wanted, start, i + 1):
                        dist = 0
                    else:
                        frag_ln = _line_for_char_pos(spans, start) or lo
                        dist = min(abs(frag_ln - ln) for ln in wanted)
                    candidates.append((frag, dist, start))
            start = i + 1
    return candidates


def _control_spans_near_lines(
    line_nums: Sequence[int],
    lt: Dict[int, str],
    bounds: Optional[Tuple[int, int]],
    *,
    lookback: int = 18,
    lookahead: int = 12,
) -> List[Tuple[str, int, int, int, str]]:
    wanted = set(_clamp_lines_to_bounds(line_nums, bounds))
    if not wanted:
        return []
    lo = max(min(lt) if lt else 1, min(wanted) - int(lookback))
    hi = min(max(lt) if lt else max(wanted), max(wanted) + int(lookahead))
    if bounds is not None:
        lo = max(lo, int(bounds[0]))
        hi = min(hi, int(bounds[1]))
    window_lines = [ln for ln in range(lo, hi + 1) if ln in lt]
    text, spans = _build_char_line_index(window_lines, lt)
    out: List[Tuple[str, int, int, int, str]] = []
    for m in re.finditer(r"\b(if|while|for|switch)\s*\(", text):
        kw = m.group(1)
        end = _control_span_end(text, m)
        if end <= m.start():
            continue
        body_span = _statement_span_from_pos(text, end, len(text))
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
        out.append((kw, m.start(), end, dist, text))
    return out


def _candidate_alignment_columns(
    record: Dict[str, Any],
    bbs: Sequence[int],
    n_stmt: int,
    *,
    rel_threshold: float = 0.10,
) -> Tuple[List[int], List[float]]:
    M = record.get("alignment_matrix")
    if not isinstance(M, list) or not bbs or n_stmt <= 0:
        return [], []
    for row in M:
        if not isinstance(row, list) or len(row) != n_stmt:
            return [], []
    masses = [0.0] * n_stmt
    for bb in bbs:
        if not isinstance(bb, (int, float)):
            continue
        bi = int(bb)
        if not (0 <= bi < len(M)) or not isinstance(M[bi], list):
            continue
        row = M[bi]
        for j in range(n_stmt):
            try:
                masses[j] += float(row[j])
            except Exception:
                pass
    max_mass = max(masses) if masses else 0.0
    if max_mass <= 0:
        return [], masses
    rel = float(rel_threshold) * max_mass
    cols = [j for j, mass in enumerate(masses) if mass > 1e-8 and mass >= rel]
    if not cols:
        cols = [int(max(range(len(masses)), key=lambda j: masses[j]))]
    return sorted(cols), masses


def _source_line_columns_target_for_slot(
    record: Dict[str, Any],
    slot_type: str,
    bbs: Sequence[int],
    expected_control_types: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    """Fallback for matrix columns that correspond directly to source lines."""
    M = record.get("alignment_matrix")
    source_lines = record.get("source_lines")
    if not isinstance(M, list) or not M or not isinstance(M[0], list):
        return "", ""
    if not isinstance(source_lines, list) or len(source_lines) != len(M[0]):
        return "", ""
    line_cols: List[int] = []
    for x in source_lines:
        if isinstance(x, (int, float)) or str(x).isdigit():
            line_cols.append(int(x))
        else:
            return "", ""
    candidates, masses = _candidate_alignment_columns(record, bbs, len(line_cols))
    if not candidates:
        return "", ""
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return "", ""
    lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    bounds = _function_body_line_range(record, lt_clean)
    lines = [line_cols[j] for j in candidates if 0 <= int(j) < len(line_cols)]
    if bounds is not None:
        lines = _clamp_lines_to_bounds(lines, bounds)
    if not lines:
        return "", ""
    if slot_type == "FILL_COND":
        cond = _condition_from_exact_source_lines(
            lt_clean,
            lines,
            bounds=bounds,
            expected_control_types=expected_control_types,
        )
        if cond and _target_is_usable(slot_type, cond):
            return cond, "source_alignment_matrix:line_condition"
        return "", ""
    if slot_type == "FILL_EXPR":
        expr = _source_expression_from_line_window(lt_clean, lines, bounds=bounds)
        if expr and _target_is_usable(slot_type, expr):
            return _clean_target_for_output(slot_type, expr), "source_alignment_matrix:line_expr"
        return "", ""
    if slot_type not in {"FILL_STMT", "FILL_EXPR"}:
        return "", ""
    target = _source_statements_from_line_window(
        lt_clean,
        lines,
        bounds=bounds,
        prefer_non_control_lines=True,
    )
    if target and _target_is_usable(slot_type, target):
        decl_rows = _needed_local_declaration_rows_from_target(record, min(lines), target)
        if decl_rows:
            decl_texts = _statement_texts_from_rows(record, decl_rows, bb_lines=set())
            if decl_texts:
                target = " ".join(decl_texts + [target])
        return _clean_target_for_output(slot_type, target), "source_alignment_matrix:line_stmt"
    return "", ""


def _field_child(node: Any, *names: str) -> Any:
    getter = getattr(node, "child_by_field_name", None)
    if getter is None:
        return None
    for name in names:
        try:
            child = getter(name)
        except Exception:
            child = None
        if child is not None:
            return child
    return None


def _node_text(source: Any, node: Any) -> str:
    try:
        start = int(node.start_byte)
        end = int(node.end_byte)
        if isinstance(source, (bytes, bytearray)):
            return bytes(source[start:end]).decode("utf-8", errors="replace")
        return str(source)[start:end]
    except Exception:
        return ""


def _source_tree_for_record(record: Dict[str, Any]) -> Tuple[str, bytes, Any]:
    p = Path(str(record.get("source_file") or ""))
    if not p.is_file():
        return "", b"", None
    parser = _get_c_parser()
    if parser is None:
        return "", b"", None
    key = str(p)
    cached = _source_tree_cache.get(key)
    if cached is not None:
        return cached
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
        source_b = source.encode("utf-8")
        tree = parser.parse(source_b)
    except Exception:
        return "", b"", None
    _source_tree_cache[key] = (source, source_b, tree)
    return source, source_b, tree


def _node_line_set(node: Any) -> Set[int]:
    try:
        lo = int(node.start_point[0]) + 1
        hi = int(node.end_point[0]) + 1
    except Exception:
        return set()
    return set(range(lo, hi + 1))


def _find_tree_sitter_node_for_row(record: Dict[str, Any], row: Dict[str, Any]) -> Any:
    _, _, tree = _source_tree_for_record(record)
    if tree is None:
        return None
    want_type = str(row.get("type") or "")
    try:
        want_start = int(row.get("start_line", 0)) - 1
        want_end = int(row.get("end_line", 0)) - 1
        want_start_col = int(row.get("start_col", -1))
        want_end_col = int(row.get("end_col", -1))
    except Exception:
        return None
    best = None
    best_score: Optional[Tuple[int, int, int]] = None
    stack: List[Any] = [tree.root_node]
    while stack:
        node = stack.pop()
        ntype = _node_type(node)
        if ntype == want_type:
            start = int(node.start_point[0])
            end = int(node.end_point[0])
            if start == want_start and end == want_end:
                sc = int(node.start_point[1])
                ec = int(node.end_point[1])
                score = (
                    1 if sc == want_start_col else 0,
                    1 if ec == want_end_col else 0,
                    -(ec - sc),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = node
        children = getattr(node, "children", None) or []
        for child in reversed(children):
            stack.append(child)
    return best


def _direct_statement_children(node: Any) -> List[Any]:
    children = getattr(node, "children", None) or []
    out: List[Any] = []
    for child in children:
        if _node_type(child) in STATEMENT_TYPES:
            out.append(child)
    return out


def _branch_statement_nodes(branch: Any, bb_lines: Set[int], *, allow_all_when_unlocated: bool) -> List[Any]:
    btype = _node_type(branch)
    if btype == "compound_statement":
        children = _direct_statement_children(branch)
        if bb_lines:
            overlapping = [c for c in children if _node_line_set(c) & bb_lines]
            if overlapping:
                return overlapping
            if not allow_all_when_unlocated:
                return []
        return children
    if btype not in STATEMENT_TYPES:
        children = _direct_statement_children(branch)
        if children:
            if bb_lines:
                overlapping = [c for c in children if _node_line_set(c) & bb_lines]
                if overlapping:
                    return overlapping
                if not allow_all_when_unlocated:
                    return []
            return children
    return [branch]


def _body_nodes_for_control_row(record: Dict[str, Any], row: Dict[str, Any], bb_lines: Set[int]) -> List[Any]:
    node = _find_tree_sitter_node_for_row(record, row)
    if node is None:
        return []
    ntype = _node_type(node)
    if ntype == "if_statement":
        consequence = _field_child(node, "consequence")
        alternative = _field_child(node, "alternative")
        branches = [x for x in (consequence, alternative) if x is not None]
        has_alternative = alternative is not None
    elif ntype in {"while_statement", "for_statement", "do_statement", "switch_statement"}:
        branches = [x for x in (_field_child(node, "body"),) if x is not None]
        has_alternative = False
    else:
        return []
    if not branches:
        return []
    if bb_lines:
        overlapping = [
            b for b in branches
            if _node_line_set(b) & bb_lines
        ]
        if overlapping:
            branches = overlapping
        else:
            return []
    else:
        return []
    out: List[Any] = []
    for branch in branches:
        out.extend(
            _branch_statement_nodes(
                branch,
                bb_lines,
                allow_all_when_unlocated=False,
            )
        )
    return out


def _condition_from_control_row(record: Dict[str, Any], row: Dict[str, Any]) -> str:
    node = _find_tree_sitter_node_for_row(record, row)
    if node is not None:
        cond = _field_child(node, "condition")
        if cond is not None:
            _, source_b, _ = _source_tree_for_record(record)
            raw = _node_text(source_b, cond)
            raw = raw.strip()
            if raw.startswith("(") and raw.endswith(")"):
                raw = raw[1:-1].strip()
            if _node_type(node) == "for_statement":
                parts = _split_top_level_semicolon_parts(raw)
                if len(parts) >= 2:
                    raw = parts[1].strip() or "1"
            cond_text = _pretty_c_fragment(raw, "FILL_COND")
            if _target_is_usable("FILL_COND", cond_text):
                return cond_text
    raw = _statement_text_from_lt(_line_to_text(record.get("source_file")), row)
    return _extract_source_condition(raw)


def _statement_texts_from_rows(
    record: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    bb_lines: Set[int],
) -> List[str]:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return []
    lt = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    out: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        rtype = str(row.get("type") or "")
        raw_texts: List[str] = []
        if rtype in {"if_statement", "while_statement", "for_statement", "do_statement", "switch_statement"}:
            body_nodes = _body_nodes_for_control_row(record, row, bb_lines)
            if body_nodes:
                _, source_b, _ = _source_tree_for_record(record)
                raw_texts = [_node_text(source_b, node) for node in body_nodes]
            else:
                raw = _statement_text_from_lt(lt, row)
                if rtype == "do_statement" and (_row_line_set(row) & bb_lines):
                    raw_texts = [raw]
                else:
                    body_stmt = _extract_simple_control_body_statement(raw)
                    if body_stmt:
                        raw_texts = [body_stmt]
        elif rtype == "source_line":
            raw_texts = [_source_lines_text(lt, sorted(_row_line_set(row)))]
            body_stmt = _extract_simple_control_body_statement(raw_texts[0])
            if body_stmt:
                raw_texts = [body_stmt]
        else:
            raw_texts = [_statement_text_from_lt(lt, row)]
        for raw in raw_texts:
            if not raw:
                continue
            if re.match(r"^\s*do\b", _normalise_source_text(raw)):
                stmts = [_pretty_c_fragment(raw, "FILL_STMT")]
            else:
                body_stmt = _extract_simple_control_body_statement(raw)
                if body_stmt:
                    raw_texts = [body_stmt]
                    raw = body_stmt
                stmts = [_pretty_c_fragment(frag, "FILL_STMT") for frag in _split_semicolon_statements(raw)]
            for stmt in stmts:
                if not stmt or not _target_is_usable("FILL_STMT", stmt):
                    continue
                key = re.sub(r"\s+", " ", stmt).strip()
                if key in seen:
                    continue
                seen.add(key)
                out.append(stmt)
    return out


def _slot_context_is_inside_control(skeleton_src: str, match_start: int) -> bool:
    prefix = str(skeleton_src or "")[: int(match_start)]
    tail = prefix[-240:]
    return bool(re.search(r"\b(?:if|while|for|switch)\s*\([^()]*$", tail))


def _strip_redundant_control_from_stmt_target(target: str) -> str:
    body = _extract_simple_control_body_statement(target)
    if not body:
        return target
    stmts: List[str] = []
    seen: Set[str] = set()
    for frag in _split_semicolon_statements(body):
        stmt = _pretty_c_fragment(frag, "FILL_STMT")
        if not stmt or not _target_is_usable("FILL_STMT", stmt):
            continue
        key = re.sub(r"\s+", " ", stmt).strip()
        if key in seen:
            continue
        seen.add(key)
        stmts.append(stmt)
    return " ".join(stmts) if stmts else target


def _identifier_tokens(text: str) -> Set[str]:
    protected, _ = _protect_c_literals(text)
    keywords = {
        "if", "else", "while", "for", "return", "switch", "case", "default",
        "break", "continue", "goto", "sizeof", "static", "extern", "const",
        "volatile", "restrict", "unsigned", "signed", "struct", "enum", "union",
        "void", "char", "short", "int", "long", "float", "double", "bool",
        "true", "false", "NULL", "nullptr",
    }
    return {
        x for x in re.findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])", protected)
        if x not in keywords
    }


def _declared_names_in_row(record: Dict[str, Any], row: Dict[str, Any]) -> Set[str]:
    lt = _line_to_text(record.get("source_file"))
    raw = _statement_text_from_lt(lt, row)
    s = _pretty_c_fragment(raw, "FILL_STMT")
    if not s:
        return set()
    if str(row.get("type") or "") != "declaration" and not _looks_like_local_declaration(s):
        return set()
    # Good enough for local declarations used by these slots, including
    # arrays/pointers and simple comma declarations.
    before_init = s.rstrip(";")
    parts = [p.strip() for p in before_init.split(",")]
    names: Set[str] = set()
    for part in parts:
        left = part.split("=", 1)[0].strip()
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$", left)
        if m:
            names.add(m.group(1))
    return names


def _looks_like_local_declaration(stmt: str) -> bool:
    s = _pretty_c_fragment(stmt, "FILL_STMT").strip()
    if not s.endswith(";"):
        return False
    if re.match(r"^(?:return|if|while|for|switch|break|continue|goto)\b", s):
        return False
    typeish = (
        r"(?:static|extern|register|auto|const|volatile|restrict|unsigned|signed|"
        r"short|long|int|char|bool|_Bool|float|double|size_t|ssize_t|idx_t|"
        r"uintmax_t|uint64_t|uint32_t|FILE|uid_t|gid_t|pid_t|mode_t|"
        r"struct\s+[A-Za-z_][A-Za-z0-9_]*|enum\s+[A-Za-z_][A-Za-z0-9_]*|"
        r"union\s+[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_t)"
    )
    return bool(re.match(rf"^\s*(?:{typeish})(?:\s+|\s*\*)+", s))


def _needed_local_declaration_rows(record: Dict[str, Any], rows: Sequence[Dict[str, Any]], target_text: str) -> List[Dict[str, Any]]:
    used = _identifier_tokens(target_text)
    if not used:
        return []
    row_keys = {
        (
            str(r.get("type")),
            int(r.get("start_line", -1)),
            int(r.get("end_line", -1)),
            int(r.get("start_col", -1)),
            int(r.get("end_col", -1)),
        )
        for r in rows
    }
    first_line = min((int(r.get("start_line", 10**9)) for r in rows), default=10**9)
    out: List[Dict[str, Any]] = []
    for row in _source_all_statement_rows(record):
        key = (
            str(row.get("type")),
            int(row.get("start_line", -1)),
            int(row.get("end_line", -1)),
            int(row.get("start_col", -1)),
            int(row.get("end_col", -1)),
        )
        if key in row_keys:
            continue
        if int(row.get("start_line", 10**9)) > first_line:
            continue
        names = _declared_names_in_row(record, row)
        if names & used:
            out.append(row)
    return sorted(out, key=lambda r: (int(r.get("start_line", 0)), int(r.get("start_col", 0))))


def _function_body_statement_target(record: Dict[str, Any], lt: Dict[int, str]) -> str:
    bounds = _function_body_line_range(record, lt)
    if bounds is None:
        return ""
    lo, hi = bounds
    body_lines = [ln for ln in range(lo + 1, hi) if ln in lt]
    raw = _source_lines_text(lt, body_lines)
    if re.search(r"\b(?:if|while|for|switch)\s*\(", _normalise_source_text(raw)):
        # A single statement slot cannot faithfully represent nested control flow.
        return ""
    stmts = [_pretty_c_fragment(s, "FILL_STMT") for s in _split_semicolon_statements(raw)]
    stmts = [s for s in stmts if s and not s.startswith(("else ", "case "))]
    return " ".join(stmts)


def _function_body_is_single_return(record: Dict[str, Any]) -> bool:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return False
    lt = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    target = _function_body_statement_target(record, lt)
    if not target:
        return False
    stmts = [
        _pretty_c_fragment(s, "FILL_STMT")
        for s in _split_semicolon_statements(target)
    ]
    stmts = [s for s in stmts if s]
    return len(stmts) == 1 and _normalise_source_text(stmts[0]).startswith("return ")


def _extract_source_condition(text: str, *, prefer_last: bool = False) -> str:
    raw = _normalise_source_text(text)
    found: List[str] = []
    for kw in ("if", "while", "for", "switch"):
        for m in re.finditer(rf"\b{kw}\s*\(", raw):
            start = raw.find("(", m.end() - 1)
            end = _balanced_param_end(raw, start)
            if end < 0:
                continue
            cond = raw[start + 1 : end].strip()
            if kw == "for":
                parts = _split_top_level_semicolon_parts(cond)
                if len(parts) >= 2:
                    cond = parts[1].strip() or "1"
            cond = _pretty_c_fragment(cond, "FILL_COND")
            if cond:
                found.append(cond)
    return found[-1] if prefer_last and found else (found[0] if found else "")


def _extract_simple_control_body_statement(text: str, *, _depth: int = 0) -> str:
    raw = _normalise_source_text(text)
    if _depth > 3:
        return ""
    raw = re.sub(r"^\s*else\s+", "", raw, count=1)
    m = re.match(r"\s*(?:if|while|for)\s*\(", raw)
    if not m:
        return ""
    open_paren = raw.find("(", m.end() - 1)
    close_paren = _balanced_param_end(raw, open_paren)
    if close_paren < 0:
        return ""
    body_span = _statement_span_from_pos(raw, close_paren + 1, len(raw))
    if body_span is None:
        return ""
    start, end = body_span
    body = raw[start:end].strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1].strip()
    nested = _extract_simple_control_body_statement(body, _depth=_depth + 1)
    if nested:
        return nested
    out: List[str] = []
    seen: Set[str] = set()
    for frag in _split_semicolon_statements(body):
        stmt = _pretty_c_fragment(frag, "FILL_STMT")
        if not stmt or not _target_is_usable("FILL_STMT", stmt):
            continue
        key = re.sub(r"\s+", " ", stmt).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(stmt)
    return " ".join(out)


def _source_statements_from_line_window(
    lt: Dict[int, str],
    line_nums: Sequence[int],
    *,
    bounds: Optional[Tuple[int, int]],
    prefer_non_control_lines: bool = True,
) -> str:
    lines = [
        int(x) for x in line_nums
        if isinstance(x, (int, float)) or str(x).isdigit()
    ]
    if bounds is not None:
        lo_b, hi_b = bounds
        lines = [ln for ln in lines if lo_b <= ln <= hi_b]
    if not lines:
        return ""

    cands = _source_statement_candidates(lines, lt, bounds)
    if cands:
        ranked = sorted(cands, key=lambda x: (x[1], len(_identifier_tokens(x[0])), x[2]))
        best = ranked[0][0]
        if re.match(r"\s*(?:if|while|for|switch)\s*\(", _pretty_c_fragment(best, "FILL_STMT")):
            body = _extract_simple_control_body_statement(best)
            if body:
                best = body
        out: List[str] = []
        seen: Set[str] = set()
        for frag in _split_semicolon_statements(best):
            stmt = _pretty_c_fragment(frag, "FILL_STMT")
            if not stmt or not _target_is_usable("FILL_STMT", stmt):
                continue
            key = re.sub(r"\s+", " ", stmt).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(stmt)
        if out:
            return " ".join(out)

    chosen = list(lines)
    if prefer_non_control_lines:
        body_lines = [
            ln for ln in lines
            if ln in lt and not _line_looks_like_control_only(lt.get(ln, ""))
        ]
        if body_lines:
            chosen = body_lines

    raw = _source_lines_text(lt, chosen)
    out: List[str] = []
    seen: Set[str] = set()
    body_stmt = _extract_simple_control_body_statement(raw)
    candidates = [body_stmt] if body_stmt else [
        _pretty_c_fragment(s, "FILL_STMT") for s in _split_semicolon_statements(raw)
    ]
    for cand in candidates:
        for frag in _split_semicolon_statements(cand):
            stmt = _pretty_c_fragment(frag, "FILL_STMT")
            if not stmt or not _target_is_usable("FILL_STMT", stmt):
                continue
            key = re.sub(r"\s+", " ", stmt).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(stmt)
    return " ".join(out)


def _source_expression_from_text(text: str) -> str:
    x = _pretty_c_fragment(text, "FILL_STMT").strip()
    if not x:
        return ""
    cond = _extract_source_condition(x, prefer_last=True)
    if cond and _target_is_usable("FILL_EXPR", cond):
        return cond
    if x.startswith("return "):
        expr = x[len("return ") :].rstrip(";").strip()
        expr = _pretty_c_fragment(expr, "FILL_EXPR")
        if _target_is_usable("FILL_EXPR", expr):
            return expr
    if x.endswith(";"):
        y = x.rstrip(";").strip()
        # For assignment/update statement slots that were typed as FILL_EXPR,
        # use the whole expression without the statement semicolon.
        if _target_is_usable("FILL_EXPR", y):
            return y
        if "=" in y:
            rhs = y.split("=", 1)[1].strip()
            rhs = _pretty_c_fragment(rhs, "FILL_EXPR")
            if _target_is_usable("FILL_EXPR", rhs):
                return rhs
    return ""


def _source_expression_from_line_window(
    lt: Dict[int, str],
    line_nums: Sequence[int],
    *,
    bounds: Optional[Tuple[int, int]],
) -> str:
    lines = _clamp_lines_to_bounds(line_nums, bounds)
    if not lines:
        return ""
    cands = _source_statement_candidates(lines, lt, bounds)
    for raw, _dist, _start in sorted(cands, key=lambda x: (x[1], x[2])):
        expr = _source_expression_from_text(raw)
        if expr:
            return expr
    raw = _source_lines_text(lt, _extend_source_span_forward(lines, lt, bounds=bounds, max_extra=12))
    return _source_expression_from_text(raw)


def _split_top_level_semicolon_parts(text: str) -> List[str]:
    s = str(text or "")
    parts: List[str] = []
    start = 0
    par = brk = 0
    in_s = in_c = esc = False
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


def _statement_from_source_start_line(
    lt: Dict[int, str],
    start_line: int,
    *,
    bounds: Optional[Tuple[int, int]],
    slot_type: str,
    max_extra: int = 18,
) -> str:
    if not lt or int(start_line) not in lt:
        return ""
    lo = int(start_line)
    hi = min(max(lt), lo + int(max_extra))
    if bounds is not None:
        lo = max(lo, int(bounds[0]))
        hi = min(hi, int(bounds[1]))
    if lo > hi:
        return ""
    raw = _source_lines_text(lt, [ln for ln in range(lo, hi + 1) if ln in lt])
    stmts = _split_semicolon_statements(raw)
    if not stmts:
        return ""
    stmt = _pretty_c_fragment(stmts[0], "FILL_STMT")
    if slot_type in {"FILL_STMT", "FILL_EXPR"}:
        if not _target_is_usable(slot_type, stmt):
            return ""
    return stmt


def _source_alignment_target_for_slot(
    record: Dict[str, Any],
    slot_type: str,
    bbs: Sequence[int],
    expected_control_types: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    """Use BB→source-statement alignment_matrix as the primary GT source."""
    if slot_type not in BODY_SLOT_TYPES:
        return "", ""
    M = record.get("alignment_matrix")
    if not isinstance(M, list) or not bbs:
        return "", ""
    stmt_rows = _source_statement_table(record)
    if not stmt_rows:
        fallback, reason = _source_line_columns_target_for_slot(record, slot_type, bbs, expected_control_types=expected_control_types)
        if fallback:
            return fallback, reason
        fallback, reason = _direct_source_lines_target_for_slot(record, slot_type, bbs, expected_control_types=expected_control_types)
        if fallback:
            return fallback, reason
        return "", "no_source_statement_columns"

    n_stmt = len(stmt_rows)
    if n_stmt <= 0:
        return "", ""
    candidates, masses = _candidate_alignment_columns(record, bbs, n_stmt)
    if not candidates:
        fallback, reason = _source_line_columns_target_for_slot(record, slot_type, bbs, expected_control_types=expected_control_types)
        if fallback:
            return fallback, reason
        fallback, reason = _direct_source_lines_target_for_slot(record, slot_type, bbs, expected_control_types=expected_control_types)
        if fallback:
            return fallback, reason
        return "", "alignment_row_has_zero_mass"
    if len(candidates) > 12:
        candidates = sorted(candidates, key=lambda j: masses[j], reverse=True)[:12]

    row_candidates: List[Dict[str, Any]] = [
        stmt_rows[j] for j in sorted(candidates)
        if 0 <= j < len(stmt_rows)
    ]
    if not row_candidates:
        return "", "alignment_columns_out_of_range"
    bb_lines = set(_line_numbers_for_bbs(record, bbs))

    if slot_type == "FILL_COND":
        # Header conditions are more reliable from the BB source-line window
        # than from compressed alignment columns: loop/else-if lowering often
        # maps a header BB to a nearby nested if column.
        fallback, reason = _direct_source_lines_target_for_slot(
            record,
            slot_type,
            bbs,
            expected_control_types=expected_control_types,
        )
        if fallback:
            return fallback, reason
        allowed = set(expected_control_types or {"if_statement", "while_statement", "for_statement", "switch_statement"})
        control_rows = [
            row for row in row_candidates
            if str(row.get("type")) in allowed
        ]
        for row in sorted(control_rows, key=lambda r: (int(r.get("start_line", 0)), int(r.get("end_line", 0)))):
            cond = _condition_from_control_row(record, row)
            if cond and _target_is_usable(slot_type, cond):
                return cond, "source_alignment_matrix:condition"
        if expected_control_types:
            # Last chance: fall back to the untyped search only after the typed
            # source projection failed. This preserves recall without letting a
            # loop header prefer an inner if condition.
            fallback, reason = _direct_source_lines_target_for_slot(
                record,
                slot_type,
                bbs,
                expected_control_types=expected_control_types,
            )
            if fallback:
                return fallback, reason + ":typed_fallback"
        return "", "no_control_condition_for_aligned_columns"

    if slot_type == "FILL_EXPR":
        lt = _line_to_text(record.get("source_file"))
        lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt)) if lt else {}
        raw = _source_text_for_statement_rows(lt_clean, row_candidates) if lt_clean else ""
        expr = _source_expression_from_text(raw)
        if expr and _target_is_usable(slot_type, expr):
            return _clean_target_for_output(slot_type, expr), "source_alignment_matrix:expr"
        fallback, reason = _direct_source_lines_target_for_slot(record, slot_type, bbs)
        if fallback:
            return fallback, reason
        return "", "no_expression_for_aligned_columns"

    extra_rows = _same_bb_extra_statement_rows(record, row_candidates, bbs)
    if extra_rows:
        row_candidates = sorted(
            list(row_candidates) + extra_rows,
            key=lambda r: (
                int(r.get("start_line", 0)),
                int(r.get("start_col", 0)),
                int(r.get("end_line", 0)),
            ),
        )

    rows_for_text = sorted(
        row_candidates,
        key=lambda r: (int(r.get("start_line", 0)), int(r.get("end_line", 0))),
    )
    out = _statement_texts_from_rows(
        record,
        rows_for_text,
        bb_lines=bb_lines,
    )
    if not out:
        fallback, reason = _direct_source_lines_target_for_slot(record, slot_type, bbs)
        if fallback:
            return fallback, reason
        return "", "no_statement_body_for_aligned_columns"
    decl_rows = _needed_local_declaration_rows(record, rows_for_text, " ".join(out))
    if decl_rows:
        rows_for_text = sorted(
            list(decl_rows) + rows_for_text,
            key=lambda r: (int(r.get("start_line", 0)), int(r.get("start_col", 0))),
        )
        out = _statement_texts_from_rows(record, rows_for_text, bb_lines=bb_lines)
        if not out:
            return "", "no_statement_body_for_aligned_columns"
    return _clean_target_for_output(slot_type, " ".join(out)), "source_alignment_matrix:stmt"


def _source_condition_candidates_for_slot(
    record: Dict[str, Any],
    bbs: Sequence[int],
    expected_control_types: Optional[Set[str]] = None,
) -> List[Tuple[str, str]]:
    """Return plausible source conditions for a regenerated condition slot."""
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    def add(cond: str, reason: str) -> None:
        clean = _pretty_c_fragment(cond, "FILL_COND")
        if not clean or not _target_is_usable("FILL_COND", clean):
            return
        key = _normalise_condition_for_compare(clean)
        if not key or key in seen:
            return
        seen.add(key)
        out.append((clean, reason))

    primary, reason = _source_alignment_target_for_slot(
        record,
        "FILL_COND",
        bbs,
        expected_control_types=expected_control_types,
    )
    if primary:
        add(primary, reason or "source_alignment_matrix:condition")

    lt = _line_to_text(record.get("source_file"))
    lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt)) if lt else {}
    bounds = _function_body_line_range(record, lt_clean) if lt_clean else None
    bb_lines = set(_clamp_lines_to_bounds(_line_numbers_for_bbs(record, bbs), bounds))

    stmt_rows = _source_statement_table(record)
    if stmt_rows:
        candidates, masses = _candidate_alignment_columns(record, bbs, len(stmt_rows))
        row_candidates = [
            stmt_rows[j] for j in candidates
            if 0 <= int(j) < len(stmt_rows)
        ]
        row_candidates = sorted(
            row_candidates,
            key=lambda r: (
                -float(masses[int(r.get("alignment_col", r.get("id", 0)))] if masses else 0.0)
                if int(r.get("alignment_col", r.get("id", 0))) < len(masses)
                else 0.0,
                int(r.get("start_line", 0)),
                int(r.get("start_col", 0)),
            ),
        )
        allowed = set(expected_control_types or {"if_statement", "while_statement", "for_statement", "switch_statement"})
        for row in row_candidates:
            if str(row.get("type") or "") in allowed:
                add(_condition_from_control_row(record, row), "source_alignment_matrix:condition_candidate")

    if bb_lines:
        allowed = set(expected_control_types or {"if_statement", "while_statement", "for_statement", "switch_statement"})
        rows = [
            row for row in _source_all_statement_rows(record)
            if str(row.get("type") or "") in allowed
            and (_row_line_set(row) & bb_lines)
        ]
        for row in sorted(rows, key=lambda r: (int(r.get("start_line", 0)), int(r.get("start_col", 0)))):
            add(_condition_from_control_row(record, row), "source_block_lines:condition_candidate")

    if lt_clean and bb_lines:
        spans = _control_spans_near_lines(sorted(bb_lines), lt_clean, bounds=bounds, lookback=8, lookahead=12)
        expected_keywords = set()
        for typ in expected_control_types or {"if_statement", "while_statement", "for_statement", "switch_statement"}:
            if typ == "if_statement":
                expected_keywords.add("if")
            elif typ == "while_statement":
                expected_keywords.add("while")
            elif typ == "for_statement":
                expected_keywords.add("for")
            elif typ == "switch_statement":
                expected_keywords.add("switch")
            elif typ == "do_statement":
                expected_keywords.add("while")
        for kw, _start, end, _dist, text in sorted(spans, key=lambda x: (x[3], x[1])):
            if kw not in expected_keywords:
                continue
            open_paren = text.rfind("(", 0, end)
            close_paren = end - 1
            if open_paren < 0 or close_paren <= open_paren:
                continue
            cond = text[open_paren + 1:close_paren].strip()
            if kw == "for":
                parts = _split_top_level_semicolon_parts(cond)
                if len(parts) >= 2:
                    cond = parts[1].strip() or "1"
            add(cond, "source_block_lines:nearby_condition_candidate")

    return out


def _condition_budget_replacement(
    record: Dict[str, Any],
    bbs: Sequence[int],
    source_cond_budget: Counter[str],
    used_cond_budget: Counter[str],
    current_target: str,
    expected_control_types: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    key = _normalise_condition_for_compare(current_target)
    if not key:
        return "", ""
    allowed = int(source_cond_budget.get(key, 0))
    if allowed <= 0 or int(used_cond_budget.get(key, 0)) < allowed:
        return "", ""
    for cand, reason in _source_condition_candidates_for_slot(
        record,
        bbs,
        expected_control_types=expected_control_types,
    ):
        cand_key = _normalise_condition_for_compare(cand)
        if not cand_key or cand_key == key:
            continue
        cand_allowed = int(source_cond_budget.get(cand_key, 0))
        if cand_allowed <= 0:
            continue
        if int(used_cond_budget.get(cand_key, 0)) >= cand_allowed:
            continue
        return cand, reason + ":condition_budget_repair"
    return "", ""


def _direct_source_lines_target_for_slot(
    record: Dict[str, Any],
    slot_type: str,
    bbs: Sequence[int],
    expected_control_types: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    """Last source-derived fallback: project directly from block_src_lines.

    This is still source GT. It is less precise than alignment columns, but it
    recovers samples where the compressed statement table cannot be rebuilt.
    """
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return "", ""
    lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    line_nums = _line_numbers_for_bbs(record, bbs)
    bounds = _function_body_line_range(record, lt_clean)
    if bounds is not None:
        lo, hi = bounds
        line_nums = [ln for ln in line_nums if lo <= ln <= hi]
    if not line_nums:
        return "", ""

    if slot_type == "FILL_COND":
        cond = _condition_from_exact_source_lines(
            lt_clean,
            line_nums,
            bounds=bounds,
            expected_control_types=expected_control_types,
        )
        if cond and _target_is_usable(slot_type, cond):
            return cond, "source_block_lines:condition"
        return "", ""

    if slot_type == "FILL_EXPR":
        expr = _source_expression_from_line_window(lt_clean, line_nums, bounds=bounds)
        if expr and _target_is_usable(slot_type, expr):
            return _clean_target_for_output(slot_type, expr), "source_block_lines:expr"
        return "", ""

    if slot_type not in {"FILL_STMT", "FILL_EXPR"}:
        return "", ""
    target = _source_statements_from_line_window(
        lt_clean,
        line_nums,
        bounds=bounds,
        prefer_non_control_lines=True,
    )
    if not target:
        return "", ""
    decl_rows = _needed_local_declaration_rows_from_target(record, min(line_nums), target)
    if decl_rows:
        decl_texts = _statement_texts_from_rows(record, decl_rows, bb_lines=set())
        if decl_texts:
            target = " ".join(decl_texts + [target])
    return _clean_target_for_output(slot_type, target), "source_block_lines:stmt"


def _condition_from_exact_source_lines(
    lt: Dict[int, str],
    line_nums: Sequence[int],
    *,
    bounds: Optional[Tuple[int, int]] = None,
    expected_control_types: Optional[Set[str]] = None,
) -> str:
    if not line_nums:
        return ""
    sorted_lines = _clamp_lines_to_bounds([int(x) for x in line_nums if int(x) in lt], bounds)
    expected_keywords = set()
    for typ in expected_control_types or {"if_statement", "while_statement", "for_statement", "switch_statement"}:
        if typ == "if_statement":
            expected_keywords.add("if")
        elif typ == "while_statement":
            expected_keywords.add("while")
        elif typ == "for_statement":
            expected_keywords.add("for")
        elif typ == "switch_statement":
            expected_keywords.add("switch")
        elif typ == "do_statement":
            expected_keywords.add("while")
    if not expected_keywords:
        expected_keywords = {"if", "while", "for", "switch"}
    # Prefer a control header that starts exactly on one of the BB source
    # lines.  This fixes loop headers whose nearby body contains nested ifs.
    for ln in sorted_lines:
        line = str(lt.get(int(ln), "") or "")
        for kw in ("if", "while", "for", "switch"):
            if kw not in expected_keywords:
                continue
            for m in re.finditer(rf"\b{kw}\s*\(", line):
                start = line.find("(", m.end() - 1)
                end = _balanced_param_end(line, start)
                if end < 0:
                    continue
                cond = line[start + 1 : end].strip()
                if kw == "for":
                    parts = _split_top_level_semicolon_parts(cond)
                    if len(parts) >= 2:
                        cond = parts[1].strip() or "1"
                cond = _pretty_c_fragment(cond, "FILL_COND")
                if cond:
                    return cond
    raw = _source_lines_text(lt, sorted_lines)
    for kw in ("if", "while", "for", "switch"):
        if kw not in expected_keywords:
            continue
        for m in re.finditer(rf"\b{kw}\s*\(", raw):
            start = raw.find("(", m.end() - 1)
            end = _balanced_param_end(raw, start)
            if end < 0:
                continue
            cond = raw[start + 1 : end].strip()
            if kw == "for":
                parts = _split_top_level_semicolon_parts(cond)
                if len(parts) >= 2:
                    cond = parts[1].strip() or "1"
            cond = _pretty_c_fragment(cond, "FILL_COND")
            if cond:
                return cond
    spans = _control_spans_near_lines(sorted_lines, lt, bounds=bounds, lookback=18, lookahead=12)
    for kw, _start, end, _dist, text in sorted(spans, key=lambda x: (x[3], x[1])):
        if kw not in expected_keywords:
            continue
        open_paren = text.rfind("(", 0, end)
        close_paren = end - 1
        if open_paren < 0 or close_paren <= open_paren:
            continue
        cond = text[open_paren + 1 : close_paren].strip()
        if kw == "for":
            parts = _split_top_level_semicolon_parts(cond)
            if len(parts) >= 2:
                cond = parts[1].strip() or "1"
        cond = _pretty_c_fragment(cond, "FILL_COND")
        if cond:
            return cond
    return ""


def _needed_local_declaration_rows_from_target(record: Dict[str, Any], first_line: int, target_text: str) -> List[Dict[str, Any]]:
    used = _identifier_tokens(target_text)
    if not used:
        return []
    out: List[Dict[str, Any]] = []
    for row in _source_all_statement_rows(record):
        if int(row.get("start_line", 10**9)) > int(first_line):
            continue
        names = _declared_names_in_row(record, row)
        if names & used:
            out.append(row)
    return sorted(out, key=lambda r: (int(r.get("start_line", 0)), int(r.get("start_col", 0))))


def _source_multistmt_target_for_slot(
    record: Dict[str, Any],
    slot_type: str,
    bbs: Sequence[int],
    *,
    body_slot_count: int,
    old_target: str,
) -> Tuple[str, str]:
    if slot_type == "FILL_COND":
        lt = _line_to_text(record.get("source_file"))
        if not lt:
            return "", ""
        lt = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
        line_nums = _line_numbers_for_bbs(record, bbs)
        bounds = _function_body_line_range(record, lt)
        if bounds is not None:
            lo, hi = bounds
            line_nums = [ln for ln in line_nums if lo <= ln <= hi]
        if not line_nums:
            return "", ""
        lo = max(min(lt), min(line_nums) - 8)
        hi = min(max(lt), max(line_nums) + 12)
        if bounds is not None:
            lo = max(lo, bounds[0])
            hi = min(hi, bounds[1])
        raw = _source_lines_text(lt, [ln for ln in range(lo, hi + 1) if ln in lt])
        cond = _extract_source_condition(raw, prefer_last=True)
        if cond and _target_is_usable(slot_type, cond):
            return cond, "source_block_lines"
        return "", ""

    if slot_type not in {"FILL_STMT", "FILL_EXPR"}:
        return "", ""
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return "", ""
    lt = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))

    if body_slot_count == 1:
        whole = _function_body_statement_target(record, lt)
        if whole:
            return _clean_target_for_output(slot_type, whole), "source_function_body"

    line_nums = _line_numbers_for_bbs(record, bbs)
    bounds = _function_body_line_range(record, lt)
    if bounds is not None:
        lo, hi = bounds
        line_nums = [ln for ln in line_nums if lo <= ln <= hi]
    if not line_nums:
        return "", ""
    target = _source_statements_from_line_window(
        lt,
        line_nums,
        bounds=bounds,
        prefer_non_control_lines=True,
    )
    if not target:
        return "", ""
    return _clean_target_for_output(slot_type, target), "source_block_lines"


def _normalise_condition_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _iter_semicolon_statement_fragments(text: str) -> List[str]:
    """Split all semicolon-terminated C statements, including inside blocks."""
    s = str(text or "")
    out: List[str] = []
    start = 0
    par = brk = 0
    in_s = in_c = esc = False
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
        elif ch in "{}" and par <= 0 and brk <= 0:
            start = i + 1
        elif ch == ";" and par <= 0 and brk <= 0:
            frag = s[start : i + 1].strip()
            frag = re.sub(r"^[{}\s]+", "", frag)
            frag = re.sub(r"[{}\s]+$", "", frag)
            if frag:
                out.append(frag)
            start = i + 1
    return out


def _normalise_statement_for_compare(text: str) -> str:
    stmt = _pretty_c_fragment(text, "FILL_STMT")
    if not stmt or not _target_is_usable("FILL_STMT", stmt):
        return ""
    return _normalise_source_text(stmt)


def _rendered_duplicate_nested_if_condition(text: str) -> bool:
    """Reject obvious AB-skeleton over-nesting: if (c) { if (c) { ... } }."""
    s = str(text or "")
    for m in re.finditer(r"\bif\s*\(", s):
        outer_open = s.find("(", m.end() - 1)
        outer_close = _balanced_param_end(s, outer_open)
        if outer_close < 0:
            continue
        outer_cond = s[outer_open + 1 : outer_close]
        j = outer_close + 1
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            continue
        j += 1
        while j < len(s) and s[j].isspace():
            j += 1
        inner = re.match(r"if\s*\(", s[j:])
        if not inner:
            continue
        inner_open = s.find("(", j + inner.end() - 1)
        inner_close = _balanced_param_end(s, inner_open)
        if inner_close < 0:
            continue
        inner_cond = s[inner_open + 1 : inner_close]
        if (
            _normalise_condition_for_compare(outer_cond)
            and _normalise_condition_for_compare(outer_cond)
            == _normalise_condition_for_compare(inner_cond)
        ):
                return True
    return False


def _source_function_text(record: Dict[str, Any]) -> str:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return ""
    bounds = _function_body_line_range(record, lt)
    if bounds is None:
        return ""
    lo, hi = bounds
    return _source_lines_text(lt, [ln for ln in range(lo, hi + 1) if ln in lt])


def _control_conditions_in_text(text: str) -> List[str]:
    raw = _normalise_source_text(text)
    out: List[str] = []
    for kw in ("if", "while", "for", "switch"):
        for m in re.finditer(rf"\b{kw}\s*\(", raw):
            start = raw.find("(", m.end() - 1)
            end = _balanced_param_end(raw, start)
            if end < 0:
                continue
            cond = raw[start + 1 : end].strip()
            if kw == "for":
                parts = _split_top_level_semicolon_parts(cond)
                if len(parts) >= 2:
                    cond = parts[1].strip() or "1"
            cond = _pretty_c_fragment(cond, "FILL_COND")
            if cond:
                out.append(_normalise_condition_for_compare(cond))
    return out


def _normalised_semicolon_statement_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for frag in _iter_semicolon_statement_fragments(text):
        key = _normalise_statement_for_compare(frag)
        if key:
            counts[key] = counts.get(key, 0) + 1
        # Source code often has single-statement controls without braces:
        #
        #   if (cond)
        #     remove_suffix (...);
        #
        # The rendered V10 output always braces the skeleton control, so the
        # rendered statement counter sees only ``remove_suffix (...);`` inside
        # the block. Count the simple control body as well, otherwise duplicate
        # slots for that body are not trimmed/rejected.
        body = _extract_simple_control_body_statement(frag)
        if body:
            for body_frag in _iter_semicolon_statement_fragments(body):
                body_key = _normalise_statement_for_compare(body_frag)
                if body_key:
                    counts[body_key] = counts.get(body_key, 0) + 1
    return counts


def _trim_stmt_target_to_source_budget(
    target: str,
    source_counts: Dict[str, int],
    used_counts: Dict[str, int],
) -> Tuple[str, bool]:
    if not source_counts:
        return target, False
    changed = False
    kept: List[str] = []
    for frag in _iter_semicolon_statement_fragments(target):
        stmt = _pretty_c_fragment(frag, "FILL_STMT")
        key = _normalise_statement_for_compare(stmt)
        if not stmt or not key:
            continue
        allowed = int(source_counts.get(key, 0))
        if allowed > 0 and int(used_counts.get(key, 0)) >= allowed:
            changed = True
            continue
        kept.append(stmt)
        if allowed > 0:
            used_counts[key] = int(used_counts.get(key, 0)) + 1
    if not kept:
        return "", True
    trimmed = _clean_target_for_output("FILL_STMT", " ".join(kept))
    if _normalise_source_text(trimmed) != _normalise_source_text(target):
        changed = True
    return trimmed, changed


def _rendered_has_extra_duplicate_source_statements(text: str, record: Dict[str, Any]) -> bool:
    source = _source_function_text(record)
    if not source:
        return False
    src_counts = _normalised_semicolon_statement_counts(source)
    if not src_counts:
        return False
    rendered_counts = _normalised_semicolon_statement_counts(text)
    for stmt, cnt in rendered_counts.items():
        if cnt > src_counts.get(stmt, 0) and src_counts.get(stmt, 0) > 0:
            return True
    return False


def _rendered_missing_source_control_condition(text: str, record: Dict[str, Any]) -> bool:
    source = _source_function_text(record)
    if not source:
        return False
    src_counts: Dict[str, int] = {}
    for cond in _control_conditions_in_text(source):
        src_counts[cond] = src_counts.get(cond, 0) + 1
    if not src_counts:
        return False
    rendered_counts: Dict[str, int] = {}
    for cond in _control_conditions_in_text(text):
        rendered_counts[cond] = rendered_counts.get(cond, 0) + 1
    return any(rendered_counts.get(cond, 0) < cnt for cond, cnt in src_counts.items())


def _rendered_has_extra_duplicate_source_conditions(text: str, record: Dict[str, Any]) -> bool:
    source = _source_function_text(record)
    if not source:
        return False
    src_counts: Dict[str, int] = {}
    for cond in _control_conditions_in_text(source):
        src_counts[cond] = src_counts.get(cond, 0) + 1
    if not src_counts:
        return False
    rendered_counts: Dict[str, int] = {}
    for cond in _control_conditions_in_text(text):
        rendered_counts[cond] = rendered_counts.get(cond, 0) + 1
    for cond, cnt in rendered_counts.items():
        src_cnt = src_counts.get(cond, 0)
        if src_cnt <= 0:
            continue
        # Binary short-circuit lowering may duplicate the same source condition
        # in adjacent header slots.  This is a skeleton-quality warning, but it
        # should not hard-drop otherwise useful source-derived slots.
        if cnt > src_cnt + 1:
            return True
    return False


def _ordered_source_statement_keys(record: Dict[str, Any]) -> List[str]:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return []
    rows = _source_all_statement_rows(record)
    if not rows:
        rows = _source_statement_table(record)
    out: List[str] = []
    seen: Set[str] = set()
    for row in sorted(
        rows,
        key=lambda r: (
            int(r.get("start_line", 0)),
            int(r.get("start_col", 0)),
            int(r.get("end_line", 0)),
            -int(r.get("depth", 0)),
        ),
    ):
        raw = _statement_text_from_lt(lt, row)
        rtype = str(row.get("type") or "")
        texts: List[str]
        if rtype in {"if_statement", "while_statement", "for_statement", "do_statement", "switch_statement"}:
            body = _extract_simple_control_body_statement(raw)
            texts = [body] if body else []
        else:
            texts = [raw]
        for txt in texts:
            for frag in _iter_semicolon_statement_fragments(txt):
                key = _normalise_statement_for_compare(frag)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out


def _rendered_statement_order_mismatch(text: str, record: Dict[str, Any]) -> bool:
    source_keys = _ordered_source_statement_keys(record)
    if len(source_keys) < 2:
        return False
    source_pos = {key: i for i, key in enumerate(source_keys)}
    seen_positions: List[int] = []
    for frag in _iter_semicolon_statement_fragments(text):
        key = _normalise_statement_for_compare(frag)
        if key in source_pos:
            seen_positions.append(source_pos[key])
    if len(seen_positions) < 2:
        return False
    max_allowed_backstep = 0
    prev = seen_positions[0]
    for pos in seen_positions[1:]:
        if pos + max_allowed_backstep < prev:
            return True
        prev = max(prev, pos)
    return False


def _all_local_declaration_names(record: Dict[str, Any], rendered_text: str) -> Set[str]:
    names: Set[str] = set()
    for row in _source_all_statement_rows(record):
        if str(row.get("type") or "") == "declaration":
            names.update(_declared_names_in_row(record, row))
    for stmt in _iter_semicolon_statement_fragments(rendered_text):
        names.update(_statement_declared_names(stmt))
    return names


def _rendered_use_before_declaration(text: str, record: Dict[str, Any]) -> bool:
    local_names = _all_local_declaration_names(record, text)
    if not local_names:
        return False
    declared = _collect_declared_names(_function_decl(record, "func")[1])
    protected, _lits = _protect_c_literals(_strip_c_comments(str(text or "")))
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[;{}()]")
    pending_decl = ""
    in_decl = False
    skip_control_condition_depth = 0
    prev_sig: List[str] = []
    for m in token_re.finditer(protected):
        tok = m.group(0)
        if tok in {"if", "while", "for", "switch"}:
            skip_control_condition_depth = 0
            prev_sig.append(tok)
            prev_sig = prev_sig[-4:]
            continue
        if tok == "(" and prev_sig and prev_sig[-1] in {"if", "while", "for", "switch"}:
            skip_control_condition_depth = 1
            continue
        if skip_control_condition_depth:
            if tok == "(":
                skip_control_condition_depth += 1
            elif tok == ")":
                skip_control_condition_depth -= 1
            elif tok in local_names and tok not in declared:
                return True
            continue
        if tok == ";":
            for name in _statement_declared_names(pending_decl + ";"):
                declared.add(name)
            pending_decl = ""
            in_decl = False
            prev_sig.clear()
            continue
        if tok in "{}":
            prev_sig.clear()
            continue
        if tok in {
            "static", "extern", "register", "auto", "const", "volatile", "restrict",
            "unsigned", "signed", "short", "long", "int", "char", "bool", "_Bool",
            "float", "double", "size_t", "ssize_t", "idx_t", "uintmax_t",
            "uint64_t", "uint32_t", "FILE", "uid_t", "gid_t", "pid_t", "mode_t",
            "struct", "enum", "union",
        }:
            in_decl = True
        if in_decl:
            pending_decl += tok + " "
        elif tok in local_names and tok not in declared:
            # Ignore function calls and member names; they are not local uses.
            tail = protected[m.end():m.end() + 8]
            head = protected[max(0, m.start() - 3):m.start()]
            if re.match(r"\s*\(", tail):
                pass
            elif re.search(r"(?:\.|->)\s*$", head):
                pass
            else:
                return True
        if tok not in {"(", ")"}:
            prev_sig.append(tok)
            prev_sig = prev_sig[-4:]
    return False


def _statement_declared_names(stmt: str) -> List[str]:
    s = _pretty_c_fragment(stmt, "FILL_STMT")
    if not s or not s.endswith(";"):
        return []
    if re.match(r"^(?:return|if|while|for|switch|break|continue|goto)\b", s):
        return []
    before = s.split("=", 1)[0].rstrip(";").strip()
    if "," in before:
        before = before.split(",", 1)[0].strip()
    typeish = (
        r"(?:static|extern|register|auto|const|volatile|restrict|unsigned|signed|"
        r"short|long|int|char|bool|_Bool|float|double|size_t|ssize_t|idx_t|"
        r"uintmax_t|uint64_t|uint32_t|FILE|struct\s+[A-Za-z_][A-Za-z0-9_]*|"
        r"enum\s+[A-Za-z_][A-Za-z0-9_]*|union\s+[A-Za-z_][A-Za-z0-9_]*)"
    )
    if not re.search(typeish, before):
        return []
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$", before)
    return [m.group(1)] if m else []


def _collect_declared_names(text: str) -> Set[str]:
    names: Set[str] = set()
    m = re.search(r"\bfunc\s*\(([^)]*)\)", str(text or ""))
    if m:
        params = m.group(1).strip()
        if params and params != "void":
            for part in params.split(","):
                pm = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$", part.strip())
                if pm:
                    names.add(pm.group(1))
    for stmt in _split_semicolon_statements(text):
        names.update(_statement_declared_names(stmt))
    return names


def _strip_redeclared_local_declarations(target: str, declared: Set[str]) -> str:
    if not target or not declared:
        return target
    out: List[str] = []
    changed = False
    for frag in _split_semicolon_statements(target):
        stmt = _pretty_c_fragment(frag, "FILL_STMT")
        names = _statement_declared_names(stmt)
        if names and any(name in declared for name in names):
            changed = True
            continue
        if stmt:
            out.append(stmt)
            declared.update(_statement_declared_names(stmt))
    return " ".join(out) if changed else target


def _can_drop_no_source_empty_stmt_slot(
    record: Dict[str, Any],
    slot_type: str,
    bbs: Sequence[int],
    old_target: str,
    reason: str,
) -> bool:
    if slot_type != "FILL_STMT":
        return False
    if str(old_target or "").strip():
        return False
    if _line_numbers_for_bbs(record, bbs):
        return False
    if not bbs:
        return False
    return str(reason or "") in {
        "alignment_row_has_zero_mass",
        "no_source_statement_columns",
        "no_source_target",
    }


def _can_drop_regenerated_unmapped_stmt_slot(
    record: Dict[str, Any],
    slot_type: str,
    bbs: Sequence[int],
    reason: str,
) -> bool:
    if slot_type != "FILL_STMT":
        return False
    if _line_numbers_for_bbs(record, bbs):
        return False
    return str(reason or "") in {
        "alignment_row_has_zero_mass",
        "no_source_statement_columns",
        "no_source_target",
        "",
    }


def _line_window_is_control_only_stmt_slot(record: Dict[str, Any], bbs: Sequence[int]) -> bool:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return False
    lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    bounds = _function_body_line_range(record, lt_clean)
    lines = _clamp_lines_to_bounds(_line_numbers_for_bbs(record, bbs), bounds)
    if not lines:
        return False
    nonempty = [str(lt_clean.get(int(ln), "") or "") for ln in lines if str(lt_clean.get(int(ln), "") or "").strip()]
    return bool(nonempty) and all(_line_looks_like_control_only(line) for line in nonempty)


def _line_window_starts_return_stmt(record: Dict[str, Any], bbs: Sequence[int]) -> bool:
    lt = _line_to_text(record.get("source_file"))
    if not lt:
        return False
    lt_clean = _strip_block_comments_from_lt(_strip_preprocessor_lines_from_lt(lt))
    bounds = _function_body_line_range(record, lt_clean)
    lines = _clamp_lines_to_bounds(_line_numbers_for_bbs(record, bbs), bounds)
    if not lines:
        return False
    target = _source_statements_from_line_window(
        lt_clean,
        lines,
        bounds=bounds,
        prefer_non_control_lines=True,
    )
    return _normalise_source_text(target).startswith("return ")


def _has_later_placeholder_type(matches: Sequence[re.Match[str]], start_index: int, slot_type: str) -> bool:
    return any(f"FILL_{m.group(1)}" == slot_type for m in matches[int(start_index) + 1:])


def _rendered_uses_obvious_undeclared_locals(text: str) -> List[str]:
    declared = _collect_declared_names(text)
    suspects = {
        "length", "len", "i", "j", "k", "n", "p", "q", "out", "in",
        "buf", "tmp", "fd", "ret", "status", "dotslash_len",
    }
    out: List[str] = []
    for name in sorted(suspects):
        if name in declared:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            # Function calls and struct fields are not local variable uses.
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(", text):
                continue
            if re.search(rf"(?:\.|->)\s*{re.escape(name)}(?![A-Za-z0-9_])", text):
                continue
            out.append(name)
    return out[:5]


def _return_type_requires_value(ret_type: str) -> bool:
    x = _normalise_source_text(ret_type)
    if not x or x.startswith("__STRUCLIFT_UNKNOWN"):
        return False
    # ``void *`` returns a value; ``static inline void`` does not.
    if "*" in x:
        return True
    ignored = {
        "static", "extern", "inline", "__inline", "__inline__",
        "const", "volatile", "restrict", "register", "auto",
    }
    toks = [
        t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", x)
        if t not in ignored
    ]
    return not (toks and toks[-1] == "void")


def _strict_reject_rendered_output(text: str, record: Dict[str, Any]) -> str:
    """Hard reject only defects that directly corrupt slot supervision."""
    if _rendered_duplicate_nested_if_condition(text):
        return "rendered_duplicate_nested_if_condition"
    if _rendered_has_extra_duplicate_source_conditions(text, record):
        return "rendered_extra_duplicate_source_condition"
    if _rendered_has_extra_duplicate_source_statements(text, record):
        return "rendered_extra_duplicate_source_statement"
    if re.search(r"\bif\s*\([^)]*;", str(text or "")):
        return "rendered_if_condition_has_semicolon"
    return ""


def _rendered_quality_warnings(text: str, record: Dict[str, Any]) -> List[str]:
    """Warnings for AB skeleton incompleteness; do not drop source-GT slots."""
    out: List[str] = []
    ret = _signature(record)[0].strip()
    if _return_type_requires_value(ret) and not re.search(r"\breturn\b", text):
        out.append("rendered_nonvoid_missing_return")
    undeclared = _rendered_uses_obvious_undeclared_locals(text)
    if undeclared:
        out.append("rendered_obvious_undeclared_locals:" + ",".join(undeclared))
    if _rendered_statement_order_mismatch(text, record):
        out.append("rendered_source_statement_order_mismatch")
    if _rendered_use_before_declaration(text, record):
        out.append("rendered_use_before_declaration")
    if _rendered_missing_source_control_condition(text, record):
        out.append("rendered_missing_source_control_condition")
    return out


def _quality_bucket(warnings: Sequence[str]) -> str:
    hardish_prefixes = (
        "rendered_obvious_undeclared_locals:",
        "rendered_source_statement_order_mismatch",
        "rendered_use_before_declaration",
    )
    if any(any(str(w).startswith(p) for p in hardish_prefixes) for w in warnings):
        return "suspicious"
    if warnings:
        return "warning"
    return "clean"


def _line_for_prompt_slot(slot_type: str, slot_idx: int, bbs: Sequence[int]) -> str:
    bb_text = (
        ",".join(f"BB_{int(x)}" for x in bbs)
        if bbs else "signature/global"
    )
    tok = _fill_token(slot_idx)
    if slot_type == "FILL_COND":
        return f"if ({tok}) {{ /* aligned to {bb_text} */\n    }}\n"
    return f"    {tok} /* aligned to {bb_text} */\n"


def _line_for_output_slot(slot_type: str, target: str) -> str:
    if not target:
        return ""
    if slot_type == "FILL_COND":
        return f"if ({target}) {{\n    }}\n"
    return f"    {target}\n"


def _strip_output_alignment_comments(text: str) -> str:
    text = re.sub(r"\s*/\*\s*aligned to .*?\*/", "", text)
    text = re.sub(r"\s*//\s*aligned to .*$", "", text, flags=re.M)
    return text


def _slot_index_from_match(m: re.Match[str], occurrence: int, n_slots: int) -> int:
    sid = m.group(2)
    sm = re.search(r"(\d+)$", sid)
    if sm:
        idx = int(sm.group(1))
        if 0 <= idx < n_slots:
            return idx
    return occurrence


def _slot_index_from_placeholder(m: re.Match[str], occurrence: int) -> int:
    sid = m.group(2)
    sm = re.search(r"(\d+)$", sid)
    if sm:
        return int(sm.group(1))
    return occurrence


def _aligned_bbs_from_slot_tail(tail: str) -> List[int]:
    """Parse the nearest ``aligned to BB_x`` comment after a skeleton slot."""
    window = tail[:300]
    m = ALIGNED_BB_RE.search(window)
    if not m:
        return []
    return [int(x) for x in re.findall(r"BB_(\d+)", m.group(1))]


def _expected_control_types_from_slot_tail(tail: str) -> Optional[Set[str]]:
    """Infer the source control kind expected by a regenerated condition slot."""
    window = str(tail or "")[:300]
    m = SLOT_KIND_RE.search(window)
    kind = str(m.group(1) if m else "")
    if "if_header" in kind:
        return {"if_statement"}
    if "loop_header" in kind:
        return {"while_statement", "for_statement", "do_statement"}
    if "switch_header" in kind:
        return {"switch_statement"}
    return None


def _render_prompt_skeleton(
    record: Dict[str, Any],
    *,
    name_for_prompt: str,
    allow_v9_teacher_fallback: bool,
    skeleton_source: str = "regenerate",
) -> Tuple[Optional[str], List[str], List[str], List[List[int]], List[int], List[str], Set[int], Optional[str]]:
    """Render prompt skeleton from the V9 Module-B skeleton.

    Function declaration placeholders are materialised. Body placeholders are
    replaced by sequential StrucLift FILL tokens, preserving the original
    skeleton structure.
    """
    skeleton_src, skeleton_src_name = _skeleton_text_for_repack(record, skeleton_source)
    if not skeleton_src:
        return None, [], [], [], [], [], set(), "missing sft_skeleton_text_pretty"

    slot_types_all = list(record.get("slot_types") or [])
    targets_all = list(record.get("sft_teacher_targets") or [])
    supervise_all = list(record.get("sft_teacher_slot_supervise") or [])
    bb_all = list(record.get("sft_slot_bb_indices") or [])

    prompt_decl, output_decl = _function_decl(record, name_for_prompt)

    keep_slot_types: List[str] = []
    keep_targets: List[str] = []
    keep_bbs: List[List[int]] = []
    keep_old_indices: List[int] = []
    keep_target_sources: List[str] = []
    dropped_old_indices: Set[int] = set()
    out_parts: List[str] = []
    skip_prompt_leading_semicolon = False
    regenerated_skeleton = _uses_regenerated_skeleton(skeleton_src_name)

    def add_skeleton_chunk(chunk: str) -> None:
        nonlocal skip_prompt_leading_semicolon
        if skip_prompt_leading_semicolon:
            chunk = re.sub(r"^\s*;", "", chunk, count=1)
            skip_prompt_leading_semicolon = False
        out_parts.append(chunk)

    pos = 0
    occurrence = 0
    matches = list(FILL_RE.finditer(skeleton_src))
    declared_so_far = _collect_declared_names(_function_decl(record, name_for_prompt)[1])
    source_stmt_budget = _normalised_semicolon_statement_counts(_source_function_text(record))
    used_stmt_budget: Dict[str, int] = {}
    source_cond_budget: Counter[str] = Counter(_control_conditions_in_text(_source_function_text(record)))
    used_cond_budget: Counter[str] = Counter()
    if regenerated_skeleton:
        supervise_all = [True] * len(matches)
    source_ast_skeleton = _uses_source_ast_skeleton(skeleton_src_name)
    n_body_slots_total = (
        sum(1 for m in matches if f"FILL_{m.group(1)}" in BODY_SLOT_TYPES)
        if regenerated_skeleton
        else sum(1 for st in slot_types_all if str(st) in BODY_SLOT_TYPES)
    )
    for mi, m in enumerate(matches):
        add_skeleton_chunk(skeleton_src[pos:m.start()])
        old_idx = _slot_index_from_match(m, occurrence, len(slot_types_all))
        if regenerated_skeleton:
            old_idx = _slot_index_from_placeholder(m, occurrence)
        occurrence += 1
        stype = (
            f"FILL_{m.group(1)}"
            if regenerated_skeleton
            else (str(slot_types_all[old_idx]) if old_idx < len(slot_types_all) else f"FILL_{m.group(1)}")
        )
        if stype == "FILL_TYPE":
            out_parts.append(_signature(record)[0])
            pos = m.end()
            continue
        if stype == "FILL_PARAM":
            out_parts.append(_signature(record)[1])
            pos = m.end()
            continue
        if stype not in BODY_SLOT_TYPES:
            out_parts.append(m.group(0))
            pos = m.end()
            continue
        keep = (
            True
            if regenerated_skeleton
            else (
                bool(supervise_all[old_idx])
                if old_idx < len(supervise_all)
                else bool(old_idx < len(targets_all) and targets_all[old_idx])
            )
        )
        target = _clean_target_for_output(
            stype,
            str(targets_all[old_idx] or "") if old_idx < len(targets_all) else "",
        )
        next_start = matches[mi + 1].start() if mi + 1 < len(matches) else len(skeleton_src)
        slot_tail = skeleton_src[m.end():next_start]
        comment_bbs = _aligned_bbs_from_slot_tail(slot_tail)
        expected_control_types = _expected_control_types_from_slot_tail(slot_tail)
        bbs_raw = bb_all[old_idx] if old_idx < len(bb_all) else []
        field_bbs = [
            int(x) for x in bbs_raw
            if isinstance(x, (int, float)) and int(x) >= 0
        ] if isinstance(bbs_raw, list) else []
        bbs = comment_bbs if regenerated_skeleton else (comment_bbs or field_bbs)
        if (
            regenerated_skeleton
            and stype == "FILL_STMT"
            and not source_ast_skeleton
            and "(cond_prefix)" not in slot_tail
            and HEADER_STMT_SLOT_RE.search(slot_tail)
        ):
            dropped_old_indices.add(old_idx)
            skip_prompt_leading_semicolon = True
            pos = m.end()
            continue
        if (
            regenerated_skeleton
            and not source_ast_skeleton
            and stype == "FILL_STMT"
            and not str(target or "").strip()
            and _line_window_is_control_only_stmt_slot(record, bbs)
        ):
            dropped_old_indices.add(old_idx)
            skip_prompt_leading_semicolon = True
            pos = m.end()
            continue
        if source_ast_skeleton:
            source_target, source_reason = _source_ast_target_for_slot(record, stype, old_idx)
        elif regenerated_skeleton and stype == "FILL_STMT" and "(cond_prefix)" in slot_tail:
            source_target, source_reason = _source_condition_prefix_target_for_slot(
                record,
                bbs,
                expected_control_types=expected_control_types,
            )
        else:
            source_target, source_reason = _source_alignment_target_for_slot(
                record,
                stype,
                bbs,
                expected_control_types=expected_control_types if stype == "FILL_COND" else None,
            )
        align_fail_reason = source_reason
        if not source_target:
            source_target, fallback_reason = _source_multistmt_target_for_slot(
                record,
                stype,
                bbs,
                body_slot_count=n_body_slots_total,
                old_target=target,
            )
            source_reason = fallback_reason or source_reason
        target_source = f"v9_teacher_target:skeleton_{skeleton_src_name}"
        if source_target:
            target = source_target
            target_source = source_reason or "source_multistmt"
            target_source = f"{target_source}:skeleton_{skeleton_src_name}"
            keep = True
        elif not allow_v9_teacher_fallback:
            detail = align_fail_reason or source_reason or "no_source_target"
            if (
                _can_drop_no_source_empty_stmt_slot(record, stype, bbs, target, detail)
                or (
                    regenerated_skeleton
                    and not source_ast_skeleton
                    and _can_drop_regenerated_unmapped_stmt_slot(record, stype, bbs, detail)
                )
                or (
                    regenerated_skeleton
                    and not source_ast_skeleton
                    and stype == "FILL_STMT"
                    and _function_body_is_single_return(record)
                    and _has_later_placeholder_type(matches, mi, "FILL_EXPR")
                )
            ):
                dropped_old_indices.add(old_idx)
                skip_prompt_leading_semicolon = True
                pos = m.end()
                continue
            return None, [], [], [], [], [], dropped_old_indices, f"missing source-derived target for {stype} slot_{old_idx}: {detail}"

        if (
            regenerated_skeleton
            and not source_ast_skeleton
            and stype == "FILL_STMT"
            and (
                _normalise_source_text(target).startswith("return ")
                or _line_window_starts_return_stmt(record, bbs)
            )
            and _has_later_placeholder_type(matches, mi, "FILL_EXPR")
        ):
            dropped_old_indices.add(old_idx)
            skip_prompt_leading_semicolon = True
            pos = m.end()
            continue

        if stype == "FILL_STMT" and _slot_context_is_inside_control(skeleton_src, m.start()):
            stripped = _strip_redundant_control_from_stmt_target(target)
            if stripped != target:
                target = _clean_target_for_output(stype, stripped)
                target_source = f"{target_source}:control_body"
        if regenerated_skeleton and not source_ast_skeleton and stype == "FILL_COND":
            replacement, replacement_source = _condition_budget_replacement(
                record,
                bbs,
                source_cond_budget,
                used_cond_budget,
                target,
                expected_control_types=expected_control_types,
            )
            if replacement:
                target = _clean_target_for_output(stype, replacement)
                target_source = f"{replacement_source}:skeleton_{skeleton_src_name}"
        if stype == "FILL_STMT":
            stripped_decls = _strip_redeclared_local_declarations(target, set(declared_so_far))
            if stripped_decls != target:
                target = _clean_target_for_output(stype, stripped_decls)
                target_source = f"{target_source}:drop_redecl"
            trimmed, trimmed_changed = _trim_stmt_target_to_source_budget(
                target,
                source_stmt_budget,
                used_stmt_budget,
            )
            if trimmed_changed:
                target = _clean_target_for_output(stype, trimmed)
                target_source = f"{target_source}:drop_duplicate_source_stmt"
        if stype == "FILL_STMT" and not str(target or "").strip():
            dropped_old_indices.add(old_idx)
            skip_prompt_leading_semicolon = True
            pos = m.end()
            continue

        if not keep or not target:
            return None, [], [], [], [], [], dropped_old_indices, f"missing required target for {stype} slot_{old_idx}"
        if not _target_is_usable(stype, target):
            return None, [], [], [], [], [], dropped_old_indices, f"unusable target for {stype} slot_{old_idx}"

        new_idx = len(keep_slot_types)
        out_parts.append(_fill_token(new_idx))
        keep_slot_types.append(stype)
        keep_targets.append(target)
        keep_bbs.append(bbs)
        keep_old_indices.append(old_idx)
        keep_target_sources.append(target_source)
        declared_so_far.update(_collect_declared_names(target))
        if stype == "FILL_COND":
            cond_key = _normalise_condition_for_compare(target)
            if cond_key:
                used_cond_budget[cond_key] += 1
        pos = m.end()

    add_skeleton_chunk(skeleton_src[pos:])
    prompt_skeleton = "".join(out_parts)
    prompt_skeleton = re.sub(r"^\s*[^()\n{};]+?\s+func\s*\([^)]*\)", prompt_decl, prompt_skeleton, count=1)
    return (
        prompt_skeleton,
        keep_slot_types,
        keep_targets,
        keep_bbs,
        keep_old_indices,
        keep_target_sources,
        dropped_old_indices,
        None,
    )


def _build_output_with_labels(
    record: Dict[str, Any],
    tokenizer: Any,
    *,
    name_for_prompt: str,
    prompt_len: int,
    slot_types: Sequence[str],
    targets: Sequence[str],
    old_indices: Sequence[int],
    dropped_old_indices: Set[int],
    skeleton_source: str = "regenerate",
) -> Tuple[str, List[int], List[int], List[List[int]], Optional[str]]:
    """Render filled complete function and token labels.

    Non-slot structure/signature tokens are labelled ``-100``. Tokens inserted
    from supervised slot targets keep their ids as labels.
    """
    skeleton_src, _skeleton_src_name = _skeleton_text_for_repack(record, skeleton_source)
    if not skeleton_src:
        return "", [], [], [], "missing sft_skeleton_text_pretty"
    regenerated_skeleton = _uses_regenerated_skeleton(_skeleton_src_name)
    output_decl = _function_decl(record, name_for_prompt)[1]
    target_by_old = {int(i): str(t or "") for i, t in zip(old_indices, targets)}
    slot_new_by_old = {int(old): new for new, old in enumerate(old_indices)}
    slot_types_all = list(record.get("slot_types") or [])
    supervise_all = list(record.get("sft_teacher_slot_supervise") or [])
    if regenerated_skeleton:
        placeholder_types = [
            f"FILL_{m.group(1)}" for m in FILL_RE.finditer(skeleton_src)
        ]
        slot_types_all = placeholder_types
        supervise_all = [True] * len(placeholder_types)

    ids: List[int] = []
    labels: List[int] = []
    spans: List[List[int]] = [[0, 0] for _ in slot_types]
    text_parts: List[str] = []
    pos = 0
    occurrence = 0
    skip_leading_semicolon = False
    replaced_decl = False

    def add_text(chunk: str) -> None:
        nonlocal skip_leading_semicolon, replaced_decl
        if skip_leading_semicolon:
            chunk = re.sub(r"^\s*;", "", chunk, count=1)
            skip_leading_semicolon = False
        chunk = _strip_output_alignment_comments(chunk)
        if not replaced_decl:
            chunk, n_sub = re.subn(
                r"^\s*[^()\n{};]+?\s+func\s*\([^)]*\)",
                output_decl,
                chunk,
                count=1,
            )
            if n_sub:
                replaced_decl = True
        if not chunk:
            return
        text_parts.append(chunk)
        toks = _tok(tokenizer, chunk)
        ids.extend(toks)
        labels.extend([-100] * len(toks))

    for m in FILL_RE.finditer(skeleton_src):
        add_text(skeleton_src[pos:m.start()])
        old_idx = _slot_index_from_match(m, occurrence, len(slot_types_all))
        if regenerated_skeleton:
            old_idx = _slot_index_from_placeholder(m, occurrence)
        occurrence += 1
        stype = (
            f"FILL_{m.group(1)}"
            if regenerated_skeleton
            else (str(slot_types_all[old_idx]) if old_idx < len(slot_types_all) else f"FILL_{m.group(1)}")
        )
        if stype == "FILL_TYPE":
            add_text(_signature(record)[0])
        elif stype == "FILL_PARAM":
            add_text(_signature(record)[1])
        elif old_idx in dropped_old_indices:
            skip_leading_semicolon = True
        elif old_idx in target_by_old:
            target = target_by_old[old_idx]
            new_idx = slot_new_by_old[old_idx]
            text_parts.append(target)
            toks = _tok(tokenizer, target)
            start = prompt_len + len(ids)
            ids.extend(toks)
            keep = (
                True
                if regenerated_skeleton
                else bool(supervise_all[old_idx])
                if old_idx < len(supervise_all)
                else bool(target)
            )
            labels.extend(toks if keep and target else [-100] * len(toks))
            end = prompt_len + len(ids)
            spans[new_idx] = [start, end] if keep and target else [0, 0]
            if target.rstrip().endswith(";"):
                skip_leading_semicolon = True
        else:
            # Unsupervised empty statement slots are rendered as empty text. For
            # required expression/condition slots we should have failed earlier.
            pass
        pos = m.end()
    add_text(skeleton_src[pos:])

    output_text = "".join(text_parts)
    return output_text, ids, labels, spans, None


def _binary_info(record: Dict[str, Any], name_for_prompt: str) -> str:
    ret, params, _ = _signature(record)
    return (
        "<|binary_info|>\n"
        "Architecture: x86_64\n"
        "Compiler: unknown (stripped)\n"
        f"Function signature hint: {ret} {name_for_prompt}({params})\n"
        f"Function: {name_for_prompt}\n"
        "<|/binary_info|>"
    )


def _function_asm_lines(record: Dict[str, Any], max_total: int, max_per_bb: int) -> List[str]:
    instr = record.get("bin_instr_text")
    lengths = record.get("bin_block_lengths")
    if not isinstance(instr, list) or not isinstance(lengths, list):
        return []
    out = ["function_asm:"]
    used = 0
    for bb, text in enumerate(instr):
        if used >= max_total:
            break
        lines: List[str]
        if isinstance(text, list):
            lines = [str(x) for x in text]
        else:
            lines = [p.strip() for p in str(text or "").split("|") if p.strip()]
        if not lines and bb < len(lengths):
            lines = [f"<{int(lengths[bb])} instructions unavailable>"]
        shown = lines[: max(1, max_per_bb)]
        for line in shown:
            if used >= max_total:
                break
            out.append(f"BB_{bb}: {line}")
            used += 1
        more = len(lines) - len(shown)
        if more > 0 and used < max_total:
            out.append(f"BB_{bb}: ... truncated {more} more instructions in BB_{bb}")
            used += 1
    return out if len(out) > 1 else []


def _slot_context_lines(
    slot_types: Sequence[str],
    bbs_all: Sequence[Sequence[int]],
    record: Dict[str, Any],
    *,
    max_instr_per_slot: int,
) -> List[str]:
    instr = record.get("bin_instr_text")
    lines: List[str] = []
    for i, (stype, bbs) in enumerate(zip(slot_types, bbs_all)):
        if bbs:
            bb_text = ",".join(f"BB_{int(x)}" for x in bbs)
            instr_bits: List[str] = []
            for bb in bbs:
                if isinstance(instr, list) and 0 <= int(bb) < len(instr):
                    raw = instr[int(bb)]
                    if isinstance(raw, list):
                        parts = [str(x) for x in raw]
                    else:
                        parts = [p.strip() for p in str(raw or "").split("|") if p.strip()]
                    instr_bits.extend(parts[:max_instr_per_slot])
            suffix = " | ".join(instr_bits[:max_instr_per_slot])
            if suffix:
                lines.append(f"slot_{i} [{stype}; {bb_text}]: {suffix}")
            else:
                lines.append(f"slot_{i} [{stype}; {bb_text}]: use graph adapter")
        else:
            lines.append(f"slot_{i} [{stype}; signature/global]: no local binary evidence")
    return lines


def _binary_context(
    record: Dict[str, Any],
    slot_types: Sequence[str],
    bbs: Sequence[Sequence[int]],
    *,
    asm_max_total_lines: int,
    asm_max_lines_per_bb: int,
    slot_max_instr: int,
) -> str:
    lines = _function_asm_lines(record, asm_max_total_lines, asm_max_lines_per_bb)
    lines.extend(_slot_context_lines(slot_types, bbs, record, max_instr_per_slot=slot_max_instr))
    return "<|binary_context|>\n" + "\n".join(lines).strip() + "\n<|/binary_context|>"


def _prompt_text(system: str, bin_info: str, skeleton: str, ctx: str) -> str:
    return "\n\n".join([
        system.strip(),
        bin_info.strip(),
        "<|skeleton|>\n" + skeleton.strip() + "\n<|/skeleton|>",
        ctx.strip(),
        "<|output|>\n",
    ])


def _build_tokenized_v10(
    record: Dict[str, Any],
    tokenizer: Any,
    *,
    system: str,
    name_for_prompt: str,
    max_length: int,
    asm_max_total_lines: int,
    asm_max_lines_per_bb: int,
    slot_max_instr: int,
    allow_v9_teacher_fallback: bool,
    skeleton_source: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    skeleton, slot_types, targets, bbs, old_indices, target_sources, dropped_old_indices, err = _render_prompt_skeleton(
        record,
        name_for_prompt=name_for_prompt,
        allow_v9_teacher_fallback=allow_v9_teacher_fallback,
        skeleton_source=skeleton_source,
    )
    if err:
        return None, err
    assert skeleton is not None
    if not slot_types:
        return None, "no supervised body slots"

    ctx = _binary_context(
        record,
        slot_types,
        bbs,
        asm_max_total_lines=asm_max_total_lines,
        asm_max_lines_per_bb=asm_max_lines_per_bb,
        slot_max_instr=slot_max_instr,
    )
    prompt = _prompt_text(system, _binary_info(record, name_for_prompt), skeleton, ctx)

    prompt_ids = _tok(tokenizer, prompt)
    output_text, output_ids, output_labels, spans, err = _build_output_with_labels(
        record,
        tokenizer,
        name_for_prompt=name_for_prompt,
        prompt_len=len(prompt_ids),
        slot_types=slot_types,
        targets=targets,
        old_indices=old_indices,
        dropped_old_indices=dropped_old_indices,
        skeleton_source=skeleton_source,
    )
    if err:
        return None, err
    rendered_reason = _strict_reject_rendered_output(output_text, record)
    if rendered_reason:
        return None, rendered_reason
    quality_warnings = _rendered_quality_warnings(output_text, record)

    input_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_labels
    if len(input_ids) != len(labels):
        return None, "internal length mismatch"
    if max_length > 0 and len(input_ids) > max_length:
        return None, f"sequence too long ({len(input_ids)} > {max_length})"

    fill_ids_seen: List[int] = []
    fill_positions: List[int] = []
    for i, tid in enumerate(input_ids):
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        if isinstance(tok, str) and tok.startswith("<|STRUCLIFT_FILL_"):
            fill_positions.append(i)
            fill_ids_seen.append(int(tid))
    if len(fill_positions) != len(slot_types):
        return None, f"fill positions mismatch ({len(fill_positions)} vs {len(slot_types)})"

    return {
        "sft_input_ids": input_ids,
        "sft_labels": labels,
        "slot_types": slot_types,
        "sft_fill_positions": fill_positions,
        "sft_slot_bb_indices": bbs,
        "sft_slot_output_spans": spans,
        "sft_teacher_targets": targets,
        "sft_teacher_target_sources": target_sources,
        "sft_v10_dropped_empty_stmt_old_indices": sorted(dropped_old_indices),
        "sft_v10_quality_warnings": quality_warnings,
        "sft_v10_quality_bucket": _quality_bucket(quality_warnings),
        "sft_teacher_slot_supervise": [True] * len(slot_types),
        "sft_body_slot_old_indices": old_indices,
        "sft_skeleton_text_pretty": skeleton,
        "sft_prompt_text_pretty": prompt,
        "sft_output_text_pretty": output_text,
        "sft_prompt_token_len": len(prompt_ids),
        "sft_output_token_len": len(output_ids),
        "sft_prompt_layout": "v10_body_slots_prompt_then_full_output",
        "sft_masked_supervised_inputs": False,
        "sft_v10_from": "v9_repack",
        "sft_v10_binary_context_before_output": True,
        "sft_v10_body_only": True,
        "sft_v10_loss_on_output_slot_spans": True,
        "sft_binary_context_lines": len(ctx.splitlines()) - 2,
        "sft_binary_context_readable_asm_in_prompt": True,
        "sft_binary_context_asm_max_total_lines": int(asm_max_total_lines),
        "sft_binary_context_asm_max_lines_per_bb": int(asm_max_lines_per_bb),
    }, None


def _build_tokenized_v10_hybrid(
    record: Dict[str, Any],
    tokenizer: Any,
    *,
    system: str,
    name_for_prompt: str,
    max_length: int,
    asm_max_total_lines: int,
    asm_max_lines_per_bb: int,
    slot_max_instr: int,
    allow_v9_teacher_fallback: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    fields, reason = _build_tokenized_v10(
        record,
        tokenizer,
        system=system,
        name_for_prompt=name_for_prompt,
        max_length=max_length,
        asm_max_total_lines=asm_max_total_lines,
        asm_max_lines_per_bb=asm_max_lines_per_bb,
        slot_max_instr=slot_max_instr,
        allow_v9_teacher_fallback=allow_v9_teacher_fallback,
        skeleton_source="regenerate",
    )
    if fields is not None:
        return fields, None

    recoverable = (
        str(reason or "").startswith("rendered_")
        or str(reason or "").startswith("missing source-derived target")
        or str(reason or "") == "no supervised body slots"
    )
    if not recoverable:
        return None, reason

    source_fields, source_reason = _build_tokenized_v10(
        record,
        tokenizer,
        system=system,
        name_for_prompt=name_for_prompt,
        max_length=max_length,
        asm_max_total_lines=asm_max_total_lines,
        asm_max_lines_per_bb=asm_max_lines_per_bb,
        slot_max_instr=slot_max_instr,
        allow_v9_teacher_fallback=False,
        skeleton_source="source_ast",
    )
    if source_fields is not None:
        source_fields["sft_v10_skeleton_fallback_from"] = "regenerate"
        source_fields["sft_v10_skeleton_fallback_reason"] = reason
        source_fields["sft_v10_from"] = "v9_repack_source_ast_fallback"
        return source_fields, None
    return None, f"{reason}; source_ast_fallback_failed: {source_reason}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--save-tokenizer", type=Path, default=None)
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--asm-max-total-lines", type=int, default=160)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=16)
    ap.add_argument("--slot-max-instr", type=int, default=12)
    ap.add_argument(
        "--allow-v9-teacher-fallback",
        action="store_true",
        help="Allow old V9 teacher targets when source-derived GT extraction fails.",
    )
    ap.add_argument(
        "--skeleton-source",
        choices=["regenerate", "v9", "source_ast", "hybrid"],
        default="regenerate",
        help="Use CFG skeleton, old V9 skeleton, source-AST skeleton, or CFG with source-AST fallback.",
    )
    ap.add_argument(
        "--prompt-func-name",
        choices=["func", "real"],
        default="func",
        help="Use generic func in prompt skeleton or the real function name.",
    )
    ap.add_argument("--keep-skipped-log", type=Path, default=None)
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(inp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fill_tokens = [_fill_token(i) for i in range(int(args.max_fill_tokens))]
    tokenizer.add_tokens(fill_tokens)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.save_tokenizer is not None:
        tok_out = args.save_tokenizer.expanduser().resolve()
        tok_out.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tok_out)
        logger.info("saved tokenizer: %s", tok_out)

    skipped: List[Dict[str, Any]] = []
    quality_bucket_counts: Counter[str] = Counter()
    quality_warning_counts: Counter[str] = Counter()
    n_in = n_out = 0
    total_len = total_sup = 0
    with outp.open("w", encoding="utf-8") as wf:
        for line_no, rec in _load_jsonl(inp):
            n_in += 1
            name_for_prompt = (
                str(rec.get("func_name") or "func")
                if args.prompt_func_name == "real"
                else "func"
            )
            if str(args.skeleton_source) == "hybrid":
                fields, reason = _build_tokenized_v10_hybrid(
                    rec,
                    tokenizer,
                    system=DEFAULT_SYSTEM,
                    name_for_prompt=name_for_prompt,
                    max_length=int(args.max_length),
                    asm_max_total_lines=int(args.asm_max_total_lines),
                    asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                    slot_max_instr=int(args.slot_max_instr),
                    allow_v9_teacher_fallback=bool(args.allow_v9_teacher_fallback),
                )
            else:
                fields, reason = _build_tokenized_v10(
                    rec,
                    tokenizer,
                    system=DEFAULT_SYSTEM,
                    name_for_prompt=name_for_prompt,
                    max_length=int(args.max_length),
                    asm_max_total_lines=int(args.asm_max_total_lines),
                    asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                    slot_max_instr=int(args.slot_max_instr),
                    allow_v9_teacher_fallback=bool(args.allow_v9_teacher_fallback),
                    skeleton_source=str(args.skeleton_source),
                )
            if fields is None:
                skipped.append({
                    "line": line_no,
                    "func_name": rec.get("func_name"),
                    "reason": reason,
                })
                continue
            out = dict(rec)
            out.update(fields)
            wf.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            n_out += 1
            total_len += len(fields["sft_input_ids"])
            total_sup += sum(1 for x in fields["sft_labels"] if int(x) != -100)
            quality_bucket_counts[str(fields.get("sft_v10_quality_bucket") or "unknown")] += 1
            for warning in fields.get("sft_v10_quality_warnings") or []:
                quality_warning_counts[str(warning)] += 1

    logger.info(
        "DONE v10 repack: input=%d output=%d skipped=%d avg_len=%.1f avg_supervised=%.1f -> %s",
        n_in,
        n_out,
        len(skipped),
        total_len / max(n_out, 1),
        total_sup / max(n_out, 1),
        outp,
    )
    if quality_bucket_counts:
        logger.info(
            "quality buckets: %s",
            " ".join(f"{k}={v}" for k, v in quality_bucket_counts.most_common()),
        )
    if quality_warning_counts:
        logger.info(
            "top quality warnings: %s",
            " ".join(f"{k}={v}" for k, v in quality_warning_counts.most_common(8)),
        )

    if args.keep_skipped_log:
        skip_path = args.keep_skipped_log.expanduser().resolve()
        skip_path.parent.mkdir(parents=True, exist_ok=True)
        with skip_path.open("w", encoding="utf-8") as f:
            for item in skipped:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("skipped log: %s", skip_path)


if __name__ == "__main__":
    main()

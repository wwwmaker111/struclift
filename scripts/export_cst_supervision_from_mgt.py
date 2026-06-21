#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export binary-only CST/slot supervision from source AST + M_gt.

This script is the first bridge between the current alignment objective and the
actual inference need:

    training-time source AST + M_gt  ->  binary-only CST / slot labels

It deliberately does not use model predictions.  It converts existing dataset
supervision into labels that a future binary-only CST predictor can learn:

* control nodes: type, source span, depth, condition BBs, region BBs
* branch/block slots: then/else/loop/switch/body BB groups
* per-BB ownership: primary slot/control labels for later classification

At inference time these labels are unavailable; they are training targets only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import export_modulea_region_control_skeleton_100 as region_export  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402


CONTROL_NODE_TYPES = {
    "if_statement": "if",
    "while_statement": "loop",
    "for_statement": "loop",
    "do_statement": "loop",
    "switch_statement": "switch",
}

LOOP_NODE_TYPES = {"while_statement", "for_statement", "do_statement"}
STMT_TYPES = set(getattr(repack, "STATEMENT_TYPES", set()))


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _node_type(node: Any) -> str:
    return repack._node_type(node)


def _node_span(node: Any) -> Tuple[int, int]:
    try:
        lo = int(node.start_point[0]) + 1
        hi = int(node.end_point[0]) + 1
        return min(lo, hi), max(lo, hi)
    except Exception:
        return 0, 0


def _node_depth(node: Any) -> int:
    depth = 0
    cur = getattr(node, "parent", None)
    while cur is not None:
        if _node_type(cur) in STMT_TYPES:
            depth += 1
        cur = getattr(cur, "parent", None)
    return depth


def _direct_stmt_children(node: Any) -> List[Any]:
    return list(repack._direct_statement_children(node) or [])


def _branch_nodes(branch: Any) -> List[Any]:
    return list(repack._source_ast_branch_statement_nodes(branch) or [])


def _field(node: Any, *names: str) -> Any:
    return repack._field_child(node, *names)


def _node_text(source_b: bytes, node: Any) -> str:
    return repack._node_text(source_b, node)


def _clean_decl_text(text: str) -> str:
    x = re.sub(r"/\*.*?\*/", " ", str(text or ""), flags=re.S)
    x = re.sub(r"//.*$", " ", x, flags=re.M)
    x = re.sub(r"\bATTRIBUTE_[A-Z0-9_]+\s*(\([^)]*\))?", " ", x)
    x = re.sub(r"\b_GL_ATTRIBUTE_[A-Z0-9_]+\s*(\([^)]*\))?", " ", x)
    x = re.sub(r"\b_[A-Z0-9_]*ATTRIBUTE[A-Z0-9_]*\s*(\([^)]*\))?", " ", x)
    x = re.sub(r"\b__attribute__\s*\(\(.*?\)\)", " ", x)
    x = re.sub(r"\b(__THROW|__nonnull\s*\([^)]*\)|__wur|__pure|__const)\b", " ", x)
    x = re.sub(r"^\s*#.*$", " ", x, flags=re.M)
    return re.sub(r"\s+", " ", x).strip()


def _function_decl_line(row: Dict[str, Any], source_b: bytes, func: Any) -> str:
    if func is None:
        return "int func(...)"
    body = _field(func, "body")
    try:
        if body is not None:
            raw = source_b[int(func.start_byte) : int(body.start_byte)].decode("utf-8", errors="replace")
        else:
            raw = _node_text(source_b, func).rsplit("{", 1)[0]
    except Exception:
        raw = ""
    raw = _clean_decl_text(raw).rstrip("{").strip()
    fn = str(row.get("func_name") or "").strip()
    if fn:
        raw = re.sub(rf"\b{re.escape(fn)}\b", "func", raw)
    if not raw or "(" not in raw or ")" not in raw:
        return "int func(...)"
    return raw


def _clean_cond_text(record: Dict[str, Any], node: Any) -> str:
    try:
        return repack._source_ast_condition_text(record, node)
    except Exception:
        return ""


def _source_function_bounds(row: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    lt = repack._line_to_text(row.get("source_file"))
    if not lt:
        return None
    try:
        return repack._function_body_line_range(row, lt)
    except Exception:
        return None


def _find_function_node(row: Dict[str, Any]) -> Tuple[bytes, Any]:
    bounds = _source_function_bounds(row)
    if bounds is None:
        return b"", None
    _src, source_b, tree = repack._source_tree_for_record(row)
    if tree is None:
        return b"", None
    lo, hi = bounds
    best = None
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if _node_type(node) == "function_definition":
            n_lo, n_hi = _node_span(node)
            if n_lo <= lo <= n_hi and n_lo <= hi <= n_hi:
                if best is None:
                    best = node
                else:
                    b_lo, b_hi = _node_span(best)
                    if (n_hi - n_lo) < (b_hi - b_lo):
                        best = node
        for child in reversed(getattr(node, "children", None) or []):
            stack.append(child)
    return source_b, best


def _alignment_matrix(row: Dict[str, Any]) -> Optional[List[List[float]]]:
    n_bb = effect._n_bb(row)
    raw = row.get("alignment_matrix")
    if not isinstance(raw, list) or len(raw) != n_bb or not raw:
        return None
    if not isinstance(raw[0], list) or not raw[0]:
        return None
    n_s = len(raw[0])
    out: List[List[float]] = []
    try:
        for r in raw:
            if not isinstance(r, list) or len(r) != n_s:
                return None
            out.append([float(x or 0.0) for x in r])
    except Exception:
        return None
    return out


def _statement_columns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = repack._source_statement_table(row)
    if not rows:
        rows = effect._source_rows(row)
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows or []):
        rr = dict(r)
        rr.setdefault("alignment_col", rr.get("id", i))
        out.append(rr)
    return out


def _row_span(r: Dict[str, Any]) -> Tuple[int, int]:
    try:
        lo = int(r.get("start_line") or 0)
        hi = int(r.get("end_line") or lo)
        return min(lo, hi), max(lo, hi)
    except Exception:
        return 0, 0


def _span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    alo, ahi = a
    blo, bhi = b
    if alo <= 0 or blo <= 0:
        return 0
    return max(0, min(ahi, bhi) - max(alo, blo) + 1)


def _cols_for_node(
    node: Any,
    stmt_rows: Sequence[Dict[str, Any]],
    *,
    type_filter: Optional[Set[str]] = None,
) -> List[int]:
    span = _node_span(node)
    ntype = _node_type(node)
    scored: List[Tuple[int, int, int]] = []
    for r in stmt_rows:
        rtype = str(r.get("type") or "")
        if type_filter is not None and rtype not in type_filter:
            continue
        ov = _span_overlap(span, _row_span(r))
        if ov <= 0:
            continue
        col = r.get("alignment_col", r.get("id"))
        try:
            c = int(col)
        except Exception:
            continue
        type_bonus = 1 if rtype == ntype else 0
        row_width = max(1, _row_span(r)[1] - _row_span(r)[0] + 1)
        scored.append((type_bonus, ov, -row_width, c))
    scored.sort(reverse=True)
    return sorted({c for _tb, _ov, _rw, c in scored})


def _bbs_for_cols(M: Sequence[Sequence[float]], cols: Sequence[int], threshold: float) -> List[int]:
    wanted = [int(c) for c in cols if c >= 0]
    if not wanted:
        return []
    out: List[int] = []
    for bb, row in enumerate(M):
        mass = 0.0
        for c in wanted:
            if c < len(row):
                mass += float(row[c])
        if mass > threshold:
            out.append(int(bb))
    return out


def _bbs_for_node(
    row: Dict[str, Any],
    node: Any,
    M: Sequence[Sequence[float]],
    stmt_rows: Sequence[Dict[str, Any]],
    threshold: float,
    *,
    type_filter: Optional[Set[str]] = None,
) -> List[int]:
    cols = _cols_for_node(node, stmt_rows, type_filter=type_filter)
    bbs = _bbs_for_cols(M, cols, threshold)
    if bbs:
        return bbs
    lo, hi = _node_span(node)
    try:
        return list(repack._bbs_for_source_lines(row, range(lo, hi + 1)) or [])
    except Exception:
        return []


def _bbs_for_lines(row: Dict[str, Any], lines: Sequence[int]) -> List[int]:
    try:
        return list(repack._bbs_for_source_lines(row, lines) or [])
    except Exception:
        return []


def _node_lines(node: Any) -> List[int]:
    lo, hi = _node_span(node)
    if lo <= 0 or hi <= 0:
        return []
    return list(range(lo, hi + 1))


def _subtree_bbs(
    row: Dict[str, Any],
    node: Any,
    M: Sequence[Sequence[float]],
    stmt_rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> List[int]:
    if node is None:
        return []
    out: Set[int] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if _node_type(cur) in STMT_TYPES:
            out.update(_bbs_for_node(row, cur, M, stmt_rows, threshold))
        for child in reversed(getattr(cur, "children", None) or []):
            stack.append(child)
    if out:
        return sorted(out)
    return _bbs_for_lines(row, _node_lines(node))


def _branch_bbs(
    row: Dict[str, Any],
    nodes: Sequence[Any],
    M: Sequence[Sequence[float]],
    stmt_rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> List[int]:
    return _union(_subtree_bbs(row, n, M, stmt_rows, threshold) for n in nodes)


def _condition_bbs(
    row: Dict[str, Any],
    node: Any,
    M: Sequence[Sequence[float]],
    stmt_rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> List[int]:
    ntype = _node_type(node)
    cond = _field(node, "condition")
    bbs: List[int] = []
    if cond is not None:
        bbs = _bbs_for_lines(row, _node_lines(cond))
    if not bbs:
        lo, _hi = _node_span(node)
        bbs = _bbs_for_lines(row, [lo])
    if not bbs:
        bbs = _bbs_for_node(row, node, M, stmt_rows, threshold, type_filter={ntype})
    return sorted(set(int(x) for x in bbs))


def _union(xs: Iterable[Sequence[int]]) -> List[int]:
    out: Set[int] = set()
    for seq in xs:
        out.update(int(x) for x in seq)
    return sorted(out)


def _subtract(a: Sequence[int], b: Sequence[int]) -> List[int]:
    bb = set(int(x) for x in b)
    return [int(x) for x in a if int(x) not in bb]


def _node_kind(node: Any) -> str:
    return CONTROL_NODE_TYPES.get(_node_type(node), "stmt")


def _stmt_preview(source_b: bytes, node: Any, max_len: int = 160) -> str:
    text = re.sub(r"\s+", " ", _node_text(source_b, node)).strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _source_control_count(row: Dict[str, Any], source_control_filter: str) -> int:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    outline = effect._source_control_outline(
        rows,
        active_lines=active,
        source_control_filter=source_control_filter,
    )
    return len(outline)


def _ast_control_outline(root_body: Any) -> List[Dict[str, Any]]:
    outline: List[Dict[str, Any]] = []
    stack: List[int] = []

    def walk_nodes(nodes: Sequence[Any]) -> None:
        for node in nodes:
            ntype = _node_type(node)
            if ntype == "compound_statement":
                walk_nodes(_direct_stmt_children(node))
                continue
            kind = _node_kind(node)
            if kind in {"if", "loop", "switch"}:
                cid = len(outline)
                parent = stack[-1] if stack else None
                span = _node_span(node)
                outline.append({
                    "control_id": cid,
                    "kind": kind,
                    "parent_control": parent,
                    "depth": len(stack),
                    "source_node_type": ntype,
                    "source_span": [int(span[0]), int(span[1])],
                })
                stack.append(cid)
                if kind == "if":
                    walk_nodes(_branch_nodes(_field(node, "consequence")))
                    alternative = _field(node, "alternative")
                    if alternative is not None:
                        walk_nodes(_branch_nodes(alternative))
                elif kind in {"loop", "switch"}:
                    walk_nodes(_branch_nodes(_field(node, "body")))
                stack.pop()

    walk_nodes(_direct_stmt_children(root_body))
    return outline


def _outline_pairs_from_records(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for r in records:
        kind = str(r.get("kind") or "")
        if kind in {"if", "loop", "switch"}:
            out.append((kind, int(r.get("depth") or 0)))
    return out


def _serialize_teacher(
    row: Dict[str, Any],
    source_b: bytes,
    func: Any,
    root_body: Any,
    M: Sequence[Sequence[float]],
    stmt_rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    controls: List[Dict[str, Any]] = []
    slots: List[Dict[str, Any]] = []
    control_stack: List[int] = []
    claimed_primary_bbs: Set[int] = set()

    def collect_control_condition_bbs(node: Any) -> Set[int]:
        out: Set[int] = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if _node_kind(cur) in {"if", "loop", "switch"}:
                out.update(_condition_bbs(row, cur, M, stmt_rows, threshold))
            for child in reversed(getattr(cur, "children", None) or []):
                stack.append(child)
        return {int(x) for x in out if int(x) >= 0}

    all_condition_bbs = collect_control_condition_bbs(root_body)

    def slot_ref(slot_id: int) -> str:
        return f"slot_{int(slot_id):03d}"

    def active_condition_bbs() -> List[int]:
        out: Set[int] = set()
        for cid in control_stack:
            if 0 <= int(cid) < len(controls):
                out.update(int(x) for x in controls[int(cid)].get("condition_bbs") or [])
        return sorted(out)

    def add_slot(
        slot_type: str,
        role: str,
        primary_bbs: Sequence[int],
        owner_control: Optional[int],
        source_span: Tuple[int, int],
        text: str = "",
        *,
        context_bbs: Optional[Sequence[int]] = None,
        render_depth: int = 1,
        trainable: bool = True,
        claim_primary: bool = True,
    ) -> int:
        sid = len(slots)
        primary = sorted(set(int(x) for x in primary_bbs))
        context = sorted(set(int(x) for x in (context_bbs if context_bbs is not None else primary)))
        if trainable and slot_type == "FILL_BLOCK":
            primary = [
                int(x) for x in primary
                if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
            ]
        elif trainable and slot_type == "FILL_COND":
            unclaimed = [int(x) for x in primary if int(x) not in claimed_primary_bbs]
            # Collapsed source controls can share one binary header BB.  Keep a
            # non-empty condition target in that rare case, but avoid letting
            # block slots claim condition BBs later.
            primary = unclaimed if unclaimed else primary
        slots.append({
            "slot_id": sid,
            "slot_type": slot_type,
            "role": role,
            "owner_control": owner_control,
            # Backward-compatible alias.  New code must use primary_bbs for
            # training labels and context_bbs for Module-C reference.
            "bbs": primary,
            "primary_bbs": primary,
            "context_bbs": context,
            "render_depth": int(max(0, render_depth)),
            "trainable": bool(trainable),
            "source_span": [int(source_span[0]), int(source_span[1])],
            "source_text_preview": text,
        })
        if trainable and claim_primary:
            claimed_primary_bbs.update(int(x) for x in primary)
        return sid

    preamble_sid = add_slot(
        "FILL_BLOCK",
        "preamble",
        [],
        None,
        (0, 0),
        "reserved preamble/context slot",
        context_bbs=[],
        render_depth=1,
        trainable=False,
        claim_primary=False,
    )

    def control_bbs(node: Any) -> List[int]:
        return _subtree_bbs(row, node, M, stmt_rows, threshold)

    def emit_plain_slot(nodes: Sequence[Any], indent: int, role: str) -> List[str]:
        real = [n for n in nodes if _node_type(n) != "compound_statement"]
        if not real:
            return []
        span_lo = min((_node_span(n)[0] for n in real if _node_span(n)[0] > 0), default=0)
        span_hi = max((_node_span(n)[1] for n in real if _node_span(n)[1] > 0), default=0)
        context = _branch_bbs(row, real, M, stmt_rows, threshold)
        primary = _subtract(context, active_condition_bbs())
        primary = [
            int(x) for x in primary
            if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
        ]
        if not primary:
            return []
        sid = add_slot(
            "FILL_BLOCK",
            role,
            primary,
            control_stack[-1] if control_stack else None,
            (span_lo, span_hi),
            " ".join(_stmt_preview(source_b, n, max_len=80) for n in real[:3]),
            context_bbs=context,
            render_depth=indent,
            trainable=bool(primary),
        )
        pad = "    " * indent
        return [f"{pad}<FILL_BLOCK:{slot_ref(sid)}>;  /* primary {primary} context {context} */"]

    def emit_block(nodes: Sequence[Any], indent: int, branch_role: Optional[str] = None) -> List[str]:
        lines: List[str] = []
        pending_plain: List[Any] = []
        role = branch_role or ("body" if control_stack else "stmt")
        for node in nodes:
            if _node_type(node) == "compound_statement":
                for child in _direct_stmt_children(node):
                    if _node_kind(child) in {"if", "loop", "switch"}:
                        lines.extend(emit_plain_slot(pending_plain, indent, role))
                        pending_plain = []
                        lines.extend(emit_stmt(child, indent))
                    else:
                        pending_plain.append(child)
                continue
            if _node_kind(node) in {"if", "loop", "switch"}:
                lines.extend(emit_plain_slot(pending_plain, indent, role))
                pending_plain = []
                lines.extend(emit_stmt(node, indent))
            else:
                pending_plain.append(node)
        lines.extend(emit_plain_slot(pending_plain, indent, role))
        return lines

    def emit_stmt(node: Any, indent: int) -> List[str]:
        pad = "    " * indent
        ntype = _node_type(node)
        kind = _node_kind(node)
        span = _node_span(node)

        if ntype == "compound_statement":
            return emit_block(_direct_stmt_children(node), indent)

        if kind in {"if", "loop", "switch"}:
            cid = len(controls)
            parent = control_stack[-1] if control_stack else None
            bbs_all = control_bbs(node)
            cond_bbs = _condition_bbs(row, node, M, stmt_rows, threshold)
            cond_sid = add_slot(
                "FILL_COND",
                "cond",
                cond_bbs,
                cid,
                span,
                _clean_cond_text(row, node),
                render_depth=indent,
                claim_primary=True,
            )
            info: Dict[str, Any] = {
                "control_id": cid,
                "kind": kind,
                "source_node_type": ntype,
                "parent_control": parent,
                "source_span": [int(span[0]), int(span[1])],
                "depth": int(_node_depth(node)),
                "condition_slot": cond_sid,
                "condition_bbs": sorted(set(cond_bbs)),
                "region_bbs": sorted(set(bbs_all)),
                "branch_slots": [],
                "branch_bbs": {},
                "source_text_preview": _stmt_preview(source_b, node),
            }
            controls.append(info)
            control_stack.append(cid)
            if kind == "if":
                consequence = _field(node, "consequence")
                then_nodes = _branch_nodes(consequence)
                then_context = _branch_bbs(row, then_nodes, M, stmt_rows, threshold)
                then_primary = _subtract(then_context, cond_bbs)
                then_primary = [
                    int(x) for x in then_primary
                    if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
                ]
                info["branch_bbs"]["then"] = then_primary
                info["branch_context_bbs"] = {"then": then_context}
                out = [f"{pad}if (<FILL_COND:{slot_ref(cond_sid)}>) {{"]
                then_lines = emit_block(then_nodes, indent + 1, "then")
                if then_lines:
                    out.extend(then_lines)
                else:
                    out.append(f"{pad}    /* empty/unmapped then branch: primary {then_primary} context {then_context} */")
                alternative = _field(node, "alternative")
                else_nodes = _branch_nodes(alternative)
                if else_nodes:
                    else_context = _branch_bbs(row, else_nodes, M, stmt_rows, threshold)
                    else_primary = _subtract(
                        else_context,
                        cond_bbs + then_primary,
                    )
                    else_primary = [
                        int(x) for x in else_primary
                        if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
                    ]
                    info["branch_bbs"]["else"] = else_primary
                    info.setdefault("branch_context_bbs", {})["else"] = else_context
                    out.append(f"{pad}}} else {{")
                    else_lines = emit_block(else_nodes, indent + 1, "else")
                    if else_lines:
                        out.extend(else_lines)
                    else:
                        out.append(f"{pad}    /* empty/unmapped else branch: primary {else_primary} context {else_context} */")
                out.append(f"{pad}}}")
                control_stack.pop()
                return out

            if kind == "loop":
                body = _field(node, "body")
                body_nodes = _branch_nodes(body)
                body_context = _branch_bbs(row, body_nodes, M, stmt_rows, threshold)
                body_primary = _subtract(body_context, cond_bbs)
                body_primary = [
                    int(x) for x in body_primary
                    if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
                ]
                info["branch_bbs"]["loop_body"] = body_primary
                info["branch_context_bbs"] = {"loop_body": body_context}
                body_lines = emit_block(body_nodes, indent + 1, "loop_body")
                head = "do" if ntype == "do_statement" else "while"
                if head == "do":
                    out = [
                        f"{pad}do {{",
                        *(body_lines or [f"{pad}    /* empty/unmapped loop body: primary {body_primary} context {body_context} */"]),
                        f"{pad}}} while (<FILL_COND:{slot_ref(cond_sid)}>);",
                    ]
                else:
                    out = [
                        f"{pad}while (<FILL_COND:{slot_ref(cond_sid)}>) {{",
                        *(body_lines or [f"{pad}    /* empty/unmapped loop body: primary {body_primary} context {body_context} */"]),
                        f"{pad}}}",
                    ]
                control_stack.pop()
                return out

            body = _field(node, "body")
            body_nodes = _branch_nodes(body)
            body_context = _branch_bbs(row, body_nodes, M, stmt_rows, threshold)
            body_primary = _subtract(body_context, cond_bbs)
            body_primary = [
                int(x) for x in body_primary
                if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
            ]
            info["branch_bbs"]["switch_body"] = body_primary
            info["branch_context_bbs"] = {"switch_body": body_context}
            body_lines = emit_block(body_nodes, indent + 1, "switch_body")
            out = [
                f"{pad}switch (<FILL_COND:{slot_ref(cond_sid)}>) {{",
                *(body_lines or [f"{pad}    /* empty/unmapped switch body: primary {body_primary} context {body_context} */"]),
                f"{pad}}}",
            ]
            control_stack.pop()
            return out

        bbs = _bbs_for_node(row, node, M, stmt_rows, threshold)
        primary = _subtract(bbs, active_condition_bbs())
        primary = [
            int(x) for x in primary
            if int(x) not in all_condition_bbs and int(x) not in claimed_primary_bbs
        ]
        if not primary:
            return []
        sid = add_slot(
            "FILL_BLOCK",
            "stmt",
            primary,
            control_stack[-1] if control_stack else None,
            span,
            _stmt_preview(source_b, node),
            context_bbs=bbs,
            render_depth=indent,
            trainable=bool(primary),
        )
        return [f"{pad}<FILL_BLOCK:{slot_ref(sid)}>;  /* primary {primary} context {bbs} */"]

    body_lines = emit_block(_direct_stmt_children(root_body), 1)
    preamble_line = f"    <FILL_BLOCK:{slot_ref(preamble_sid)}>;  /* reserved preamble/context */"
    text = f"{_function_decl_line(row, source_b, func)} {{\n" + "\n".join([preamble_line] + body_lines) + "\n}"

    for c in controls:
        cid = int(c["control_id"])
        desc: Set[int] = set(int(x) for x in c.get("condition_bbs") or [])
        for s in slots:
            owner = s.get("owner_control")
            if owner is None:
                continue
            try:
                owner_i = int(owner)
            except Exception:
                continue
            cur: Optional[int] = owner_i
            while cur is not None:
                if cur == cid:
                    desc.update(int(x) for x in s.get("primary_bbs") or [])
                    break
                parent = controls[cur].get("parent_control") if 0 <= cur < len(controls) else None
                cur = int(parent) if parent is not None else None
        c["region_primary_bbs"] = sorted(desc)
    return text, controls, slots


def _per_bb_labels(n_bb: int, controls: Sequence[Dict[str, Any]], slots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    owned_slots: Dict[int, List[int]] = defaultdict(list)
    owned_controls: Dict[int, List[int]] = defaultdict(list)
    roles: Dict[int, List[str]] = defaultdict(list)
    kinds: Dict[int, List[str]] = defaultdict(list)
    for slot in slots:
        if not bool(slot.get("trainable", True)):
            continue
        sid = int(slot["slot_id"])
        role = str(slot.get("role") or "")
        owner = slot.get("owner_control")
        for bb in slot.get("primary_bbs", slot.get("bbs")) or []:
            owned_slots[int(bb)].append(sid)
            roles[int(bb)].append(role)
            if owner is not None:
                owned_controls[int(bb)].append(int(owner))
    for ctrl in controls:
        cid = int(ctrl["control_id"])
        kind = str(ctrl.get("kind") or "")
        for bb in ctrl.get("region_bbs") or []:
            kinds[int(bb)].append(kind)
            owned_controls[int(bb)].append(cid)
    out: List[Dict[str, Any]] = []
    for bb in range(n_bb):
        out.append({
            "bb": bb,
            "primary_slot": owned_slots.get(bb, [None])[0],
            "slots": sorted(set(owned_slots.get(bb, []))),
            "controls": sorted(set(owned_controls.get(bb, []))),
            "roles": sorted(set(roles.get(bb, []))),
            "control_kinds": sorted(set(kinds.get(bb, []))),
        })
    return out


def _supervision_stats(controls: Sequence[Dict[str, Any]], slots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    branch_slots = [s for s in slots if str(s.get("slot_type") or "") == "FILL_BRANCH"]
    cond_slots = [s for s in slots if str(s.get("slot_type") or "") == "FILL_COND"]
    fill_blocks = [s for s in slots if str(s.get("slot_type") or "") == "FILL_BLOCK"]
    trainable_slots = [s for s in slots if bool(s.get("trainable", True))]
    preamble_slots = [s for s in slots if str(s.get("role") or "") == "preamble"]
    empty_branch_controls = 0
    empty_branch_refs = []
    for c in controls:
        bad_roles = [
            role for role, bbs in (c.get("branch_bbs") or {}).items()
            if not bbs
        ]
        if bad_roles:
            empty_branch_controls += 1
            empty_branch_refs.append({
                "control_id": c.get("control_id"),
                "kind": c.get("kind"),
                "roles": bad_roles,
            })
    return {
        "n_controls": len(controls),
        "n_slots": len(slots),
        "n_cond_slots": len(cond_slots),
        "n_branch_slots": len(branch_slots),
        "n_fill_block_slots": len(fill_blocks),
        "n_trainable_slots": len(trainable_slots),
        "n_preamble_slots": len(preamble_slots),
        "slot_primary_empty": sum(1 for s in trainable_slots if not s.get("primary_bbs", s.get("bbs"))),
        "slot_context_empty": sum(1 for s in trainable_slots if not s.get("context_bbs", s.get("bbs"))),
        "cond_primary_empty": sum(1 for s in cond_slots if bool(s.get("trainable", True)) and not s.get("primary_bbs", s.get("bbs"))),
        "branch_primary_empty": sum(1 for s in branch_slots if bool(s.get("trainable", True)) and not s.get("primary_bbs", s.get("bbs"))),
        "fill_block_primary_empty": sum(1 for s in fill_blocks if bool(s.get("trainable", True)) and not s.get("primary_bbs", s.get("bbs"))),
        "control_region_bbs_empty": sum(1 for c in controls if not c.get("region_primary_bbs", c.get("region_bbs"))),
        "controls_with_empty_branch": empty_branch_controls,
        "empty_branch_refs_first10": empty_branch_refs[:10],
    }


def _teacher_quality_reason(
    n_bb: int,
    controls: Sequence[Dict[str, Any]],
    slots: Sequence[Dict[str, Any]],
    skeleton: str,
    *,
    max_bb: int,
    max_controls: int,
    max_slots: int,
) -> str:
    if n_bb <= 0:
        return "bad_n_bb"
    if max_bb > 0 and n_bb > max_bb:
        return "too_many_bbs"
    if not controls:
        return "no_controls_after_projection"
    if max_controls > 0 and len(controls) > max_controls:
        return "too_many_controls"
    if max_slots > 0 and len(slots) > max_slots:
        return "too_many_slots"
    preamble = [s for s in slots if str(s.get("role") or "") == "preamble"]
    if len(preamble) != 1 or int(preamble[0].get("slot_id", -1)) != 0:
        return "bad_preamble"
    if "<FILL_BLOCK:slot_000>" not in skeleton:
        return "missing_preamble_in_skeleton"
    trainable = [s for s in slots if bool(s.get("trainable", True))]
    if any(not s.get("primary_bbs", s.get("bbs")) for s in trainable):
        return "empty_trainable_primary"
    if any(not s.get("context_bbs", s.get("bbs")) for s in trainable):
        return "empty_trainable_context"
    for s in trainable:
        for bb in s.get("primary_bbs", s.get("bbs")) or []:
            if not isinstance(bb, int) or bb < 0 or bb >= n_bb:
                return "primary_bb_oob"
    for c in controls:
        if not c.get("condition_bbs"):
            return "empty_condition_bbs"
        if not c.get("region_primary_bbs", c.get("region_bbs")):
            return "empty_control_region_primary"
    return "ok"


def _eligible(row: Dict[str, Any], source_control_filter: str, min_source_controls: int) -> Tuple[bool, str]:
    if not region_export.is_valid_bin(row):
        return False, "bad_binary_fields"
    if _alignment_matrix(row) is None:
        return False, "missing_alignment_matrix"
    if _source_control_count(row, source_control_filter) < min_source_controls:
        return False, "no_or_below_min_source_control"
    source_b, func = _find_function_node(row)
    if not source_b or func is None:
        return False, "source_ast_unavailable"
    body = _field(func, "body")
    if body is None:
        return False, "function_body_unavailable"
    return True, "ok"


def export(args: argparse.Namespace) -> None:
    jsonl = Path(args.jsonl).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    skipped: Counter[str] = Counter()
    quality_skipped: Counter[str] = Counter()
    scanned = 0

    records: List[Dict[str, Any]] = []
    write_errors: Counter[str] = Counter()
    with out_jsonl.open("w", encoding="utf-8") as jf:
        for line_no, row in _iter_jsonl(jsonl, int(args.limit)):
            scanned += 1
            ok, reason = _eligible(row, str(args.source_control_filter), int(args.min_source_controls))
            if not ok:
                skipped[reason] += 1
                continue
            try:
                M = _alignment_matrix(row)
                assert M is not None
                source_b, func = _find_function_node(row)
                body = _field(func, "body")
                stmt_rows = _statement_columns(row)
                ast_outline = _ast_control_outline(body)
                teacher_text, controls, slots = _serialize_teacher(
                    row,
                    source_b,
                    func,
                    body,
                    M,
                    stmt_rows,
                    float(args.align_threshold),
                )
                n_bb = effect._n_bb(row)
                stats = _supervision_stats(controls, slots)
                q_reason = _teacher_quality_reason(
                    n_bb,
                    controls,
                    slots,
                    teacher_text,
                    max_bb=int(args.max_bb),
                    max_controls=int(args.max_controls),
                    max_slots=int(args.max_slots),
                )
                if q_reason != "ok":
                    quality_skipped[q_reason] += 1
                    continue
                rec = {
                    "example": len(records),
                    "input_line": line_no,
                    "func_name": row.get("func_name"),
                    "source_file": row.get("source_file"),
                    "n_bb": n_bb,
                    "n_src_cols": len(M[0]) if M else 0,
                    "pattern_labels": row.get("pattern_labels"),
                    "source_control_count": _source_control_count(row, str(args.source_control_filter)),
                    "source_ast_outline": ast_outline,
                    "source_ast_outline_pairs": _outline_pairs_from_records(ast_outline),
                    "teacher_outline_pairs": _outline_pairs_from_records(controls),
                    "teacher_skeleton": teacher_text,
                    "controls": controls,
                    "slots": slots,
                    "per_bb": _per_bb_labels(n_bb, controls, slots),
                    "stats": stats,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
                if int(args.num_examples) > 0 and len(records) >= int(args.num_examples):
                    break
            except Exception as e:
                write_errors[type(e).__name__] += 1
                if not args.skip_errors:
                    raise
        print(
            f"written_jsonl={len(records)} scanned={scanned} skipped={dict(skipped)} "
            f"quality_skipped={dict(quality_skipped)} errors={dict(write_errors)}",
            flush=True,
        )

    with out_txt.open("w", encoding="utf-8") as wf:
        wf.write("===== CST/SLOT SUPERVISION FROM M_gt + SOURCE AST =====\n")
        wf.write(f"input = {jsonl}\n")
        wf.write(f"out_jsonl = {out_jsonl}\n")
        wf.write(f"source_control_filter = {args.source_control_filter}\n")
        wf.write(f"align_threshold = {float(args.align_threshold)}\n")
        wf.write(f"scanned = {scanned}\n")
        wf.write(f"exported = {len(records)}\n")
        wf.write(f"skipped = {dict(skipped)}\n")
        wf.write(f"quality_skipped = {dict(quality_skipped)}\n")
        wf.write(f"errors = {dict(write_errors)}\n")
        wf.write(f"elapsed_sec = {time.time() - t0:.1f}\n\n")
        for rec in records[: int(args.preview_examples)]:
            wf.write("=" * 120 + "\n")
            wf.write(
                f"example={rec['example']} input_line={rec['input_line']} "
                f"func={rec.get('func_name')} n_bb={rec['n_bb']} "
                f"controls={rec['stats']['n_controls']} slots={rec['stats']['n_slots']}\n"
            )
            wf.write(f"source_file={rec.get('source_file')}\n")
            wf.write(f"stats={rec['stats']}\n")
            wf.write(f"source_ast_outline_pairs={rec.get('source_ast_outline_pairs')}\n")
            wf.write(f"teacher_outline_pairs={rec.get('teacher_outline_pairs')}\n")
            wf.write("\n-- CONTROLS --\n")
            for c in rec["controls"]:
                wf.write(
                    f"#{c['control_id']:02d} {c['kind']:6s} parent={c['parent_control']} "
                    f"span={c['source_span']} cond_bbs={c['condition_bbs']} "
                    f"region_bbs={c['region_bbs']} branches={c['branch_bbs']}\n"
                )
                wf.write(f"    src: {c.get('source_text_preview','')}\n")
            wf.write("\n-- SLOTS --\n")
            for s in rec["slots"]:
                wf.write(
                    f"slot_{s['slot_id']:03d} {s['slot_type']:10s} role={s['role']:12s} "
                    f"owner={s['owner_control']} trainable={s.get('trainable', True)} "
                    f"depth={s.get('render_depth')} "
                    f"primary={s.get('primary_bbs', s.get('bbs'))} "
                    f"context={s.get('context_bbs', s.get('bbs'))} span={s['source_span']}\n"
                )
            wf.write("\n-- TEACHER REGION-SLOT SKELETON --\n")
            wf.write(str(rec["teacher_skeleton"]).rstrip() + "\n\n")

    print(f"saved_jsonl = {out_jsonl}")
    print(f"saved_txt = {out_txt}")
    print(f"elapsed_total_sec = {time.time() - t0:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export binary-only CST/slot supervision from source AST + alignment_matrix(M_gt).",
    )
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--num-examples", type=int, default=100, help="0 means export all eligible samples")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source-control-filter", default="bb-start-line", choices=["raw", "bb-start-line", "bb-range"])
    ap.add_argument("--min-source-controls", type=int, default=1)
    ap.add_argument("--align-threshold", type=float, default=1e-8)
    ap.add_argument("--max-bb", type=int, default=40)
    ap.add_argument("--max-controls", type=int, default=10)
    ap.add_argument("--max-slots", type=int, default=45)
    ap.add_argument("--preview-examples", type=int, default=100)
    ap.add_argument("--skip-errors", action="store_true")
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()

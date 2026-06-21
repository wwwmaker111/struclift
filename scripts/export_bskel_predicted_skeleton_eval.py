#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export and audit B_skel predicted CST skeletons against AST teachers.

This script evaluates the inference-side path:

    frozen Module A -> trained BinaryCSTParser -> predicted control outline

It compares the predicted control outline with the AST-derived teacher from
``export_cst_supervision_from_mgt.py``.  The source AST is used only as the
stored teacher target; model inputs remain binary-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.train_module_b_skel import (  # noqa: E402
    BSkelTeacherDataset,
    cache_frozen_binary_features,
    collate_bskel,
    load_stage2_model,
    move_bskel_batch_to_device,
)
from struclift.config import StrucLiftConfig  # noqa: E402
from struclift.models.module_b_skel import (  # noqa: E402
    CTRL_ID_TO_NAME,
    CTRL_NAME_TO_ID,
    CTRL_NONE,
    CTRL_IF,
    CTRL_LOOP,
    CTRL_SWITCH,
    SLOT_ID_TO_NAME,
    SLOT_NONE,
    BinaryCSTParser,
    decode_slot_groups,
    decode_parent_tree,
    render_predicted_canonical_skeleton,
)


CONTROL_IDS = {CTRL_IF, CTRL_LOOP, CTRL_SWITCH}
FILL_SLOT_RE = re.compile(r"<FILL_(COND|BLOCK|BRANCH):slot_(\d+)>")


def _outline_kind_label(kind: str) -> str:
    kind = str(kind or "unknown")
    if kind == "loop":
        return "while"
    return kind


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def teacher_outline(teacher: Dict[str, Any]) -> List[Tuple[str, int]]:
    pairs = teacher.get("teacher_outline_pairs")
    if isinstance(pairs, list):
        out: List[Tuple[str, int]] = []
        for x in pairs:
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append((str(x[0]), int(x[1])))
        if out:
            return out

    controls = teacher.get("controls") if isinstance(teacher.get("controls"), list) else []

    def depth_of(idx: int) -> int:
        depth = 0
        seen = set()
        cur = idx
        while 0 <= cur < len(controls):
            if cur in seen:
                break
            seen.add(cur)
            parent = controls[cur].get("parent_control")
            if parent is None:
                break
            try:
                cur = int(parent)
            except Exception:
                break
            depth += 1
        return depth

    return [(str(c.get("kind") or "unknown"), depth_of(i)) for i, c in enumerate(controls)]


def predicted_outline_from_output(pred: Any) -> List[Tuple[str, int, int, int]]:
    ctrl = pred.ctrl_logits.argmax(dim=-1).detach().cpu().tolist()
    parents = decode_parent_tree(pred.parent_scores, pred.root_scores)
    n = len(ctrl)

    def depth_of(bb: int) -> int:
        depth = 0
        seen = set()
        cur = bb
        while 0 <= cur < n:
            if cur in seen:
                break
            seen.add(cur)
            parent = int(parents[cur])
            if parent >= n or parent < 0:
                break
            cur = parent
            depth += 1
        return depth

    out: List[Tuple[str, int, int, int]] = []
    for bb, cid in enumerate(ctrl):
        cid = int(cid)
        if cid in CONTROL_IDS:
            out.append((CTRL_ID_TO_NAME.get(cid, "unknown"), depth_of(bb), bb, int(parents[bb])))
    return out


def strip_bb(outline: Sequence[Tuple[str, int, int, int]]) -> List[Tuple[str, int]]:
    return [(str(kind), int(depth)) for kind, depth, _bb, _parent in outline]


def predicted_outline_from_outline_head(pred: Any) -> Optional[List[Tuple[str, int]]]:
    if not hasattr(pred, "outline_kind_logits") or pred.outline_kind_logits.numel() == 0:
        return None
    kind_ids = pred.outline_kind_logits.argmax(dim=-1)
    depth_ids = pred.outline_depth_logits.argmax(dim=-1)
    count_id: Optional[int] = None
    if hasattr(pred, "outline_count_logits") and pred.outline_count_logits.numel() > 0:
        count_logits = pred.outline_count_logits
        if count_logits.dim() == 2:
            count_logits = count_logits[0]
        count_id = int(count_logits.argmax(dim=-1).detach().cpu().item())
    if kind_ids.dim() == 2:
        kind_ids = kind_ids[0]
        depth_ids = depth_ids[0]
    out: List[Tuple[str, int]] = []
    kind_list = kind_ids.detach().cpu().tolist()
    depth_list = depth_ids.detach().cpu().tolist()
    if count_id is None:
        iter_pairs = zip(kind_list, depth_list)
    else:
        count_id = max(0, min(int(count_id), len(kind_list)))
        iter_pairs = zip(kind_list[:count_id], depth_list[:count_id])
    for k, d in iter_pairs:
        k = int(k)
        if count_id is None and k == CTRL_NONE:
            break
        out.append((CTRL_ID_TO_NAME.get(k, "unknown"), int(d)))
    return out


def predicted_slot_layout_from_head(pred: Any) -> Optional[List[Tuple[str, int]]]:
    """Decode the explicit slot-layout head as (FILL kind, render depth)."""
    if not hasattr(pred, "slot_layout_kind_logits") or pred.slot_layout_kind_logits.numel() == 0:
        return None
    kind_ids = pred.slot_layout_kind_logits.argmax(dim=-1)
    depth_ids = pred.slot_layout_depth_logits.argmax(dim=-1)
    count_id: Optional[int] = None
    if hasattr(pred, "slot_layout_count_logits") and pred.slot_layout_count_logits.numel() > 0:
        count_logits = pred.slot_layout_count_logits
        if count_logits.dim() == 2:
            count_logits = count_logits[0]
        count_id = int(count_logits.argmax(dim=-1).detach().cpu().item())
    if kind_ids.dim() == 2:
        kind_ids = kind_ids[0]
        depth_ids = depth_ids[0]
    kind_list = kind_ids.detach().cpu().tolist()
    depth_list = depth_ids.detach().cpu().tolist()
    if count_id is None:
        count = len(kind_list)
    else:
        count = max(0, min(int(count_id), len(kind_list)))
    out: List[Tuple[str, int]] = []
    for k, d in zip(kind_list[:count], depth_list[:count]):
        k = int(k)
        if count_id is None and k == SLOT_NONE:
            break
        name = SLOT_ID_TO_NAME.get(k, "none")
        if name == "none":
            continue
        out.append((str(name), int(d)))
    return out


def predicted_slot_grounding_from_head(
    pred: Any,
    slot_layout: Optional[Sequence[Tuple[str, int]]],
) -> List[Dict[str, Any]]:
    if slot_layout is None:
        return []
    out: List[Dict[str, Any]] = []
    for pos, (kind, depth) in enumerate(slot_layout):
        primary, context = _decode_slot_grounding(pred, slot_pos=int(pos))
        out.append(
            {
                "slot_pos": int(pos),
                "slot_id": int(pos) + 1,
                "slot_type": str(kind),
                "depth": int(depth),
                "primary_bbs": primary,
                "context_bbs": context,
            }
        )
    return out


def kindseq(outline: Sequence[Tuple[str, int]]) -> List[str]:
    return [str(k) for k, _d in outline]


def normalize_depths(outline: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not outline:
        return []
    levels = {d: i for i, d in enumerate(sorted({int(d) for _k, d in outline}))}
    return [(str(k), int(levels[int(d)])) for k, d in outline]


def render_pred_outline(outline: Sequence[Tuple[str, int, int, int]]) -> str:
    if not outline:
        return "<none>"
    lines = []
    for kind, depth, bb, parent in outline:
        lines.append(f"{'  ' * int(depth)}{kind}  bb={bb} parent={parent}")
    return "\n".join(lines)


def render_kind_depth_outline(outline: Sequence[Tuple[str, int]]) -> str:
    if not outline:
        return "<none>"
    return "\n".join(f"{'  ' * int(depth)}{kind}" for kind, depth in outline)


def _teacher_decl_line(teacher_skeleton: Any) -> str:
    text = str(teacher_skeleton or "")
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.endswith("{"):
        first = first[:-1].rstrip()
    if first and "(" in first and ")" in first:
        return first
    return "int func(...)"


def render_outline_head_skeleton(
    outline: Sequence[Tuple[str, int]],
    *,
    decl_line: str = "int func(...)",
) -> str:
    decl_line = str(decl_line or "int func(...)").strip() or "int func(...)"
    if not outline:
        return (
            f"{decl_line} {{\n"
            "    <FILL_BLOCK:slot_000>;  /* reserved preamble/context; BB assignment not decoded */\n"
            "}"
        )

    lines = [f"{decl_line} {{"]
    stack: List[int] = []
    slot = 1

    def close_to(depth: int) -> None:
        while len(stack) > int(depth):
            lines.append(f"{'    ' * len(stack)}}}")
            stack.pop()

    for idx, (kind, depth) in enumerate(outline):
        # Keep the rendered C-like text syntactically valid even when the model
        # predicts an impossible depth jump; metrics still use the raw outline.
        depth = min(max(0, int(depth)), len(stack))
        close_to(depth)
        indent = "    " * (len(stack) + 1)
        label = _outline_kind_label(str(kind))
        if str(kind) == "switch":
            lines.append(f"{indent}switch (<FILL_COND:slot_{slot:03d}>) {{  /* outline[{idx}]; BB assignment not decoded */")
            slot += 1
            lines.append(f"{indent}    case <FILL_CASE:slot_{slot:03d}>:")
            slot += 1
            lines.append(f"{indent}        <FILL_BLOCK:slot_{slot:03d}>;  /* predicted slot; BB assignment not decoded */")
            slot += 1
        else:
            lines.append(f"{indent}{label} (<FILL_COND:slot_{slot:03d}>) {{  /* outline[{idx}]; BB assignment not decoded */")
            slot += 1
            lines.append(f"{indent}    <FILL_BLOCK:slot_{slot:03d}>;  /* predicted slot; BB assignment not decoded */")
            slot += 1
        stack.append(depth)

    close_to(0)
    lines.insert(1, "    <FILL_BLOCK:slot_000>;  /* reserved preamble/context; BB assignment not decoded */")
    lines.append(f"    <FILL_BLOCK:slot_{slot:03d}>;  /* predicted follow/context; BB assignment not decoded */")
    lines.append("}")
    return "\n".join(lines)


def _fmt_bbs(bbs: Sequence[int], *, max_items: int = 16) -> str:
    vals = [int(x) for x in bbs if int(x) >= 0]
    vals = sorted(dict.fromkeys(vals))
    if len(vals) > max_items:
        shown = ", ".join(str(x) for x in vals[:max_items])
        return f"[{shown}, ...]"
    return "[" + ", ".join(str(x) for x in vals) + "]"


def _ctrl_kind_id(kind: str) -> int:
    return int(CTRL_NAME_TO_ID.get(str(kind or "none"), CTRL_NONE))


def _normalize_full_outline_with_bbs(
    outline: Sequence[Tuple[str, int, int, int]],
) -> List[Tuple[str, int, int, int]]:
    norm = normalize_depths([(k, d) for k, d, _bb, _parent in outline])
    out: List[Tuple[str, int, int, int]] = []
    for i, (kind, _depth, bb, parent) in enumerate(outline):
        nd = int(norm[i][1]) if i < len(norm) else 0
        out.append((str(kind), nd, int(bb), int(parent)))
    return out


@torch.no_grad()
def _assign_outline_controls_to_bbs(
    pred: Any,
    outline: Sequence[Tuple[str, int]],
) -> List[Dict[str, Any]]:
    """Attach outline-head control nodes to likely BB anchors.

    The outline head gives the most stable count/kind/depth prediction.  BB
    grounding comes from the ctrl/parent heads.  We greedily match outline
    nodes to predicted ctrl anchors of the same type, preferring the same
    normalized depth and preserving left-to-right order when possible.
    """
    n = int(pred.ctrl_logits.size(0))
    if n == 0:
        return []
    probs = torch.softmax(pred.ctrl_logits[:n].float(), dim=-1).detach().cpu()
    parents = decode_parent_tree(pred.parent_scores[:n, :n], pred.root_scores[:n])
    full = _normalize_full_outline_with_bbs(predicted_outline_from_output(pred))
    used: set[int] = set()
    assigned: List[Dict[str, Any]] = []
    last_pos = -1

    for idx, (kind, depth) in enumerate(outline):
        kind = str(kind)
        depth = int(depth)
        cid = _ctrl_kind_id(kind)
        candidates = [x for x in full if int(x[2]) not in used and str(x[0]) == kind]
        if candidates:
            def cand_key(x: Tuple[str, int, int, int]) -> Tuple[int, int, int]:
                _k, d, bb, _p = x
                return (abs(int(d) - depth), 0 if int(bb) >= last_pos else 1, abs(int(bb) - max(last_pos, 0)))

            _ck, cd, bb, parent = min(candidates, key=cand_key)
        else:
            ranked = probs[:, cid].argsort(descending=True).tolist() if 0 <= cid < probs.size(1) else []
            bb = next((int(x) for x in ranked if int(x) not in used), -1)
            cd = depth
            parent = int(parents[bb]) if 0 <= bb < len(parents) else n
        if 0 <= int(bb) < n:
            used.add(int(bb))
            last_pos = int(bb)
        assigned.append(
            {
                "outline_index": int(idx),
                "kind": kind,
                "depth": depth,
                "bb": int(bb),
                "parent": int(parent),
                "ctrl_prob": float(probs[int(bb), cid].item()) if 0 <= int(bb) < n and 0 <= cid < probs.size(1) else 0.0,
                "ctrl_depth": int(cd),
            },
        )
    return assigned


@torch.no_grad()
def _region_context(pred: Any, anchor: int, *, threshold: float = 0.5) -> List[int]:
    n = int(pred.ctrl_logits.size(0))
    if not (0 <= int(anchor) < n) or pred.region_logits.numel() == 0:
        return [int(anchor)] if 0 <= int(anchor) < n else []
    scores = torch.sigmoid(pred.region_logits[int(anchor), :n].float()).detach().cpu()
    ctx = [int(i) for i, v in enumerate(scores.tolist()) if float(v) >= float(threshold)]
    if int(anchor) not in ctx:
        ctx.insert(0, int(anchor))
    if not ctx:
        top = scores.argsort(descending=True)[: min(4, n)].tolist()
        ctx = [int(x) for x in top]
    return sorted(dict.fromkeys(ctx))


@torch.no_grad()
def _slot_group_for_context(
    pred: Any,
    ctx: Sequence[int],
    used_controls: set[int],
    used_primary: Optional[set[int]] = None,
) -> List[int]:
    n = int(pred.ctrl_logits.size(0))
    ctx_set = {int(x) for x in ctx if 0 <= int(x) < n}
    if not ctx_set:
        return []
    used_primary = used_primary or set()
    groups = decode_slot_groups(pred, threshold=0.55)
    bb_to_group: Dict[int, int] = {}
    for gi, group in enumerate(groups):
        for bb in group:
            if 0 <= int(bb) < n:
                bb_to_group[int(bb)] = int(gi)
    slot_ids = pred.slot_logits[:n].argmax(dim=-1).detach().cpu().tolist()
    candidates = [
        bb for bb in sorted(ctx_set)
        if bb not in used_controls and bb not in used_primary
    ]
    if not candidates:
        candidates = [bb for bb in sorted(ctx_set) if bb not in used_controls]
    if not candidates:
        return []
    non_cond = [bb for bb in candidates if int(slot_ids[bb]) != 1]
    seed = non_cond[0] if non_cond else candidates[0]
    gi = bb_to_group.get(int(seed))
    if gi is not None and 0 <= gi < len(groups):
        group = [int(x) for x in groups[gi] if int(x) in ctx_set]
        if group:
            return sorted(dict.fromkeys(group))
    return sorted(dict.fromkeys(candidates))


def _teacher_layout_from_skeleton(teacher: Dict[str, Any]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for line in str(teacher.get("teacher_skeleton") or "").splitlines():
        m = FILL_SLOT_RE.search(line)
        if not m:
            continue
        try:
            sid = int(m.group(2))
        except Exception:
            sid = -1
        if sid == 0:
            continue
        depth = (len(line) - len(line.lstrip(" "))) // 4
        out.append((f"FILL_{m.group(1)}", int(depth)))
    return out


def _normalize_slot_layout_depths(layout: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not layout:
        return []
    min_depth = min(int(d) for _k, d in layout)
    return [(str(k), max(1, int(d) - min_depth + 1)) for k, d in layout]


def _slot_anchor_groups(pred: Any) -> Dict[int, List[int]]:
    n = int(pred.ctrl_logits.size(0))
    groups = decode_slot_groups(pred, threshold=0.55)
    out: Dict[int, List[int]] = {}
    for bb in range(n):
        out[int(bb)] = [int(bb)]
    for group in groups:
        g = sorted(dict.fromkeys(int(x) for x in group if 0 <= int(x) < n))
        for bb in g:
            out[int(bb)] = g
    return out


@torch.no_grad()
def _decode_slot_grounding(
    pred: Any,
    *,
    slot_pos: int,
    primary_threshold: float = 0.5,
    context_threshold: float = 0.5,
) -> Tuple[List[int], List[int]]:
    n = int(pred.ctrl_logits.size(0))
    if (
        n <= 0
        or not hasattr(pred, "slot_primary_logits")
        or pred.slot_primary_logits.numel() == 0
        or int(slot_pos) < 0
        or int(slot_pos) >= int(pred.slot_primary_logits.size(1))
    ):
        return [], []
    p_scores = torch.sigmoid(pred.slot_primary_logits[0, int(slot_pos), :n].float()).detach().cpu()
    c_scores = torch.sigmoid(pred.slot_context_logits[0, int(slot_pos), :n].float()).detach().cpu()
    primary = [int(i) for i, v in enumerate(p_scores.tolist()) if float(v) >= float(primary_threshold)]
    context = [int(i) for i, v in enumerate(c_scores.tolist()) if float(v) >= float(context_threshold)]
    if not primary and p_scores.numel() > 0:
        top = int(p_scores.argmax().item())
        if float(p_scores[top].item()) >= 0.20:
            primary = [top]
    if not context:
        context = list(primary)
    return sorted(dict.fromkeys(primary)), sorted(dict.fromkeys(context))


@torch.no_grad()
def render_slot_layout_head_skeleton_with_bb_alignment(
    slot_layout: Sequence[Tuple[str, int]],
    outline: Sequence[Tuple[str, int]],
    *,
    pred: Any,
    decl_line: str = "int func(...)",
) -> str:
    """Render the explicit slot-layout head without inventing extra slots."""
    decl_line = str(decl_line or "int func(...)").strip() or "int func(...)"
    n = int(pred.ctrl_logits.size(0))
    assignments = _assign_outline_controls_to_bbs(pred, outline)
    ctrl_cursor = 0
    slot_id = 1
    lines = [f"{decl_line} {{"]
    lines.append("    <FILL_BLOCK:slot_000>;  /* primary [] context [] */")
    stack_depths: List[int] = []

    def close_to(depth: int) -> None:
        while stack_depths and stack_depths[-1] >= int(depth):
            lines.append(f"{'    ' * stack_depths[-1]}}}")
            stack_depths.pop()

    def note(primary: Sequence[int], context: Sequence[int]) -> str:
        p = sorted(dict.fromkeys(int(x) for x in primary if 0 <= int(x) < n))
        c = sorted(dict.fromkeys(int(x) for x in context if 0 <= int(x) < n))
        return f"primary {_fmt_bbs(p)} context {_fmt_bbs(c)}"

    for slot_pos, (kind, raw_depth) in enumerate(slot_layout):
        depth = max(1, int(raw_depth))
        close_to(depth)
        pad = "    " * depth
        kind = str(kind)
        pred_primary, pred_context = _decode_slot_grounding(pred, slot_pos=int(slot_pos))
        if kind == "FILL_COND":
            assn = assignments[ctrl_cursor] if ctrl_cursor < len(assignments) else {"bb": -1, "parent": n, "kind": "if"}
            ctrl_cursor += 1
            anchor = int(assn.get("bb", -1))
            if not pred_primary and 0 <= anchor < n:
                pred_primary = [anchor]
            if not pred_context:
                pred_context = _region_context(pred, anchor)
            label = _outline_kind_label(str(assn.get("kind") or "if"))
            comment = note(pred_primary, pred_context)
            if str(assn.get("kind")) == "switch":
                lines.append(
                    f"{pad}switch (<FILL_COND:slot_{slot_id:03d}>) {{  "
                    f"/* outline[{int(assn.get('outline_index', ctrl_cursor - 1))}] "
                    f"BB_{anchor} parent={int(assn.get('parent', n))} {comment} */",
                )
            else:
                lines.append(
                    f"{pad}{label} (<FILL_COND:slot_{slot_id:03d}>) {{  "
                    f"/* outline[{int(assn.get('outline_index', ctrl_cursor - 1))}] "
                    f"BB_{anchor} parent={int(assn.get('parent', n))} {comment} */",
                )
            slot_id += 1
            stack_depths.append(depth)
            continue
        if kind == "FILL_BRANCH":
            lines.append(f"{pad}case <FILL_BRANCH:slot_{slot_id:03d}>:  /* {note(pred_primary, pred_context)} */")
            slot_id += 1
            continue
        if kind == "FILL_BLOCK":
            lines.append(f"{pad}<FILL_BLOCK:slot_{slot_id:03d}>;  /* {note(pred_primary, pred_context)} */")
            slot_id += 1

    close_to(1)
    lines.append("}")
    return "\n".join(lines)


def _segment_top_level_block(
    pred: Any,
    candidates: Sequence[int],
    used_controls: set[int],
    used_primary: set[int],
) -> List[int]:
    """Pick one predicted top-level block group from candidate BBs.

    The outline-head renderer knows the order of control headers, but the old
    renderer only emitted body slots and one final follow slot.  That misses
    ordinary statements before a top-level control.  This helper uses the
    same-slot decoder to keep such pre/follow blocks grouped instead of dumping
    all remaining BBs into one tail slot.
    """
    n = int(pred.ctrl_logits.size(0))
    cand = [
        int(bb) for bb in sorted(dict.fromkeys(candidates))
        if 0 <= int(bb) < n and int(bb) not in used_controls and int(bb) not in used_primary
    ]
    if not cand:
        return []

    slot_ids = pred.slot_logits[:n].argmax(dim=-1).detach().cpu().tolist()
    block_cand = [bb for bb in cand if int(slot_ids[bb]) != 1]
    cand = block_cand or cand

    groups = decode_slot_groups(pred, threshold=0.55)
    bb_to_group: Dict[int, int] = {}
    for gi, group in enumerate(groups):
        for bb in group:
            if 0 <= int(bb) < n:
                bb_to_group[int(bb)] = int(gi)

    seed = cand[0]
    gi = bb_to_group.get(seed)
    if gi is not None and 0 <= gi < len(groups):
        group = [
            int(bb) for bb in groups[gi]
            if int(bb) in set(cand) and int(bb) not in used_controls and int(bb) not in used_primary
        ]
        if group:
            return sorted(dict.fromkeys(group))
    return [seed]


def _emit_top_level_blocks_before(
    lines: List[str],
    *,
    pred: Any,
    stop_bb: int,
    slot: int,
    note: Any,
    used_controls: set[int],
    used_primary: set[int],
) -> int:
    """Emit available top-level non-control blocks before ``stop_bb``."""
    n = int(pred.ctrl_logits.size(0))
    if not (0 <= int(stop_bb) < n):
        return slot
    while True:
        candidates = [
            bb for bb in range(int(stop_bb))
            if bb not in used_controls and bb not in used_primary
        ]
        block = _segment_top_level_block(pred, candidates, used_controls, used_primary)
        if not block:
            return slot
        lines.append(f"    <FILL_BLOCK:slot_{slot:03d}>;  /* {note(block, block)} */")
        slot += 1


def _emit_remaining_top_level_blocks(
    lines: List[str],
    *,
    pred: Any,
    slot: int,
    note: Any,
    used_controls: set[int],
    used_primary: set[int],
) -> int:
    """Emit remaining top-level blocks after all controls are closed."""
    n = int(pred.ctrl_logits.size(0))
    while True:
        candidates = [
            bb for bb in range(n)
            if bb not in used_controls and bb not in used_primary
        ]
        block = _segment_top_level_block(pred, candidates, used_controls, used_primary)
        if not block:
            return slot
        lines.append(f"    <FILL_BLOCK:slot_{slot:03d}>;  /* {note(block, block)} */")
        slot += 1


@torch.no_grad()
def render_outline_head_skeleton_with_bb_alignment(
    outline: Sequence[Tuple[str, int]],
    *,
    pred: Any,
    decl_line: str = "int func(...)",
) -> str:
    """Render outline-head skeleton and annotate each slot with predicted BBs."""
    decl_line = str(decl_line or "int func(...)").strip() or "int func(...)"
    n = int(pred.ctrl_logits.size(0))
    assignments = _assign_outline_controls_to_bbs(pred, outline)
    used_controls = {int(x["bb"]) for x in assignments if 0 <= int(x.get("bb", -1)) < n}
    slot_primary: List[List[int]] = []
    used_primary: set[int] = set()

    def note(primary: Sequence[int], context: Sequence[int], *, track: bool = True) -> str:
        p = sorted(dict.fromkeys(int(x) for x in primary if 0 <= int(x) < n))
        c = sorted(dict.fromkeys(int(x) for x in context if 0 <= int(x) < n))
        if track:
            slot_primary.append(p)
            used_primary.update(p)
        return f"primary {_fmt_bbs(p)} context {_fmt_bbs(c)}"

    if n == 0:
        return f"{decl_line} {{\n    <FILL_BLOCK:slot_000>;  /* primary [] context [] */\n}}"

    lines = [f"{decl_line} {{"]
    lines.append(f"    <FILL_BLOCK:slot_000>;  /* {note([], [], track=False)} */")
    stack: List[int] = []
    slot = 1

    def close_to(depth: int) -> None:
        while len(stack) > int(depth):
            lines.append(f"{'    ' * len(stack)}}}")
            stack.pop()

    for idx, (kind, raw_depth) in enumerate(outline):
        depth = min(max(0, int(raw_depth)), len(stack))
        close_to(depth)
        assn = assignments[idx] if idx < len(assignments) else {"bb": -1, "parent": n, "ctrl_prob": 0.0}
        anchor = int(assn.get("bb", -1))
        if depth == 0:
            slot = _emit_top_level_blocks_before(
                lines,
                pred=pred,
                stop_bb=anchor,
                slot=slot,
                note=note,
                used_controls=used_controls,
                used_primary=used_primary,
            )
        indent = "    " * (len(stack) + 1)
        label = _outline_kind_label(str(kind))
        ctx = _region_context(pred, anchor)
        body = _slot_group_for_context(pred, ctx, used_controls, used_primary)
        ctrl_comment = note([anchor] if 0 <= anchor < n else [], ctx)
        if str(kind) == "switch":
            lines.append(
                f"{indent}switch (<FILL_COND:slot_{slot:03d}>) {{  "
                f"/* outline[{idx}] BB_{anchor} parent={int(assn.get('parent', n))} {ctrl_comment} */",
            )
            slot += 1
            body_comment = note(body, ctx) if body else "primary [] context []"
            lines.append(f"{indent}    case <FILL_BRANCH:slot_{slot:03d}>:  /* primary [] context [] */")
            slot += 1
            if body:
                lines.append(f"{indent}        <FILL_BLOCK:slot_{slot:03d}>;  /* {body_comment} */")
                slot += 1
            else:
                lines.append(f"{indent}        /* empty/unmapped switch body */")
        else:
            lines.append(
                f"{indent}{label} (<FILL_COND:slot_{slot:03d}>) {{  "
                f"/* outline[{idx}] BB_{anchor} parent={int(assn.get('parent', n))} {ctrl_comment} */",
            )
            slot += 1
            if body:
                body_comment = note(body, ctx)
                lines.append(f"{indent}    <FILL_BLOCK:slot_{slot:03d}>;  /* {body_comment} */")
                slot += 1
            else:
                lines.append(f"{indent}    /* empty/unmapped body */")
        stack.append(depth)

    close_to(0)
    slot = _emit_remaining_top_level_blocks(
        lines,
        pred=pred,
        slot=slot,
        note=note,
        used_controls=used_controls,
        used_primary=used_primary,
    )
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate B_skel predicted skeletons against AST teacher.")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--a-ckpt", type=Path, required=True)
    ap.add_argument("--bskel-ckpt", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-batch-size", type=int, default=64)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("===== B_skel predicted skeleton eval =====", flush=True)
    print("source_jsonl =", args.source_jsonl, flush=True)
    print("teacher_jsonl =", args.teacher_jsonl, flush=True)
    print("a_ckpt =", args.a_ckpt, flush=True)
    print("bskel_ckpt =", args.bskel_ckpt, flush=True)
    print("out_txt =", args.out_txt, flush=True)
    print("device =", device, flush=True)

    ckpt = torch.load(args.bskel_ckpt.expanduser().resolve(), map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    a_model = load_stage2_model(args.a_ckpt.expanduser().resolve(), device)
    parser = BinaryCSTParser(
        d_model=StrucLiftConfig().module_b.d_model,
        num_layers=int(ckpt_args.get("num_layers", 2)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        max_outline_controls=int(ckpt_args.get("max_outline_controls", 12)),
        max_outline_depth=int(ckpt_args.get("max_outline_depth", 8)),
        max_slot_layout_items=int(ckpt_args.get("max_slot_layout_items", 45)),
        max_slot_layout_depth=int(ckpt_args.get("max_slot_layout_depth", 8)),
    ).to(device)
    state = ckpt["parser"] if isinstance(ckpt, dict) and "parser" in ckpt else ckpt
    missing, unexpected = parser.load_state_dict(state, strict=False)
    print(f"loaded B_skel = {args.bskel_ckpt}", flush=True)
    print(f"missing = {len(missing)} unexpected = {len(unexpected)}", flush=True)
    parser.eval()

    ds = BSkelTeacherDataset(
        args.source_jsonl,
        args.teacher_jsonl,
        max_examples=int(args.max_examples),
        max_outline_controls=int(ckpt_args.get("max_outline_controls", 12)),
        max_outline_depth=int(ckpt_args.get("max_outline_depth", 8)),
        max_slot_layout_items=int(ckpt_args.get("max_slot_layout_items", 45)),
        max_slot_layout_depth=int(ckpt_args.get("max_slot_layout_depth", 8)),
    )
    cache_frozen_binary_features(
        a_model=a_model,
        dataset=ds,
        device=device,
        batch_size=int(args.cache_batch_size),
        label="eval",
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_bskel,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    rows: List[Dict[str, Any]] = []
    global_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_bskel_batch_to_device(batch, device, non_blocking=True)
            pred = parser(
                binary_node_embs=batch["bskel_cached_binary_node_embs"],
                pattern_logits=batch["bskel_cached_pattern_logits"],
                bin_struct_features=batch["bin_struct_features"],
                edge_index=batch["bin_edge_index"],
                edge_type=batch["bin_edge_type"],
                bin_batch=batch.get("bin_batch"),
                cfg_features=batch.get("bskel_cached_cfg_features"),
            )
            bin_batch = batch["bin_batch"]
            n_graphs = int(bin_batch.max().item()) + 1 if bin_batch.numel() else 0
            for gi in range(n_graphs):
                nodes = (bin_batch == gi).nonzero(as_tuple=True)[0]
                sub_pred = type(pred)(
                    node_states=pred.node_states[nodes],
                    ctrl_logits=pred.ctrl_logits[nodes],
                    role_logits=pred.role_logits[nodes],
                    role_pair_logits=pred.role_pair_logits[nodes][:, nodes],
                    root_role_logits=pred.root_role_logits[nodes],
                    slot_logits=pred.slot_logits[nodes],
                    parent_scores=pred.parent_scores[nodes][:, nodes],
                    root_scores=pred.root_scores[nodes],
                    region_logits=pred.region_logits[nodes][:, nodes],
                    same_slot_logits=pred.same_slot_logits[nodes][:, nodes],
                    slot_anchor_scores=pred.slot_anchor_scores[nodes][:, nodes],
                    outline_count_logits=pred.outline_count_logits[gi : gi + 1],
                    outline_kind_logits=pred.outline_kind_logits[gi : gi + 1],
                    outline_depth_logits=pred.outline_depth_logits[gi : gi + 1],
                    slot_layout_count_logits=pred.slot_layout_count_logits[gi : gi + 1],
                    slot_layout_kind_logits=pred.slot_layout_kind_logits[gi : gi + 1],
                    slot_layout_depth_logits=pred.slot_layout_depth_logits[gi : gi + 1],
                    slot_primary_logits=pred.slot_primary_logits[gi : gi + 1][:, :, nodes],
                    slot_context_logits=pred.slot_context_logits[gi : gi + 1][:, :, nodes],
                    structural_bias=pred.structural_bias[:, nodes][:, :, nodes]
                    if pred.structural_bias.dim() == 3 else pred.structural_bias,
                )
                item = ds.items[global_idx]
                teach = item.teacher
                t_outline = teacher_outline(teach)
                p_outline_full = predicted_outline_from_output(sub_pred)
                p_outline_head = predicted_outline_from_outline_head(sub_pred)
                p_outline = p_outline_head if p_outline_head is not None else strip_bb(p_outline_full)
                t_outline_norm = normalize_depths(t_outline)
                p_outline_norm = normalize_depths(p_outline)
                t_slot_layout = _teacher_layout_from_skeleton(teach)
                p_slot_layout_head = predicted_slot_layout_from_head(sub_pred)
                p_slot_layout_norm = (
                    _normalize_slot_layout_depths(p_slot_layout_head)
                    if p_slot_layout_head is not None
                    else None
                )
                p_slot_grounding = predicted_slot_grounding_from_head(sub_pred, p_slot_layout_norm)
                old_ctrl_parent_skeleton = render_predicted_canonical_skeleton(sub_pred)
                decl_line = _teacher_decl_line(teach.get("teacher_skeleton"))
                outline_only_skeleton = render_outline_head_skeleton(p_outline_norm, decl_line=decl_line)
                if p_slot_layout_norm is not None:
                    predicted_skeleton = render_slot_layout_head_skeleton_with_bb_alignment(
                        p_slot_layout_norm,
                        p_outline_norm,
                        pred=sub_pred,
                        decl_line=decl_line,
                    )
                    pred_source = "slot_layout_head_with_bb_alignment"
                elif p_outline_head is not None:
                    predicted_skeleton = render_outline_head_skeleton_with_bb_alignment(
                        p_outline_norm,
                        pred=sub_pred,
                        decl_line=decl_line,
                    )
                    pred_source = "outline_head_with_bb_alignment"
                else:
                    predicted_skeleton = old_ctrl_parent_skeleton
                    pred_source = "bb_ctrl_parent"
                rec = {
                    "example": global_idx,
                    "input_line": teach.get("input_line"),
                    "func_name": teach.get("func_name"),
                    "n_bb": teach.get("n_bb"),
                    "teacher_outline": t_outline,
                    "teacher_outline_normalized": t_outline_norm,
                    "pred_outline": p_outline,
                    "pred_outline_source": "outline_head" if p_outline_head is not None else "bb_ctrl_parent",
                    "pred_outline_normalized": p_outline_norm,
                    "pred_outline_full": p_outline_full,
                    "teacher_slot_layout": t_slot_layout,
                    "pred_slot_layout": p_slot_layout_head,
                    "pred_slot_layout_normalized": p_slot_layout_norm,
                    "pred_slot_grounding": p_slot_grounding,
                    "count_equal": len(t_outline) == len(p_outline),
                    "kindseq_equal": kindseq(t_outline) == kindseq(p_outline),
                    "outline_raw_equal": t_outline == p_outline,
                    "outline_equal": t_outline_norm == p_outline_norm,
                    "function_decl": decl_line,
                    "teacher_skeleton": teach.get("teacher_skeleton"),
                    "predicted_skeleton": predicted_skeleton,
                    "predicted_skeleton_source": pred_source,
                    "diagnostic_outline_only_skeleton": outline_only_skeleton,
                    "diagnostic_ctrl_parent_skeleton": old_ctrl_parent_skeleton,
                }
                rows.append(rec)
                global_idx += 1

    n = len(rows)
    count_ok = sum(1 for r in rows if r["count_equal"])
    kind_ok = sum(1 for r in rows if r["kindseq_equal"])
    outline_ok = sum(1 for r in rows if r["outline_equal"])
    raw_outline_ok = sum(1 for r in rows if r["outline_raw_equal"])
    layout_rows = [r for r in rows if r.get("pred_slot_layout_normalized") is not None]
    layout_count_ok = sum(
        1 for r in layout_rows
        if len(r.get("teacher_slot_layout") or []) == len(r.get("pred_slot_layout_normalized") or [])
    )
    layout_kind_ok = sum(
        1 for r in layout_rows
        if [k for k, _d in (r.get("teacher_slot_layout") or [])]
        == [k for k, _d in (r.get("pred_slot_layout_normalized") or [])]
    )
    layout_exact_ok = sum(
        1 for r in layout_rows
        if (r.get("teacher_slot_layout") or []) == (r.get("pred_slot_layout_normalized") or [])
    )
    gap_dist = Counter((len(r["pred_outline"]) - len(r["teacher_outline"])) for r in rows)
    layout_gap_dist = Counter(
        len(r.get("pred_slot_layout_normalized") or []) - len(r.get("teacher_slot_layout") or [])
        for r in layout_rows
    )
    kind_gap = Counter((tuple(kindseq(r["teacher_outline"])), tuple(kindseq(r["pred_outline"]))) for r in rows if not r["kindseq_equal"])

    with args.out_txt.open("w", encoding="utf-8") as wf:
        wf.write("===== B_SKEL PREDICTED SKELETON EVAL =====\n")
        wf.write(f"source_jsonl = {args.source_jsonl}\n")
        wf.write(f"teacher_jsonl = {args.teacher_jsonl}\n")
        wf.write(f"a_ckpt = {args.a_ckpt}\n")
        wf.write(f"bskel_ckpt = {args.bskel_ckpt}\n")
        wf.write(f"n = {n}\n")
        wf.write(f"control_count_equal = {count_ok}/{n} ({count_ok / max(1, n) * 100:.1f}%)\n")
        wf.write(f"kindseq_equal = {kind_ok}/{n} ({kind_ok / max(1, n) * 100:.1f}%)\n")
        wf.write(f"outline_equal = {outline_ok}/{n} ({outline_ok / max(1, n) * 100:.1f}%)\n")
        wf.write(f"outline_raw_equal = {raw_outline_ok}/{n} ({raw_outline_ok / max(1, n) * 100:.1f}%)\n")
        wf.write(
            f"slot_layout_count_equal = {layout_count_ok}/{len(layout_rows)} "
            f"({layout_count_ok / max(1, len(layout_rows)) * 100:.1f}%)\n"
        )
        wf.write(
            f"slot_layout_kindseq_equal = {layout_kind_ok}/{len(layout_rows)} "
            f"({layout_kind_ok / max(1, len(layout_rows)) * 100:.1f}%)\n"
        )
        wf.write(
            f"slot_layout_equal = {layout_exact_ok}/{len(layout_rows)} "
            f"({layout_exact_ok / max(1, len(layout_rows)) * 100:.1f}%)\n"
        )
        wf.write(f"count_gap_dist = {gap_dist.most_common()}\n")
        wf.write(f"slot_layout_count_gap_dist = {layout_gap_dist.most_common()}\n")
        wf.write(f"top_kindseq_errors = {kind_gap.most_common(20)}\n")
        wf.write(f"elapsed_sec = {time.time() - t0:.1f}\n\n")
        for r in rows:
            wf.write("=" * 120 + "\n")
            wf.write(
                f"example={r['example']} input_line={r['input_line']} "
                f"func={r['func_name']} n_bb={r['n_bb']} "
                f"count_equal={r['count_equal']} kindseq_equal={r['kindseq_equal']} "
                f"outline_equal={r['outline_equal']}\n"
            )
            wf.write("\n-- TEACHER OUTLINE --\n")
            wf.write(render_pred_outline([(k, d, -1, -1) for k, d in r["teacher_outline"]]) + "\n")
            wf.write("\n-- TEACHER OUTLINE NORMALIZED --\n")
            wf.write(render_pred_outline([(k, d, -1, -1) for k, d in r["teacher_outline_normalized"]]) + "\n")
            wf.write("\n-- PREDICTED OUTLINE HEAD --\n")
            wf.write(render_kind_depth_outline(r["pred_outline"]) + "\n")
            wf.write("\n-- PREDICTED OUTLINE HEAD NORMALIZED --\n")
            wf.write(render_kind_depth_outline(r["pred_outline_normalized"]) + "\n")
            wf.write("\n-- TEACHER SLOT LAYOUT --\n")
            wf.write(str(r.get("teacher_slot_layout") or []) + "\n")
            wf.write("\n-- PREDICTED SLOT LAYOUT HEAD NORMALIZED --\n")
            wf.write(str(r.get("pred_slot_layout_normalized") or []) + "\n")
            wf.write("\n-- DIAGNOSTIC CTRL/PARENT OUTLINE, NOT USED FOR OUTLINE METRICS --\n")
            wf.write(render_pred_outline(r["pred_outline_full"]) + "\n")
            wf.write("\n-- TEACHER CANONICAL SKELETON --\n")
            wf.write(str(r.get("teacher_skeleton") or "").rstrip() + "\n")
            wf.write("\n-- PREDICTED CANONICAL SKELETON --\n")
            wf.write(str(r.get("predicted_skeleton") or "").rstrip() + "\n\n")
            wf.write("\n-- DIAGNOSTIC OUTLINE-ONLY SKELETON --\n")
            wf.write(str(r.get("diagnostic_outline_only_skeleton") or "").rstrip() + "\n\n")

    if args.out_jsonl is not None:
        with args.out_jsonl.open("w", encoding="utf-8") as jf:
            for r in rows:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"n = {n}", flush=True)
    print(f"control_count_equal = {count_ok}/{n} ({count_ok / max(1, n) * 100:.1f}%)", flush=True)
    print(f"kindseq_equal = {kind_ok}/{n} ({kind_ok / max(1, n) * 100:.1f}%)", flush=True)
    print(f"outline_equal = {outline_ok}/{n} ({outline_ok / max(1, n) * 100:.1f}%)", flush=True)
    print(f"outline_raw_equal = {raw_outline_ok}/{n} ({raw_outline_ok / max(1, n) * 100:.1f}%)", flush=True)
    print(
        f"slot_layout_count_equal = {layout_count_ok}/{len(layout_rows)} "
        f"({layout_count_ok / max(1, len(layout_rows)) * 100:.1f}%)",
        flush=True,
    )
    print(
        f"slot_layout_kindseq_equal = {layout_kind_ok}/{len(layout_rows)} "
        f"({layout_kind_ok / max(1, len(layout_rows)) * 100:.1f}%)",
        flush=True,
    )
    print(
        f"slot_layout_equal = {layout_exact_ok}/{len(layout_rows)} "
        f"({layout_exact_ok / max(1, len(layout_rows)) * 100:.1f}%)",
        flush=True,
    )
    print("count_gap_dist =", gap_dist.most_common(), flush=True)
    print("slot_layout_count_gap_dist =", layout_gap_dist.most_common(), flush=True)
    print("saved_txt =", args.out_txt, flush=True)
    if args.out_jsonl is not None:
        print("saved_jsonl =", args.out_jsonl, flush=True)
    print("elapsed_sec =", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()

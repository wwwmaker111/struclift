#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Module-A control skeletons with region-level slots.

This is an audit-only exporter for the "control skeleton + region slots" idea:

* keep only control syntax: ``if`` / ``else`` / ``while`` / ``switch`` / braces;
* each control condition is one ``FILL_COND`` slot;
* consecutive non-control statements are merged into one ``FILL_BLOCK`` slot;
* every slot carries primary BB ids plus read-only context BB ids; a block slot
  can own many BBs and the downstream decoder can generate multiple C
  statements.

The script does not run Module B/alignment.  It uses Module A
``pattern_logits.argmax`` and the current deterministic CST builder, then
serializes a different region-level skeleton for manual audit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import augment_binskel_sft as sft_aug  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402
from struclift.config import StrucLiftConfig  # noqa: E402
from struclift.data.binskel_jsonl import (  # noqa: E402
    collate_binskel_batch,
    move_binskel_batch_to_device,
)
from struclift.data.structures import EdgeType, PatternType  # noqa: E402
from struclift.models.struclift import StrucLift  # noqa: E402
from struclift.utils.skeleton import (  # noqa: E402
    CSTNode,
    build_control_structure_tree_with_stats,
)


PATTERN_NAMES = {
    0: "sequential",
    1: "if_header",
    2: "if_then_body",
    3: "if_else_body",
    4: "loop_header",
    5: "loop_body",
    6: "loop_exit",
    7: "switch_header",
    8: "switch_case_body",
    9: "function_prologue_epilogue",
}


@dataclass
class RegionSlot:
    slot_id: str
    slot_type: str
    role: str
    # Primary BBs are the supervised owner set for this slot.  They stay
    # single-owner; context BBs are read-only evidence and may repeat.
    aligned_bbs: List[int]
    context_bbs: List[int]
    context: str


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        k = k.replace("._orig_mod.", ".")
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        out[k] = v
    return out


def load_model(path: Path, device: torch.device) -> StrucLift:
    cfg = StrucLiftConfig()
    model = StrucLift(cfg)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    state = normalize_state(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded ckpt = {path}", flush=True)
    print(f"missing = {len(missing)} unexpected = {len(unexpected)}", flush=True)
    if missing[:10]:
        print("missing examples:", missing[:10], flush=True)
    if unexpected[:10]:
        print("unexpected examples:", unexpected[:10], flush=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def n_bb(row: Dict[str, Any]) -> int:
    bl = row.get("bin_block_lengths")
    return len(bl) if isinstance(bl, list) else 0


def is_valid_bin(row: Dict[str, Any]) -> bool:
    bl = row.get("bin_block_lengths")
    pl = row.get("pattern_labels")
    if not isinstance(bl, list) or not bl:
        return False
    if not isinstance(pl, list) or len(pl) != len(bl):
        return False
    n_instr = sum(int(x) for x in bl)
    if not isinstance(row.get("bin_opcodes"), list) or len(row["bin_opcodes"]) != n_instr:
        return False
    if not isinstance(row.get("bin_operand_types"), list) or len(row["bin_operand_types"]) != n_instr * 3:
        return False
    if not isinstance(row.get("bin_operand_values"), list) or len(row["bin_operand_values"]) != n_instr * 3:
        return False
    if not isinstance(row.get("bin_struct_features"), list) or len(row["bin_struct_features"]) != len(bl):
        return False
    ei = row.get("bin_edge_index")
    et = row.get("bin_edge_type") or []
    if not isinstance(ei, list) or len(ei) != 2:
        return False
    if len(ei[0]) != len(ei[1]) or len(ei[0]) != len(et):
        return False
    return True


def bin_only_record(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "bin_block_lengths",
        "bin_opcodes",
        "bin_operand_types",
        "bin_operand_values",
        "bin_struct_features",
        "bin_edge_index",
        "bin_edge_type",
        "pattern_labels",
    ]
    return {k: row[k] for k in keys if k in row}


def source_control_outline(row: Dict[str, Any], source_control_filter: str) -> List[Tuple[str, int, int]]:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    outline = effect._source_control_outline(
        rows,
        active_lines=active,
        source_control_filter=source_control_filter,
    )
    if not outline and active:
        outline = effect._active_line_control_outline(
            row,
            active,
            source_control_filter=source_control_filter,
        )
    return outline


def has_control_by_source(row: Dict[str, Any], source_control_filter: str) -> bool:
    return len(source_control_outline(row, source_control_filter)) > 0


def source_function_with_numbers(row: Dict[str, Any]) -> Tuple[str, Optional[Tuple[int, int]]]:
    lt = repack._line_to_text(row.get("source_file"))
    if not lt:
        return "<source file unreadable>", None
    bounds = repack._function_body_line_range(row, lt)
    if bounds is None:
        return "<function bounds unavailable>", None
    lo, hi = bounds
    lines = []
    for ln in range(lo, hi + 1):
        if ln in lt:
            lines.append(f"{ln:5d}: {lt[ln]}")
    return "\n".join(lines) if lines else "<empty source range>", bounds


def source_control_spans_by_kind(
    row: Dict[str, Any],
    source_control_filter: str,
) -> Dict[str, List[Tuple[int, int]]]:
    """Return source control spans grouped as if/loop/switch.

    This is audit-only metadata.  The real region-slot builder is intentionally
    source-free: it uses Module-A labels plus the binary CFG, because source is
    unavailable at inference time.
    """
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    out: Dict[str, List[Tuple[int, int]]] = {"if": [], "loop": [], "switch": []}
    for r in rows:
        typ = str(r.get("type") or "")
        kind: Optional[str] = None
        if typ == "if_statement":
            kind = "if"
        elif typ in {"while_statement", "for_statement", "do_statement"}:
            kind = "loop"
        elif typ == "switch_statement":
            kind = "switch"
        if kind is None:
            continue
        if not effect._row_passes_source_filter(r, active, source_control_filter):
            continue
        try:
            lo = int(r.get("start_line") or 0)
            hi = int(r.get("end_line") or lo)
        except Exception:
            continue
        if lo <= 0:
            continue
        if hi < lo:
            lo, hi = hi, lo
        out[kind].append((lo, hi))
    for kind in out:
        out[kind] = sorted(set(out[kind]))
    return out


def source_control_rows_by_kind(
    row: Dict[str, Any],
    source_control_filter: str,
) -> Dict[str, List[Dict[str, Any]]]:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    out: Dict[str, List[Dict[str, Any]]] = {"if": [], "loop": [], "switch": []}
    for r in rows:
        typ = str(r.get("type") or "")
        kind: Optional[str] = None
        if typ == "if_statement":
            kind = "if"
        elif typ in {"while_statement", "for_statement", "do_statement"}:
            kind = "loop"
        elif typ == "switch_statement":
            kind = "switch"
        if kind is None:
            continue
        if not effect._row_passes_source_filter(r, active, source_control_filter):
            continue
        out[kind].append(dict(r))
    for kind in out:
        out[kind].sort(key=lambda r: (int(r.get("start_line") or 0), int(r.get("end_line") or 0)))
    return out


def _numbered_source_code(src_text: str) -> str:
    lines: List[str] = []
    for raw in str(src_text or "").splitlines():
        m = re.match(r"\s*\d+:\s?(.*)$", raw)
        lines.append(m.group(1) if m else raw)
    return "\n".join(lines)


def _strip_cish_for_braces(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", str(code or ""), flags=re.S)
    out: List[str] = []
    i = 0
    in_str: Optional[str] = None
    esc = False
    while i < len(code):
        ch = code[i]
        if in_str:
            if ch == "\n":
                out.append("\n")
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {"'", '"'}:
            in_str = ch
            i += 1
            continue
        if ch == "/" and i + 1 < len(code) and code[i + 1] == "/":
            j = code.find("\n", i)
            if j < 0:
                break
            out.append("\n")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def source_reliability_meta(
    row: Dict[str, Any],
    src_text: str,
    bounds: Tuple[int, int],
    smeta: Dict[str, Any],
    source_control_filter: str,
) -> Dict[str, Any]:
    """Audit whether the source snippet is a trustworthy reference.

    Hard-unreliable means the source lines used by BB alignment/control outline
    fall outside the function-body bounds we printed.  Brace imbalance is kept
    as a warning because preprocessor conditionals and large literal tables can
    make a simple brace scan noisy.
    """
    lo, hi = int(bounds[0]), int(bounds[1])
    if hi < lo:
        lo, hi = hi, lo

    active = sorted(int(x) for x in effect._active_source_lines(row) if int(x) > 0)
    outline = source_control_outline(row, source_control_filter)
    outline_lines = sorted({int(ln) for _typ, _depth, ln in outline if int(ln) > 0})
    active_outside = [ln for ln in active if ln < lo or ln > hi]
    active_inside = [ln for ln in active if lo <= ln <= hi]
    outline_outside = [ln for ln in outline_lines if ln < lo or ln > hi]

    hard: List[str] = []
    warn: List[str] = []
    if smeta.get("source_parse_status") != "statement_rows_ok":
        warn.append(f"source_parse:{smeta.get('source_parse_status')}")
    if smeta.get("source_outline_source") == "active_line_fallback":
        warn.append("active_line_fallback")
    if active and not active_inside:
        hard.append("active_lines_disjoint_source_bounds")
    elif active and len(active_outside) > max(4, len(active) // 2):
        hard.append("most_active_lines_outside_source_bounds")
    if outline_outside:
        hard.append("control_outline_outside_source_bounds")

    code = _strip_cish_for_braces(_numbered_source_code(src_text))
    bal = 0
    min_bal = 0
    for ch in code:
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
            min_bal = min(min_bal, bal)
    nonempty = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if bal != 0:
        warn.append(f"brace_balance_{bal:+d}")
    if min_bal < 0:
        warn.append("brace_underflow")
    if nonempty and not nonempty[-1].endswith("}"):
        warn.append("last_line_not_close_brace")

    status = "reliable"
    if hard:
        status = "unreliable"
    elif warn:
        status = "warning"
    return {
        "source_reliability": status,
        "source_reliability_hard_reasons": hard,
        "source_reliability_warnings": warn,
        "source_bounds": (lo, hi),
        "source_counts": dict(smeta.get("source_counts") or {}),
        "active_line_minmax": (
            min(active) if active else None,
            max(active) if active else None,
        ),
        "outline_line_minmax": (
            min(outline_lines) if outline_lines else None,
            max(outline_lines) if outline_lines else None,
        ),
        "n_active_lines_outside_bounds": len(active_outside),
        "n_outline_lines_outside_bounds": len(outline_outside),
        "source_control_spans": source_control_spans_by_kind(row, source_control_filter),
        "source_control_rows": source_control_rows_by_kind(row, source_control_filter),
    }


def compact_source_reliability_for_log(source_aux: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_reliability": source_aux.get("source_reliability"),
        "source_reliability_hard_reasons": source_aux.get("source_reliability_hard_reasons") or [],
        "source_reliability_warnings": source_aux.get("source_reliability_warnings") or [],
        "source_bounds": source_aux.get("source_bounds"),
        "active_line_minmax": source_aux.get("active_line_minmax"),
        "outline_line_minmax": source_aux.get("outline_line_minmax"),
        "n_active_lines_outside_bounds": source_aux.get("n_active_lines_outside_bounds"),
        "n_outline_lines_outside_bounds": source_aux.get("n_outline_lines_outside_bounds"),
        "source_counts": source_aux.get("source_counts") or {},
        "source_control_span_counts": {
            k: len(v) for k, v in (source_aux.get("source_control_spans") or {}).items()
        },
    }


def source_control_meta(row: Dict[str, Any], source_control_filter: str) -> Dict[str, Any]:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    outline = effect._source_control_outline(
        rows,
        active_lines=active,
        source_control_filter=source_control_filter,
    )
    outline_source = "statement_rows"
    if not outline and active:
        fallback = effect._active_line_control_outline(
            row,
            active,
            source_control_filter=source_control_filter,
        )
        if fallback:
            outline = fallback
            outline_source = "active_line_fallback"
    seq = [(t, d) for t, d, _ln in outline]
    lines_by_kind: Dict[str, List[int]] = {"if": [], "loop": [], "switch": []}
    for typ, _depth, ln in outline:
        if typ in lines_by_kind:
            lines_by_kind[typ].append(int(ln))
    return {
        "source_control_count": len(seq),
        "source_counts": dict(effect._counts(seq)),
        "source_control_lines": {
            k: sorted(set(v)) for k, v in lines_by_kind.items()
        },
        "source_outline": effect._source_outline_lines(outline, limit=160),
        "source_outline_source": outline_source,
        "source_statement_rows": len(rows),
        "active_source_lines": len(active),
        "source_parse_status": effect._source_parse_status(row, rows),
    }


@torch.inference_mode()
def predict_labels_batch(
    model: StrucLift,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
) -> List[Tuple[List[int], List[float]]]:
    small = [bin_only_record(r) for r in rows]
    batch = collate_binskel_batch(
        small,
        inst_dropout_p=0.0,
        edge_dropout_p=0.0,
        add_func_uid=False,
    )
    batch = move_binskel_batch_to_device(batch, device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        enc = model.encode(
            bin_opcodes=batch["bin_opcodes"],
            bin_operand_types=batch["bin_operand_types"],
            bin_operand_values=batch["bin_operand_values"],
            bin_block_lengths=batch["bin_block_lengths"],
            bin_edge_index=batch["bin_edge_index"],
            bin_edge_type=batch["bin_edge_type"],
            bin_struct_features=batch["bin_struct_features"],
            bin_batch=batch.get("bin_batch"),
            run_alignment=False,
        )
    logits = enc.pattern_logits.float()
    labels_all = logits.argmax(dim=-1).detach().cpu().tolist()
    probs_all = torch.softmax(logits, dim=-1).amax(dim=-1).detach().cpu().tolist()
    out: List[Tuple[List[int], List[float]]] = []
    off = 0
    for row in rows:
        n = n_bb(row)
        out.append((
            [int(x) for x in labels_all[off : off + n]],
            [float(x) for x in probs_all[off : off + n]],
        ))
        off += n
    return out


def label_summary(labels: Sequence[int]) -> str:
    c = Counter(int(x) for x in labels)
    return ", ".join(
        f"{PATTERN_NAMES.get(k, 'unknown_' + str(k))}={c[k]}"
        for k in sorted(c)
    )


def label_sequence(labels: Sequence[int], max_items: int = 220) -> str:
    xs = []
    for i, lab in enumerate(labels[:max_items]):
        xs.append(f"BB_{i}:{PATTERN_NAMES.get(int(lab), 'unknown_' + str(lab))}")
    if len(labels) > max_items:
        xs.append("...")
    return "\n".join(xs)


def _bb_context(hints: Dict[int, str], bbs: Sequence[int], max_chars: int = 1200) -> str:
    parts = []
    for bb in bbs:
        txt = hints.get(int(bb), "")
        if txt:
            parts.append(f"BB_{int(bb)}: {txt}")
        else:
            parts.append(f"BB_{int(bb)}")
    out = " | ".join(parts)
    if len(out) > max_chars:
        return out[:max_chars] + " ..."
    return out


def _bb_comment(bbs: Sequence[int], label: Optional[int] = None) -> str:
    if not bbs:
        return "/* aligned to <none> */"
    bb_s = ", ".join(f"BB_{int(x)}" for x in bbs)
    if label is None:
        return f"/* aligned to {bb_s} */"
    return f"/* aligned to {bb_s} ({PATTERN_NAMES.get(int(label), str(label))}) */"


def _unique_bbs(bbs: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for bb in bbs:
        b = int(bb)
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def _edge_lists_from_row(row: Dict[str, Any], n: int) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], List[Tuple[int, int, int]]]:
    succs: Dict[int, List[int]] = {i: [] for i in range(n)}
    preds: Dict[int, List[int]] = {i: [] for i in range(n)}
    edges: List[Tuple[int, int, int]] = []
    ei = row.get("bin_edge_index") or [[], []]
    et = row.get("bin_edge_type") or []
    if not isinstance(ei, list) or len(ei) != 2:
        return succs, preds, edges
    for k, (s_raw, d_raw) in enumerate(zip(ei[0], ei[1])):
        try:
            s, d = int(s_raw), int(d_raw)
        except Exception:
            continue
        if not (0 <= s < n and 0 <= d < n):
            continue
        try:
            t = int(et[k]) if k < len(et) else int(EdgeType.UNCOND)
        except Exception:
            t = int(EdgeType.UNCOND)
        succs[s].append(d)
        preds[d].append(s)
        edges.append((s, d, t))
    for i in range(n):
        succs[i] = sorted(set(succs[i]))
        preds[i] = sorted(set(preds[i]))
    return succs, preds, edges


def _bb_source_line_sets(row: Dict[str, Any], n: int) -> List[Set[int]]:
    groups = row.get("block_src_lines") or []
    out: List[Set[int]] = [set() for _ in range(n)]
    if not isinstance(groups, list):
        return out
    for i, group in enumerate(groups[:n]):
        if not isinstance(group, list):
            continue
        vals: Set[int] = set()
        for x in group:
            try:
                v = int(x)
            except Exception:
                continue
            if v > 0:
                vals.add(v)
        out[i] = vals
    return out


def _immediate_postdominators_from_succs(succs: Dict[int, List[int]], n: int) -> List[int]:
    if n <= 0:
        return []
    exits = [i for i in range(n) if not succs.get(i)]
    if not exits:
        return [-1] * n
    universe = set(range(n))
    postdom: List[Set[int]] = [set(universe) for _ in range(n)]
    for e in exits:
        postdom[e] = {e}
    changed = True
    while changed:
        changed = False
        for bb in range(n - 1, -1, -1):
            if bb in exits:
                continue
            ss = succs.get(bb, [])
            if not ss:
                new_set = {bb}
            else:
                common = set(universe)
                for s in ss:
                    common &= postdom[int(s)]
                new_set = {bb} | common
            if new_set != postdom[bb]:
                postdom[bb] = new_set
                changed = True
    out: List[int] = [-1] * n
    for bb in range(n):
        candidates = [c for c in postdom[bb] if c != bb]
        if candidates:
            out[bb] = max(candidates, key=lambda c: (len(postdom[c]), -int(c)))
    return out


def refine_labels_for_region_builder(
    row: Dict[str, Any],
    labels: Sequence[int],
) -> Tuple[List[int], Dict[str, Any]]:
    """Apply conservative CFG sanity checks before region-slot structuring.

    The Module-A argmax can mark routing/join blocks as headers.  Those false
    headers are what produced many extra ``if`` nodes in the train/val audits.
    This pass does not look at source text; it only rejects headers whose CFG
    shape cannot support the claimed control construct.
    """
    refined = [int(x) for x in labels]
    n = len(refined)
    succs, preds, edges = _edge_lists_from_row(row, n)

    demoted_if: List[int] = []
    demoted_switch: List[int] = []
    demoted_loop: List[int] = []
    promoted_if_to_switch: List[int] = []

    back_targets: Set[int] = {
        d for s, d, t in edges
        if t == int(EdgeType.BACK) or d <= s
    }

    for bb, lab in enumerate(list(refined)):
        succ = [s for s in succs.get(bb, []) if s != bb]
        if lab == int(PatternType.IF_HEADER):
            # A source-level if needs an actual binary branch.  Single-successor
            # IF_HEADER predictions are usually join/routing blocks; multiway
            # branches are stronger binary evidence for a switch dispatch.
            if len(succ) >= 3:
                refined[bb] = int(PatternType.SWITCH_HEADER)
                promoted_if_to_switch.append(bb)
            elif len(succ) < 2:
                refined[bb] = int(PatternType.SEQUENTIAL)
                demoted_if.append(bb)
        elif lab == int(PatternType.SWITCH_HEADER):
            # Keep real multiway switches; very small switch-looking nodes are
            # better treated as ordinary code than emitted as a bogus switch.
            if len(succ) < 2:
                refined[bb] = int(PatternType.SEQUENTIAL)
                demoted_switch.append(bb)
        elif lab == int(PatternType.LOOP_HEADER):
            # Preserve loop headers that are backedge targets or at least branch
            # out of the loop.  A single-successor non-backedge loop header is
            # usually a noisy label.
            has_backedge = bb in back_targets or any(p >= bb for p in preds.get(bb, []))
            if not has_backedge and len(succ) < 2:
                refined[bb] = int(PatternType.SEQUENTIAL)
                demoted_loop.append(bb)

    stats = {
        "demoted_if_headers": demoted_if,
        "demoted_switch_headers": demoted_switch,
        "demoted_loop_headers": demoted_loop,
        "promoted_if_to_switch_headers": promoted_if_to_switch,
        "n_demoted_if_headers": len(demoted_if),
        "n_demoted_switch_headers": len(demoted_switch),
        "n_promoted_if_to_switch_headers": len(promoted_if_to_switch),
        "n_demoted_loop_headers": len(demoted_loop),
    }
    return refined, stats


class RegionSerializer:
    def __init__(
        self,
        labels: Sequence[int],
        hints: Dict[int, str],
        row: Dict[str, Any],
    ):
        self.labels = [int(x) for x in labels]
        self.hints = hints
        self.succs, self.preds, _edges = _edge_lists_from_row(row, len(self.labels))
        self.ipdom = _immediate_postdominators_from_succs(self.succs, len(self.labels))
        self.slots: List[RegionSlot] = []
        self.slot_idx = 0
        self.control_condition_bbs: Set[int] = set()
        self.fallback_loop_headers: Set[int] = set()
        self.fallback_switch_headers: Set[int] = set()
        self.claimed_bbs: Set[int] = set()
        self.suppressed_duplicate_bbs: List[int] = []
        self.n_empty_block_slots_suppressed = 0
        self.n_empty_condition_slots = 0
        self.n_condition_routing_blocks = 0
        self.n_collapsed_condition_ifs = 0
        self.n_rejected_condition_if_collapses = 0
        self.rejected_condition_if_collapses: List[Tuple[int, int, str]] = []

    def context_bbs_for(self, seed_bbs: Sequence[int], role: str) -> List[int]:
        seeds = _unique_bbs(seed_bbs)
        if not seeds:
            return []
        ctx: List[int] = []
        if role == "function_signature":
            return []
        for bb in seeds:
            # Give the decoder local CFG evidence without making those BBs
            # supervised owners of this slot.
            ctx.extend(self.preds.get(bb, []))
            ctx.append(bb)
            ctx.extend(self.succs.get(bb, []))
        return _unique_bbs(x for x in ctx if 0 <= int(x) < len(self.labels))

    def new_slot(
        self,
        slot_type: str,
        role: str,
        bbs: Sequence[int],
        *,
        context_bbs: Optional[Sequence[int]] = None,
    ) -> str:
        sid = f"slot_{self.slot_idx}"
        self.slot_idx += 1
        bb_list_raw = _unique_bbs(bbs)
        bb_list: List[int] = []
        if slot_type in {"FILL_COND", "FILL_BLOCK"}:
            for bb in bb_list_raw:
                if bb in self.claimed_bbs:
                    self.suppressed_duplicate_bbs.append(bb)
                    continue
                bb_list.append(bb)
                self.claimed_bbs.add(bb)
            if slot_type == "FILL_COND" and not bb_list:
                self.n_empty_condition_slots += 1
        else:
            bb_list = bb_list_raw
        if context_bbs is None:
            ctx_bbs = self.context_bbs_for(bb_list_raw or bb_list, role)
        else:
            ctx_bbs = _unique_bbs(context_bbs)
        self.slots.append(
            RegionSlot(
                slot_id=sid,
                slot_type=slot_type,
                role=role,
                aligned_bbs=bb_list,
                context_bbs=ctx_bbs,
                context=_bb_context(self.hints, ctx_bbs),
            )
        )
        return sid

    def slot_bbs(self, slot_id: str) -> List[int]:
        if self.slots and self.slots[-1].slot_id == slot_id:
            return list(self.slots[-1].aligned_bbs)
        for slot in reversed(self.slots):
            if slot.slot_id == slot_id:
                return list(slot.aligned_bbs)
        return []

    def node_cond_bbs(self, node: CSTNode) -> List[int]:
        bbs = list(node.aligned_bbs)
        if not bbs and node.condition_bb is not None:
            bbs = [int(node.condition_bb)]
        return _unique_bbs(bbs)

    def split_if_children(self, node: CSTNode) -> Tuple[List[CSTNode], List[CSTNode]]:
        then_children: List[CSTNode] = []
        else_children: List[CSTNode] = []
        for child in node.children:
            child_branch = child.branch_label if child.branch_label is not None else child.pattern_label
            if child_branch == int(PatternType.IF_ELSE_BODY):
                else_children.append(child)
            else:
                then_children.append(child)
        return then_children, else_children

    def reaches(self, start: int, target: int, stop: Set[int]) -> bool:
        if target < 0:
            return False
        stack = [int(start)]
        seen: Set[int] = set()
        while stack:
            bb = stack.pop()
            if bb == target:
                return True
            if bb in seen or bb in stop:
                continue
            seen.add(bb)
            for succ in self.succs.get(bb, []):
                stack.append(int(succ))
        return False

    def child_has_branch_to_merge(self, child_header: int, merge: int, parent_header: int) -> bool:
        if merge < 0:
            return False
        for succ in self.succs.get(int(child_header), []):
            if int(succ) == merge:
                return True
            if self.reaches(int(succ), int(merge), {int(parent_header), int(child_header)}):
                return True
        return False

    def direct_or_near_condition_edge(self, parent_header: int, child_header: int, parent_bbs: Sequence[int]) -> bool:
        if int(child_header) in self.succs.get(int(parent_header), []):
            return True
        # Short-circuit lowering often puts a tiny routing/return BB between two
        # condition headers.  Keep this narrow; broader reachability is handled
        # by postdom/merge checks below.
        parent_last = max(int(x) for x in parent_bbs) if parent_bbs else int(parent_header)
        return 0 < int(child_header) - int(parent_last) <= 2

    def children_have_control(self, children: Sequence[CSTNode], *, kinds: Optional[Set[str]] = None) -> bool:
        kinds = kinds or {"if_else", "while_loop", "for_loop", "switch"}
        stack = list(children)
        while stack:
            ch = stack.pop()
            if ch.type in kinds:
                return True
            stack.extend(ch.children)
        return False

    def count_child_controls(self, children: Sequence[CSTNode], *, kinds: Optional[Set[str]] = None) -> int:
        kinds = kinds or {"if_else", "while_loop", "for_loop", "switch"}
        total = 0
        stack = list(children)
        while stack:
            ch = stack.pop()
            if ch.type in kinds:
                total += 1
            stack.extend(ch.children)
        return total

    def statement_bbs_in_children(self, children: Sequence[CSTNode]) -> List[int]:
        out: List[int] = []
        stack = list(children)
        while stack:
            ch = stack.pop()
            if ch.type == "statement":
                out.extend(int(x) for x in getattr(ch, "aligned_bbs", []) or [])
            stack.extend(ch.children)
        return _unique_bbs(out)

    def single_if_child(self, children: Sequence[CSTNode]) -> Optional[CSTNode]:
        controls = [ch for ch in children if ch.type != "statement"]
        if len(controls) == 1 and controls[0].type == "if_else":
            return controls[0]
        return None

    def child_has_real_body(self, child: CSTNode, child_bbs: Sequence[int]) -> bool:
        child_then, child_else = self.split_if_children(child)
        if child_else:
            return True
        if self.children_have_control(child_then, kinds={"while_loop", "for_loop", "switch"}):
            return True
        # A single nested IF in the condition path is still commonly just
        # short-circuit lowering.  Do not reject a chain merely because it
        # contains more IF nodes: those are exactly the binary condition BBs we
        # want one region-level FILL_COND to own.  The child's final body is
        # preserved after the condition BBs are merged.
        return False

    def can_collapse_condition_if(
        self,
        parent_bbs: Sequence[int],
        child: CSTNode,
        child_bbs: Sequence[int],
    ) -> Tuple[bool, str]:
        if not parent_bbs or not child_bbs:
            return False, "missing_condition_bbs"
        parent_header = int(parent_bbs[0])
        child_header = int(child_bbs[0])
        if not self.direct_or_near_condition_edge(parent_header, child_header, parent_bbs):
            return False, "not_direct_or_near_condition_edge"

        parent_merge = self.ipdom[parent_header] if 0 <= parent_header < len(self.ipdom) else -1
        child_merge = self.ipdom[child_header] if 0 <= child_header < len(self.ipdom) else -1
        same_merge = parent_merge >= 0 and parent_merge == child_merge
        child_to_parent_merge = self.child_has_branch_to_merge(child_header, parent_merge, parent_header)
        merge_ok = same_merge or child_to_parent_merge
        if not merge_ok:
            return False, "no_shared_merge"

        if self.child_has_real_body(child, child_bbs):
            return False, "child_has_real_body"

        reason = "same_merge" if same_merge else "child_branch_to_parent_merge"
        return True, reason

    def collapse_condition_only_if_chain(self, node: CSTNode) -> Tuple[List[int], List[CSTNode], List[CSTNode]]:
        """Merge binary short-circuit condition chains into one condition slot.

        The audits show many false extra ``if`` nodes where the only child of an
        if is another if.  In stripped binary CFGs this usually means one C
        condition compiled into several conditional BBs.  We merge only the
        no-statement/no-else chain shape, leaving normal if bodies intact.
        """
        cond_bbs = self.node_cond_bbs(node)
        then_children, else_children = self.split_if_children(node)
        seen_nodes: Set[int] = set()

        # Only collapse a pure adjacent header chain.  Broader OR/AND branch
        # collapse looked tempting, but in audit it swallowed many real source
        # if-statements.  The final inference path must stay source-free, so we
        # keep the binary-only rule deliberately narrow here.
        while not else_children and len(then_children) == 1 and then_children[0].type == "if_else":
            child = then_children[0]
            ident = id(child)
            if ident in seen_nodes:
                break
            child_bbs = self.node_cond_bbs(child)
            ok, reason = self.can_collapse_condition_if(cond_bbs, child, child_bbs)
            if not ok:
                self.n_rejected_condition_if_collapses += 1
                if cond_bbs and child_bbs:
                    self.rejected_condition_if_collapses.append((int(cond_bbs[0]), int(child_bbs[0]), reason))
                break
            seen_nodes.add(ident)
            cond_bbs.extend(child_bbs)
            self.n_collapsed_condition_ifs += 1
            then_children, else_children = self.split_if_children(child)
        return _unique_bbs(cond_bbs), then_children, else_children

    def serialize(self, root: CSTNode, *, fallback_headers: bool = True) -> Tuple[str, List[RegionSlot], Dict[str, Any]]:
        sig = self.new_slot("FILL_FUNC_SIG", "function_signature", [])
        lines = [f"<FILL_FUNC_SIG:{sig}> {{"]
        lines.extend(self.emit_children(root.children, 1, role="function_body"))
        if fallback_headers:
            fallback_lines = self.emit_unstructured_loop_switch_fallbacks(1)
            if fallback_lines:
                lines.append("    /* unresolved predicted loop/switch headers */")
                lines.extend(fallback_lines)
        lines.append("}")
        stats = {
            "n_region_slots": len(self.slots),
            "condition_bbs": sorted(self.control_condition_bbs),
            "claimed_bbs": len(self.claimed_bbs),
            "suppressed_duplicate_bbs": sorted(set(self.suppressed_duplicate_bbs)),
            "n_suppressed_duplicate_bbs": len(self.suppressed_duplicate_bbs),
            "n_empty_block_slots_suppressed": self.n_empty_block_slots_suppressed,
            "n_empty_condition_slots": self.n_empty_condition_slots,
            "n_condition_routing_blocks": self.n_condition_routing_blocks,
            "n_collapsed_condition_ifs": self.n_collapsed_condition_ifs,
            "n_rejected_condition_if_collapses": self.n_rejected_condition_if_collapses,
            "rejected_condition_if_collapse_examples": self.rejected_condition_if_collapses[:12],
            "fallback_loop_headers": sorted(self.fallback_loop_headers),
            "fallback_switch_headers": sorted(self.fallback_switch_headers),
            "unstructured_loop_headers": [
                i for i, lab in enumerate(self.labels)
                if lab == int(PatternType.LOOP_HEADER)
                and i not in self.control_condition_bbs
                and i not in self.fallback_loop_headers
            ],
            "unstructured_switch_headers": [
                i for i, lab in enumerate(self.labels)
                if lab == int(PatternType.SWITCH_HEADER)
                and i not in self.control_condition_bbs
                and i not in self.fallback_switch_headers
            ],
        }
        return "\n".join(lines), self.slots, stats

    def emit_children(self, children: Sequence[CSTNode], indent: int, role: str) -> List[str]:
        lines: List[str] = []
        pending: List[int] = []

        def flush_pending() -> None:
            nonlocal pending
            bbs = _unique_bbs(pending)
            pending = []
            if not bbs:
                return
            bbs = [bb for bb in bbs if bb not in self.claimed_bbs]
            if not bbs:
                self.n_empty_block_slots_suppressed += 1
                return
            sid = self.new_slot("FILL_BLOCK", role, bbs)
            pad = "    " * indent
            lines.append(f"{pad}<FILL_BLOCK:{sid}>  {_bb_comment(bbs)}")

        for child in children:
            if child.type == "statement":
                pending.extend(child.aligned_bbs)
                continue
            flush_pending()
            rendered = self.emit_control(child, indent)
            if rendered:
                lines.extend(rendered)
        flush_pending()
        return lines

    def emit_control(self, node: CSTNode, indent: int) -> List[str]:
        pad = "    " * indent
        cond_bbs = self.node_cond_bbs(node)
        self.control_condition_bbs.update(cond_bbs)

        if node.type in {"while_loop", "for_loop"}:
            sid = self.new_slot("FILL_COND", "loop_condition", cond_bbs)
            slot_cond_bbs = self.slot_bbs(sid)
            lines = [f"{pad}while (<FILL_COND:{sid}>) {{ {_bb_comment(slot_cond_bbs, int(PatternType.LOOP_HEADER))}"]
            body = self.emit_children(node.children, indent + 1, role="loop_body")
            if body:
                lines.extend(body)
            else:
                bid = self.new_slot("FILL_BLOCK", "loop_body", [])
                lines.append(f"{'    ' * (indent + 1)}<FILL_BLOCK:{bid}>  /* empty loop body slot */")
            lines.append(f"{pad}}}")
            return lines

        if node.type == "switch":
            sid = self.new_slot("FILL_COND", "switch_condition", cond_bbs)
            slot_cond_bbs = self.slot_bbs(sid)
            lines = [f"{pad}switch (<FILL_COND:{sid}>) {{ {_bb_comment(slot_cond_bbs, int(PatternType.SWITCH_HEADER))}"]
            body = self.emit_children(node.children, indent + 1, role="switch_body")
            if body:
                lines.extend(body)
            else:
                bid = self.new_slot("FILL_BLOCK", "switch_body", [])
                lines.append(f"{'    ' * (indent + 1)}<FILL_BLOCK:{bid}>  /* empty switch body slot */")
            lines.append(f"{pad}}}")
            return lines

        if node.type == "if_else":
            cond_bbs, then_children, else_children = self.collapse_condition_only_if_chain(node)
            self.control_condition_bbs.update(cond_bbs)
            if not then_children and not else_children:
                self.n_condition_routing_blocks += 1
                sid = self.new_slot("FILL_BLOCK", "condition_routing_block", cond_bbs)
                slot_bbs = self.slot_bbs(sid)
                return [f"{pad}<FILL_BLOCK:{sid}>  {_bb_comment(slot_bbs)}  /* condition-only CFG routing */"]
            sid = self.new_slot("FILL_COND", "if_condition", cond_bbs)
            slot_cond_bbs = self.slot_bbs(sid)

            lines = [f"{pad}if (<FILL_COND:{sid}>) {{ {_bb_comment(slot_cond_bbs, int(PatternType.IF_HEADER))}"]
            then_body = self.emit_children(then_children, indent + 1, role="if_then_body")
            if then_body:
                lines.extend(then_body)
            else:
                bid = self.new_slot("FILL_BLOCK", "if_then_body", [])
                lines.append(f"{'    ' * (indent + 1)}<FILL_BLOCK:{bid}>  /* empty then body slot */")
            lines.append(f"{pad}}}")
            if else_children:
                lines[-1] += " else {"
                else_body = self.emit_children(else_children, indent + 1, role="if_else_body")
                if else_body:
                    lines.extend(else_body)
                else:
                    bid = self.new_slot("FILL_BLOCK", "if_else_body", [])
                    lines.append(f"{'    ' * (indent + 1)}<FILL_BLOCK:{bid}>  /* empty else body slot */")
                lines.append(f"{pad}}}")
            return lines

        # Unknown control-like node: fall back to one region block.
        bbs = _unique_bbs(node.aligned_bbs)
        sid = self.new_slot("FILL_BLOCK", f"unknown_{node.type}", bbs)
        return [f"{pad}<FILL_BLOCK:{sid}>  {_bb_comment(bbs)}"]

    def fallback_body_bbs(self, header: int, body_label: int, max_scan: int = 48, max_take: int = 16) -> List[int]:
        bbs: List[int] = []
        for i in range(int(header) + 1, min(len(self.labels), int(header) + 1 + max_scan)):
            if self.labels[i] in {
                int(PatternType.LOOP_HEADER),
                int(PatternType.SWITCH_HEADER),
            }:
                break
            if self.labels[i] == body_label:
                bbs.append(i)
                if len(bbs) >= max_take:
                    break
        return bbs

    def emit_unstructured_loop_switch_fallbacks(self, indent: int) -> List[str]:
        lines: List[str] = []
        pad = "    " * indent
        for h, lab in enumerate(self.labels):
            if h in self.control_condition_bbs:
                continue
            if lab == int(PatternType.LOOP_HEADER):
                self.fallback_loop_headers.add(int(h))
                sid = self.new_slot("FILL_COND", "fallback_loop_condition", [h])
                slot_cond_bbs = self.slot_bbs(sid)
                body_bbs = self.fallback_body_bbs(h, int(PatternType.LOOP_BODY))
                bid = self.new_slot("FILL_BLOCK", "fallback_loop_body", body_bbs)
                slot_body_bbs = self.slot_bbs(bid)
                lines.append(f"{pad}while (<FILL_COND:{sid}>) {{ {_bb_comment(slot_cond_bbs, lab)}")
                lines.append(f"{pad}    <FILL_BLOCK:{bid}>  {_bb_comment(slot_body_bbs)}")
                lines.append(f"{pad}}}")
            elif lab == int(PatternType.SWITCH_HEADER):
                self.fallback_switch_headers.add(int(h))
                sid = self.new_slot("FILL_COND", "fallback_switch_condition", [h])
                slot_cond_bbs = self.slot_bbs(sid)
                body_bbs = self.fallback_body_bbs(h, int(PatternType.SWITCH_CASE_BODY))
                bid = self.new_slot("FILL_BLOCK", "fallback_switch_body", body_bbs)
                slot_body_bbs = self.slot_bbs(bid)
                lines.append(f"{pad}switch (<FILL_COND:{sid}>) {{ {_bb_comment(slot_cond_bbs, lab)}")
                lines.append(f"{pad}    <FILL_BLOCK:{bid}>  {_bb_comment(slot_body_bbs)}")
                lines.append(f"{pad}}}")
        return lines


def region_skeleton_from_pred_labels(
    row: Dict[str, Any],
    labels: Sequence[int],
) -> Tuple[str, List[RegionSlot], Dict[str, Any], Dict[str, int], List[int], Dict[str, Any]]:
    bei = row.get("bin_edge_index") or [[], []]
    edge_index = torch.tensor(
        [[int(x) for x in bei[0]], [int(x) for x in bei[1]]],
        dtype=torch.int64,
    )
    refined_labels, refine_stats = refine_labels_for_region_builder(row, labels)
    pl_t = torch.tensor([int(x) for x in refined_labels], dtype=torch.int64)
    cst, build_stats = build_control_structure_tree_with_stats(pl_t, edge_index, len(labels))
    hints = sft_aug._bb_opcode_hints(row)
    serializer = RegionSerializer(refined_labels, hints, row)
    text, slots, region_stats = serializer.serialize(cst, fallback_headers=True)
    return text, slots, region_stats, build_stats, refined_labels, refine_stats


def control_meta_from_skeleton(text: str, nbb: int) -> Dict[str, Any]:
    pred_seq = effect._control_outline_from_text(text)
    pred_counts = effect._counts(pred_seq)
    return {
        "pred_control_count": len(pred_seq),
        "pred_counts": dict(pred_counts),
        "pred_outline": effect._outline_lines(pred_seq, limit=200),
        "bb_stats": effect._bb_marker_stats(text, nbb),
    }


def slot_map_text(slots: Sequence[RegionSlot], max_context_chars: int = 600) -> str:
    lines: List[str] = []
    for slot in slots:
        bbs = ", ".join(f"BB_{bb}" for bb in slot.aligned_bbs) if slot.aligned_bbs else "<none>"
        ctx_bbs = ", ".join(f"BB_{bb}" for bb in slot.context_bbs) if slot.context_bbs else "<none>"
        ctx = slot.context
        if len(ctx) > max_context_chars:
            ctx = ctx[:max_context_chars] + " ..."
        lines.append(
            f"{slot.slot_id} {slot.slot_type} role={slot.role} "
            f"primary_bbs=[{bbs}] context_bbs=[{ctx_bbs}]"
        )
        if ctx:
            lines.append(f"  asm_context: {ctx}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--stage2-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--source-control-filter",
        choices=["raw", "bb-start-line", "bb-range"],
        default="bb-start-line",
    )
    ap.add_argument("--min-source-controls", type=int, default=1)
    ap.add_argument("--candidate-pool-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=530)
    args = ap.parse_args()

    jsonl_path = args.jsonl.expanduser().resolve()
    ckpt_path = args.stage2_ckpt.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("===== first source-control functions, region-slot skeleton export =====", flush=True)
    print("jsonl =", jsonl_path, flush=True)
    print("ckpt =", ckpt_path, flush=True)
    print("out =", out_path, flush=True)
    print("device =", device, flush=True)

    t0 = time.time()
    candidates: List[Tuple[int, Dict[str, Any], str, Tuple[int, int]]] = []
    seen = set()
    scanned = 0
    skipped: Counter[str] = Counter()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            scanned += 1
            if args.limit and scanned > int(args.limit):
                break
            try:
                row = json.loads(line)
            except Exception:
                skipped["bad_json"] += 1
                continue
            if not is_valid_bin(row):
                skipped["bad_bin"] += 1
                continue
            src_text, bounds = source_function_with_numbers(row)
            if bounds is None:
                skipped["source_bounds_unavailable"] += 1
                continue
            key = (str(row.get("source_file") or ""), str(row.get("func_name") or ""), bounds)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            if not has_control_by_source(row, str(args.source_control_filter)):
                skipped["no_source_control"] += 1
                continue
            smeta = source_control_meta(row, str(args.source_control_filter))
            if int(smeta.get("source_control_count") or 0) < int(args.min_source_controls):
                skipped["below_min_source_controls"] += 1
                continue
            seen.add(key)
            candidates.append((line_no, row, src_text, bounds))
            if len(candidates) >= int(args.num_examples):
                break

    print(
        f"selected candidates={len(candidates)} scanned={scanned} "
        f"skipped={dict(skipped)} select_sec={time.time() - t0:.1f}",
        flush=True,
    )

    model = load_model(ckpt_path, device)
    pred_map: Dict[int, Tuple[List[int], List[float]]] = {}
    bs = max(1, int(args.batch_size))
    for start in range(0, len(candidates), bs):
        chunk = candidates[start : start + bs]
        rows = [x[1] for x in chunk]
        preds = predict_labels_batch(model, rows, device)
        for item, pred in zip(chunk, preds):
            pred_map[item[0]] = pred
        print(f"predicted {min(start + bs, len(candidates))}/{len(candidates)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        wf.write("===== REGION-SLOT MODULE-A CONTROL SKELETON VS REAL SOURCE EXPORT =====\n")
        wf.write(f"input = {jsonl_path}\n")
        wf.write(f"ckpt = {ckpt_path}\n")
        wf.write("skeleton_source = ModuleA argmax + CST + region-level serializer; B/alignment is NOT run\n")
        wf.write("slot_policy = control condition is FILL_COND; consecutive non-control BBs merge into FILL_BLOCK; primary_bbs are single-owner; context_bbs are read-only and may repeat\n")
        wf.write("source_reliability_policy = unreliable source bounds are flagged for audit only; skeleton construction is source-free and uses only ModuleA labels plus binary CFG\n")
        wf.write(f"device = {device}\n")
        wf.write(f"scanned = {scanned}\n")
        wf.write(f"exported = {len(candidates)}\n")
        wf.write(f"requested = {int(args.num_examples)}\n")
        wf.write(f"skipped = {dict(skipped)}\n")

        for idx, (line_no, row, src_text, bounds) in enumerate(candidates):
            if idx % 10 == 0:
                print(f"writing example {idx}/{len(candidates)}", flush=True)
            labels, conf = pred_map[line_no]
            try:
                smeta = source_control_meta(row, str(args.source_control_filter))
                source_aux = source_reliability_meta(
                    row,
                    src_text,
                    bounds,
                    smeta,
                    str(args.source_control_filter),
                )
                (
                    skeleton_text,
                    slots,
                    region_stats,
                    build_stats,
                    refined_labels,
                    refine_stats,
                ) = region_skeleton_from_pred_labels(row, labels)
                pmeta = control_meta_from_skeleton(skeleton_text, len(labels))
                src_counts = Counter(smeta.get("source_counts") or {})
                pred_counts = Counter(pmeta.get("pred_counts") or {})
                bb_stats = pmeta.get("bb_stats") or {}

                vals = [float(x) for x in conf]
                vals_sorted = sorted(vals)
                med = vals_sorted[len(vals_sorted) // 2] if vals_sorted else 0.0
                conf_line = (
                    f"min/median/max={min(vals):.3f}/{med:.3f}/{max(vals):.3f}"
                    if vals else "n/a"
                )

                wf.write("\n" + "=" * 120 + "\n")
                wf.write(f"example = {idx}  input_line = {line_no}  func = {row.get('func_name')}\n")
                wf.write(f"source_file = {row.get('source_file')}\n")
                wf.write(f"source_bounds = {bounds[0]}-{bounds[1]}\n")
                wf.write("skeleton_source = ModuleA_argmax_plus_region_slot_builder_no_B_alignment\n")
                wf.write(f"bb = {len(labels)}\n")
                wf.write(
                    f"source_parse = {smeta.get('source_parse_status')}  "
                    f"source_ref = {smeta.get('source_outline_source')}  "
                    f"src_rows = {smeta.get('source_statement_rows')}  "
                    f"active_source_lines = {smeta.get('active_source_lines')}\n"
                )
                wf.write(f"source_reliability = {compact_source_reliability_for_log(source_aux)}\n")
                wf.write(
                    f"ctrl(pred/source) = {pmeta.get('pred_control_count')}/{smeta.get('source_control_count')}  "
                    f"gaps(if,loop,switch)=("
                    f"{int(pred_counts.get('if', 0)) - int(src_counts.get('if', 0))},"
                    f"{int(pred_counts.get('loop', 0)) - int(src_counts.get('loop', 0))},"
                    f"{int(pred_counts.get('switch', 0)) - int(src_counts.get('switch', 0))})\n"
                )
                wf.write(
                    f"bb_cover = {bb_stats.get('bb_covered')}/{len(labels)}  "
                    f"bb_missing = {bb_stats.get('bb_missing')}  "
                    f"bb_duplicated_first20 = {bb_stats.get('bb_duplicated')}\n"
                )
                wf.write(f"ab_pred_10class_label_counts = {label_summary(labels)}\n")
                wf.write(f"refined_10class_label_counts = {label_summary(refined_labels)}\n")
                wf.write(f"ab_pred_10class_label_confidence = {conf_line}\n")
                wf.write(f"region_refine_stats = {refine_stats}\n")
                wf.write(f"cst_build_stats = {build_stats}\n")
                wf.write(f"region_slot_stats = {region_stats}\n")

                wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
                wf.write("\n".join(smeta.get("source_outline") or ["<none>"]) + "\n")

                wf.write("\n-- REGION SKELETON CONTROL OUTLINE --\n")
                wf.write("\n".join(pmeta.get("pred_outline") or ["<none>"]) + "\n")

                wf.write("\n-- AB PREDICTED 10CLASS BB LABEL SEQUENCE --\n")
                wf.write(label_sequence(labels) + "\n")

                wf.write("\n-- REFINED 10CLASS BB LABEL SEQUENCE --\n")
                wf.write(label_sequence(refined_labels) + "\n")

                wf.write("\n-- REGION SLOT TO BB MAP --\n")
                wf.write(slot_map_text(slots) + "\n")

                wf.write("\n-- REAL SOURCE FUNCTION --\n")
                wf.write(src_text.rstrip() + "\n")

                wf.write("\n-- REGION-LEVEL GENERATED CONTROL SKELETON --\n")
                wf.write(skeleton_text.rstrip() + "\n")
            except Exception as e:
                wf.write("\n" + "=" * 120 + "\n")
                wf.write(f"example = {idx}  input_line = {line_no}  func = {row.get('func_name')}\n")
                wf.write(f"ERROR writing example: {type(e).__name__}: {e}\n")

    print("saved_to =", out_path, flush=True)
    print("elapsed_total_sec =", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()

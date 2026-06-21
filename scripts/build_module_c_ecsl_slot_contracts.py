#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Module-C ECSL O0 SemanticSlot contracts from Module-B v3 output.

This is the first, deliberately conservative ECSL bridge:

* it treats the verified Module-B skeleton as fixed input;
* it lowers skeleton placeholders to slot anchors when skeleton text is present;
* it builds one SemanticSlot-lite contract per Module-B slot;
* it records what is proven by Module B and what is still unavailable in O0;
* it assigns a training policy tier without projecting source AST labels.

The script does not synthesize teacher labels, does not modify the skeleton, and
does not claim C3+ validation.  Later ECSL stages can attach candidate labels and
multi-level validator results to these contracts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA = "ModuleC.ECSL.SemanticSlotContractSet.v0"
SLOT_SCHEMA = "SemanticSlotLite.v0"
LOWERER_SCHEMA = "SkeletonSlotLowerer.v0"

ESCAPE_SLOT_TYPES = {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}
STRUCTURAL_SLOT_TYPES = {"COND", "BLOCK", "CASE"}
UNSUPPORTED_SLOT_TYPES = {"UNSTRUCTURED_REGION"}
DOC_SLOT_TYPES = {
    "COND",
    "LOOP_EXIT",
    "SWITCH_DISPATCH",
    "CASE_LABEL",
    "EXPR",
    "LVALUE",
    "RETURN",
    "CALL_ARG",
    "CALL_TARGET",
    "MEMORY_EFFECT",
    "PHI_OR_MERGE",
    "STMT",
    "BLOCK",
    "EXIT",
}

PLACEHOLDER_RE = re.compile(r"<(?P<tag>[A-Z_]+):(?P<slot_ref>slot_(?P<sid>\d+))>")

ATTENTION_WEIGHTS = {
    "COND": {"region": 0.55, "boundary": 0.35, "dataflow": 0.10},
    "SWITCH_DISPATCH": {"region": 0.45, "boundary": 0.40, "dataflow": 0.15},
    "BLOCK": {"region": 0.50, "boundary": 0.15, "dataflow": 0.35},
    "STMT": {"region": 0.50, "boundary": 0.15, "dataflow": 0.35},
    "EXIT": {"region": 0.45, "boundary": 0.40, "dataflow": 0.15},
    "LOOP_EXIT": {"region": 0.45, "boundary": 0.40, "dataflow": 0.15},
    "LOOP_CONTINUE": {"region": 0.45, "boundary": 0.40, "dataflow": 0.15},
    "RETURN": {"region": 0.45, "boundary": 0.10, "dataflow": 0.45},
    "PHI_OR_MERGE": {"region": 0.10, "boundary": 0.45, "dataflow": 0.45},
}


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def as_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    out: List[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def compact_counter(counter: Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    head = items[:limit]
    head.append(("__other__", sum(v for _, v in items[limit:])))
    return head


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def q(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def verified_skeleton(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("verified_skeleton")
    return value if isinstance(value, dict) else {}


def selected_report(row: Dict[str, Any]) -> Dict[str, Any]:
    rep = q(row, "validator", "selected_report")
    if isinstance(rep, dict):
        return rep
    rep = q(row, "verified_skeleton", "validator_report")
    return rep if isinstance(rep, dict) else {}


def structuredness(row: Dict[str, Any]) -> Dict[str, Any]:
    value = q(row, "document_metric_contract", "structuredness")
    if isinstance(value, dict):
        return value
    value = q(row, "teacher_diagnostics", "ranker_metrics")
    return value if isinstance(value, dict) else {}


def function_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("function_ir")
    return value if isinstance(value, dict) else {}


def slot_grounding(row: Dict[str, Any]) -> Dict[str, Any]:
    value = q(row, "verified_skeleton", "slot_grounding")
    return value if isinstance(value, dict) else {}


def slot_pointers(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    grounding = slot_grounding(row)
    out: Dict[int, Dict[str, Any]] = {}
    for ptr in grounding.get("pointers") or []:
        if not isinstance(ptr, dict):
            continue
        sid = safe_int(ptr.get("slot_id"), -1)
        if sid >= 0:
            out[sid] = ptr
    return out


def edge_dicts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    fi = function_ir(row)
    edges = fi.get("cfg_edges")
    if isinstance(edges, list) and edges:
        out = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = safe_int(edge.get("src"), -1)
            dst = safe_int(edge.get("dst"), -1)
            if src < 0 or dst < 0:
                continue
            slim = {
                "src": src,
                "dst": dst,
                "kind": edge.get("kind"),
                "condition_ssa": edge.get("condition_ssa"),
                "provenance_instruction": edge.get("provenance_instruction"),
                "confidence": edge.get("confidence"),
            }
            out.append(slim)
        if out:
            return out

    raw_edges = row.get("bin_edge_index")
    if not (isinstance(raw_edges, list) and len(raw_edges) == 2):
        return []
    srcs = raw_edges[0] if isinstance(raw_edges[0], list) else []
    dsts = raw_edges[1] if isinstance(raw_edges[1], list) else []
    edge_types = row.get("bin_edge_type") if isinstance(row.get("bin_edge_type"), list) else []
    out = []
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        s = safe_int(src, -1)
        d = safe_int(dst, -1)
        if s < 0 or d < 0:
            continue
        out.append(
            {
                "src": s,
                "dst": d,
                "kind": edge_types[idx] if idx < len(edge_types) else None,
                "condition_ssa": None,
                "provenance_instruction": None,
                "confidence": None,
            }
        )
    return out


def basic_block_preview(row: Dict[str, Any], bbs: Sequence[int], *, max_blocks: int = 12) -> List[Dict[str, Any]]:
    fi = function_ir(row)
    blocks = fi.get("basic_blocks")
    if not isinstance(blocks, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw_bb in bbs:
        bb = safe_int(raw_bb, -1)
        if bb in seen or not (0 <= bb < len(blocks)):
            continue
        seen.add(bb)
        block = blocks[bb] if isinstance(blocks[bb], dict) else {}
        out.append(
            {
                "bb_id": bb,
                "stable_id": block.get("stable_id") or f"bb_{bb}",
                "terminator": block.get("terminator"),
                "instruction_index_range": block.get("instruction_index_range"),
                "instruction_count": block.get("instruction_count"),
                "address_range": block.get("address_range"),
            }
        )
        if len(out) >= max_blocks:
            break
    return out


def instruction_index(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fi = function_ir(row)
    out: Dict[str, Dict[str, Any]] = {}
    for item in fi.get("lifted_instructions") or row.get("lifted_instructions") or []:
        if not isinstance(item, dict):
            continue
        instr_id = str(item.get("instruction_id") or item.get("id") or "")
        if instr_id:
            out[instr_id] = item
    return out


def branch_metadata_index(row: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    fi = function_ir(row)
    out: Dict[int, Dict[str, Any]] = {}
    for item in fi.get("branch_metadata") or row.get("branch_metadata") or []:
        if not isinstance(item, dict):
            continue
        bb = safe_int(item.get("bb_id"), -1)
        if bb >= 0:
            out[bb] = item
    return out


def function_ir_availability(row: Dict[str, Any]) -> Dict[str, Any]:
    fi = function_ir(row)
    src = fi.get("availability_source") if isinstance(fi.get("availability_source"), dict) else {}
    lifted_src = str(src.get("lifted_instructions") or "").lower()
    ssa_src = str(src.get("ssa") or fi.get("ssa_level") or "").lower()
    def_use_src = str(src.get("def_use") or "").lower()
    real_lifted = lifted_src == "real" or lifted_src.startswith("binary")
    real_ssa = ssa_src == "real" or ssa_src.startswith("binary")
    real_def_use = def_use_src == "real" or def_use_src.startswith("binary")
    return {
        "lifted_instructions": str(src.get("lifted_instructions") or "unknown"),
        "ssa": str(src.get("ssa") or fi.get("ssa_level") or "unknown"),
        "def_use": str(src.get("def_use") or "unknown"),
        "instruction_provenance": str(src.get("instruction_provenance") or "unknown"),
        "lifted_instructions_available": bool(fi.get("lifted_instructions_available")),
        "ssa_available": bool(fi.get("ssa_available")),
        "def_use_available": bool(fi.get("def_use_available")),
        "call_metadata_available": bool(fi.get("call_metadata_available")),
        "memory_access_summary_available": bool(fi.get("memory_access_summary_available")),
        "branch_metadata_available": bool(fi.get("branch_metadata_available") or fi.get("branch_metadata")),
        "real_lifted_available": bool(real_lifted),
        "real_ssa_available": bool(real_ssa or any(
            isinstance(v, dict) and v.get("is_proxy") is False
            for v in (fi.get("ssa_values") or row.get("ssa_values") or [])
        )),
        "real_def_use_available": bool(real_def_use),
    }


def instruction_ids_for_bbs(row: Dict[str, Any], bbs: Sequence[int]) -> List[str]:
    fi = function_ir(row)
    blocks = fi.get("basic_blocks") if isinstance(fi.get("basic_blocks"), list) else []
    out: List[str] = []
    seen = set()
    for raw_bb in bbs:
        bb = safe_int(raw_bb, -1)
        if not (0 <= bb < len(blocks)):
            continue
        block = blocks[bb] if isinstance(blocks[bb], dict) else {}
        ids = block.get("instruction_ids") if isinstance(block.get("instruction_ids"), list) else []
        if not ids and isinstance(block.get("instruction_index_range"), list) and len(block.get("instruction_index_range")) >= 2:
            start = safe_int(block["instruction_index_range"][0], -1)
            end = safe_int(block["instruction_index_range"][1], -1)
            ids = [f"bb_{bb}:ins_{i}" for i in range(start, max(start, end))]
        for item in ids:
            sid = str(item)
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def ssa_value_index(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fi = function_ir(row)
    out: Dict[str, Dict[str, Any]] = {}
    for item in fi.get("ssa_values") or row.get("ssa_values") or []:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("ssa_id") or item.get("id") or item.get("name") or "")
        if sid:
            out[sid] = item
    return out


def ssa_def_provenance(row: Dict[str, Any], ssa_id: Any) -> Dict[str, Any]:
    sid = str(ssa_id or "").strip()
    if not sid:
        return {"status": "missing_ssa_id"}
    ssa = ssa_value_index(row).get(sid) or {}
    instr_id = str(ssa.get("defined_by_instruction") or "")
    instructions = instruction_index(row)
    instr = instructions.get(instr_id) if instr_id else None
    status = "available" if isinstance(instr, dict) and instr else "missing_def_instruction"
    return {
        "status": status,
        "ssa_id": sid,
        "register": ssa.get("register"),
        "ssa_kind": ssa.get("kind"),
        "defined_by_instruction": instr_id or None,
        "def_bb_id": ssa.get("bb_id"),
        "def_instruction": instr if isinstance(instr, dict) else None,
        "def_text": instr.get("text") if isinstance(instr, dict) else None,
        "def_mnemonic": instr.get("mnemonic") if isinstance(instr, dict) else None,
        "def_operands": instr.get("operands") if isinstance(instr, dict) else None,
        "is_proxy": bool(ssa.get("is_proxy")) if isinstance(ssa, dict) else None,
    }


def condition_branch_provenance(row: Dict[str, Any], ptr: Dict[str, Any]) -> Dict[str, Any]:
    cond = ptr.get("condition_ssa")
    primary = as_int_list(ptr.get("primary_bbs"))
    condition_bb = safe_int(ptr.get("condition_bb"), primary[0] if primary else -1)
    meta = branch_metadata_index(row).get(condition_bb) or {}
    branch_id = str(ptr.get("branch_instruction") or meta.get("branch_instruction") or "")
    branch_instr = instruction_index(row).get(branch_id) if branch_id else None
    outgoing = ptr.get("outgoing_edges") or ptr.get("function_ir_outgoing_edges") or meta.get("outgoing_edges") or []
    return {
        "schema": "ModuleC.ECSL.ConditionProvenance.v0",
        "status": "available" if cond and branch_id and branch_instr else "partial_or_missing",
        "condition_ssa": cond,
        "ssa_def": ssa_def_provenance(row, cond),
        "condition_bb": condition_bb if condition_bb >= 0 else None,
        "branch_instruction": branch_id or None,
        "branch_instruction_record": branch_instr if isinstance(branch_instr, dict) else None,
        "branch_text": branch_instr.get("text") if isinstance(branch_instr, dict) else meta.get("branch_text"),
        "branch_mnemonic": branch_instr.get("mnemonic") if isinstance(branch_instr, dict) else meta.get("branch_mnemonic"),
        "branch_address": branch_instr.get("address") if isinstance(branch_instr, dict) else meta.get("branch_address"),
        "outgoing_edges": outgoing if isinstance(outgoing, list) else [],
        "true_false_polarity_available": bool(ptr.get("true_false_polarity_available") or meta.get("true_false_polarity_available")),
        "source": meta.get("source") or "function_ir",
    }


def def_use_maps(row: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    fi = function_ir(row)
    du = fi.get("def_use_graph") or row.get("def_use_graph") or row.get("ssa_def_use") or {}
    uses: Dict[str, List[str]] = {}
    defs: Dict[str, List[str]] = {}
    if isinstance(du, dict):
        raw_uses = du.get("instruction_uses")
        if isinstance(raw_uses, dict):
            for key, vals in raw_uses.items():
                uses[str(key)] = [str(x) for x in (vals if isinstance(vals, list) else [vals]) if x not in (None, "", [], {})]
        raw_defs = du.get("instruction_defs")
        if isinstance(raw_defs, dict):
            for key, vals in raw_defs.items():
                defs[str(key)] = [str(x) for x in (vals if isinstance(vals, list) else [vals]) if x not in (None, "", [], {})]
        for edge in du.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            kind = str(edge.get("kind") or "")
            d = edge.get("def")
            u = edge.get("use")
            if kind.endswith("read") or kind in {"reg_read", "mem_read"}:
                if u not in (None, "", [], {}) and d not in (None, "", [], {}):
                    uses.setdefault(str(u), []).append(str(d))
            elif kind.endswith("write") or kind in {"reg_write", "mem_write"}:
                if d not in (None, "", [], {}) and u not in (None, "", [], {}):
                    defs.setdefault(str(d), []).append(str(u))
            else:
                if isinstance(d, str) and d.startswith("bb_") and u not in (None, "", [], {}):
                    defs.setdefault(d, []).append(str(u))
    return uses, defs


def unique_keep_order(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def instruction_use_ids(value: Any) -> List[str]:
    """Best-effort extraction of instruction ids from SSA use records."""
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        for key in ("instruction_id", "instr_id", "id", "use_instruction", "used_by_instruction"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                out.extend(instruction_use_ids(item))
        return out
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(instruction_use_ids(item))
    return out


def dataflow_summary_for_region(row: Dict[str, Any], bbs: Sequence[int]) -> Dict[str, Any]:
    availability = function_ir_availability(row)
    instr_ids = instruction_ids_for_bbs(row, bbs)
    instr_id_set = set(instr_ids)
    uses_by_instr, defs_by_instr = def_use_maps(row)
    raw_defs: List[str] = []
    raw_uses: List[str] = []
    for instr_id in instr_ids:
        raw_defs.extend(defs_by_instr.get(instr_id, []))
        raw_uses.extend(uses_by_instr.get(instr_id, []))

    # Build observable live-in/live-out facts instead of treating every
    # register-temporary def/use as a source-level effect.  ECSL Obs(slot)
    # cares about values that cross the slot boundary; internal SSA artifacts
    # are recorded for audit but are not hard labels.
    bb_set = {safe_int(x, -1) for x in bbs}
    live_in: List[str] = []
    live_out: List[str] = []
    internal_defs: List[str] = []
    internal_uses: List[str] = []
    for sid, item in ssa_value_index(row).items():
        bb = safe_int(item.get("bb_id"), -999999)
        defined_by = str(item.get("defined_by_instruction") or "")
        use_ids = instruction_use_ids(item.get("uses"))
        defined_in_region = bb in bb_set or (defined_by and defined_by in instr_id_set)
        used_in_region = any(uid in instr_id_set for uid in use_ids)
        used_outside_region = any(uid and uid not in instr_id_set for uid in use_ids)
        if defined_in_region:
            raw_defs.append(sid)
            if used_outside_region:
                live_out.append(sid)
            else:
                internal_defs.append(sid)
        if used_in_region:
            raw_uses.append(sid)
            if not defined_in_region:
                live_in.append(sid)
            else:
                internal_uses.append(sid)

    raw_defs = unique_keep_order(raw_defs)
    raw_uses = unique_keep_order(raw_uses)
    live_in = unique_keep_order(live_in)
    live_out = unique_keep_order(live_out)
    internal_defs = unique_keep_order(internal_defs)
    internal_uses = unique_keep_order(internal_uses)
    status = "not_available_o0"
    fact_state = "not_available"
    if live_out:
        status = "available_from_real_function_ir" if availability["real_ssa_available"] else "partial_from_proxy_function_ir"
        fact_state = "nonempty_real" if availability["real_ssa_available"] else "nonempty_proxy"
    elif instr_ids and (availability.get("real_ssa_available") or availability.get("real_def_use_available")):
        status = "available_empty_from_real_function_ir"
        fact_state = "proven_empty_real"
    elif instr_ids and (availability.get("ssa_available") or availability.get("def_use_available")):
        status = "partial_empty_from_proxy_function_ir"
        fact_state = "empty_proxy_needs_v3"
    return {
        "status": status,
        "fact_state": fact_state,
        "defs": live_out,
        "uses": live_in,
        "live_in": live_in,
        "live_out": live_out,
        "internal_defs": internal_defs,
        "internal_uses": internal_uses,
        "raw_defs": raw_defs,
        "raw_uses": raw_uses,
        "instruction_ids": instr_ids,
        "availability": availability,
    }


RETURN_SSA_RE = re.compile(r"^(?:r?ax|e?ax|ax|al|ret(?:urn)?(?:_value)?)(?:[@_.#-]?\d+)?$", re.I)


def infer_return_ssa_for_slot(slot: Dict[str, Any], flow: Dict[str, Any]) -> Optional[str]:
    """Best-effort return-value grounding for Module-B early-exit slots.

    This is still a contract field, not a label.  It only gives later candidate
    builders a binary-pivot symbol to test with V3.
    """
    typ = str(slot.get("slot_type") or "").upper()
    if typ not in {"EARLY_EXIT", "CLEANUP_EXIT"}:
        return None
    values: List[str] = []
    for key in ("defs", "live_out", "uses", "live_in", "raw_defs", "internal_defs"):
        values.extend(str(x) for x in (flow.get(key) or []) if str(x or "").strip())
    for value in values:
        text = str(value or "").strip()
        simple = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
        head = re.split(r"[@_.#-]", text, 1)[0]
        if "ret_value" in text.lower() or RETURN_SSA_RE.match(text) or RETURN_SSA_RE.match(simple) or RETURN_SSA_RE.match(head):
            return text
    return None


def infer_return_constant_from_def(row: Dict[str, Any], return_ssa: Optional[str]) -> Optional[str]:
    if not return_ssa:
        return None
    prov = ssa_def_provenance(row, return_ssa)
    text = str(prov.get("def_text") or "").strip().lower()
    if not text:
        return None
    # Common compiler idioms for integer return constants.
    if re.search(r"\bxor\s+e?ax\s*,\s*e?ax\b", text) or re.search(r"\bsub\s+e?ax\s*,\s*e?ax\b", text):
        return "0"
    m = re.search(r"\bmov(?:abs)?\s+[re]?ax\s*,\s*([-+]?(?:0x[0-9a-f]+|\d+))\b", text)
    if m:
        raw = m.group(1)
        try:
            return str(int(raw, 0))
        except Exception:
            return raw
    return None


def effect_summary_for_region(row: Dict[str, Any], bbs: Sequence[int]) -> Dict[str, Any]:
    fi = function_ir(row)
    instr_ids = set(instruction_ids_for_bbs(row, bbs))
    lifted = fi.get("lifted_instructions") if isinstance(fi.get("lifted_instructions"), list) else []
    read_set: List[str] = []
    write_set: List[str] = []
    register_reads: List[str] = []
    register_writes: List[str] = []
    calls: List[Dict[str, Any]] = []
    volatile_ops: List[str] = []

    for inst in lifted:
        if not isinstance(inst, dict):
            continue
        instr_id = str(inst.get("instruction_id") or "")
        if instr_id not in instr_ids:
            continue
        text = str(inst.get("text") or inst.get("mnemonic") or "").strip()
        for reg in inst.get("regs_read") or []:
            register_reads.append(str(reg))
        for reg in inst.get("regs_write") or []:
            register_writes.append(str(reg))
        groups = {str(x).lower() for x in (inst.get("groups") or [])}
        mnemonic = str(inst.get("mnemonic") or "").lower()
        operands_text = " ".join(str(x) for x in (inst.get("operands") or []))
        text_l = text.lower()
        # Register traffic is normally a compiler artifact and not a
        # source-visible memory/call effect.  Keep it for audit but only
        # promote explicit memory/global-like operands into read/write sets.
        if "[" in text or "]" in text or " ptr " in text_l or "rip" in text_l:
            if any(mnemonic.startswith(prefix) for prefix in ("mov", "cmp", "test", "lea", "push")):
                read_set.append(text or operands_text or instr_id)
            if mnemonic.startswith(("mov", "stos", "xchg", "cmpxchg")) and re.search(r"\[[^\]]+\]\s*,", text):
                write_set.append(text or operands_text or instr_id)
        if "call" in groups or mnemonic.startswith("call"):
            calls.append({"instruction_id": instr_id, "text": text, "effect": "call"})
        if "volatile" in text.lower():
            volatile_ops.append(instr_id)

    call_meta = fi.get("call_metadata") if isinstance(fi.get("call_metadata"), dict) else {}
    for call in call_meta.get("calls") or []:
        if isinstance(call, dict) and str(call.get("instruction_id") or "") in instr_ids:
            calls.append(call)

    mem = fi.get("memory_access_summary") if isinstance(fi.get("memory_access_summary"), dict) else {}
    memory_instruction_count = safe_int(mem.get("memory_instruction_count"), 0)
    availability = function_ir_availability(row)
    status = "not_available_o0"
    fact_state = "not_available"
    if read_set or write_set or calls or volatile_ops:
        status = "available_from_real_function_ir" if availability["real_lifted_available"] else "partial_from_proxy_function_ir"
        fact_state = "nonempty_real" if availability["real_lifted_available"] else "nonempty_proxy"
    elif instr_ids and availability.get("real_lifted_available") and (
        availability.get("call_metadata_available") or availability.get("memory_access_summary_available")
    ):
        status = "available_empty_from_real_function_ir"
        fact_state = "proven_empty_real"
    elif memory_instruction_count > 0 and instr_ids:
        status = "summary_only_memory_effects"
        fact_state = "summary_only_memory_effects"
    elif instr_ids and availability.get("lifted_instructions_available"):
        status = "partial_empty_from_proxy_function_ir"
        fact_state = "empty_proxy_needs_v3"
    return {
        "status": status,
        "fact_state": fact_state,
        "read_set": unique_keep_order(read_set),
        "write_set": unique_keep_order(write_set),
        "register_reads": unique_keep_order(register_reads),
        "register_writes": unique_keep_order(register_writes),
        "call_effects": calls,
        "volatile_ops": unique_keep_order(volatile_ops),
        "side_effect_order": [str(c.get("instruction_id") or c.get("text") or "") for c in calls if isinstance(c, dict)],
        "may_throw_or_longjmp": False,
        "memory_instruction_count": memory_instruction_count,
        "availability": availability,
    }


def edge_context(row: Dict[str, Any], bbs: Sequence[int]) -> Dict[str, List[Dict[str, Any]]]:
    bb_set = {safe_int(x, -1) for x in bbs}
    internal: List[Dict[str, Any]] = []
    incoming: List[Dict[str, Any]] = []
    outgoing: List[Dict[str, Any]] = []
    for edge in edge_dicts(row):
        src = safe_int(edge.get("src"), -1)
        dst = safe_int(edge.get("dst"), -1)
        if src in bb_set and dst in bb_set:
            internal.append(edge)
        elif src in bb_set:
            outgoing.append(edge)
        elif dst in bb_set:
            incoming.append(edge)
    return {"incoming": incoming, "internal": internal, "outgoing": outgoing}


def b_gate_report(row: Dict[str, Any], lowerer_success: bool, contract_conflicts: Sequence[str]) -> Dict[str, Any]:
    rep = selected_report(row)
    st = structuredness(row)
    components = {
        "validator_valid": bool(rep.get("valid")),
        "bb_coverage_ok": bool(rep.get("bb_coverage_ok")),
        "edge_preservation_ok": bool(rep.get("edge_preservation_ok")),
        "strict_edge_preservation_ok": bool(rep.get("strict_edge_preservation_ok")),
        "slot_grounding_ok": bool(rep.get("slot_grounding_ok")),
        "parse_success": bool(rep.get("parse_success")),
        "fatal_error_zero": not bool(rep.get("fatal_errors")),
        "slot_locality_ready": bool(st.get("slot_locality_ready")),
        "empty_control_with_context_zero": safe_int(rep.get("empty_control_with_context_count"), 0) == 0,
        "body_bb_outside_control_zero": safe_int(rep.get("body_bb_outside_control_count"), 0) == 0,
        "skeleton_slot_lowerer_success": bool(lowerer_success),
        "contract_consistency_pass": len(contract_conflicts) == 0,
    }
    required = [
        "validator_valid",
        "bb_coverage_ok",
        "edge_preservation_ok",
        "strict_edge_preservation_ok",
        "slot_grounding_ok",
        "parse_success",
        "fatal_error_zero",
        "slot_locality_ready",
        "empty_control_with_context_zero",
        "body_bb_outside_control_zero",
        "skeleton_slot_lowerer_success",
        "contract_consistency_pass",
    ]
    failed = [key for key in required if not components.get(key)]
    return {
        "pass": not failed,
        "failed_components": failed,
        "components": components,
        "raw_validator": {
            "valid": rep.get("valid"),
            "bb_coverage_ok": rep.get("bb_coverage_ok"),
            "edge_preservation_ok": rep.get("edge_preservation_ok"),
            "strict_edge_preservation_ok": rep.get("strict_edge_preservation_ok"),
            "slot_grounding_ok": rep.get("slot_grounding_ok"),
            "parse_success": rep.get("parse_success"),
            "fatal_errors": rep.get("fatal_errors") or [],
        },
        "locality": {
            "slot_locality_ready": st.get("slot_locality_ready"),
            "slot_locality_score": st.get("slot_locality_score"),
            "empty_control_count": rep.get("empty_control_count"),
            "empty_control_with_context_count": rep.get("empty_control_with_context_count"),
            "body_bb_outside_control_count": rep.get("body_bb_outside_control_count"),
        },
        "contract_conflicts": list(contract_conflicts),
    }


def strip_slot_comments(line: str) -> str:
    return re.sub(r"/\*.*?\*/", "", line)


def count_char(line: str, ch: str) -> int:
    return line.count(ch)


def classify_construct(line_without_comments: str, placeholder_tag: str) -> str:
    stripped = line_without_comments.strip()
    if stripped.startswith("if "):
        return "IF"
    if stripped.startswith("while "):
        return "LOOP"
    if stripped.startswith("switch "):
        return "SWITCH"
    if placeholder_tag == "CASE_SLOT":
        return "SWITCH_CASE"
    if placeholder_tag in {"GOTO_SLOT", "BREAK_SLOT", "CONTINUE_SLOT", "EARLY_EXIT_SLOT", "CLEANUP_EXIT_SLOT"}:
        return "CONTROL_TRANSFER"
    return "STMT"


def update_stack_before_line(stack: List[Dict[str, Any]], raw_line: str) -> None:
    code = strip_slot_comments(raw_line)
    stripped = code.strip()
    leading_closes = 0
    i = 0
    while i < len(stripped) and stripped[i] == "}":
        leading_closes += 1
        i += 1
        while i < len(stripped) and stripped[i].isspace():
            i += 1
    for _ in range(leading_closes):
        if len(stack) > 1:
            stack.pop()
    if "else" in stripped and stack and stack[-1]["kind"] == "IF":
        stack[-1]["kind"] = "ELSE"
        stack[-1]["path_segment"] = stack[-1]["path_segment"].replace("IF", "ELSE")


def update_stack_after_line(stack: List[Dict[str, Any]], raw_line: str, anchor_kind: Optional[str], line_no: int) -> None:
    code = strip_slot_comments(raw_line)
    stripped = code.strip()
    opens = count_char(code, "{")
    closes = count_char(code, "}")
    leading_closes = 0
    i = 0
    while i < len(stripped) and stripped[i] == "}":
        leading_closes += 1
        i += 1
        while i < len(stripped) and stripped[i].isspace():
            i += 1
    nonleading_closes = max(0, closes - leading_closes)
    pushes = max(0, opens - nonleading_closes)
    if pushes <= 0:
        return
    kind = anchor_kind or ("ELSE" if "else" in stripped else "BLOCK")
    for _ in range(pushes):
        idx = sum(1 for item in stack if item.get("kind") == kind)
        stack.append(
            {
                "kind": kind,
                "line": line_no,
                "path_segment": f"{kind}[{idx}]@L{line_no}",
            }
        )


def lower_skeleton_slots(skeleton: str, slots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Lower placeholder occurrences to AST-path-like anchors.

    The lowerer is intentionally text based because Module-B already rendered a
    balanced skeleton and this O0 bridge must not modify the skeleton AST.  It
    records enough location metadata for later deterministic render/validation.
    """
    slot_ids = {safe_int(slot.get("slot_id"), -1) for slot in slots if safe_int(slot.get("slot_id"), -1) >= 0}
    anchors: Dict[int, Dict[str, Any]] = {}
    issues: List[str] = []
    if not skeleton:
        return {
            "schema": LOWERER_SCHEMA,
            "success": False,
            "anchors": anchors,
            "issues": ["missing_verified_skeleton_text"],
            "slot_count": len(slot_ids),
            "lowered_slot_count": 0,
        }

    stack: List[Dict[str, Any]] = [{"kind": "FUNCTION", "line": 1, "path_segment": "FUNCTION[0]"}]
    dfs_order = 0
    for line_no, raw_line in enumerate(skeleton.splitlines(), 1):
        update_stack_before_line(stack, raw_line)
        code = strip_slot_comments(raw_line)
        line_anchor_kind: Optional[str] = None
        for match in PLACEHOLDER_RE.finditer(raw_line):
            sid = safe_int(match.group("sid"), -1)
            tag = match.group("tag")
            placeholder = match.group(0)
            construct = classify_construct(code, tag)
            parent = stack[-1] if stack else {"kind": "UNKNOWN", "path_segment": "UNKNOWN"}
            path = "/".join(str(item.get("path_segment")) for item in stack)
            if construct in {"IF", "LOOP", "SWITCH"}:
                path = f"{path}/{construct}@L{line_no}"
                line_anchor_kind = construct
            elif construct == "SWITCH_CASE":
                path = f"{path}/CASE_SLOT@L{line_no}"
            elif construct == "CONTROL_TRANSFER":
                path = f"{path}/CONTROL_TRANSFER@L{line_no}"
            else:
                path = f"{path}/STMT_SLOT@L{line_no}"
            if sid in anchors:
                issues.append(f"duplicate_placeholder_slot_{sid:03d}_line_{line_no}")
                continue
            anchors[sid] = {
                "skeleton_line": line_no,
                "indent": max(0, len(raw_line) - len(raw_line.lstrip(" "))),
                "dfs_order": dfs_order,
                "parent_construct": str(parent.get("kind") or "UNKNOWN"),
                "skeleton_ast_path": path,
                "placeholder_text": placeholder,
                "placeholder_tag": tag,
                "line_text": raw_line.rstrip(),
            }
            dfs_order += 1
        update_stack_after_line(stack, raw_line, line_anchor_kind, line_no)

    missing = sorted(slot_ids - set(anchors))
    extra = sorted(set(anchors) - slot_ids)
    for sid in missing:
        issues.append(f"slot_{sid:03d}_missing_placeholder_anchor")
    for sid in extra:
        issues.append(f"slot_{sid:03d}_placeholder_without_slot_metadata")
    return {
        "schema": LOWERER_SCHEMA,
        "success": not issues,
        "anchors": anchors,
        "issues": issues,
        "slot_count": len(slot_ids),
        "lowered_slot_count": len(anchors),
    }


def doc_slot_type(module_b_slot_kind: str, role: str) -> str:
    """Map Module-B skeleton slot kinds to the ECSL SemanticSlot enum.

    Module-B exposes action-oriented kinds such as BREAK/GOTO/CLEANUP_EXIT.
    The ECSL document uses semantic slot types.  Keep the original kind in
    ``module_b_slot_kind`` and report this document-aligned type separately so
    later validators can avoid both schema drift and action ambiguity.
    """
    typ = str(module_b_slot_kind or "").upper()
    role = str(role or "").lower()
    if typ == "COND":
        if "switch" in role:
            return "SWITCH_DISPATCH"
        return "COND"
    if typ == "BLOCK":
        return "BLOCK"
    if typ == "CASE":
        return "CASE_LABEL"
    if typ in {"BREAK", "CONTINUE"}:
        return "LOOP_EXIT"
    if typ == "EARLY_EXIT":
        return "RETURN"
    if typ in {"GOTO", "CLEANUP_EXIT"}:
        return "EXIT"
    if typ == "STMT":
        return "STMT"
    # The design doc has no UNSTRUCTURED_REGION enum.  Keep this as a BLOCK-like
    # unmatched region and let confidence/policy mark it C0 via module_b kind.
    if typ == "UNSTRUCTURED_REGION":
        return "BLOCK"
    return "STMT"


def semantic_slot_type(slot_type: str, role: str) -> str:
    return doc_slot_type(slot_type, role)


def grammar_category(slot_type: str, sem_type: str) -> str:
    typ = str(slot_type or "").upper()
    if typ == "COND":
        return "condition_only"
    if typ in {"BLOCK", "CASE"}:
        return "statement_list"
    if typ in ESCAPE_SLOT_TYPES:
        return "control_transfer"
    if typ == "UNSTRUCTURED_REGION":
        return "unsupported_region"
    if sem_type == "RETURN":
        return "return_expr"
    return "unknown"


def allowed_control_effects(slot_type: str) -> List[str]:
    typ = str(slot_type or "").upper()
    if typ == "BREAK":
        return ["break"]
    if typ == "CONTINUE":
        return ["continue"]
    if typ == "GOTO":
        return ["goto_like_internal"]
    if typ in {"EARLY_EXIT", "CLEANUP_EXIT"}:
        return ["return", "goto_like_internal"]
    if typ == "UNSTRUCTURED_REGION":
        return ["return", "break", "continue", "goto_like_internal"]
    return ["none"]


def forbidden_control_transfer(slot_type: str) -> List[str]:
    typ = str(slot_type or "").upper()
    if typ in {"COND"}:
        return ["return", "break", "continue", "goto"]
    if typ in {"BLOCK", "CASE"}:
        return ["goto"]
    if typ in ESCAPE_SLOT_TYPES:
        return []
    return ["return", "break", "continue", "goto"]


def pointer_for_slot(row: Dict[str, Any], slot: Dict[str, Any]) -> Dict[str, Any]:
    sid = safe_int(slot.get("slot_id"), -1)
    return slot_pointers(row).get(sid, {})


def infer_entry_bb(primary: Sequence[int], context: Sequence[int], edge: Optional[Sequence[int]]) -> Optional[str]:
    if primary:
        return f"BB_{safe_int(primary[0])}"
    if edge and len(edge) >= 1:
        return f"BB_{safe_int(edge[0])}"
    if context:
        return f"BB_{safe_int(context[0])}"
    return None


def infer_exit_bbs(row: Dict[str, Any], primary: Sequence[int], context: Sequence[int], edge: Optional[Sequence[int]]) -> List[str]:
    if edge and len(edge) >= 2:
        return [f"BB_{safe_int(edge[1])}"]
    bbs = list(primary or context)
    outgoing = edge_context(row, bbs).get("outgoing", [])
    return sorted({f"BB_{safe_int(e.get('dst'))}" for e in outgoing})


def covered_edges(row: Dict[str, Any], bbs: Sequence[int], edge: Optional[Sequence[int]]) -> List[List[Any]]:
    if edge and len(edge) >= 2:
        return [[safe_int(edge[0]), safe_int(edge[1]), "explicit_slot_edge"]]
    bb_set = {safe_int(x, -1) for x in bbs}
    out: List[List[Any]] = []
    for e in edge_dicts(row):
        src = safe_int(e.get("src"), -1)
        dst = safe_int(e.get("dst"), -1)
        if src in bb_set and dst in bb_set:
            out.append([src, dst, e.get("kind")])
    return out


def control_contract(row: Dict[str, Any], slot: Dict[str, Any], ptr: Dict[str, Any]) -> Dict[str, Any]:
    typ = str(slot.get("slot_type") or "").upper()
    edge = slot.get("edge")
    primary = as_int_list(slot.get("primary_bbs"))
    context = as_int_list(slot.get("context_bbs"))
    outgoing = ptr.get("outgoing_edges") or ptr.get("function_ir_outgoing_edges") or []

    true_succ: Optional[str] = None
    false_succ: Optional[str] = None
    fallthrough: Optional[str] = None
    if typ == "COND" and isinstance(outgoing, list):
        slim = [e for e in outgoing if isinstance(e, dict)]
        if len(slim) >= 1:
            true_succ = f"BB_{safe_int(slim[0].get('dst'))}"
        if len(slim) >= 2:
            false_succ = f"BB_{safe_int(slim[1].get('dst'))}"
        if len(slim) == 1:
            fallthrough = true_succ

    loop_header = None
    if typ == "COND" and str(slot.get("role") or "").lower().startswith("loop"):
        header = primary[0] if primary else ptr.get("condition_bb")
        if header is not None:
            loop_header = f"BB_{safe_int(header)}"

    break_target = None
    continue_target = None
    if typ == "BREAK" and edge and len(edge) >= 2:
        break_target = f"BB_{safe_int(edge[1])}"
    if typ == "CONTINUE" and edge and len(edge) >= 2:
        continue_target = f"BB_{safe_int(edge[1])}"

    return {
        "status": "partial_from_module_b_o0",
        "true_successor": true_succ,
        "false_successor": false_succ,
        "fallthrough_successor": fallthrough,
        "postdom_successor": None,
        "loop_header": loop_header,
        "loop_latch": None,
        "break_target": break_target,
        "continue_target": continue_target,
        "allowed_control_effects": allowed_control_effects(typ),
        "edge": list(edge) if isinstance(edge, list) else edge,
        "outgoing_edges": outgoing if isinstance(outgoing, list) else [],
        "notes": [
            "O0 records Module-B successor metadata when available; V1+ validator must confirm polarity and preservation."
        ],
    }


def data_contract(row: Dict[str, Any], slot: Dict[str, Any], ptr: Dict[str, Any]) -> Dict[str, Any]:
    primary = as_int_list(slot.get("primary_bbs"))
    context = as_int_list(slot.get("context_bbs"))
    flow = dataflow_summary_for_region(row, primary or context)
    availability = flow.get("availability") if isinstance(flow.get("availability"), dict) else {}
    status = str(flow.get("status") or "not_available_o0")
    condition_prov = condition_branch_provenance(row, {**ptr, "primary_bbs": primary}) if ptr.get("condition_ssa") else {}
    return_ssa = infer_return_ssa_for_slot(slot, flow)
    return_constant = infer_return_constant_from_def(row, return_ssa)
    return {
        "status": status,
        "live_in": list(flow.get("live_in") or []),
        "live_out": list(flow.get("live_out") or []),
        "defs": list(flow.get("defs") or []),
        "uses": list(flow.get("uses") or []),
        "return_ssa": return_ssa,
        "return_value_ssa": return_ssa,
        "return_constant": return_constant,
        "return_value_constant": return_constant,
        "fact_state": flow.get("fact_state"),
        "type_hints": {},
        "canonical_symbols": {},
        "ssa_grounding": {
            "condition_ssa": ptr.get("condition_ssa"),
            "return_ssa": return_ssa,
            "return_value_ssa": return_ssa,
            "return_constant": return_constant,
            "return_value_constant": return_constant,
            "real_ssa_available": ptr.get("real_ssa_available"),
            "branch_instruction": ptr.get("branch_instruction"),
            "branch_instruction_index": ptr.get("branch_instruction_index"),
            "condition_provenance": condition_prov,
            "condition_ssa_def": condition_prov.get("ssa_def") if isinstance(condition_prov, dict) else None,
            "condition_branch_instruction": condition_prov.get("branch_instruction_record") if isinstance(condition_prov, dict) else None,
            "function_ir_ssa_source": availability.get("ssa"),
            "function_ir_def_use_source": availability.get("def_use"),
            "function_ir_branch_metadata_available": availability.get("branch_metadata_available"),
            "instruction_ids": list(flow.get("instruction_ids") or []),
        },
        "bb_context_preview": basic_block_preview(row, primary or context),
        "notes": [
            "Dataflow facts are copied from existing FunctionIR/def-use evidence when present; proxy facts remain weak until V3."
        ],
    }


def effect_contract(row: Dict[str, Any], slot: Dict[str, Any]) -> Dict[str, Any]:
    primary = as_int_list(slot.get("primary_bbs"))
    context = as_int_list(slot.get("context_bbs"))
    effect = effect_summary_for_region(row, primary or context)
    return {
        "status": str(effect.get("status") or "not_available_o0"),
        "read_set": list(effect.get("read_set") or []),
        "write_set": list(effect.get("write_set") or []),
        "call_effects": list(effect.get("call_effects") or []),
        "volatile_ops": list(effect.get("volatile_ops") or []),
        "register_reads": list(effect.get("register_reads") or []),
        "register_writes": list(effect.get("register_writes") or []),
        "side_effect_order": list(effect.get("side_effect_order") or []),
        "may_throw_or_longjmp": bool(effect.get("may_throw_or_longjmp")),
        "memory_instruction_count": safe_int(effect.get("memory_instruction_count"), 0),
        "fact_state": effect.get("fact_state"),
        "availability": effect.get("availability") if isinstance(effect.get("availability"), dict) else {},
        "notes": [
            "Effect facts are copied from existing FunctionIR/call/memory summaries when present; missing memory SSA remains V2/V3 work."
        ],
    }


def observable_semantics_contract(slot_type: str, sem_type: str) -> Dict[str, Any]:
    typ = str(slot_type or "").upper()
    sem = str(sem_type or "").upper()
    observations: List[str] = []
    if typ == "COND":
        observations.extend(["branch_successor_decision"])
    if typ in {"BLOCK", "CASE", "UNSTRUCTURED_REGION"}:
        observations.extend(["live_out_variable_values", "memory_write_footprint", "externally_visible_call_effects"])
    if typ in {"BREAK", "CONTINUE", "GOTO", "EARLY_EXIT", "CLEANUP_EXIT"} or sem in {"EXIT", "LOOP_EXIT", "LOOP_CONTINUE"}:
        observations.extend(["required_control_transfer_target"])
    if typ in {"EARLY_EXIT", "CLEANUP_EXIT"}:
        observations.append("return_value_if_affected")
    return {
        "status": "declared_o0",
        "observable_items": observations,
        "do_not_compare": [
            "temporary_variable_names",
            "source_ast_node_kind",
            "non_live_out_intermediate_values",
            "statement_split_or_merge_shape",
            "surface_difference_between_early_return_and_if_else",
        ],
        "notes": [
            "Obs(slot) defines what later validators must compare; O0 records the contract but does not prove equivalence."
        ],
    }


def unified_ir_contract(
    *,
    slot: Dict[str, Any],
    sem_type: str,
    region: Dict[str, Any],
    data: Dict[str, Any],
    effect: Dict[str, Any],
) -> Dict[str, Any]:
    slot_id = str(slot.get("slot_id") or "")
    typ = str(slot.get("slot_type") or "").upper()
    kind = {
        "COND": "cond",
        "BLOCK": "stmt_region",
        "CASE": "stmt_region",
        "BREAK": "jump",
        "CONTINUE": "jump",
        "GOTO": "jump",
        "EARLY_EXIT": "return",
        "CLEANUP_EXIT": "jump",
        "UNSTRUCTURED_REGION": "opaque_region",
    }.get(typ, "unknown")
    node = {
        "id": f"ir_{slot_id}",
        "kind": kind,
        "bb_id": region.get("entry_bb"),
        "operands": [],
        "defs": list(data.get("defs") or []),
        "uses": list(data.get("uses") or []),
        "effects": list(effect.get("call_effects") or []),
        "type": None,
        "source_span": region.get("source_span"),
        "binary_span": region.get("binary_inst_span"),
        "semantic_slot_type": sem_type,
    }
    edges = []
    for raw in region.get("covered_edges") or []:
        if isinstance(raw, list) and len(raw) >= 2:
            edges.append(
                {
                    "src": f"BB_{safe_int(raw[0])}",
                    "dst": f"BB_{safe_int(raw[1])}",
                    "edge_type": raw[2] if len(raw) >= 3 else None,
                }
            )
    return {
        "status": "ir_lite_from_module_b_contract",
        "nodes": [node],
        "edges": edges,
        "canonical_symbol_policy": "arg_*, stack_slot_*, global_addr_*, ret_value, tmp_*, mem_region_*",
        "source_teacher_ir_status": "KEEP_EXTERNAL_source_lowering_not_connected",
        "binary_ir_status": "partial_from_function_ir_and_slot_grounding",
        "skeleton_ir_status": "available_from_skeleton_slot_lowerer",
    }


def attention_mask_contract(sem_type: str, region: Dict[str, Any], control: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    weights = ATTENTION_WEIGHTS.get(str(sem_type or "").upper(), ATTENTION_WEIGHTS.get("BLOCK"))
    boundary = set(region.get("exit_bbs") or [])
    for key in ("true_successor", "false_successor", "fallthrough_successor", "postdom_successor", "loop_header", "loop_latch", "break_target", "continue_target"):
        val = control.get(key)
        if val:
            boundary.add(str(val))
    if region.get("entry_bb"):
        boundary.add(str(region.get("entry_bb")))
    dataflow = set(data.get("live_in") or []) | set(data.get("live_out") or [])
    return {
        "status": "available_o0_bb_mask_without_dataflow",
        "weights": dict(weights or {}),
        "region_bb": list(region.get("region_bbs") or []),
        "boundary_bb": sorted(boundary),
        "dataflow_symbols": sorted(str(x) for x in dataflow),
        "dataflow_bb": [],
        "notes": [
            "This implements the ECSL three-ring mask schema; dataflow_bb remains empty until V2 dataflow extraction is connected."
        ],
    }


def grammar_contract(slot_type: str, sem_type: str, category: str) -> Dict[str, Any]:
    typ = str(slot_type or "").upper()
    return {
        "status": "available_o0",
        "category": category,
        "allowed_tokens": None,
        "forbidden_control_transfer": forbidden_control_transfer(typ),
        "scope_constraints": [
            "slot_fill_must_not_modify_skeleton_tokens",
            "slot_fill_must_parse_inside_placeholder_context",
        ],
        "render_policy": {
            "COND": "replace_with_c_condition_expression",
            "BLOCK": "replace_with_c_statement_list",
            "CASE": "replace_with_case_body_or_case_label_template",
            "GOTO": "emit_control_transfer_template_from_cfg_edge",
            "BREAK": "emit_break_template_from_cfg_edge",
            "CONTINUE": "emit_continue_template_from_cfg_edge",
            "EARLY_EXIT": "emit_early_exit_template_from_cfg_edge",
            "CLEANUP_EXIT": "emit_cleanup_exit_template_from_cfg_edges",
            "UNSTRUCTURED_REGION": "skip_or_emit_labeled_cfg_region",
        }.get(typ, "skip_unknown_slot_type"),
        "notes": [
            "O0 grammar is category-level; token-level grammar checker belongs to the next ECSL phase."
        ],
    }


def confidence_and_policy(
    *,
    b_gate_pass: bool,
    slot_type: str,
    is_escape: bool,
    is_repair: bool,
    has_anchor: bool,
    escape_count: int,
    strong_max_escape_count: int,
    weak_max_escape_count: int,
) -> Dict[str, Any]:
    typ = str(slot_type or "").upper()
    reasons: List[str] = []
    if not b_gate_pass:
        reasons.append("b_gate_failed")
    if not has_anchor:
        reasons.append("missing_skeleton_anchor")
    if typ in UNSUPPORTED_SLOT_TYPES:
        reasons.append("unsupported_unstructured_region")
    if typ not in STRUCTURAL_SLOT_TYPES and typ not in ESCAPE_SLOT_TYPES and typ not in UNSUPPORTED_SLOT_TYPES:
        reasons.append("unknown_slot_type")

    if reasons:
        return {
            "tier": "C0",
            "tier_name": "invalid_or_unmatched",
            "initial_alignment_score": 0.0,
            "validation_score": 0.0,
            "final_weight": 0.0,
            "label_policy": "skip",
            "hard_ce_allowed": False,
            "reasons": reasons,
        }

    if is_escape:
        return {
            "tier": "C1",
            "tier_name": "binary_only_or_template_prior",
            "initial_alignment_score": 0.25,
            "validation_score": 0.0,
            "final_weight": 0.0,
            "label_policy": "binary_template_prior_only",
            "hard_ce_allowed": False,
            "reasons": ["escape_or_control_transfer_slot"],
        }

    if typ in STRUCTURAL_SLOT_TYPES and is_repair:
        return {
            "tier": "C2",
            "tier_name": "cfg_liveout_compatible_weak",
            "initial_alignment_score": 0.45,
            "validation_score": 0.2,
            "final_weight": 0.2,
            "label_policy": "weak_supervision_after_candidate_validation",
            "hard_ce_allowed": False,
            "reasons": ["module_b_repair_slot_requires_extra_validation"],
        }

    if typ in STRUCTURAL_SLOT_TYPES and escape_count <= strong_max_escape_count:
        return {
            "tier": "C2",
            "tier_name": "contract_ready_needs_candidate_validator",
            "initial_alignment_score": 0.6,
            "validation_score": 0.2,
            "final_weight": 0.2,
            "label_policy": "eligible_for_scts_candidates_not_hard_label_yet",
            "hard_ce_allowed": False,
            "reasons": ["contract_ready_but_no_slot_label_candidate_validated"],
        }

    if typ in STRUCTURAL_SLOT_TYPES and escape_count <= weak_max_escape_count:
        return {
            "tier": "C1",
            "tier_name": "heuristic_prior_escape_moderate",
            "initial_alignment_score": 0.35,
            "validation_score": 0.0,
            "final_weight": 0.0,
            "label_policy": "candidate_prior_only_until_validator_passes",
            "hard_ce_allowed": False,
            "reasons": ["moderate_function_escape_count"],
        }

    return {
        "tier": "C0",
        "tier_name": "too_ambiguous_for_o0",
        "initial_alignment_score": 0.0,
        "validation_score": 0.0,
        "final_weight": 0.0,
        "label_policy": "skip",
        "hard_ce_allowed": False,
        "reasons": ["escape_heavy_function_requires_later_global_validation"],
    }


def make_semantic_slot(
    row: Dict[str, Any],
    slot: Dict[str, Any],
    *,
    function_key: str,
    function_slot_index: int,
    anchor: Optional[Dict[str, Any]],
    b_gate: Dict[str, Any],
    strong_max_escape_count: int,
    weak_max_escape_count: int,
) -> Dict[str, Any]:
    rep = selected_report(row)
    st = structuredness(row)
    sid = safe_int(slot.get("slot_id"), -1)
    slot_ref = f"slot_{sid:03d}" if sid >= 0 else str(slot.get("slot_id"))
    module_b_slot_kind = str(slot.get("slot_type") or "UNKNOWN").upper()
    role = str(slot.get("role") or "")
    primary = as_int_list(slot.get("primary_bbs"))
    context = as_int_list(slot.get("context_bbs"))
    edge = slot.get("edge") if isinstance(slot.get("edge"), list) else None
    region_bbs = sorted(set(primary or context))
    ptr = pointer_for_slot(row, slot)
    slot_type = doc_slot_type(module_b_slot_kind, role)
    sem_type = slot_type
    category = grammar_category(module_b_slot_kind, sem_type)
    is_escape = module_b_slot_kind in ESCAPE_SLOT_TYPES
    reason = str(slot.get("reason") or "")
    is_repair = "repair" in reason or "validator_repaired_missing_bb_coverage" in reason
    escape_count = safe_int(rep.get("escape_count"), safe_int(rep.get("goto_count")))
    has_anchor = bool(anchor and anchor.get("skeleton_ast_path"))

    confidence = confidence_and_policy(
        b_gate_pass=bool(b_gate.get("pass")),
        slot_type=module_b_slot_kind,
        is_escape=is_escape,
        is_repair=is_repair,
        has_anchor=has_anchor,
        escape_count=escape_count,
        strong_max_escape_count=strong_max_escape_count,
        weak_max_escape_count=weak_max_escape_count,
    )

    anchor_obj = {
        "skeleton_ast_path": (anchor or {}).get("skeleton_ast_path"),
        "dfs_order": (anchor or {}).get("dfs_order", function_slot_index),
        "parent_construct": (anchor or {}).get("parent_construct"),
        "placeholder_text": (anchor or {}).get("placeholder_text"),
        "skeleton_line": (anchor or {}).get("skeleton_line"),
        "indent": (anchor or {}).get("indent"),
        "line_text": (anchor or {}).get("line_text"),
        "anchor_status": "lowered_from_skeleton" if has_anchor else "missing_in_o0_lowerer",
    }

    region = {
        "entry_bb": infer_entry_bb(primary, context, edge),
        "exit_bbs": infer_exit_bbs(row, primary, context, edge),
        "region_bbs": [f"BB_{bb}" for bb in region_bbs],
        "primary_bbs": [f"BB_{bb}" for bb in primary],
        "context_bbs": [f"BB_{bb}" for bb in context],
        "covered_edges": covered_edges(row, region_bbs, edge),
        "binary_inst_span": ptr.get("instruction_ranges") or None,
        "source_span": None,
        "edge_context": edge_context(row, region_bbs),
        "region_status": "grounded_to_module_b_bbs" if (primary or context or edge) else "missing_region_bbs",
    }

    control = control_contract(row, slot, ptr)
    data = data_contract(row, slot, ptr)
    effect = effect_contract(row, slot)
    condition_prov = q(data, "ssa_grounding", "condition_provenance") or {}
    condition_def_status = q(condition_prov, "ssa_def", "status")
    branch_status = condition_prov.get("status") if isinstance(condition_prov, dict) else None

    return {
        "schema": SLOT_SCHEMA,
        "function_key": function_key,
        "function_slot_index": function_slot_index,
        "example": row.get("example"),
        "input_line": row.get("input_line"),
        "func_name": row.get("func_name") or function_ir(row).get("function_name"),
        "slot_id": slot_ref,
        "slot_id_raw": sid,
        "slot_type": slot_type,
        "module_b_slot_kind": module_b_slot_kind,
        "semantic_slot_type": sem_type,
        "role": role,
        "anchor": anchor_obj,
        "region": region,
        "control_contract": control,
        "data_contract": data,
        "effect_contract": effect,
        "observable_semantics": observable_semantics_contract(module_b_slot_kind, sem_type),
        "unified_ir": unified_ir_contract(slot={**slot, "slot_type": module_b_slot_kind}, sem_type=sem_type, region=region, data=data, effect=effect),
        "attention_mask": attention_mask_contract(sem_type, region, control, data),
        "grammar_contract": grammar_contract(module_b_slot_kind, sem_type, category),
        "polarity": {
            "status": "branch_metadata_available_needs_v3_confirmation"
            if module_b_slot_kind == "COND" and condition_def_status == "available" and branch_status == "available"
            else ("unknown_o0" if module_b_slot_kind == "COND" else "not_applicable"),
            "value": "unknown" if module_b_slot_kind == "COND" else "normal",
            "true_false_polarity_available": ptr.get("true_false_polarity_available"),
            "condition_def_status": condition_def_status,
            "branch_provenance_status": branch_status,
            "condition_def_text": q(condition_prov, "ssa_def", "def_text") if isinstance(condition_prov, dict) else None,
            "branch_text": condition_prov.get("branch_text") if isinstance(condition_prov, dict) else None,
        },
        "validation_rule": {
            "status": "stub_o0",
            "local_stub_signature": "fill placeholder in fixed Module-B skeleton without changing skeleton tokens",
            "compile_context": "requires later deterministic renderer and local C parser/type checker",
            "local_equivalence_observation": [
                "parser/type validation",
                "CFG successor preservation",
                "SSA live-out/effect validation",
                "optional symbolic or differential validation",
            ],
        },
        "confidence": confidence,
        "quality": {
            "b_gate_pass": bool(b_gate.get("pass")),
            "slot_locality_ready": bool(st.get("slot_locality_ready")),
            "slot_locality_score": st.get("slot_locality_score"),
            "is_escape_slot": is_escape,
            "is_repair_slot": is_repair,
            "function_escape_count": escape_count,
            "function_goto_count": safe_int(rep.get("goto_count")),
            "function_break_count": safe_int(rep.get("break_count")),
            "function_continue_count": safe_int(rep.get("continue_count")),
            "function_early_exit_count": safe_int(rep.get("early_exit_count")),
            "function_cleanup_exit_count": safe_int(rep.get("cleanup_exit_count")),
            "has_skeleton_anchor": has_anchor,
            "skip_reasons": list(confidence.get("reasons") or []),
        },
        "provenance": {
            "module": "module_b_v3",
            "candidate_id": verified_skeleton(row).get("candidate_id"),
            "candidate_name": verified_skeleton(row).get("candidate_name"),
            "selection_source": verified_skeleton(row).get("selection_source"),
            "slot_reason": reason,
            "slot_owner": slot.get("owner"),
            "module_b_slot_kind": module_b_slot_kind,
            "edge": edge,
            "source_projection_used": False,
            "note": "ECSL O0 contract only; no source AST label projection is used.",
        },
        "b_gate": b_gate,
    }


def build_record(
    line_no: int,
    row: Dict[str, Any],
    *,
    include_skeleton_snippets: bool,
    strong_max_escape_count: int,
    weak_max_escape_count: int,
) -> Dict[str, Any]:
    vs = verified_skeleton(row)
    slots = [slot for slot in (vs.get("slots") or []) if isinstance(slot, dict)]
    skeleton = str(vs.get("skeleton") or "")
    lowerer = lower_skeleton_slots(skeleton, slots)
    contract_conflicts = list(lowerer.get("issues") or [])
    b_gate = b_gate_report(row, bool(lowerer.get("success")), contract_conflicts)
    function_key = f"ex{row.get('example')}_line{row.get('input_line')}_{row.get('func_name') or function_ir(row).get('function_name') or line_no}"
    anchors: Dict[int, Dict[str, Any]] = lowerer.get("anchors") or {}
    semantic_slots = [
        make_semantic_slot(
            row,
            slot,
            function_key=function_key,
            function_slot_index=idx,
            anchor=anchors.get(safe_int(slot.get("slot_id"), -1)),
            b_gate=b_gate,
            strong_max_escape_count=strong_max_escape_count,
            weak_max_escape_count=weak_max_escape_count,
        )
        for idx, slot in enumerate(slots)
    ]
    record = {
        "schema": SCHEMA,
        "source_line_no": line_no,
        "function_key": function_key,
        "example": row.get("example"),
        "input_line": row.get("input_line"),
        "func_name": row.get("func_name") or function_ir(row).get("function_name"),
        "module_b": {
            "schema_version": row.get("schema_version"),
            "candidate_id": vs.get("candidate_id"),
            "candidate_name": vs.get("candidate_name"),
            "selection_source": vs.get("selection_source"),
            "outline": vs.get("outline") or [],
        },
        "skeleton_cfg": {
            "schema": "SkeletonCFG.o0_metadata_only",
            "lowerer": {
                "schema": lowerer.get("schema"),
                "success": lowerer.get("success"),
                "slot_count": lowerer.get("slot_count"),
                "lowered_slot_count": lowerer.get("lowered_slot_count"),
                "issues": lowerer.get("issues") or [],
            },
            "fixed_skeleton_sha_source": "module_b_verified_skeleton_text",
            "renderer_policy": "preserve_all_non_placeholder_skeleton_tokens",
        },
        "b_gate": b_gate,
        "semantic_slots": semantic_slots,
        "audit": {
            "slot_count": len(semantic_slots),
            "all_slots_lowered": bool(lowerer.get("success")) and len(semantic_slots) == safe_int(lowerer.get("lowered_slot_count")),
            "slot_type_dist": dict(Counter(slot.get("slot_type") for slot in semantic_slots)),
            "module_b_slot_kind_dist": dict(Counter(slot.get("module_b_slot_kind") for slot in semantic_slots)),
            "confidence_tier_dist": dict(Counter(q(slot, "confidence", "tier") for slot in semantic_slots)),
            "label_policy_dist": dict(Counter(q(slot, "confidence", "label_policy") for slot in semantic_slots)),
        },
    }
    if include_skeleton_snippets:
        record["verified_skeleton_text"] = skeleton
        record["verified_skeleton_preview"] = vs.get("skeleton_preview") or "\n".join(skeleton.splitlines()[:40])
    return record


def update_stats(stats: Dict[str, Any], record: Dict[str, Any]) -> None:
    stats["functions_total"] += 1
    slots = record.get("semantic_slots") or []
    b_gate = record.get("b_gate") or {}
    lowerer = q(record, "skeleton_cfg", "lowerer") or {}
    if b_gate.get("pass"):
        stats["functions_b_gate_pass"] += 1
    if lowerer.get("success"):
        stats["functions_lowerer_success"] += 1
    if q(record, "audit", "all_slots_lowered"):
        stats["functions_all_slots_lowered"] += 1
    if any(q(slot, "quality", "is_repair_slot") for slot in slots):
        stats["functions_with_repair"] += 1
    max_escape = max([safe_int(q(slot, "quality", "function_escape_count")) for slot in slots] or [0])
    if max_escape >= stats["escape_heavy_threshold"]:
        stats["functions_escape_heavy"] += 1
    stats["function_escape_count_dist"][max_escape] += 1
    stats["slots_total"] += len(slots)
    for slot in slots:
        slot_type = str(slot.get("slot_type") or "UNKNOWN")
        module_b_slot_kind = str(slot.get("module_b_slot_kind") or "UNKNOWN")
        sem_type = str(slot.get("semantic_slot_type") or "UNKNOWN")
        category = str(q(slot, "grammar_contract", "category") or "unknown")
        tier = str(q(slot, "confidence", "tier") or "UNKNOWN")
        policy = str(q(slot, "confidence", "label_policy") or "UNKNOWN")
        stats["slot_type_dist"][slot_type] += 1
        stats["module_b_slot_kind_dist"][module_b_slot_kind] += 1
        stats["semantic_slot_type_dist"][sem_type] += 1
        stats["grammar_category_dist"][category] += 1
        stats["confidence_tier_dist"][tier] += 1
        stats["label_policy_dist"][policy] += 1
        if q(slot, "quality", "has_skeleton_anchor"):
            stats["slots_with_anchor"] += 1
        else:
            stats["slots_missing_anchor"] += 1
        if q(slot, "quality", "is_escape_slot"):
            stats["escape_slot_count"] += 1
        if q(slot, "quality", "is_repair_slot"):
            stats["repair_slot_count"] += 1
        if tier == "C2":
            stats["weak_candidate_slots"] += 1
        if tier in {"C3", "C4", "C5"}:
            stats["strong_candidate_slots"] += 1
        if tier == "C1":
            stats["binary_only_slots"] += 1
        if tier == "C0":
            stats["skip_slots"] += 1

    if not b_gate.get("pass") and len(stats["b_gate_fail_examples"]) < 20:
        stats["b_gate_fail_examples"].append(
            {
                "example": record.get("example"),
                "input_line": record.get("input_line"),
                "func_name": record.get("func_name"),
                "failed": b_gate.get("failed_components") or [],
            }
        )
    if any(q(slot, "quality", "is_repair_slot") for slot in slots) and len(stats["repair_examples"]) < 20:
        stats["repair_examples"].append(
            {
                "example": record.get("example"),
                "input_line": record.get("input_line"),
                "func_name": record.get("func_name"),
                "repair_slots": [
                    slot.get("slot_id")
                    for slot in slots
                    if q(slot, "quality", "is_repair_slot")
                ],
            }
        )
    if max_escape >= stats["escape_heavy_threshold"] and len(stats["escape_heavy_examples"]) < 20:
        stats["escape_heavy_examples"].append(
            {
                "example": record.get("example"),
                "input_line": record.get("input_line"),
                "func_name": record.get("func_name"),
                "escape_count": max_escape,
            }
        )


def initial_stats(escape_heavy_threshold: int) -> Dict[str, Any]:
    return {
        "escape_heavy_threshold": escape_heavy_threshold,
        "functions_total": 0,
        "functions_b_gate_pass": 0,
        "functions_lowerer_success": 0,
        "functions_all_slots_lowered": 0,
        "functions_with_repair": 0,
        "functions_escape_heavy": 0,
        "slots_total": 0,
        "slots_with_anchor": 0,
        "slots_missing_anchor": 0,
        "repair_slot_count": 0,
        "escape_slot_count": 0,
        "strong_candidate_slots": 0,
        "weak_candidate_slots": 0,
        "binary_only_slots": 0,
        "skip_slots": 0,
        "slot_type_dist": Counter(),
        "module_b_slot_kind_dist": Counter(),
        "semantic_slot_type_dist": Counter(),
        "grammar_category_dist": Counter(),
        "confidence_tier_dist": Counter(),
        "label_policy_dist": Counter(),
        "function_escape_count_dist": Counter(),
        "b_gate_fail_examples": [],
        "repair_examples": [],
        "escape_heavy_examples": [],
    }


def write_report(path: Path, *, input_jsonl: Path, out_jsonl: Path, stats: Dict[str, Any]) -> None:
    n = stats["functions_total"]
    s = stats["slots_total"]
    lines: List[str] = []
    lines.append("=== Module C ECSL O0 SemanticSlot Contract Builder ===")
    lines.append(f"input_jsonl = {input_jsonl}")
    lines.append(f"out_jsonl = {out_jsonl}")
    lines.append(f"schema = {SCHEMA}")
    lines.append("")
    lines.append(f"functions_total = {n}")
    lines.append(f"functions_b_gate_pass = {pct(stats['functions_b_gate_pass'], n)}")
    lines.append(f"functions_lowerer_success = {pct(stats['functions_lowerer_success'], n)}")
    lines.append(f"functions_all_slots_lowered = {pct(stats['functions_all_slots_lowered'], n)}")
    lines.append(f"functions_with_repair = {stats['functions_with_repair']}")
    lines.append(f"functions_escape_heavy = {stats['functions_escape_heavy']}")
    lines.append(f"function_escape_count_dist = {compact_counter(stats['function_escape_count_dist'])}")
    lines.append("")
    lines.append(f"slots_total = {s}")
    lines.append(f"slots_with_anchor = {pct(stats['slots_with_anchor'], s)}")
    lines.append(f"slots_missing_anchor = {stats['slots_missing_anchor']}")
    lines.append(f"repair_slot_count = {stats['repair_slot_count']}")
    lines.append(f"escape_slot_count = {stats['escape_slot_count']}")
    lines.append(f"strong_candidate_slots = {stats['strong_candidate_slots']}")
    lines.append(f"weak_candidate_slots = {stats['weak_candidate_slots']}")
    lines.append(f"binary_only_slots = {stats['binary_only_slots']}")
    lines.append(f"skip_slots = {stats['skip_slots']}")
    lines.append("")
    lines.append(f"slot_type_dist = {compact_counter(stats['slot_type_dist'])}")
    lines.append(f"module_b_slot_kind_dist = {compact_counter(stats['module_b_slot_kind_dist'])}")
    lines.append(f"semantic_slot_type_dist = {compact_counter(stats['semantic_slot_type_dist'])}")
    lines.append(f"grammar_category_dist = {compact_counter(stats['grammar_category_dist'])}")
    lines.append(f"confidence_tier_dist = {compact_counter(stats['confidence_tier_dist'])}")
    lines.append(f"label_policy_dist = {compact_counter(stats['label_policy_dist'])}")
    lines.append("")
    lines.append("-- b_gate_fail_examples --")
    for item in stats["b_gate_fail_examples"]:
        lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("-- repair_examples --")
    for item in stats["repair_examples"]:
        lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("-- escape_heavy_examples --")
    for item in stats["escape_heavy_examples"]:
        lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    lines.append("")
    lines.append("Notes:")
    lines.append("- C2 here means contract-ready / weak by ECSL definition, not hard SFT label ready.")
    lines.append("- C1 escape/control-transfer slots are binary-template priors unless later validator upgrades them.")
    lines.append("- C0 slots/functions should not enter hard CE.")
    lines.append("- O0 does not project source AST labels and does not claim symbolic/effect equivalence.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contracts(args: argparse.Namespace) -> Dict[str, Any]:
    input_jsonl = Path(args.input_jsonl)
    out_jsonl = Path(args.out_jsonl)
    out_txt = Path(args.out_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    stats = initial_stats(escape_heavy_threshold=int(args.escape_heavy_threshold))
    with out_jsonl.open("w", encoding="utf-8") as wf:
        for line_no, row in iter_jsonl(input_jsonl, limit=int(args.limit)):
            record = build_record(
                line_no,
                row,
                include_skeleton_snippets=bool(args.include_skeleton_snippets),
                strong_max_escape_count=int(args.strong_max_escape_count),
                weak_max_escape_count=int(args.weak_max_escape_count),
            )
            wf.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            update_stats(stats, record)

    write_report(out_txt, input_jsonl=input_jsonl, out_jsonl=out_jsonl, stats=stats)
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build ECSL O0 SemanticSlot contracts from Module-B v3 full-pipeline JSONL."
    )
    ap.add_argument("--input-jsonl", required=True, help="Module-B v3 fullcandidates JSONL.")
    ap.add_argument("--out-jsonl", required=True, help="Output contract JSONL, one function per line.")
    ap.add_argument("--out-txt", required=True, help="Human-readable audit report.")
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    ap.add_argument(
        "--include-skeleton-snippets",
        action="store_true",
        help="Include full skeleton text and preview in each output record for manual audit.",
    )
    ap.add_argument(
        "--strong-max-escape-count",
        type=int,
        default=2,
        help="Max function escape count for structural slots to remain C2 contract-ready.",
    )
    ap.add_argument(
        "--weak-max-escape-count",
        type=int,
        default=7,
        help="Max function escape count for structural slots to remain C1 candidate-prior.",
    )
    ap.add_argument(
        "--escape-heavy-threshold",
        type=int,
        default=8,
        help="Report threshold for escape-heavy functions.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_contracts(args)
    print(f"functions_total = {stats['functions_total']}")
    print(f"functions_b_gate_pass = {pct(stats['functions_b_gate_pass'], stats['functions_total'])}")
    print(f"slots_total = {stats['slots_total']}")
    print(f"slots_with_anchor = {pct(stats['slots_with_anchor'], stats['slots_total'])}")
    print(f"confidence_tier_dist = {compact_counter(stats['confidence_tier_dist'])}")
    print(f"label_policy_dist = {compact_counter(stats['label_policy_dist'])}")
    print(f"wrote_jsonl = {args.out_jsonl}")
    print(f"wrote_txt = {args.out_txt}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrich existing CFG JSONL into a complete Module-B v3 FunctionIR schema.

The current StructLift JSONL already contains function-level CFG fields
(``bin_block_lengths``, ``bin_opcodes``, ``bin_edge_index``, ``bin_edge_type``).
It usually does not contain a real lifted SSA / def-use graph.  This script does
not pretend otherwise: by default it builds an auditable CFG-level proxy
FunctionIR with stable BB/instruction ids, edge provenance, branch-condition
proxy values, and explicit availability flags.

When future pipelines add real lifted IR/SSA fields, this script preserves them
and marks their source as ``real`` instead of ``cfg_proxy``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_int_list(x: Any) -> List[int]:
    if not isinstance(x, list):
        return []
    out: List[int] = []
    for v in x:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _has_any(row: Dict[str, Any], keys: Sequence[str]) -> bool:
    return any(k in row and row.get(k) not in (None, [], {}, "") for k in keys)


def _edge_kind(row: Dict[str, Any], idx: int) -> Optional[Any]:
    for key in ("cfg_edge_kinds", "edge_kinds", "edge_types", "bin_edge_type"):
        raw = row.get(key)
        if isinstance(raw, list) and idx < len(raw):
            return raw[idx]
    return None


def _instr_text(row: Dict[str, Any]) -> List[str]:
    for key in ("bin_instr_text", "bin_asm", "asm", "instructions_text", "disasm"):
        raw = row.get(key)
        if isinstance(raw, list):
            # Some backfills store per-BB instruction text.
            if raw and all(isinstance(x, list) for x in raw):
                flat: List[str] = []
                for block in raw:
                    flat.extend(str(x) for x in block)
                return flat
            return [str(x) for x in raw]
    return []


def _block_ranges(block_lengths: Sequence[int]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    cur = 0
    for length in block_lengths:
        start = cur
        cur += max(0, int(length))
        ranges.append((start, cur))
    return ranges


def _normalize_edges(row: Dict[str, Any], n_bb: int) -> List[Tuple[int, int, int]]:
    raw = row.get("bin_edge_index")
    if not (isinstance(raw, list) and len(raw) == 2):
        return []
    srcs = raw[0] if isinstance(raw[0], list) else []
    dsts = raw[1] if isinstance(raw[1], list) else []
    out: List[Tuple[int, int, int]] = []
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        s = _safe_int(src, -1)
        d = _safe_int(dst, -1)
        if 0 <= s < n_bb and 0 <= d < n_bb:
            out.append((s, d, idx))
    return out


def enrich_row(row: Dict[str, Any], *, input_line: int, add_proxy_ssa: bool = True) -> Dict[str, Any]:
    out = dict(row)
    block_lengths = _as_int_list(row.get("bin_block_lengths"))
    n_bb = len(block_lengths)
    ranges = _block_ranges(block_lengths)
    opcodes = _as_int_list(row.get("bin_opcodes"))
    operand_types = row.get("bin_operand_types") if isinstance(row.get("bin_operand_types"), list) else []
    operand_values = row.get("bin_operand_values") if isinstance(row.get("bin_operand_values"), list) else []
    text = _instr_text(row)
    edges = _normalize_edges(row, n_bb)
    succs: Dict[int, List[int]] = {i: [] for i in range(n_bb)}
    preds: Dict[int, List[int]] = {i: [] for i in range(n_bb)}
    for s, d, _idx in edges:
        succs[s].append(d)
        preds[d].append(s)
    for i in range(n_bb):
        succs[i] = sorted(set(succs[i]))
        preds[i] = sorted(set(preds[i]))

    func_name = row.get("func_name") or row.get("function") or f"line_{input_line}"
    function_id = str(row.get("function_id") or row.get("func_id") or f"line_{input_line}:{func_name}")
    basic_blocks: List[Dict[str, Any]] = []
    lifted: List[Dict[str, Any]] = []
    instruction_provenance: Dict[str, Dict[str, Any]] = {}
    for bb, (start, end) in enumerate(ranges):
        outgoing = succs.get(bb, [])
        terminator = "return" if not outgoing else ("cond_branch" if len(outgoing) >= 2 else "jump")
        instr_ids = [f"bb_{bb}:ins_{i}" for i in range(start, end)]
        for local_idx, global_idx in enumerate(range(start, end)):
            instr_id = f"bb_{bb}:ins_{global_idx}"
            lifted_id = f"lir_{global_idx}"
            opcode_id = opcodes[global_idx] if global_idx < len(opcodes) else None
            lifted.append(
                {
                    "lifted_id": lifted_id,
                    "instruction_id": instr_id,
                    "bb_id": bb,
                    "index": int(global_idx),
                    "local_index": int(local_idx),
                    "kind": "opcode_proxy",
                    "opcode_id": opcode_id,
                    "text": text[global_idx] if global_idx < len(text) else None,
                    "is_proxy": True,
                }
            )
            instruction_provenance[instr_id] = {
                "instruction_id": instr_id,
                "lifted_id": lifted_id,
                "bb_id": int(bb),
                "global_instruction_index": int(global_idx),
                "source": "cfg_opcode_proxy",
            }
        basic_blocks.append(
            {
                "bb_id": int(bb),
                "stable_id": f"bb_{bb}",
                "address_range": None,
                "instruction_ids": instr_ids,
                "instruction_index_range": [int(start), int(end)],
                "instruction_count": int(max(0, end - start)),
                "opcode_ids": opcodes[start:end] if opcodes else [],
                "operand_type_ids": operand_types[start:end] if operand_types else [],
                "operand_value_ids": operand_values[start:end] if operand_values else [],
                "terminator": terminator,
                "has_call": None,
                "has_return": bool(not outgoing),
                "side_effect_summary": [],
            }
        )

    ssa_values: List[Dict[str, Any]] = []
    def_use_edges: List[Dict[str, Any]] = []
    branch_metadata: List[Dict[str, Any]] = []
    if add_proxy_ssa:
        for bb in range(n_bb):
            outgoing = succs.get(bb, [])
            if len(outgoing) < 2:
                continue
            start, end = ranges[bb] if bb < len(ranges) else (0, 0)
            branch_instr = f"bb_{bb}:ins_{end - 1}" if end > start else None
            value_id = f"v_branch_bb_{bb}"
            ssa_values.append(
                {
                    "ssa_id": value_id,
                    "kind": "branch_condition_proxy",
                    "defined_by_instruction": branch_instr,
                    "bb_id": int(bb),
                    "uses": [f"edge_{idx}" for s, d, idx in edges if s == bb],
                    "is_proxy": True,
                }
            )
            if branch_instr is not None:
                def_use_edges.append(
                    {
                        "def": branch_instr,
                        "use": value_id,
                        "kind": "defines_proxy_condition",
                        "is_proxy": True,
                    }
                )
            branch_metadata.append(
                {
                    "bb_id": int(bb),
                    "condition_ssa": value_id,
                    "branch_instruction": branch_instr,
                    "outgoing_edges": [
                        {
                            "edge_id": int(idx),
                            "src": int(s),
                            "dst": int(d),
                            "kind": _edge_kind(row, idx),
                        }
                        for s, d, idx in edges
                        if s == bb
                    ],
                    "true_false_polarity_available": False,
                    "source": "cfg_proxy",
                }
            )

    cfg_edges: List[Dict[str, Any]] = []
    branch_by_bb = {int(b.get("bb_id")): b for b in branch_metadata}
    for s, d, idx in edges:
        branch = branch_by_bb.get(s)
        start, end = ranges[s] if s < len(ranges) else (0, 0)
        prov = (
            branch.get("branch_instruction")
            if branch
            else (f"bb_{s}:ins_{end - 1}" if end > start else None)
        )
        cfg_edges.append(
            {
                "edge_id": int(idx),
                "src": int(s),
                "dst": int(d),
                "src_id": f"bb_{s}",
                "dst_id": f"bb_{d}",
                "kind": _edge_kind(row, idx),
                "condition_ssa": branch.get("condition_ssa") if branch else None,
                "provenance_instruction": prov,
                "confidence": None,
            }
        )

    real_lifted = _has_any(row, ["lifted_instructions", "ir_instructions", "llvm_ir", "vex_ir", "pcode"])
    real_ssa = _has_any(row, ["ssa_values"]) and not all(
        isinstance(x, dict) and x.get("is_proxy") for x in (row.get("ssa_values") or [])
    )
    real_def_use = _has_any(row, ["def_use_graph", "ssa_def_use", "du_edges"])
    real_prov = _has_any(row, ["instruction_provenance", "source_instruction_map", "asm_instruction_ids"])

    function_ir = {
        "schema": "FunctionIR.v1",
        "function_id": function_id,
        "function_name": func_name,
        "source_path": row.get("source_path") or row.get("src_path") or row.get("path"),
        "binary_path": row.get("binary_path") or row.get("bin_path"),
        "basic_blocks": basic_blocks,
        "cfg_edges": cfg_edges,
        "lifted_instructions": row.get("lifted_instructions") or row.get("ir_instructions") or lifted,
        "ssa_values": row.get("ssa_values") or ssa_values,
        "def_use_graph": row.get("def_use_graph") or row.get("ssa_def_use") or {"edges": def_use_edges},
        "instruction_provenance": row.get("instruction_provenance") or instruction_provenance,
        "branch_metadata": row.get("branch_metadata") or branch_metadata,
        "call_metadata": row.get("call_metadata") or {},
        "memory_access_summary": row.get("memory_access_summary") or {},
        "lifted_instructions_available": bool(real_lifted or lifted),
        "ssa_available": bool(real_ssa or ssa_values),
        "def_use_available": bool(real_def_use or def_use_edges),
        "instruction_provenance_available": bool(real_prov or instruction_provenance),
        "branch_metadata_available": bool(branch_metadata or _has_any(row, ["branch_metadata", "bin_edge_type"])),
        "call_metadata_available": _has_any(row, ["call_metadata", "callsites", "call_edges"]),
        "memory_access_summary_available": _has_any(row, ["memory_access_summary", "mem_accesses", "memory_ops"]),
        "availability_source": {
            "lifted_instructions": "real" if real_lifted else "cfg_opcode_proxy",
            "ssa": "real" if real_ssa else ("cfg_branch_proxy" if ssa_values else "missing"),
            "def_use": "real" if real_def_use else ("cfg_branch_proxy" if def_use_edges else "missing"),
            "instruction_provenance": "real" if real_prov else "cfg_opcode_proxy",
        },
    }
    out["function_ir"] = function_ir
    out["lifted_instructions"] = function_ir["lifted_instructions"]
    out["ssa_values"] = function_ir["ssa_values"]
    out["def_use_graph"] = function_ir["def_use_graph"]
    out["instruction_provenance"] = function_ir["instruction_provenance"]
    out["branch_metadata"] = function_ir["branch_metadata"]
    out["function_ir_enrichment"] = {
        "schema": "ModuleBv3FunctionIREnrichment.v1",
        "mode": "preserve_real_fields_plus_cfg_proxy",
        "cfg_proxy_ssa_enabled": bool(add_proxy_ssa),
        "n_basic_blocks": int(n_bb),
        "n_cfg_edges": int(len(cfg_edges)),
        "n_lifted_instructions": int(len(function_ir["lifted_instructions"] or [])),
        "n_ssa_values": int(len(function_ir["ssa_values"] or [])),
        "n_def_use_edges": int(len((function_ir["def_use_graph"] or {}).get("edges") or []))
        if isinstance(function_ir.get("def_use_graph"), dict)
        else 0,
        "real_lifted_available": bool(real_lifted),
        "real_ssa_available": bool(real_ssa),
        "real_def_use_available": bool(real_def_use),
        "real_instruction_provenance_available": bool(real_prov),
        "note": (
            "CFG proxy fields satisfy FunctionIR schema and stable id requirements; "
            "they are not semantic SSA recovered by a lifter."
        ),
    }
    return out


def render_summary(rows: Sequence[Dict[str, Any]], *, source_jsonl: Path, out_jsonl: Path, elapsed: float) -> str:
    n = len(rows)
    counter = Counter()
    for r in rows:
        e = r.get("function_ir_enrichment") or {}
        if e.get("real_lifted_available"):
            counter["real_lifted_available"] += 1
        if e.get("real_ssa_available"):
            counter["real_ssa_available"] += 1
        if e.get("real_def_use_available"):
            counter["real_def_use_available"] += 1
        if e.get("real_instruction_provenance_available"):
            counter["real_instruction_provenance_available"] += 1
        if (r.get("function_ir") or {}).get("lifted_instructions_available"):
            counter["lifted_available"] += 1
        if (r.get("function_ir") or {}).get("ssa_available"):
            counter["ssa_available"] += 1
        if (r.get("function_ir") or {}).get("def_use_available"):
            counter["def_use_available"] += 1
        if (r.get("function_ir") or {}).get("instruction_provenance_available"):
            counter["provenance_available"] += 1
        if (r.get("function_ir") or {}).get("branch_metadata_available"):
            counter["branch_metadata_available"] += 1
    lines = [
        "===== MODULE B-v3 FUNCTIONIR ENRICHMENT =====",
        f"source_jsonl = {source_jsonl}",
        f"out_jsonl = {out_jsonl}",
        f"n = {n}",
        "",
        "-- schema availability after enrichment --",
        f"lifted_instructions_available = {counter['lifted_available']}/{n}",
        f"ssa_available = {counter['ssa_available']}/{n}",
        f"def_use_available = {counter['def_use_available']}/{n}",
        f"instruction_provenance_available = {counter['provenance_available']}/{n}",
        f"branch_metadata_available = {counter['branch_metadata_available']}/{n}",
        "",
        "-- real semantic availability, not proxy --",
        f"real_lifted_available = {counter['real_lifted_available']}/{n}",
        f"real_ssa_available = {counter['real_ssa_available']}/{n}",
        f"real_def_use_available = {counter['real_def_use_available']}/{n}",
        f"real_instruction_provenance_available = {counter['real_instruction_provenance_available']}/{n}",
        "",
        "note = proxy SSA/provenance makes FunctionIR schema complete for Module-B v3 audits; "
        "do not claim lifter-level SSA semantics unless real_* availability is nonzero.",
        f"elapsed_sec = {elapsed:.1f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich CFG JSONL with complete Module-B v3 FunctionIR proxy fields")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--no-proxy-ssa", action="store_true")
    args = ap.parse_args()

    source_jsonl = args.source_jsonl.expanduser().resolve()
    out_jsonl = args.out_jsonl.expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt = args.out_txt.expanduser().resolve() if args.out_txt else None
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for line_no, row in iter_jsonl(source_jsonl, int(args.max_examples)):
            enriched = enrich_row(row, input_line=line_no, add_proxy_ssa=not bool(args.no_proxy_ssa))
            rows.append(enriched)
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    summary = render_summary(rows, source_jsonl=source_jsonl, out_jsonl=out_jsonl, elapsed=time.time() - t0)
    print(summary, end="")
    if out_txt is not None:
        out_txt.write_text(summary, encoding="utf-8")
    print(f"saved_jsonl = {out_jsonl}")
    if out_txt is not None:
        print(f"saved_txt = {out_txt}")


if __name__ == "__main__":
    main()

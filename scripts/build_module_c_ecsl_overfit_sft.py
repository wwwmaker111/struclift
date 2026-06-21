#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Stage3-readable Module-C ECSL overfit SFT set.

This is deliberately an *overfit plumbing* builder, not the strict ECSL
teacher.  The strict candidate pipeline correctly keeps C1 priors out of hard
SFT labels while V2/V3 validators are unavailable.  For a 500-row sanity check,
however, we still need tokenized Stage3 examples that can be fed to the LLM.

The default ``hybrid`` label policy therefore:

1. Uses the fixed Module-B verified skeleton and ECSL slot contracts.
2. Projects weak source-line teacher fragments through ``augment_binskel_sft``.
3. Fills otherwise-unsupervised control-transfer slots from V0/V1-passing ECSL
   candidates such as ``break;`` / ``continue;`` / ``goto label_BB_*;``.
4. Marks every output row as an overfit pseudo-label row so it cannot be
   mistaken for a C3+ semantic teacher.

The resulting JSONL contains normal Stage3 fields:
``sft_input_ids``, ``sft_labels``, ``slot_types``, ``sft_fill_positions``, and
``sft_slot_bb_indices``.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FILL_RE = re.compile(r"<FILL_([A-Z_]+):([^>]+)>")
PLACEHOLDER_RE = re.compile(r"<(?P<tag>[A-Z_]+):slot_(?P<sid>\d+)>(?P<trailing>\s*;)?")
TIER_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
CONTROL_SLOT_TYPES = {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}
STRUCTURAL_SLOT_TYPES = {"COND", "BLOCK", "CASE"}


@dataclass
class SlotInfo:
    slot_id: str
    slot_type: str
    aligned_bb_indices: List[int]
    binary_context_text: str = ""


@dataclass
class Skeleton:
    skeleton_text: str
    slots: List[SlotInfo]
    reliability: Optional[float] = None


@dataclass
class CandidateChoice:
    slot_id: str
    slot_type: str
    text: str
    candidate_id: str
    confidence_level: str
    source: str
    reason: str


def load_augment_module() -> Any:
    path = ROOT / "scripts" / "augment_binskel_sft.py"
    spec = importlib.util.spec_from_file_location("_augment_binskel_sft_ecsl_overfit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if line:
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


def q(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def function_keys(row: Dict[str, Any], *, line_no: Optional[int] = None) -> List[Tuple[Any, Any, str]]:
    funcs = [
        row.get("func_name"),
        q(row, "function_ir", "function_name"),
        row.get("function_name"),
        row.get("name"),
    ]
    ex = row.get("example")
    inp = row.get("input_line")
    out: List[Tuple[Any, Any, str]] = []
    for func in funcs:
        if not func:
            continue
        fs = str(func)
        out.append((ex, inp, fs))
        out.append((None, inp, fs))
        out.append((ex, None, fs))
        out.append((None, None, fs))
    if line_no is not None:
        for func in funcs:
            if func:
                out.append((None, line_no, str(func)))
    return out


def contract_function_key(row: Dict[str, Any]) -> Tuple[Any, Any, str]:
    return (row.get("example"), row.get("input_line"), str(row.get("func_name") or ""))


def load_module_b_index(paths: Sequence[Path], *, limit: int = 0) -> Dict[Tuple[Any, Any, str], Dict[str, Any]]:
    index: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    for path in paths:
        for line_no, row in iter_jsonl(path, limit=limit):
            for key in function_keys(row, line_no=line_no):
                index.setdefault(key, row)
    return index


def merge_source_and_module_b(source_row: Optional[Dict[str, Any]], module_b_row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep heavy graph/source tensors from the original row and B metadata from Module-B output."""
    out = dict(source_row or {})
    always_overlay = {
        "example",
        "input_line",
        "func_name",
        "schema_version",
        "verified_skeleton",
        "validator",
        "document_metric_contract",
        "teacher_diagnostics",
        "function_ir",
    }
    for key, value in module_b_row.items():
        if value is None:
            continue
        if key in always_overlay or key not in out or out.get(key) is None:
            out[key] = value
    return out


def find_module_b_row(index: Dict[Tuple[Any, Any, str], Dict[str, Any]], cand_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    keys = [
        contract_function_key(cand_record),
        (None, cand_record.get("input_line"), str(cand_record.get("func_name") or "")),
        (cand_record.get("example"), None, str(cand_record.get("func_name") or "")),
        (None, None, str(cand_record.get("func_name") or "")),
    ]
    for key in keys:
        if key in index:
            return index[key]
    return None


def canonical_slot_id(value: Any) -> str:
    s = str(value or "").strip()
    m = re.search(r"slot_(\d+)", s)
    if m:
        return f"slot_{int(m.group(1)):03d}"
    try:
        return f"slot_{int(s):03d}"
    except Exception:
        return s


def normalize_skeleton_placeholders(text: str) -> str:
    """Convert Module-B placeholder names to Stage3 ``FILL_*`` placeholders."""
    s = str(text or "")

    def repl(match: re.Match[str]) -> str:
        tag = match.group("tag")
        sid = int(match.group("sid"))
        trailing = match.group("trailing") or ""
        slot = f"slot_{sid:03d}"
        if tag == "FILL_COND":
            return f"<FILL_COND:{slot}>" + trailing
        if tag in {"FILL_BLOCK", "FILL_STMT", "CASE_SLOT"}:
            return f"<FILL_STMT:{slot}>"
        if tag in {"GOTO_SLOT", "BREAK_SLOT", "CONTINUE_SLOT", "EARLY_EXIT_SLOT", "CLEANUP_EXIT_SLOT"}:
            return f"<FILL_STMT:{slot}>"
        if tag.startswith("FILL_"):
            return f"<{tag}:{slot}>" + trailing
        return f"<FILL_STMT:{slot}>"

    s = PLACEHOLDER_RE.sub(repl, s)
    s = re.sub(r"<FILL_COND:slot_(\d+)>", lambda m: f"<FILL_COND:slot_{int(m.group(1)):03d}>", s)
    s = re.sub(r"<FILL_(?:BLOCK|STMT):slot_(\d+)>\s*;?", lambda m: f"<FILL_STMT:slot_{int(m.group(1)):03d}>", s)
    return s


def parse_bb(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    s = str(value or "")
    m = re.search(r"BB_(\d+)", s)
    if m:
        return int(m.group(1))
    try:
        return int(s)
    except Exception:
        return None


def slot_fill_type(slot_type: str) -> str:
    typ = str(slot_type or "").upper()
    if typ == "COND":
        return "FILL_COND"
    return "FILL_STMT"


def slot_bbs(slot: Dict[str, Any]) -> List[int]:
    raw: List[Any] = []
    for path in (
        ("region", "primary_bbs"),
        ("region", "region_bbs"),
        ("region", "context_bbs"),
        ("attention_mask", "region_bb"),
    ):
        val = q(slot, *path)
        if isinstance(val, list) and val:
            raw = list(val)
            break
    if not raw:
        edge = q(slot, "control_contract", "edge")
        if isinstance(edge, list):
            raw = list(edge)
    out: List[int] = []
    for item in raw:
        bb = parse_bb(item)
        if bb is not None and bb >= 0:
            out.append(bb)
    return sorted(set(out))


def slot_context_text(slot: Dict[str, Any]) -> str:
    slim = {
        "slot_type": slot.get("slot_type"),
        "semantic_slot_type": slot.get("semantic_slot_type"),
        "role": slot.get("role"),
        "anchor": q(slot, "anchor", "skeleton_ast_path"),
        "region": slot.get("region"),
        "control": slot.get("control_contract"),
        "data": q(slot, "data_contract", "ssa_grounding"),
        "confidence": slot.get("confidence"),
    }
    try:
        return json.dumps(slim, ensure_ascii=False, sort_keys=True)[:4096]
    except Exception:
        return ""


def external_skeleton_from_candidate_record(cand_record: Dict[str, Any], module_b_row: Dict[str, Any]) -> Skeleton:
    skeleton = (
        q(module_b_row, "verified_skeleton", "skeleton")
        or cand_record.get("verified_skeleton_text")
        or cand_record.get("verified_skeleton_preview")
        or ""
    )
    skeleton = normalize_skeleton_placeholders(str(skeleton))
    slots: List[SlotInfo] = []
    for slot_set in cand_record.get("slot_candidate_sets") or []:
        slot = slot_set.get("slot") if isinstance(slot_set, dict) else None
        if not isinstance(slot, dict):
            continue
        sid = canonical_slot_id(slot.get("slot_id"))
        typ = slot_fill_type(str(slot.get("slot_type") or ""))
        slots.append(
            SlotInfo(
                slot_id=sid,
                slot_type=typ,
                aligned_bb_indices=slot_bbs(slot),
                binary_context_text=slot_context_text(slot),
            )
        )
    if not skeleton:
        raise ValueError("missing verified skeleton text")
    if not slots:
        raise ValueError("missing ECSL slots")
    skeleton_slot_ids = [canonical_slot_id(m.group(2)) for m in FILL_RE.finditer(skeleton)]
    slot_by_id = {s.slot_id: s for s in slots}
    ordered = [slot_by_id[sid] for sid in skeleton_slot_ids if sid in slot_by_id]
    if len(ordered) != len(slots):
        missing = sorted(set(slot_by_id) - set(skeleton_slot_ids))
        if missing:
            raise ValueError(f"{len(missing)} ECSL slots missing from normalized skeleton, e.g. {missing[:5]}")
    return Skeleton(skeleton_text=skeleton, slots=ordered, reliability=1.0)


def candidate_text(cand: Dict[str, Any]) -> str:
    rows = cand.get("c_renderings")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("text") or "").strip():
                return str(row.get("text") or "").strip()
    return ""


def candidate_is_generated(cand: Dict[str, Any]) -> bool:
    if q(cand, "keep", "status") == "KEEP_EXTERNAL":
        return False
    text = candidate_text(cand)
    if not text:
        return False
    if "ECSL_BINARY_BLOCK" in text:
        return False
    if "ECSL_UNKNOWN_COND" in text:
        return False
    return True


def candidate_ok(cand: Dict[str, Any], *, min_tier: str, require_v0_v1: bool) -> bool:
    if not candidate_is_generated(cand):
        return False
    level = str(q(cand, "confidence", "confidence_level") or "C0")
    if TIER_ORDER.get(level, 0) < TIER_ORDER.get(min_tier, 1):
        return False
    if require_v0_v1 and (not q(cand, "validation", "V0", "passed") or not q(cand, "validation", "V1", "passed")):
        return False
    if TIER_ORDER.get(level, 0) >= TIER_ORDER["C2"] and str(q(cand, "validation", "V2", "status") or "") not in {"pass", ""}:
        return False
    return True


def candidate_sort_key(cand: Dict[str, Any]) -> Tuple[int, float, float, str]:
    level = str(q(cand, "confidence", "confidence_level") or "C0")
    return (
        TIER_ORDER.get(level, 0),
        safe_float(q(cand, "confidence", "confidence_weight")),
        safe_float(q(cand, "matching", "base_alignment_score")),
        str(cand.get("candidate_id") or ""),
    )


def clean_candidate_for_slot(aug: Any, slot_type: str, text: str) -> Tuple[str, bool]:
    fill_type = slot_fill_type(slot_type)
    cleaned, usable = aug._clean_teacher_target_for_slot(fill_type, text)
    if not usable:
        return "", False
    reason = aug._strict_reject_teacher_target(fill_type, cleaned, text)
    if reason:
        # break/continue/goto templates are intentionally simple statement slots.
        if not (fill_type == "FILL_STMT" and cleaned.startswith(("break", "continue", "goto "))):
            return "", False
    return cleaned, True


def select_candidates_for_overfit(
    cand_record: Dict[str, Any],
    aug: Any,
    *,
    min_tier: str,
    require_v0_v1: bool,
) -> Dict[str, CandidateChoice]:
    out: Dict[str, CandidateChoice] = {}
    for slot_set in cand_record.get("slot_candidate_sets") or []:
        if not isinstance(slot_set, dict):
            continue
        slot = slot_set.get("slot") if isinstance(slot_set.get("slot"), dict) else {}
        sid = canonical_slot_id(slot.get("slot_id"))
        slot_type = str(slot.get("slot_type") or "")
        rows = [c for c in slot_set.get("candidates") or [] if isinstance(c, dict)]
        rows = [c for c in rows if candidate_ok(c, min_tier=min_tier, require_v0_v1=require_v0_v1)]
        rows.sort(key=candidate_sort_key, reverse=True)
        for cand in rows:
            raw = candidate_text(cand)
            cleaned, usable = clean_candidate_for_slot(aug, slot_type, raw)
            if not usable:
                continue
            src = ",".join(str(x) for x in (q(cand, "provenance", "source") or []))
            out[sid] = CandidateChoice(
                slot_id=sid,
                slot_type=slot_type,
                text=cleaned,
                candidate_id=str(cand.get("candidate_id") or ""),
                confidence_level=str(q(cand, "confidence", "confidence_level") or "C0"),
                source=src,
                reason="ecsl_v0_v1_candidate_fallback",
            )
            break
    return out


def ensure_minimal_stage_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    n_bb = len(out.get("bin_block_lengths") or [])
    if n_bb <= 0:
        fi = out.get("function_ir") if isinstance(out.get("function_ir"), dict) else {}
        bbs = fi.get("basic_blocks") if isinstance(fi.get("basic_blocks"), list) else []
        if bbs:
            out["bin_block_lengths"] = [max(1, safe_int(b.get("instruction_count"), 1)) for b in bbs if isinstance(b, dict)]
            n_bb = len(out["bin_block_lengths"])
    if n_bb > 0:
        out.setdefault("pattern_labels", [0] * n_bb)
        out.setdefault("bin_struct_features", [[0.0] * 12 for _ in range(n_bb)])
        if "bin_edge_index" not in out:
            fi = out.get("function_ir") if isinstance(out.get("function_ir"), dict) else {}
            pairs = []
            for edge in fi.get("cfg_edges") or []:
                if not isinstance(edge, dict):
                    continue
                s = safe_int(edge.get("src"), -1)
                d = safe_int(edge.get("dst"), -1)
                if 0 <= s < n_bb and 0 <= d < n_bb:
                    pairs.append((s, d))
            out["bin_edge_index"] = [[s for s, _ in pairs], [d for _, d in pairs]]
            out["bin_edge_type"] = [0] * len(pairs)
    ni = sum(safe_int(x) for x in out.get("bin_block_lengths") or [])
    out.setdefault("bin_opcodes", [0] * ni)
    out.setdefault("bin_operand_types", [0] * (ni * 3))
    out.setdefault("bin_operand_values", [0] * (ni * 3))
    return out


def count_supervised_tokens(record: Dict[str, Any]) -> int:
    return sum(1 for x in record.get("sft_labels") or [] if safe_int(x, -100) != -100)


def prompt_order_flags(record: Dict[str, Any]) -> List[bool]:
    flags = list(
        record.get("sft_teacher_slot_supervise_prompt_order")
        or record.get("sft_teacher_slot_supervise")
        or []
    )
    slot_ids = record.get("sft_prompt_slot_ids") if isinstance(record.get("sft_prompt_slot_ids"), list) else []
    while len(flags) < len(slot_ids):
        flags.append(False)
    return [bool(x) for x in flags[: len(slot_ids)]] if slot_ids else [bool(x) for x in flags]


def count_supervised_slots(record: Dict[str, Any]) -> int:
    return sum(1 for x in prompt_order_flags(record) if bool(x))


def rebuild_sft_with_prompt_targets(
    record: Dict[str, Any],
    aug: Any,
    tokenizer: Any,
    fill_ids: List[int],
    prompt_targets: List[str],
    prompt_flags: List[bool],
) -> None:
    skeleton = str(record.get("sft_skeleton_text_pretty") or "")
    slot_types = [str(x) for x in (record.get("slot_types") or [])]
    slot_ids = [str(x) for x in (record.get("sft_prompt_slot_ids") or [])]
    bb_indices = record.get("sft_slot_bb_indices") if isinstance(record.get("sft_slot_bb_indices"), list) else []
    if not skeleton or not slot_types or not slot_ids:
        raise ValueError("cannot rebuild SFT without prompt skeleton/slots")
    slots = [
        SlotInfo(
            slot_id=slot_ids[i],
            slot_type=slot_types[i],
            aligned_bb_indices=[
                safe_int(x, -1)
                for x in (bb_indices[i] if i < len(bb_indices) and isinstance(bb_indices[i], list) else [])
                if safe_int(x, -1) >= 0
            ],
        )
        for i in range(len(slot_types))
    ]
    body_ids, body_sup, fill_pos_body = aug._interleave_skeleton_token_ids(
        skeleton,
        slots,
        prompt_targets,
        fill_ids,
        tokenizer,
        slot_supervise=prompt_flags,
    )
    old_ids = list(record.get("sft_input_ids") or [])
    old_labels = list(record.get("sft_labels") or [])
    old_fill_positions = [int(x) for x in (record.get("sft_fill_positions") or [])]
    if old_fill_positions and fill_pos_body:
        head_len = max(0, old_fill_positions[0] - fill_pos_body[0])
    else:
        head_len = 0
    tail_start_old = 0
    if old_fill_positions:
        tail_start_old = old_fill_positions[-1] + 1
        while tail_start_old < len(old_labels) and safe_int(old_labels[tail_start_old], -100) != -100:
            tail_start_old += 1
    old_head = old_ids[:head_len]
    old_tail = old_ids[tail_start_old:] if tail_start_old < len(old_ids) else []

    input_ids = old_head + body_ids + old_tail
    labels = list(input_ids)
    supervised_mask = [False] * len(old_head) + body_sup + [False] * len(old_tail)
    for i, keep in enumerate(supervised_mask):
        if not keep:
            labels[i] = -100

    if record.get("sft_masked_supervised_inputs"):
        mask_id = record.get("sft_supervised_input_mask_id")
        if mask_id is None:
            mask_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        mask_id = int(mask_id or 0)
        for i, keep in enumerate(supervised_mask):
            if keep:
                input_ids[i] = mask_id

    record["sft_input_ids"] = input_ids
    record["sft_labels"] = labels
    record["sft_fill_positions"] = [len(old_head) + int(p) for p in fill_pos_body]
    record["sft_teacher_targets"] = list(prompt_targets)
    record["sft_teacher_targets_prompt_order"] = list(prompt_targets)
    record["sft_teacher_slot_supervise"] = list(prompt_flags)
    record["sft_teacher_slot_supervise_prompt_order"] = list(prompt_flags)
    record["sft_output_text_pretty"] = aug._render_skeleton_with_targets(skeleton, prompt_targets, slots)


def apply_candidate_fallbacks(
    record: Dict[str, Any],
    candidate_choices: Dict[str, CandidateChoice],
    *,
    prefer_candidates: bool,
) -> Dict[str, Any]:
    slot_ids = [str(x) for x in (record.get("sft_prompt_slot_ids") or [])]
    targets = list(record.get("sft_teacher_targets_prompt_order") or record.get("sft_teacher_targets") or [])
    flags = list(
        record.get("sft_teacher_slot_supervise_prompt_order")
        or record.get("sft_teacher_slot_supervise")
        or [bool(str(x or "").strip()) for x in targets]
    )
    while len(targets) < len(slot_ids):
        targets.append("")
    while len(flags) < len(slot_ids):
        flags.append(False)

    applied: List[Dict[str, Any]] = []
    source_projected = 0
    for i, sid in enumerate(slot_ids):
        has_source = bool(flags[i] and str(targets[i] or "").strip())
        if has_source:
            source_projected += 1
        choice = candidate_choices.get(canonical_slot_id(sid))
        if choice is None:
            continue
        force_candidate = str(choice.slot_type or "").upper() in CONTROL_SLOT_TYPES
        if has_source and not prefer_candidates and not force_candidate:
            continue
        old = targets[i]
        targets[i] = choice.text
        flags[i] = True
        applied.append(
            {
                "slot_index": i,
                "slot_id": sid,
                "old_target": old,
                "new_target": choice.text,
                "candidate_id": choice.candidate_id,
                "slot_type": choice.slot_type,
                "confidence_level": choice.confidence_level,
                "source": choice.source,
                "reason": "control_transfer_candidate_preferred" if force_candidate else choice.reason,
            }
        )
    return {
        "targets": targets[: len(slot_ids)],
        "flags": [bool(x) for x in flags[: len(slot_ids)]],
        "candidate_fallbacks_applied": applied,
        "source_projected_slot_count_before_fallback": source_projected,
    }


def build_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    aug = load_augment_module()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"需要 transformers: {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(int(args.max_fill_tokens))]
    tokenizer.add_tokens(fill_tokens)
    fill_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in fill_tokens]
    if args.save_tokenizer:
        save_tok = Path(args.save_tokenizer).expanduser().resolve()
        save_tok.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_tok))

    module_b_paths = [Path(p).expanduser().resolve() for p in args.module_b_jsonl]
    module_b_index = load_module_b_index(module_b_paths, limit=int(args.max_module_b_rows))
    source_index: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    if args.source_jsonl:
        source_paths = [Path(p).expanduser().resolve() for p in args.source_jsonl]
        source_index = load_module_b_index(source_paths, limit=int(args.max_source_rows))
    candidate_path = Path(args.candidates_jsonl).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    prefix_rewrites = [(str(a), str(b)) for a, b in (args.rewrite_source_prefix or [])]

    out_records: List[Dict[str, Any]] = []
    skips: List[str] = []
    stats: Dict[str, Any] = {
        "rows_seen": 0,
        "records_ok": 0,
        "slot_type_dist": collections.Counter(),
        "label_source_dist": collections.Counter(),
        "skip_reason_dist": collections.Counter(),
        "fallback_count_dist": collections.Counter(),
    }

    for _line_no, cand_record in iter_jsonl(candidate_path, limit=int(args.limit)):
        stats["rows_seen"] += 1
        key = contract_function_key(cand_record)
        mb = find_module_b_row(module_b_index, cand_record)
        if mb is None:
            reason = f"{key}: module_b_row_not_found"
            skips.append(reason)
            stats["skip_reason_dist"]["module_b_row_not_found"] += 1
            continue
        src = find_module_b_row(source_index, cand_record) if source_index else None
        if source_index and src is None:
            src = find_module_b_row(source_index, mb)
        try:
            skel = external_skeleton_from_candidate_record(cand_record, mb)
        except Exception as e:
            reason = f"{key}: skeleton_error: {e}"
            skips.append(reason)
            stats["skip_reason_dist"]["skeleton_error"] += 1
            continue

        candidate_choices = select_candidates_for_overfit(
            cand_record,
            aug,
            min_tier=str(args.min_candidate_tier),
            require_v0_v1=not bool(args.allow_unvalidated_candidates),
        )
        work = ensure_minimal_stage_fields(merge_source_and_module_b(src, mb))
        try:
            rec, err = aug.process_record(
                work,
                tokenizer,
                fill_ids,
                source_root,
                emit_rl_fields=True,
                num_patterns=10,
                mask_prompt_labels=True,
                mask_non_fill_labels=True,
                mask_supervised_inputs=bool(args.mask_supervised_inputs),
                supervised_input_mask_prob=float(args.supervised_input_mask_prob),
                max_seq_len=int(args.max_seq_len),
                prefix_rewrites=prefix_rewrites,
                external_skeleton=skel,
                skeleton_source="module_c_ecsl_overfit_verified_skeleton",
                strict_teacher=bool(args.strict_source_teacher),
                strict_teacher_drop_record=bool(args.strict_source_teacher_drop_record),
                min_supervised_semantic_slots=int(args.min_source_supervised_semantic_slots),
                asm_max_total_lines=int(args.asm_max_total_lines),
                asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                prompt_compact_mode=str(args.prompt_compact_mode),
                max_slot_context_chars=int(args.max_slot_context_chars),
            )
        except Exception as e:
            rec, err = None, f"process_record_exception: {e}"
        if rec is None:
            reason = f"{key}: {err}"
            skips.append(reason)
            stats["skip_reason_dist"][str(err).split(":")[0][:80]] += 1
            continue

        applied = apply_candidate_fallbacks(
            rec,
            candidate_choices,
            prefer_candidates=bool(args.prefer_candidate_labels),
        )
        try:
            rebuild_sft_with_prompt_targets(
                rec,
                aug,
                tokenizer,
                fill_ids,
                applied["targets"],
                applied["flags"],
            )
        except Exception as e:
            reason = f"{key}: rebuild_error: {e}"
            skips.append(reason)
            stats["skip_reason_dist"]["rebuild_error"] += 1
            continue

        supervised_slots = count_supervised_slots(rec)
        if supervised_slots < int(args.min_supervised_slots_after_fallback):
            reason = f"{key}: too_few_supervised_slots_after_fallback={supervised_slots}"
            skips.append(reason)
            stats["skip_reason_dist"]["too_few_supervised_slots_after_fallback"] += 1
            continue
        if count_supervised_tokens(rec) <= 0:
            reason = f"{key}: no_supervised_tokens"
            skips.append(reason)
            stats["skip_reason_dist"]["no_supervised_tokens"] += 1
            continue

        fallback_rows = applied["candidate_fallbacks_applied"]
        rec["module_c_ecsl_overfit"] = {
            "schema": "ModuleC.ECSL.OverfitSFT.v0",
            "pseudo_label": True,
            "not_strict_semantic_teacher": True,
            "label_policy": "hybrid_source_projection_plus_ecsl_v0_v1_candidate_fallback",
            "candidate_fallbacks_applied": fallback_rows,
            "source_projected_slot_count_before_fallback": applied["source_projected_slot_count_before_fallback"],
            "supervised_slot_count": supervised_slots,
            "supervised_token_count": count_supervised_tokens(rec),
            "candidate_record_function_key": cand_record.get("function_key"),
            "module_b_candidate_name": q(mb, "verified_skeleton", "candidate_name"),
            "module_b_selection_source": q(mb, "verified_skeleton", "selection_source"),
            "warning": "For 500-row overfit plumbing only; do not scale as C3+ hard labels without V2/V3 validation.",
        }
        rec["module_c_ecsl_candidate_record_summary"] = {
            "function_key": cand_record.get("function_key"),
            "candidate_count": q(cand_record, "summary", "candidate_count"),
            "slot_count": q(cand_record, "summary", "slot_count"),
            "sft_eligible_slot_count_strict": q(cand_record, "summary", "sft_eligible_slot_count"),
        }
        out_records.append(rec)
        stats["records_ok"] += 1
        stats["slot_type_dist"].update(rec.get("slot_types") or [])
        stats["fallback_count_dist"][len(fallback_rows)] += 1
        fallback_indices = {int(x.get("slot_index", -1)) for x in fallback_rows}
        for i, keep in enumerate(prompt_order_flags(rec)):
            if not keep:
                continue
            stats["label_source_dist"]["ecsl_candidate_fallback" if i in fallback_indices else "source_projection"] += 1

    return out_records, skips, stats


def summarize(records: Sequence[Dict[str, Any]], skips: Sequence[str], stats: Dict[str, Any], args: argparse.Namespace) -> str:
    token_counts = [count_supervised_tokens(r) for r in records]
    slot_counts = [count_supervised_slots(r) for r in records]
    lines: List[str] = []
    lines.append("=== Module C ECSL Overfit Stage3 SFT Build ===")
    lines.append(f"candidates_jsonl = {args.candidates_jsonl}")
    lines.append(f"module_b_jsonl = {list(args.module_b_jsonl)}")
    lines.append(f"records_ok = {len(records)}")
    lines.append(f"records_skipped = {len(skips)}")
    lines.append(f"rows_seen = {stats.get('rows_seen', 0)}")
    lines.append("")
    lines.append(f"slot_type_dist = {stats.get('slot_type_dist', collections.Counter()).most_common()}")
    lines.append(f"label_source_dist = {stats.get('label_source_dist', collections.Counter()).most_common()}")
    lines.append(f"fallback_count_dist = {stats.get('fallback_count_dist', collections.Counter()).most_common()}")
    if slot_counts:
        lines.append(f"supervised_slots_total = {sum(slot_counts)}")
        lines.append(f"supervised_slots_min = {min(slot_counts)}")
        lines.append(f"supervised_slots_max = {max(slot_counts)}")
    if token_counts:
        lines.append(f"supervised_tokens_total = {sum(token_counts)}")
        lines.append(f"supervised_tokens_min = {min(token_counts)}")
        lines.append(f"supervised_tokens_max = {max(token_counts)}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- This file is trainable by Stage3, unlike strict ECSL train records with sft_eligible=0.")
    lines.append("- Labels are overfit pseudo-labels: source-line projection plus selected V0/V1 ECSL control-transfer fallbacks.")
    lines.append("- Passing overfit only proves the Module-C data/training plumbing can learn these 500 rows.")
    lines.append("- Scaling to 19w still requires a real label policy: V2/V3 validators, source CFG lowering, or explicitly accepted weak supervision.")
    lines.append("")
    lines.append("-- first target previews --")
    shown = 0
    for rec in records[:20]:
        slot_ids = rec.get("sft_prompt_slot_ids") or []
        slot_types = rec.get("slot_types") or []
        targets = rec.get("sft_teacher_targets_prompt_order") or rec.get("sft_teacher_targets") or []
        flags = prompt_order_flags(rec)
        for sid, typ, tgt, keep in zip(slot_ids, slot_types, targets, flags):
            if keep and str(tgt or "").strip():
                lines.append(f"{rec.get('func_name')} {sid} {typ}: {str(tgt)[:160]}")
                shown += 1
                if shown >= 20:
                    break
        if shown >= 20:
            break
    if not shown:
        lines.append("<none>")
    if skips:
        lines.append("")
        lines.append(f"skip_reason_dist = {stats.get('skip_reason_dist', collections.Counter()).most_common()}")
        lines.append("-- first skips --")
        lines.extend(str(x) for x in skips[:80])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True, help="ECSL slot candidate JSONL.")
    ap.add_argument("--module-b-jsonl", nargs="+", required=True, help="Module-B fullcandidates JSONL used to recover graph/source tensors.")
    ap.add_argument("--source-jsonl", nargs="+", default=None, help="Optional original enriched binskel/source JSONL to recover bin/src/alignment tensors when Module-B output is slim.")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--save-tokenizer", default=None)
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-module-b-rows", type=int, default=0)
    ap.add_argument("--max-source-rows", type=int, default=0)
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--min-candidate-tier", default="C1", choices=["C1", "C2", "C3", "C4", "C5"])
    ap.add_argument("--allow-unvalidated-candidates", action="store_true")
    ap.add_argument("--prefer-candidate-labels", action="store_true")
    ap.add_argument("--strict-source-teacher", action="store_true")
    ap.add_argument("--strict-source-teacher-drop-record", action="store_true")
    ap.add_argument("--min-source-supervised-semantic-slots", type=int, default=1)
    ap.add_argument("--min-supervised-slots-after-fallback", type=int, default=1)
    ap.add_argument("--mask-supervised-inputs", action="store_true")
    ap.add_argument("--supervised-input-mask-prob", type=float, default=1.0)
    ap.add_argument("--asm-max-total-lines", type=int, default=0)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=0)
    ap.add_argument("--prompt-compact-mode", default="ultra", choices=["auto", "full", "compact", "ultra"])
    ap.add_argument("--max-slot-context-chars", type=int, default=256)
    ap.add_argument(
        "--rewrite-source-prefix",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Rewrite source_file prefixes before source teacher projection.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records, skips, stats = build_records(args)
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    out_txt.write_text(summarize(records, skips, stats, args), encoding="utf-8")
    print(f"records_ok = {len(records)}")
    print(f"records_skipped = {len(skips)}")
    print(f"saved_jsonl = {out_jsonl}")
    print(f"saved_txt = {out_txt}")


if __name__ == "__main__":
    main()

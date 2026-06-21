#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a tiny Module-C overfit SFT set from B-v3 slot tasks.

This script is the bridge for the first C/D overfit experiment:

1. Read the 10-row ``ModuleCDSlotTaskFromB3.v1`` JSONL exported from B-v3.
2. Join each row back to the original compact/enriched training record.
3. Reuse ``augment_binskel_sft.process_record`` with an external B-v3
   skeleton, so labels are projected from ``block_src_lines`` + source files
   onto B-v3 slots instead of the old binskel skeleton.
4. Emit normal Stage3 fields: ``sft_input_ids``, ``sft_labels``,
   ``slot_types``, ``sft_fill_positions``, and ``sft_slot_bb_indices``.

The resulting JSONL is intentionally for a 10-sample overfit sanity check, not
for claiming generalization.
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

@dataclass
class ExternalSlotInfo:
    slot_id: str
    slot_type: str
    aligned_bb_indices: List[int]
    binary_context_text: str = ""


@dataclass
class ExternalSkeleton:
    skeleton_text: str
    slots: List[ExternalSlotInfo]
    reliability: Optional[float] = None


def load_augment_module() -> Any:
    path = ROOT / "scripts" / "augment_binskel_sft.py"
    spec = importlib.util.spec_from_file_location("_augment_binskel_sft_b3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def task_key(task: Dict[str, Any]) -> Tuple[Optional[int], str]:
    src = task.get("source") if isinstance(task.get("source"), dict) else {}
    line_no = src.get("b3_input_line")
    func = str(task.get("func_name") or "")
    return safe_int(line_no, -1), func


def source_record_keys(line_no: int, row: Dict[str, Any]) -> List[Tuple[Optional[int], str]]:
    funcs = [
        row.get("func_name"),
        row.get("function"),
        row.get("name"),
        row.get("function_name"),
    ]
    out: List[Tuple[Optional[int], str]] = []
    for f in funcs:
        if f:
            out.append((line_no, str(f)))
            out.append((None, str(f)))
    return out


def load_source_index(paths: Sequence[Path], *, limit: int = 0) -> Dict[Tuple[Optional[int], str], Dict[str, Any]]:
    index: Dict[Tuple[Optional[int], str], Dict[str, Any]] = {}
    for path in paths:
        for line_no, row in iter_jsonl(path, limit=limit):
            for key in source_record_keys(line_no, row):
                index.setdefault(key, row)
    return index


def b3_slot_to_slotinfo(slot: Dict[str, Any]) -> Optional[ExternalSlotInfo]:
    sid = safe_int(slot.get("slot_id"), -1)
    if sid < 0:
        return None
    b3_type = str(slot.get("b3_slot_type") or slot.get("slot_type") or "")
    mc = slot.get("module_c") if isinstance(slot.get("module_c"), dict) else {}
    fill_type = str(mc.get("module_c_fill_type") or "")
    if not fill_type:
        if b3_type == "COND":
            fill_type = "FILL_COND"
        elif b3_type == "BLOCK":
            fill_type = "FILL_STMT"
        else:
            return None
    if fill_type.startswith("TEMPLATE_"):
        return None
    if fill_type not in {"FILL_COND", "FILL_STMT", "FILL_EXPR", "FILL_TYPE", "FILL_PARAM"}:
        return None
    bbs = slot.get("slot_mask_bb_indices") or slot.get("primary_bbs") or slot.get("context_bbs") or []
    good_bbs = [safe_int(x, -1) for x in bbs]
    good_bbs = [x for x in good_bbs if x >= 0]
    ctx = ""
    try:
        ctx = json.dumps(slot.get("grounding") or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        ctx = ""
    return ExternalSlotInfo(
        slot_id=f"slot_{sid:03d}",
        slot_type=fill_type,
        aligned_bb_indices=good_bbs,
        binary_context_text=ctx[:4096],
    )


def normalize_b3_skeleton_text(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"<FILL_COND:slot_(\d+)>", lambda m: f"<FILL_COND:slot_{int(m.group(1)):03d}>", s)
    s = re.sub(r"<FILL_BLOCK:slot_(\d+)>", lambda m: f"<FILL_STMT:slot_{int(m.group(1)):03d}>", s)
    s = re.sub(r"<FILL_STMT:slot_(\d+)>", lambda m: f"<FILL_STMT:slot_{int(m.group(1)):03d}>", s)
    s = re.sub(
        r"<CASE_SLOT:slot_(\d+)>;?",
        lambda m: f"/* <CASE_SLOT:slot_{int(m.group(1)):03d}> */ ;",
        s,
    )
    return s


def external_skeleton_from_task(task: Dict[str, Any]) -> ExternalSkeleton:
    vs = task.get("verified_skeleton") if isinstance(task.get("verified_skeleton"), dict) else {}
    skeleton_text = normalize_b3_skeleton_text(str(vs.get("skeleton") or ""))
    slots: List[ExternalSlotInfo] = []
    for slot in task.get("module_c_slots") or []:
        if not isinstance(slot, dict):
            continue
        info = b3_slot_to_slotinfo(slot)
        if info is not None:
            slots.append(info)
    if not skeleton_text or not slots:
        raise ValueError("task has no usable B-v3 skeleton/slots")
    return ExternalSkeleton(skeleton_text=skeleton_text, slots=slots, reliability=1.0)


def task_label_alignment_ok(task: Dict[str, Any]) -> Tuple[bool, str]:
    quality = task.get("module_b_quality") if isinstance(task.get("module_b_quality"), dict) else {}
    required_true = [
        "hard_contract_ok",
        "doc_quality_ready",
        "rank_trainable",
        "c_fillable_ranker_ready",
        "modulec_safe_for_rank",
        "modulec_teacher_render_structural_ready",
        "modulec_cond_count_match",
    ]
    for key in required_true:
        if not bool(quality.get(key)):
            return False, f"{key}=False"
    if safe_float(quality.get("modulec_cond_slot_match_ratio"), 0.0) < 0.98:
        return False, "modulec_cond_slot_match_ratio<0.98"
    if safe_int(quality.get("modulec_unsafe_nonempty_slot_count")) > 0:
        return False, "modulec_unsafe_nonempty_slot_count>0"
    if safe_int(quality.get("modulec_missing_teacher_primary_count")) > 0:
        return False, "modulec_missing_teacher_primary_count>0"
    return True, "ok"


def ensure_minimal_stage_fields(row: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    fi = task.get("function_ir") if isinstance(task.get("function_ir"), dict) else {}
    if not out.get("func_name") and task.get("func_name"):
        out["func_name"] = task.get("func_name")
    if not out.get("source_file"):
        src = fi.get("source_path") or fi.get("source_file") or fi.get("src_file")
        if src:
            out["source_file"] = src
    if not out.get("binary_path") and fi.get("binary_path"):
        out["binary_path"] = fi.get("binary_path")

    n_bb = len(out.get("bin_block_lengths") or [])
    if n_bb <= 0:
        bbs = fi.get("basic_blocks") if isinstance(fi.get("basic_blocks"), list) else []
        if bbs:
            out["bin_block_lengths"] = [safe_int(b.get("instruction_count"), 1) for b in bbs if isinstance(b, dict)]
            n_bb = len(out["bin_block_lengths"])
    if n_bb > 0:
        out.setdefault("pattern_labels", [0] * n_bb)
        out.setdefault("bin_struct_features", [[0.0] * 12 for _ in range(n_bb)])
        if "bin_edge_index" not in out:
            edges = fi.get("cfg_edges") if isinstance(fi.get("cfg_edges"), list) else []
            srcs = [safe_int(e.get("src"), -1) for e in edges if isinstance(e, dict)]
            dsts = [safe_int(e.get("dst"), -1) for e in edges if isinstance(e, dict)]
            pairs = [(s, d) for s, d in zip(srcs, dsts) if 0 <= s < n_bb and 0 <= d < n_bb]
            out["bin_edge_index"] = [[s for s, _ in pairs], [d for _, d in pairs]]
            out["bin_edge_type"] = [0] * len(pairs)
    ni = sum(safe_int(x) for x in out.get("bin_block_lengths") or [])
    out.setdefault("bin_opcodes", [0] * ni)
    out.setdefault("bin_operand_types", [0] * (ni * 3))
    out.setdefault("bin_operand_values", [0] * (ni * 3))
    return out


def count_supervised_tokens(record: Dict[str, Any]) -> int:
    return sum(1 for x in record.get("sft_labels") or [] if safe_int(x, -100) != -100)


def count_supervised_slots(record: Dict[str, Any]) -> int:
    flags = record.get("sft_teacher_slot_supervise")
    if isinstance(flags, list):
        return sum(1 for x in flags if bool(x))
    targets = record.get("sft_teacher_targets") or []
    return sum(1 for x in targets if str(x or "").strip())


CONTROL_LIKE_STMT_RE = re.compile(
    r"(^|[;{}\s])(?:if|while|for|switch|else|case|default)\b"
)


def prune_control_like_stmt_labels(record: Dict[str, Any]) -> int:
    """Drop noisy statement labels that duplicate control/switch structure.

    B-v3 already owns control structure.  For the first overfit experiment,
    a FILL_STMT target containing ``if (...)`` or ``case/default:`` teaches the
    model to put control syntax inside a block slot, which is exactly the bad
    failure mode we want to avoid.
    """
    slot_types = list(record.get("slot_types") or [])
    targets = list(record.get("sft_teacher_targets") or [])
    raw_targets = list(record.get("sft_teacher_targets_raw") or [])
    flags = list(record.get("sft_teacher_slot_supervise") or [bool(str(x or "").strip()) for x in targets])
    changed = 0
    n = min(len(slot_types), len(targets), len(flags))
    drops: List[Dict[str, Any]] = list(record.get("b3_overfit_pruned_teacher_labels") or [])
    for i in range(n):
        typ = str(slot_types[i])
        tgt = str(targets[i] or "").strip()
        if typ != "FILL_STMT" or not flags[i] or not tgt:
            continue
        if CONTROL_LIKE_STMT_RE.search(tgt):
            flags[i] = False
            targets[i] = ""
            if i < len(raw_targets):
                raw_targets[i] = ""
            drops.append(
                {
                    "slot": int(i),
                    "slot_type": typ,
                    "reason": "control_like_stmt_label_pruned_for_b3_overfit",
                    "target": tgt,
                }
            )
            changed += 1
    record["sft_teacher_targets"] = targets
    if raw_targets:
        record["sft_teacher_targets_raw"] = raw_targets
    record["sft_teacher_slot_supervise"] = flags
    record["b3_overfit_pruned_teacher_labels"] = drops
    return changed


def rebuild_sft_with_targets(
    record: Dict[str, Any],
    aug: Any,
    tokenizer: Any,
    fill_ids: List[int],
    *,
    mask_non_fill_labels: bool = True,
) -> None:
    """Rebuild body/input labels after pruning teacher targets.

    ``augment_binskel_sft.process_record`` has already produced the skeleton
    prompt.  Re-interleave the same skeleton with the pruned target list so the
    token-level SFT labels match ``sft_teacher_slot_supervise``.
    """
    skeleton = str(record.get("sft_skeleton_text_pretty") or "")
    slot_types = list(record.get("slot_types") or [])
    slot_ids = [str(x) for x in (record.get("sft_prompt_slot_ids") or [])]
    if not skeleton or not slot_types:
        return
    slots = [
        ExternalSlotInfo(slot_id=slot_ids[i] if i < len(slot_ids) else f"slot_{i:03d}", slot_type=str(t), aligned_bb_indices=[])
        for i, t in enumerate(slot_types)
    ]
    targets = list(record.get("sft_teacher_targets") or [])
    flags = list(record.get("sft_teacher_slot_supervise") or [bool(str(x or "").strip()) for x in targets])
    body_ids, body_sup, fill_pos_body = aug._interleave_skeleton_token_ids(
        skeleton,
        slots,
        targets,
        fill_ids,
        tokenizer,
        slot_supervise=flags,
    )
    old_ids = list(record.get("sft_input_ids") or [])
    old_labels = list(record.get("sft_labels") or [])
    old_fill_positions = [int(x) for x in (record.get("sft_fill_positions") or [])]
    if old_fill_positions:
        head_len = max(0, old_fill_positions[0] - fill_pos_body[0])
    else:
        head_len = 0
    tail_start_old = 0
    if old_fill_positions:
        last_old_fill = old_fill_positions[-1]
        tail_start_old = last_old_fill + 1
        while tail_start_old < len(old_labels) and int(old_labels[tail_start_old]) != -100:
            tail_start_old += 1
    old_head = old_ids[:head_len]
    old_tail = old_ids[tail_start_old:] if tail_start_old < len(old_ids) else []
    input_ids = old_head + body_ids + old_tail
    labels = list(input_ids)
    supervised_mask = [False] * len(old_head) + body_sup + [False] * len(old_tail)
    if mask_non_fill_labels:
        for i, keep in enumerate(supervised_mask):
            if not keep:
                labels[i] = -100
    record["sft_input_ids"] = input_ids
    record["sft_labels"] = labels
    record["sft_fill_positions"] = [len(old_head) + int(p) for p in fill_pos_body]


def summarize(records: Sequence[Dict[str, Any]], skips: Sequence[str]) -> str:
    lines: List[str] = []
    n = len(records)
    slot_type_dist = collections.Counter()
    slot_supervised = 0
    slot_total = 0
    token_counts: List[int] = []
    target_examples: List[str] = []
    pruned = 0
    for rec in records:
        slot_type_dist.update(rec.get("slot_types") or [])
        flags = rec.get("sft_teacher_slot_supervise")
        targets = rec.get("sft_teacher_targets") or []
        if isinstance(flags, list):
            slot_total += len(flags)
            slot_supervised += sum(1 for x in flags if bool(x))
        else:
            slot_total += len(targets)
            slot_supervised += sum(1 for x in targets if str(x or "").strip())
        token_counts.append(count_supervised_tokens(rec))
        for typ, tgt in zip(rec.get("slot_types") or [], targets):
            if tgt and len(target_examples) < 16:
                target_examples.append(f"{rec.get('func_name')} {typ}: {str(tgt)[:120]}")
        pruned += len(rec.get("b3_overfit_pruned_teacher_labels") or [])
    lines.append("=== Module C B-v3 10-row overfit SFT build ===")
    lines.append(f"records_ok = {n}")
    lines.append(f"records_skipped = {len(skips)}")
    lines.append(f"slot_type_dist = {sorted(slot_type_dist.items())}")
    lines.append(f"supervised_slots = {pct(slot_supervised, slot_total)}")
    lines.append(f"control_like_stmt_labels_pruned = {pruned}")
    if token_counts:
        lines.append(f"supervised_tokens_total = {sum(token_counts)}")
        lines.append(f"supervised_tokens_min = {min(token_counts)}")
        lines.append(f"supervised_tokens_max = {max(token_counts)}")
    lines.append("")
    lines.append("-- target preview --")
    lines.extend(target_examples or ["<none>"])
    if skips:
        lines.append("")
        lines.append("-- skips --")
        lines.extend(skips[:50])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot-tasks-jsonl", required=True)
    ap.add_argument("--source-jsonl", nargs="+", required=True, help="Original compact/enriched rows used by B-v3")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--save-tokenizer", default=None)
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-source-rows", type=int, default=0)
    ap.add_argument("--max-fill-tokens", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--strict-teacher", action="store_true")
    ap.add_argument(
        "--allow-relaxed-label-alignment",
        action="store_true",
        help=(
            "Allow building SFT labels from slot-task rows that do not pass the "
            "high-confidence B-v3/teacher alignment contract. Use only for debugging."
        ),
    )
    ap.add_argument(
        "--drop-control-like-stmt-labels",
        action="store_true",
        help="Drop FILL_STMT labels containing if/while/for/switch/case/default before overfit training.",
    )
    ap.add_argument("--min-supervised-semantic-slots", type=int, default=1)
    ap.add_argument("--asm-max-total-lines", type=int, default=80)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=8)
    ap.add_argument(
        "--rewrite-source-prefix",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Passed through to augment source path resolver.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    aug = load_augment_module()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"需要 transformers: {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    extra = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(int(args.max_fill_tokens))]
    tokenizer.add_tokens(extra)
    fill_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in extra]
    if args.save_tokenizer:
        Path(args.save_tokenizer).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(Path(args.save_tokenizer).expanduser().resolve()))

    source_paths = [Path(p).expanduser().resolve() for p in args.source_jsonl]
    source_index = load_source_index(source_paths, limit=int(args.max_source_rows))
    tasks = [row for _line, row in iter_jsonl(Path(args.slot_tasks_jsonl).expanduser().resolve(), int(args.max_examples))]

    out_records: List[Dict[str, Any]] = []
    skips: List[str] = []
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    prefix_rewrites = [(str(a), str(b)) for a, b in (args.rewrite_source_prefix or [])]

    for task in tasks:
        key = task_key(task)
        if not bool(args.allow_relaxed_label_alignment):
            ok, reason = task_label_alignment_ok(task)
            if not ok:
                skips.append(f"{key}: low-confidence B-v3 label alignment: {reason}")
                continue
        src = source_index.get(key) or source_index.get((None, key[1]))
        if src is None:
            skips.append(f"{key}: source row not found")
            continue
        try:
            skel = external_skeleton_from_task(task)
        except Exception as e:
            skips.append(f"{key}: skeleton error: {e}")
            continue
        work = ensure_minimal_stage_fields(src, task)
        rec, err = aug.process_record(
            work,
            tokenizer,
            fill_ids,
            source_root,
            emit_rl_fields=True,
            num_patterns=10,
            mask_prompt_labels=True,
            mask_non_fill_labels=True,
            mask_supervised_inputs=True,
            supervised_input_mask_prob=1.0,
            max_seq_len=int(args.max_seq_len),
            prefix_rewrites=prefix_rewrites,
            external_skeleton=skel,
            skeleton_source="module_b_v3_verified_skeleton",
            strict_teacher=bool(args.strict_teacher),
            strict_teacher_drop_record=False,
            min_supervised_semantic_slots=int(args.min_supervised_semantic_slots),
            asm_max_total_lines=int(args.asm_max_total_lines),
            asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
        )
        if rec is None:
            skips.append(f"{key}: {err}")
            continue
        if bool(args.drop_control_like_stmt_labels):
            n_pruned = prune_control_like_stmt_labels(rec)
            if n_pruned:
                rebuild_sft_with_targets(rec, aug, tokenizer, fill_ids)
        rec["b3_module_cd_task"] = {
            "func_name": task.get("func_name"),
            "b3_source": task.get("source"),
            "module_b_quality": task.get("module_b_quality"),
        }
        rec["b3_verified_skeleton_text"] = (task.get("verified_skeleton") or {}).get("skeleton")
        rec["b3_slot_tasks"] = task.get("module_c_slots") or []
        out_records.append(rec)

    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    out_txt.write_text(summarize(out_records, skips), encoding="utf-8")
    print(f"records_ok = {len(out_records)}/{len(tasks)}")
    print(f"saved_jsonl = {out_jsonl}")
    print(f"saved_txt = {out_txt}")
    if skips:
        print(f"skips = {len(skips)}")


if __name__ == "__main__":
    main()

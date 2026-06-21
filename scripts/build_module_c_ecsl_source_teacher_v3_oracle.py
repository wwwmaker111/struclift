#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a training-time source-teacher projection JSONL.

This script is for the supervised-training setting where the original source
teacher is available.  It does not claim binary differential execution.  It
can provide cleaned skeleton-compatible targets for bootstrap/weak supervision,
but those targets are not V3/C3 semantic proof.  This follows the ECSL document's
rule that unmatched B slots are masked out rather than forcing a bad label.  A
legacy function-level strict mode is available with ``--label-policy function``.

The output JSONL is useful for auditing source projection coverage.  It should
not be supplied as true ``--v3-results-jsonl`` for hard SFT unless a separate
differential/trace/symbolic validator has promoted the rows.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FILL_PLACEHOLDER_RE = re.compile(r"<FILL_[A-Z_]+:[^>]+>")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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


def q(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def compact_counter(counter: collections.Counter, *, limit: int = 40) -> List[Tuple[Any, int]]:
    items = counter.most_common()
    if len(items) <= limit:
        return items
    return items[:limit] + [("__other__", sum(v for _, v in items[limit:]))]


def compile_reward_like(code: str, *, cc: str, timeout: int) -> Tuple[float, Dict[str, Any]]:
    boilerplate = "\n".join(
        [
            "#include <stddef.h>",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "",
        ]
    )
    full = boilerplate + "\n" + str(code or "")
    try:
        proc = subprocess.run(
            [cc, "-fsyntax-only", "-x", "c", "-", "-w"],
            input=full.encode("utf-8", errors="replace"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout),
        )
    except Exception as e:
        return 0.0, {"status": "exception", "reason": str(e), "cc": cc}
    stderr = proc.stderr.decode("utf-8", errors="replace")
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode == 0:
        return 1.0, {"status": "pass", "returncode": 0, "cc": cc}
    error_lines = [line for line in stderr.splitlines() if ": error:" in line or "error:" in line]
    score = max(0.0, 1.0 - 0.1 * len(error_lines))
    return score, {
        "status": "fail",
        "returncode": proc.returncode,
        "cc": cc,
        "error_count": len(error_lines),
        "stdout_head": stdout[:2000],
        "stderr_head": stderr[:4000],
    }


def source_row_for(helper: Any, source_index: Dict[Tuple[Any, Any, str], Dict[str, Any]], cand_record: Dict[str, Any], mb: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    src = helper.find_module_b_row(source_index, cand_record) if source_index else None
    if source_index and src is None:
        src = helper.find_module_b_row(source_index, mb)
    return src


def confidence_weight(level: str) -> float:
    return {
        "C5": 1.0,
        "C4": 0.8,
        "C3": 0.5,
        "C2": 0.2,
        "C1": 0.0,
        "C0": 0.0,
    }.get(str(level), 0.0)


def prompt_slot_rows(rec: Dict[str, Any], *, pass_tier: str) -> List[Dict[str, Any]]:
    slot_ids = [str(x) for x in (rec.get("sft_prompt_slot_ids") or [])]
    slot_types = [str(x) for x in (rec.get("slot_types") or [])]
    targets = list(rec.get("sft_teacher_targets_prompt_order") or rec.get("sft_teacher_targets") or [])
    flags = list(rec.get("sft_teacher_slot_supervise_prompt_order") or rec.get("sft_teacher_slot_supervise") or [])
    raw_targets = list(rec.get("sft_teacher_targets_raw_prompt_order") or rec.get("sft_teacher_targets_raw") or [])
    original_indices = [
        safe_int(x, i)
        for i, x in enumerate(rec.get("sft_prompt_slot_original_indices") or list(range(max(len(slot_ids), len(targets)))))
    ]
    drop_by_slot: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in rec.get("sft_teacher_strict_drop_reasons") or []:
        if not isinstance(item, dict):
            continue
        idx = safe_int(item.get("slot"), -1)
        if idx >= 0:
            drop_by_slot[idx].append(item)

    n = max(len(slot_ids), len(slot_types), len(targets), len(flags))
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        sid = slot_ids[i] if i < len(slot_ids) else f"slot_{i:03d}"
        typ = slot_types[i] if i < len(slot_types) else ""
        target = str(targets[i] or "") if i < len(targets) else ""
        raw = str(raw_targets[i] or "") if i < len(raw_targets) else ""
        kept = bool(flags[i]) if i < len(flags) else False
        source_teacher_passed = bool(kept and target.strip())
        hard_ce_allowed = False
        level = "C2" if source_teacher_passed else "C0"
        reason = "source_teacher_slot_projection_pass" if source_teacher_passed else "unmatched_or_unusable_source_teacher_slot"
        original_idx = original_indices[i] if i < len(original_indices) else i
        if not source_teacher_passed and drop_by_slot.get(original_idx):
            reason = ",".join(str(x.get("reason") or "drop") for x in drop_by_slot[original_idx][:4])
        rows.append(
            {
                "slot_index": i,
                "original_slot_index": original_idx,
                "slot_id": sid,
                "slot_type": typ,
                "passed": source_teacher_passed,
                "source_teacher_passed": source_teacher_passed,
                "v3_pass": False,
                "differential_pass": False,
                "trace_match": False,
                "symbolic_match": False,
                "confidence_level": level,
                "confidence_weight": confidence_weight(level),
                "hard_ce_allowed": hard_ce_allowed,
                "target_text": target if source_teacher_passed else "",
                "raw_teacher_text": raw[:1000],
                "reason": reason,
                "oracle_type": "training_source_teacher_slot_projection",
                "validation": {
                    "V3": {
                        "passed": hard_ce_allowed,
                        "status": "source_teacher_projection_not_v3" if source_teacher_passed else "fail",
                        "reason": reason,
                    }
                },
            }
        )
    return rows


def build(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    helper = load_module("_ecsl_overfit_helpers_for_v3_oracle", ROOT / "scripts" / "build_module_c_ecsl_overfit_sft.py")
    strong = load_module("_ecsl_strong_helpers_for_v3_oracle", ROOT / "scripts" / "build_module_c_ecsl_strong_sft.py")
    aug = helper.load_augment_module()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"需要 transformers: {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fill_tokens = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(int(args.max_fill_tokens))]
    tokenizer.add_tokens(fill_tokens)
    fill_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in fill_tokens]

    module_b_paths = [Path(p).expanduser().resolve() for p in args.module_b_jsonl]
    source_paths = [Path(p).expanduser().resolve() for p in args.source_jsonl]
    module_b_index = helper.load_module_b_index(module_b_paths, limit=int(args.max_module_b_rows))
    source_index = helper.load_module_b_index(source_paths, limit=int(args.max_source_rows))
    candidate_path = Path(args.candidates_jsonl).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    prefix_rewrites = [(str(a), str(b)) for a, b in (args.rewrite_source_prefix or [])]

    pass_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "rows_seen": 0,
        "source_teacher_pass": 0,
        "source_teacher_fail": 0,
        "hard_v3_pass": 0,
        "skip_reason_dist": collections.Counter(),
        "tier_dist": collections.Counter(),
        "slot_tier_dist": collections.Counter(),
        "compile_status_dist": collections.Counter(),
        "function_v2_proxy_status_dist": collections.Counter(),
        "supervised_slot_count_dist": collections.Counter(),
        "slot_count": 0,
        "source_teacher_pass_slot_count": 0,
        "hard_v3_pass_slot_count": 0,
    }

    for _line_no, cand_record in iter_jsonl(candidate_path, limit=int(args.limit)):
        stats["rows_seen"] += 1
        key = helper.contract_function_key(cand_record)
        reasons: List[str] = []
        mb = helper.find_module_b_row(module_b_index, cand_record)
        if mb is None:
            reasons.append("module_b_row_not_found")
            row = {
                "function_key": cand_record.get("function_key"),
                "example": cand_record.get("example"),
                "input_line": cand_record.get("input_line"),
                "func_name": cand_record.get("func_name"),
                "passed": False,
                "v3_pass": False,
                "confidence_level": "C0",
                "reason": ";".join(reasons),
                "oracle_type": "source_teacher_strict_projection",
            }
            audit_rows.append(row)
            stats["source_teacher_fail"] += 1
            stats["tier_dist"]["C0"] += 1
            stats["skip_reason_dist"][reasons[0]] += 1
            continue
        src = source_row_for(helper, source_index, cand_record, mb)
        try:
            skel = helper.external_skeleton_from_candidate_record(cand_record, mb)
        except Exception as e:
            reasons.append(f"skeleton_error:{e}")
            skel = None

        rec = None
        err = ""
        if skel is not None:
            work = helper.ensure_minimal_stage_fields(helper.merge_source_and_module_b(src, mb))
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
                    skeleton_source="module_c_ecsl_source_teacher_projection",
                    strict_teacher=True,
                    strict_teacher_drop_record=str(args.label_policy) == "function",
                    min_supervised_semantic_slots=int(args.min_supervised_semantic_slots),
                    asm_max_total_lines=int(args.asm_max_total_lines),
                    asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                    prompt_compact_mode=str(args.prompt_compact_mode),
                    max_slot_context_chars=int(args.max_slot_context_chars),
                )
            except Exception as e:
                rec, err = None, f"process_record_exception:{e}"
        if rec is None:
            reasons.append(str(err or "process_record_failed"))

        bg_ok = v0_ok = v1_ok = function_v2_proxy_ok = False
        bg_reasons: List[str] = []
        v01_reasons: List[str] = []
        function_v2_proxy_reasons: List[str] = []
        function_v2_proxy_evidence: Dict[str, Any] = {}
        rendered = ""
        supervised_slots = 0
        supervised_tokens = 0
        slot_rows: List[Dict[str, Any]] = []
        slot_pass_count = 0
        hard_v3_slot_count = 0
        if rec is not None:
            bg_ok, bg_reasons = strong.b_gate_pass(cand_record, mb)
            v0_ok, v1_ok, v01_reasons = strong.v0_v1_pass(rec)
            function_v2_proxy_ok, function_v2_proxy_reasons, function_v2_proxy_evidence = strong.v2_static_constructability_pass(mb, rec)
            rendered = str(rec.get("sft_output_text_pretty") or "")
            supervised_slots = strong.count_supervised_slots(rec)
            supervised_tokens = strong.count_supervised_tokens(rec)
            stats["supervised_slot_count_dist"][supervised_slots] += 1
            slot_rows = prompt_slot_rows(rec, pass_tier=str(args.pass_tier))
            slot_pass_count = sum(1 for row in slot_rows if row.get("source_teacher_passed", row.get("passed")))
            hard_v3_slot_count = sum(1 for row in slot_rows if row.get("v3_pass") and row.get("hard_ce_allowed"))
            stats["slot_count"] += len(slot_rows)
            stats["source_teacher_pass_slot_count"] += slot_pass_count
            stats["hard_v3_pass_slot_count"] += hard_v3_slot_count
            stats["slot_tier_dist"].update(str(row.get("confidence_level") or "C0") for row in slot_rows)
        if not bg_ok:
            reasons.extend(bg_reasons)
        if not v0_ok or not v1_ok:
            reasons.extend(v01_reasons)
        if str(args.label_policy) == "function" and bool(args.require_v2) and not function_v2_proxy_ok:
            reasons.extend(function_v2_proxy_reasons or ["function_v2_proxy_failed"])
        if rec is not None and FILL_PLACEHOLDER_RE.search(rendered):
            reasons.append("remaining_fill_placeholder_after_render")
        if rec is not None and not rendered.strip():
            reasons.append("empty_rendered_teacher")

        compile_score = None
        compile_report: Dict[str, Any] = {"status": "skipped"}
        if str(args.label_policy) == "slot" and str(args.compile_policy) in {"record", "require"}:
            compile_report = {
                "status": "skipped_slot_level_partial_render",
                "reason": "slot-level source-teacher oracle validates targets individually; full rendered code may intentionally contain masked slots",
            }
            stats["compile_status_dist"][str(compile_report.get("status"))] += 1
        elif str(args.compile_policy) in {"record", "require"} and rec is not None:
            compile_score, compile_report = compile_reward_like(
                rendered,
                cc=str(args.cc),
                timeout=int(args.compile_timeout),
            )
            stats["compile_status_dist"][str(compile_report.get("status") or "unknown")] += 1
            if str(args.compile_policy) == "require" and float(compile_score) < float(args.min_compile_reward):
                reasons.append(f"compile_reward_below_threshold={compile_score:.4f}<{float(args.min_compile_reward):.4f}")
        else:
            stats["compile_status_dist"]["skipped"] += 1

        if str(args.label_policy) == "slot":
            passed = bool(
                rec is not None
                and bg_ok
                and v0_ok
                and v1_ok
                and slot_pass_count >= max(1, int(args.min_supervised_semantic_slots))
                and not reasons
            )
        else:
            passed = bool(rec is not None and bg_ok and v0_ok and v1_ok and (function_v2_proxy_ok or not bool(args.require_v2)) and not reasons)
        level = "C2" if passed else "C0"
        row = {
            "schema": "ModuleC.ECSL.SourceTeacherProjection.v1",
            "function_key": cand_record.get("function_key"),
            "example": cand_record.get("example"),
            "input_line": cand_record.get("input_line"),
            "func_name": cand_record.get("func_name"),
            "passed": passed,
            "source_teacher_pass": passed,
            "v3_pass": False,
            "differential_pass": False,
            "trace_match": False,
            "symbolic_match": False,
            "confidence_level": level,
            "tier": level,
            "reason": (
                f"source_teacher_projection_pass_slots={slot_pass_count}/{len(slot_rows)}"
                if passed and str(args.label_policy) == "slot"
                else ("source_teacher_strict_projection_pass" if passed else ";".join(str(x) for x in reasons[:12]))
            ),
            "status": "pass" if passed else "fail",
            "oracle_type": "training_source_teacher_slot_projection" if str(args.label_policy) == "slot" else "training_source_teacher_strict_render",
            "oracle_note": "Training-time source-teacher projection; this is bootstrap/weak evidence, not binary differential V3.",
            "label_policy": str(args.label_policy),
            "source_teacher_available": src is not None,
            "b_gate_pass": bool(bg_ok),
            "v0_pass": bool(v0_ok),
            "v1_pass": bool(v1_ok),
            "function_v2_proxy_pass": bool(function_v2_proxy_ok),
            "function_v2_proxy_reasons": function_v2_proxy_reasons,
            "function_v2_proxy_evidence": function_v2_proxy_evidence,
            "function_v2_proxy_note": "Diagnostic source/Module-B constructability proxy; slot-label source projection pass is controlled by source_teacher_pass_slot_count, not this field.",
            "supervised_slot_count": supervised_slots,
            "source_teacher_pass_slot_count": slot_pass_count,
            "v3_pass_slot_count": 0,
            "slot_v3_results": slot_rows,
            "supervised_token_count": supervised_tokens,
            "sft_seq_len": len(rec.get("sft_input_ids") or []) if rec is not None else None,
            "sft_prompt_compact_mode": rec.get("sft_prompt_compact_mode") if rec is not None else None,
            "sft_prompt_retry_trace": rec.get("sft_prompt_retry_trace") if rec is not None else None,
            "compile_policy": str(args.compile_policy),
            "compile_reward": compile_score,
            "compile_report": compile_report,
            "module_b_candidate_name": q(mb, "verified_skeleton", "candidate_name"),
            "module_b_selection_source": q(mb, "verified_skeleton", "selection_source"),
        }
        audit_rows.append(row)
        stats["tier_dist"][level] += 1
        stats["function_v2_proxy_status_dist"]["pass" if function_v2_proxy_ok else "fail"] += 1
        if passed:
            pass_rows.append(row)
            stats["source_teacher_pass"] += 1
            if row.get("v3_pass"):
                stats["hard_v3_pass"] += 1
        else:
            stats["source_teacher_fail"] += 1
            first = reasons[0] if reasons else "unknown_fail"
            stats["skip_reason_dist"][str(first).split(":")[0][:120]] += 1

    return pass_rows, audit_rows, stats


def write_report(path: Path, *, args: argparse.Namespace, pass_rows: Sequence[Dict[str, Any]], audit_rows: Sequence[Dict[str, Any]], stats: Dict[str, Any]) -> None:
    n = int(stats.get("rows_seen") or 0)
    lines: List[str] = []
    lines.append("=== Module C ECSL Source-Teacher Projection Build ===")
    lines.append(f"candidates_jsonl = {args.candidates_jsonl}")
    lines.append(f"module_b_jsonl = {list(args.module_b_jsonl)}")
    lines.append(f"source_jsonl = {list(args.source_jsonl)}")
    lines.append(f"out_jsonl = {args.out_jsonl}")
    lines.append("")
    lines.append(f"rows_seen = {n}")
    hard_v3_rows = sum(1 for row in pass_rows if row.get("v3_pass"))
    lines.append(f"source_teacher_pass_rows = {pct(len(pass_rows), n)}")
    lines.append(f"hard_v3_rows = {pct(hard_v3_rows, n)}")
    lines.append(f"source_teacher_fail_rows = {pct(len(audit_rows) - len(pass_rows), n)}")
    lines.append(f"tier_dist = {compact_counter(stats['tier_dist'])}")
    lines.append(f"slot_tier_dist = {compact_counter(stats['slot_tier_dist'])}")
    lines.append(f"source_teacher_pass_slot_count = {stats['source_teacher_pass_slot_count']}/{max(1, stats['slot_count'])} ({100.0 * stats['source_teacher_pass_slot_count'] / max(1, stats['slot_count']):.1f}%)")
    lines.append(f"hard_v3_pass_slot_count = {stats['hard_v3_pass_slot_count']}/{max(1, stats['slot_count'])} ({100.0 * stats['hard_v3_pass_slot_count'] / max(1, stats['slot_count']):.1f}%)")
    lines.append(f"function_v2_proxy_status_dist = {compact_counter(stats['function_v2_proxy_status_dist'])}")
    lines.append(f"compile_status_dist = {compact_counter(stats['compile_status_dist'])}")
    lines.append(f"supervised_slot_count_dist = {compact_counter(stats['supervised_slot_count_dist'])}")
    lines.append(f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Passing rows are source-teacher projections, not V3 semantic proofs.")
    lines.append("- function_v2_proxy_status_dist is diagnostic only in slot label_policy; it is not the final candidate-table V2 coverage.")
    lines.append("- They may be used for bootstrap/weak audits, but hard CE still requires separate C3+ differential/trace/symbolic evidence.")
    lines.append("- --pass-tier is retained for CLI compatibility; this script now emits C2 source projection rows and v3_pass=false.")
    lines.append("")
    lines.append("-- first pass rows --")
    for row in list(pass_rows)[:20]:
        lines.append(
            json.dumps(
                {
                    "example": row.get("example"),
                    "input_line": row.get("input_line"),
                    "func_name": row.get("func_name"),
                    "supervised_slot_count": row.get("supervised_slot_count"),
                    "source_teacher_pass_slot_count": row.get("source_teacher_pass_slot_count"),
                    "v3_pass_slot_count": row.get("v3_pass_slot_count"),
                    "sft_seq_len": row.get("sft_seq_len"),
                    "compile_reward": row.get("compile_reward"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if not pass_rows:
        lines.append("<none>")
    lines.append("")
    lines.append("-- first fail rows --")
    shown = 0
    for row in audit_rows:
        if row.get("passed"):
            continue
        lines.append(
            json.dumps(
                {
                    "example": row.get("example"),
                    "input_line": row.get("input_line"),
                    "func_name": row.get("func_name"),
                    "reason": row.get("reason"),
                    "sft_seq_len": row.get("sft_seq_len"),
                    "compile_reward": row.get("compile_reward"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        shown += 1
        if shown >= 30:
            break
    if not shown:
        lines.append("<none>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--module-b-jsonl", nargs="+", required=True)
    ap.add_argument("--source-jsonl", nargs="+", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-audit-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-module-b-rows", type=int, default=0)
    ap.add_argument("--max-source-rows", type=int, default=0)
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--min-supervised-semantic-slots", type=int, default=1)
    ap.add_argument("--asm-max-total-lines", type=int, default=0)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=0)
    ap.add_argument("--prompt-compact-mode", default="ultra", choices=["auto", "full", "compact", "ultra"])
    ap.add_argument("--max-slot-context-chars", type=int, default=256)
    ap.add_argument("--mask-supervised-inputs", action="store_true")
    ap.add_argument("--supervised-input-mask-prob", type=float, default=1.0)
    ap.add_argument("--rewrite-source-prefix", nargs=2, action="append", metavar=("OLD", "NEW"))
    ap.add_argument("--require-v2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--label-policy", default="slot", choices=["slot", "function"], help="slot follows ECSL per-slot unmatched-mass semantics; function preserves legacy all-or-nothing strict render.")
    ap.add_argument("--pass-tier", default="C2", choices=["C2", "C3", "C4", "C5"], help="Deprecated compatibility option; source-teacher projection is emitted as C2 and never as true V3.")
    ap.add_argument("--compile-policy", default="record", choices=["skip", "record", "require"])
    ap.add_argument("--min-compile-reward", type=float, default=1.0)
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--compile-timeout", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pass_rows, audit_rows, stats = build(args)
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_audit = Path(args.out_audit_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    for path in (out_jsonl, out_audit, out_txt):
        path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in pass_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with out_audit.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(out_txt, args=args, pass_rows=pass_rows, audit_rows=audit_rows, stats=stats)

    n = int(stats.get("rows_seen") or 0)
    print(f"rows_seen = {n}")
    hard_v3_rows = sum(1 for row in pass_rows if row.get("v3_pass"))
    print(f"source_teacher_pass_rows = {pct(len(pass_rows), n)}")
    print(f"hard_v3_rows = {pct(hard_v3_rows, n)}")
    print(f"tier_dist = {compact_counter(stats['tier_dist'])}")
    print(f"function_v2_proxy_status_dist = {compact_counter(stats['function_v2_proxy_status_dist'])}")
    print(f"compile_status_dist = {compact_counter(stats['compile_status_dist'])}")
    print(f"skip_reason_dist = {compact_counter(stats['skip_reason_dist'])}")
    print(f"saved_source_teacher_projection_jsonl = {out_jsonl}")
    print(f"saved_audit_jsonl = {out_audit}")
    print(f"saved_report = {out_txt}")


if __name__ == "__main__":
    main()

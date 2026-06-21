#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit V10 repack skipped slots against source-alignment candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import repack_modulec_v9_to_v10 as repack  # noqa: E402


MISS_RE = re.compile(r"missing source-derived target for (FILL_[A-Z_]+) slot_(\d+)")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _slot_bbs(record: Dict[str, Any], old_idx: int) -> List[int]:
    skeleton, skeleton_src_name = repack._skeleton_text_for_repack(record, "regenerate")
    regenerated_skeleton = repack._uses_regenerated_skeleton(skeleton_src_name)
    slot_types = list(record.get("slot_types") or [])
    bb_all = list(record.get("sft_slot_bb_indices") or [])
    matches = list(repack.FILL_RE.finditer(skeleton))
    occurrence = 0
    for mi, m in enumerate(matches):
        idx = repack._slot_index_from_match(m, occurrence, len(slot_types))
        if regenerated_skeleton:
            idx = repack._slot_index_from_placeholder(m, occurrence)
        occurrence += 1
        if idx != old_idx:
            continue
        next_start = matches[mi + 1].start() if mi + 1 < len(matches) else len(skeleton)
        comment_bbs = repack._aligned_bbs_from_slot_tail(skeleton[m.end():next_start])
        bbs_raw = bb_all[old_idx] if old_idx < len(bb_all) else []
        field_bbs = [
            int(x) for x in bbs_raw
            if isinstance(x, (int, float)) and int(x) >= 0
        ] if isinstance(bbs_raw, list) else []
        return comment_bbs if regenerated_skeleton else (comment_bbs or field_bbs)
    return []


def _candidate_rows(
    record: Dict[str, Any],
    bbs: Sequence[int],
) -> Tuple[str, List[Tuple[int, float, Dict[str, int], str, bool]]]:
    M = record.get("alignment_matrix")
    if not isinstance(M, list):
        return "no_alignment_matrix", []
    lt = repack._line_to_text(record.get("source_file"))
    if not lt:
        return "no_source_file_text", []
    lt_clean = repack._strip_block_comments_from_lt(repack._strip_preprocessor_lines_from_lt(lt))
    stmt_rows = repack._source_statement_table(record)
    if not stmt_rows:
        return "no_statement_table", []
    n_stmt = len(stmt_rows)
    if any(isinstance(row, list) and len(row) != n_stmt for row in M):
        lens = sorted({len(row) for row in M if isinstance(row, list)})
        return f"matrix_stmt_count_mismatch matrix_lens={lens} stmt_rows={n_stmt}", []

    masses = [0.0] * n_stmt
    for bb in bbs:
        bi = int(bb)
        if not (0 <= bi < len(M)) or not isinstance(M[bi], list):
            continue
        for j, val in enumerate(M[bi]):
            if j >= n_stmt:
                break
            try:
                masses[j] += float(val)
            except Exception:
                pass
    if not masses or max(masses) <= 0:
        return "alignment_row_has_zero_mass", []
    top = sorted(range(n_stmt), key=lambda j: masses[j], reverse=True)[:8]
    out: List[Tuple[int, float, Dict[str, int], str, bool]] = []
    for j in top:
        if masses[j] <= 0:
            continue
        row = stmt_rows[j]
        raw = repack._source_text_for_statement_rows(lt_clean, [row])
        pretty = repack._pretty_c_fragment(raw, "FILL_STMT")
        usable_stmt = repack._target_is_usable("FILL_STMT", pretty)
        out.append((j, masses[j], row, pretty, usable_stmt))
    return "", out


def _body_slot_count(record: Dict[str, Any]) -> int:
    return sum(
        1 for st in list(record.get("slot_types") or [])
        if str(st) in repack.BODY_SLOT_TYPES
    )


def _current_source_target(
    record: Dict[str, Any],
    stype: str,
    bbs: Sequence[int],
    old_target: str,
) -> Tuple[str, str]:
    target, reason = repack._source_alignment_target_for_slot(record, stype, bbs)
    if target:
        return target, reason or "source_alignment_matrix"
    target, reason2 = repack._source_multistmt_target_for_slot(
        record,
        stype,
        bbs,
        body_slot_count=_body_slot_count(record),
        old_target=old_target,
    )
    return target, reason2 or reason


def _fallback_source_targets(
    record: Dict[str, Any],
    stype: str,
    bbs: Sequence[int],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for fn in (
        repack._source_line_columns_target_for_slot,
        repack._direct_source_lines_target_for_slot,
    ):
        try:
            target, reason = fn(record, stype, bbs)
        except Exception as exc:
            target, reason = "", f"{getattr(fn, '__name__', 'fallback')}_error:{exc}"
        if target:
            out.append((reason, target))
    return out


def _source_context(source_file: Any, row: Dict[str, int], radius: int = 2) -> str:
    lt = repack._line_to_text(source_file)
    if not lt:
        return ""
    lo = max(1, int(row.get("start_line", 0)) - radius)
    hi = min(max(lt), int(row.get("end_line", 0)) + radius)
    lines = []
    for ln in range(lo, hi + 1):
        mark = ">>" if int(row.get("start_line", 0)) <= ln <= int(row.get("end_line", 0)) else "  "
        lines.append(f"{mark} {ln:5d}: {lt.get(ln, '')}")
    return "\n".join(lines)


def _real_source_function_context(record: Dict[str, Any]) -> str:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt:
        return ""
    bounds = repack._function_body_line_range(record, lt)
    if bounds is None:
        return ""
    lo, hi = bounds
    return "\n".join(f"{ln:5d}: {lt.get(ln, '')}" for ln in range(lo, hi + 1))


def _render_output_without_tokenizer(
    record: Dict[str, Any],
    targets: Sequence[str],
    old_indices: Sequence[int],
    *,
    name_for_prompt: str = "func",
) -> str:
    skeleton_src, skeleton_src_name = repack._skeleton_text_for_repack(record, "regenerate")
    regenerated_skeleton = repack._uses_regenerated_skeleton(skeleton_src_name)
    output_decl = repack._function_decl(record, name_for_prompt)[1]
    target_by_old = {int(i): str(t or "") for i, t in zip(old_indices, targets)}
    slot_types_all = list(record.get("slot_types") or [])
    if regenerated_skeleton:
        slot_types_all = [f"FILL_{m.group(1)}" for m in repack.FILL_RE.finditer(skeleton_src)]

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
        chunk = repack._strip_output_alignment_comments(chunk)
        if not replaced_decl:
            chunk, n_sub = re.subn(
                r"^\s*[^()\n{};]+?\s+func\s*\([^)]*\)",
                output_decl,
                chunk,
                count=1,
            )
            if n_sub:
                replaced_decl = True
        if chunk:
            text_parts.append(chunk)

    for m in repack.FILL_RE.finditer(skeleton_src):
        add_text(skeleton_src[pos:m.start()])
        old_idx = repack._slot_index_from_match(m, occurrence, len(slot_types_all))
        if regenerated_skeleton:
            old_idx = repack._slot_index_from_placeholder(m, occurrence)
        occurrence += 1
        stype = (
            f"FILL_{m.group(1)}"
            if regenerated_skeleton
            else (str(slot_types_all[old_idx]) if old_idx < len(slot_types_all) else f"FILL_{m.group(1)}")
        )
        if stype == "FILL_TYPE":
            add_text(repack._signature(record)[0])
        elif stype == "FILL_PARAM":
            add_text(repack._signature(record)[1])
        elif old_idx in target_by_old:
            target = target_by_old[old_idx]
            text_parts.append(target)
            if target.rstrip().endswith(";"):
                skip_leading_semicolon = True
        pos = m.end()
    add_text(skeleton_src[pos:])
    return "".join(text_parts)


def _rendered_skip_payload(record: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[List[int]], List[int], List[str], str]:
    skeleton, slot_types, targets, bbs, old_indices, target_sources, _dropped, err = repack._render_prompt_skeleton(
        record,
        name_for_prompt="func",
        allow_v9_teacher_fallback=False,
        skeleton_source="regenerate",
    )
    output = ""
    if not err:
        output = _render_output_without_tokenizer(record, targets, old_indices, name_for_prompt="func")
    return skeleton or "", slot_types, targets, bbs, old_indices, target_sources, err or output


def _write_report(
    rows: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    out_path: Path,
    max_examples_per_reason: int,
    kept: Optional[List[Dict[str, Any]]] = None,
    max_kept_examples: int = 12,
) -> None:
    by_line = {i + 1: row for i, row in enumerate(rows)}
    reason_counts = Counter(str(x.get("reason")) for x in skipped)
    shown = defaultdict(int)

    with out_path.open("w", encoding="utf-8") as wf:
        if kept is not None:
            wf.write("===== KEPT QUALITY BUCKETS =====\n")
            bucket_counts = Counter(str(r.get("sft_v10_quality_bucket") or ("warning" if r.get("sft_v10_quality_warnings") else "clean")) for r in kept)
            for bucket, cnt in bucket_counts.most_common():
                wf.write(f"{cnt:4d}  {bucket}\n")

            wf.write("\n===== KEPT WARNING COUNTS =====\n")
            warning_counts = Counter()
            for row in kept:
                for warning in row.get("sft_v10_quality_warnings") or []:
                    warning_counts[str(warning)] += 1
            for warning, cnt in warning_counts.most_common():
                wf.write(f"{cnt:4d}  {warning}\n")

            wf.write("\n===== SUSPICIOUS KEPT EXAMPLES =====\n")
            shown_kept = 0
            for i, row in enumerate(kept):
                bucket = str(row.get("sft_v10_quality_bucket") or ("warning" if row.get("sft_v10_quality_warnings") else "clean"))
                warnings = [str(x) for x in (row.get("sft_v10_quality_warnings") or [])]
                if bucket == "clean" and not warnings:
                    continue
                if shown_kept >= max_kept_examples:
                    break
                shown_kept += 1
                wf.write("\n" + "=" * 100 + "\n")
                wf.write(f"kept_index = {i} func = {row.get('func_name')} bucket = {bucket}\n")
                wf.write(f"warnings = {warnings}\n")
                wf.write(f"slots = {len(row.get('slot_types') or [])}\n")
                wf.write("\nPROMPT SKELETON:\n")
                wf.write(str(row.get("sft_skeleton_text_pretty") or "") + "\n")
                wf.write("\nSLOT TARGETS:\n")
                for si, (stype, bbs, src, tgt) in enumerate(zip(
                    row.get("slot_types") or [],
                    row.get("sft_slot_bb_indices") or [],
                    row.get("sft_teacher_target_sources") or [],
                    row.get("sft_teacher_targets") or [],
                )):
                    wf.write("-" * 80 + "\n")
                    wf.write(f"slot_{si} type={stype} bbs={bbs} source={src}\n")
                    wf.write(str(tgt) + "\n")
                wf.write("\nRENDERED OUTPUT:\n")
                wf.write(str(row.get("sft_output_text_pretty") or "") + "\n")
                real = _real_source_function_context(row)
                if real:
                    wf.write("\nREAL SOURCE FUNCTION:\n")
                    wf.write(real + "\n")
            wf.write("\n")

        wf.write("===== SKIPPED REASON COUNTS =====\n")
        for reason, cnt in reason_counts.most_common():
            wf.write(f"{cnt:4d}  {reason}\n")

        wf.write("\n===== DETAILED MISSING-SLOT EXAMPLES =====\n")
        for item in skipped:
            reason = str(item.get("reason") or "")
            if shown[reason] >= max_examples_per_reason:
                continue
            m = MISS_RE.search(reason)
            if not m:
                continue
            stype = m.group(1)
            old_idx = int(m.group(2))
            rec = by_line.get(int(item.get("line") or 0))
            if not rec:
                continue
            shown[reason] += 1

            bbs = _slot_bbs(rec, old_idx)
            target_all = list(rec.get("sft_teacher_targets") or [])
            old_target = target_all[old_idx] if old_idx < len(target_all) else ""
            status, cands = _candidate_rows(rec, bbs)
            current_target, current_reason = _current_source_target(rec, stype, bbs, str(old_target or ""))
            fallback_targets = _fallback_source_targets(rec, stype, bbs)

            wf.write("\n" + "=" * 100 + "\n")
            wf.write(f"reason = {reason}\n")
            wf.write(f"jsonl_line = {item.get('line')} func = {item.get('func_name')}\n")
            wf.write(f"source_file = {rec.get('source_file')}\n")
            wf.write(f"slot = slot_{old_idx} type = {stype} bbs = {bbs}\n")
            wf.write(f"v9_teacher_target = {old_target!r}\n")
            wf.write(f"candidate_status = {status or 'ok'}\n")
            if current_target:
                wf.write(f"current_code_source_target = {current_target!r}\n")
                wf.write(f"current_code_source_reason = {current_reason}\n")
            else:
                wf.write(f"current_code_source_target = ''\n")
                wf.write(f"current_code_source_reason = {current_reason}\n")

            bb_lines = rec.get("block_src_lines") or []
            for bb in bbs:
                if isinstance(bb, int) and 0 <= bb < len(bb_lines):
                    wf.write(f"block_src_lines[BB_{bb}] = {bb_lines[bb]}\n")

            if fallback_targets:
                wf.write("\nDIRECT SOURCE FALLBACKS:\n")
                for src, tgt in fallback_targets:
                    wf.write(f"{src}: {tgt!r}\n")

            if not cands:
                continue
            wf.write("\nTOP ALIGNMENT CANDIDATES:\n")
            for j, mass, row, text, usable in cands:
                wf.write("-" * 80 + "\n")
                wf.write(
                    f"col={j} mass={mass:.4f} type={row.get('type')} "
                    f"lines={row.get('start_line')}-{row.get('end_line')} "
                    f"depth={row.get('depth')} usable_as_stmt={usable}\n"
                )
                wf.write(f"pretty = {text!r}\n")
                ctx = _source_context(rec.get("source_file"), row)
                if ctx:
                    wf.write("source_context:\n")
                    wf.write(ctx + "\n")

        wf.write("\n===== DETAILED RENDERED-SKIP EXAMPLES =====\n")
        rendered_shown = defaultdict(int)
        for item in skipped:
            reason = str(item.get("reason") or "")
            if MISS_RE.search(reason):
                continue
            if rendered_shown[reason] >= max_examples_per_reason:
                continue
            rec = by_line.get(int(item.get("line") or 0))
            if not rec:
                continue
            rendered_shown[reason] += 1
            skeleton, slot_types, targets, bbs, old_indices, target_sources, payload = _rendered_skip_payload(rec)

            wf.write("\n" + "=" * 100 + "\n")
            wf.write(f"reason = {reason}\n")
            wf.write(f"jsonl_line = {item.get('line')} func = {item.get('func_name')}\n")
            wf.write(f"source_file = {rec.get('source_file')}\n")
            wf.write("\nPROMPT SKELETON:\n")
            wf.write((skeleton or "<render_prompt_skeleton_failed>") + "\n")
            wf.write("\nSLOT TARGETS:\n")
            if not slot_types:
                wf.write(f"render_error = {payload}\n")
            for i, (stype, bb, old_idx, src, tgt) in enumerate(zip(slot_types, bbs, old_indices, target_sources, targets)):
                wf.write("-" * 80 + "\n")
                wf.write(f"new_slot_{i} old_slot_{old_idx} type={stype} bbs={bb} source={src}\n")
                wf.write(str(tgt) + "\n")
                bb_lines = rec.get("block_src_lines") or []
                for one_bb in bb:
                    if isinstance(one_bb, int) and 0 <= one_bb < len(bb_lines):
                        wf.write(f"block_src_lines[BB_{one_bb}] = {bb_lines[one_bb]}\n")
            wf.write("\nRENDERED OUTPUT:\n")
            wf.write(payload + "\n")
            real = _real_source_function_context(rec)
            if real:
                wf.write("\nREAL SOURCE FUNCTION:\n")
                wf.write(real + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v9-jsonl", "--input", dest="v9_jsonl", type=Path, required=True)
    ap.add_argument("--skipped-jsonl", "--skipped", dest="skipped_jsonl", type=Path, required=True)
    ap.add_argument("--kept-jsonl", "--kept", dest="kept_jsonl", type=Path, default=None)
    ap.add_argument("--out", "--output", dest="out", type=Path, required=True)
    ap.add_argument(
        "--max-examples-per-reason",
        "--limit",
        dest="max_examples_per_reason",
        type=int,
        default=3,
    )
    ap.add_argument("--max-kept-examples", type=int, default=12)
    args = ap.parse_args()

    rows = _load_jsonl(args.v9_jsonl.expanduser().resolve())
    skipped = _load_jsonl(args.skipped_jsonl.expanduser().resolve())
    kept = _load_jsonl(args.kept_jsonl.expanduser().resolve()) if args.kept_jsonl is not None else None
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        rows,
        skipped,
        out_path,
        int(args.max_examples_per_reason),
        kept=kept,
        max_kept_examples=int(args.max_kept_examples),
    )
    print(f"saved_to = {out_path}")


if __name__ == "__main__":
    main()

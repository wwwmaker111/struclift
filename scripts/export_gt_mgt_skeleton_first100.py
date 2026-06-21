#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit training-time §6.5.1 skeletons built from GT labels + M_gt.

This answers a narrow question:

    If training data already has ``pattern_labels`` and ``alignment_matrix``,
    how good is the skeleton obtained by the current §6.5.1 implementation?

The builder is still source-free at construction time: it receives only
``pattern_labels``, binary CFG, and the alignment matrix as the document says.
Source is used only after construction as an audit reference.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import augment_binskel_sft as sft_aug  # noqa: E402
import export_modulea_region_control_skeleton_100 as region_export  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402
from struclift.models.module_b import (  # noqa: E402
    _bb_source_order_keys_from_alignment,
    _skeleton_hints_from_alignment,
)
from struclift.utils.skeleton import build_skeleton_for_binary_cfg  # noqa: E402


PATTERN_NAMES = region_export.PATTERN_NAMES


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def _n_bb(row: Dict[str, Any]) -> int:
    return effect._n_bb(row)


def _labels(row: Dict[str, Any]) -> List[int]:
    raw = row.get("pattern_labels")
    return [int(x) for x in raw] if isinstance(raw, list) else []


def _edge_index(row: Dict[str, Any]) -> torch.Tensor:
    raw = row.get("bin_edge_index") or [[], []]
    if not isinstance(raw, list) or len(raw) != 2:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(
        [[int(x) for x in raw[0]], [int(x) for x in raw[1]]],
        dtype=torch.long,
    )


def _alignment_matrix(row: Dict[str, Any], n_bb: int) -> Optional[torch.Tensor]:
    raw = row.get("alignment_matrix")
    if not isinstance(raw, list) or len(raw) != n_bb:
        return None
    try:
        if not raw or not isinstance(raw[0], list) or len(raw[0]) == 0:
            return None
        width = len(raw[0])
        if any(not isinstance(r, list) or len(r) != width for r in raw):
            return None
        return torch.tensor(raw, dtype=torch.float32)
    except Exception:
        return None


def _ast_types(row: Dict[str, Any], n_s: int) -> torch.Tensor:
    raw = row.get("src_ast_types")
    if isinstance(raw, list) and len(raw) >= n_s:
        vals = [int(x) for x in raw[:n_s]]
    else:
        vals = [0 for _ in range(n_s)]
    return torch.tensor(vals, dtype=torch.long)


def _source_function_with_numbers(row: Dict[str, Any]) -> Tuple[str, Optional[Tuple[int, int]]]:
    lt = repack._line_to_text(row.get("source_file"))
    if not lt:
        return "<source file unreadable>", None
    bounds = repack._function_body_line_range(row, lt)
    if bounds is None:
        return "<function bounds unavailable>", None
    lo, hi = bounds
    lines: List[str] = []
    for ln in range(lo, hi + 1):
        if ln in lt:
            lines.append(f"{ln:5d}: {lt[ln]}")
    return "\n".join(lines) if lines else "<empty source range>", bounds


def _source_meta(row: Dict[str, Any], source_control_filter: str) -> Dict[str, Any]:
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
    return {
        "source_parse_status": effect._source_parse_status(row, rows),
        "source_outline_source": outline_source,
        "source_statement_rows": len(rows),
        "active_source_lines": len(active),
        "source_control_count": len(seq),
        "source_counts": dict(effect._counts(seq)),
        "source_outline": effect._source_outline_lines(outline, limit=200),
    }


def _outline_pairs(lines: Sequence[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for line in lines:
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        typ = line.strip().split()[0]
        if typ in {"if", "loop", "switch"}:
            out.append((typ, leading // 2))
    return out


def _skeleton_meta(text: str, n_bb: int) -> Dict[str, Any]:
    seq = effect._control_outline_from_text(text)
    counts = effect._counts(seq)
    outline = effect._outline_lines(seq, limit=200)
    return {
        "pred_control_count": len(seq),
        "pred_counts": dict(counts),
        "pred_outline_lines": outline,
        "pred_outline": _outline_pairs(outline),
        "bb_stats": effect._bb_marker_stats(text, n_bb),
    }


def _label_summary(labels: Sequence[int]) -> str:
    c = Counter(int(x) for x in labels)
    return ", ".join(
        f"{PATTERN_NAMES.get(k, 'unknown_' + str(k))}={c[k]}"
        for k in sorted(c)
    )


def _label_sequence(labels: Sequence[int], max_items: int = 240) -> str:
    out: List[str] = []
    for i, lab in enumerate(labels[:max_items]):
        out.append(f"BB_{i}:{PATTERN_NAMES.get(int(lab), 'unknown_' + str(lab))}")
    if len(labels) > max_items:
        out.append("...")
    return "\n".join(out)


def _build_gt_mgt_skeleton(row: Dict[str, Any]) -> Tuple[str, Any, Dict[str, Any]]:
    labels = _labels(row)
    n_bb = len(labels)
    M = _alignment_matrix(row, n_bb)
    if M is None:
        raise RuntimeError("missing_or_invalid_alignment_matrix")
    pl = torch.tensor(labels, dtype=torch.long)
    ast_types = _ast_types(row, int(M.size(1)))
    prefix_hints, skip_bbs = _skeleton_hints_from_alignment(M, pl, ast_types)
    hints = sft_aug._bb_opcode_hints(row)
    skel = build_skeleton_for_binary_cfg(
        pl,
        _edge_index(row),
        n_bb,
        block_instructions_text=hints,
        bb_order_key=_bb_source_order_keys_from_alignment(M),
        alignment_matrix=M,
        header_prefix_slots=prefix_hints,
        skip_statement_bbs=skip_bbs,
    )
    text = str(getattr(skel, "skeleton_text", "") or "")
    if hasattr(sft_aug, "_format_skeleton_text_for_prompt"):
        text = sft_aug._format_skeleton_text_for_prompt(text)
    aux = {
        "build_stats": getattr(skel, "build_stats", None),
        "reliability": getattr(skel, "reliability", None),
        "n_slots": len(getattr(skel, "slots", []) or []),
        "skip_bbs": sorted(int(x) for x in skip_bbs),
        "prefix_hints": dict(prefix_hints),
        "m_shape": list(M.shape),
        "m_nonzero_rows": int((M.sum(dim=1) > 1e-12).sum().item()),
    }
    return text, skel, aux


def _is_valid_training_row(row: Dict[str, Any]) -> bool:
    if not region_export.is_valid_bin(row):
        return False
    labels = _labels(row)
    if not labels or len(labels) != _n_bb(row):
        return False
    return _alignment_matrix(row, len(labels)) is not None


def _pick_candidates(
    jsonl: Path,
    num_examples: int,
    limit: int,
    source_control_filter: str,
    min_source_controls: int,
) -> Tuple[List[Tuple[int, Dict[str, Any], str, Tuple[int, int], Dict[str, Any]]], int, Counter]:
    candidates: List[Tuple[int, Dict[str, Any], str, Tuple[int, int], Dict[str, Any]]] = []
    seen = set()
    skipped: Counter[str] = Counter()
    scanned = 0
    for line_no, row in _iter_jsonl(jsonl, limit):
        scanned = line_no
        if not _is_valid_training_row(row):
            skipped["invalid_bin_labels_or_mgt"] += 1
            continue
        source_text, bounds = _source_function_with_numbers(row)
        if bounds is None:
            skipped["source_bounds_unavailable"] += 1
            continue
        key = (str(row.get("source_file") or ""), str(row.get("func_name") or ""), bounds)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        smeta = _source_meta(row, source_control_filter)
        if int(smeta.get("source_control_count") or 0) < int(min_source_controls):
            skipped["no_or_below_min_source_control"] += 1
            continue
        seen.add(key)
        candidates.append((line_no, row, source_text, bounds, smeta))
        if len(candidates) >= int(num_examples):
            break
    return candidates, scanned, skipped


def _fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument(
        "--source-control-filter",
        choices=["raw", "bb-start-line", "bb-range"],
        default="bb-start-line",
    )
    ap.add_argument("--min-source-controls", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    jsonl = args.jsonl.expanduser().resolve()
    out = args.out.expanduser().resolve()
    candidates, scanned, skipped = _pick_candidates(
        jsonl,
        int(args.num_examples),
        int(args.limit),
        str(args.source_control_filter),
        int(args.min_source_controls),
    )

    rows: List[Dict[str, Any]] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== TRAINING-STAGE GT-PATTERN + M_GT SKELETON AUDIT =====\n")
        wf.write(f"input = {jsonl}\n")
        wf.write("skeleton_source = jsonl pattern_labels + jsonl alignment_matrix(M_gt) + CFG via build_skeleton_for_binary_cfg\n")
        wf.write("source_usage = source is used only for audit reference, not for skeleton construction\n")
        wf.write(f"scanned = {scanned}\n")
        wf.write(f"exported = {len(candidates)}\n")
        wf.write(f"requested = {int(args.num_examples)}\n")
        wf.write(f"source_control_filter = {args.source_control_filter}\n")
        wf.write(f"skipped = {dict(skipped)}\n")

        for idx, (line_no, row, src_text, bounds, smeta) in enumerate(candidates):
            if idx % 10 == 0:
                print(f"writing example {idx}/{len(candidates)}", flush=True)
            labels = _labels(row)
            src_counts = Counter(smeta.get("source_counts") or {})
            src_outline = _outline_pairs(smeta.get("source_outline") or [])
            try:
                skeleton_text, skel, aux = _build_gt_mgt_skeleton(row)
                pmeta = _skeleton_meta(skeleton_text, len(labels))
                pred_counts = Counter(pmeta.get("pred_counts") or {})
                pred_outline = list(pmeta.get("pred_outline") or [])
                gaps = (
                    int(pred_counts.get("if", 0)) - int(src_counts.get("if", 0)),
                    int(pred_counts.get("loop", 0)) - int(src_counts.get("loop", 0)),
                    int(pred_counts.get("switch", 0)) - int(src_counts.get("switch", 0)),
                )
                kind_ok = [x[0] for x in pred_outline] == [x[0] for x in src_outline]
                outline_ok = pred_outline == src_outline
                gap0 = gaps == (0, 0, 0)
                err = ""
            except Exception as e:
                skeleton_text = f"<ERROR: {type(e).__name__}: {e}>"
                pmeta = {"pred_control_count": 0, "pred_counts": {}, "pred_outline_lines": [], "bb_stats": {}}
                aux = {}
                gaps = (
                    -int(src_counts.get("if", 0)),
                    -int(src_counts.get("loop", 0)),
                    -int(src_counts.get("switch", 0)),
                )
                kind_ok = False
                outline_ok = False
                gap0 = False
                err = str(e)

            rows.append(
                {
                    "example": idx,
                    "func": row.get("func_name"),
                    "line_no": line_no,
                    "gap0": gap0,
                    "kind_ok": kind_ok,
                    "outline_ok": outline_ok,
                    "gaps": gaps,
                    "source_ref": smeta.get("source_outline_source"),
                    "error": err,
                }
            )

            wf.write("\n" + "=" * 120 + "\n")
            wf.write(f"example = {idx}  input_line = {line_no}  func = {row.get('func_name')}\n")
            wf.write(f"source_file = {row.get('source_file')}\n")
            wf.write(f"source_bounds = {bounds[0]}-{bounds[1]}\n")
            wf.write("skeleton_source = GT_pattern_labels_plus_M_gt_current_651_builder\n")
            wf.write(f"bb = {len(labels)}\n")
            wf.write(
                f"source_parse = {smeta.get('source_parse_status')}  "
                f"source_ref = {smeta.get('source_outline_source')}  "
                f"src_rows = {smeta.get('source_statement_rows')}  "
                f"active_source_lines = {smeta.get('active_source_lines')}\n"
            )
            wf.write(
                f"ctrl(pred/source) = {pmeta.get('pred_control_count')}/{smeta.get('source_control_count')}  "
                f"gaps(if,loop,switch)=({gaps[0]},{gaps[1]},{gaps[2]})  "
                f"gap0={int(gap0)} kindseq={int(kind_ok)} outline={int(outline_ok)}\n"
            )
            wf.write(f"gt_10class_label_counts = {_label_summary(labels)}\n")
            wf.write(f"mgt_builder_aux = {aux}\n")
            wf.write(f"bb_stats = {pmeta.get('bb_stats')}\n")

            wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
            wf.write("\n".join(smeta.get("source_outline") or ["<none>"]) + "\n")

            wf.write("\n-- GT+MGT SKELETON CONTROL OUTLINE --\n")
            wf.write("\n".join(pmeta.get("pred_outline_lines") or ["<none>"]) + "\n")

            wf.write("\n-- GT 10CLASS BB LABEL SEQUENCE --\n")
            wf.write(_label_sequence(labels) + "\n")

            wf.write("\n-- REAL SOURCE FUNCTION --\n")
            wf.write(src_text.rstrip() + "\n")

            wf.write("\n-- GT+MGT GENERATED SKELETON --\n")
            wf.write(skeleton_text.rstrip() + "\n")

        n = len(rows)
        gap0_n = sum(1 for r in rows if r["gap0"])
        kind_n = sum(1 for r in rows if r["kind_ok"])
        outline_n = sum(1 for r in rows if r["outline_ok"])
        reliable_rows = [r for r in rows if r.get("source_ref") == "statement_rows"]
        rel_n = len(reliable_rows)
        rel_outline = sum(1 for r in reliable_rows if r["outline_ok"])

        wf.write("\n" + "=" * 120 + "\n")
        wf.write("===== SUMMARY =====\n")
        wf.write(
            f"all: n={n} gap0={_fmt_pct(gap0_n, n)} "
            f"kindseq={_fmt_pct(kind_n, n)} outline={_fmt_pct(outline_n, n)}\n"
        )
        wf.write(f"statement_rows_only: n={rel_n} outline={_fmt_pct(rel_outline, rel_n)}\n")
        wf.write(f"gap_dist = {Counter(r['gaps'] for r in rows).most_common(30)}\n")
        wf.write(
            "bad_outline_first30 = "
            f"{[(r['example'], r['func'], r['gaps']) for r in rows if not r['outline_ok']][:30]}\n"
        )

    print("saved_to =", out)
    print(
        f"all: n={len(rows)} gap0={_fmt_pct(sum(1 for r in rows if r['gap0']), len(rows))} "
        f"kindseq={_fmt_pct(sum(1 for r in rows if r['kind_ok']), len(rows))} "
        f"outline={_fmt_pct(sum(1 for r in rows if r['outline_ok']), len(rows))}",
        flush=True,
    )
    print("elapsed_total_sec =", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()

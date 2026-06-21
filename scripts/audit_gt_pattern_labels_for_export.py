#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit GT pattern labels for examples already exported to txt.

The region skeleton exporter writes the jsonl ``input_line`` for each example,
but it does not currently dump the original ``pattern_labels``.  This helper
re-opens the source jsonl, fetches those rows, and compares:

* AB predicted 10-class labels from the exported txt vs jsonl GT labels;
* skeleton built from GT labels vs the source control outline;
* skeleton built from AB predicted labels vs the source control outline.

Source is used only as an audit reference here.  The builder itself remains
the same source-free CFG + pattern-label builder used by the exporter.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_modulea_region_control_skeleton_100 as region_export  # noqa: E402


KIND_TO_LABELS = {
    "if": {1},
    "loop": {4},
    "switch": {7},
}


def parse_exported_examples(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    examples: List[Dict[str, Any]] = []
    for chunk in re.split(r"\n={20,}\n", text):
        if not chunk.lstrip().startswith("example ="):
            continue
        m = re.search(r"example = (\d+)\s+input_line = (\d+)\s+func = ([^\n]+)", chunk)
        if not m:
            continue
        ex = int(m.group(1))
        input_line = int(m.group(2))
        func = m.group(3).strip()
        pred = parse_label_section(chunk, "-- AB PREDICTED 10CLASS BB LABEL SEQUENCE --")
        refined = parse_label_section(chunk, "-- REFINED 10CLASS BB LABEL SEQUENCE --")
        examples.append(
            {
                "example": ex,
                "input_line": input_line,
                "func": func,
                "pred_labels": pred,
                "refined_labels": refined,
            }
        )
    return examples


def parse_label_section(chunk: str, marker: str) -> List[int]:
    labels: List[int] = []
    in_sec = False
    rev = {v: k for k, v in region_export.PATTERN_NAMES.items()}
    for line in chunk.splitlines():
        if line.strip() == marker:
            in_sec = True
            continue
        if in_sec:
            if line.startswith("-- ") or line.startswith("=") or line.startswith("example ="):
                break
            m = re.match(r"BB_(\d+):([A-Za-z0-9_]+)", line.strip())
            if not m:
                continue
            labels.append(int(rev.get(m.group(2), -1)))
    return labels


def load_jsonl_rows(path: Path, line_numbers: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    wanted = set(int(x) for x in line_numbers)
    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no not in wanted:
                continue
            out[line_no] = json.loads(line)
            if len(out) >= len(wanted):
                break
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"jsonl missing requested input_line values: {missing[:20]}")
    return out


def outline_pairs(lines: Sequence[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for line in lines:
        if not str(line).strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        depth = leading // 2
        typ = line.strip().split()[0]
        if typ in {"if", "loop", "switch"}:
            out.append((typ, depth))
    return out


def source_meta(row: Dict[str, Any], source_control_filter: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    src_text, bounds = region_export.source_function_with_numbers(row)
    if bounds is None:
        raise RuntimeError("source bounds unavailable")
    smeta = region_export.source_control_meta(row, source_control_filter)
    srel = region_export.source_reliability_meta(row, src_text, bounds, smeta, source_control_filter)
    return smeta, srel


def skeleton_meta_from_labels(row: Dict[str, Any], labels: Sequence[int]) -> Dict[str, Any]:
    (
        skeleton_text,
        _slots,
        region_stats,
        build_stats,
        refined_labels,
        refine_stats,
    ) = region_export.region_skeleton_from_pred_labels(row, labels)
    pmeta = region_export.control_meta_from_skeleton(skeleton_text, len(labels))
    return {
        "text": skeleton_text,
        "pred_counts": Counter(pmeta.get("pred_counts") or {}),
        "pred_outline": outline_pairs(pmeta.get("pred_outline") or []),
        "region_stats": region_stats,
        "build_stats": build_stats,
        "refined_labels": [int(x) for x in refined_labels],
        "refine_stats": refine_stats,
    }


def control_label_counts(labels: Sequence[int]) -> Counter:
    c: Counter[str] = Counter()
    for x in labels:
        lab = int(x)
        if lab == 1:
            c["if"] += 1
        elif lab == 4:
            c["loop"] += 1
        elif lab == 7:
            c["switch"] += 1
    return c


def label_match_stats(pred: Sequence[int], gt: Sequence[int]) -> Dict[str, Any]:
    n = min(len(pred), len(gt))
    correct = sum(1 for i in range(n) if int(pred[i]) == int(gt[i]))
    pred_control = {i for i, x in enumerate(pred[:n]) if int(x) in {1, 4, 7}}
    gt_control = {i for i, x in enumerate(gt[:n]) if int(x) in {1, 4, 7}}
    ctrl_tp = len(pred_control & gt_control)
    ctrl_fp = len(pred_control - gt_control)
    ctrl_fn = len(gt_control - pred_control)
    per_class = {}
    for lab, name in region_export.PATTERN_NAMES.items():
        idx = [i for i in range(n) if int(gt[i]) == int(lab)]
        if idx:
            per_class[name] = {
                "gt": len(idx),
                "correct": sum(1 for i in idx if int(pred[i]) == int(lab)),
            }
    return {
        "n": n,
        "len_pred": len(pred),
        "len_gt": len(gt),
        "correct": correct,
        "acc": correct / n if n else 0.0,
        "ctrl_tp": ctrl_tp,
        "ctrl_fp": ctrl_fp,
        "ctrl_fn": ctrl_fn,
        "ctrl_precision": ctrl_tp / (ctrl_tp + ctrl_fp) if ctrl_tp + ctrl_fp else 0.0,
        "ctrl_recall": ctrl_tp / (ctrl_tp + ctrl_fn) if ctrl_tp + ctrl_fn else 0.0,
        "per_class": per_class,
    }


def fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(n / d):.1%})" if d else "0/0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-txt", type=Path, action="append", required=True)
    ap.add_argument("--jsonl", type=Path, action="append", required=True)
    ap.add_argument("--name", action="append")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-control-filter", default="bb-start-line", choices=["raw", "bb-start-line", "bb-range"])
    args = ap.parse_args()

    if len(args.export_txt) != len(args.jsonl):
        raise SystemExit("--export-txt and --jsonl counts must match")
    names = args.name or []
    while len(names) < len(args.export_txt):
        names.append(args.export_txt[len(names)].stem)

    all_rows: List[Dict[str, Any]] = []
    per_split: Dict[str, List[Dict[str, Any]]] = {}

    for name, export_txt, jsonl in zip(names, args.export_txt, args.jsonl):
        examples = parse_exported_examples(export_txt)
        source_rows = load_jsonl_rows(jsonl, [x["input_line"] for x in examples])
        split_rows: List[Dict[str, Any]] = []
        for item in examples:
            row = source_rows[item["input_line"]]
            gt_labels = [int(x) for x in row.get("pattern_labels") or []]
            pred_labels = [int(x) for x in item.get("pred_labels") or []]
            smeta, srel = source_meta(row, args.source_control_filter)
            src_counts = Counter(smeta.get("source_counts") or {})
            src_outline = outline_pairs(smeta.get("source_outline") or [])

            gt_skel = skeleton_meta_from_labels(row, gt_labels)
            pred_skel = skeleton_meta_from_labels(row, pred_labels)

            gt_label_counts = control_label_counts(gt_labels)
            pred_label_counts = control_label_counts(pred_labels)
            gt_label_gaps = (
                int(gt_label_counts.get("if", 0)) - int(src_counts.get("if", 0)),
                int(gt_label_counts.get("loop", 0)) - int(src_counts.get("loop", 0)),
                int(gt_label_counts.get("switch", 0)) - int(src_counts.get("switch", 0)),
            )
            pred_label_gaps = (
                int(pred_label_counts.get("if", 0)) - int(src_counts.get("if", 0)),
                int(pred_label_counts.get("loop", 0)) - int(src_counts.get("loop", 0)),
                int(pred_label_counts.get("switch", 0)) - int(src_counts.get("switch", 0)),
            )
            gt_gaps = (
                int(gt_skel["pred_counts"].get("if", 0)) - int(src_counts.get("if", 0)),
                int(gt_skel["pred_counts"].get("loop", 0)) - int(src_counts.get("loop", 0)),
                int(gt_skel["pred_counts"].get("switch", 0)) - int(src_counts.get("switch", 0)),
            )
            pred_gaps = (
                int(pred_skel["pred_counts"].get("if", 0)) - int(src_counts.get("if", 0)),
                int(pred_skel["pred_counts"].get("loop", 0)) - int(src_counts.get("loop", 0)),
                int(pred_skel["pred_counts"].get("switch", 0)) - int(src_counts.get("switch", 0)),
            )
            rec = {
                "split": name,
                "example": item["example"],
                "input_line": item["input_line"],
                "func": item["func"],
                "source_reliability": srel.get("source_reliability"),
                "source_counts": dict(src_counts),
                "source_outline": src_outline,
                "gt_label_counts": dict(gt_label_counts),
                "pred_label_counts": dict(pred_label_counts),
                "gt_label_gaps": gt_label_gaps,
                "pred_label_gaps": pred_label_gaps,
                "gt_gaps": gt_gaps,
                "pred_gaps": pred_gaps,
                "gt_kindseq_equal": [x[0] for x in gt_skel["pred_outline"]] == [x[0] for x in src_outline],
                "pred_kindseq_equal": [x[0] for x in pred_skel["pred_outline"]] == [x[0] for x in src_outline],
                "gt_outline_equal": gt_skel["pred_outline"] == src_outline,
                "pred_outline_equal": pred_skel["pred_outline"] == src_outline,
                "label_match": label_match_stats(pred_labels, gt_labels),
                "gt_refine_stats": gt_skel["refine_stats"],
                "pred_refine_stats": pred_skel["refine_stats"],
            }
            split_rows.append(rec)
            all_rows.append(rec)
        per_split[name] = split_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as wf:
        for name, rows in per_split.items():
            write_summary(wf, name, rows)
        write_summary(wf, "ALL_200", all_rows)
        wf.write("\n===== suspicious examples: GT-built skeleton not count-correct but AB pred is also wrong =====\n")
        for r in all_rows:
            if r["source_reliability"] == "unreliable":
                continue
            if r["gt_label_gaps"] != (0, 0, 0) or r["gt_gaps"] != (0, 0, 0):
                wf.write(
                    f"{r['split']} #{r['example']:02d} line={r['input_line']} func={r['func']} "
                    f"rel={r['source_reliability']} "
                    f"gt_label_gaps={r['gt_label_gaps']} gt_builder_gaps={r['gt_gaps']} "
                    f"pred_label_gaps={r['pred_label_gaps']} pred_builder_gaps={r['pred_gaps']} "
                    f"gt_counts={r['gt_label_counts']} src_counts={r['source_counts']}\n"
                )
        wf.write("\n===== per-example compact table =====\n")
        for r in all_rows:
            lm = r["label_match"]
            wf.write(
                f"{r['split']:5s} #{r['example']:02d} {r['func'][:36]:36s} rel={r['source_reliability']:10s} "
                f"bb_acc={lm['acc']:.3f} ctrlP/R={lm['ctrl_precision']:.3f}/{lm['ctrl_recall']:.3f} "
                f"gt_label_gap={r['gt_label_gaps']} gt_builder_gap={r['gt_gaps']} "
                f"pred_label_gap={r['pred_label_gaps']} pred_builder_gap={r['pred_gaps']} "
                f"gt_kind={int(r['gt_kindseq_equal'])} gt_outline={int(r['gt_outline_equal'])}\n"
            )
    print("saved_to =", args.out)


def write_summary(wf, name: str, rows: Sequence[Dict[str, Any]]) -> None:
    wf.write(f"\n===== {name} =====\n")
    n = len(rows)
    wf.write(f"n = {n}\n")
    wf.write(f"source_reliability = {dict(Counter(r['source_reliability'] for r in rows))}\n")
    for title, subset in [
        ("all", list(rows)),
        ("main_no_unreliable", [r for r in rows if r["source_reliability"] != "unreliable"]),
        ("reliable_only", [r for r in rows if r["source_reliability"] == "reliable"]),
    ]:
        if not subset:
            continue
        d = len(subset)
        gt_label_gap0 = sum(r["gt_label_gaps"] == (0, 0, 0) for r in subset)
        pred_label_gap0 = sum(r["pred_label_gaps"] == (0, 0, 0) for r in subset)
        gt_gap0 = sum(r["gt_gaps"] == (0, 0, 0) for r in subset)
        gt_kind = sum(r["gt_kindseq_equal"] for r in subset)
        gt_outline = sum(r["gt_outline_equal"] for r in subset)
        pred_gap0 = sum(r["pred_gaps"] == (0, 0, 0) for r in subset)
        pred_kind = sum(r["pred_kindseq_equal"] for r in subset)
        pred_outline = sum(r["pred_outline_equal"] for r in subset)
        total_bb = sum(r["label_match"]["n"] for r in subset)
        correct_bb = sum(r["label_match"]["correct"] for r in subset)
        ctrl_tp = sum(r["label_match"]["ctrl_tp"] for r in subset)
        ctrl_fp = sum(r["label_match"]["ctrl_fp"] for r in subset)
        ctrl_fn = sum(r["label_match"]["ctrl_fn"] for r in subset)
        wf.write(
            f"{title}: n={d} | "
            f"raw-GT-label count={fmt_pct(gt_label_gap0, d)} raw-AB-label count={fmt_pct(pred_label_gap0, d)} | "
            f"GT-builder gap0={fmt_pct(gt_gap0, d)} kindseq={fmt_pct(gt_kind, d)} outline={fmt_pct(gt_outline, d)} | "
            f"AB-pred-builder gap0={fmt_pct(pred_gap0, d)} kindseq={fmt_pct(pred_kind, d)} outline={fmt_pct(pred_outline, d)} | "
            f"pred-vs-GT BB acc={correct_bb}/{total_bb} ({correct_bb/total_bb:.1%}) "
            f"control P/R={ctrl_tp/(ctrl_tp+ctrl_fp) if ctrl_tp+ctrl_fp else 0.0:.1%}/"
            f"{ctrl_tp/(ctrl_tp+ctrl_fn) if ctrl_tp+ctrl_fn else 0.0:.1%}\n"
        )
    wf.write(f"raw GT label gap dist = {Counter(r['gt_label_gaps'] for r in rows).most_common(20)}\n")
    wf.write(f"raw AB label gap dist = {Counter(r['pred_label_gaps'] for r in rows).most_common(20)}\n")
    wf.write(f"GT-builder gap dist = {Counter(r['gt_gaps'] for r in rows).most_common(20)}\n")
    wf.write(f"AB-pred-builder gap dist = {Counter(r['pred_gaps'] for r in rows).most_common(20)}\n")
    buckets = defaultdict(list)
    for r in rows:
        if r["source_reliability"] == "unreliable":
            buckets["unreliable"].append(r)
        if r["gt_label_gaps"] != (0, 0, 0):
            buckets["raw_GT_label_count_bad"].append(r)
        if r["gt_gaps"] == (0, 0, 0) and not r["pred_gaps"] == (0, 0, 0):
            buckets["GT_count_ok_AB_count_bad"].append(r)
        if r["gt_gaps"] != (0, 0, 0):
            buckets["GT_count_bad"].append(r)
        if r["gt_gaps"] == (0, 0, 0) and not r["gt_outline_equal"]:
            buckets["GT_count_ok_but_tree_bad"].append(r)
    for key in ["raw_GT_label_count_bad", "GT_count_ok_AB_count_bad", "GT_count_bad", "GT_count_ok_but_tree_bad", "unreliable"]:
        xs = buckets[key]
        wf.write(
            f"{key} {len(xs)} = "
            f"{[(r['example'], r['func'], r['gt_label_gaps'], r['gt_gaps'], r['pred_gaps'], r['source_reliability']) for r in xs[:40]]}\n"
        )


if __name__ == "__main__":
    main()

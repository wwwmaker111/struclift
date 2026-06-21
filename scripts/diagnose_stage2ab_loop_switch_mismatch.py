#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose why Stage2AB skeletons over-generate loop/switch structures.

Input is the manual export TXT produced by
``export_stage2ab_control_source_skeleton_50.py`` plus the JSONL it references.
For each selected mismatching example, this script maps ``input_line`` back to
the original JSONL record and prints:

* source and AB skeleton control outlines,
* 10-class BB label counts,
* LOOP/SWITCH header BBs and their source-line alignments,
* CFG backedges and high-outdegree/indirect-jump switch-like BBs,
* loop_exit/switch_case_body coverage.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402


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

EDGE_NAMES = {
    0: "COND",
    1: "UNCOND",
    2: "BACK",
    3: "CALL",
    4: "RET",
}


def _read_jsonl_lines(path: Path, wanted: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    want = set(int(x) for x in wanted)
    out: Dict[int, Dict[str, Any]] = {}
    if not want:
        return out
    max_line = max(want)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > max_line and len(out) >= len(want):
                break
            if i not in want:
                continue
            line = line.strip()
            if line:
                out[i] = json.loads(line)
    return out


def _parse_export_examples(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n={20,}\n", txt)
    examples: List[Dict[str, Any]] = []
    for block in blocks:
        if not block.startswith("example = "):
            continue
        h = re.search(
            r"^example = (\d+)\s+input_line = (\d+)\s+func = (.*?)\s+verdict = (.*)$",
            block,
            re.M,
        )
        c = re.search(
            r"^ctrl\(pred/source\) = (\d+)/(\d+)\s+"
            r"gaps\(if,loop,switch\)=\(([-\d]+),([-\d]+),([-\d]+)\)\s+"
            r"sim\(type,depth\)=([\d.]+)/([\d.]+)",
            block,
            re.M,
        )
        r = re.search(r"^bb_cover = .*?reasons = (.*)$", block, re.M)
        if not h or not c:
            continue
        reasons: List[str] = []
        if r:
            try:
                reasons = [str(x) for x in ast.literal_eval(r.group(1))]
            except Exception:
                reasons = []
        examples.append(
            {
                "example": int(h.group(1)),
                "input_line": int(h.group(2)),
                "func": h.group(3).strip(),
                "verdict": h.group(4).strip(),
                "pred_ctrl": int(c.group(1)),
                "src_ctrl": int(c.group(2)),
                "if_gap": int(c.group(3)),
                "loop_gap": int(c.group(4)),
                "switch_gap": int(c.group(5)),
                "type_sim": float(c.group(6)),
                "depth_sim": float(c.group(7)),
                "reasons": reasons,
                "block": block,
            },
        )
    return examples


def _edge_list(record: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    raw = record.get("bin_edge_index")
    typ = record.get("bin_edge_type") or []
    out: List[Tuple[int, int, int]] = []
    if isinstance(raw, list) and len(raw) == 2:
        srcs, dsts = raw
        for k, (s, d) in enumerate(zip(srcs or [], dsts or [])):
            try:
                et = int(typ[k]) if k < len(typ) else -1
                out.append((int(s), int(d), et))
            except Exception:
                continue
    edges2 = record.get("edges")
    if not out and isinstance(edges2, list):
        for e in edges2:
            if isinstance(e, list) and len(e) >= 3:
                try:
                    out.append((int(e[0]), int(e[1]), int(e[2])))
                except Exception:
                    pass
    return out


def _block_lines(record: Dict[str, Any], bb: int) -> List[int]:
    raw = record.get("block_src_lines") or []
    if not isinstance(raw, list) or bb >= len(raw):
        return []
    vals = raw[bb]
    if not isinstance(vals, list):
        return []
    out: List[int] = []
    for x in vals:
        try:
            out.append(int(x))
        except Exception:
            pass
    return sorted(set(out))


def _line_snippets(record: Dict[str, Any], lines: Sequence[int], *, max_lines: int = 6) -> str:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt or not lines:
        return "-"
    parts = []
    for ln in list(lines)[:max_lines]:
        txt = str(lt.get(int(ln), "")).strip()
        if txt:
            parts.append(f"L{ln}:{txt}")
        else:
            parts.append(f"L{ln}")
    if len(lines) > max_lines:
        parts.append("...")
    return " | ".join(parts) if parts else "-"


def _bb_struct_features(record: Dict[str, Any], bb: int) -> str:
    feats = record.get("bin_struct_features")
    if not isinstance(feats, list) or bb >= len(feats) or not isinstance(feats[bb], list):
        return "-"
    vals = feats[bb]
    names = [
        "in", "out", "is_loop_header", "is_loop_exit",
        "dom_depth", "postdom_depth", "block_size", "entropy",
        "has_call", "has_indirect_jump", "dfs", "bfs",
    ]
    parts = []
    for i, name in enumerate(names):
        if i < len(vals):
            v = vals[i]
            if isinstance(v, float):
                parts.append(f"{name}={v:.3g}")
            else:
                parts.append(f"{name}={v}")
    return ", ".join(parts) if parts else "-"


def _header_rows(record: Dict[str, Any], labels: Sequence[int], label_id: int) -> List[str]:
    edges = _edge_list(record)
    succs: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    preds: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for s, d, et in edges:
        succs[s].append((d, et))
        preds[d].append((s, et))
    rows: List[str] = []
    for bb, lab in enumerate(labels):
        if int(lab) != int(label_id):
            continue
        src_lines = _block_lines(record, bb)
        succ = ", ".join(f"{d}:{EDGE_NAMES.get(et, et)}" for d, et in sorted(succs.get(bb, []))) or "-"
        pred = ", ".join(f"{s}:{EDGE_NAMES.get(et, et)}" for s, et in sorted(preds.get(bb, []))) or "-"
        back_in = [s for s, et in preds.get(bb, []) if et == 2]
        back_out = [d for d, et in succs.get(bb, []) if et == 2]
        rows.append(
            f"BB_{bb} label={PATTERN_NAMES.get(label_id)} "
            f"src_lines={src_lines or '-'} src_text={_line_snippets(record, src_lines)} "
            f"pred=[{pred}] succ=[{succ}] back_in={back_in or '-'} back_out={back_out or '-'} "
            f"struct=({_bb_struct_features(record, bb)})"
        )
    return rows


def _write_example(wf: Any, ex: Dict[str, Any], rec: Dict[str, Any]) -> None:
    labels = [int(x) for x in (rec.get("pattern_labels") or [])]
    counts = collections.Counter(labels)
    edges = _edge_list(rec)
    backedges = [(s, d, et) for s, d, et in edges if et == 2]
    outdeg: Dict[int, int] = collections.Counter(s for s, _d, _et in edges)
    switch_like = sorted(bb for bb, deg in outdeg.items() if deg >= 3)
    indirect_like = []
    feats = rec.get("bin_struct_features")
    if isinstance(feats, list):
        for i, row in enumerate(feats):
            if isinstance(row, list) and len(row) > 9:
                try:
                    if float(row[9]) > 0:
                        indirect_like.append(i)
                except Exception:
                    pass

    audit = effect._audit_row(
        int(ex["input_line"]),
        rec,
        strict_filled_terminals=False,
        source_control_filter="bb-start-line",
    )
    wf.write("\n" + "=" * 120 + "\n")
    wf.write(
        f"example={ex['example']} input_line={ex['input_line']} func={ex['func']} "
        f"verdict={ex['verdict']}\n"
    )
    wf.write(f"source_file={rec.get('source_file')}\n")
    wf.write(
        f"ctrl(pred/source)={ex['pred_ctrl']}/{ex['src_ctrl']} "
        f"gaps(if,loop,switch)=({ex['if_gap']},{ex['loop_gap']},{ex['switch_gap']}) "
        f"sim={ex['type_sim']:.3f}/{ex['depth_sim']:.3f} reasons={ex['reasons']}\n"
    )
    wf.write(
        "label_counts="
        + ", ".join(f"{PATTERN_NAMES.get(k, k)}={v}" for k, v in sorted(counts.items()))
        + "\n"
    )
    wf.write(
        f"loop_header_bbs={[i for i, x in enumerate(labels) if x == 4]}\n"
        f"loop_body_bbs={[i for i, x in enumerate(labels) if x == 5]}\n"
        f"loop_exit_bbs={[i for i, x in enumerate(labels) if x == 6]}\n"
        f"switch_header_bbs={[i for i, x in enumerate(labels) if x == 7]}\n"
        f"switch_case_body_bbs={[i for i, x in enumerate(labels) if x == 8]}\n"
    )
    wf.write(
        "backedges="
        + (", ".join(f"{s}->{d}:{EDGE_NAMES.get(et, et)}" for s, d, et in backedges) or "-")
        + "\n"
    )
    wf.write(f"switch_like_outdegree_ge3_bbs={switch_like or '-'}\n")
    wf.write(f"has_indirect_jump_bbs={indirect_like or '-'}\n")

    wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
    wf.write("\n".join(audit.get("source_outline") or ["<none>"]) + "\n")
    wf.write("\n-- AB SKELETON CONTROL OUTLINE --\n")
    wf.write("\n".join(audit.get("pred_outline") or ["<none>"]) + "\n")

    wf.write("\n-- LOOP_HEADER DETAILS --\n")
    wf.write("\n".join(_header_rows(rec, labels, 4) or ["<none>"]) + "\n")
    wf.write("\n-- SWITCH_HEADER DETAILS --\n")
    wf.write("\n".join(_header_rows(rec, labels, 7) or ["<none>"]) + "\n")
    wf.write("\n-- LOOP_EXIT DETAILS --\n")
    wf.write("\n".join(_header_rows(rec, labels, 6) or ["<none>"]) + "\n")
    wf.write("\n-- SWITCH_CASE_BODY DETAILS --\n")
    wf.write("\n".join(_header_rows(rec, labels, 8) or ["<none>"]) + "\n")

    wf.write("\n-- 10CLASS LABEL SEQUENCE --\n")
    seq = [f"BB_{i}:{PATTERN_NAMES.get(x, x)}" for i, x in enumerate(labels)]
    wf.write("\n".join(seq[:260]) + ("\n...\n" if len(seq) > 260 else "\n"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-txt", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-examples", type=int, default=0)
    ap.add_argument(
        "--verdict",
        action="append",
        default=None,
        help="Verdict to include. Can be repeated. Default: LOOP_SWITCH_MISMATCH and CONTROL_COUNT_MISMATCH.",
    )
    ap.add_argument("--min-abs-loop-gap", type=int, default=1)
    ap.add_argument("--min-abs-switch-gap", type=int, default=1)
    ap.add_argument(
        "--include-count-mismatch",
        action="store_true",
        help="Also include CONTROL_COUNT_MISMATCH even when loop/switch gap is 0.",
    )
    args = ap.parse_args()

    examples = _parse_export_examples(args.export_txt.expanduser().resolve())
    verdicts = set(args.verdict or ["LOOP_SWITCH_MISMATCH"])
    selected: List[Dict[str, Any]] = []
    for ex in examples:
        loop_switch_bad = (
            abs(int(ex["loop_gap"])) >= int(args.min_abs_loop_gap)
            or abs(int(ex["switch_gap"])) >= int(args.min_abs_switch_gap)
        )
        if str(ex["verdict"]) in verdicts and loop_switch_bad:
            selected.append(ex)
        elif args.include_count_mismatch and str(ex["verdict"]) == "CONTROL_COUNT_MISMATCH":
            selected.append(ex)
        if args.limit_examples and len(selected) >= int(args.limit_examples):
            break

    records = _read_jsonl_lines(
        args.jsonl.expanduser().resolve(),
        [int(ex["input_line"]) for ex in selected],
    )

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as wf:
        wf.write("===== STAGE2AB LOOP/SWITCH MISMATCH DIAGNOSTIC =====\n")
        wf.write(f"export_txt={args.export_txt.expanduser().resolve()}\n")
        wf.write(f"jsonl={args.jsonl.expanduser().resolve()}\n")
        wf.write(f"total_examples_in_export={len(examples)}\n")
        wf.write(f"selected={len(selected)}\n")
        wf.write(f"verdict_filter={sorted(verdicts)}\n")
        wf.write(f"min_abs_loop_gap={int(args.min_abs_loop_gap)}\n")
        wf.write(f"min_abs_switch_gap={int(args.min_abs_switch_gap)}\n")
        wf.write("\n===== SELECTED SUMMARY =====\n")
        for ex in selected:
            wf.write(
                f"{ex['example']:3d} line={ex['input_line']} func={ex['func']} "
                f"verdict={ex['verdict']} ctrl={ex['pred_ctrl']}/{ex['src_ctrl']} "
                f"gaps=({ex['if_gap']},{ex['loop_gap']},{ex['switch_gap']}) "
                f"sim={ex['type_sim']:.3f}/{ex['depth_sim']:.3f}\n"
            )
        for ex in selected:
            rec = records.get(int(ex["input_line"]))
            if rec is None:
                continue
            _write_example(wf, ex, rec)

    print(f"saved_to = {out}")
    print(f"total_examples_in_export = {len(examples)}")
    print(f"selected = {len(selected)}")


if __name__ == "__main__":
    main()

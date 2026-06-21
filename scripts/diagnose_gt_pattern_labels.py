#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose whether stored GT ``pattern_labels`` are structurally plausible.

This script does *not* run Stage2 A/B.  It treats the JSONL's existing
``pattern_labels`` as the ground-truth labels and checks them against:

* CFG backedge targets / stored structural loop-header features,
* source control statements recovered from the source file,
* switch/case role coverage.

The goal is to separate "GT label noise" from "AB prediction noise".
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import re
import sys
from pathlib import Path
from typing import Any, Counter, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402


_ORIG_LINE_TO_TEXT = repack._line_to_text


@functools.lru_cache(maxsize=512)
def _cached_line_to_text(source_file: str) -> Dict[int, str]:
    return _ORIG_LINE_TO_TEXT(source_file)


def _line_to_text_cached(source_file: Any) -> Dict[int, str]:
    return _cached_line_to_text(str(source_file or ""))


repack._line_to_text = _line_to_text_cached


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

CONTROL_LINE_RE = {
    "if": re.compile(r"\bif\s*\("),
    "loop": re.compile(r"\b(for|while)\s*\(|\bdo\b"),
    "switch": re.compile(r"\bswitch\s*\("),
    "case": re.compile(r"\bcase\b[^:]*:|default\s*:"),
}


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.2f}%" if d else "0.00%"


def _iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception:
                continue


def _labels(record: Dict[str, Any]) -> List[int]:
    raw = record.get("pattern_labels")
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except Exception:
            out.append(-1)
    return out


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
    if not isinstance(raw, list) or bb < 0 or bb >= len(raw):
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


def _line_snippets(record: Dict[str, Any], lines: Sequence[int], *, max_lines: int = 5) -> str:
    lt = repack._line_to_text(record.get("source_file"))
    if not lt or not lines:
        return "-"
    parts: List[str] = []
    for ln in list(lines)[:max_lines]:
        txt = str(lt.get(int(ln), "")).strip()
        parts.append(f"L{ln}:{txt}" if txt else f"L{ln}")
    if len(lines) > max_lines:
        parts.append("...")
    return " | ".join(parts) if parts else "-"


def _line_has(record: Dict[str, Any], bb: int, kind: str) -> bool:
    lt = repack._line_to_text(record.get("source_file"))
    pat = CONTROL_LINE_RE[kind]
    for ln in _block_lines(record, bb):
        if pat.search(str(lt.get(int(ln), ""))):
            return True
    return False


def _struct_loop_headers(record: Dict[str, Any]) -> Set[int]:
    feats = record.get("bin_struct_features")
    out: Set[int] = set()
    if isinstance(feats, list):
        for i, row in enumerate(feats):
            if isinstance(row, list) and len(row) > 2:
                try:
                    if float(row[2]) > 0.5:
                        out.add(i)
                except Exception:
                    pass
    return out


def _struct_loop_exits(record: Dict[str, Any]) -> Set[int]:
    feats = record.get("bin_struct_features")
    out: Set[int] = set()
    if isinstance(feats, list):
        for i, row in enumerate(feats):
            if isinstance(row, list) and len(row) > 3:
                try:
                    if float(row[3]) > 0.5:
                        out.add(i)
                except Exception:
                    pass
    return out


def _indirect_jump_bbs(record: Dict[str, Any]) -> Set[int]:
    feats = record.get("bin_struct_features")
    out: Set[int] = set()
    if isinstance(feats, list):
        for i, row in enumerate(feats):
            if isinstance(row, list) and len(row) > 9:
                try:
                    if float(row[9]) > 0.5:
                        out.add(i)
                except Exception:
                    pass
    return out


def _source_control_counts(
    record: Dict[str, Any],
    source_control_filter: str,
    *,
    fast_source_lines: bool,
) -> Tuple[Counter[str], List[str], str]:
    if fast_source_lines:
        active_lines = effect._active_source_lines(record)
        outline = effect._active_line_control_outline(
            record,
            active_lines,
            source_control_filter=source_control_filter,
        )
        counts: Counter[str] = collections.Counter(t for t, _depth, _line in outline)
        return counts, effect._source_outline_lines(outline, limit=120), "fast_active_line_scan"

    rows = effect._source_rows(record)
    parse_status = effect._source_parse_status(record, rows)
    active_lines = effect._active_source_lines(record)
    outline = effect._source_control_outline(
        rows,
        active_lines=active_lines,
        source_control_filter=source_control_filter,
    )
    counts: Counter[str] = collections.Counter(t for t, _depth, _line in outline)
    return counts, effect._source_outline_lines(outline, limit=120), parse_status


def _record_diag(
    line_no: int,
    record: Dict[str, Any],
    *,
    source_control_filter: str,
    fast_source_lines: bool,
) -> Dict[str, Any]:
    labels = _labels(record)
    counts = collections.Counter(labels)
    edges = _edge_list(record)
    backedges = [(s, d) for s, d, et in edges if et == 2]
    back_targets = {d for _s, d in backedges}
    struct_lh = _struct_loop_headers(record)
    expected_lh = back_targets | struct_lh
    loop_headers = {i for i, x in enumerate(labels) if x == 4}
    loop_bodies = {i for i, x in enumerate(labels) if x == 5}
    loop_exits = {i for i, x in enumerate(labels) if x == 6}
    struct_le = _struct_loop_exits(record)

    outdeg: Counter[int] = collections.Counter(s for s, _d, _et in edges)
    switch_like = {bb for bb, deg in outdeg.items() if deg >= 3}
    indirect = _indirect_jump_bbs(record)
    switch_headers = {i for i, x in enumerate(labels) if x == 7}
    switch_cases = {i for i, x in enumerate(labels) if x == 8}

    src_counts, src_outline, parse_status = _source_control_counts(
        record,
        source_control_filter,
        fast_source_lines=fast_source_lines,
    )
    src_loop_count = int(src_counts.get("loop", 0))
    src_switch_count = int(src_counts.get("switch", 0))
    src_if_count = int(src_counts.get("if", 0))

    false_lh = loop_headers - expected_lh
    missed_lh = expected_lh - loop_headers
    false_lh_no_loop_line = {bb for bb in false_lh if not _line_has(record, bb, "loop")}
    switch_headers_no_switch_or_case_line = {
        bb for bb in switch_headers
        if not (_line_has(record, bb, "switch") or _line_has(record, bb, "case"))
    }

    return {
        "line_no": line_no,
        "func": record.get("func_name"),
        "source_file": record.get("source_file"),
        "n_bb": len(labels),
        "label_counts": counts,
        "source_counts": src_counts,
        "source_outline": src_outline,
        "parse_status": parse_status,
        "loop_headers": sorted(loop_headers),
        "loop_bodies": sorted(loop_bodies),
        "loop_exits": sorted(loop_exits),
        "struct_loop_exits": sorted(struct_le),
        "backedges": sorted(backedges),
        "backedge_targets": sorted(back_targets),
        "struct_loop_headers": sorted(struct_lh),
        "false_loop_headers": sorted(false_lh),
        "missed_loop_headers": sorted(missed_lh),
        "false_loop_headers_no_loop_line": sorted(false_lh_no_loop_line),
        "switch_headers": sorted(switch_headers),
        "switch_cases": sorted(switch_cases),
        "switch_like_outdegree_ge3": sorted(switch_like),
        "indirect_jump_bbs": sorted(indirect),
        "switch_headers_no_switch_or_case_line": sorted(switch_headers_no_switch_or_case_line),
        "src_loop_count": src_loop_count,
        "src_switch_count": src_switch_count,
        "src_if_count": src_if_count,
        "switch_header_over_source": max(0, len(switch_headers) - src_switch_count),
        "loop_header_over_source": max(0, len(loop_headers) - src_loop_count),
        "record": record,
    }


def _fmt_counts(counts: Counter[int]) -> str:
    parts = []
    for k in sorted(counts):
        name = PATTERN_NAMES.get(k, f"unknown_{k}")
        parts.append(f"{name}={counts[k]}")
    return ", ".join(parts) if parts else "-"


def _write_bb_details(wf: Any, record: Dict[str, Any], bbs: Sequence[int], label_name: str) -> None:
    if not bbs:
        wf.write("<none>\n")
        return
    edges = _edge_list(record)
    preds: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    succs: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for s, d, et in edges:
        preds[d].append((s, et))
        succs[s].append((d, et))
    feats = record.get("bin_struct_features")
    for bb in bbs:
        pred = ", ".join(f"{s}:{EDGE_NAMES.get(et, et)}" for s, et in sorted(preds.get(bb, []))) or "-"
        succ = ", ".join(f"{d}:{EDGE_NAMES.get(et, et)}" for d, et in sorted(succs.get(bb, []))) or "-"
        struct = "-"
        if isinstance(feats, list) and 0 <= bb < len(feats) and isinstance(feats[bb], list):
            row = feats[bb]
            struct = (
                f"in={row[0] if len(row) > 0 else '?'} "
                f"out={row[1] if len(row) > 1 else '?'} "
                f"is_loop_header={row[2] if len(row) > 2 else '?'} "
                f"is_loop_exit={row[3] if len(row) > 3 else '?'} "
                f"has_indirect_jump={row[9] if len(row) > 9 else '?'}"
            )
        lines = _block_lines(record, bb)
        wf.write(
            f"BB_{bb} gt={label_name} src_lines={lines or '-'} "
            f"src_text={_line_snippets(record, lines)} pred=[{pred}] succ=[{succ}] "
            f"struct=({struct})\n"
        )


def _write_example(wf: Any, diag: Dict[str, Any]) -> None:
    rec = diag["record"]
    wf.write("\n" + "=" * 120 + "\n")
    wf.write(
        f"line={diag['line_no']} func={diag['func']} source_file={diag['source_file']}\n"
        f"n_bb={diag['n_bb']} parse={diag['parse_status']}\n"
        f"gt_label_counts={_fmt_counts(diag['label_counts'])}\n"
        f"source_counts={dict(diag['source_counts'])}\n"
        f"loop_headers={diag['loop_headers']} backedge_targets={diag['backedge_targets']} "
        f"struct_loop_headers={diag['struct_loop_headers']}\n"
        f"false_loop_headers={diag['false_loop_headers']} "
        f"missed_loop_headers={diag['missed_loop_headers']}\n"
        f"switch_headers={diag['switch_headers']} switch_cases={diag['switch_cases']} "
        f"switch_like_outdegree_ge3={diag['switch_like_outdegree_ge3']} "
        f"indirect_jump_bbs={diag['indirect_jump_bbs']}\n"
    )
    wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
    wf.write("\n".join(diag["source_outline"] or ["<none>"]) + "\n")
    wf.write("\n-- FALSE LOOP_HEADER DETAILS --\n")
    _write_bb_details(wf, rec, diag["false_loop_headers"], "loop_header")
    wf.write("\n-- MISSED LOOP HEADER DETAILS --\n")
    _write_bb_details(wf, rec, diag["missed_loop_headers"], "expected_loop_header")
    wf.write("\n-- SWITCH_HEADER DETAILS --\n")
    _write_bb_details(wf, rec, diag["switch_headers"], "switch_header")
    wf.write("\n-- SWITCH_CASE_BODY DETAILS --\n")
    _write_bb_details(wf, rec, diag["switch_cases"], "switch_case_body")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means scan the whole JSONL")
    ap.add_argument("--max-examples", type=int, default=40)
    ap.add_argument("--source-control-filter", choices=["raw", "bb-start-line", "bb-range"], default="bb-start-line")
    ap.add_argument("--min-source-controls", type=int, default=1)
    ap.add_argument(
        "--fast-source-lines",
        action="store_true",
        help="Use active source-line regex scan instead of full statement-row extraction.",
    )
    ap.add_argument("--only-suspicious", action="store_true", help="Only export examples with loop/switch GT suspicion")
    args = ap.parse_args()

    scanned = 0
    eligible = 0
    malformed = 0
    label_counts_total: Counter[int] = collections.Counter()
    source_counts_total: Counter[str] = collections.Counter()
    parse_status_counts: Counter[str] = collections.Counter()

    loop_header_total = 0
    loop_body_total = 0
    loop_exit_total = 0
    struct_loop_exit_total = 0
    backedge_target_total = 0
    struct_loop_header_total = 0
    false_loop_header_total = 0
    missed_loop_header_total = 0
    false_loop_no_loop_line_total = 0

    switch_header_total = 0
    switch_case_total = 0
    switch_like_total = 0
    indirect_total = 0
    switch_header_over_source_total = 0
    switch_header_no_switch_or_case_line_total = 0

    funcs_with_false_loop_header = 0
    funcs_with_missed_loop_header = 0
    funcs_with_switch_header = 0
    funcs_with_switch_header_no_case = 0
    funcs_switch_header_gt_source_switch = 0
    funcs_switch_header_without_source_switch = 0

    loop_examples: List[Dict[str, Any]] = []
    switch_examples: List[Dict[str, Any]] = []
    exported_examples: List[Dict[str, Any]] = []

    for line_no, record in _iter_jsonl(args.jsonl.expanduser().resolve(), int(args.limit)):
        scanned += 1
        labels = _labels(record)
        n_bb = len(record.get("bin_block_lengths") or record.get("block_src_lines") or labels)
        if not labels or len(labels) != n_bb:
            malformed += 1
            continue
        diag = _record_diag(
            line_no,
            record,
            source_control_filter=str(args.source_control_filter),
            fast_source_lines=bool(args.fast_source_lines),
        )
        if sum(diag["source_counts"].values()) < int(args.min_source_controls):
            continue
        eligible += 1

        label_counts_total.update(diag["label_counts"])
        source_counts_total.update(diag["source_counts"])
        parse_status_counts[str(diag["parse_status"])] += 1

        loop_header_total += len(diag["loop_headers"])
        loop_body_total += len(diag["loop_bodies"])
        loop_exit_total += len(diag["loop_exits"])
        struct_loop_exit_total += len(diag["struct_loop_exits"])
        backedge_target_total += len(diag["backedge_targets"])
        struct_loop_header_total += len(diag["struct_loop_headers"])
        false_loop_header_total += len(diag["false_loop_headers"])
        missed_loop_header_total += len(diag["missed_loop_headers"])
        false_loop_no_loop_line_total += len(diag["false_loop_headers_no_loop_line"])

        switch_header_total += len(diag["switch_headers"])
        switch_case_total += len(diag["switch_cases"])
        switch_like_total += len(diag["switch_like_outdegree_ge3"])
        indirect_total += len(diag["indirect_jump_bbs"])
        switch_header_over_source_total += int(diag["switch_header_over_source"])
        switch_header_no_switch_or_case_line_total += len(diag["switch_headers_no_switch_or_case_line"])

        if diag["false_loop_headers"]:
            funcs_with_false_loop_header += 1
            loop_examples.append(diag)
        if diag["missed_loop_headers"]:
            funcs_with_missed_loop_header += 1
        if diag["switch_headers"]:
            funcs_with_switch_header += 1
            if not diag["switch_cases"]:
                funcs_with_switch_header_no_case += 1
            if len(diag["switch_headers"]) > int(diag["src_switch_count"]):
                funcs_switch_header_gt_source_switch += 1
            if int(diag["src_switch_count"]) == 0:
                funcs_switch_header_without_source_switch += 1
            switch_examples.append(diag)

        suspicious = bool(diag["false_loop_headers"]) or (
            bool(diag["switch_headers"])
            and (not diag["switch_cases"] or len(diag["switch_headers"]) > int(diag["src_switch_count"]))
        )
        if (not args.only_suspicious or suspicious) and len(exported_examples) < int(args.max_examples):
            exported_examples.append(diag)

    loop_examples.sort(key=lambda d: (len(d["false_loop_headers"]), len(d["loop_headers"])), reverse=True)
    switch_examples.sort(key=lambda d: (len(d["switch_headers"]), int(d["switch_header_over_source"])), reverse=True)

    outp = args.out.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as wf:
        wf.write("===== GT PATTERN_LABEL DIAGNOSTIC =====\n")
        wf.write(f"jsonl = {args.jsonl.expanduser().resolve()}\n")
        wf.write("label_source = stored_jsonl_pattern_labels_treated_as_GT\n")
        wf.write(f"scanned = {scanned}\n")
        wf.write(f"eligible_with_source_control = {eligible}\n")
        wf.write(f"malformed_or_missing_labels = {malformed}\n")
        wf.write(f"source_control_filter = {args.source_control_filter}\n")
        wf.write(f"fast_source_lines = {bool(args.fast_source_lines)}\n")
        wf.write(f"min_source_controls = {args.min_source_controls}\n")
        wf.write(f"parse_status_counts = {dict(parse_status_counts)}\n")
        wf.write("\n===== GT LABEL COUNTS =====\n")
        total_labels = sum(label_counts_total.values())
        for lab in sorted(PATTERN_NAMES):
            c = int(label_counts_total.get(lab, 0))
            wf.write(f"{lab:2d} {PATTERN_NAMES[lab]:28s} {c:8d} {_pct(c, total_labels):>8s}\n")
        wf.write(f"total_label_bbs = {total_labels}\n")
        wf.write(f"source_control_counts = {dict(source_counts_total)}\n")

        wf.write("\n===== LOOP GT CHECK =====\n")
        wf.write(f"gt_loop_header_labels = {loop_header_total}\n")
        wf.write(f"cfg_backedge_targets = {backedge_target_total}\n")
        wf.write(f"struct_loop_header_features = {struct_loop_header_total}\n")
        wf.write(f"gt_false_loop_headers_vs_cfg_or_struct = {false_loop_header_total} ({_pct(false_loop_header_total, loop_header_total)})\n")
        wf.write(f"gt_missed_loop_headers_vs_cfg_or_struct = {missed_loop_header_total}\n")
        wf.write(f"gt_false_loop_headers_without_loop_source_line = {false_loop_no_loop_line_total}\n")
        wf.write(f"gt_loop_body_labels = {loop_body_total}\n")
        wf.write(f"gt_loop_exit_labels = {loop_exit_total}\n")
        wf.write(f"struct_loop_exit_features = {struct_loop_exit_total}\n")
        wf.write(f"funcs_with_false_loop_header = {funcs_with_false_loop_header} ({_pct(funcs_with_false_loop_header, eligible)})\n")
        wf.write(f"funcs_with_missed_loop_header = {funcs_with_missed_loop_header} ({_pct(funcs_with_missed_loop_header, eligible)})\n")

        wf.write("\n===== SWITCH GT CHECK =====\n")
        wf.write(f"gt_switch_header_labels = {switch_header_total}\n")
        wf.write(f"gt_switch_case_body_labels = {switch_case_total}\n")
        wf.write(f"cfg_outdegree_ge3_switch_like_bbs = {switch_like_total}\n")
        wf.write(f"struct_indirect_jump_bbs = {indirect_total}\n")
        wf.write(f"gt_switch_header_over_source_switch_count_sum = {switch_header_over_source_total}\n")
        wf.write(f"gt_switch_headers_without_switch_or_case_source_line = {switch_header_no_switch_or_case_line_total}\n")
        wf.write(f"funcs_with_switch_header = {funcs_with_switch_header} ({_pct(funcs_with_switch_header, eligible)})\n")
        wf.write(f"funcs_with_switch_header_and_no_case_body = {funcs_with_switch_header_no_case} ({_pct(funcs_with_switch_header_no_case, funcs_with_switch_header)})\n")
        wf.write(f"funcs_switch_header_count_gt_source_switch_count = {funcs_switch_header_gt_source_switch} ({_pct(funcs_switch_header_gt_source_switch, funcs_with_switch_header)})\n")
        wf.write(f"funcs_switch_header_without_source_switch = {funcs_switch_header_without_source_switch} ({_pct(funcs_switch_header_without_source_switch, funcs_with_switch_header)})\n")

        wf.write("\n===== TOP LOOP GT SUSPICIOUS EXAMPLES =====\n")
        for d in loop_examples[:20]:
            wf.write(
                f"line={d['line_no']:6d} func={str(d['func'])[:34]:34s} "
                f"loop_headers={d['loop_headers']} backedge_targets={d['backedge_targets']} "
                f"struct_loop_headers={d['struct_loop_headers']} "
                f"false_headers={d['false_loop_headers']} missed={d['missed_loop_headers']}\n"
            )

        wf.write("\n===== TOP SWITCH GT SUSPICIOUS EXAMPLES =====\n")
        for d in switch_examples[:20]:
            wf.write(
                f"line={d['line_no']:6d} func={str(d['func'])[:34]:34s} "
                f"source_switch={d['src_switch_count']} switch_headers={d['switch_headers']} "
                f"case_bodies={d['switch_cases']} outdeg>=3={d['switch_like_outdegree_ge3']} "
                f"indirect={d['indirect_jump_bbs']}\n"
            )

        wf.write("\n===== DETAILED EXAMPLES =====\n")
        for d in exported_examples:
            _write_example(wf, d)

    print(f"saved_to = {outp}")
    print(f"scanned = {scanned}")
    print(f"eligible_with_source_control = {eligible}")
    print(f"gt_false_loop_headers_vs_cfg_or_struct = {false_loop_header_total}/{loop_header_total}")
    print(f"gt_switch_header_labels = {switch_header_total}")
    print(f"gt_switch_case_body_labels = {switch_case_total}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair AB-stage ``pattern_labels`` and emit deduplicated opt-level splits.

This is a conservative GT-label repair pass for the AB/Stage2 dataset.  It is
designed for the current failure mode:

* many non-loop-header BBs are labelled ``LOOP_HEADER`` just because they align
  to a source ``for``/``while``/``do`` line;
* switch lowering creates many ``SWITCH_HEADER`` labels while
  ``SWITCH_CASE_BODY`` is almost unused.

The script streams large JSONL files, rewrites only ``pattern_labels``, dedups
within each optimization level by original source function, and writes:

* combined repaired train/val JSONL,
* five per-opt train JSONL files,
* five per-opt val JSONL files,
* one stats report.

It never overwrites inputs by default.
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


SEQUENTIAL = 0
IF_HEADER = 1
IF_THEN_BODY = 2
IF_ELSE_BODY = 3
LOOP_HEADER = 4
LOOP_BODY = 5
LOOP_EXIT = 6
SWITCH_HEADER = 7
SWITCH_CASE_BODY = 8
FUNC_PROLOGUE_EPILOGUE = 9

EDGE_COND = 0
EDGE_UNCOND = 1
EDGE_BACK = 2

PATTERN_NAMES = {
    SEQUENTIAL: "sequential",
    IF_HEADER: "if_header",
    IF_THEN_BODY: "if_then_body",
    IF_ELSE_BODY: "if_else_body",
    LOOP_HEADER: "loop_header",
    LOOP_BODY: "loop_body",
    LOOP_EXIT: "loop_exit",
    SWITCH_HEADER: "switch_header",
    SWITCH_CASE_BODY: "switch_case_body",
    FUNC_PROLOGUE_EPILOGUE: "function_prologue_epilogue",
}

OPT_ORDER = ("o0", "o1", "o2", "o3", "os")

IF_RE = re.compile(r"\bif\s*\(")
LOOP_RE = re.compile(r"\b(for|while)\s*\(|\bdo\b")
SWITCH_RE = re.compile(r"\bswitch\s*\(")
CASE_RE = re.compile(r"\bcase\b[^:]*:|default\s*:")
TERMINAL_RE = re.compile(r"\b(return|break|continue|goto)\b")
BREAK_CONTINUE_RE = re.compile(r"\b(break|continue)\b")
COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.S | re.M)


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "0.00%"


def _strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", str(text or ""))


@functools.lru_cache(maxsize=768)
def _line_to_text(source_file: str) -> Dict[int, str]:
    if not source_file:
        return {}
    p = Path(source_file)
    if not p.is_file():
        return {}
    try:
        return {
            i: line
            for i, line in enumerate(
                p.read_text(encoding="utf-8", errors="replace").splitlines(),
                1,
            )
        }
    except Exception:
        return {}


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _n_bb(row: Dict[str, Any]) -> int:
    for key in ("pattern_labels", "bin_block_lengths", "block_src_lines"):
        val = row.get(key)
        if isinstance(val, list):
            return len(val)
    return 0


def _labels(row: Dict[str, Any], n_bb: int) -> List[int]:
    raw = row.get("pattern_labels")
    if not isinstance(raw, list):
        return [SEQUENTIAL] * n_bb
    out: List[int] = []
    for x in raw[:n_bb]:
        try:
            v = int(x)
        except Exception:
            v = SEQUENTIAL
        out.append(v if v in PATTERN_NAMES else SEQUENTIAL)
    if len(out) < n_bb:
        out.extend([SEQUENTIAL] * (n_bb - len(out)))
    return out


def _edge_list(row: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    raw = row.get("bin_edge_index")
    typ = row.get("bin_edge_type") or []
    out: List[Tuple[int, int, int]] = []
    if isinstance(raw, list) and len(raw) == 2:
        srcs, dsts = raw
        for k, (s, d) in enumerate(zip(srcs or [], dsts or [])):
            try:
                et = int(typ[k]) if k < len(typ) else -1
                out.append((int(s), int(d), et))
            except Exception:
                continue
    return out


def _block_lines(row: Dict[str, Any], bb: int) -> List[int]:
    raw = row.get("block_src_lines") or []
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


def _bb_text(row: Dict[str, Any], bb: int) -> str:
    lt = _line_to_text(str(row.get("source_file") or ""))
    parts = []
    for ln in _block_lines(row, bb):
        txt = lt.get(int(ln), "")
        if txt:
            parts.append(txt)
    return _strip_comments("\n".join(parts))


def _struct_feature_set(row: Dict[str, Any], idx: int, threshold: float = 0.5) -> Set[int]:
    feats = row.get("bin_struct_features")
    out: Set[int] = set()
    if isinstance(feats, list):
        for i, vals in enumerate(feats):
            if not isinstance(vals, list) or len(vals) <= idx:
                continue
            try:
                if float(vals[idx]) > threshold:
                    out.add(i)
            except Exception:
                pass
    return out


def _outdegree_map(edges: Sequence[Tuple[int, int, int]]) -> Counter[int]:
    c: Counter[int] = collections.Counter()
    for s, _d, _et in edges:
        c[s] += 1
    return c


def _succs_map(edges: Sequence[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for s, d, et in edges:
        out[s].append((d, et))
    return out


def _preds_map(edges: Sequence[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for s, d, et in edges:
        out[d].append((s, et))
    return out


def _has_cond_branch(succs: Dict[int, List[Tuple[int, int]]], bb: int) -> bool:
    ss = succs.get(bb, [])
    if any(et == EDGE_COND for _d, et in ss):
        return True
    return len(ss) >= 2


def _choose_canonical_per_source_line(
    row: Dict[str, Any],
    candidates: Set[int],
    pattern: re.Pattern[str],
    preferred: Optional[Set[int]] = None,
) -> Set[int]:
    """Keep at most one canonical header for each source control line."""
    if not candidates:
        return set()
    preferred = set(preferred or set())
    grouped: Dict[int, List[int]] = collections.defaultdict(list)
    no_line: List[int] = []
    lt = _line_to_text(str(row.get("source_file") or ""))
    for bb in sorted(candidates):
        matched = False
        for ln in _block_lines(row, bb):
            if pattern.search(_strip_comments(lt.get(int(ln), ""))):
                grouped[int(ln)].append(bb)
                matched = True
        if not matched:
            no_line.append(bb)

    kept: Set[int] = set()
    for _ln, bbs in grouped.items():
        bbs = sorted(set(bbs))
        best = sorted(bbs, key=lambda b: (0 if b in preferred else 1, b))[0]
        kept.add(best)
    kept.update(no_line)
    return kept


def _clusters_by_distance(values: Set[int], max_gap: int = 3) -> List[List[int]]:
    clusters: List[List[int]] = []
    cur: List[int] = []
    prev: Optional[int] = None
    for bb in sorted(values):
        if prev is None or bb - prev <= max_gap:
            cur.append(bb)
        else:
            if cur:
                clusters.append(cur)
            cur = [bb]
        prev = bb
    if cur:
        clusters.append(cur)
    return clusters


def _repair_labels(row: Dict[str, Any]) -> Tuple[List[int], Dict[str, int]]:
    n = _n_bb(row)
    old = _labels(row, n)
    edges = _edge_list(row)
    succs = _succs_map(edges)
    preds = _preds_map(edges)
    outdeg = _outdegree_map(edges)

    back_targets = {d for _s, d, et in edges if et == EDGE_BACK and 0 <= d < n}
    back_sources = {s for s, _d, et in edges if et == EDGE_BACK and 0 <= s < n}
    struct_loop_headers = {i for i in _struct_feature_set(row, 2) if 0 <= i < n}
    struct_loop_exits = {i for i in _struct_feature_set(row, 3) if 0 <= i < n}
    indirect_jump_bbs = {i for i in _struct_feature_set(row, 9) if 0 <= i < n}
    switch_like_bbs = {bb for bb, deg in outdeg.items() if deg >= 3 and 0 <= bb < n}

    # High-precision loop headers: CFG/structural evidence first.  This is the
    # key fix for the observed false LOOP_HEADER explosion.
    loop_headers = back_targets | struct_loop_headers
    loop_headers = _choose_canonical_per_source_line(
        row,
        loop_headers,
        LOOP_RE,
        preferred=back_targets,
    )

    # Canonical switch header: one source switch line -> one header.  Also keep
    # binary switch-like dispatch nodes when source-line evidence is available.
    # If neither survives, keep one old switch header per local cluster as a
    # conservative fallback; otherwise we would erase source switches that were
    # lowered to a compare chain and whose BBs no longer carry the switch line.
    old_switch_headers = {i for i, x in enumerate(old) if x == SWITCH_HEADER}
    switch_source_candidates = {
        bb for bb in range(n)
        if SWITCH_RE.search(_bb_text(row, bb))
    }
    switch_headers = _choose_canonical_per_source_line(
        row,
        switch_source_candidates,
        SWITCH_RE,
        preferred=indirect_jump_bbs | switch_like_bbs,
    )
    if not switch_headers:
        switch_headers = indirect_jump_bbs | switch_like_bbs
    old_switch_clusters = _clusters_by_distance(old_switch_headers, max_gap=3)
    if not switch_headers and old_switch_clusters:
        for cluster in old_switch_clusters:
            switch_headers.add(cluster[0])
    switch_headers = {bb for bb in switch_headers if 0 <= bb < n}

    # If old GT encoded one source switch as a chain of many SWITCH_HEADER BBs,
    # keep one canonical header per cluster and reinterpret the remaining old
    # switch headers as SWITCH_CASE_BODY.  This directly fixes the observed
    # "many switch_header, almost no switch_case_body" supervision bug.
    old_switch_cluster_cases: Set[int] = set()
    for cluster in old_switch_clusters:
        in_cluster_headers = sorted(bb for bb in cluster if bb in switch_headers)
        if in_cluster_headers:
            canonical = in_cluster_headers[0]
        else:
            canonical = cluster[0]
            switch_headers.add(canonical)
        for bb in cluster:
            if bb != canonical:
                old_switch_cluster_cases.add(bb)

    switch_cases = {
        bb for bb in range(n)
        if bb not in switch_headers and CASE_RE.search(_bb_text(row, bb))
    }
    switch_cases.update(bb for bb in old_switch_cluster_cases if bb not in switch_headers)

    new = list(old)
    stats: Dict[str, int] = collections.Counter()

    # First demote old over-broad control headers.  We keep body labels unless a
    # stronger canonical role below overwrites them.
    for bb, lab in enumerate(old):
        txt = _bb_text(row, bb)
        if lab == LOOP_HEADER and bb not in loop_headers:
            stats["demoted_loop_header"] += 1
            if IF_RE.search(txt) and _has_cond_branch(succs, bb):
                new[bb] = IF_HEADER
            elif CASE_RE.search(txt):
                new[bb] = SWITCH_CASE_BODY
            elif TERMINAL_RE.search(txt) and (bb in struct_loop_exits or not succs.get(bb)):
                new[bb] = LOOP_EXIT
            else:
                new[bb] = SEQUENTIAL
        elif lab == SWITCH_HEADER and bb not in switch_headers:
            stats["demoted_switch_header"] += 1
            if CASE_RE.search(txt):
                new[bb] = SWITCH_CASE_BODY
            elif IF_RE.search(txt) and _has_cond_branch(succs, bb):
                new[bb] = IF_HEADER
            else:
                new[bb] = SEQUENTIAL

    # Repair canonical roles.
    for bb in loop_headers:
        if old[bb] != LOOP_HEADER:
            stats["promoted_loop_header"] += 1
        new[bb] = LOOP_HEADER

    for bb in switch_headers:
        if old[bb] != SWITCH_HEADER:
            stats["promoted_switch_header"] += 1
        new[bb] = SWITCH_HEADER

    for bb in switch_cases:
        if new[bb] != SWITCH_CASE_BODY:
            stats["promoted_switch_case_body"] += 1
        new[bb] = SWITCH_CASE_BODY

    # Loop body and exit are secondary roles.  Do not overwrite headers or
    # switch cases.
    for bb in back_sources:
        if new[bb] in {SEQUENTIAL, IF_THEN_BODY, IF_ELSE_BODY}:
            if old[bb] != LOOP_BODY:
                stats["promoted_loop_body"] += 1
            new[bb] = LOOP_BODY

    for bb in struct_loop_exits:
        if new[bb] in {SEQUENTIAL, IF_THEN_BODY, IF_ELSE_BODY}:
            txt = _bb_text(row, bb)
            # Be conservative: loop exits should be terminal-like or have a CFG
            # loop-exit feature; this avoids flooding the class.
            if TERMINAL_RE.search(txt) or bb in struct_loop_exits:
                if old[bb] != LOOP_EXIT:
                    stats["promoted_loop_exit"] += 1
                new[bb] = LOOP_EXIT

    # break/continue are strong loop-exit body signals even when the structural
    # feature extractor did not mark loop exits.
    for bb in range(n):
        if new[bb] in {
            IF_HEADER, LOOP_HEADER, LOOP_BODY, LOOP_EXIT,
            SWITCH_HEADER, SWITCH_CASE_BODY,
        }:
            continue
        if BREAK_CONTINUE_RE.search(_bb_text(row, bb)):
            if old[bb] != LOOP_EXIT:
                stats["promoted_loop_exit_break_continue"] += 1
            new[bb] = LOOP_EXIT

    # Fix obvious if headers that source/CFG both support, but avoid overriding
    # stronger loop/switch roles.
    for bb in range(n):
        if new[bb] in {LOOP_HEADER, SWITCH_HEADER, SWITCH_CASE_BODY}:
            continue
        txt = _bb_text(row, bb)
        if IF_RE.search(txt) and _has_cond_branch(succs, bb):
            if new[bb] != IF_HEADER:
                stats["promoted_if_header"] += 1
            new[bb] = IF_HEADER

    # Function exits: only when there is no outgoing edge and no stronger role.
    for bb in range(n):
        if new[bb] in {
            IF_HEADER, LOOP_HEADER, LOOP_BODY, LOOP_EXIT,
            SWITCH_HEADER, SWITCH_CASE_BODY,
        }:
            continue
        txt = _bb_text(row, bb)
        if TERMINAL_RE.search(txt) and not succs.get(bb):
            if new[bb] != FUNC_PROLOGUE_EPILOGUE:
                stats["promoted_func_exit"] += 1
            new[bb] = FUNC_PROLOGUE_EPILOGUE

    stats["changed_bbs"] = sum(1 for a, b in zip(old, new) if a != b)
    stats["old_loop_header"] = sum(1 for x in old if x == LOOP_HEADER)
    stats["new_loop_header"] = sum(1 for x in new if x == LOOP_HEADER)
    stats["old_switch_header"] = sum(1 for x in old if x == SWITCH_HEADER)
    stats["new_switch_header"] = sum(1 for x in new if x == SWITCH_HEADER)
    stats["old_switch_case_body"] = sum(1 for x in old if x == SWITCH_CASE_BODY)
    stats["new_switch_case_body"] = sum(1 for x in new if x == SWITCH_CASE_BODY)
    stats["old_loop_exit"] = sum(1 for x in old if x == LOOP_EXIT)
    stats["new_loop_exit"] = sum(1 for x in new if x == LOOP_EXIT)
    return new, dict(stats)


def _normalise_opt(raw: Any, path: Optional[str] = None) -> str:
    vals = [
        raw,
        path,
    ]
    for val in vals:
        s = str(val or "").lower()
        for opt in OPT_ORDER:
            if re.search(rf"(^|[^a-z0-9]){opt}([^a-z0-9]|$)", s):
                return opt
        # Some datasets store -O0 / -Os.
        m = re.search(r"-o([0123s])\b", s)
        if m:
            return "o" + m.group(1)
    return "unknown"


def _record_opt(row: Dict[str, Any]) -> str:
    for key in ("opt_level", "optimization", "opt", "compiler_opt"):
        if key in row:
            opt = _normalise_opt(row.get(key))
            if opt != "unknown":
                return opt
    for key in ("binary", "binary_path", "object_file", "source_file"):
        opt = _normalise_opt(None, str(row.get(key) or ""))
        if opt != "unknown":
            return opt
    return "unknown"


def _function_key(row: Dict[str, Any], opt: str) -> Tuple[str, str, str]:
    source_file = str(row.get("source_file") or "")
    func = str(row.get("func_name") or row.get("function") or row.get("name") or "")
    if source_file and func:
        return (opt, source_file, func)
    binary = str(row.get("binary") or row.get("binary_path") or row.get("object_file") or "")
    addr = str(row.get("func_addr") or row.get("address") or row.get("entry") or row.get("idx") or "")
    if source_file or func:
        return (opt, source_file or binary, func or addr)
    return (
        opt,
        binary,
        addr,
    )


class WriterSet:
    def __init__(self, out_dir: Path, prefix: str, combined_name: str):
        self.out_dir = out_dir
        self.prefix = prefix
        self.combined_path = out_dir / combined_name
        self.combined = self.combined_path.open("w", encoding="utf-8")
        self.by_opt = {
            opt: (out_dir / f"{prefix}_{opt}.patternfix.dedup.jsonl").open("w", encoding="utf-8")
            for opt in OPT_ORDER
        }
        self.unknown_path = out_dir / f"{prefix}_unknown.patternfix.dedup.jsonl"
        self.unknown = self.unknown_path.open("w", encoding="utf-8")

    def write(self, opt: str, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.combined.write(line)
        if opt in self.by_opt:
            self.by_opt[opt].write(line)
        else:
            self.unknown.write(line)

    def close(self) -> None:
        self.combined.close()
        for f in self.by_opt.values():
            f.close()
        self.unknown.close()

    def paths(self) -> List[Path]:
        return [self.combined_path] + [
            self.out_dir / f"{self.prefix}_{opt}.patternfix.dedup.jsonl"
            for opt in OPT_ORDER
        ] + [self.unknown_path]


def _process_file(
    *,
    input_path: Path,
    writer: WriterSet,
    seen_by_opt: Dict[str, Set[Tuple[str, str, str]]],
    split_name: str,
    limit: int,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "split": split_name,
        "input": str(input_path),
        "read": 0,
        "written": 0,
        "duplicates_skipped": 0,
        "unknown_opt_written": 0,
        "malformed_skipped": 0,
        "by_opt_read": collections.Counter(),
        "by_opt_written": collections.Counter(),
        "label_counts_before": collections.Counter(),
        "label_counts_after": collections.Counter(),
        "repair_stats": collections.Counter(),
    }
    for line_no, row in _iter_jsonl(input_path):
        if limit and stats["read"] >= limit:
            break
        stats["read"] += 1
        n = _n_bb(row)
        if n <= 0:
            stats["malformed_skipped"] += 1
            continue
        opt = _record_opt(row)
        stats["by_opt_read"][opt] += 1
        key = _function_key(row, opt)
        if key in seen_by_opt.setdefault(opt, set()):
            stats["duplicates_skipped"] += 1
            continue
        old_labels = _labels(row, n)
        new_labels, repair_stats = _repair_labels(row)
        if len(new_labels) != n:
            stats["malformed_skipped"] += 1
            continue
        out = dict(row)
        out["pattern_labels"] = [int(x) for x in new_labels]
        out["pattern_labels_repair"] = {
            "version": "patternfix_loop_switch_v1",
            "source": "scripts/repair_ab_pattern_labels_dedup.py",
            "original_file": str(input_path),
            "original_line": int(line_no),
            "changed_bbs": int(repair_stats.get("changed_bbs", 0)),
        }
        writer.write(opt, out)
        seen_by_opt[opt].add(key)
        stats["written"] += 1
        stats["by_opt_written"][opt] += 1
        if opt == "unknown":
            stats["unknown_opt_written"] += 1
        stats["label_counts_before"].update(old_labels)
        stats["label_counts_after"].update(new_labels)
        stats["repair_stats"].update(repair_stats)
    return stats


def _counter_to_dict(c: Counter[Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in c.items()}


def _write_stats(path: Path, stats_list: Sequence[Dict[str, Any]], writer_paths: Sequence[Path]) -> None:
    with path.open("w", encoding="utf-8") as wf:
        wf.write("===== AB PATTERN_LABEL REPAIR + DEDUP STATS =====\n")
        wf.write("repair_version = patternfix_loop_switch_v1\n")
        wf.write("dedup_key = (opt_level, source_file, func_name)\n")
        wf.write("\n===== OUTPUT FILES =====\n")
        for p in writer_paths:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            wf.write(f"{size / 1024 / 1024 / 1024:.2f}G\t{p}\n")
        for stats in stats_list:
            wf.write("\n" + "=" * 100 + "\n")
            wf.write(f"split = {stats['split']}\n")
            wf.write(f"input = {stats['input']}\n")
            for key in ("read", "written", "duplicates_skipped", "unknown_opt_written", "malformed_skipped"):
                wf.write(f"{key} = {stats[key]}\n")
            wf.write(f"by_opt_read = {_counter_to_dict(stats['by_opt_read'])}\n")
            wf.write(f"by_opt_written = {_counter_to_dict(stats['by_opt_written'])}\n")
            wf.write("\n-- label counts before/after --\n")
            before = stats["label_counts_before"]
            after = stats["label_counts_after"]
            total_before = sum(before.values())
            total_after = sum(after.values())
            for lab in sorted(PATTERN_NAMES):
                b = before.get(lab, 0)
                a = after.get(lab, 0)
                wf.write(
                    f"{lab:2d} {PATTERN_NAMES[lab]:28s} "
                    f"before={b:10d} ({_pct(b, total_before):>7s}) "
                    f"after={a:10d} ({_pct(a, total_after):>7s})\n"
                )
            wf.write("\n-- repair stats --\n")
            for key, val in sorted(stats["repair_stats"].items()):
                wf.write(f"{key} = {int(val)}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True, help="Usually AB_2048.jsonl")
    ap.add_argument("--val", type=Path, required=True, help="Usually AB_val.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug only; 0 means full input")
    args = ap.parse_args()

    train_path = args.train.expanduser().resolve()
    val_path = args.val.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else train_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_writer = WriterSet(
        out_dir,
        "AB_train_2048",
        "AB_2048.patternfix.dedup.jsonl",
    )
    val_writer = WriterSet(
        out_dir,
        "AB_val",
        "AB_val.patternfix.dedup.jsonl",
    )
    try:
        train_stats = _process_file(
            input_path=train_path,
            writer=train_writer,
            seen_by_opt={},
            split_name="train",
            limit=int(args.limit),
        )
        val_stats = _process_file(
            input_path=val_path,
            writer=val_writer,
            seen_by_opt={},
            split_name="val",
            limit=int(args.limit),
        )
    finally:
        train_writer.close()
        val_writer.close()

    stats_path = out_dir / "AB_2048_AB_val.patternfix.dedup.stats.txt"
    _write_stats(stats_path, [train_stats, val_stats], train_writer.paths() + val_writer.paths())

    print(f"train_out = {train_writer.combined_path}")
    print(f"val_out = {val_writer.combined_path}")
    print(f"stats = {stats_path}")
    print(f"train_read/written = {train_stats['read']}/{train_stats['written']}")
    print(f"train_duplicates_skipped = {train_stats['duplicates_skipped']}")
    print(f"val_read/written = {val_stats['read']}/{val_stats['written']}")
    print(f"val_duplicates_skipped = {val_stats['duplicates_skipped']}")


if __name__ == "__main__":
    main()

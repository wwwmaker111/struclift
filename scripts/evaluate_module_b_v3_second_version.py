#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Module-B v3 second-version ranker + slot grounding smoke quality.

The deterministic MVP candidate generator and strict validator remain the safety
boundary.  This script adds the second-version layer from
``module_b_v3_practical_candidate_ranker_strategy.md``:

* build multiple valid candidates;
* score them offline against the teacher to create L_rank supervision;
* train/evaluate a small candidate ranker;
* audit slot grounding pointers back to BB/instruction/CFG edges.

The model never creates skeleton structure.  It only chooses among candidates
that have already passed deterministic validation or can fall back safely.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MVP_PATH = ROOT / "struclift" / "utils" / "module_b_v3_mvp.py"
MVP_SPEC = importlib.util.spec_from_file_location("module_b_v3_mvp_standalone", MVP_PATH)
if MVP_SPEC is None or MVP_SPEC.loader is None:
    raise RuntimeError(f"cannot load {MVP_PATH}")
MVP_MODULE = importlib.util.module_from_spec(MVP_SPEC)
sys.modules[MVP_SPEC.name] = MVP_MODULE
MVP_SPEC.loader.exec_module(MVP_MODULE)

RANKER_PATH = ROOT / "struclift" / "utils" / "module_b_v3_ranker.py"
RANKER_SPEC = importlib.util.spec_from_file_location("module_b_v3_ranker_standalone", RANKER_PATH)
if RANKER_SPEC is None or RANKER_SPEC.loader is None:
    raise RuntimeError(f"cannot load {RANKER_PATH}")
RANKER_MODULE = importlib.util.module_from_spec(RANKER_SPEC)
sys.modules[RANKER_SPEC.name] = RANKER_MODULE
RANKER_SPEC.loader.exec_module(RANKER_MODULE)

C_AUDIT_PATH = ROOT / "scripts" / "audit_module_b_v3_c_label_constructability.py"
C_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "module_b_v3_c_label_constructability_for_second_version",
    C_AUDIT_PATH,
)
if C_AUDIT_SPEC is None or C_AUDIT_SPEC.loader is None:
    raise RuntimeError(f"cannot load {C_AUDIT_PATH}")
C_AUDIT_MODULE = importlib.util.module_from_spec(C_AUDIT_SPEC)
sys.modules[C_AUDIT_SPEC.name] = C_AUDIT_MODULE
C_AUDIT_SPEC.loader.exec_module(C_AUDIT_MODULE)


analyse_cfg = MVP_MODULE.analyse_cfg
candidate_outline = MVP_MODULE.candidate_outline
explain_candidate_edges = MVP_MODULE.explain_candidate_edges
generate_candidates = MVP_MODULE.generate_candidates
render_candidate = MVP_MODULE.render_candidate
select_candidate = MVP_MODULE.select_candidate
validate_candidate = MVP_MODULE.validate_candidate

cfg_features = RANKER_MODULE.cfg_features
candidate_features = RANKER_MODULE.candidate_features
slot_grounding_pointers = RANKER_MODULE.slot_grounding_pointers
train_pairwise_ranker = RANKER_MODULE.train_pairwise_ranker
select_with_ranker = RANKER_MODULE.select_with_ranker

audit_candidate = C_AUDIT_MODULE.audit_candidate
slots_from_candidate = C_AUDIT_MODULE.slots_from_candidate
slots_from_teacher_record = C_AUDIT_MODULE.slots_from_teacher_record


def _safe_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        value = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def read_source_by_line(path: Path, wanted: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    wanted_set = {int(x) for x in wanted if int(x) > 0}
    out: Dict[int, Dict[str, Any]] = {}
    if not wanted_set:
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no not in wanted_set:
                continue
            line = line.strip()
            if line:
                out[line_no] = json.loads(line)
            if len(out) >= len(wanted_set):
                break
    return out


def read_jsonl_rows(path: Path, limit: int = 0) -> List[Tuple[int, Dict[str, Any]]]:
    return [(line_no, row) for line_no, row in iter_jsonl(path, limit)]


def _row_input_line(row: Dict[str, Any]) -> int:
    for key in ("input_line", "source_input_line", "original_input_line", "line_no"):
        try:
            value = int(row.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return int(value)
    return 0


def _simple_func_name(row: Optional[Dict[str, Any]]) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("func_name") or row.get("function") or row.get("name") or "")


def _names_compatible(row: Optional[Dict[str, Any]], teacher: Optional[Dict[str, Any]]) -> bool:
    rn = _simple_func_name(row)
    tn = _simple_func_name(teacher)
    return bool(not rn or not tn or rn == tn)


def _join_keys(row: Optional[Dict[str, Any]], teacher: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    base = row if isinstance(row, dict) else {}
    teach = teacher if isinstance(teacher, dict) else {}
    keys: List[Tuple[str, str]] = []
    for field in ("function_id", "func_id", "sample_id"):
        value = str(base.get(field) or teach.get(field) or "")
        if value:
            keys.append((field, value))
    func = str(base.get("func_name") or base.get("function") or teach.get("func_name") or teach.get("function") or "")
    src = str(base.get("source_path") or base.get("src_path") or teach.get("source_path") or teach.get("src_path") or "")
    binary = str(base.get("binary_path") or base.get("bin_path") or teach.get("binary_path") or teach.get("bin_path") or "")
    if func and binary:
        keys.append(("func_binary", f"{func}\0{binary}"))
    if func and src:
        keys.append(("func_source", f"{func}\0{src}"))
    if func:
        keys.append(("func", func))
    return keys


def _unique_key_index(rows: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for _line_no, row in rows:
        for key in _join_keys(row):
            buckets.setdefault(key, []).append(row)
    return {key: vals[0] for key, vals in buckets.items() if len(vals) == 1}


def _source_row_by_input_field(rows: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    seen: set[int] = set()
    for _line_no, row in rows:
        line = _row_input_line(row)
        if line <= 0:
            continue
        if line in seen:
            out.pop(line, None)
        else:
            seen.add(line)
            out[line] = row
    return out


def _pick_joined_source_row(
    *,
    idx: int,
    teacher: Dict[str, Any],
    physical_line: int,
    source_by_physical_line: Dict[int, Dict[str, Any]],
    source_by_input_field: Dict[int, Dict[str, Any]],
    source_by_unique_key: Dict[Tuple[str, str], Dict[str, Any]],
    sequential_source_rows: Sequence[Tuple[int, Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    wanted_line = _row_input_line(teacher) or int(physical_line)
    candidates: List[Tuple[str, Optional[Dict[str, Any]]]] = [
        ("teacher_input_line_as_source_physical_line", source_by_physical_line.get(wanted_line)),
        ("source_row_input_line_field", source_by_input_field.get(wanted_line)),
    ]
    for key in _join_keys(None, teacher):
        candidates.append((f"unique_key:{key[0]}", source_by_unique_key.get(key)))
    if idx < len(sequential_source_rows):
        candidates.append(("sequential_compact_order", sequential_source_rows[idx][1]))
    for method, row in candidates:
        if row is not None and _names_compatible(row, teacher):
            return row, method
    return None, "missing_or_name_mismatch"


def prepare_rows_with_diagnostics(
    source_jsonl: Path,
    teacher_jsonl: Path,
    max_examples: int,
) -> Tuple[List[Tuple[int, Dict[str, Any], Dict[str, Any]]], Dict[str, Any]]:
    teacher_items = read_jsonl_rows(teacher_jsonl, max_examples)
    wanted_lines = [_row_input_line(t) or line_no for line_no, t in teacher_items]
    source_by_physical_line = read_source_by_line(source_jsonl, wanted_lines)
    sequential_source_rows = read_jsonl_rows(source_jsonl, max_examples)
    source_by_input_field = _source_row_by_input_field(sequential_source_rows)
    source_by_unique_key = _unique_key_index(sequential_source_rows)

    rows: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    method_counts: Counter[str] = Counter()
    missing_preview: List[Dict[str, Any]] = []
    for idx, (teacher_physical_line, teacher) in enumerate(teacher_items):
        source_row, method = _pick_joined_source_row(
            idx=idx,
            teacher=teacher,
            physical_line=teacher_physical_line,
            source_by_physical_line=source_by_physical_line,
            source_by_input_field=source_by_input_field,
            source_by_unique_key=source_by_unique_key,
            sequential_source_rows=sequential_source_rows,
        )
        method_counts[method] += 1
        if source_row is None:
            if len(missing_preview) < 20:
                missing_preview.append(
                    {
                        "teacher_physical_line": int(teacher_physical_line),
                        "teacher_input_line": int(_row_input_line(teacher) or 0),
                        "teacher_func": _simple_func_name(teacher),
                    }
                )
            continue
        rows.append((int(_row_input_line(teacher) or teacher_physical_line), source_row, teacher))

    diagnostics = {
        "teacher_rows_seen": int(len(teacher_items)),
        "source_sequential_rows_seen": int(len(sequential_source_rows)),
        "source_physical_line_matches_loaded": int(len(source_by_physical_line)),
        "matched_rows": int(len(rows)),
        "missing_rows": int(len(teacher_items) - len(rows)),
        "match_method_dist": sorted(method_counts.items()),
        "missing_preview": missing_preview,
    }
    return rows, diagnostics


def teacher_outline(teacher: Optional[Dict[str, Any]]) -> List[Tuple[str, int]]:
    if not teacher:
        return []
    pairs = teacher.get("teacher_outline_pairs")
    if isinstance(pairs, list):
        out: List[Tuple[str, int]] = []
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((str(item[0]), int(item[1])))
                except Exception:
                    continue
        return out
    controls = teacher.get("controls") if isinstance(teacher.get("controls"), list) else []
    return [
        (str(c.get("kind") or "unknown"), int(c.get("depth") or 0))
        for c in controls
        if str(c.get("kind") or "") in {"if", "loop", "switch"}
    ]


def teacher_slot_kindseq(teacher: Optional[Dict[str, Any]]) -> List[str]:
    if not teacher:
        return []
    slots = teacher.get("slots") if isinstance(teacher.get("slots"), list) else []
    out: List[str] = []
    for slot in slots:
        if not bool(slot.get("trainable", True)):
            continue
        raw = str(slot.get("slot_type") or "")
        if raw == "FILL_COND":
            out.append("COND")
        elif raw == "FILL_BLOCK":
            out.append("BLOCK")
        elif raw == "FILL_BRANCH":
            out.append("BRANCH")
    return out


def pred_slot_kindseq(slots: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for slot in slots:
        raw = str(slot.get("slot_type") or "")
        if raw in {"COND", "BLOCK"}:
            out.append(raw)
        elif raw == "UNSTRUCTURED_REGION":
            out.append("BLOCK")
        elif raw == "CASE":
            out.append("BRANCH")
        elif raw in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
            out.append("BRANCH")
    return out


def decl_from_teacher(teacher: Optional[Dict[str, Any]]) -> str:
    text = str((teacher or {}).get("teacher_skeleton") or "")
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.endswith("{"):
        first = first[:-1].rstrip()
    if first and "(" in first and ")" in first:
        return first
    return "int func(...)"


def normalize_depths(outline: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not outline:
        return []
    levels = {d: i for i, d in enumerate(sorted({int(d) for _k, d in outline}))}
    return [(str(k), int(levels[int(d)])) for k, d in outline]


def kindseq(outline: Sequence[Tuple[str, int]]) -> List[str]:
    return [str(k) for k, _d in outline]


def lcs_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return int(prev[-1])


def seq_f1(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    hit = lcs_len(pred, gold)
    precision = hit / max(1, len(pred))
    recall = hit / max(1, len(gold))
    return float(2.0 * precision * recall / max(1e-8, precision + recall))


def count_similarity(pred_n: int, gold_n: int) -> float:
    if pred_n == gold_n == 0:
        return 1.0
    return max(0.0, 1.0 - abs(int(pred_n) - int(gold_n)) / max(1, int(gold_n)))


def _coerce_edge_index(raw: Any) -> set[Tuple[int, int]]:
    edges: set[Tuple[int, int]] = set()
    if isinstance(raw, list) and len(raw) == 2:
        srcs = raw[0] if isinstance(raw[0], list) else []
        dsts = raw[1] if isinstance(raw[1], list) else []
        for s, d in zip(srcs, dsts):
            try:
                edges.add((int(s), int(d)))
            except Exception:
                continue
    return edges


def _bb_to_source_statements(
    row: Dict[str, Any],
    *,
    n_bb: int,
    topk: int = 3,
    min_frac: float = 0.5,
    min_weight: float = 1e-8,
) -> List[set[int]]:
    n_src = len(row.get("src_stmt_lengths") or [])
    matrix = row.get("alignment_matrix")
    out: List[set[int]] = [set() for _ in range(int(n_bb))]
    if not isinstance(matrix, list) or not matrix or n_src <= 0:
        return out
    for bb in range(min(int(n_bb), len(matrix))):
        vals = matrix[bb]
        if not isinstance(vals, list) or not vals:
            continue
        pairs: List[Tuple[float, int]] = []
        for j, val in enumerate(vals[:n_src]):
            weight = _safe_float(val)
            if weight > 0.0:
                pairs.append((weight, int(j)))
        if not pairs:
            continue
        pairs.sort(reverse=True)
        peak = pairs[0][0]
        keep_threshold = max(float(min_weight), float(peak) * float(min_frac))
        kept = [j for weight, j in pairs if weight >= keep_threshold]
        if topk > 0:
            kept = kept[: int(topk)]
        out[bb].update(kept)
    return out


def _project_bin_edges(edges: Iterable[Tuple[int, int]], bb_to_src: Sequence[set[int]]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for b0, b1 in edges:
        if not (0 <= int(b0) < len(bb_to_src) and 0 <= int(b1) < len(bb_to_src)):
            continue
        for s0 in bb_to_src[int(b0)]:
            for s1 in bb_to_src[int(b1)]:
                if int(s0) != int(s1):
                    out.add((int(s0), int(s1)))
    return out


def _prf(pred: set[Tuple[int, int]], gold: set[Tuple[int, int]]) -> Dict[str, float]:
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(pred & gold)
    precision = hit / max(1, len(pred))
    recall = hit / max(1, len(gold))
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


def source_cfg_candidate_score(
    row: Dict[str, Any],
    cand_rec: Dict[str, Any],
    *,
    n_bb: int,
) -> Dict[str, Any]:
    gold = _coerce_edge_index(row.get("src_edge_index"))
    bb_to_src = _bb_to_source_statements(row, n_bb=int(n_bb))
    aligned_src = set().union(*bb_to_src) if bb_to_src else set()
    edge_audit = cand_rec.get("edge_audit") or {}
    cats = edge_audit.get("edge_category_edges") or edge_audit.get("category_edges") or {}
    if not gold or not aligned_src or not isinstance(cats, dict):
        return {
            "src_cfg_available": False,
            "src_cfg_score": 0.0,
            "src_cfg_struct_f1": 0.0,
            "src_cfg_struct_recall": 0.0,
            "src_cfg_all_f1": 0.0,
            "src_cfg_all_recall": 0.0,
            "structured_src_stmt_coverage": 0.0,
        }

    def edges_for(keys: Sequence[str]) -> set[Tuple[int, int]]:
        out: set[Tuple[int, int]] = set()
        for key in keys:
            raw = cats.get(key) or []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            out.add((int(item[0]), int(item[1])))
                        except Exception:
                            continue
        return out

    structured_bin = edges_for([
        "structured_semantic",
        "explicit_structured",
        "linear_slot",
        "interval_internal",
    ])
    fallback_bin = edges_for(["fallback_internal"])
    escape_bin = edges_for(["goto_slot", "break_slot", "continue_slot", "early_exit_slot", "cleanup_exit_slot"])
    all_bin = structured_bin | fallback_bin | escape_bin
    all_prf = _prf(_project_bin_edges(all_bin, bb_to_src), gold)
    struct_projected = _project_bin_edges(structured_bin, bb_to_src)
    struct_prf = _prf(struct_projected, gold)

    structured_bbs: set[int] = set()
    fallback_bbs: set[int] = set()
    escape_bbs: set[int] = set()
    for slot in cand_rec.get("slots") or []:
        typ = str(slot.get("slot_type") or "")
        bbs = [int(x) for x in (slot.get("primary_bbs") or []) + (slot.get("context_bbs") or [])]
        if typ == "UNSTRUCTURED_REGION":
            fallback_bbs.update(bbs)
        elif typ in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
            escape_bbs.update(bbs)
        else:
            structured_bbs.update(bbs)

    def project_bbs(bbs: Iterable[int]) -> set[int]:
        out: set[int] = set()
        for bb in bbs:
            if 0 <= int(bb) < len(bb_to_src):
                out.update(int(x) for x in bb_to_src[int(bb)])
        return out

    n_aligned = max(1, len(aligned_src))
    structured_src_cov = len(project_bbs(structured_bbs - fallback_bbs)) / n_aligned
    fallback_src_cov = len(project_bbs(fallback_bbs)) / n_aligned
    escape_src_cov = len(project_bbs(escape_bbs)) / n_aligned
    src_cfg_score = (
        0.44 * float(struct_prf["recall"])
        + 0.26 * float(struct_prf["f1"])
        + 0.18 * float(structured_src_cov)
        + 0.08 * float(all_prf["recall"])
        - 0.08 * float(fallback_src_cov)
        - 0.06 * float(escape_src_cov)
    )
    src_cfg_score = max(0.0, min(1.0, src_cfg_score))
    return {
        "src_cfg_available": True,
        "src_cfg_score": round(float(src_cfg_score), 6),
        "src_cfg_struct_precision": struct_prf["precision"],
        "src_cfg_struct_recall": struct_prf["recall"],
        "src_cfg_struct_f1": struct_prf["f1"],
        "src_cfg_all_precision": all_prf["precision"],
        "src_cfg_all_recall": all_prf["recall"],
        "src_cfg_all_f1": all_prf["f1"],
        "structured_src_stmt_coverage": round(float(structured_src_cov), 6),
        "fallback_src_stmt_coverage": round(float(fallback_src_cov), 6),
        "escape_src_stmt_coverage": round(float(escape_src_cov), 6),
    }


CONTROL_LIKE_STMT_RE = re.compile(
    r"^\s*(?:case\b|default\s*:|if\s*\(|else\b|for\s*\(|while\s*\(|do\b|switch\s*\(|"
    r"break\s*;|continue\s*;|goto\b)"
)


def modulec_constructability_score(
    cand_rec: Dict[str, Any],
    *,
    teacher: Optional[Dict[str, Any]],
    source_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score whether a strict-valid B candidate is usable as Module-C supervision.

    This is a preference target for L_rank, not a correctness validator.  The
    validator still owns CFG safety; this score only says whether the skeleton
    is likely source-like / C-fillable enough to build high-confidence labels.
    """
    teacher_slots = slots_from_teacher_record(teacher or {})
    audit = audit_candidate(
        pred_slots=slots_from_candidate(cand_rec),
        teacher_slots=teacher_slots,
        validator_report=cand_rec.get("validator_report"),
        outline_equal=(cand_rec.get("source_style") or {}).get("outline_equal"),
    )
    pred_slots = [s for s in slots_from_candidate(cand_rec) if s.kind in {"COND", "BLOCK", "BRANCH"}]
    unsafe_count = int(len(audit.get("unsafe_nonempty_pred_slots") or []))
    missing_count = int(len(audit.get("missing_teacher_primary_bbs") or []))
    pred_count = max(1, int(audit.get("pred_fill_slot_count") or len(pred_slots) or 1))
    teacher_primary = int(len(audit.get("teacher_primary_bbs") or []))
    covered_teacher = int(len(audit.get("covered_teacher_primary_bbs") or []))
    matched_teacher = int(len(audit.get("matched_teacher_slot_ids") or []))
    teacher_slots_count = max(1, len([s for s in teacher_slots if s.trainable]))
    labelable_ratio = max(0.0, 1.0 - unsafe_count / pred_count)
    teacher_primary_coverage = (
        1.0 if teacher_primary <= 0 else min(1.0, covered_teacher / max(1, teacher_primary))
    )
    teacher_slot_match_ratio = min(1.0, matched_teacher / teacher_slots_count)
    cond_total = int(audit.get("pred_cond_count") or 0)
    teacher_cond_count = int(audit.get("teacher_cond_count") or 0)
    cond_count_match = bool(cond_total == teacher_cond_count)
    cond_slot_match_ratio = (
        1.0
        if max(cond_total, teacher_cond_count) <= 0
        else min(cond_total, teacher_cond_count) / max(cond_total, teacher_cond_count)
    )

    slots = cand_rec.get("slots") or []
    block_slots = [
        s for s in slots
        if str(s.get("slot_type") or "") in {"BLOCK", "UNSTRUCTURED_REGION", "CASE"}
    ]
    escape_slots = [
        s for s in slots
        if str(s.get("slot_type") or "") in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}
    ]
    switch_like = bool(
        any(str(k) == "switch" for k, _d in (cand_rec.get("outline") or []))
        or "switch" in str(cand_rec.get("name") or "")
    )
    switch_case_slot_ready = bool(
        not switch_like
        or any("switch_case" in str(s.get("role") or "") for s in block_slots)
    )
    # Until CASE_SLOT is fully rendered, switch case bodies are usable for smoke
    # tests but unsafe for raw source-line labels unless marked explicitly.
    explicit_case_slot_ready = bool(
        not switch_like
        or any(str(s.get("slot_type") or "") == "CASE" for s in slots)
    )
    case_slot_gap = 0 if explicit_case_slot_ready else 1

    edge_audit = cand_rec.get("edge_audit") or {}
    edge_counts = edge_audit.get("edge_category_counts") or {}
    structured_case_edges = int((edge_audit.get("structured_semantic_reasons") or {}).get("switch_case_edge") or 0)
    switch_edge_ready = bool(not switch_like or structured_case_edges > 0 or int(edge_counts.get("explicit_structured") or 0) > 0)

    report = cand_rec.get("validator_report") or {}
    goto_count = int(report.get("goto_count") or 0)
    early_exit_count = int(report.get("early_exit_count") or 0)
    break_count = int(report.get("break_count") or 0)
    continue_count = int(report.get("continue_count") or 0)
    fallback_bb = int(report.get("fallback_bb_count") or 0)
    fallback_region = int(report.get("fallback_region_count") or 0)
    source_style = cand_rec.get("source_style") or {}
    outline_f1 = float(source_style.get("outline_lcs_f1") or 0.0)
    kind_f1 = float(source_style.get("kind_lcs_f1") or 0.0)
    slot_kind_f1 = float(source_style.get("slot_kind_lcs_f1") or 0.0)
    outline_gap_abs = abs(int(source_style.get("outline_count_gap") or 0))
    ast_source_like_score = (
        0.42 * outline_f1
        + 0.30 * kind_f1
        + 0.18 * slot_kind_f1
        + 0.10 * max(0.0, 1.0 - min(1.0, outline_gap_abs / 3.0))
    )
    source_cfg = source_cfg or {}
    if source_cfg.get("src_cfg_available"):
        source_like_score = (
            0.50 * float(source_cfg.get("src_cfg_score") or 0.0)
            + 0.20 * float(source_cfg.get("src_cfg_struct_recall") or 0.0)
            + 0.15 * float(source_cfg.get("src_cfg_struct_f1") or 0.0)
            + 0.15 * ast_source_like_score
        )
    else:
        source_like_score = ast_source_like_score

    branch_conflict_penalty = unsafe_count + missing_count
    escape_template_ready = bool(all(str(s.get("slot_type") or "") != "GOTO" for s in escape_slots) or goto_count <= 1)
    teacher_render_structural_ready = bool(
        audit.get("safe_for_modulec")
        and switch_case_slot_ready
        and switch_edge_ready
        and goto_count <= 2
        and fallback_bb <= max(1, len(block_slots))
    )
    c_fillable_score = (
        0.36 * labelable_ratio
        + 0.28 * teacher_primary_coverage
        + 0.18 * teacher_slot_match_ratio
        + 0.06 * cond_slot_match_ratio
        + 0.10 * float(switch_case_slot_ready)
        + 0.08 * float(switch_edge_ready)
        - 0.08 * min(1.0, branch_conflict_penalty / 4.0)
        - 0.05 * min(1.0, goto_count / 3.0)
        - 0.04 * min(1.0, fallback_region / 3.0)
        - 0.05 * float(case_slot_gap)
    )
    c_fillable_score = max(0.0, min(1.0, c_fillable_score))
    second_version_ready = bool(
        audit.get("safe_for_modulec")
        and teacher_render_structural_ready
        and cond_slot_match_ratio >= 0.98
        and labelable_ratio >= 0.98
        and teacher_primary_coverage >= 0.98
        and source_like_score >= 0.45
    )
    return {
        "safe_for_modulec": bool(audit.get("safe_for_modulec")),
        "pred_slots_labelable": bool(audit.get("pred_slots_labelable")),
        "teacher_primary_coverage_ok": bool(audit.get("teacher_primary_coverage_ok")),
        "labelable_ratio": round(float(labelable_ratio), 6),
        "teacher_primary_coverage": round(float(teacher_primary_coverage), 6),
        "teacher_slot_match_ratio": round(float(teacher_slot_match_ratio), 6),
        "cond_slot_match_ratio": round(float(cond_slot_match_ratio), 6),
        "cond_count_match": bool(cond_count_match),
        "unsafe_nonempty_slot_count": int(unsafe_count),
        "missing_teacher_primary_count": int(missing_count),
        "source_like_score": round(float(source_like_score), 6),
        "ast_source_like_score": round(float(ast_source_like_score), 6),
        "c_fillable_score": round(float(c_fillable_score), 6),
        "teacher_render_structural_ready": bool(teacher_render_structural_ready),
        "branch_label_conflict_free": bool(unsafe_count == 0 and missing_count == 0),
        "switch_like": bool(switch_like),
        "switch_case_slot_ready": bool(switch_case_slot_ready),
        "explicit_case_slot_ready": bool(explicit_case_slot_ready),
        "case_slot_gap": int(case_slot_gap),
        "switch_edge_ready": bool(switch_edge_ready),
        "escape_template_ready": bool(escape_template_ready),
        "break_continue_explicit_count": int(break_count + continue_count),
        "early_exit_explicit_count": int(early_exit_count),
        "goto_count": int(goto_count),
        "fallback_bb_count": int(fallback_bb),
        "second_version_ready": bool(second_version_ready),
    }


def edge_explanation_audit(cfg: Any, cand: Any) -> Dict[str, Any]:
    core = explain_candidate_edges(cfg, cand)
    n_edges = max(1, len(cfg.edges))
    counts = dict(core.get("counts") or {})
    explained = len(cfg.edges) - int(counts.get("uncovered") or 0)
    return {
        "edge_count": int(len(cfg.edges)),
        "edge_category_counts": counts,
        "edge_category_edges": dict(core.get("category_edges") or {}),
        "edge_category_preview": dict(core.get("preview") or {}),
        "structured_semantic_reasons": dict(core.get("semantic_reasons") or {}),
        "strict_edge_preservation_ok": int(counts.get("uncovered") or 0) == 0,
        "edge_preservation_ratio": round(float(explained / n_edges), 6),
        "interval_edge_ratio": round(float(int(counts.get("interval_internal") or 0) / n_edges), 6),
        "structured_semantic_edge_ratio": round(
            float(int(counts.get("structured_semantic") or 0) / n_edges),
            6,
        ),
        "goto_edge_ratio": round(float(int(counts.get("goto_slot") or 0) / n_edges), 6),
        "break_edge_ratio": round(float(int(counts.get("break_slot") or 0) / n_edges), 6),
        "continue_edge_ratio": round(float(int(counts.get("continue_slot") or 0) / n_edges), 6),
        "fallback_edge_ratio": round(float(int(counts.get("fallback_internal") or 0) / n_edges), 6),
    }


def outline_metrics(pred: Sequence[Tuple[str, int]], gold: Sequence[Tuple[str, int]]) -> Dict[str, Any]:
    pn = normalize_depths(pred)
    gn = normalize_depths(gold)
    pk = kindseq(pred)
    gk = kindseq(gold)
    p_depth = max((d for _k, d in pn), default=-1) + 1
    g_depth = max((d for _k, d in gn), default=-1) + 1
    return {
        "has_teacher_outline": bool(gold),
        "outline_equal": bool(gold) and pn == gn,
        "kindseq_equal": bool(gold) and pk == gk,
        "outline_lcs_f1": round(seq_f1(pn, gn), 6),
        "kind_lcs_f1": round(seq_f1(pk, gk), 6),
        "control_count_gap": int(len(pred) - len(gold)),
        "max_depth_gap": int(p_depth - g_depth),
        "pred_outline": [[k, d] for k, d in pred],
        "teacher_outline": [[k, d] for k, d in gold],
    }


def source_style_score(
    cand_rec: Dict[str, Any],
    *,
    t_outline: Sequence[Tuple[str, int]],
    t_slot_kind: Sequence[str],
) -> Dict[str, Any]:
    report = cand_rec.get("validator_report") or {}
    slots = cand_rec.get("slots") or []
    pred_outline = [(str(k), int(d)) for k, d in cand_rec.get("outline") or []]
    m = outline_metrics(pred_outline, t_outline)
    p_slot_kind = pred_slot_kindseq(slots)
    slot_kind_f1 = seq_f1(p_slot_kind, t_slot_kind)
    control_count_score = count_similarity(len(pred_outline), len(t_outline))
    valid = bool(report.get("valid"))
    fallback_bb = int(report.get("fallback_bb_count") or 0)
    goto = int(report.get("goto_count") or 0)
    slot_count = int(report.get("slot_count") or len(slots))
    pointer_acc = float((cand_rec.get("grounding") or {}).get("slot_grounding_pointer_acc") or 0.0)
    safety = 1.0 if valid else -1.0
    compact_penalty = min(0.35, 0.010 * fallback_bb + 0.009 * goto + 0.0015 * slot_count)
    quality = (
        0.38 * float(m["outline_lcs_f1"])
        + 0.22 * float(m["kind_lcs_f1"])
        + 0.12 * control_count_score
        + 0.10 * slot_kind_f1
        + 0.08 * pointer_acc
        + 0.10 * safety
        - compact_penalty
    )
    return {
        "quality": round(float(quality), 6),
        "outline_lcs_f1": float(m["outline_lcs_f1"]),
        "kind_lcs_f1": float(m["kind_lcs_f1"]),
        "slot_kind_lcs_f1": round(float(slot_kind_f1), 6),
        "control_count_score": round(float(control_count_score), 6),
        "outline_equal": bool(m["outline_equal"]),
        "kindseq_equal": bool(m["kindseq_equal"]),
        "slot_kindseq_equal": bool(t_slot_kind) and p_slot_kind == list(t_slot_kind),
        "outline_count_gap": int(m["control_count_gap"]),
    }


def is_strict_valid_candidate(cand_rec: Dict[str, Any]) -> bool:
    report = cand_rec.get("validator_report") or {}
    edge = cand_rec.get("edge_audit") or {}
    return bool(
        report.get("valid")
        and report.get("bb_coverage_ok")
        and report.get("unique_primary_ok")
        and report.get("slot_grounding_ok")
        and report.get("parse_success")
        and not (report.get("fatal_errors") or [])
        and (
            edge.get("strict_edge_preservation_ok")
            or report.get("strict_edge_preservation_ok")
        )
    )


def document_quality_components(cand_rec: Dict[str, Any], *, n_bb: int) -> Dict[str, Any]:
    """Document-aligned candidate quality for Module-B v3.

    This intentionally does not use source-AST outline similarity.  The
    algorithm document defines Module B as binary CFG/SSA -> verified skeleton,
    so the ranking target rewards validator correctness, CFG/SSA/provenance
    grounding, and structured coverage while penalizing fallback/goto debt.
    """
    report = cand_rec.get("validator_report") or {}
    features = cand_rec.get("features") or {}
    grounding = cand_rec.get("grounding") or {}
    edge = cand_rec.get("edge_audit") or {}
    denom_bb = max(1, int(n_bb or 0))
    hard_valid = is_strict_valid_candidate(cand_rec)
    fallback_bb = int(report.get("fallback_bb_count") or 0)
    fallback_region_count = int(report.get("fallback_region_count") or 0)
    goto_count = int(report.get("goto_count") or 0)
    break_count = int(report.get("break_count") or 0)
    continue_count = int(report.get("continue_count") or 0)
    early_exit_count = int(report.get("early_exit_count") or 0)
    cleanup_exit_count = int(report.get("cleanup_exit_count") or 0)
    escape_count = int(
        report.get("escape_count")
        or (goto_count + break_count + continue_count + early_exit_count + cleanup_exit_count)
    )
    slot_count = int(report.get("slot_count") or len(cand_rec.get("slots") or []))
    slot_locality_score = _safe_float(
        report.get("slot_locality_score"),
        _safe_float(features.get("slot_locality_score"), 1.0),
    )
    empty_control_count = int(report.get("empty_control_count") or features.get("empty_control_count") or 0)
    empty_loop_count = int(report.get("empty_loop_count") or features.get("empty_loop_count") or 0)
    empty_if_count = int(report.get("empty_if_count") or features.get("empty_if_count") or 0)
    empty_switch_count = int(report.get("empty_switch_count") or features.get("empty_switch_count") or 0)
    empty_control_with_context_count = int(
        report.get("empty_control_with_context_count")
        or features.get("empty_control_with_context_count")
        or 0
    )
    control_body_total_bb_count = int(
        report.get("control_body_total_bb_count")
        or features.get("control_body_total_bb_count")
        or 0
    )
    control_body_nested_bb_count = int(
        report.get("control_body_nested_bb_count")
        or features.get("control_body_nested_bb_count")
        or 0
    )
    body_bb_outside_control_count = int(
        report.get("body_bb_outside_control_count")
        or features.get("body_bb_outside_control_count")
        or 0
    )
    fallback_ratio = float(fallback_bb) / denom_bb
    structured_bb_ratio = _safe_float(
        features.get("structured_bb_ratio"),
        _safe_float(report.get("structured_bb_count")) / denom_bb,
    )
    structured_control_ratio = _safe_float(features.get("structured_control_ratio"))
    edge_semantic_ratio = _safe_float(features.get("edge_semantic_ratio"))
    edge_explicit_ratio = _safe_float(features.get("edge_explicit_ratio"))
    edge_interval_ratio = _safe_float(features.get("edge_interval_ratio"))
    edge_linear_ratio = _safe_float(features.get("edge_linear_ratio"))
    edge_goto_ratio = _safe_float(features.get("edge_goto_ratio"))
    edge_break_ratio = _safe_float(features.get("edge_break_ratio"))
    edge_continue_ratio = _safe_float(features.get("edge_continue_ratio"))
    edge_early_exit_ratio = _safe_float(features.get("edge_early_exit_ratio"))
    edge_fallback_ratio = _safe_float(features.get("edge_fallback_ratio"))
    edge_uncovered_ratio = _safe_float(features.get("edge_uncovered_ratio"))
    slot_pointer_acc = _safe_float(
        grounding.get("slot_grounding_pointer_acc"),
        _safe_float(features.get("slot_pointer_acc")),
    )
    cond_pointer_acc = _safe_float(grounding.get("cond_pointer_acc"), _safe_float(features.get("cond_pointer_acc")))
    block_pointer_acc = _safe_float(grounding.get("block_pointer_acc"), _safe_float(features.get("block_pointer_acc")))
    goto_pointer_acc = _safe_float(grounding.get("goto_pointer_acc"), _safe_float(features.get("goto_pointer_acc")))
    branch_header_cond_slot_coverage = _safe_float(
        features.get("branch_header_cond_slot_coverage"),
        1.0,
    )
    branch_header_cond_slot_gap = _safe_float(
        features.get("branch_header_cond_slot_gap"),
        max(0.0, 1.0 - branch_header_cond_slot_coverage),
    )
    condition_slot_grounded = _safe_float(
        grounding.get("condition_slot_grounded_rate"),
        _safe_float(features.get("condition_slot_grounded_rate")),
    )
    condition_ssa_rate = _safe_float(
        grounding.get("condition_ssa_available_rate"),
        _safe_float(features.get("condition_ssa_available_rate")),
    )
    real_condition_ssa_rate = _safe_float(
        grounding.get("real_condition_ssa_available_rate"),
        _safe_float(features.get("real_condition_ssa_available_rate")),
    )
    branch_cov = _safe_float(
        grounding.get("branch_provenance_coverage"),
        _safe_float(features.get("branch_provenance_coverage")),
    )
    condition_branch_meta = _safe_float(
        grounding.get("condition_branch_metadata_rate"),
        _safe_float(features.get("condition_branch_metadata_rate")),
    )
    if_ground = _safe_float(grounding.get("if_cond_grounded_rate"), _safe_float(features.get("if_cond_grounded_rate")))
    loop_ground = _safe_float(
        grounding.get("loop_cond_grounded_rate"),
        _safe_float(features.get("loop_cond_grounded_rate")),
    )
    switch_ground = _safe_float(
        grounding.get("switch_cond_grounded_rate"),
        _safe_float(features.get("switch_cond_grounded_rate")),
    )
    polarity_available = _safe_float(
        grounding.get("condition_polarity_available_rate"),
        _safe_float(features.get("condition_polarity_available_rate")),
    )
    strict_edge = bool(
        report.get("strict_edge_preservation_ok")
        or edge.get("strict_edge_preservation_ok")
        or features.get("strict_edge_preservation_ok")
    )
    edge_preserved = bool(
        report.get("edge_preservation_ok")
        or strict_edge
        or features.get("edge_preservation_ok")
    )
    parse_success = bool(report.get("parse_success") or features.get("parse_success"))
    slot_grounding_ok = bool(report.get("slot_grounding_ok") or features.get("slot_grounding_ok"))
    cost_feasible = bool(hard_valid and fallback_ratio <= 0.30 and goto_count <= 2 and escape_count <= 4)
    grounding_ready = bool(
        slot_pointer_acc >= 0.98
        and branch_header_cond_slot_coverage >= 0.90
        and branch_cov >= 0.90
        and condition_slot_grounded >= 0.90
    )
    structured_ready = bool(structured_bb_ratio >= 0.60 and fallback_ratio <= 0.30)
    slot_locality_ready = bool(
        slot_locality_score >= 0.98
        and empty_control_with_context_count == 0
        and body_bb_outside_control_count == 0
    )
    doc_quality_ready = bool(
        hard_valid
        and structured_ready
        and grounding_ready
        and slot_locality_ready
        and goto_count <= 2
        and escape_count <= 4
    )
    doc_mvp_ready = bool(hard_valid)
    if hard_valid:
        rank_quality = (
            1000.0
            + 28.0 * structured_bb_ratio
            + 6.0 * structured_control_ratio
            + 8.0 * edge_semantic_ratio
            + 3.0 * edge_explicit_ratio
            + 4.0 * edge_interval_ratio
            + 1.5 * edge_linear_ratio
            + 10.0 * slot_pointer_acc
            + 8.0 * branch_header_cond_slot_coverage
            + 5.0 * cond_pointer_acc
            + 3.0 * block_pointer_acc
            + 1.0 * goto_pointer_acc
            + 7.0 * condition_slot_grounded
            + 4.0 * branch_cov
            + 3.0 * condition_branch_meta
            + 2.0 * condition_ssa_rate
            + 2.0 * real_condition_ssa_rate
            + 1.0 * if_ground
            + 1.0 * loop_ground
            + 1.0 * switch_ground
            + 1.5 * polarity_available
            + 2.0 * float(strict_edge)
            + 1.0 * float(edge_preserved)
            + 0.5 * float(parse_success)
            + 0.5 * float(slot_grounding_ok)
            + 8.0 * slot_locality_score
            - 24.0 * fallback_ratio
            - 2.5 * float(fallback_region_count)
            - 1.8 * float(goto_count)
            - 0.35 * float(break_count)
            - 0.20 * float(continue_count)
            - 0.45 * float(early_exit_count)
            - 0.35 * float(cleanup_exit_count)
            - 5.0 * edge_goto_ratio
            - 1.2 * edge_break_ratio
            - 0.8 * edge_continue_ratio
            - 1.0 * edge_early_exit_ratio
            - 3.0 * edge_fallback_ratio
            - 20.0 * edge_uncovered_ratio
            - 12.0 * branch_header_cond_slot_gap
            - 0.35 * float(slot_count) / denom_bb
            - 20.0 * float(empty_control_with_context_count)
            - 4.0 * float(empty_control_count)
            - 3.0 * float(body_bb_outside_control_count)
        )
    else:
        # Invalid structures stay far below every strict-valid candidate.
        rank_quality = (
            -1000.0
            + 0.5 * float(bool(report.get("valid")))
            + 0.5 * float(bool(report.get("bb_coverage_ok")))
            + 0.5 * float(edge_preserved)
            - 20.0 * edge_uncovered_ratio
            - 5.0 * fallback_ratio
        )
    return {
        "target_policy": "document_cfg_ssa_quality_base_v2",
        "rank_quality": round(float(rank_quality), 6),
        "doc_quality_score": round(float(rank_quality), 6),
        "hard_valid": bool(hard_valid),
        "doc_mvp_ready": bool(doc_mvp_ready),
        "doc_quality_ready": bool(doc_quality_ready),
        "decompile_ready_proxy": bool(doc_quality_ready),
        "cost_feasible": bool(cost_feasible),
        "grounding_ready": bool(grounding_ready),
        "structured_ready": bool(structured_ready),
        "fallback_bb_ratio": round(float(fallback_ratio), 6),
        "fallback_bb_count": int(fallback_bb),
        "fallback_region_count": int(fallback_region_count),
        "goto_count": int(goto_count),
        "break_count": int(break_count),
        "continue_count": int(continue_count),
        "early_exit_count": int(early_exit_count),
        "cleanup_exit_count": int(cleanup_exit_count),
        "escape_count": int(escape_count),
        "slot_count": int(slot_count),
        "slot_locality_score": round(float(slot_locality_score), 6),
        "slot_locality_ready": bool(slot_locality_ready),
        "empty_control_count": int(empty_control_count),
        "empty_loop_count": int(empty_loop_count),
        "empty_if_count": int(empty_if_count),
        "empty_switch_count": int(empty_switch_count),
        "empty_control_with_context_count": int(empty_control_with_context_count),
        "control_body_total_bb_count": int(control_body_total_bb_count),
        "control_body_nested_bb_count": int(control_body_nested_bb_count),
        "body_bb_outside_control_count": int(body_bb_outside_control_count),
        "structured_bb_ratio": round(float(structured_bb_ratio), 6),
        "structured_control_ratio": round(float(structured_control_ratio), 6),
        "edge_semantic_ratio": round(float(edge_semantic_ratio), 6),
        "edge_explicit_ratio": round(float(edge_explicit_ratio), 6),
        "edge_interval_ratio": round(float(edge_interval_ratio), 6),
        "edge_linear_ratio": round(float(edge_linear_ratio), 6),
        "edge_goto_ratio": round(float(edge_goto_ratio), 6),
        "edge_break_ratio": round(float(edge_break_ratio), 6),
        "edge_continue_ratio": round(float(edge_continue_ratio), 6),
        "edge_early_exit_ratio": round(float(edge_early_exit_ratio), 6),
        "edge_fallback_ratio": round(float(edge_fallback_ratio), 6),
        "edge_uncovered_ratio": round(float(edge_uncovered_ratio), 6),
        "strict_edge_preservation_ok": bool(strict_edge),
        "edge_preservation_ok": bool(edge_preserved),
        "parse_success": bool(parse_success),
        "slot_grounding_ok": bool(slot_grounding_ok),
        "slot_grounding_pointer_acc": round(float(slot_pointer_acc), 6),
        "cond_pointer_acc": round(float(cond_pointer_acc), 6),
        "block_pointer_acc": round(float(block_pointer_acc), 6),
        "goto_pointer_acc": round(float(goto_pointer_acc), 6),
        "condition_slot_grounded_rate": round(float(condition_slot_grounded), 6),
        "branch_header_cond_slot_coverage": round(float(branch_header_cond_slot_coverage), 6),
        "branch_header_cond_slot_gap": round(float(branch_header_cond_slot_gap), 6),
        "condition_ssa_available_rate": round(float(condition_ssa_rate), 6),
        "real_condition_ssa_available_rate": round(float(real_condition_ssa_rate), 6),
        "branch_provenance_coverage": round(float(branch_cov), 6),
        "condition_branch_metadata_rate": round(float(condition_branch_meta), 6),
        "condition_polarity_available_rate": round(float(polarity_available), 6),
        "if_cond_grounded_rate": round(float(if_ground), 6),
        "loop_cond_grounded_rate": round(float(loop_ground), 6),
        "switch_cond_grounded_rate": round(float(switch_ground), 6),
    }


def source_soft_ready_from_metrics(
    *,
    hard_valid: bool,
    doc_quality_ready: bool,
    src_cfg: Dict[str, Any],
    outline_f1: float,
    kind_f1: float,
    count_gap: int,
) -> bool:
    if not (hard_valid and doc_quality_ready):
        return False
    if bool(src_cfg.get("src_cfg_available")):
        return bool(
            float(src_cfg.get("src_cfg_score") or 0.0) >= 0.68
            and float(src_cfg.get("src_cfg_struct_recall") or 0.0) >= 0.75
        )
    return bool(outline_f1 >= 0.80 and kind_f1 >= 0.80 and abs(int(count_gap)) <= 1)


def modulec_high_confidence_label_ready(
    *,
    doc_quality_ready: bool,
    c_fill: Dict[str, Any],
    source_like_score_min: float = 0.45,
) -> bool:
    return bool(
        doc_quality_ready
        and c_fill.get("safe_for_modulec")
        and c_fill.get("teacher_render_structural_ready")
        and c_fill.get("cond_count_match")
        and c_fill.get("branch_label_conflict_free")
        and float(c_fill.get("cond_slot_match_ratio") or 0.0) >= 0.98
        and float(c_fill.get("labelable_ratio") or 0.0) >= 0.98
        and float(c_fill.get("teacher_primary_coverage") or 0.0) >= 0.98
        and float(c_fill.get("source_like_score") or 0.0) >= float(source_like_score_min)
        and int(c_fill.get("unsafe_nonempty_slot_count") or 0) == 0
        and int(c_fill.get("missing_teacher_primary_count") or 0) == 0
    )


def structural_source_like_target(
    *,
    out: Dict[str, Any],
    src_cfg: Dict[str, Any],
    outline_f1: float,
    kind_f1: float,
    count_gap: int,
) -> Dict[str, Any]:
    """Teacher-derived structural L_rank target, independent of slot labels.

    This is the supervision signal for Module-B v3's second-version ranker:
    among already-safe candidates, prefer the control-flow skeleton that looks
    most like the source skeleton.  Slot/teacher label constructability is kept
    in separate Module-C fields and must not decide whether a skeleton is a
    structural positive for L_rank.
    """
    hard_valid = bool(out.get("hard_valid"))
    if not hard_valid:
        return {
            "score": 0.0,
            "ast_score": 0.0,
            "tier": 0,
            "status": "invalid_not_trainable",
            "trainable": False,
            "confidence": "low",
            "exact": False,
            "soft": False,
            "target_policy": "structural_source_like_l_rank_v1",
        }
    doc_ready = bool(out.get("doc_quality_ready"))
    control_count_score = max(0.0, 1.0 - min(1.0, abs(int(count_gap)) / 3.0))
    outline_exact = bool(outline_f1 >= 0.999 and abs(int(count_gap)) == 0)
    kind_exact = bool(kind_f1 >= 0.999)
    fallback_ratio = float(out.get("fallback_bb_ratio") or 0.0)
    goto_count = int(out.get("goto_count") or 0)
    escape_count = int(out.get("escape_count") or 0)
    ast_score = (
        0.46 * float(outline_f1)
        + 0.34 * float(kind_f1)
        + 0.14 * float(control_count_score)
        + 0.04 * float(outline_exact)
        + 0.02 * float(kind_exact)
    )
    if src_cfg.get("src_cfg_available"):
        score = (
            0.54 * ast_score
            + 0.22 * float(src_cfg.get("src_cfg_score") or 0.0)
            + 0.14 * float(src_cfg.get("src_cfg_struct_recall") or 0.0)
            + 0.10 * float(src_cfg.get("src_cfg_struct_f1") or 0.0)
        )
    else:
        score = ast_score
    score -= 0.10 * min(1.0, fallback_ratio)
    score -= 0.03 * min(1.0, goto_count / 3.0)
    score -= 0.02 * min(1.0, escape_count / 5.0)
    score = max(0.0, min(1.0, score))

    source_ready = source_soft_ready_from_metrics(
        hard_valid=hard_valid,
        doc_quality_ready=doc_ready,
        src_cfg=src_cfg,
        outline_f1=outline_f1,
        kind_f1=kind_f1,
        count_gap=count_gap,
    )

    high_conf = bool(
        doc_ready
        and score >= 0.74
        and (outline_f1 >= 0.78 or kind_f1 >= 0.86 or outline_exact or kind_exact)
        and fallback_ratio <= 0.20
    )
    medium_conf = bool(
        doc_ready
        and (
            source_ready
            or (
                score >= 0.62
                and (outline_f1 >= 0.64 or kind_f1 >= 0.74)
                and fallback_ratio <= 0.30
            )
        )
    )
    if high_conf:
        tier, status, trainable = 6, "high_confidence_structural_source_like", True
    elif medium_conf:
        tier, status, trainable = 5, "medium_confidence_structural_source_like", True
    elif source_ready:
        tier, status, trainable = 4, "low_margin_structural_source_like", False
    elif doc_ready:
        tier, status, trainable = 3, "binary_doc_quality_only", False
    elif out.get("cost_feasible"):
        tier, status, trainable = 2, "binary_cost_feasible_only", False
    else:
        tier, status, trainable = 1, "strict_valid_low_confidence", False
    return {
        "score": round(float(score), 6),
        "ast_score": round(float(ast_score), 6),
        "tier": int(tier),
        "status": str(status),
        "trainable": bool(trainable),
        "confidence": "high" if int(tier) >= 6 else ("medium" if int(tier) >= 5 else ("low" if int(tier) < 4 else "low_margin")),
        "exact": bool(outline_exact or kind_exact),
        "soft": bool(source_ready or medium_conf or high_conf),
        "target_policy": "structural_source_like_l_rank_v1",
    }


def rank_label_tier(
    *,
    out: Dict[str, Any],
    c_fill: Dict[str, Any],
    src_cfg: Dict[str, Any],
    outline_f1: float,
    kind_f1: float,
    count_gap: int,
) -> Tuple[int, str, bool]:
    structural = structural_source_like_target(
        out=out,
        src_cfg=src_cfg,
        outline_f1=outline_f1,
        kind_f1=kind_f1,
        count_gap=count_gap,
    )
    return (
        int(structural.get("tier") or 0),
        str(structural.get("status") or "strict_valid_low_confidence"),
        bool(structural.get("trainable")),
    )


def rank_target_score(
    cand_rec: Dict[str, Any],
    *,
    n_bb: int,
    teacher: Optional[Dict[str, Any]],
    source_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Second-version teacher target for L_rank.

    CFG/SSA safety is still the hard gate.  Source-like and Module-C-fillable
    signals are preference targets used only after the candidate is strict-valid.
    """
    source_style = cand_rec.get("source_style") or {}
    outline_f1 = float(source_style.get("outline_lcs_f1") or 0.0)
    kind_f1 = float(source_style.get("kind_lcs_f1") or 0.0)
    slot_kind_f1 = float(source_style.get("slot_kind_lcs_f1") or 0.0)
    count_gap = abs(int(source_style.get("outline_count_gap") or 0))
    out = document_quality_components(cand_rec, n_bb=n_bb)
    src_cfg = source_cfg_candidate_score(source_row or {}, cand_rec, n_bb=n_bb)
    c_fill = modulec_constructability_score(cand_rec, teacher=teacher, source_cfg=src_cfg)
    structural = structural_source_like_target(
        out=out,
        src_cfg=src_cfg,
        outline_f1=outline_f1,
        kind_f1=kind_f1,
        count_gap=count_gap,
    )
    base_quality = float(out.get("rank_quality") or -1000.0)
    modulec_trainable = modulec_high_confidence_label_ready(
        doc_quality_ready=bool(out.get("doc_quality_ready")),
        c_fill=c_fill,
    )
    if out.get("hard_valid"):
        tier = int(structural.get("tier") or 0)
        label_status = str(structural.get("status") or "strict_valid_low_confidence")
        rank_trainable = bool(structural.get("trainable"))
        preference_score = (
            base_quality
            + 44.0 * float(structural.get("score") or 0.0)
            + 18.0 * float(structural.get("ast_score") or 0.0)
            + 10.0 * float(outline_f1)
            + 8.0 * float(kind_f1)
            + 4.0 * float(bool(structural.get("exact")))
            + 4.0 * float(src_cfg.get("src_cfg_score") or 0.0)
            + 3.0 * float(src_cfg.get("src_cfg_struct_recall") or 0.0)
            + 2.0 * float(src_cfg.get("src_cfg_struct_f1") or 0.0)
            - 6.0 * float(src_cfg.get("fallback_src_stmt_coverage") or 0.0)
            - 4.0 * float(src_cfg.get("escape_src_stmt_coverage") or 0.0)
        )
        rank_quality = 10000.0 * float(tier) + preference_score
    else:
        tier = int(structural.get("tier") or 0)
        label_status = str(structural.get("status") or "invalid_not_trainable")
        rank_trainable = bool(structural.get("trainable"))
        rank_quality = base_quality
    out.update({
        "target_policy": "structural_source_like_l_rank_v1",
        "rank_quality": round(float(rank_quality), 6),
        "doc_quality_score": round(float(base_quality), 6),
        "rank_label_tier": int(tier),
        "rank_label_status": str(label_status),
        "rank_trainable": bool(rank_trainable),
        "rank_label_confidence": structural.get("confidence"),
        "structural_source_like_score": structural.get("score"),
        "structural_ast_source_like_score": structural.get("ast_score"),
        "structural_source_like": bool(structural.get("soft")),
        "structural_source_like_exact": bool(structural.get("exact")),
        "structural_rank_trainable": bool(rank_trainable),
        "structural_rank_label_tier": int(tier),
        "structural_rank_label_status": str(label_status),
        "structural_target_policy": structural.get("target_policy"),
        "source_ast_metrics_used_for_rank": True,
        "source_ast_metrics_role": "structural_L_rank_label_only_after_strict_validator",
        "source_cfg_metrics_used_for_rank": bool(src_cfg.get("src_cfg_available")),
        "source_cfg_score": src_cfg.get("src_cfg_score"),
        "source_cfg_struct_precision": src_cfg.get("src_cfg_struct_precision"),
        "source_cfg_struct_recall": src_cfg.get("src_cfg_struct_recall"),
        "source_cfg_struct_f1": src_cfg.get("src_cfg_struct_f1"),
        "source_cfg_all_precision": src_cfg.get("src_cfg_all_precision"),
        "source_cfg_all_recall": src_cfg.get("src_cfg_all_recall"),
        "source_cfg_all_f1": src_cfg.get("src_cfg_all_f1"),
        "source_cfg_structured_src_stmt_coverage": src_cfg.get("structured_src_stmt_coverage"),
        "source_cfg_fallback_src_stmt_coverage": src_cfg.get("fallback_src_stmt_coverage"),
        "source_cfg_escape_src_stmt_coverage": src_cfg.get("escape_src_stmt_coverage"),
        "source_soft_recovered": source_soft_ready_from_metrics(
            hard_valid=bool(out.get("hard_valid")),
            doc_quality_ready=bool(out.get("doc_quality_ready")),
            src_cfg=src_cfg,
            outline_f1=outline_f1,
            kind_f1=kind_f1,
            count_gap=count_gap,
        ),
        "outline_lcs_f1": round(float(outline_f1), 6),
        "kind_lcs_f1": round(float(kind_f1), 6),
        "slot_kind_lcs_f1": round(float(slot_kind_f1), 6),
        "outline_count_gap_abs": int(count_gap),
        "outline_equal": bool(source_style.get("outline_equal")),
        "kindseq_equal": bool(source_style.get("kindseq_equal")),
        "slot_kindseq_equal": bool(source_style.get("slot_kindseq_equal")),
        "modulec_safe_for_rank": bool(c_fill.get("safe_for_modulec")),
        "modulec_pred_slots_labelable": bool(c_fill.get("pred_slots_labelable")),
        "modulec_teacher_primary_coverage_ok": bool(c_fill.get("teacher_primary_coverage_ok")),
        "modulec_labelable_ratio": c_fill.get("labelable_ratio"),
        "modulec_teacher_primary_coverage": c_fill.get("teacher_primary_coverage"),
        "modulec_teacher_slot_match_ratio": c_fill.get("teacher_slot_match_ratio"),
        "modulec_cond_slot_match_ratio": c_fill.get("cond_slot_match_ratio"),
        "modulec_cond_count_match": bool(c_fill.get("cond_count_match")),
        "modulec_c_fillable_score": c_fill.get("c_fillable_score"),
        "modulec_source_like_score": c_fill.get("source_like_score"),
        "modulec_ast_source_like_score": c_fill.get("ast_source_like_score"),
        "slot_locality_score": out.get("slot_locality_score"),
        "slot_locality_ready": bool(out.get("slot_locality_ready")),
        "empty_control_count": int(out.get("empty_control_count") or 0),
        "empty_loop_count": int(out.get("empty_loop_count") or 0),
        "empty_if_count": int(out.get("empty_if_count") or 0),
        "empty_switch_count": int(out.get("empty_switch_count") or 0),
        "empty_control_with_context_count": int(out.get("empty_control_with_context_count") or 0),
        "control_body_total_bb_count": int(out.get("control_body_total_bb_count") or 0),
        "control_body_nested_bb_count": int(out.get("control_body_nested_bb_count") or 0),
        "body_bb_outside_control_count": int(out.get("body_bb_outside_control_count") or 0),
        "modulec_label_ready": bool(modulec_trainable),
        "modulec_trainable": bool(modulec_trainable),
        "modulec_teacher_render_structural_ready": bool(c_fill.get("teacher_render_structural_ready")),
        "modulec_branch_label_conflict_free": bool(c_fill.get("branch_label_conflict_free")),
        "modulec_switch_like": bool(c_fill.get("switch_like")),
        "modulec_switch_case_slot_ready": bool(c_fill.get("switch_case_slot_ready")),
        "modulec_explicit_case_slot_ready": bool(c_fill.get("explicit_case_slot_ready")),
        "modulec_case_slot_gap": int(c_fill.get("case_slot_gap") or 0),
        "modulec_switch_edge_ready": bool(c_fill.get("switch_edge_ready")),
        "modulec_escape_template_ready": bool(c_fill.get("escape_template_ready")),
        "modulec_unsafe_nonempty_slot_count": int(c_fill.get("unsafe_nonempty_slot_count") or 0),
        "modulec_missing_teacher_primary_count": int(c_fill.get("missing_teacher_primary_count") or 0),
        "second_version_ready": bool(rank_trainable),
        "c_fillable_ranker_ready": bool(c_fill.get("second_version_ready")),
    })
    return out


def rank_sort_key(cand_rec: Dict[str, Any]) -> Tuple[float, int]:
    target = cand_rec.get("rank_target") or {}
    quality = target.get("rank_quality")
    return (
        float(quality) if isinstance(quality, (int, float)) else -1e9,
        -int(cand_rec.get("candidate_id") or 0),
    )


def choose_eval_candidate(
    *,
    args: argparse.Namespace,
    rec: Dict[str, Any],
    ranker_selected: Optional[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]],
    oracle: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    mode = str(getattr(args, "selection_mode", "ranker") or "ranker")
    candidates = rec.get("candidates") or []
    fallback = next(
        (c for c in candidates if str(c.get("name") or "") == "full_fallback"),
        candidates[0] if candidates else None,
    )
    if mode == "oracle":
        return oracle or baseline or ranker_selected or fallback
    if mode == "deterministic":
        return baseline or ranker_selected or fallback
    return ranker_selected or baseline or fallback


def _candidate_family_name(cand_rec: Dict[str, Any]) -> str:
    name = str(cand_rec.get("name") or "")
    if name == "full_fallback":
        return "fallback"
    if name == "linear_goto":
        return "linear_goto"
    if "sese_interval" in name or "interval" in name:
        return "sese_interval"
    if "switch_fanin" in name:
        return "switch_fanin"
    if "switch_chain" in name:
        return "switch_chain"
    if "mixed" in name:
        return "mixed_if_policy"
    if "depth" in name:
        return "depth_limited"
    if "skip_tiny" in name or "skip_small" in name:
        return "sparse_control"
    if "wide" in name:
        return "wide_region"
    if "then_only" in name:
        return "then_only"
    if "else_only" in name:
        return "else_only"
    if "recursive" in name:
        return "recursive"
    if "shallow" in name:
        return "shallow"
    return "other"


def _compact_signature(cand_rec: Dict[str, Any]) -> Tuple[Any, ...]:
    report = cand_rec.get("validator_report") or {}
    slots = cand_rec.get("slots") or []
    outline = tuple((str(k), int(d)) for k, d in (cand_rec.get("outline") or []))
    slot_types = tuple(str(s.get("slot_type") or "") for s in slots)
    return (
        outline,
        slot_types,
        int(report.get("fallback_bb_count") or 0),
        int(report.get("goto_count") or 0),
    )


def _compact_score(cand_rec: Dict[str, Any]) -> Tuple[Any, ...]:
    report = cand_rec.get("validator_report") or {}
    features = cand_rec.get("features") or {}
    hard = bool((cand_rec.get("rank_target") or {}).get("hard_valid"))
    name = str(cand_rec.get("name") or "")
    core_bonus = 1 if name in {
        "cfg_shallow_mvp",
        "cfg_recursive_mvp",
        "cfg_shallow_then_only",
        "cfg_shallow_else_only",
        "cfg_recursive_then_only",
        "cfg_recursive_else_only",
        "cfg_recursive_sese_interval",
        "cfg_sese_interval_seq",
        "cfg_switch_fanin",
        "linear_goto",
        "full_fallback",
    } or name.startswith("cfg_switch_fanin") else 0
    return (
        int(hard),
        core_bonus,
        float(features.get("slot_locality_score") or report.get("slot_locality_score") or 0.0),
        -int(report.get("empty_control_with_context_count") or features.get("empty_control_with_context_count") or 0),
        -int(report.get("body_bb_outside_control_count") or features.get("body_bb_outside_control_count") or 0),
        -int(report.get("empty_control_count") or features.get("empty_control_count") or 0),
        -float(features.get("fallback_bb_ratio") or 0.0),
        -int(report.get("goto_count") or 0),
        float(features.get("structured_bb_ratio") or 0.0),
        float(features.get("switch_semantic_ratio") or 0.0),
        float(features.get("edge_semantic_ratio") or 0.0),
        int(report.get("structured_control_count") or 0),
        -int(report.get("slot_count") or 0),
        -int(cand_rec.get("candidate_id") or 0),
    )


def compact_candidate_records(
    candidates: Sequence[Dict[str, Any]],
    *,
    max_candidates_final: int = 0,
    candidate_family_cap: int = 0,
    candidate_signature_cap: int = 0,
    mixed_candidate_cap: int = 0,
    always_keep_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before = len(candidates)
    if before <= 0:
        return [], {
            "enabled": False,
            "before": 0,
            "after": 0,
            "dropped": 0,
            "reason": "empty_candidate_pool",
        }
    max_final = int(max_candidates_final or 0)
    family_cap = int(candidate_family_cap or 0)
    signature_cap = int(candidate_signature_cap or 0)
    mixed_cap = int(mixed_candidate_cap or 0)
    if max_final <= 0 and family_cap <= 0 and signature_cap <= 0 and mixed_cap <= 0:
        return list(candidates), {
            "enabled": False,
            "before": before,
            "after": before,
            "dropped": 0,
            "reason": "no_compaction_requested",
        }

    core_names = {
        "cfg_shallow_mvp",
        "cfg_recursive_mvp",
        "cfg_shallow_then_only",
        "cfg_shallow_else_only",
        "cfg_recursive_then_only",
        "cfg_recursive_else_only",
        "cfg_recursive_sese_interval",
        "cfg_sese_interval_seq",
        "linear_goto",
        "full_fallback",
    }
    keep_names = set(core_names)
    keep_names.update(str(x) for x in (always_keep_names or []) if x)
    by_id = {int(c.get("candidate_id") or 0): c for c in candidates}
    kept_ids: List[int] = []

    def add(cand: Dict[str, Any]) -> None:
        cid = int(cand.get("candidate_id") or 0)
        if cid not in kept_ids:
            kept_ids.append(cid)

    for cand in candidates:
        if str(cand.get("name") or "") in keep_names:
            add(cand)
    if max_final > 0 and len(kept_ids) > max_final:
        full_fallback_ids = [
            cid for cid in kept_ids
            if str(by_id[cid].get("name") or "") == "full_fallback"
        ][:1]
        other_protected = [
            cid for cid in kept_ids
            if str(by_id[cid].get("name") or "") == "linear_goto"
        ][: max(0, max_final - len(full_fallback_ids))]
        protected = full_fallback_ids + other_protected
        rest = [
            cid for cid in kept_ids
            if cid not in protected
        ]
        rest = sorted(rest, key=lambda cid: _compact_score(by_id[cid]), reverse=True)
        kept_ids = protected + rest[: max(0, max_final - len(protected))]

    family_counts: Dict[str, int] = {}
    signature_counts: Dict[Tuple[Any, ...], int] = {}
    mixed_count = 0
    for cid in list(kept_ids):
        cand = by_id[cid]
        fam = _candidate_family_name(cand)
        sig = _compact_signature(cand)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        signature_counts[sig] = signature_counts.get(sig, 0) + 1
        if fam == "mixed_if_policy":
            mixed_count += 1

    ranked = sorted(candidates, key=_compact_score, reverse=True)
    for cand in ranked:
        if max_final > 0 and len(kept_ids) >= max_final:
            break
        cid = int(cand.get("candidate_id") or 0)
        if cid in kept_ids:
            continue
        fam = _candidate_family_name(cand)
        sig = _compact_signature(cand)
        if family_cap > 0 and family_counts.get(fam, 0) >= family_cap:
            continue
        if signature_cap > 0 and signature_counts.get(sig, 0) >= signature_cap:
            continue
        if mixed_cap > 0 and fam == "mixed_if_policy" and mixed_count >= mixed_cap:
            continue
        kept_ids.append(cid)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        signature_counts[sig] = signature_counts.get(sig, 0) + 1
        if fam == "mixed_if_policy":
            mixed_count += 1

    kept = [by_id[cid] for cid in sorted(kept_ids)]
    old_to_new: Dict[int, int] = {}
    for new_id, cand in enumerate(kept):
        old_id = int(cand.get("candidate_id") or 0)
        old_to_new[old_id] = new_id
        cand = dict(cand)
        cand["original_candidate_id"] = old_id
        cand["candidate_id"] = new_id
        kept[new_id] = cand
    family_dist = {}
    for cand in kept:
        fam = _candidate_family_name(cand)
        family_dist[fam] = family_dist.get(fam, 0) + 1
    return kept, {
        "enabled": True,
        "before": int(before),
        "after": int(len(kept)),
        "dropped": int(before - len(kept)),
        "max_candidates_final": int(max_final),
        "candidate_family_cap": int(family_cap),
        "candidate_signature_cap": int(signature_cap),
        "mixed_candidate_cap": int(mixed_cap),
        "family_dist": family_dist,
        "old_to_new_candidate_id": {str(k): int(v) for k, v in old_to_new.items()},
    }


def row_func_name(row: Dict[str, Any], teacher: Optional[Dict[str, Any]]) -> Any:
    return row.get("func_name") or row.get("function") or (teacher or {}).get("func_name")


def prepare_rows(
    source_jsonl: Path,
    teacher_jsonl: Path,
    max_examples: int,
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    rows, _diagnostics = prepare_rows_with_diagnostics(source_jsonl, teacher_jsonl, max_examples)
    return rows


def build_candidate_records(
    *,
    row: Dict[str, Any],
    teacher: Dict[str, Any],
    max_region_nodes: int,
    max_mixed_if_headers: int,
    candidate_beam: int,
    include_skeletons: bool,
    include_switch_chain: bool = False,
    include_sese_interval: bool = True,
    max_candidates_final: int = 0,
    candidate_family_cap: int = 0,
    candidate_signature_cap: int = 0,
    mixed_candidate_cap: int = 0,
) -> Dict[str, Any]:
    cfg = analyse_cfg(row)
    cfg_feat = cfg_features(cfg)
    n_bb = int(cfg_feat.get("n_bb") or 0)
    t_outline = teacher_outline(teacher)
    t_slot_kind = teacher_slot_kindseq(teacher)
    candidate_objs = generate_candidates(
        cfg,
        max_region_nodes=max_region_nodes,
        max_mixed_headers=max_mixed_if_headers,
        candidate_beam=candidate_beam,
        include_switch_chain=include_switch_chain,
        include_sese_interval=include_sese_interval,
    )
    deterministic_cand, deterministic_report, _ = select_candidate(cfg, candidate_objs)
    candidates: List[Dict[str, Any]] = []
    for cid, cand in enumerate(candidate_objs):
        rendered = render_candidate(cand, decl_line=decl_from_teacher(teacher))
        report = validate_candidate(cfg, cand, rendered_text=rendered).asdict()
        slots = [s.asdict() for s in cand.slots]
        edge_audit = edge_explanation_audit(cfg, cand)
        grounding = slot_grounding_pointers(row, slots)
        rec: Dict[str, Any] = {
            "candidate_id": cid,
            "name": cand.name,
            "notes": list(cand.notes),
            "outline": [[k, d] for k, d in candidate_outline(cand)],
            "slots": slots,
            "validator_report": report,
            "edge_audit": edge_audit,
            "grounding": grounding,
        }
        rec["features"] = candidate_features(rec, cfg_feat)
        rec["source_style"] = source_style_score(
            rec,
            t_outline=t_outline,
            t_slot_kind=t_slot_kind,
        )
        rec["rank_target"] = rank_target_score(rec, n_bb=n_bb, teacher=teacher, source_row=row)
        if include_skeletons:
            rec["skeleton"] = rendered
        else:
            rec["skeleton_preview"] = "\n".join(rendered.splitlines()[:16])
        candidates.append(rec)

    candidates, compaction_report = compact_candidate_records(
        candidates,
        max_candidates_final=int(max_candidates_final),
        candidate_family_cap=int(candidate_family_cap),
        candidate_signature_cap=int(candidate_signature_cap),
        mixed_candidate_cap=int(mixed_candidate_cap),
    )

    deterministic_pool = [c for c in candidates if is_strict_valid_candidate(c)] or candidates
    deterministic_rec = max(deterministic_pool, key=_compact_score) if deterministic_pool else None
    deterministic_id = deterministic_rec.get("candidate_id") if deterministic_rec else None
    deterministic_name = deterministic_rec.get("name") if deterministic_rec else deterministic_cand.name
    valid_candidates = [c for c in candidates if bool((c.get("rank_target") or {}).get("hard_valid"))]
    trainable_candidates = [
        c for c in valid_candidates
        if bool((c.get("rank_target") or {}).get("rank_trainable"))
    ]
    cfillable_candidates = [
        c for c in valid_candidates
        if bool((c.get("rank_target") or {}).get("c_fillable_ranker_ready"))
    ]
    modulec_safe_candidates = [
        c for c in valid_candidates
        if bool((c.get("rank_target") or {}).get("modulec_safe_for_rank"))
    ]
    second_version_candidates = [
        c for c in valid_candidates
        if bool((c.get("rank_target") or {}).get("second_version_ready"))
    ]
    oracle_pool = valid_candidates or candidates
    oracle = max(oracle_pool, key=rank_sort_key) if oracle_pool else None
    ranked = sorted(oracle_pool, key=rank_sort_key, reverse=True)
    ranked_ids = [c.get("candidate_id") for c in ranked]
    top16_ids = ranked_ids[:16]
    top32_ids = ranked_ids[:32]
    return {
        "cfg": cfg,
        "cfg_features": cfg_feat,
        "teacher": {
            "outline": [[k, d] for k, d in t_outline],
            "outline_kindseq": kindseq(t_outline),
            "slot_kindseq": t_slot_kind,
        },
        "deterministic_candidate_id": deterministic_id,
        "deterministic_name": deterministic_name,
        "deterministic_validator_report": (
            deterministic_rec.get("validator_report")
            if deterministic_rec is not None
            else deterministic_report.asdict()
        ),
        "oracle_candidate_id": oracle.get("candidate_id") if oracle else None,
        "oracle_name": oracle.get("name") if oracle else None,
        "oracle_selection": (
            "high_confidence_rank_trainable"
            if trainable_candidates
            else ("strict_valid_ready_first" if valid_candidates else "fallback_no_strict_valid")
        ),
        "strict_valid_candidate_count": len(valid_candidates),
        "rank_trainable_candidate_count": len(trainable_candidates),
        "c_fillable_candidate_count": len(cfillable_candidates),
        "modulec_safe_candidate_count": len(modulec_safe_candidates),
        "second_version_candidate_count": len(second_version_candidates),
        "any_rank_trainable_candidate": bool(trainable_candidates),
        "any_c_fillable_candidate": bool(cfillable_candidates),
        "any_modulec_safe_candidate": bool(modulec_safe_candidates),
        "any_second_version_candidate": bool(second_version_candidates),
        "oracle_rank_trainable": bool((oracle.get("rank_target") or {}).get("rank_trainable")) if oracle else False,
        "oracle_c_fillable_ranker_ready": bool((oracle.get("rank_target") or {}).get("c_fillable_ranker_ready")) if oracle else False,
        "oracle_modulec_safe_for_rank": bool((oracle.get("rank_target") or {}).get("modulec_safe_for_rank")) if oracle else False,
        "top3_oracle_candidate_ids": top16_ids[:3],
        "top5_oracle_candidate_ids": top16_ids[:5],
        "top16_oracle_candidate_ids": top16_ids,
        "top32_oracle_candidate_ids": top32_ids,
        "top16_contains_rank_trainable": any(c.get("candidate_id") in top16_ids for c in trainable_candidates),
        "top32_contains_rank_trainable": any(c.get("candidate_id") in top32_ids for c in trainable_candidates),
        "top16_contains_c_fillable": any(c.get("candidate_id") in top16_ids for c in cfillable_candidates),
        "top32_contains_c_fillable": any(c.get("candidate_id") in top32_ids for c in cfillable_candidates),
        "candidate_compaction": compaction_report,
        "include_sese_interval": bool(include_sese_interval),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def selected_metrics(record: Dict[str, Any], cand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if cand is None:
        return {}
    t_outline = [
        (str(k), int(d))
        for k, d in ((record.get("teacher") or {}).get("outline") or [])
    ]
    pred_outline = [(str(k), int(d)) for k, d in cand.get("outline") or []]
    report = cand.get("validator_report") or {}
    grounding = cand.get("grounding") or {}
    rank_target = cand.get("rank_target") or {}
    metrics = outline_metrics(pred_outline, t_outline)
    fallback_ratio = float(report.get("fallback_bb_count") or 0) / max(1, int(record.get("n_bb") or 0))
    hard_valid = bool(rank_target.get("hard_valid", is_strict_valid_candidate(cand)))
    cost_feasible = bool(
        rank_target.get(
            "cost_feasible",
            hard_valid and fallback_ratio <= 0.30 and int(report.get("goto_count") or 0) <= 2,
        )
    )
    source_soft = (
        hard_valid
        and metrics["has_teacher_outline"]
        and float(metrics["outline_lcs_f1"]) >= 0.80
        and float(metrics["kind_lcs_f1"]) >= 0.80
        and abs(int(metrics["control_count_gap"])) <= 1
    )
    doc_quality_ready = bool(
        rank_target.get(
            "doc_quality_ready",
            rank_target.get("decompile_ready_proxy", cost_feasible),
        )
    )
    return {
        "candidate_id": cand.get("candidate_id"),
        "name": cand.get("name"),
        "quality": (cand.get("source_style") or {}).get("quality"),
        "rank_quality": rank_target.get("rank_quality"),
        "target_policy": rank_target.get("target_policy"),
        "rank_label_tier": rank_target.get("rank_label_tier"),
        "rank_label_status": rank_target.get("rank_label_status"),
        "rank_trainable": bool(rank_target.get("rank_trainable", False)),
        "rank_label_confidence": rank_target.get("rank_label_confidence"),
        "doc_quality_score": rank_target.get("doc_quality_score", rank_target.get("rank_quality")),
        "hard_valid": hard_valid,
        "doc_mvp_ready": bool(rank_target.get("doc_mvp_ready", hard_valid)),
        "doc_quality_ready": bool(doc_quality_ready),
        "second_version_ready": bool(rank_target.get("second_version_ready", False)),
        "c_fillable_ranker_ready": bool(rank_target.get("c_fillable_ranker_ready", False)),
        "modulec_safe_for_rank": bool(rank_target.get("modulec_safe_for_rank", False)),
        "modulec_pred_slots_labelable": bool(rank_target.get("modulec_pred_slots_labelable", False)),
        "modulec_teacher_primary_coverage_ok": bool(rank_target.get("modulec_teacher_primary_coverage_ok", False)),
        "modulec_teacher_render_structural_ready": bool(
            rank_target.get("modulec_teacher_render_structural_ready", False)
        ),
        "modulec_branch_label_conflict_free": bool(
            rank_target.get("modulec_branch_label_conflict_free", False)
        ),
        "modulec_switch_case_slot_ready": bool(rank_target.get("modulec_switch_case_slot_ready", False)),
        "modulec_explicit_case_slot_ready": bool(rank_target.get("modulec_explicit_case_slot_ready", False)),
        "modulec_switch_edge_ready": bool(rank_target.get("modulec_switch_edge_ready", False)),
        "modulec_escape_template_ready": bool(rank_target.get("modulec_escape_template_ready", False)),
        "modulec_labelable_ratio": float(rank_target.get("modulec_labelable_ratio") or 0.0),
        "modulec_teacher_primary_coverage": float(rank_target.get("modulec_teacher_primary_coverage") or 0.0),
        "modulec_teacher_slot_match_ratio": float(rank_target.get("modulec_teacher_slot_match_ratio") or 0.0),
        "modulec_cond_slot_match_ratio": float(rank_target.get("modulec_cond_slot_match_ratio") or 0.0),
        "modulec_cond_count_match": bool(rank_target.get("modulec_cond_count_match", False)),
        "modulec_c_fillable_score": float(rank_target.get("modulec_c_fillable_score") or 0.0),
        "modulec_source_like_score": float(rank_target.get("modulec_source_like_score") or 0.0),
        "modulec_ast_source_like_score": float(rank_target.get("modulec_ast_source_like_score") or 0.0),
        "slot_locality_score": float(rank_target.get("slot_locality_score") or 0.0),
        "slot_locality_ready": bool(rank_target.get("slot_locality_ready", False)),
        "empty_control_count": int(rank_target.get("empty_control_count") or 0),
        "empty_loop_count": int(rank_target.get("empty_loop_count") or 0),
        "empty_if_count": int(rank_target.get("empty_if_count") or 0),
        "empty_switch_count": int(rank_target.get("empty_switch_count") or 0),
        "empty_control_with_context_count": int(rank_target.get("empty_control_with_context_count") or 0),
        "control_body_total_bb_count": int(rank_target.get("control_body_total_bb_count") or 0),
        "control_body_nested_bb_count": int(rank_target.get("control_body_nested_bb_count") or 0),
        "body_bb_outside_control_count": int(rank_target.get("body_bb_outside_control_count") or 0),
        "source_cfg_metrics_used_for_rank": bool(rank_target.get("source_cfg_metrics_used_for_rank", False)),
        "source_cfg_score": float(rank_target.get("source_cfg_score") or 0.0),
        "source_cfg_struct_recall": float(rank_target.get("source_cfg_struct_recall") or 0.0),
        "source_cfg_struct_f1": float(rank_target.get("source_cfg_struct_f1") or 0.0),
        "source_cfg_all_recall": float(rank_target.get("source_cfg_all_recall") or 0.0),
        "source_cfg_all_f1": float(rank_target.get("source_cfg_all_f1") or 0.0),
        "source_cfg_structured_src_stmt_coverage": float(rank_target.get("source_cfg_structured_src_stmt_coverage") or 0.0),
        "source_cfg_fallback_src_stmt_coverage": float(rank_target.get("source_cfg_fallback_src_stmt_coverage") or 0.0),
        "source_cfg_escape_src_stmt_coverage": float(rank_target.get("source_cfg_escape_src_stmt_coverage") or 0.0),
        "modulec_unsafe_nonempty_slot_count": int(rank_target.get("modulec_unsafe_nonempty_slot_count") or 0),
        "modulec_missing_teacher_primary_count": int(rank_target.get("modulec_missing_teacher_primary_count") or 0),
        "modulec_case_slot_gap": int(rank_target.get("modulec_case_slot_gap") or 0),
        "cost_feasible": bool(cost_feasible),
        "source_soft_recovered": bool(source_soft),
        "decompile_ready_proxy": bool(doc_quality_ready),
        "outline_equal": bool(metrics["outline_equal"]),
        "kindseq_equal": bool(metrics["kindseq_equal"]),
        "outline_lcs_f1": float(metrics["outline_lcs_f1"]),
        "kind_lcs_f1": float(metrics["kind_lcs_f1"]),
        "control_count_gap": int(metrics["control_count_gap"]),
        "fallback_bb_ratio": round(float(fallback_ratio), 6),
        "fallback_bb_count": int(report.get("fallback_bb_count") or 0),
        "fallback_region_count": int(report.get("fallback_region_count") or 0),
        "goto_count": int(report.get("goto_count") or 0),
        "break_count": int(report.get("break_count") or 0),
        "continue_count": int(report.get("continue_count") or 0),
        "early_exit_count": int(report.get("early_exit_count") or 0),
        "cleanup_exit_count": int(report.get("cleanup_exit_count") or 0),
        "escape_count": int(
            report.get("escape_count")
            or (
                int(report.get("goto_count") or 0)
                + int(report.get("break_count") or 0)
                + int(report.get("continue_count") or 0)
                + int(report.get("early_exit_count") or 0)
                + int(report.get("cleanup_exit_count") or 0)
            )
        ),
        "structured_control_count": int(report.get("structured_control_count") or 0),
        "structured_bb_ratio": float(rank_target.get(
            "structured_bb_ratio",
            float(report.get("structured_bb_count") or 0) / max(1, int(record.get("n_bb") or 0)),
        )),
        "edge_semantic_ratio": float(rank_target.get("edge_semantic_ratio") or 0.0),
        "edge_goto_ratio": float(rank_target.get("edge_goto_ratio") or 0.0),
        "edge_early_exit_ratio": float(rank_target.get("edge_early_exit_ratio") or 0.0),
        "edge_fallback_ratio": float(rank_target.get("edge_fallback_ratio") or 0.0),
        "edge_uncovered_ratio": float(rank_target.get("edge_uncovered_ratio") or 0.0),
        "slot_grounding_pointer_acc": float(grounding.get("slot_grounding_pointer_acc") or 0.0),
        "cond_pointer_acc": float(grounding.get("cond_pointer_acc") or 0.0),
        "cond_polarity_checkable_rate": float(grounding.get("cond_polarity_checkable_rate") or 0.0),
        "block_pointer_acc": float(grounding.get("block_pointer_acc") or 0.0),
        "goto_pointer_acc": float(grounding.get("goto_pointer_acc") or 0.0),
        "condition_polarity_error_proxy": grounding.get("condition_polarity_error_proxy"),
        "condition_polarity_note": grounding.get("condition_polarity_note"),
        "ssa_available": bool(grounding.get("ssa_available")),
        "real_ssa_available": bool(grounding.get("real_ssa_available")),
        "condition_ssa_available_rate": float(grounding.get("condition_ssa_available_rate") or 0.0),
        "real_condition_ssa_available_rate": float(grounding.get("real_condition_ssa_available_rate") or 0.0),
        "branch_provenance_coverage": float(grounding.get("branch_provenance_coverage") or 0.0),
        "condition_branch_metadata_rate": float(grounding.get("condition_branch_metadata_rate") or 0.0),
        "condition_polarity_available_rate": float(grounding.get("condition_polarity_available_rate") or 0.0),
        "condition_slot_grounded_rate": float(grounding.get("condition_slot_grounded_rate") or 0.0),
        "branch_header_cond_slot_coverage": float(
            rank_target.get("branch_header_cond_slot_coverage", 0.0) or 0.0
        ),
        "branch_header_cond_slot_gap": float(
            rank_target.get("branch_header_cond_slot_gap", 0.0) or 0.0
        ),
        "if_cond_grounded_rate": float(grounding.get("if_cond_grounded_rate") or 0.0),
        "loop_cond_grounded_rate": float(grounding.get("loop_cond_grounded_rate") or 0.0),
        "switch_cond_grounded_rate": float(grounding.get("switch_cond_grounded_rate") or 0.0),
    }


def find_candidate(record: Dict[str, Any], candidate_id: Any) -> Optional[Dict[str, Any]]:
    for cand in record.get("candidates") or []:
        if cand.get("candidate_id") == candidate_id:
            return cand
    return None


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / max(1, den):.1f}%)"


def mean_val(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows]
    return float(statistics.mean(vals)) if vals else 0.0


def median_val(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows]
    return float(statistics.median(vals)) if vals else 0.0


def dist(rows: Sequence[Dict[str, Any]], key: str) -> List[Tuple[Any, int]]:
    c = Counter(r.get(key) for r in rows)
    return sorted(c.items(), key=lambda kv: (str(kv[0]), kv[1]))


def contains_kind(record: Dict[str, Any], kind: str) -> bool:
    return any(x[0] == kind for x in (record.get("teacher") or {}).get("outline") or [])


def build_summary(records: Sequence[Dict[str, Any]], args: argparse.Namespace, elapsed: float) -> str:
    n = len(records)
    views = {
        "baseline": [r["baseline_metrics"] for r in records],
        "ranker": [r["ranker_metrics"] for r in records],
        "oracle": [r["oracle_metrics"] for r in records],
    }

    def line_metrics(prefix: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
        out: List[str] = [f"-- {prefix} --"]
        for key in (
            "hard_valid",
            "doc_mvp_ready",
            "doc_quality_ready",
            "second_version_ready",
            "c_fillable_ranker_ready",
            "rank_trainable",
            "modulec_safe_for_rank",
            "modulec_teacher_render_structural_ready",
            "modulec_branch_label_conflict_free",
            "modulec_cond_count_match",
            "cost_feasible",
            "source_soft_recovered",
            "slot_locality_ready",
            "outline_equal",
            "kindseq_equal",
        ):
            out.append(f"{key} = {pct(sum(1 for r in rows if r.get(key)), len(rows))}")
        out.append(
            "source_ast_metrics_used_for_rank = "
            f"{pct(sum(1 for r in rows if r.get('target_policy') == 'document_cfg_ssa_cfillable_quality_v3'), len(rows))}"
        )
        out.append(
            "source_cfg_metrics_used_for_rank = "
            f"{pct(sum(1 for r in rows if r.get('source_cfg_metrics_used_for_rank')), len(rows))}"
        )
        out.append(f"doc_quality_score_mean = {mean_val(rows, 'doc_quality_score'):.4f}")
        out.append(f"source_cfg_score_mean = {mean_val(rows, 'source_cfg_score'):.4f}")
        out.append(f"source_cfg_struct_recall_mean = {mean_val(rows, 'source_cfg_struct_recall'):.4f}")
        out.append(f"source_cfg_struct_f1_mean = {mean_val(rows, 'source_cfg_struct_f1'):.4f}")
        out.append(f"source_cfg_structured_src_stmt_coverage_mean = {mean_val(rows, 'source_cfg_structured_src_stmt_coverage'):.4f}")
        out.append(f"source_cfg_fallback_src_stmt_coverage_mean = {mean_val(rows, 'source_cfg_fallback_src_stmt_coverage'):.4f}")
        out.append(f"source_cfg_escape_src_stmt_coverage_mean = {mean_val(rows, 'source_cfg_escape_src_stmt_coverage'):.4f}")
        out.append(f"modulec_c_fillable_score_mean = {mean_val(rows, 'modulec_c_fillable_score'):.4f}")
        out.append(f"modulec_source_like_score_mean = {mean_val(rows, 'modulec_source_like_score'):.4f}")
        out.append(f"modulec_ast_source_like_score_mean = {mean_val(rows, 'modulec_ast_source_like_score'):.4f}")
        out.append(f"modulec_labelable_ratio_mean = {mean_val(rows, 'modulec_labelable_ratio'):.4f}")
        out.append(f"modulec_teacher_primary_coverage_mean = {mean_val(rows, 'modulec_teacher_primary_coverage'):.4f}")
        out.append(f"modulec_cond_slot_match_ratio_mean = {mean_val(rows, 'modulec_cond_slot_match_ratio'):.4f}")
        out.append(f"modulec_unsafe_nonempty_slot_count_mean = {mean_val(rows, 'modulec_unsafe_nonempty_slot_count'):.4f}")
        out.append(f"modulec_missing_teacher_primary_count_mean = {mean_val(rows, 'modulec_missing_teacher_primary_count'):.4f}")
        out.append(f"modulec_case_slot_gap_mean = {mean_val(rows, 'modulec_case_slot_gap'):.4f}")
        out.append(f"slot_locality_score_mean = {mean_val(rows, 'slot_locality_score'):.4f}")
        out.append(f"empty_control_count_mean = {mean_val(rows, 'empty_control_count'):.4f}")
        out.append(f"empty_control_with_context_count_mean = {mean_val(rows, 'empty_control_with_context_count'):.4f}")
        out.append(f"body_bb_outside_control_count_mean = {mean_val(rows, 'body_bb_outside_control_count'):.4f}")
        out.append(f"structured_bb_ratio_mean = {mean_val(rows, 'structured_bb_ratio'):.4f}")
        out.append(f"edge_semantic_ratio_mean = {mean_val(rows, 'edge_semantic_ratio'):.4f}")
        out.append(f"outline_lcs_f1_mean = {mean_val(rows, 'outline_lcs_f1'):.4f}")
        out.append(f"outline_lcs_f1_median = {median_val(rows, 'outline_lcs_f1'):.4f}")
        out.append(f"kind_lcs_f1_mean = {mean_val(rows, 'kind_lcs_f1'):.4f}")
        out.append(f"fallback_bb_ratio_mean = {mean_val(rows, 'fallback_bb_ratio'):.4f}")
        out.append(f"goto_count_mean = {mean_val(rows, 'goto_count'):.4f}")
        out.append(f"goto_count_dist = {dist(rows, 'goto_count')}")
        out.append(f"selected_name_dist = {Counter(r.get('name') for r in rows).most_common(30)}")
        out.append(f"slot_grounding_pointer_acc_mean = {mean_val(rows, 'slot_grounding_pointer_acc'):.4f}")
        out.append(f"cond_pointer_acc_mean = {mean_val(rows, 'cond_pointer_acc'):.4f}")
        out.append(f"cond_polarity_checkable_rate_mean = {mean_val(rows, 'cond_polarity_checkable_rate'):.4f}")
        out.append(f"real_ssa_available = {pct(sum(1 for r in rows if r.get('real_ssa_available')), len(rows))}")
        out.append(f"condition_ssa_available_rate_mean = {mean_val(rows, 'condition_ssa_available_rate'):.4f}")
        out.append(f"real_condition_ssa_available_rate_mean = {mean_val(rows, 'real_condition_ssa_available_rate'):.4f}")
        out.append(f"branch_provenance_coverage_mean = {mean_val(rows, 'branch_provenance_coverage'):.4f}")
        out.append(f"condition_branch_metadata_rate_mean = {mean_val(rows, 'condition_branch_metadata_rate'):.4f}")
        out.append(f"condition_slot_grounded_rate_mean = {mean_val(rows, 'condition_slot_grounded_rate'):.4f}")
        out.append(f"if_cond_grounded_rate_mean = {mean_val(rows, 'if_cond_grounded_rate'):.4f}")
        out.append(f"loop_cond_grounded_rate_mean = {mean_val(rows, 'loop_cond_grounded_rate'):.4f}")
        out.append(f"switch_cond_grounded_rate_mean = {mean_val(rows, 'switch_cond_grounded_rate'):.4f}")
        return out

    top1_match_oracle = sum(
        1 for r in records
        if r["ranker_metrics"].get("candidate_id") == r["oracle_metrics"].get("candidate_id")
    )
    top3_contains = sum(
        1 for r in records
        if r["ranker_metrics"].get("candidate_id") in (r.get("top3_oracle_candidate_ids") or [])
    )
    top5_contains = sum(
        1 for r in records
        if r["ranker_metrics"].get("candidate_id") in (r.get("top5_oracle_candidate_ids") or [])
    )
    baseline_vs_ranker_doc_ready_delta = sum(
        1 for r in records
        if r["ranker_metrics"].get("doc_quality_ready")
    ) - sum(1 for r in records if r["baseline_metrics"].get("doc_quality_ready"))
    baseline_vs_ranker_goto_delta = (
        mean_val(views["ranker"], "goto_count") - mean_val(views["baseline"], "goto_count")
    )
    baseline_vs_ranker_fallback_delta = (
        mean_val(views["ranker"], "fallback_bb_ratio") - mean_val(views["baseline"], "fallback_bb_ratio")
    )

    lines: List[str] = []
    lines.append("===== MODULE B-v3 SECOND VERSION SMOKE EVAL =====")
    lines.append(f"source_jsonl = {Path(args.source_jsonl).expanduser().resolve()}")
    lines.append(f"teacher_jsonl = {Path(args.teacher_jsonl).expanduser().resolve()}")
    lines.append(f"n = {n}")
    join_diag = getattr(args, "_input_join_diagnostics", {}) or {}
    if join_diag:
        lines.append(
            "input_join = "
            f"matched {join_diag.get('matched_rows')}/{join_diag.get('teacher_rows_seen')} "
            f"missing={join_diag.get('missing_rows')} "
            f"methods={join_diag.get('match_method_dist')}"
        )
    lines.append(f"selection_mode = {str(args.selection_mode)}")
    lines.append(
        "ranker_feature_policy = binary_only_cfg_ssa_candidate_validator_provenance; "
        "source_teacher_metrics_are_L_rank_labels_and_audit_only"
    )
    lines.append(
        "note = candidate ranker only chooses among strict-validator candidates; "
        "validator remains the final safety boundary."
    )
    lines.append(
        "polarity_note = this CFG JSONL lacks reliable true/false edge metadata, "
        "so the script reports condition polarity checkability, not a fake polarity accuracy."
    )
    lines.append("")
    lines.append("-- ranker / oracle headline --")
    lines.append(
        "note = use --selection-mode oracle only to measure candidate-generator upper bound "
        "or export teacher-forced Module-C labels."
    )
    lines.append(f"ranker_top1_matches_oracle = {pct(top1_match_oracle, n)}")
    lines.append(f"ranker_selected_in_oracle_top3 = {pct(top3_contains, n)}")
    lines.append(f"ranker_selected_in_oracle_top5 = {pct(top5_contains, n)}")
    lines.append(f"ranker_doc_quality_ready_delta_vs_baseline = {baseline_vs_ranker_doc_ready_delta:+d}")
    lines.append(f"ranker_goto_mean_delta_vs_baseline = {baseline_vs_ranker_goto_delta:+.4f}")
    lines.append(f"ranker_fallback_ratio_delta_vs_baseline = {baseline_vs_ranker_fallback_delta:+.4f}")
    lines.append(
        "invalid_oracle_labels = "
        f"{sum(1 for r in records if not r['oracle_metrics'].get('hard_valid'))}/{n}"
    )
    lines.append(f"oracle_selection_dist = {Counter(r.get('oracle_selection') for r in records).most_common()}")
    lines.append(f"strict_valid_candidate_count_dist = {sorted(Counter(r.get('strict_valid_candidate_count') for r in records).items())}")
    lines.append(f"rank_trainable_candidate_count_dist = {sorted(Counter(r.get('rank_trainable_candidate_count') for r in records).items())}")
    lines.append(f"c_fillable_candidate_count_dist = {sorted(Counter(r.get('c_fillable_candidate_count') for r in records).items())}")
    lines.append(f"any_rank_trainable_candidate = {pct(sum(1 for r in records if r.get('any_rank_trainable_candidate')), n)}")
    lines.append(f"any_c_fillable_candidate = {pct(sum(1 for r in records if r.get('any_c_fillable_candidate')), n)}")
    lines.append(f"oracle_rank_trainable = {pct(sum(1 for r in records if r.get('oracle_rank_trainable')), n)}")
    lines.append(f"top16_contains_rank_trainable = {pct(sum(1 for r in records if r.get('top16_contains_rank_trainable')), n)}")
    lines.append(f"top32_contains_rank_trainable = {pct(sum(1 for r in records if r.get('top32_contains_rank_trainable')), n)}")
    lines.append(f"candidate_count_dist = {sorted(Counter(r.get('candidate_count') for r in records).items())}")
    g1_rate = sum(1 for r in records if r.get("top16_contains_rank_trainable")) / max(1, n)
    g4_rate = top1_match_oracle / max(1, n)
    g5_rate = sum(1 for r in records if r["ranker_metrics"].get("rank_trainable")) / max(1, n)
    if g1_rate < 0.90:
        stop_go = "STOP_G1_FIX_CANDIDATE_GENERATOR_OR_LABEL_ALIGNMENT"
    elif g4_rate < 0.90 or g5_rate < 0.90:
        stop_go = "STOP_G4_FIX_RANKER_SELECTION"
    else:
        stop_go = "GO_SECOND_VERSION_RANKER_READY_FOR_C_D_SMOKE"
    lines.append(f"stop_go_decision = {stop_go}")
    lines.append("")
    for prefix in ("baseline", "ranker", "oracle"):
        lines.extend(line_metrics(prefix, views[prefix]))
        lines.append("")

    lines.append("-- switch / loop / if buckets, ranker --")
    for kind in ("switch", "loop", "if"):
        bucket = [r for r in records if contains_kind(r, kind)]
        rows = [r["ranker_metrics"] for r in bucket]
        lines.append(
            f"{kind}: n={len(rows)} doc_quality_ready={pct(sum(1 for x in rows if x.get('doc_quality_ready')), len(rows))} "
            f"structured_cov_mean={mean_val(rows, 'structured_bb_ratio'):.4f} "
            f"goto_mean={mean_val(rows, 'goto_count'):.4f}"
        )
    lines.append("")
    lines.append("-- BB buckets, ranker --")
    bucket_defs = [
        ("0-5", lambda nbb: nbb <= 5),
        ("6-10", lambda nbb: 6 <= nbb <= 10),
        ("11-20", lambda nbb: 11 <= nbb <= 20),
        ("21+", lambda nbb: nbb >= 21),
    ]
    for name, fn in bucket_defs:
        bucket = [r for r in records if fn(int(r.get("n_bb") or 0))]
        rows = [r["ranker_metrics"] for r in bucket]
        lines.append(
            f"{name}: n={len(rows)} doc_quality_ready={pct(sum(1 for x in rows if x.get('doc_quality_ready')), len(rows))} "
            f"structured_cov_mean={mean_val(rows, 'structured_bb_ratio'):.4f} "
            f"goto_mean={mean_val(rows, 'goto_count'):.4f} "
            f"fallback_mean={mean_val(rows, 'fallback_bb_ratio'):.4f}"
        )
    lines.append("")
    lines.append(f"-- badcase preview, first {int(args.max_bad_preview)} --")
    bad = sorted(
        records,
        key=lambda r: (
            0 if r["ranker_metrics"].get("doc_quality_ready") else 1,
            -float(r["ranker_metrics"].get("structured_bb_ratio") or 0.0),
            int(r["ranker_metrics"].get("goto_count") or 0),
            float(r["ranker_metrics"].get("fallback_bb_ratio") or 0.0),
        ),
        reverse=True,
    )
    for r in bad[: int(args.max_bad_preview)]:
        b = r["baseline_metrics"]
        m = r["ranker_metrics"]
        o = r["oracle_metrics"]
        lines.append("-" * 120)
        lines.append(
            f"example={r.get('example')} input_line={r.get('input_line')} func={r.get('func_name')} "
            f"n_bb={r.get('n_bb')} candidates={r.get('candidate_count')} "
            f"baseline={b.get('name')} ranker={m.get('name')} oracle={o.get('name')} "
            f"ranker_doc_ready={m.get('doc_quality_ready')} oracle_doc_ready={o.get('doc_quality_ready')} "
            f"ranker_structured={m.get('structured_bb_ratio')} oracle_structured={o.get('structured_bb_ratio')} "
            f"ranker_goto={m.get('goto_count')} oracle_goto={o.get('goto_count')} "
            f"ranker_fb={m.get('fallback_bb_ratio')} oracle_fb={o.get('fallback_bb_ratio')} "
            f"ranker_ground={m.get('slot_grounding_pointer_acc')} "
            f"cond_ssa={m.get('condition_ssa_available_rate')} "
            f"branch_cov={m.get('branch_provenance_coverage')}"
        )
        lines.append(f"teacher_outline={(r.get('teacher') or {}).get('outline')}")
        top = sorted(
            r.get("candidates") or [],
            key=rank_sort_key,
            reverse=True,
        )[:6]
        for cand in top:
            st = cand.get("source_style") or {}
            rt = cand.get("rank_target") or {}
            rep = cand.get("validator_report") or {}
            lines.append(
                f"  cand={cand.get('candidate_id')} name={cand.get('name')} "
                f"ranker_score={cand.get('ranker_score')} rank_q={rt.get('rank_quality')} "
                f"doc_ready={rt.get('doc_quality_ready')} source_q={st.get('quality')} "
                f"structured={rt.get('structured_bb_ratio')} edge_sem={rt.get('edge_semantic_ratio')} "
                f"outline_f1={st.get('outline_lcs_f1')} kind_f1={st.get('kind_lcs_f1')} "
                f"valid={rep.get('valid')} fallback={rep.get('fallback_bb_count')} "
                f"goto={rep.get('goto_count')} outline={cand.get('outline')}"
            )
    lines.append("")
    lines.append(f"elapsed_sec = {elapsed:.1f}")
    return "\n".join(lines) + "\n"


def evaluate(args: argparse.Namespace) -> None:
    source_jsonl = Path(args.source_jsonl).expanduser().resolve()
    teacher_jsonl = Path(args.teacher_jsonl).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_ranker = Path(args.out_ranker).expanduser().resolve() if args.out_ranker else None
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if out_ranker:
        out_ranker.parent.mkdir(parents=True, exist_ok=True)

    raw_rows, input_join_diagnostics = prepare_rows_with_diagnostics(
        source_jsonl,
        teacher_jsonl,
        int(args.max_examples),
    )
    setattr(args, "_input_join_diagnostics", input_join_diagnostics)
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    for example, (line_no, row, teacher) in enumerate(raw_rows):
        rec = build_candidate_records(
            row=row,
            teacher=teacher,
            max_region_nodes=int(args.max_region_nodes),
            max_mixed_if_headers=int(args.max_mixed_if_headers),
            candidate_beam=int(args.candidate_beam),
            include_skeletons=bool(args.include_skeletons),
            include_switch_chain=bool(args.include_switch_chain),
            include_sese_interval=not bool(args.disable_sese_interval),
            max_candidates_final=int(args.max_candidates_final),
            candidate_family_cap=int(args.candidate_family_cap),
            candidate_signature_cap=int(args.candidate_signature_cap),
            mixed_candidate_cap=int(args.mixed_candidate_cap),
        )
        rec.update(
            {
                "example": example,
                "input_line": line_no,
                "func_name": row_func_name(row, teacher),
                "n_bb": int(rec["cfg_features"].get("n_bb") or 0),
                "n_edges": int(rec["cfg_features"].get("n_edges") or 0),
            }
        )
        rec.pop("cfg", None)
        records.append(rec)

    # Leave-one-out ranking is small but honest for the 100-sample smoke.  For a
    # large run this can be disabled with --train-on-all.
    if bool(args.train_on_all):
        ranker = train_pairwise_ranker(
            records,
            epochs=int(args.ranker_epochs),
            lr=float(args.ranker_lr),
            l2=float(args.ranker_l2),
            seed=int(args.seed),
        )
        rankers = [ranker for _ in records]
    else:
        rankers = []
        for i in range(len(records)):
            train_rows = records[:i] + records[i + 1 :]
            rankers.append(
                train_pairwise_ranker(
                    train_rows,
                    epochs=int(args.ranker_epochs),
                    lr=float(args.ranker_lr),
                    l2=float(args.ranker_l2),
                    seed=int(args.seed) + i,
                )
            )
        ranker = train_pairwise_ranker(
            records,
            epochs=int(args.ranker_epochs),
            lr=float(args.ranker_lr),
            l2=float(args.ranker_l2),
            seed=int(args.seed),
        )

    with out_jsonl.open("w", encoding="utf-8") as jf:
        for i, rec in enumerate(records):
            baseline = find_candidate(rec, rec.get("deterministic_candidate_id"))
            oracle = find_candidate(rec, rec.get("oracle_candidate_id"))
            ranker_selected = select_with_ranker(rec.get("candidates") or [], rec.get("cfg_features") or {}, rankers[i])
            selected = choose_eval_candidate(
                args=args,
                rec=rec,
                ranker_selected=ranker_selected,
                baseline=baseline,
                oracle=oracle,
            )
            rec["baseline_metrics"] = selected_metrics(rec, baseline)
            rec["oracle_metrics"] = selected_metrics(rec, oracle)
            rec["binary_ranker_metrics"] = selected_metrics(rec, ranker_selected)
            rec["ranker_metrics"] = selected_metrics(rec, selected)
            rec["rank_supervision"] = {
                "loss": "L_rank",
                "target_policy": (oracle.get("rank_target") or {}).get("target_policy") if oracle else "document_cfg_ssa_cfillable_quality_v3",
                "oracle_selection": rec.get("oracle_selection"),
                "target_candidate_id": rec.get("oracle_candidate_id"),
                "target_name": rec.get("oracle_name"),
                "rank_quality": (oracle.get("rank_target") or {}).get("rank_quality") if oracle else None,
                "source_style_quality_diagnostic": (oracle.get("source_style") or {}).get("quality") if oracle else None,
                "hard_valid": bool((oracle.get("rank_target") or {}).get("hard_valid")) if oracle else False,
                "eval_mode": "train_on_all" if args.train_on_all else "leave_one_out",
                "selection_mode": str(args.selection_mode),
                "binary_only_ranker_candidate_id": ranker_selected.get("candidate_id") if ranker_selected else None,
                "selected_candidate_id": selected.get("candidate_id") if selected else None,
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if out_ranker is not None:
        out_ranker.write_text(json.dumps(ranker.asdict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = build_summary(records, args, time.time() - t0)
    out_txt.write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)
    print(f"saved_txt = {out_txt}", flush=True)
    print(f"saved_jsonl = {out_jsonl}", flush=True)
    if out_ranker is not None:
        print(f"saved_ranker = {out_ranker}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Module-B v3 second-version ranker + grounding")
    ap.add_argument("--source-jsonl", type=Path, required=True)
    ap.add_argument("--teacher-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--out-ranker", type=Path, default=None)
    ap.add_argument("--max-examples", type=int, default=100)
    ap.add_argument("--max-region-nodes", type=int, default=64)
    ap.add_argument("--max-mixed-if-headers", type=int, default=4)
    ap.add_argument("--candidate-beam", type=int, default=160)
    ap.add_argument("--ranker-epochs", type=int, default=80)
    ap.add_argument("--ranker-lr", type=float, default=0.05)
    ap.add_argument("--ranker-l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-on-all", action="store_true")
    ap.add_argument(
        "--selection-mode",
        choices=["ranker", "oracle", "deterministic"],
        default="ranker",
        help=(
            "ranker = binary-only learned selection; oracle = source/teacher-forced upper bound; "
            "deterministic = no learned ranker."
        ),
    )
    ap.add_argument("--max-bad-preview", type=int, default=80)
    ap.add_argument("--include-skeletons", action="store_true")
    ap.add_argument("--include-switch-chain", action="store_true")
    ap.add_argument("--disable-sese-interval", action="store_true")
    ap.add_argument("--max-candidates-final", type=int, default=0)
    ap.add_argument("--candidate-family-cap", type=int, default=0)
    ap.add_argument("--candidate-signature-cap", type=int, default=0)
    ap.add_argument("--mixed-candidate-cap", type=int, default=0)
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

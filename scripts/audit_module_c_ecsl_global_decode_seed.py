#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit ECSL global decode seed tables.

This script performs a deterministic, conservative smoke audit for the global
decoding stage.  It does not run an LLM or a compiler.  It checks whether each
function has at least one candidate per slot, whether local top choices can be
rendered back into the fixed Module-B skeleton, and whether any placeholders
remain.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PLACEHOLDER_RE = re.compile(r"<(?P<tag>[A-Z_]+):(?P<slot_ref>slot_(?P<sid>\d+))>(?P<trailing>\s*;)?")
TIER_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit > 0 and line_no > limit:
                break
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def q(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def candidate_sort_key(cand: Dict[str, Any]) -> Tuple[int, float, str]:
    level = str(q(cand, "confidence", "confidence_level") or "C0")
    return (
        TIER_ORDER.get(level, 0),
        safe_float(q(cand, "confidence", "confidence_weight")),
        str(cand.get("candidate_id") or ""),
    )


def normalize_fill(text: str, tag: str, trailing: str) -> str:
    out = str(text or "").strip()
    if tag != "FILL_COND":
        if out and not out.endswith((";", "}", "*/")):
            out += ";"
        if out.endswith("*/"):
            out += ";"
    if tag == "FILL_COND":
        return out
    if trailing and out.endswith(";"):
        return out
    return out


def render_top1(seed: Dict[str, Any]) -> Dict[str, Any]:
    skeleton = str(seed.get("skeleton") or "")
    if not skeleton:
        return {"ok": False, "errors": ["missing_skeleton"], "rendered": None}
    table = seed.get("candidate_table") if isinstance(seed.get("candidate_table"), dict) else {}
    errors: List[str] = []
    fills: Dict[str, Dict[str, Any]] = {}
    for slot_id, candidates in table.items():
        rows = [c for c in candidates if isinstance(c, dict) and str(c.get("text") or "").strip()]
        if not rows:
            errors.append(f"{slot_id}:no_nonempty_candidate")
            continue
        rows.sort(key=candidate_sort_key, reverse=True)
        fills[str(slot_id)] = rows[0]

    def repl(match: re.Match[str]) -> str:
        slot_id = match.group("slot_ref")
        tag = match.group("tag")
        trailing = match.group("trailing") or ""
        cand = fills.get(slot_id)
        if cand is None:
            errors.append(f"{slot_id}:missing_candidate_for_placeholder")
            return match.group(0)
        return normalize_fill(str(cand.get("text") or ""), tag, trailing)

    rendered = PLACEHOLDER_RE.sub(repl, skeleton)
    remaining = [m.group("slot_ref") for m in PLACEHOLDER_RE.finditer(rendered)]
    if remaining:
        errors.append("remaining_placeholders=" + ",".join(remaining[:20]))
    return {
        "ok": not errors,
        "errors": errors,
        "rendered": rendered,
        "selected_fills": {
            slot_id: {
                "candidate_id": cand.get("candidate_id"),
                "confidence": cand.get("confidence"),
                "text": cand.get("text"),
            }
            for slot_id, cand in fills.items()
        },
        "remaining_placeholder_count": len(remaining),
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    stats = {
        "rows": 0,
        "render_ok": 0,
        "missing_skeleton": 0,
        "remaining_placeholder": 0,
        "empty_candidate_slot": 0,
        "candidate_slot_count_dist": Counter(),
        "error_dist": Counter(),
        "examples": [],
    }
    out_jsonl = Path(args.out_jsonl)
    out_txt = Path(args.out_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as wf:
        for _line_no, seed in iter_jsonl(Path(args.decode_jsonl), limit=int(args.limit)):
            stats["rows"] += 1
            result = render_top1(seed)
            table = seed.get("candidate_table") if isinstance(seed.get("candidate_table"), dict) else {}
            stats["candidate_slot_count_dist"][len(table)] += 1
            if result["ok"]:
                stats["render_ok"] += 1
            for err in result.get("errors") or []:
                if "missing_skeleton" in err:
                    stats["missing_skeleton"] += 1
                if "remaining_placeholders" in err:
                    stats["remaining_placeholder"] += 1
                if "no_nonempty_candidate" in err or "missing_candidate_for_placeholder" in err:
                    stats["empty_candidate_slot"] += 1
                stats["error_dist"][err.split("=")[0]] += 1
            audit_row = {
                "function_key": seed.get("function_key"),
                "example": seed.get("example"),
                "input_line": seed.get("input_line"),
                "func_name": seed.get("func_name"),
                "render_ok": result["ok"],
                "errors": result.get("errors") or [],
                "remaining_placeholder_count": result.get("remaining_placeholder_count"),
                "selected_fills": result.get("selected_fills"),
            }
            if bool(args.include_rendered):
                audit_row["rendered_top1"] = result.get("rendered")
            wf.write(json.dumps(audit_row, ensure_ascii=False, sort_keys=True) + "\n")
            if not result["ok"] and len(stats["examples"]) < 20:
                stats["examples"].append(audit_row)

    lines = [
        "=== Module C ECSL Global Decode Seed Audit ===",
        f"decode_jsonl = {args.decode_jsonl}",
        f"out_jsonl = {args.out_jsonl}",
        "",
        f"rows = {stats['rows']}",
        f"render_ok = {stats['render_ok']}/{max(1, stats['rows'])}",
        f"missing_skeleton = {stats['missing_skeleton']}",
        f"remaining_placeholder = {stats['remaining_placeholder']}",
        f"empty_candidate_slot = {stats['empty_candidate_slot']}",
        f"candidate_slot_count_dist = {stats['candidate_slot_count_dist'].most_common()}",
        f"error_dist = {stats['error_dist'].most_common()}",
        "",
        "-- first bad examples --",
    ]
    for item in stats["examples"]:
        lines.append(json.dumps({k: item.get(k) for k in ("example", "input_line", "func_name", "errors")}, ensure_ascii=False, sort_keys=True))
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit ECSL global decode seed records.")
    ap.add_argument("--decode-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-rendered", action="store_true")
    return ap.parse_args()


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()

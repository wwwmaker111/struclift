#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deduplicate binskel-style JSONL by original function identity.

The key is ``(source_file or binary, func_name)``. Records without a usable
function key are kept as line-unique entries so malformed data is not silently
collapsed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FuncKey = Tuple[str, str, str]


def _clean_path(s: object) -> str:
    return str(s or "").strip().replace("\\", "/")


def _func_key(rec: dict, line_no: int) -> FuncKey:
    src = _clean_path(rec.get("source_file"))
    binary = _clean_path(rec.get("binary"))
    func = str(rec.get("func_name") or "").strip()
    ident = src or binary
    if not ident or not func:
        return ("__MISSING_KEY__", str(line_no), "")
    return ("source_or_binary", ident, func)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deduplicate binskel JSONL by (source_file or binary, func_name)",
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--keep",
        choices=("first", "last"),
        default="last",
        help="Which record to keep when the same function key appears multiple times.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts; do not write output.",
    )
    ap.add_argument(
        "--examples",
        type=int,
        default=10,
        help="How many duplicate-key examples to print.",
    )
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    kept: Dict[FuncKey, Tuple[int, str]] = {}
    counts: Dict[FuncKey, int] = {}
    dup_examples: List[Tuple[FuncKey, int]] = []
    total = 0
    bad_json = 0
    missing_key = 0

    with inp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.rstrip("\n\r")
            if not s.strip():
                continue
            total += 1
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                bad_json += 1
                raise SystemExit(f"{inp}:{line_no}: JSON decode failed: {e}") from e
            if not isinstance(rec, dict):
                bad_json += 1
                raise SystemExit(f"{inp}:{line_no}: expected JSON object")

            key = _func_key(rec, line_no)
            if key[0] == "__MISSING_KEY__":
                missing_key += 1
            prev = counts.get(key, 0)
            counts[key] = prev + 1
            if prev and len(dup_examples) < max(0, int(args.examples)):
                dup_examples.append((key, line_no))

            if args.keep == "first":
                kept.setdefault(key, (line_no, s))
            else:
                kept[key] = (line_no, s)

    duplicate_extra = sum(c - 1 for c in counts.values() if c > 1)
    unique = len(kept)

    logger.info("input: %s", inp)
    logger.info("records=%d unique=%d duplicate_extra=%d", total, unique, duplicate_extra)
    logger.info(
        "missing_key_kept_line_unique=%d bad_json=%d keep=%s",
        missing_key,
        bad_json,
        args.keep,
    )
    if dup_examples:
        logger.info("duplicate examples:")
        for key, line_no in dup_examples:
            logger.info("  line %d key=%r", line_no, key)

    if args.dry_run:
        logger.info("dry-run: not writing output")
        return

    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(kept.values(), key=lambda x: x[0])
    with outp.open("w", encoding="utf-8") as out:
        for _, raw in rows:
            out.write(raw)
            out.write("\n")
    logger.info("wrote: %s (%d records)", outp, len(rows))


if __name__ == "__main__":
    main()

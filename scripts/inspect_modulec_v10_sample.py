#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print a readable V10 Module-C SFT sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer


def _read_row(path: Path, row: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row:
                return json.loads(line)
    raise IndexError(f"row {row} out of range: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    rec = _read_row(args.jsonl.expanduser().resolve(), args.row)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("=" * 100)
    print("META")
    print("=" * 100)
    for k in [
        "func_name",
        "source_file",
        "binary",
        "opt_level",
        "sft_prompt_layout",
        "sft_v10_from",
        "sft_v10_body_only",
        "sft_v10_loss_on_output_slot_spans",
    ]:
        print(f"{k} = {rec.get(k)}")
    labels: List[int] = list(rec.get("sft_labels") or [])
    print(f"len(sft_input_ids) = {len(rec.get('sft_input_ids') or [])}")
    print(f"num_supervised_tokens = {sum(1 for x in labels if int(x) != -100)}")
    print(f"slot_types = {rec.get('slot_types')}")
    print(f"sft_fill_positions = {rec.get('sft_fill_positions')}")
    print(f"sft_slot_bb_indices = {rec.get('sft_slot_bb_indices')}")
    print(f"sft_slot_output_spans = {rec.get('sft_slot_output_spans')}")
    print(f"sft_teacher_target_sources = {rec.get('sft_teacher_target_sources')}")

    print("\n" + "=" * 100)
    print("PROMPT / EMPTY-SLOT INPUT")
    print("=" * 100)
    print(rec.get("sft_prompt_text_pretty", ""))

    print("\n" + "=" * 100)
    print("OUTPUT / COMPLETE FILLED FUNCTION")
    print("=" * 100)
    print(rec.get("sft_output_text_pretty", ""))

    print("\n" + "=" * 100)
    print("SUPERVISED LABEL SPANS")
    print("=" * 100)
    for i, span in enumerate(rec.get("sft_slot_output_spans") or []):
        if not isinstance(span, list) or len(span) != 2:
            continue
        s, e = int(span[0]), int(span[1])
        ids = [int(x) for x in labels[s:e] if int(x) != -100]
        print("-" * 80)
        print(f"slot_{i} type={rec.get('slot_types', [None])[i] if i < len(rec.get('slot_types') or []) else None}")
        print(f"span=[{s}, {e}) bb={rec.get('sft_slot_bb_indices', [None])[i] if i < len(rec.get('sft_slot_bb_indices') or []) else None}")
        print(f"source={rec.get('sft_teacher_target_sources', [None])[i] if i < len(rec.get('sft_teacher_target_sources') or []) else None}")
        print(tok.decode(ids, skip_special_tokens=False))

    print("\n" + "=" * 100)
    print("DECODED FULL INPUT IDS")
    print("=" * 100)
    print(tok.decode(rec.get("sft_input_ids") or [], skip_special_tokens=False))


if __name__ == "__main__":
    main()

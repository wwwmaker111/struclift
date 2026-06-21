#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit Stage2AB SFT JSONL format before overfit/full rebuild."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


BAD_SPACING_PATTERNS = [
    re.compile(r"\breturn(?=(?:0x[0-9A-Fa-f]+|\d|true\b|false\b|NULL\b|nullptr\b|sizeof\b|[(!*&~+\-]))"),
    re.compile(r"\b(if|while|for|switch)\("),
    re.compile(r"\b(static|extern|inline|const|volatile|signed|unsigned)(?=(?:void|char|short|int|long|float|double|bool|size_t)\b)"),
    re.compile(r"\bstruct(?=[A-Za-z_][A-Za-z0-9_]*\b)"),
]


def _iter_jsonl(path: Path, limit: int) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if limit and i > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _count_supervised(labels: Any) -> int:
    return sum(1 for x in labels if x != -100) if isinstance(labels, list) else 0


def _bad_spacing(texts: List[str]) -> List[str]:
    bad: List[str] = []
    for txt in texts:
        s = str(txt or "")
        if any(p.search(s) for p in BAD_SPACING_PATTERNS):
            bad.append(s[:240].replace("\n", "\\n"))
            if len(bad) >= 5:
                break
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Stage2AB SFT JSONL format")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    vocab = None
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        vocab = len(tok)

    n = 0
    counters: "collections.Counter[str]" = collections.Counter()
    lens: List[int] = []
    sups: List[int] = []
    sem_slots: List[int] = []
    max_id = -1
    bad_id = 0
    examples: List[Dict[str, Any]] = []
    bad_spacing_examples: List[tuple[int, str, List[str]]] = []

    for line_no, d in _iter_jsonl(args.jsonl.expanduser().resolve(), args.limit):
        n += 1
        ids = d.get("sft_input_ids") or []
        labels = d.get("sft_labels") or []
        slot_types = d.get("slot_types") or []
        targets = d.get("sft_teacher_targets") or []
        supervise = d.get("sft_teacher_slot_supervise") or []
        prompt = str(d.get("sft_prompt_text_pretty") or "")

        L = len(ids)
        S = _count_supervised(labels)
        lens.append(L)
        sups.append(S)
        sem_slots.append(
            sum(
                1
                for t, keep in zip(slot_types, supervise)
                if keep and t not in {"FILL_TYPE", "FILL_PARAM"}
            )
        )
        if ids:
            max_id = max(max_id, max(int(x) for x in ids))
        if vocab is not None and any((not isinstance(x, int)) or x < 0 or x >= vocab for x in ids):
            bad_id += 1

        if d.get("sft_skeleton_source") == "stage2_module_b":
            counters["stage2_skeleton"] += 1
        if d.get("sft_stage2_ckpt"):
            counters["has_stage2_ckpt"] += 1
        if d.get("bin_instr_text"):
            counters["has_bin_instr_text"] += 1
        if d.get("sft_binary_context_readable_asm") or d.get("sft_binary_context_readable_asm_in_prompt"):
            counters["asm_in_prompt"] += 1
        if "function_asm:" in prompt:
            counters["prompt_contains_function_asm"] += 1
        if d.get("sft_signature_teacher_ret_type") not in (None, "", "__STRUCLIFT_UNKNOWN_RET__"):
            counters["ret_supervised"] += 1
        if d.get("sft_signature_teacher_params") not in (None, "", "__STRUCLIFT_UNKNOWN_PARAMS__"):
            counters["param_supervised"] += 1
        if S > 0:
            counters["has_supervision"] += 1
        if L <= 16384:
            counters["len_le_16384"] += 1

        bad_space = _bad_spacing([str(x) for x in targets])
        if bad_space and len(bad_spacing_examples) < args.show:
            bad_spacing_examples.append((line_no, str(d.get("func_name") or "?"), bad_space))

        if len(examples) < args.show:
            examples.append(
                {
                    "line": line_no,
                    "func": d.get("func_name"),
                    "len": L,
                    "supervised_tokens": S,
                    "slot_types": slot_types[:16],
                    "semantic_supervised_slots": sem_slots[-1],
                    "ret": d.get("sft_signature_teacher_ret_type"),
                    "params": d.get("sft_signature_teacher_params"),
                    "asm_in_prompt": bool(
                        d.get("sft_binary_context_readable_asm")
                        or d.get("sft_binary_context_readable_asm_in_prompt")
                    ),
                    "asm_budget": [
                        d.get("sft_binary_context_asm_max_total_lines"),
                        d.get("sft_binary_context_asm_max_lines_per_bb"),
                    ],
                    "stage2_shape": d.get("sft_stage2_alignment_shape"),
                    "prompt_head": prompt[:600],
                    "targets_head": targets[:8],
                }
            )

    def pct(k: str) -> str:
        return f"{counters[k]}/{n} ({(100.0 * counters[k] / max(1, n)):.1f}%)"

    print("== summary ==")
    print("records:", n)
    if vocab is not None:
        print("tokenizer_len:", vocab)
        print("max_input_id:", max_id)
        print("bad_id_records:", bad_id)
    for k in [
        "stage2_skeleton",
        "has_stage2_ckpt",
        "has_bin_instr_text",
        "asm_in_prompt",
        "prompt_contains_function_asm",
        "ret_supervised",
        "param_supervised",
        "has_supervision",
        "len_le_16384",
    ]:
        print(f"{k}:", pct(k))
    if lens:
        xs = sorted(lens)
        ys = sorted(sups)
        zs = sorted(sem_slots)
        print("input_len min/median/max:", xs[0], xs[len(xs) // 2], xs[-1])
        print("supervised_tokens min/median/max:", ys[0], ys[len(ys) // 2], ys[-1])
        print("semantic_supervised_slots min/median/max:", zs[0], zs[len(zs) // 2], zs[-1])

    print("\n== examples ==")
    for ex in examples:
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    print("\n== bad spacing suspects ==")
    if not bad_spacing_examples:
        print("none")
    for line_no, fn, bads in bad_spacing_examples:
        print(f"line {line_no} {fn}:")
        for b in bads:
            print("  ", b)


if __name__ == "__main__":
    main()

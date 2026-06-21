#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack ECSL slot training JSONL into an auditable Module-C training manifest.

The candidate builder may output both hard/weak references and C1 priors.  This
packer creates curriculum shards and a manifest for actual Module-C fine-tuning.
It never promotes C1/C0 priors into hard labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MANIFEST_SCHEMA = "ModuleC.ECSL.TrainingManifest.v0"


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


def tier_rank(tier: str) -> int:
    return {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}.get(str(tier), 0)


def normalize_reference_weights(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = sum(max(0.0, safe_float(ref.get("weight"))) for ref in refs)
    if total <= 0:
        return refs
    out = []
    for ref in refs:
        item = dict(ref)
        item["normalized_weight"] = round(max(0.0, safe_float(ref.get("weight"))) / total, 8)
        out.append(item)
    return out


def make_slot_training_item(rec: Dict[str, Any], *, min_tier: str) -> Dict[str, Any] | None:
    refs = []
    for ref in rec.get("references") or []:
        level = str(ref.get("confidence_level") or "C0")
        if tier_rank(level) >= tier_rank(min_tier) and safe_float(ref.get("weight")) > 0.0:
            refs.append(dict(ref))
    if not refs:
        return None
    refs = normalize_reference_weights(refs)
    completion = {
        "fills": [
            {
                "slot_id": rec.get("slot_id"),
                "text": refs[0].get("text"),
            }
        ]
    }
    return {
        "schema": "ModuleC.ECSL.SlotSFTItem.v0",
        "function_key": rec.get("function_key"),
        "example": rec.get("example"),
        "input_line": rec.get("input_line"),
        "func_name": rec.get("func_name"),
        "slot_id": rec.get("slot_id"),
        "slot_type": rec.get("slot_type"),
        "prompt": rec.get("prompt"),
        "completion": json.dumps(completion, ensure_ascii=False, sort_keys=True),
        "references": refs,
        "loss": {
            "type": "confidence_weighted_multi_reference_sft",
            "formula": "-log sum_j c_ij * P_theta(y_ij | x_i, slot_i)",
            "reference_count": len(refs),
            "min_tier": min_tier,
        },
        "candidate_priors": rec.get("candidate_priors") or [],
        "safety": {
            "hard_ce_uses_only_references": True,
            "c1_c0_priors_are_not_hard_labels": True,
        },
    }


def curriculum_bucket(item: Dict[str, Any]) -> str:
    slot_type = str(item.get("slot_type") or "UNKNOWN")
    ref_tiers = [str(ref.get("confidence_level") or "C0") for ref in item.get("references") or []]
    best = max([tier_rank(t) for t in ref_tiers] or [0])
    if best >= tier_rank("C3"):
        prefix = "strong"
    elif best >= tier_rank("C2"):
        prefix = "weak"
    else:
        prefix = "prior"
    if slot_type == "COND":
        return f"{prefix}_condition"
    if slot_type in {"BLOCK", "CASE"}:
        return f"{prefix}_block"
    if slot_type in {"GOTO", "BREAK", "CONTINUE", "EARLY_EXIT", "CLEANUP_EXIT"}:
        return f"{prefix}_control_transfer"
    return f"{prefix}_other"


def pack(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "ecsl_sft_items.jsonl"
    manifest_path = out_dir / "ecsl_training_manifest.json"
    report_path = out_dir / "ecsl_training_manifest_report.txt"

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    stats = {
        "input_records": 0,
        "sft_items": 0,
        "skipped_no_reference": 0,
        "bucket_dist": Counter(),
        "slot_type_dist": Counter(),
        "best_tier_dist": Counter(),
        "reference_count_dist": Counter(),
    }

    with all_path.open("w", encoding="utf-8") as wf:
        for _line_no, rec in iter_jsonl(Path(args.train_jsonl), limit=int(args.limit)):
            stats["input_records"] += 1
            item = make_slot_training_item(rec, min_tier=str(args.min_tier))
            if item is None:
                stats["skipped_no_reference"] += 1
                continue
            bucket = curriculum_bucket(item)
            item["curriculum_bucket"] = bucket
            wf.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            buckets.setdefault(bucket, []).append(item)
            stats["sft_items"] += 1
            stats["bucket_dist"][bucket] += 1
            stats["slot_type_dist"][str(item.get("slot_type") or "UNKNOWN")] += 1
            best_tier = max(
                (str(ref.get("confidence_level") or "C0") for ref in item.get("references") or []),
                key=tier_rank,
                default="C0",
            )
            stats["best_tier_dist"][best_tier] += 1
            stats["reference_count_dist"][len(item.get("references") or [])] += 1

    shard_paths = {}
    for bucket, rows in sorted(buckets.items()):
        path = out_dir / f"{bucket}.jsonl"
        shard_paths[bucket] = str(path)
        with path.open("w", encoding="utf-8") as wf:
            for item in rows:
                wf.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "input_train_jsonl": str(args.train_jsonl),
        "all_sft_items_jsonl": str(all_path),
        "curriculum_shards": shard_paths,
        "min_tier": str(args.min_tier),
        "training_objective": {
            "name": "confidence_weighted_multi_reference_sft",
            "loss": "-log sum_j c_ij * P_theta(y_ij | x_i, slot_i)",
            "c1_c0_policy": "priors_only_not_hard_ce",
            "module_scope": "fine-tune Module C adapter/LoRA only; keep Modules A/B frozen",
        },
        "latent_em": {
            "status": "KEEP_TRAINING_STAGE",
            "objective": "q_ij proportional to c_ij * P_theta(y_ij | x_i, slot_i); M-step weighted CE",
            "guardrails": ["warm up SFT first", "use entropy floor", "cap confidence", "exclude C1/C0 from hard updates"],
        },
        "consistency_training": {
            "status": "KEEP_TRAINING_STAGE",
            "views": [
                "branch inversion",
                "early-return view",
                "if-else view",
                "loop+break view",
                "source-like view",
                "skeleton-like view",
            ],
            "requirement": "different views must satisfy the same Obs(slot)",
        },
        "self_training": {
            "status": "KEEP_TRAINING_STAGE",
            "promotion_rule": "only predictions validated at C3/C4/C5 may enter strong supervision",
        },
        "grpo_rl": {
            "status": "KEEP_LATER_STAGE",
            "action_space": "slot text only",
            "trainable_scope": "Module C adapter/LoRA only; Modules A/B frozen",
            "hard_constraints": ["parse_ok", "compile_ok", "no skeleton mutation", "slot grammar valid", "no illegal control transfer"],
        },
        "recommended_sequence": [
            "warmup on strong_condition/strong_block if available",
            "train weak_condition/weak_block with low weight or consistency objective",
            "use priors for retrieval/reranking only",
            "connect V2/V3 validators before promoting C2 to C3+ hard supervision",
        ],
        "stats": {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in stats.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "=== Module C ECSL Training Manifest ===",
        f"input_train_jsonl = {args.train_jsonl}",
        f"out_dir = {out_dir}",
        f"all_sft_items_jsonl = {all_path}",
        f"manifest = {manifest_path}",
        "",
        f"input_records = {stats['input_records']}",
        f"sft_items = {stats['sft_items']}",
        f"skipped_no_reference = {stats['skipped_no_reference']}",
        f"bucket_dist = {stats['bucket_dist'].most_common()}",
        f"slot_type_dist = {stats['slot_type_dist'].most_common()}",
        f"best_tier_dist = {stats['best_tier_dist'].most_common()}",
        f"reference_count_dist = {stats['reference_count_dist'].most_common()}",
        "",
        "Notes:",
        "- This packer does not train the model; it prepares auditable data for Module-C SFT.",
        "- C1/C0 candidate_priors are never used as hard CE references.",
        "- If sft_items is zero, connect stronger candidate synthesis/validators before fine-tuning.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pack ECSL training records into SFT manifest and curriculum shards.")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-tier", default="C2", choices=["C2", "C3", "C4", "C5"])
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    pack(parse_args())


if __name__ == "__main__":
    main()

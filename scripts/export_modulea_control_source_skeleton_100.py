#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export first N source-control functions with Module-A-only skeletons.

This is intentionally the simple fast path:
  - scan JSONL in order;
  - keep only rows whose real source has control flow;
  - run Module A on binary-only records with ``collate_binskel_batch``;
  - build skeletons from ``pattern_logits.argmax`` + current CFG builder;
  - write the full real source function and full generated skeleton.

It does not run Module B/alignment and does not use Stage2 collate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402

import audit_stage2ab_skeleton_effect as effect  # noqa: E402
import augment_binskel_sft as sft_aug  # noqa: E402
import repack_modulec_v9_to_v10 as repack  # noqa: E402
from struclift.config import StrucLiftConfig  # noqa: E402
from struclift.data.binskel_jsonl import (  # noqa: E402
    collate_binskel_batch,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift  # noqa: E402


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


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        k = k.replace("._orig_mod.", ".")
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        out[k] = v
    return out


def load_model(path: Path, device: torch.device) -> StrucLift:
    cfg = StrucLiftConfig()
    model = StrucLift(cfg)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    state = normalize_state(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded ckpt = {path}", flush=True)
    print(f"missing = {len(missing)} unexpected = {len(unexpected)}", flush=True)
    if missing[:10]:
        print("missing examples:", missing[:10], flush=True)
    if unexpected[:10]:
        print("unexpected examples:", unexpected[:10], flush=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def source_control_outline(row: Dict[str, Any], source_control_filter: str) -> List[Tuple[str, int, int]]:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    outline = effect._source_control_outline(
        rows,
        active_lines=active,
        source_control_filter=source_control_filter,
    )
    if not outline and active:
        outline = effect._active_line_control_outline(
            row,
            active,
            source_control_filter=source_control_filter,
        )
    return outline


def has_control_by_source(row: Dict[str, Any], source_control_filter: str) -> bool:
    return len(source_control_outline(row, source_control_filter)) > 0


def n_bb(row: Dict[str, Any]) -> int:
    bl = row.get("bin_block_lengths")
    return len(bl) if isinstance(bl, list) else 0


def is_valid_bin(row: Dict[str, Any]) -> bool:
    bl = row.get("bin_block_lengths")
    pl = row.get("pattern_labels")
    if not isinstance(bl, list) or not bl:
        return False
    if not isinstance(pl, list) or len(pl) != len(bl):
        return False
    n_instr = sum(int(x) for x in bl)
    if not isinstance(row.get("bin_opcodes"), list) or len(row["bin_opcodes"]) != n_instr:
        return False
    if not isinstance(row.get("bin_operand_types"), list) or len(row["bin_operand_types"]) != n_instr * 3:
        return False
    if not isinstance(row.get("bin_operand_values"), list) or len(row["bin_operand_values"]) != n_instr * 3:
        return False
    if not isinstance(row.get("bin_struct_features"), list) or len(row["bin_struct_features"]) != len(bl):
        return False
    ei = row.get("bin_edge_index")
    et = row.get("bin_edge_type") or []
    if not isinstance(ei, list) or len(ei) != 2:
        return False
    if len(ei[0]) != len(ei[1]) or len(ei[0]) != len(et):
        return False
    return True


def bin_only_record(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "bin_block_lengths",
        "bin_opcodes",
        "bin_operand_types",
        "bin_operand_values",
        "bin_struct_features",
        "bin_edge_index",
        "bin_edge_type",
        "pattern_labels",
    ]
    return {k: row[k] for k in keys if k in row}


@torch.inference_mode()
def predict_labels_batch(
    model: StrucLift,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
) -> List[Tuple[List[int], List[float]]]:
    small = [bin_only_record(r) for r in rows]
    batch = collate_binskel_batch(
        small,
        inst_dropout_p=0.0,
        edge_dropout_p=0.0,
        add_func_uid=False,
    )
    batch = move_binskel_batch_to_device(batch, device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        enc = model.encode(
            bin_opcodes=batch["bin_opcodes"],
            bin_operand_types=batch["bin_operand_types"],
            bin_operand_values=batch["bin_operand_values"],
            bin_block_lengths=batch["bin_block_lengths"],
            bin_edge_index=batch["bin_edge_index"],
            bin_edge_type=batch["bin_edge_type"],
            bin_struct_features=batch["bin_struct_features"],
            bin_batch=batch.get("bin_batch"),
            run_alignment=False,
        )
    logits = enc.pattern_logits.float()
    labels_all = logits.argmax(dim=-1).detach().cpu().tolist()
    probs_all = torch.softmax(logits, dim=-1).amax(dim=-1).detach().cpu().tolist()

    out: List[Tuple[List[int], List[float]]] = []
    off = 0
    for row in rows:
        n = n_bb(row)
        out.append((
            [int(x) for x in labels_all[off : off + n]],
            [float(x) for x in probs_all[off : off + n]],
        ))
        off += n
    return out


def source_function_with_numbers(row: Dict[str, Any]) -> Tuple[str, Optional[Tuple[int, int]]]:
    lt = repack._line_to_text(row.get("source_file"))
    if not lt:
        return "<source file unreadable>", None
    bounds = repack._function_body_line_range(row, lt)
    if bounds is None:
        return "<function bounds unavailable>", None
    lo, hi = bounds
    lines = []
    for ln in range(lo, hi + 1):
        if ln in lt:
            lines.append(f"{ln:5d}: {lt[ln]}")
    return "\n".join(lines) if lines else "<empty source range>", bounds


def source_control_meta(row: Dict[str, Any], source_control_filter: str) -> Dict[str, Any]:
    rows = effect._source_rows(row)
    active = effect._active_source_lines(row)
    outline = effect._source_control_outline(
        rows,
        active_lines=active,
        source_control_filter=source_control_filter,
    )
    outline_source = "statement_rows"
    if not outline and active:
        fallback = effect._active_line_control_outline(
            row,
            active,
            source_control_filter=source_control_filter,
        )
        if fallback:
            outline = fallback
            outline_source = "active_line_fallback"
    seq = [(t, d) for t, d, _ln in outline]
    return {
        "source_control_count": len(seq),
        "source_counts": dict(effect._counts(seq)),
        "source_outline": effect._source_outline_lines(outline, limit=160),
        "source_outline_source": outline_source,
        "source_statement_rows": len(rows),
        "active_source_lines": len(active),
        "source_parse_status": effect._source_parse_status(row, rows),
    }


def skeleton_from_pred_labels(row: Dict[str, Any], labels: Sequence[int]) -> Tuple[Any, str]:
    bei = row.get("bin_edge_index") or [[], []]
    edge_index = torch.tensor(
        [[int(x) for x in bei[0]], [int(x) for x in bei[1]]],
        dtype=torch.int64,
    )
    pl_t = torch.tensor([int(x) for x in labels], dtype=torch.int64)
    hints = sft_aug._bb_opcode_hints(row)
    skel = sft_aug.build_skeleton_for_binary_cfg(pl_t, edge_index, len(labels), hints)
    text = skel.skeleton_text
    if hasattr(sft_aug, "_format_skeleton_text_for_prompt"):
        text = sft_aug._format_skeleton_text_for_prompt(text)
    return skel, text


def label_summary(labels: Sequence[int]) -> str:
    c = Counter(int(x) for x in labels)
    return ", ".join(
        f"{PATTERN_NAMES.get(k, 'unknown_' + str(k))}={c[k]}"
        for k in sorted(c)
    )


def label_sequence(labels: Sequence[int], max_items: int = 220) -> str:
    xs = []
    for i, lab in enumerate(labels[:max_items]):
        xs.append(f"BB_{i}:{PATTERN_NAMES.get(int(lab), 'unknown_' + str(lab))}")
    if len(labels) > max_items:
        xs.append("...")
    return "\n".join(xs)


def control_meta_from_skeleton(text: str, nbb: int) -> Dict[str, Any]:
    pred_seq = effect._control_outline_from_text(text)
    pred_counts = effect._counts(pred_seq)
    return {
        "pred_control_count": len(pred_seq),
        "pred_counts": dict(pred_counts),
        "pred_outline": effect._outline_lines(pred_seq, limit=160),
        "bb_stats": effect._bb_marker_stats(text, nbb),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--stage2-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--source-control-filter",
        choices=["raw", "bb-start-line", "bb-range"],
        default="bb-start-line",
    )
    ap.add_argument("--min-source-controls", type=int, default=1)
    # Accepted for compatibility with older commands; intentionally unused.
    ap.add_argument("--candidate-pool-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=530)
    args = ap.parse_args()

    jsonl_path = args.jsonl.expanduser().resolve()
    ckpt_path = args.stage2_ckpt.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("===== first source-control functions, A-only skeleton export =====", flush=True)
    print("jsonl =", jsonl_path, flush=True)
    print("ckpt =", ckpt_path, flush=True)
    print("out =", out_path, flush=True)
    print("device =", device, flush=True)

    t0 = time.time()
    candidates: List[Tuple[int, Dict[str, Any], str, Tuple[int, int]]] = []
    seen = set()
    scanned = 0
    skipped: Counter[str] = Counter()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            scanned += 1
            if args.limit and scanned > int(args.limit):
                break
            try:
                row = json.loads(line)
            except Exception:
                skipped["bad_json"] += 1
                continue
            if not is_valid_bin(row):
                skipped["bad_bin"] += 1
                continue
            src_text, bounds = source_function_with_numbers(row)
            if bounds is None:
                skipped["source_bounds_unavailable"] += 1
                continue
            key = (str(row.get("source_file") or ""), str(row.get("func_name") or ""), bounds)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            if not has_control_by_source(row, str(args.source_control_filter)):
                skipped["no_source_control"] += 1
                continue
            smeta = source_control_meta(row, str(args.source_control_filter))
            if int(smeta.get("source_control_count") or 0) < int(args.min_source_controls):
                skipped["below_min_source_controls"] += 1
                continue
            seen.add(key)
            candidates.append((line_no, row, src_text, bounds))
            if len(candidates) >= int(args.num_examples):
                break

    print(
        f"selected candidates={len(candidates)} scanned={scanned} "
        f"skipped={dict(skipped)} select_sec={time.time() - t0:.1f}",
        flush=True,
    )

    model = load_model(ckpt_path, device)

    pred_map: Dict[int, Tuple[List[int], List[float]]] = {}
    bs = max(1, int(args.batch_size))
    for start in range(0, len(candidates), bs):
        chunk = candidates[start : start + bs]
        rows = [x[1] for x in chunk]
        preds = predict_labels_batch(model, rows, device)
        for item, pred in zip(chunk, preds):
            pred_map[item[0]] = pred
        print(
            f"predicted {min(start + bs, len(candidates))}/{len(candidates)}",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        wf.write("===== FAST FIRST-100 MODULE-A CONTROL SKELETON VS REAL SOURCE EXPORT =====\n")
        wf.write(f"input = {jsonl_path}\n")
        wf.write(f"ckpt = {ckpt_path}\n")
        wf.write("skeleton_source = ModuleA pattern_logits.argmax + CFG rule builder; B/alignment is NOT run\n")
        wf.write(f"device = {device}\n")
        wf.write(f"scanned = {scanned}\n")
        wf.write(f"exported = {len(candidates)}\n")
        wf.write(f"requested = {int(args.num_examples)}\n")
        wf.write(f"skipped = {dict(skipped)}\n")

        for idx, (line_no, row, src_text, bounds) in enumerate(candidates):
            if idx % 10 == 0:
                print(f"writing example {idx}/{len(candidates)}", flush=True)
            labels, conf = pred_map[line_no]
            try:
                skel, skeleton_text = skeleton_from_pred_labels(row, labels)
                smeta = source_control_meta(row, str(args.source_control_filter))
                pmeta = control_meta_from_skeleton(skeleton_text, len(labels))
                src_counts = Counter(smeta.get("source_counts") or {})
                pred_counts = Counter(pmeta.get("pred_counts") or {})
                bb_stats = pmeta.get("bb_stats") or {}

                vals = [float(x) for x in conf]
                vals_sorted = sorted(vals)
                med = vals_sorted[len(vals_sorted) // 2] if vals_sorted else 0.0
                conf_line = (
                    f"min/median/max={min(vals):.3f}/{med:.3f}/{max(vals):.3f}"
                    if vals else "n/a"
                )

                wf.write("\n" + "=" * 120 + "\n")
                wf.write(f"example = {idx}  input_line = {line_no}  func = {row.get('func_name')}\n")
                wf.write(f"source_file = {row.get('source_file')}\n")
                wf.write(f"source_bounds = {bounds[0]}-{bounds[1]}\n")
                wf.write("skeleton_source = ModuleA_argmax_plus_current_rule_builder_no_B_alignment\n")
                wf.write(f"bb = {len(labels)}\n")
                wf.write(
                    f"source_parse = {smeta.get('source_parse_status')}  "
                    f"source_ref = {smeta.get('source_outline_source')}  "
                    f"src_rows = {smeta.get('source_statement_rows')}  "
                    f"active_source_lines = {smeta.get('active_source_lines')}\n"
                )
                wf.write(
                    f"ctrl(pred/source) = {pmeta.get('pred_control_count')}/{smeta.get('source_control_count')}  "
                    f"gaps(if,loop,switch)=("
                    f"{int(pred_counts.get('if', 0)) - int(src_counts.get('if', 0))},"
                    f"{int(pred_counts.get('loop', 0)) - int(src_counts.get('loop', 0))},"
                    f"{int(pred_counts.get('switch', 0)) - int(src_counts.get('switch', 0))})\n"
                )
                wf.write(
                    f"bb_cover = {bb_stats.get('bb_covered')}/{len(labels)}  "
                    f"bb_missing = {bb_stats.get('bb_missing')}  "
                    f"bb_duplicated_first20 = {bb_stats.get('bb_duplicated')}\n"
                )
                wf.write(f"ab_pred_10class_label_counts = {label_summary(labels)}\n")
                wf.write(f"ab_pred_10class_label_confidence = {conf_line}\n")
                wf.write(f"skeleton_reliability = {getattr(skel, 'reliability', None)}\n")
                wf.write(f"skeleton_build_stats = {getattr(skel, 'build_stats', None)}\n")

                wf.write("\n-- SOURCE CONTROL OUTLINE --\n")
                wf.write("\n".join(smeta.get("source_outline") or ["<none>"]) + "\n")

                wf.write("\n-- AB SKELETON CONTROL OUTLINE --\n")
                wf.write("\n".join(pmeta.get("pred_outline") or ["<none>"]) + "\n")

                wf.write("\n-- AB PREDICTED 10CLASS BB LABEL SEQUENCE --\n")
                wf.write(label_sequence(labels) + "\n")

                wf.write("\n-- REAL SOURCE FUNCTION --\n")
                wf.write(src_text.rstrip() + "\n")

                wf.write("\n-- CURRENT AB GENERATED FULL SKELETON --\n")
                wf.write(skeleton_text.rstrip() + "\n")
            except Exception as e:
                wf.write("\n" + "=" * 120 + "\n")
                wf.write(f"example = {idx}  input_line = {line_no}  func = {row.get('func_name')}\n")
                wf.write(f"ERROR writing example: {type(e).__name__}: {e}\n")

    print("saved_to =", out_path, flush=True)
    print("elapsed_total_sec =", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()

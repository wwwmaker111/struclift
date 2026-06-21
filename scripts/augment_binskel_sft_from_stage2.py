#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Stage3 SFT JSONL from Stage2 Module A/B outputs.

This is the proposal-aligned path:

1. Load the trained Stage2 checkpoint.
2. Run Module A + Module B on each binskel sample.
3. Use Module B's emitted skeleton / slot alignment as the Stage3 prompt.
4. Use the source side as teacher to create supervised fill targets.

The older ``augment_binskel_sft.py`` path remains useful as a GT-skeleton
diagnostic, but this script trains Module C on the distribution it will see at
deployment time.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from struclift.config import StrucLiftConfig
from struclift.data.binskel_jsonl import (
    BinSkelJsonlDataset,
    collate_binskel_stage2_batch,
    move_binskel_batch_to_device,
)
from struclift.models.struclift import StrucLift
from struclift.utils.state_dict_compat import load_state_dict_skip_shape_mismatch

import augment_binskel_sft as sft_aug

try:
    import audit_v10_kept_vs_source as source_oracle_audit
except Exception:  # pragma: no cover - optional CLI diagnostic dependency
    source_oracle_audit = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_SOFT_SOURCE_ORACLE_REASONS = {"format_or_control_shape_diff_only"}


def _attach_source_oracle_audit_fields(record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Attach rendered-GT-vs-source audit fields used to gate Stage2AB SFT data."""
    if source_oracle_audit is None:
        raise RuntimeError("audit_v10_kept_vs_source.py is not importable")
    audit = source_oracle_audit._audit_row(record)  # type: ignore[attr-defined]
    reasons = [str(x) for x in (audit.get("reasons") or [])]
    hard_reasons = [r for r in reasons if r not in _SOFT_SOURCE_ORACLE_REASONS]
    out = dict(record)
    out["sft_source_oracle_exact_body_match"] = bool(audit.get("exact_body_match"))
    out["sft_source_oracle_ok"] = bool(audit.get("oracle_ok"))
    out["sft_source_oracle_reasons"] = reasons
    out["sft_source_oracle_hard_reasons"] = hard_reasons
    return out, reasons, hard_reasons


class _RecordTimeoutError(TimeoutError):
    pass


@contextlib.contextmanager
def _record_timeout(seconds: float, label: str):
    """Bound per-record teacher construction so one bad sample cannot stall a run."""
    if seconds <= 0:
        yield
        return
    try:
        old_handler = signal.getsignal(signal.SIGALRM)
        old_timer = signal.getitimer(signal.ITIMER_REAL)
    except (AttributeError, ValueError):
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ARG001
        raise _RecordTimeoutError(f"record timeout after {seconds:.1f}s: {label}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def _load_stage2_model(ckpt_path: Path, device: torch.device) -> StrucLift:
    cfg = StrucLiftConfig()
    model = StrucLift(cfg)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a state_dict-like object: {ckpt_path}")
    missing, unexpected = load_state_dict_skip_shape_mismatch(model, state, log=logger)
    if missing:
        logger.info("Stage2 load missing keys, examples: %s", missing[:5])
    if unexpected:
        logger.info("Stage2 load unexpected keys, examples: %s", unexpected[:5])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.alignment_module.cfg.emit_skeleton = True
    return model


def _read_records(path: Path, max_lines: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f):
            if max_lines is not None and li >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skip bad JSON line %d", li + 1)
    return records


def _has_readable_binary_text(record: Dict[str, Any]) -> bool:
    for name in ("bin_instr_text", "bin_asm", "asm", "instructions_text", "disasm"):
        val = record.get(name)
        if isinstance(val, list) and val:
            return True
    return False


def _strip_alignment_tag(ctx: str) -> str:
    parts = [p.strip() for p in str(ctx or "").split("|") if p.strip()]
    tags = [
        p for p in parts
        if p.startswith("aligned_stmt_idx=") or p.startswith("role=")
    ]
    return " | ".join(tags)


def _upgrade_skeleton_binary_context(skel: Any, record: Dict[str, Any]) -> Tuple[Any, bool]:
    """Replace Module-B opcode-only slot text with richer per-BB text when present."""
    hints = sft_aug._bb_opcode_hints(record)
    readable = _has_readable_binary_text(record)
    for slot in getattr(skel, "slots", []) or []:
        bbs = [int(x) for x in getattr(slot, "aligned_bb_indices", []) or []]
        parts = []
        for bb in bbs:
            h = str(hints.get(bb, "")).strip()
            if h:
                parts.append(f"BB{bb}: {h}")
        tag = _strip_alignment_tag(getattr(slot, "binary_context_text", ""))
        if parts:
            slot.binary_context_text = " | ".join(parts + ([tag] if tag else []))
        elif tag:
            slot.binary_context_text = tag
    return skel, readable


def _run_stage2_for_record(
    model: StrucLift,
    record: Dict[str, Any],
    device: torch.device,
    *,
    use_m_gt_for_ab: bool,
) -> Tuple[Optional[Any], Optional[torch.Tensor], str]:
    try:
        batch = collate_binskel_stage2_batch([record])
        batch = move_binskel_batch_to_device(batch, device)
        with torch.inference_mode():
            out = model.encode(
                bin_opcodes=batch["bin_opcodes"],
                bin_operand_types=batch["bin_operand_types"],
                bin_operand_values=batch["bin_operand_values"],
                bin_block_lengths=batch["bin_block_lengths"],
                bin_edge_index=batch["bin_edge_index"],
                bin_edge_type=batch["bin_edge_type"],
                bin_struct_features=batch["bin_struct_features"],
                bin_batch=batch.get("bin_batch"),
                src_token_ids=batch.get("src_token_ids"),
                src_stmt_lengths=batch.get("src_stmt_lengths"),
                src_edge_index=batch.get("src_edge_index"),
                src_edge_type=batch.get("src_edge_type"),
                src_ast_types=batch.get("src_ast_types"),
                src_compound_id=batch.get("src_compound_id"),
                src_batch=batch.get("src_batch"),
                bin_adj=batch.get("adj_bin"),
                src_adj=batch.get("adj_src_gt"),
                M_gt=batch.get("M_gt") if use_m_gt_for_ab else None,
                run_alignment=True,
            )
        skel = None
        if out.skeleton_list:
            skel = out.skeleton_list[0]
        if skel is None:
            return None, None, "Stage2 Module B did not emit skeleton"
        M = None
        if out.alignment is not None and isinstance(out.alignment.alignment_matrix, list):
            M = out.alignment.alignment_matrix[0].detach().cpu()
        return skel, M, ""
    except Exception as e:
        return None, None, f"Stage2 A/B failed: {e}"


def _run_stage2_for_records(
    model: StrucLift,
    records: List[Dict[str, Any]],
    device: torch.device,
    *,
    use_m_gt_for_ab: bool,
) -> List[Tuple[Optional[Any], Optional[torch.Tensor], str]]:
    try:
        batch = collate_binskel_stage2_batch(records)
        batch = move_binskel_batch_to_device(batch, device)
        with torch.inference_mode():
            out = model.encode(
                bin_opcodes=batch["bin_opcodes"],
                bin_operand_types=batch["bin_operand_types"],
                bin_operand_values=batch["bin_operand_values"],
                bin_block_lengths=batch["bin_block_lengths"],
                bin_edge_index=batch["bin_edge_index"],
                bin_edge_type=batch["bin_edge_type"],
                bin_struct_features=batch["bin_struct_features"],
                bin_batch=batch.get("bin_batch"),
                src_token_ids=batch.get("src_token_ids"),
                src_stmt_lengths=batch.get("src_stmt_lengths"),
                src_edge_index=batch.get("src_edge_index"),
                src_edge_type=batch.get("src_edge_type"),
                src_ast_types=batch.get("src_ast_types"),
                src_compound_id=batch.get("src_compound_id"),
                src_batch=batch.get("src_batch"),
                bin_adj=batch.get("adj_bin"),
                src_adj=batch.get("adj_src_gt"),
                M_gt=batch.get("M_gt") if use_m_gt_for_ab else None,
                run_alignment=True,
            )
        skels = list(out.skeleton_list or [])
        Ms = []
        if out.alignment is not None and isinstance(out.alignment.alignment_matrix, list):
            Ms = [m.detach().cpu() for m in out.alignment.alignment_matrix]
        res: List[Tuple[Optional[Any], Optional[torch.Tensor], str]] = []
        for i in range(len(records)):
            skel = skels[i] if i < len(skels) else None
            M = Ms[i] if i < len(Ms) else None
            if skel is None:
                res.append((None, None, "Stage2 Module B did not emit skeleton"))
            else:
                res.append((skel, M, ""))
        return res
    except RuntimeError as e:
        if len(records) > 1 and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(records) // 2
            return (
                _run_stage2_for_records(
                    model, records[:mid], device, use_m_gt_for_ab=use_m_gt_for_ab,
                )
                + _run_stage2_for_records(
                    model, records[mid:], device, use_m_gt_for_ab=use_m_gt_for_ab,
                )
            )
        return [(None, None, f"Stage2 A/B failed: {e}") for _ in records]
    except Exception as e:
        return [(None, None, f"Stage2 A/B failed: {e}") for _ in records]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Proposal-aligned Stage3 SFT augmentation using Stage2 A/B skeletons",
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage2-ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--save-tokenizer", type=Path, default=None)
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument(
        "--rewrite-source-prefix",
        nargs=2,
        metavar=("FROM_PREFIX", "TO_PREFIX"),
        action="append",
        dest="prefix_pairs",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=0)
    ap.add_argument(
        "--asm-max-total-lines",
        type=int,
        default=160,
        help="Maximum readable asm lines to include in prompt; auto-shrinks when sequence is too long.",
    )
    ap.add_argument(
        "--asm-max-lines-per-bb",
        type=int,
        default=16,
        help="Maximum readable asm lines per BB to include in prompt.",
    )
    ap.add_argument(
        "--require-readable-asm",
        action="store_true",
        help="Require prompt-ready readable asm fields such as bin_instr_text; fail records without them.",
    )
    ap.add_argument("--fill-id-min", type=int, default=sft_aug.DEFAULT_FILL_ID_MIN)
    ap.add_argument("--num-patterns", type=int, default=10)
    ap.add_argument("--emit-rl-fields", action="store_true")
    ap.add_argument("--mask-prompt-labels", action="store_true")
    ap.add_argument("--mask-non-fill-labels", action="store_true")
    ap.add_argument("--mask-supervised-inputs", action="store_true")
    ap.add_argument("--supervised-input-mask-prob", type=float, default=1.0)
    ap.add_argument("--supervised-input-mask-id", type=int, default=None)
    ap.add_argument("--mask-token", type=str, default=sft_aug.DEFAULT_MASK_TOKEN)
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument(
        "--record-timeout-sec",
        type=float,
        default=60.0,
        help=(
            "Maximum wall-clock seconds for per-record teacher/SFT construction. "
            "Timed-out records are skipped; set <=0 to disable."
        ),
    )
    ap.add_argument(
        "--strict-teacher",
        action="store_true",
        help="Drop noisy teacher slots: duplicate body targets, non-control conditions, and multi-statement targets.",
    )
    ap.add_argument(
        "--strict-teacher-drop-record",
        action="store_true",
        help="With --strict-teacher, skip the whole record when any dangerous teacher slot is rejected.",
    )
    ap.add_argument(
        "--min-supervised-semantic-slots",
        type=int,
        default=1,
        help="Drop records with fewer supervised non-signature slots after teacher cleaning.",
    )
    ap.add_argument(
        "--use-m-gt-for-ab",
        action="store_true",
        help=(
            "Feed M_gt into Module B while generating skeletons. Default is off, "
            "so skeleton/alignment follows inference-time A/B behavior."
        ),
    )
    ap.add_argument(
        "--source-oracle-filter",
        choices=("off", "strict", "hard"),
        default="off",
        help=(
            "Audit rendered teacher output against the real source while building Stage2AB SFT. "
            "'strict' keeps only oracle_ok rows; 'hard' also keeps rows whose only reason is "
            "format_or_control_shape_diff_only."
        ),
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers is required")
        sys.exit(1)

    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    ckpt = args.stage2_ckpt.expanduser().resolve()
    if not inp.is_file():
        logger.error("input not found: %s", inp)
        sys.exit(1)
    if not ckpt.is_file():
        logger.error("Stage2 checkpoint not found: %s", ckpt)
        sys.exit(1)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = _load_stage2_model(ckpt, device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    extra = [f"<|STRUCLIFT_FILL_{i:04d}|>" for i in range(args.max_fill_tokens)]
    if args.mask_supervised_inputs and args.supervised_input_mask_id is None:
        extra.append(str(args.mask_token))
    tokenizer.add_tokens(extra)
    fill_ids = [
        tokenizer.convert_tokens_to_ids(f"<|STRUCLIFT_FILL_{i:04d}|>")
        for i in range(args.max_fill_tokens)
    ]
    low = min(fill_ids)
    if low < args.fill_id_min:
        logger.warning(
            "fill token min id=%d < %d; ensure Stage3 fill range matches tokenizer",
            low,
            args.fill_id_min,
        )

    records = _read_records(inp, args.max_lines)
    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    prefix_pairs: Optional[List[Tuple[str, str]]] = None
    if args.prefix_pairs:
        prefix_pairs = [(str(a), str(b)) for a, b in args.prefix_pairs]

    n_ok = 0
    n_err = 0
    err_examples: List[str] = []
    err_counter: "collections.Counter[str]" = collections.Counter()
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None

    with outp.open("w", encoding="utf-8") as fout:
        t0 = time.time()
        total = len(records)
        bs = max(1, int(args.batch_size))
        for start in range(0, total, bs):
            chunk = records[start : start + bs]
            stage2_res = _run_stage2_for_records(
                model,
                chunk,
                device,
                use_m_gt_for_ab=bool(args.use_m_gt_for_ab),
            )
            for j, (rec, (skel, M, err)) in enumerate(zip(chunk, stage2_res)):
                idx = start + j
                if skel is None:
                    n_err += 1
                    err_counter[str(err).split(":", 1)[0]] += 1
                    if len(err_examples) < 8:
                        err_examples.append(f"line {idx + 1} {rec.get('func_name', '?')}: {err}")
                    continue
                label = f"line {idx + 1} {rec.get('func_name', '?')}"
                try:
                    with _record_timeout(float(args.record_timeout_sec), label):
                        skel, readable_ctx = _upgrade_skeleton_binary_context(skel, rec)
                        new_rec, err2 = sft_aug.process_record(
                            rec,
                            tokenizer,
                            fill_ids,
                            source_root,
                            emit_rl_fields=args.emit_rl_fields,
                            num_patterns=args.num_patterns,
                            mask_prompt_labels=args.mask_prompt_labels,
                            mask_non_fill_labels=args.mask_non_fill_labels,
                            mask_supervised_inputs=args.mask_supervised_inputs,
                            supervised_input_mask_prob=args.supervised_input_mask_prob,
                            supervised_input_mask_id=args.supervised_input_mask_id,
                            mask_token=args.mask_token,
                            max_seq_len=max_seq_len,
                            prefix_rewrites=prefix_pairs,
                            external_skeleton=skel,
                            skeleton_source="stage2_module_b",
                            strict_teacher=bool(args.strict_teacher),
                            strict_teacher_drop_record=bool(args.strict_teacher_drop_record),
                            min_supervised_semantic_slots=int(args.min_supervised_semantic_slots),
                            asm_max_total_lines=int(args.asm_max_total_lines),
                            asm_max_lines_per_bb=int(args.asm_max_lines_per_bb),
                            require_readable_asm=bool(args.require_readable_asm),
                        )
                except _RecordTimeoutError as e:
                    n_err += 1
                    err_counter["record timeout"] += 1
                    if len(err_examples) < 8:
                        err_examples.append(f"{label}: {e}")
                    logger.warning("skip %s", e)
                    continue
                except Exception as e:
                    n_err += 1
                    kind = type(e).__name__
                    err_counter[f"process_record failed: {kind}"] += 1
                    if len(err_examples) < 8:
                        err_examples.append(f"{label}: process_record failed: {kind}: {e}")
                    logger.warning("skip %s: process_record failed: %s: %s", label, kind, e)
                    continue
                if new_rec is None:
                    n_err += 1
                    err_counter[str(err2).split("(", 1)[0].strip()] += 1
                    if len(err_examples) < 8:
                        err_examples.append(f"line {idx + 1} {rec.get('func_name', '?')}: {err2}")
                    continue
                new_rec["sft_stage2_ckpt"] = str(ckpt)
                new_rec["sft_stage2_use_m_gt_for_ab"] = bool(args.use_m_gt_for_ab)
                new_rec["sft_stage2_alignment_shape"] = list(M.shape) if M is not None else None
                new_rec["sft_stage2_skeleton_reliability"] = getattr(skel, "reliability", None)
                new_rec["sft_stage2_skeleton_build_stats"] = getattr(skel, "build_stats", None)
                new_rec["sft_binary_context_readable_asm"] = bool(readable_ctx)
                if args.source_oracle_filter != "off":
                    try:
                        new_rec, oracle_reasons, oracle_hard_reasons = _attach_source_oracle_audit_fields(new_rec)
                    except Exception as e:
                        n_err += 1
                        kind = type(e).__name__
                        err_counter[f"source oracle failed: {kind}"] += 1
                        if len(err_examples) < 8:
                            err_examples.append(f"{label}: source oracle failed: {kind}: {e}")
                        logger.warning("skip %s: source oracle failed: %s: %s", label, kind, e)
                        continue
                    if args.source_oracle_filter == "strict":
                        oracle_keep = bool(new_rec.get("sft_source_oracle_ok"))
                    else:
                        oracle_keep = not oracle_hard_reasons
                    if not oracle_keep:
                        n_err += 1
                        shown = ",".join(oracle_hard_reasons or oracle_reasons[:4] or ["unknown"])
                        err_counter[f"source oracle rejected: {shown}"] += 1
                        if len(err_examples) < 8:
                            err_examples.append(f"{label}: source oracle rejected: {shown}")
                        continue
                fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush()
            done = min(total, start + len(chunk))
            if done % max(1, int(args.log_every)) == 0 or done == total:
                dt = max(1e-6, time.time() - t0)
                speed = done / dt
                eta = (total - done) / speed if speed > 0 else 0.0
                logger.info(
                    "processed input=%d/%d ok=%d err=%d speed=%.2f rec/s eta=%.1f min errors=%s",
                    done,
                    total,
                    n_ok,
                    n_err,
                    speed,
                    eta / 60.0,
                    dict(err_counter.most_common(4)),
                )

    if args.save_tokenizer is not None:
        save_dir = args.save_tokenizer.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        logger.info("saved tokenizer with fill/mask tokens -> %s", save_dir)

    logger.info("done: wrote %d records, failed %d -> %s", n_ok, n_err, outp)
    if err_counter:
        logger.info("failure summary: %s", dict(err_counter.most_common()))
    if err_examples:
        logger.info("failure examples:\n  %s", "\n  ".join(err_examples))


if __name__ == "__main__":
    main()

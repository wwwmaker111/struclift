#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the 500-row Module-C ECSL overfit pipeline.

This runner is intentionally separate from the strict ECSL data builder.  It
first builds/reuses ECSL contracts and candidate tables, then creates a
Stage3-readable overfit pseudo-label JSONL, and can optionally run the Stage3
LLM dry-run or actual overfit fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def run(cmd: List[str], *, cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log.write(proc.stdout)
        log.flush()
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def default_doc(repo_root: Path) -> str:
    local = repo_root / "StrucLift_ModuleC_ECSL_方案.md"
    if local.exists():
        return str(local)
    return "StrucLift_ModuleC_ECSL_方案.md"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Module-C ECSL overfit data + optional Stage3 training.")
    ap.add_argument("--module-b-jsonl", required=True, help="Module-B fullcandidates JSONL, e.g. train500.")
    ap.add_argument("--source-jsonl", default=None, help="Optional original enriched binskel/source JSONL for graph/source tensors.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--doc-md", default=None)
    ap.add_argument("--tokenizer", required=True, help="Base tokenizer/model path before adding StrucLift fill tokens.")
    ap.add_argument("--llm", default=None, help="HF LLM path for Stage3 dry-run/train. Required if --dry-run-stage3 or --train-stage3.")
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-ecsl-rebuild", action="store_true")
    ap.add_argument("--max-candidates-per-slot", type=int, default=8)
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--overfit-max-seq-len", type=int, default=16384)
    ap.add_argument("--stage3-max-sft-seq-len", type=int, default=16384)
    ap.add_argument("--asm-max-total-lines", type=int, default=80)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=8)
    ap.add_argument("--min-candidate-tier", default="C1", choices=["C1", "C2", "C3", "C4", "C5"])
    ap.add_argument("--prefer-candidate-labels", action="store_true")
    ap.add_argument("--strict-source-teacher", action="store_true")
    ap.add_argument("--mask-supervised-inputs", action="store_true")
    ap.add_argument("--rewrite-source-prefix", nargs=2, action="append", metavar=("OLD", "NEW"))
    ap.add_argument("--dry-run-stage3", action="store_true")
    ap.add_argument("--train-stage3", action="store_true")
    ap.add_argument("--stage3-save-dir", default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--llm-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    ap.add_argument("--llm-gradient-checkpointing", action="store_true")
    ap.add_argument("--llm-device-map-auto", action="store_true")
    ap.add_argument("--llm-per-gpu-max-gib", type=float, default=None)
    ap.add_argument("--s3-lambda-compile", type=float, default=0.0)
    ap.add_argument("--enable-v2-validator", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--v2-require-full-dataflow", action="store_true")
    ap.add_argument("--external-v3-jsonl", default=None)
    ap.add_argument("--v3-command", default=None)
    ap.add_argument("--enable-v3-compile-smoke", action="store_true")
    ap.add_argument("--v3-cc", default="cc")
    ap.add_argument("--v3-compile-keep-dir", default=None)
    ap.add_argument("--allow-compile-smoke-as-v3", action="store_true", help="Deprecated compatibility flag; ignored because compile smoke is diagnostic only.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "overfit_pipeline.log"
    if log_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path.rename(out_dir / f"overfit_pipeline.{stamp}.previous.log")

    py = sys.executable
    module_b_jsonl = str(Path(args.module_b_jsonl).expanduser().resolve())
    doc_md = str(Path(args.doc_md).expanduser().resolve()) if args.doc_md else default_doc(repo_root)

    candidates_jsonl = out_dir / "ecsl_slot_candidates.jsonl"
    contracts_jsonl = out_dir / "ecsl_slot_contracts.jsonl"
    decode_jsonl = out_dir / "ecsl_global_decode_seed.jsonl"
    candidate_report = out_dir / "ecsl_candidate_report.txt"
    overfit_jsonl = out_dir / "ecsl_overfit_sft.jsonl"
    overfit_report = out_dir / "ecsl_overfit_sft_report.txt"
    tokenizer_saved = out_dir / "tokenizer_with_struclift_fill"
    stage3_save_dir = Path(args.stage3_save_dir).expanduser().resolve() if args.stage3_save_dir else out_dir / "stage3_overfit"

    if args.force_ecsl_rebuild or not candidates_jsonl.is_file():
        cmd = [
            py,
            "scripts/run_module_c_ecsl_pipeline_local.py",
            "--input-jsonl",
            module_b_jsonl,
            "--out-dir",
            str(out_dir),
            "--repo-root",
            str(repo_root),
            "--doc-md",
            doc_md,
            "--max-candidates-per-slot",
            str(args.max_candidates_per_slot),
            "--include-skeleton-snippets",
            "--emit-keep-external",
            "--emit-llm-proposal-requests",
            "--include-c1-priors",
            "--include-rendered-audit",
        ]
        if int(args.limit) > 0:
            cmd.extend(["--limit", str(args.limit)])
        if bool(args.enable_v2_validator):
            cmd.append("--enable-v2-validator")
        else:
            cmd.append("--no-enable-v2-validator")
        if bool(args.v2_require_full_dataflow):
            cmd.append("--v2-require-full-dataflow")
        if args.external_v3_jsonl:
            cmd.extend(["--external-v3-jsonl", str(Path(args.external_v3_jsonl).expanduser().resolve())])
        if args.v3_command:
            cmd.extend(["--v3-command", str(args.v3_command)])
        if bool(args.enable_v3_compile_smoke):
            cmd.append("--enable-v3-compile-smoke")
        if args.v3_cc:
            cmd.extend(["--v3-cc", str(args.v3_cc)])
        if args.v3_compile_keep_dir:
            cmd.extend(["--v3-compile-keep-dir", str(Path(args.v3_compile_keep_dir).expanduser().resolve())])
        run(cmd, cwd=repo_root, log_path=log_path)
    else:
        print(f"reuse existing ECSL candidates: {candidates_jsonl}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"reuse existing ECSL candidates: {candidates_jsonl}\n")

    cmd = [
        py,
        "scripts/build_module_c_ecsl_overfit_sft.py",
        "--candidates-jsonl",
        str(candidates_jsonl),
        "--module-b-jsonl",
        module_b_jsonl,
        "--out-jsonl",
        str(overfit_jsonl),
        "--out-txt",
        str(overfit_report),
        "--tokenizer",
        str(args.tokenizer),
        "--save-tokenizer",
        str(tokenizer_saved),
        "--max-fill-tokens",
        str(args.max_fill_tokens),
        "--max-seq-len",
        str(args.overfit_max_seq_len),
        "--min-candidate-tier",
        str(args.min_candidate_tier),
        "--asm-max-total-lines",
        str(args.asm_max_total_lines),
        "--asm-max-lines-per-bb",
        str(args.asm_max_lines_per_bb),
    ]
    if args.source_jsonl:
        cmd.extend(["--source-jsonl", str(Path(args.source_jsonl).expanduser().resolve())])
    if int(args.limit) > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.source_root:
        cmd.extend(["--source-root", str(Path(args.source_root).expanduser().resolve())])
    if args.prefer_candidate_labels:
        cmd.append("--prefer-candidate-labels")
    if args.strict_source_teacher:
        cmd.append("--strict-source-teacher")
    if args.mask_supervised_inputs:
        cmd.append("--mask-supervised-inputs")
    for old, new in args.rewrite_source_prefix or []:
        cmd.extend(["--rewrite-source-prefix", old, new])
    run(cmd, cwd=repo_root, log_path=log_path)

    if args.dry_run_stage3 or args.train_stage3:
        if not args.llm:
            raise SystemExit("--llm is required with --dry-run-stage3 or --train-stage3")
        base = [
            py,
            "scripts/train_stage3_binskel_autoconfig.py",
            "--jsonl",
            str(overfit_jsonl),
            "--llm",
            str(args.llm),
            "--tokenizer-saved",
            str(tokenizer_saved),
            "--save-dir",
            str(stage3_save_dir),
            "--batch-size",
            str(args.batch_size),
            "--max-sft-seq-len",
            str(args.stage3_max_sft_seq_len),
            "--s3-lambda-compile",
            str(args.s3_lambda_compile),
            "--llm-dtype",
            str(args.llm_dtype),
        ]
        if args.device:
            base.extend(["--device", str(args.device)])
        if args.init_from:
            base.extend(["--init-from", str(Path(args.init_from).expanduser().resolve())])
        if args.resume:
            base.extend(["--resume", str(Path(args.resume).expanduser().resolve())])
        if args.llm_gradient_checkpointing:
            base.append("--llm-gradient-checkpointing")
        if args.llm_device_map_auto:
            base.append("--llm-device-map-auto")
        if args.llm_per_gpu_max_gib is not None:
            base.extend(["--llm-per-gpu-max-gib", str(args.llm_per_gpu_max_gib)])

        if args.dry_run_stage3:
            run(base + ["--dry-run"], cwd=repo_root, log_path=log_path)
        if args.train_stage3:
            run(base + ["--epochs", str(args.epochs)], cwd=repo_root, log_path=log_path)

    summary = {
        "out_dir": str(out_dir),
        "module_b_jsonl": module_b_jsonl,
        "contracts_jsonl": str(contracts_jsonl),
        "candidates_jsonl": str(candidates_jsonl),
        "decode_jsonl": str(decode_jsonl),
        "candidate_report": str(candidate_report),
        "overfit_jsonl": str(overfit_jsonl),
        "overfit_report": str(overfit_report),
        "tokenizer_saved": str(tokenizer_saved),
        "stage3_save_dir": str(stage3_save_dir),
        "overfit_pipeline_log": str(log_path),
    }
    summary_path = out_dir / "overfit_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== Module C ECSL overfit pipeline finished ===")
    for key, value in summary.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Module-C ECSL strong-label data build for a 500-row audit/overfit pass.

This runner follows the strong-label-first policy:

1. Build/reuse ECSL slot contracts and candidates from verified Module-B rows.
2. Build a strict Stage3 SFT JSONL only from C3+ rows.
3. Run V2 contract validation in the ECSL candidate builder.
4. Require explicit external V3/differential results before any row becomes a hard label.
5. Refuse Stage3 dry-run/train when the hard SFT JSONL is empty.

The older overfit runner remains useful only as a plumbing smoke test.  This
runner is the one to use before claiming semantic Module-C training data.
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


def run_shell(command: str, *, cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + command, flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + command + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            shell=True,
        )
        log.write(proc.stdout)
        log.flush()
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}: {command}")


def default_doc(repo_root: Path) -> str:
    local = repo_root / "StrucLift_ModuleC_ECSL_方案.md"
    if local.exists():
        return str(local)
    return "StrucLift_ModuleC_ECSL_方案.md"


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Module-C ECSL strong-label build + optional Stage3 strong overfit.")
    ap.add_argument("--module-b-jsonl", required=True, help="Module-B guardedrepair fullcandidates JSONL.")
    ap.add_argument("--source-jsonl", default=None, help="Optional original enriched binskel/source JSONL for graph/source tensors.")
    ap.add_argument("--v3-results-jsonl", default=None, help="External symbolic/trace/differential V3 validation JSONL. Required to produce C3+ hard labels.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--doc-md", default=None)
    ap.add_argument("--tokenizer", required=True, help="Base tokenizer/model path before adding StrucLift fill tokens.")
    ap.add_argument("--llm", default=None, help="HF LLM path for Stage3 dry-run/train. Required if --dry-run-stage3 or --train-stage3.")
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-ecsl-rebuild", action="store_true")
    ap.add_argument(
        "--build-real-function-ir",
        action="store_true",
        help=(
            "Build binary-derived FunctionIR/register SSA from --source-jsonl and merge it back into "
            "--module-b-jsonl before Module-C contracts are built."
        ),
    )
    ap.add_argument(
        "--real-function-ir-jsonl",
        default=None,
        help="Existing enriched source JSONL from enrich_module_b_v3_real_function_ir100.py to merge into Module-B.",
    )
    ap.add_argument("--real-function-ir-max-examples", type=int, default=0)
    ap.add_argument("--real-function-ir-binary-root-map", action="append", default=[])
    ap.add_argument("--real-function-ir-path-search-root", action="append", default=[])
    ap.add_argument("--real-function-ir-binary-search-root", action="append", default=[])
    ap.add_argument("--real-function-ir-recover-missing-binary-path", action="store_true")
    ap.add_argument("--real-function-ir-require-cfg-shape-match", action="store_true")
    ap.add_argument("--real-function-ir-no-proxy-fallback", action="store_true")
    ap.add_argument("--max-candidates-per-slot", type=int, default=8)
    ap.add_argument("--max-fill-tokens", type=int, default=512)
    ap.add_argument("--strong-max-seq-len", type=int, default=16384)
    ap.add_argument("--stage3-max-sft-seq-len", type=int, default=16384)
    ap.add_argument("--asm-max-total-lines", type=int, default=0)
    ap.add_argument("--asm-max-lines-per-bb", type=int, default=0)
    ap.add_argument("--prompt-compact-mode", default="ultra", choices=["auto", "full", "compact", "ultra"])
    ap.add_argument("--max-slot-context-chars", type=int, default=256)
    ap.add_argument("--min-hard-tier", default="C3", choices=["C3", "C4", "C5"])
    ap.add_argument("--min-supervised-semantic-slots", type=int, default=1)
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
    ap.add_argument(
        "--build-source-teacher-v3-oracle",
        action="store_true",
        help="Build a training-time source-teacher projection JSONL for coverage audit; it is not true V3 and is not auto-fed to hard SFT.",
    )
    ap.add_argument(
        "--build-v3-differential-lite",
        action="store_true",
        help=(
            "Build diagnostic V3-lite differential results from candidates plus source-teacher projection. "
            "This is a conservative subset audit and is not fed to hard SFT unless "
            "--use-v3-lite-as-experimental-hard-v3 is also set."
        ),
    )
    ap.add_argument(
        "--use-v3-lite-as-experimental-hard-v3",
        action="store_true",
        help=(
            "Experimental only: feed V3-lite rows into hard SFT as C3 evidence. "
            "Do not use when claiming one-to-one compliance with the ECSL document's full V3 "
            "trace/symbolic/differential validator."
        ),
    )
    ap.add_argument(
        "--v3-lite-source-projection-jsonl",
        default=None,
        help="Existing source-teacher projection JSONL to use as V3-lite reference. Defaults to this run's projection output.",
    )
    ap.add_argument("--v3-lite-cc", default="cc")
    ap.add_argument("--v3-lite-timeout", type=int, default=10)
    ap.add_argument("--v3-lite-max-tests", type=int, default=256)
    ap.add_argument("--v3-lite-keep-dir", default=None)
    ap.add_argument("--v3-lite-require-candidate-v2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--v3-lite-allow-single-var-alpha-equivalence", action="store_true")
    ap.add_argument("--v3-lite-allow-n-var-alpha-equivalence", action="store_true")
    ap.add_argument("--source-teacher-v3-compile-policy", default="record", choices=["skip", "record", "require"])
    ap.add_argument("--source-teacher-v3-min-compile-reward", type=float, default=1.0)
    ap.add_argument("--source-teacher-v3-cc", default="gcc")
    ap.add_argument("--source-teacher-v3-label-policy", default="slot", choices=["slot", "function"])
    ap.add_argument("--source-teacher-v3-pass-tier", default="C2", choices=["C2", "C3", "C4", "C5"])
    ap.add_argument("--enable-v2-validator", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--v0-cc", default="cc")
    ap.add_argument("--v2-require-full-dataflow", action="store_true")
    ap.add_argument("--external-v3-jsonl", default=None, help="External slot-candidate V3 JSONL consumed during candidate validation.")
    ap.add_argument("--v3-command", default=None, help="Optional shell command consumed during candidate validation for V3.")
    ap.add_argument(
        "--export-v3-validation-tasks",
        action="store_true",
        help=(
            "Export V0/V1/V2-passing candidate tasks for a real external V3 "
            "trace/symbolic/differential validator. This task file is not V3 evidence."
        ),
    )
    ap.add_argument("--v3-task-require-v2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--v3-task-allow-partial-v2", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--v3-task-allow-source-projection", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--v3-task-max-tasks-per-function", type=int, default=0)
    ap.add_argument(
        "--v3-validator-command",
        default=None,
        help=(
            "Shell command for a real V3 validator. Placeholders: {tasks_jsonl}, {out_jsonl}, "
            "{out_txt}, {out_dir}, {repo_root}. The command must write JSONL rows with "
            "differential_pass, trace_match, or symbolic_match plus C3+ confidence."
        ),
    )
    ap.add_argument(
        "--build-v3-backend",
        action="store_true",
        help=(
            "Run the built-in V3 backend over exported V3 tasks. It performs compiled differential "
            "or symbolic structural checks against source-teacher slot references and writes "
            "ecsl_v3_backend_results.jsonl."
        ),
    )
    ap.add_argument("--v3-validator-results-jsonl", default=None)
    ap.add_argument("--v3-validator-report", default=None)
    ap.add_argument("--v3-backend-cc", default="cc")
    ap.add_argument("--v3-backend-timeout", type=int, default=10)
    ap.add_argument("--v3-backend-max-tests", type=int, default=256)
    ap.add_argument("--v3-backend-keep-dir", default=None)
    ap.add_argument("--v3-backend-allow-single-var-alpha-equivalence", action="store_true")
    ap.add_argument("--v3-backend-allow-n-var-alpha-equivalence", action="store_true")
    ap.add_argument("--v3-backend-allow-source-projection-candidate-self-proof", action="store_true")
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
    log_path = out_dir / "strong500_pipeline.log"
    if log_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path.rename(out_dir / f"strong500_pipeline.{stamp}.previous.log")

    py = sys.executable
    module_b_jsonl = str(Path(args.module_b_jsonl).expanduser().resolve())
    doc_md = str(Path(args.doc_md).expanduser().resolve()) if args.doc_md else default_doc(repo_root)

    real_function_ir_jsonl = out_dir / "module_b_source_real_function_ir.jsonl"
    real_function_ir_report = out_dir / "module_b_source_real_function_ir_report.txt"
    merged_module_b_jsonl = out_dir / "module_b_with_real_function_ir.jsonl"
    merged_module_b_report = out_dir / "module_b_with_real_function_ir_report.txt"
    candidates_jsonl = out_dir / "ecsl_slot_candidates.jsonl"
    effective_candidates_jsonl = candidates_jsonl
    contracts_jsonl = out_dir / "ecsl_slot_contracts.jsonl"
    decode_jsonl = out_dir / "ecsl_global_decode_seed.jsonl"
    candidate_report = out_dir / "ecsl_candidate_report.txt"
    strong_jsonl = out_dir / "ecsl_strong_sft.jsonl"
    strong_audit_jsonl = out_dir / "ecsl_strong_sft_audit.jsonl"
    strong_report = out_dir / "ecsl_strong_sft_report.txt"
    source_teacher_v3_jsonl = out_dir / "ecsl_source_teacher_projection_results.jsonl"
    source_teacher_v3_audit_jsonl = out_dir / "ecsl_source_teacher_projection_audit.jsonl"
    source_teacher_v3_report = out_dir / "ecsl_source_teacher_projection_report.txt"
    candidates_with_source_projection_jsonl = out_dir / "ecsl_slot_candidates_with_source_projection.jsonl"
    source_projection_injection_report = out_dir / "ecsl_source_projection_candidate_injection_report.txt"
    v3_tasks_jsonl = out_dir / "ecsl_v3_validation_tasks.jsonl"
    v3_tasks_report = out_dir / "ecsl_v3_validation_tasks_report.txt"
    v3_backend_results_jsonl = (
        Path(args.v3_validator_results_jsonl).expanduser().resolve()
        if args.v3_validator_results_jsonl
        else out_dir / "ecsl_v3_backend_results.jsonl"
    )
    v3_backend_audit_jsonl = out_dir / "ecsl_v3_backend_audit.jsonl"
    v3_backend_report = (
        Path(args.v3_validator_report).expanduser().resolve()
        if args.v3_validator_report
        else out_dir / "ecsl_v3_backend_report.txt"
    )
    v3_lite_jsonl = out_dir / "ecsl_v3_differential_lite_results.jsonl"
    v3_lite_audit_jsonl = out_dir / "ecsl_v3_differential_lite_audit.jsonl"
    v3_lite_report = out_dir / "ecsl_v3_differential_lite_report.txt"
    tokenizer_saved = out_dir / "tokenizer_with_struclift_fill"
    stage3_save_dir = Path(args.stage3_save_dir).expanduser().resolve() if args.stage3_save_dir else out_dir / "stage3_strong_overfit"

    if args.build_real_function_ir or args.real_function_ir_jsonl:
        if not args.source_jsonl and not args.real_function_ir_jsonl:
            raise SystemExit("--source-jsonl is required with --build-real-function-ir")
        source_real_ir = (
            Path(args.real_function_ir_jsonl).expanduser().resolve()
            if args.real_function_ir_jsonl
            else real_function_ir_jsonl
        )
        if args.build_real_function_ir and not source_real_ir.is_file():
            max_examples = int(args.real_function_ir_max_examples or args.limit or 500)
            cmd = [
                py,
                "scripts/enrich_module_b_v3_real_function_ir100.py",
                "--source-jsonl",
                str(Path(args.source_jsonl).expanduser().resolve()),
                "--teacher-jsonl",
                module_b_jsonl,
                "--out-jsonl",
                str(source_real_ir),
                "--out-txt",
                str(real_function_ir_report),
                "--max-examples",
                str(max_examples),
            ]
            for item in args.real_function_ir_binary_root_map or []:
                cmd.extend(["--binary-root-map", str(item)])
            for item in args.real_function_ir_path_search_root or []:
                cmd.extend(["--path-search-root", str(Path(item).expanduser().resolve())])
            for item in args.real_function_ir_binary_search_root or []:
                cmd.extend(["--binary-search-root", str(Path(item).expanduser().resolve())])
            if bool(args.real_function_ir_recover_missing_binary_path):
                cmd.append("--recover-missing-binary-path")
            if bool(args.real_function_ir_require_cfg_shape_match):
                cmd.append("--require-cfg-shape-match")
            if bool(args.real_function_ir_no_proxy_fallback):
                cmd.append("--no-proxy-fallback")
            run(cmd, cwd=repo_root, log_path=log_path)
        elif source_real_ir.is_file():
            print(f"reuse existing real FunctionIR JSONL: {source_real_ir}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"reuse existing real FunctionIR JSONL: {source_real_ir}\n")
        else:
            raise SystemExit(f"real FunctionIR JSONL not found: {source_real_ir}")

        cmd = [
            py,
            "scripts/merge_module_b_v3_real_function_ir.py",
            "--module-b-jsonl",
            module_b_jsonl,
            "--real-ir-jsonl",
            str(source_real_ir),
            "--out-jsonl",
            str(merged_module_b_jsonl),
            "--out-txt",
            str(merged_module_b_report),
        ]
        # Do not forward the function-row limit to the V3 backend.  At this
        # point ``ecsl_v3_validation_tasks.jsonl`` may contain many candidates
        # per function; truncating it to the first N tasks silently drops most
        # V3 opportunities and makes hard-SFT coverage look much worse than it
        # is.
        run(cmd, cwd=repo_root, log_path=log_path)
        module_b_jsonl = str(merged_module_b_jsonl)
        print("real FunctionIR/SSA was merged into Module-B rows and will be used for Module-C V2 contracts.")

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
        if args.v0_cc:
            cmd.extend(["--v0-cc", str(args.v0_cc)])
        if bool(args.v2_require_full_dataflow):
            cmd.append("--v2-require-full-dataflow")
        candidate_v3_jsonl = args.external_v3_jsonl or args.v3_results_jsonl
        if candidate_v3_jsonl:
            cmd.extend(["--external-v3-jsonl", str(Path(candidate_v3_jsonl).expanduser().resolve())])
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

    effective_v3_results_jsonl = args.v3_results_jsonl
    true_v3_required = bool(args.dry_run_stage3 or args.train_stage3)
    if args.build_source_teacher_v3_oracle:
        if not args.source_jsonl:
            raise SystemExit("--source-jsonl is required with --build-source-teacher-v3-oracle")
        cmd = [
            py,
            "scripts/build_module_c_ecsl_source_teacher_v3_oracle.py",
            "--candidates-jsonl",
            str(candidates_jsonl),
            "--module-b-jsonl",
            module_b_jsonl,
            "--source-jsonl",
            str(Path(args.source_jsonl).expanduser().resolve()),
            "--out-jsonl",
            str(source_teacher_v3_jsonl),
            "--out-audit-jsonl",
            str(source_teacher_v3_audit_jsonl),
            "--out-txt",
            str(source_teacher_v3_report),
            "--tokenizer",
            str(args.tokenizer),
            "--max-fill-tokens",
            str(args.max_fill_tokens),
            "--max-seq-len",
            str(args.strong_max_seq_len),
            "--min-supervised-semantic-slots",
            str(args.min_supervised_semantic_slots),
            "--asm-max-total-lines",
            str(args.asm_max_total_lines),
            "--asm-max-lines-per-bb",
            str(args.asm_max_lines_per_bb),
            "--prompt-compact-mode",
            str(args.prompt_compact_mode),
            "--max-slot-context-chars",
            str(args.max_slot_context_chars),
            "--compile-policy",
            str(args.source_teacher_v3_compile_policy),
            "--min-compile-reward",
            str(args.source_teacher_v3_min_compile_reward),
            "--cc",
            str(args.source_teacher_v3_cc),
            "--label-policy",
            str(args.source_teacher_v3_label_policy),
            "--pass-tier",
            str(args.source_teacher_v3_pass_tier),
        ]
        if int(args.limit) > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.source_root:
            cmd.extend(["--source-root", str(Path(args.source_root).expanduser().resolve())])
        if args.mask_supervised_inputs:
            cmd.append("--mask-supervised-inputs")
        for old, new in args.rewrite_source_prefix or []:
            cmd.extend(["--rewrite-source-prefix", old, new])
        run(cmd, cwd=repo_root, log_path=log_path)
        print(
            "source-teacher projection was built for audit only; "
            "not using it as --v3-results-jsonl because it is not differential/trace/symbolic V3."
        )

        cmd = [
            py,
            "scripts/inject_module_c_ecsl_source_projection_candidates.py",
            "--candidates-jsonl",
            str(candidates_jsonl),
            "--source-projection-jsonl",
            str(source_teacher_v3_jsonl),
            "--out-candidates-jsonl",
            str(candidates_with_source_projection_jsonl),
            "--out-txt",
            str(source_projection_injection_report),
        ]
        if int(args.limit) > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.v0_cc:
            cmd.extend(["--v0-cc", str(args.v0_cc)])
        run(cmd, cwd=repo_root, log_path=log_path)
        effective_candidates_jsonl = candidates_with_source_projection_jsonl
        print(
            "source-teacher projection candidates were injected as concrete candidates; "
            "only clean V0/V1/V2 rows become C2, and they still require independent V3 or human audit before hard SFT."
        )

    if args.export_v3_validation_tasks:
        cmd = [
            py,
            "scripts/export_module_c_ecsl_v3_validation_tasks.py",
            "--candidates-jsonl",
            str(effective_candidates_jsonl),
            "--out-jsonl",
            str(v3_tasks_jsonl),
            "--out-txt",
            str(v3_tasks_report),
        ]
        if int(args.limit) > 0:
            cmd.extend(["--limit", str(args.limit)])
        if int(args.v3_task_max_tasks_per_function) > 0:
            cmd.extend(["--max-tasks-per-function", str(args.v3_task_max_tasks_per_function)])
        if bool(args.v3_task_require_v2):
            cmd.append("--require-v2")
        else:
            cmd.append("--no-require-v2")
        if bool(args.v3_task_allow_partial_v2):
            cmd.append("--allow-partial-v2-tasks")
        if bool(args.v3_task_allow_source_projection):
            cmd.append("--allow-source-projection-tasks")
        else:
            cmd.append("--no-allow-source-projection-tasks")
        run(cmd, cwd=repo_root, log_path=log_path)
        print(
            "V3 validation tasks were exported for an external trace/symbolic/differential backend. "
            "They are tasks, not labels; convert backend pass results into --v3-results-jsonl before hard SFT."
        )

    if args.build_v3_backend:
        if not v3_tasks_jsonl.is_file():
            raise SystemExit("--build-v3-backend requires --export-v3-validation-tasks")
        projection_jsonl = source_teacher_v3_jsonl
        if not projection_jsonl.is_file():
            raise SystemExit("--build-v3-backend requires --build-source-teacher-v3-oracle in this training-source backend")
        cmd = [
            py,
            "scripts/run_module_c_ecsl_v3_backend.py",
            "--tasks-jsonl",
            str(v3_tasks_jsonl),
            "--source-projection-jsonl",
            str(projection_jsonl),
            "--out-jsonl",
            str(v3_backend_results_jsonl),
            "--out-audit-jsonl",
            str(v3_backend_audit_jsonl),
            "--out-txt",
            str(v3_backend_report),
            "--cc",
            str(args.v3_backend_cc),
            "--timeout",
            str(args.v3_backend_timeout),
            "--max-tests",
            str(args.v3_backend_max_tests),
        ]
        if args.v3_backend_keep_dir:
            cmd.extend(["--keep-dir", str(Path(args.v3_backend_keep_dir).expanduser().resolve())])
        if bool(args.v3_backend_allow_single_var_alpha_equivalence):
            cmd.append("--allow-single-var-alpha-equivalence")
        if bool(args.v3_backend_allow_n_var_alpha_equivalence):
            cmd.append("--allow-n-var-alpha-equivalence")
        if bool(args.v3_backend_allow_source_projection_candidate_self_proof):
            cmd.append("--allow-source-projection-candidate-self-proof")
        run(cmd, cwd=repo_root, log_path=log_path)
        effective_v3_results_jsonl = str(v3_backend_results_jsonl)
        print(
            "built-in V3 backend results will be used as --v3-results-jsonl; "
            "only compiled differential or symbolic structural pass rows can become C3 hard labels."
        )

    if args.v3_validator_command:
        if not v3_tasks_jsonl.is_file():
            raise SystemExit(
                "--v3-validator-command needs exported V3 tasks. Add --export-v3-validation-tasks "
                "or provide --v3-results-jsonl from a previous backend run."
            )
        rendered_command = str(args.v3_validator_command).format(
            tasks_jsonl=str(v3_tasks_jsonl),
            out_jsonl=str(v3_backend_results_jsonl),
            out_txt=str(v3_backend_report),
            out_dir=str(out_dir),
            repo_root=str(repo_root),
        )
        run_shell(rendered_command, cwd=repo_root, log_path=log_path)
        if not v3_backend_results_jsonl.is_file():
            raise SystemExit(
                f"V3 validator command finished but did not create expected results JSONL: {v3_backend_results_jsonl}"
            )
        effective_v3_results_jsonl = str(v3_backend_results_jsonl)
        print(
            "external V3 backend results will be used as --v3-results-jsonl; "
            "only rows with differential_pass/trace_match/symbolic_match and C3+ confidence can become hard labels."
        )

    if args.build_v3_differential_lite:
        projection_jsonl = (
            Path(args.v3_lite_source_projection_jsonl).expanduser().resolve()
            if args.v3_lite_source_projection_jsonl
            else source_teacher_v3_jsonl
        )
        if not projection_jsonl.is_file():
            raise SystemExit(
                "--build-v3-differential-lite needs a source projection JSONL. "
                "Use --build-source-teacher-v3-oracle or pass --v3-lite-source-projection-jsonl."
            )
        cmd = [
            py,
            "scripts/run_module_c_ecsl_v3_differential_lite.py",
            "--candidates-jsonl",
            str(effective_candidates_jsonl),
            "--source-projection-jsonl",
            str(projection_jsonl),
            "--out-jsonl",
            str(v3_lite_jsonl),
            "--out-audit-jsonl",
            str(v3_lite_audit_jsonl),
            "--out-txt",
            str(v3_lite_report),
            "--cc",
            str(args.v3_lite_cc),
            "--timeout",
            str(args.v3_lite_timeout),
            "--max-tests",
            str(args.v3_lite_max_tests),
        ]
        if int(args.limit) > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.v3_lite_keep_dir:
            cmd.extend(["--keep-dir", str(Path(args.v3_lite_keep_dir).expanduser().resolve())])
        if bool(args.v3_lite_require_candidate_v2):
            cmd.append("--require-candidate-v2")
        else:
            cmd.append("--no-require-candidate-v2")
        if bool(args.v3_lite_allow_single_var_alpha_equivalence):
            cmd.append("--allow-single-var-alpha-equivalence")
        if bool(args.v3_lite_allow_n_var_alpha_equivalence):
            cmd.append("--allow-n-var-alpha-equivalence")
        if bool(args.use_v3_lite_as_experimental_hard_v3):
            cmd.append("--mark-hard-ce-allowed")
        run(cmd, cwd=repo_root, log_path=log_path)
        if bool(args.use_v3_lite_as_experimental_hard_v3):
            effective_v3_results_jsonl = str(v3_lite_jsonl)
            print(
                "WARNING: V3-lite diagnostic rows are being used as experimental hard V3 evidence. "
                "This is not the full ECSL V3 trace/symbolic/differential validator."
            )
        else:
            print(
                "V3-lite diagnostic rows were built for audit only; they are not used as --v3-results-jsonl. "
                "Provide --v3-results-jsonl from a true trace/symbolic/differential backend for C3 hard labels."
            )

    if true_v3_required and not effective_v3_results_jsonl:
        raise SystemExit(
            "No true --v3-results-jsonl was supplied. Refusing strong Stage3 run because ECSL C3 hard labels "
            "require trace/symbolic/differential V3 evidence. Build/export V3 validation tasks first, run a real "
            "validator, then rerun with --v3-results-jsonl."
        )

    cmd = [
        py,
        "scripts/build_module_c_ecsl_strong_sft.py",
        "--candidates-jsonl",
        str(effective_candidates_jsonl),
        "--module-b-jsonl",
        module_b_jsonl,
        "--out-hard-jsonl",
        str(strong_jsonl),
        "--out-audit-jsonl",
        str(strong_audit_jsonl),
        "--out-txt",
        str(strong_report),
        "--tokenizer",
        str(args.tokenizer),
        "--save-tokenizer",
        str(tokenizer_saved),
        "--max-fill-tokens",
        str(args.max_fill_tokens),
        "--max-seq-len",
        str(args.strong_max_seq_len),
        "--min-hard-tier",
        str(args.min_hard_tier),
        "--min-supervised-semantic-slots",
        str(args.min_supervised_semantic_slots),
        "--asm-max-total-lines",
        str(args.asm_max_total_lines),
        "--asm-max-lines-per-bb",
        str(args.asm_max_lines_per_bb),
        "--prompt-compact-mode",
        str(args.prompt_compact_mode),
        "--max-slot-context-chars",
        str(args.max_slot_context_chars),
    ]
    if args.source_jsonl:
        cmd.extend(["--source-jsonl", str(Path(args.source_jsonl).expanduser().resolve())])
    if effective_v3_results_jsonl:
        cmd.extend(["--v3-results-jsonl", str(Path(effective_v3_results_jsonl).expanduser().resolve())])
    if bool(args.use_v3_lite_as_experimental_hard_v3):
        cmd.append("--allow-diagnostic-v3-as-hard-label")
    if int(args.limit) > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.source_root:
        cmd.extend(["--source-root", str(Path(args.source_root).expanduser().resolve())])
    if args.mask_supervised_inputs:
        cmd.append("--mask-supervised-inputs")
    for old, new in args.rewrite_source_prefix or []:
        cmd.extend(["--rewrite-source-prefix", old, new])
    run(cmd, cwd=repo_root, log_path=log_path)

    hard_rows = count_jsonl_rows(strong_jsonl)
    if (args.dry_run_stage3 or args.train_stage3) and hard_rows <= 0:
        raise SystemExit(
            "hard_sft_records=0; refusing Stage3 training. "
            "This usually means no external V3 C3+ validation was supplied. "
            f"Inspect {strong_report} and produce --v3-results-jsonl first."
        )

    if args.dry_run_stage3 or args.train_stage3:
        if not args.llm:
            raise SystemExit("--llm is required with --dry-run-stage3 or --train-stage3")
        base = [
            py,
            "scripts/train_stage3_binskel_autoconfig.py",
            "--jsonl",
            str(strong_jsonl),
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
        "source_jsonl": str(Path(args.source_jsonl).expanduser().resolve()) if args.source_jsonl else None,
        "real_function_ir_jsonl": str(real_function_ir_jsonl) if args.build_real_function_ir else (str(Path(args.real_function_ir_jsonl).expanduser().resolve()) if args.real_function_ir_jsonl else None),
        "real_function_ir_report": str(real_function_ir_report) if args.build_real_function_ir else None,
        "merged_module_b_jsonl": str(merged_module_b_jsonl) if (args.build_real_function_ir or args.real_function_ir_jsonl) else None,
        "merged_module_b_report": str(merged_module_b_report) if (args.build_real_function_ir or args.real_function_ir_jsonl) else None,
        "v3_results_jsonl": str(Path(effective_v3_results_jsonl).expanduser().resolve()) if effective_v3_results_jsonl else None,
        "v3_backend_results_jsonl": str(v3_backend_results_jsonl) if (args.build_v3_backend or args.v3_validator_command) else None,
        "v3_backend_audit_jsonl": str(v3_backend_audit_jsonl) if args.build_v3_backend else None,
        "v3_backend_report": str(v3_backend_report) if (args.build_v3_backend or args.v3_validator_command) else None,
        "source_teacher_projection_jsonl": str(source_teacher_v3_jsonl) if args.build_source_teacher_v3_oracle else None,
        "source_teacher_projection_report": str(source_teacher_v3_report) if args.build_source_teacher_v3_oracle else None,
        "candidates_with_source_projection_jsonl": str(candidates_with_source_projection_jsonl) if args.build_source_teacher_v3_oracle else None,
        "source_projection_injection_report": str(source_projection_injection_report) if args.build_source_teacher_v3_oracle else None,
        "v3_validation_tasks_jsonl": str(v3_tasks_jsonl) if args.export_v3_validation_tasks else None,
        "v3_validation_tasks_report": str(v3_tasks_report) if args.export_v3_validation_tasks else None,
        "v3_differential_lite_jsonl": str(v3_lite_jsonl) if args.build_v3_differential_lite else None,
        "v3_differential_lite_audit_jsonl": str(v3_lite_audit_jsonl) if args.build_v3_differential_lite else None,
        "v3_differential_lite_report": str(v3_lite_report) if args.build_v3_differential_lite else None,
        "contracts_jsonl": str(contracts_jsonl),
        "candidates_jsonl": str(candidates_jsonl),
        "effective_candidates_jsonl": str(effective_candidates_jsonl),
        "decode_jsonl": str(decode_jsonl),
        "candidate_report": str(candidate_report),
        "strong_jsonl": str(strong_jsonl),
        "strong_audit_jsonl": str(strong_audit_jsonl),
        "strong_report": str(strong_report),
        "hard_sft_records": hard_rows,
        "tokenizer_saved": str(tokenizer_saved),
        "stage3_save_dir": str(stage3_save_dir),
        "strong500_pipeline_log": str(log_path),
    }
    summary_path = out_dir / "strong500_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== Module C ECSL strong500 pipeline finished ===")
    for key, value in summary.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the local Module-C ECSL data-building pipeline.

The runner builds contracts, slot candidates, validator reports, manifests, and
global decode seeds.  V2 is a local conservative contract validator.  V3 can be
connected through an external symbolic/trace/differential JSONL/command, and an
optional compile-smoke harness records diagnostic compile evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def run(cmd: List[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    ap = argparse.ArgumentParser(description="Run local Module-C ECSL builders and audits.")
    ap.add_argument("--input-jsonl", required=True, help="Module-B fullcandidates JSONL.")
    ap.add_argument("--out-dir", required=True, help="Output directory for all Module-C ECSL artifacts.")
    ap.add_argument("--repo-root", default=".", help="StructLift repo root.")
    ap.add_argument("--doc-md", default=None, help="ECSL design markdown for coverage audit.")
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    ap.add_argument("--max-candidates-per-slot", type=int, default=8)
    ap.add_argument("--min-train-tier", default="C2", choices=["C1", "C2", "C3", "C4", "C5"])
    ap.add_argument("--pack-min-tier", default="C2", choices=["C2", "C3", "C4", "C5"])
    ap.add_argument("--include-skeleton-snippets", action="store_true")
    ap.add_argument("--emit-keep-external", action="store_true")
    ap.add_argument("--emit-llm-proposal-requests", action="store_true")
    ap.add_argument("--include-c1-priors", action="store_true")
    ap.add_argument("--include-rendered-audit", action="store_true")
    ap.add_argument("--enable-v2-validator", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--v0-cc", default="cc")
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
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"
    if log_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path.rename(out_dir / f"pipeline.{stamp}.previous.log")

    py = sys.executable
    input_jsonl = str(Path(args.input_jsonl).resolve())
    doc_md = str(Path(args.doc_md).resolve()) if args.doc_md else default_doc(repo_root)

    contracts_jsonl = out_dir / "ecsl_slot_contracts.jsonl"
    contracts_report = out_dir / "ecsl_slot_contracts_report.txt"
    candidates_jsonl = out_dir / "ecsl_slot_candidates.jsonl"
    train_jsonl = out_dir / "ecsl_slot_train_records.jsonl"
    decode_jsonl = out_dir / "ecsl_global_decode_seed.jsonl"
    candidate_report = out_dir / "ecsl_candidate_report.txt"
    manifest_dir = out_dir / "training_manifest"
    decode_audit_jsonl = out_dir / "ecsl_global_decode_audit.jsonl"
    decode_audit_txt = out_dir / "ecsl_global_decode_audit.txt"
    coverage_json = out_dir / "ecsl_algorithm_coverage.json"
    coverage_txt = out_dir / "ecsl_algorithm_coverage.txt"

    cmd = [
        py,
        "scripts/build_module_c_ecsl_slot_contracts.py",
        "--input-jsonl",
        input_jsonl,
        "--out-jsonl",
        str(contracts_jsonl),
        "--out-txt",
        str(contracts_report),
    ]
    if int(args.limit) > 0:
        cmd.extend(["--limit", str(args.limit)])
    if bool(args.include_skeleton_snippets):
        cmd.append("--include-skeleton-snippets")
    run(cmd, cwd=repo_root, log_path=log_path)

    cmd = [
        py,
        "scripts/build_module_c_ecsl_candidates.py",
        "--contracts-jsonl",
        str(contracts_jsonl),
        "--out-candidates-jsonl",
        str(candidates_jsonl),
        "--out-train-jsonl",
        str(train_jsonl),
        "--out-decode-jsonl",
        str(decode_jsonl),
        "--out-txt",
        str(candidate_report),
        "--max-candidates-per-slot",
        str(args.max_candidates_per_slot),
        "--min-train-tier",
        str(args.min_train_tier),
    ]
    if int(args.limit) > 0:
        cmd.extend(["--limit", str(args.limit)])
    if bool(args.emit_keep_external):
        cmd.append("--emit-keep-external")
    if bool(args.emit_llm_proposal_requests):
        cmd.append("--emit-llm-proposal-requests")
    if bool(args.include_c1_priors):
        cmd.append("--include-c1-priors")
    if bool(args.enable_v2_validator):
        cmd.append("--enable-v2-validator")
    else:
        cmd.append("--no-enable-v2-validator")
    if args.v0_cc:
        cmd.extend(["--v0-cc", str(args.v0_cc)])
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

    run(
        [
            py,
            "scripts/pack_module_c_ecsl_training_manifest.py",
            "--train-jsonl",
            str(train_jsonl),
            "--out-dir",
            str(manifest_dir),
            "--min-tier",
            str(args.pack_min_tier),
        ],
        cwd=repo_root,
        log_path=log_path,
    )

    cmd = [
        py,
        "scripts/audit_module_c_ecsl_global_decode_seed.py",
        "--decode-jsonl",
        str(decode_jsonl),
        "--out-jsonl",
        str(decode_audit_jsonl),
        "--out-txt",
        str(decode_audit_txt),
    ]
    if int(args.limit) > 0:
        cmd.extend(["--limit", str(args.limit)])
    if bool(args.include_rendered_audit):
        cmd.append("--include-rendered")
    run(cmd, cwd=repo_root, log_path=log_path)

    run(
        [
            py,
            "scripts/audit_module_c_ecsl_algorithm_coverage.py",
            "--doc-md",
            doc_md,
            "--repo-root",
            str(repo_root),
            "--out-json",
            str(coverage_json),
            "--out-txt",
            str(coverage_txt),
        ],
        cwd=repo_root,
        log_path=log_path,
    )

    summary = {
        "out_dir": str(out_dir),
        "contracts_jsonl": str(contracts_jsonl),
        "contracts_report": str(contracts_report),
        "candidates_jsonl": str(candidates_jsonl),
        "train_jsonl": str(train_jsonl),
        "decode_jsonl": str(decode_jsonl),
        "candidate_report": str(candidate_report),
        "training_manifest_dir": str(manifest_dir),
        "decode_audit_txt": str(decode_audit_txt),
        "algorithm_coverage_txt": str(coverage_txt),
        "pipeline_log": str(log_path),
    }
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=== Module C ECSL pipeline finished ===")
    for key, value in summary.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()

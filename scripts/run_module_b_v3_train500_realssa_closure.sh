#!/usr/bin/env bash
set -euo pipefail

cd /data/chaoni/WQM/model_code/structlift

export PYTHON="${PYTHON:-/data/chaoni/miniconda3/envs/DeepseekV4_env/bin/python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/structlift_pycache_b3_train500_realssa}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /data/chaoni/WQM/audits/module_b_v3_gt98_train500_YYYYMMDD_HHMMSS"
  exit 2
fi

export B3_500_OUT="$1"
export B3_500_SRC="$B3_500_OUT/A_train_first500_compact.jsonl"
export B3_500_TEACHER="$B3_500_OUT/teacher_first500_compact.jsonl"
export B3_500_REALSSA="$B3_500_OUT/A_train_first500_compact_realssa_recovered_enriched.jsonl"

echo "===== Module B-v3 train500 real SSA closure ====="
echo "B3_500_OUT=$B3_500_OUT"
echo "PYTHON=$PYTHON"

echo
echo "===== input checks ====="
ls -lh "$B3_500_SRC" "$B3_500_TEACHER"
wc -l "$B3_500_SRC" "$B3_500_TEACHER"

echo
echo "===== py_compile ====="
"$PYTHON" -m py_compile \
  scripts/enrich_module_b_v3_real_function_ir100.py \
  scripts/export_module_b_v3_full_pipeline.py \
  scripts/audit_module_b_v3_doc_contract.py \
  struclift/utils/module_b_v3_mvp.py \
  struclift/utils/module_b_v3_ranker.py

echo
echo "===== 1/4 enrich real SSA with missing binary_path recovery ====="
"$PYTHON" scripts/enrich_module_b_v3_real_function_ir100.py \
  --source-jsonl "$B3_500_SRC" \
  --teacher-jsonl "$B3_500_TEACHER" \
  --out-jsonl "$B3_500_REALSSA" \
  --out-txt "$B3_500_OUT/enrich_real_function_ir500_recovered.txt" \
  --max-examples 500 \
  --recover-missing-binary-path \
  --require-cfg-shape-match \
  --binary-search-root /data/chaoni/WQM/source_datasets/coreutils/coreutils-9.4/multibin_o0 \
  --binary-search-root /data/chaoni/WQM/source_datasets/coreutils/coreutils-9.4/multibin_o1 \
  --binary-search-root /data/chaoni/WQM/source_datasets/coreutils/coreutils-9.4/multibin_o2 \
  --binary-search-root /data/chaoni/WQM/source_datasets/coreutils/coreutils-9.4/multibin_o3 \
  --binary-search-root /data/chaoni/WQM/source_datasets/coreutils/coreutils-9.4/multibin_os \
  --binary-search-root /data/chaoni/WQM/source_datasets/coreutils/coreutils-9.4/src \
  2>&1 | tee "$B3_500_OUT/enrich_real_function_ir500_recovered.log"

echo
echo "===== recovered real SSA summary ====="
grep -En \
  "target_rows_seen|real_register_ssa|proxy_fallback|mode_dist|cfg_shape_matches_source_jsonl|instr_text_matches_source_jsonl|binary_path_recovered|binary_path_recovery_method_dist|binary_index_stats|fallback reasons|missing_or_unresolved_binary_path|binary_recovery|function_symbol_not_found|rebuilt_cfg_shape_mismatch|pyelftools" \
  "$B3_500_OUT/enrich_real_function_ir500_recovered.txt" || true

echo
echo "===== 2/4 run Module B-v3 on recovered real SSA train500 ====="
"$PYTHON" scripts/export_module_b_v3_full_pipeline.py \
  --source-jsonl "$B3_500_REALSSA" \
  --teacher-jsonl "$B3_500_TEACHER" \
  --max-examples 500 \
  --max-region-nodes 96 \
  --max-mixed-if-headers 4 \
  --candidate-beam 96 \
  --ranker-epochs 80 \
  --ranker-lr 0.05 \
  --ranker-l2 0.0001 \
  --seed 13 \
  --train-on-all \
  --include-switch-chain \
  --include-skeletons \
  --include-all-candidate-summaries \
  --max-candidates-final 32 \
  --candidate-family-cap 6 \
  --candidate-signature-cap 3 \
  --mixed-candidate-cap 12 \
  --max-preview 120 \
  --out-jsonl "$B3_500_OUT/train500_module_b_v3_gt98_realssa_full_pipeline.jsonl" \
  --out-txt "$B3_500_OUT/train500_module_b_v3_gt98_realssa_full_pipeline.txt" \
  --out-ranker "$B3_500_OUT/module_b_v3_gt98_train500_realssa_ranker.json" \
  2>&1 | tee "$B3_500_OUT/export_train500_module_b_v3_gt98_realssa_full_pipeline.log"

echo
echo "===== 3/4 doc-contract audit ====="
"$PYTHON" scripts/audit_module_b_v3_doc_contract.py \
  --input-jsonl "$B3_500_OUT/train500_module_b_v3_gt98_realssa_full_pipeline.jsonl" \
  --max-examples 500 \
  --max-preview 120 \
  --out-txt "$B3_500_OUT/train500_module_b_v3_gt98_realssa_doc_contract_audit.txt" \
  2>&1 | tee "$B3_500_OUT/audit_train500_module_b_v3_gt98_realssa_doc_contract.log"

echo
echo "===== 4/4 proxy-vs-real quick comparison ====="
grep -En \
  "doc_contract_label|notes =|valid =|BB coverage|CFG edge preservation|validator lowering edge preservation|False verified|validator fatal error|skeleton parse success|Verified Skeleton Rate|Fallback Region Rate|GOTO count per function mean|EARLY_EXIT count per function mean|CLEANUP_EXIT count per function mean|Escape slot count per function mean|Structured block coverage mean|doc_quality_ready =|hard_valid =|failure_bucket_dist|condition_slot_grounded_rate_mean|branch_header_cond_slot_coverage_mean|branch_provenance_coverage_mean|condition_branch_metadata_rate_mean|ranker_top1_matches_oracle|ranker_selected_in_oracle_top3" \
  "$B3_500_OUT/train500_module_b_v3_gt98_doc_contract_audit.txt" \
  "$B3_500_OUT/train500_module_b_v3_gt98_full_pipeline.txt" \
  "$B3_500_OUT/train500_module_b_v3_gt98_realssa_doc_contract_audit.txt" \
  "$B3_500_OUT/train500_module_b_v3_gt98_realssa_full_pipeline.txt" || true

echo
echo "===== files ====="
ls -lh "$B3_500_OUT"
echo "B3_500_OUT=$B3_500_OUT"

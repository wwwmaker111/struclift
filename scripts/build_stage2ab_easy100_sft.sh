#!/usr/bin/env bash
# Build a small Stage2-A/B-skeleton SFT JSONL for format audit / overfit validation.
#
# This intentionally uses scripts/augment_binskel_sft_from_stage2.py:
#   Stage2 checkpoint -> model.encode(run_alignment=True) -> Module B skeleton
#   -> Stage3 fill-target SFT fields.
#
# By default it first backfills readable per-BB asm into a temporary input,
# then runs Stage2 A/B to build skeletons. It does not delete or overwrite the
# preserved CD_*/C_* baseline JSONL files.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"
NORM="${ROOT}/scripts/normalize_binskel_source_paths.py"
BACKFILL="${ROOT}/scripts/backfill_binskel_asm_text.py"
AUG="${ROOT}/scripts/augment_binskel_sft_from_stage2.py"
DEDUP="${ROOT}/scripts/dedup_binskel_jsonl.py"

BASE="${BASE:-/data/chaoni/WQM/datasets}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/chaoni/WQM/source_datasets}"

SPLIT="${SPLIT:-train}"
TAG="${TAG:-o0}"
N="${N:-50}"
if [[ "${N}" =~ ^[0-9]+$ ]] && [[ "${N}" -le 0 ]]; then
  LIMIT_LABEL="all"
else
  LIMIT_LABEL="${N}"
fi

INPUT_PREFIX="${INPUT_PREFIX:-CD}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-C}"
INPUT_SUFFIX="${INPUT_SUFFIX:-}"

TOKENIZER="${TOKENIZER:-/data/chaoni/WQM/models/deepseek-coder-6.7b-instruct}"
TOKENIZER_SAVED="${TOKENIZER_SAVED:-${BASE}/structlift_tokenizer_deepseek_6.7b_instruct_fill_stage2ab_easy100_v7}"
STAGE2_CKPT="${STAGE2_CKPT:-/data/chaoni/WQM/checkpoints/stage2_ddp2/best_stage2_curriculum_stage4.pt}"

OUT_TAG="${OUT_TAG:-deepseek_coder6.7b_stage2ab_asmctx_pretty_easy16style_full_nofname_masktok_len16384_v7_probe50}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"

DRY_RUN="${DRY_RUN:-0}"
SAVE_TOKENIZER="${SAVE_TOKENIZER:-1}"
USE_EXISTING_REPATH="${USE_EXISTING_REPATH:-1}"
BACKFILL_ASM="${BACKFILL_ASM:-1}"
DEDUP_INPUT="${DEDUP_INPUT:-0}"
DEDUP_KEEP="${DEDUP_KEEP:-last}"
ALLOW_ASM_MISMATCH="${ALLOW_ASM_MISMATCH:-0}"
ASM_MAX_TOTAL_LINES="${ASM_MAX_TOTAL_LINES:-160}"
ASM_MAX_LINES_PER_BB="${ASM_MAX_LINES_PER_BB:-16}"
USE_M_GT_FOR_AB="${USE_M_GT_FOR_AB:-0}"
STRICT_TEACHER="${STRICT_TEACHER:-0}"
MIN_SUPERVISED_SEMANTIC_SLOTS="${MIN_SUPERVISED_SEMANTIC_SLOTS:-1}"

NORM_REWRITE_OPTS=(
  --rewrite-prefix /mnt/e/structlift_src_datasets "${SOURCE_ROOT}"
  --rewrite-prefix /home/wuqiongmin/structlift_wsl/datasets "${SOURCE_ROOT}"
  --rewrite-prefix /home/wuqiongmin/struclift_wsl/datasets "${SOURCE_ROOT}"
  --rewrite-prefix /tmp/libxml2-2.12.7 "${SOURCE_ROOT}/libxml2/libxml2-2.12.7"
)
AUG_REWRITE_OPTS=(
  --rewrite-source-prefix /mnt/e/structlift_src_datasets "${SOURCE_ROOT}"
  --rewrite-source-prefix /home/wuqiongmin/structlift_wsl/datasets "${SOURCE_ROOT}"
  --rewrite-source-prefix /home/wuqiongmin/struclift_wsl/datasets "${SOURCE_ROOT}"
  --rewrite-source-prefix /tmp/libxml2-2.12.7 "${SOURCE_ROOT}/libxml2/libxml2-2.12.7"
)

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    printf '>>'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  fi
}

if [[ ! -f "${NORM}" || ! -f "${AUG}" || ! -f "${BACKFILL}" ]]; then
  echo "error: missing ${NORM}, ${BACKFILL}, or ${AUG}" >&2
  exit 1
fi
if [[ "${DEDUP_INPUT}" == "1" && ! -f "${DEDUP}" ]]; then
  echo "error: DEDUP_INPUT=1 but missing ${DEDUP}" >&2
  exit 1
fi

in_stem="${INPUT_PREFIX}_${SPLIT}_${TAG}"
out_stem="${OUTPUT_PREFIX}_${SPLIT}_${TAG}"
inp="${BASE}/${in_stem}${INPUT_SUFFIX}.jsonl"
repath="${BASE}/${in_stem}.repath.jsonl"
probe_repath="${BASE}/${in_stem}.repath.${OUT_TAG}.input${LIMIT_LABEL}.jsonl"
dedup_repath="${BASE}/${in_stem}.repath.${OUT_TAG}.dedup${LIMIT_LABEL}.jsonl"
asm_inp="${BASE}/${in_stem}.repath.${OUT_TAG}.asm${LIMIT_LABEL}.jsonl"
out="${BASE}/${out_stem}_${OUT_TAG}.jsonl"

if [[ "${DRY_RUN}" != "1" && ! -f "${inp}" ]]; then
  echo "error: input JSONL not found: ${inp}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${STAGE2_CKPT}" ]]; then
  echo "error: Stage2 checkpoint not found: ${STAGE2_CKPT}" >&2
  exit 1
fi

echo "== Stage2-A/B easy overfit SFT build =="
echo "input      : ${inp}"
echo "repath     : ${repath}"
echo "output     : ${out}"
echo "stage2 ckpt: ${STAGE2_CKPT}"
echo "tokenizer  : ${TOKENIZER}"
echo "max lines  : ${LIMIT_LABEL}"
echo "max seq len: ${MAX_SEQ_LEN}"
echo "batch size : ${BATCH_SIZE}"
echo "dedup input: ${DEDUP_INPUT} keep=${DEDUP_KEEP}"
echo "asm input  : ${asm_inp}"
echo
echo "This script preserves existing CD_*/C_* JSONL files; temporary probe inputs use OUT_TAG names."
echo

if [[ "${USE_EXISTING_REPATH}" == "1" && -f "${repath}" ]]; then
  echo "using existing repath JSONL: ${repath}"
else
  run_cmd "${PY}" "${NORM}" \
    --input "${inp}" \
    --output "${repath}" \
    "${NORM_REWRITE_OPTS[@]}" \
    --also-binary
fi

run_cmd "${PY}" "${NORM}" \
  --input "${repath}" \
  --output "${probe_repath}" \
  "${NORM_REWRITE_OPTS[@]}" \
  --also-binary

stage_input="${probe_repath}"
if [[ "${DEDUP_INPUT}" == "1" ]]; then
  run_cmd "${PY}" "${DEDUP}" \
    --input "${probe_repath}" \
    --output "${dedup_repath}" \
    --keep "${DEDUP_KEEP}"
  stage_input="${dedup_repath}"
fi

aug_input="${stage_input}"
if [[ "${BACKFILL_ASM}" == "1" ]]; then
  bf_args=(
    "${PY}" "${BACKFILL}"
    --input "${stage_input}"
    --output "${asm_inp}"
  )
  if [[ "${LIMIT_LABEL}" != "all" ]]; then
    bf_args+=(--max-lines "${N}")
  fi
  if [[ "${ALLOW_ASM_MISMATCH}" == "1" ]]; then
    bf_args+=(--allow-mismatch)
  fi
  run_cmd "${bf_args[@]}"
  aug_input="${asm_inp}"
fi

aug_args=(
  "${PY}" "${AUG}"
  --input "${aug_input}"
  --output "${out}"
  --stage2-ckpt "${STAGE2_CKPT}"
  --tokenizer "${TOKENIZER}"
  --source-root "${SOURCE_ROOT}"
  "${AUG_REWRITE_OPTS[@]}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --asm-max-total-lines "${ASM_MAX_TOTAL_LINES}"
  --asm-max-lines-per-bb "${ASM_MAX_LINES_PER_BB}"
  --mask-prompt-labels
  --mask-non-fill-labels
  --mask-supervised-inputs
  --emit-rl-fields
  --min-supervised-semantic-slots "${MIN_SUPERVISED_SEMANTIC_SLOTS}"
)

if [[ "${LIMIT_LABEL}" != "all" ]]; then
  aug_args+=(--max-lines "${N}")
fi

if [[ "${SAVE_TOKENIZER}" == "1" ]]; then
  aug_args+=(--save-tokenizer "${TOKENIZER_SAVED}")
fi
if [[ "${USE_M_GT_FOR_AB}" == "1" ]]; then
  aug_args+=(--use-m-gt-for-ab)
fi
if [[ "${STRICT_TEACHER}" == "1" ]]; then
  aug_args+=(--strict-teacher)
fi

run_cmd "${aug_args[@]}"

echo
echo "done:"
echo "  ${out}"
echo "  tokenizer with fill/mask tokens: ${TOKENIZER_SAVED}"

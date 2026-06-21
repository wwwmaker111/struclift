#!/usr/bin/env bash
# 由 build_*_binskel.sh source。需已设置 WORKDIR（仓库根），且已激活含 transformers 的 venv（若 EMIT_SFT_JSONL=1）。
#
# 提供: binskel_augment_sft_jsonl <jsonl路径> <source_root>
# 在已有对齐 jsonl 上追加 SFT/RL 字段（sft_input_ids、sft_labels、slot_types、bin_pattern_dist 等）。
#
# 环境变量:
#   EMIT_SFT_JSONL=1       默认 1；0 则跳过
#   SFT_TOKENIZER          默认 deepseek-ai/deepseek-coder-6.7b-base
#   SFT_MAX_SEQ_LEN        默认 16384
#   SFT_MASK_PROMPT_LABELS 默认 1；1 则传 --mask-prompt-labels
#   SFT_MASK_NON_FILL_LABELS 默认 1；1 则传 --mask-non-fill-labels
#
# shellcheck shell=bash

: "${EMIT_SFT_JSONL:=1}"
: "${SFT_TOKENIZER:=deepseek-ai/deepseek-coder-6.7b-base}"
: "${SFT_MAX_SEQ_LEN:=16384}"
: "${SFT_MASK_PROMPT_LABELS:=1}"
: "${SFT_MASK_NON_FILL_LABELS:=1}"

binskel_augment_sft_jsonl() {
  local jsonl="${1:?}"
  local src_root="${2:?}"
  if [[ "${EMIT_SFT_JSONL}" != "1" ]]; then
    return 0
  fi
  [[ -f "${jsonl}" ]] || return 0
  local aug_py="${WORKDIR}/scripts/augment_binskel_sft.py"
  [[ -f "${aug_py}" ]] || {
    echo "错误: 未找到 ${aug_py}" >&2
    return 1
  }
  local tmp="${jsonl}.__sft_tmp__"
  rm -f "${tmp}"
  echo "  [SFT] augment_binskel_sft.py → $(basename "${jsonl}")"
  python "${aug_py}" \
    --input "${jsonl}" \
    --output "${tmp}" \
    --tokenizer "${SFT_TOKENIZER}" \
    --source-root "${src_root}" \
    --emit-rl-fields \
    --max-seq-len "${SFT_MAX_SEQ_LEN}" \
    $( [[ "${SFT_MASK_PROMPT_LABELS}" == "1" ]] && echo "--mask-prompt-labels" ) \
    $( [[ "${SFT_MASK_NON_FILL_LABELS}" == "1" ]] && echo "--mask-non-fill-labels" ) \
    && mv -f "${tmp}" "${jsonl}"
}

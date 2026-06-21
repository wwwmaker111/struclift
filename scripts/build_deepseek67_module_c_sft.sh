#!/usr/bin/env bash
# 为 Module C（Stage3）批量生成 DeepSeek-Coder-6.7B 口径的 SFT jsonl。
# augment 只加载 **tokenizer**（权重 .safetensors 不必下完）；训练 Stage3 时才需要完整 --llm。
#
# 流程: 对齐 jsonl → normalize 路径 → augment_binskel_sft
#
# 输出命名（默认）:
#   C_train_o0_deepseek_coder6.7b.jsonl … C_train_os_…
#   C_val_o0_deepseek_coder6.7b.jsonl  …
# 由 NAME_PREFIX=C 与 OUT_NAME_TAG=deepseek_coder6.7b 控制。
#
# 用法:
#   export TOKENIZER=/data/chaoni/WQM/models/deepseek-coder-6.7b-instruct
#   export NAME_PREFIX=C
#   bash scripts/build_deepseek67_module_c_sft.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"
NORM="${ROOT}/scripts/normalize_binskel_source_paths.py"
AUG="${ROOT}/scripts/augment_binskel_sft.py"

BASE="${BASE:-/data/chaoni/WQM/datasets}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/chaoni/WQM/source_datasets}"
NAME_PREFIX="${NAME_PREFIX:-C}"
# 输入 jsonl 前缀（若磁盘上仍是 CD_train_o0.jsonl，而输出要 C_train_o0_…，则 INPUT_PREFIX=CD NAME_PREFIX=C）
INPUT_NAME_PREFIX="${INPUT_NAME_PREFIX:-${NAME_PREFIX}}"

# HuggingFace 名或本地目录（含 tokenizer 文件即可，不必有完整权重）
TOKENIZER="${TOKENIZER:-/data/chaoni/WQM/models/deepseek-coder-6.7b-instruct}"
TOKENIZER_SAVED="${TOKENIZER_SAVED:-${BASE}/struclift_tokenizer_deepseek_6.7b_instruct_fill}"

# 输出: ${NAME_PREFIX}_${split}_${tag}_${OUT_NAME_TAG}.jsonl
OUT_NAME_TAG="${OUT_NAME_TAG:-deepseek_coder6.7b}"

# 输入: 含 alignment_matrix 的对齐 jsonl（通常无 _sft）
INPUT_SUFFIX="${INPUT_SUFFIX:-}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_SPLIT="${ONLY_SPLIT:-}"
ONLY_TAG="${ONLY_TAG:-}"
SKIP_TOKENIZER_CHECK="${SKIP_TOKENIZER_CHECK:-0}"

# normalize 用 --rewrite-prefix；augment 用 --rewrite-source-prefix（参数名不同）
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

AUG_EXTRA=(--mask-non-fill-labels --emit-rl-fields)
if [[ -n "${MAX_SEQ_LEN:-}" ]]; then
  AUG_EXTRA+=(--max-seq-len "${MAX_SEQ_LEN}")
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    echo ">> $*"
    "$@"
  fi
}

if [[ ! -f "${NORM}" || ! -f "${AUG}" ]]; then
  echo "错误: 缺少 ${NORM} 或 ${AUG}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && "${SKIP_TOKENIZER_CHECK}" != "1" ]]; then
  run_cmd "${PY}" -c "
from transformers import AutoTokenizer
tok = '${TOKENIZER}'
print('加载 tokenizer:', tok)
AutoTokenizer.from_pretrained(tok, trust_remote_code=True)
print('tokenizer OK')
"
fi

mkdir -p "$(dirname "${TOKENIZER_SAVED}")" 2>/dev/null || true

SPLITS=(train val)
TAGS=(o0 o1 o2 o3 os)
[[ -n "${ONLY_SPLIT}" ]] && SPLITS=("${ONLY_SPLIT}")
[[ -n "${ONLY_TAG}" ]] && TAGS=("${ONLY_TAG}")

SAVE_TOK_ONCE="${SAVE_TOK_ONCE:-1}"
tok_saved_flag=0

for split in "${SPLITS[@]}"; do
  for tag in "${TAGS[@]}"; do
    in_stem="${INPUT_NAME_PREFIX}_${split}_${tag}"
    out_stem="${NAME_PREFIX}_${split}_${tag}"
    inp="${BASE}/${in_stem}${INPUT_SUFFIX}.jsonl"
    repath="${BASE}/${in_stem}.repath.jsonl"
    out="${BASE}/${out_stem}_${OUT_NAME_TAG}.jsonl"

    if [[ ! -f "${inp}" ]]; then
      echo "跳过（不存在）: ${inp}" >&2
      continue
    fi

    echo "======== ${in_stem} → $(basename "${out}") ========"
    run_cmd "${PY}" "${NORM}" \
      --input "${inp}" \
      --output "${repath}" \
      "${NORM_REWRITE_OPTS[@]}" \
      --also-binary

    aug_args=(
      "${PY}" "${AUG}"
      --input "${repath}"
      --output "${out}"
      --tokenizer "${TOKENIZER}"
      --source-root "${SOURCE_ROOT}"
      "${AUG_REWRITE_OPTS[@]}"
      "${AUG_EXTRA[@]}"
    )
    if [[ "${SAVE_TOK_ONCE}" == "1" && "${tok_saved_flag}" == "0" ]]; then
      aug_args+=(--save-tokenizer "${TOKENIZER_SAVED}")
      tok_saved_flag=1
    elif [[ "${SAVE_TOK_ONCE}" != "1" ]]; then
      aug_args+=(--save-tokenizer "${TOKENIZER_SAVED}")
    fi

    run_cmd "${aug_args[@]}"
  done
done

echo ""
echo "完成。"
echo "  输出: ${BASE}/${NAME_PREFIX}_{train,val}_{o0..os}_${OUT_NAME_TAG}.jsonl"
echo "  增广用 tokenizer: ${TOKENIZER}"
echo "  已保存（含 FILL）: ${TOKENIZER_SAVED}"
echo "  Stage3 训练时再指定完整权重: --llm ${TOKENIZER}"

#!/usr/bin/env bash
#
# 九项目 × 五档 = 45 个 jsonl：与 scripts/build_binskel_9proj_5opt.sh 一致。
#
# 张量（sft_input_ids / sft_labels / slot_types / 可选 RL 字段）的算法来源
#   - 本仓库当前各 **build_*_binskel.sh 默认未** source scripts/_binskel_md_extra.sh，
#     因此 export FULL_STAGES_JSONL=1 **不会**在 build 阶段内联 SFT（除非已用 _patch_md_extra 等方式改过脚本）。
#   - **与训练一致、与 augment_all_45_to_full_sft.py 子进程一致**的算法：scripts/augment_binskel_sft.py
#     （--tokenize / --source-root / --emit-rl-fields，与 rebuild_loss_datasets_full.sh 的 run_augment 相同）。
#
# 默认：RUN_SEPARATE_AUGMENT=1 → 步骤 2 必跑 augment_all_45_to_full_sft.py，保证 Stage3/4 字段齐全。
# 若你已把各 build 脚本改为 source _binskel_md_extra 且 FULL_STAGES_JSONL=1 真正生效，可设 RUN_SEPARATE_AUGMENT=0 避免重复。
#
# 前置：各项目已按 scripts/compile_*.sh 编译；设置全部 *_DIR 与 OUT_ROOT。
#
#   export OUT_ROOT=/mnt/e/structlift_datasets
#   export ZLIB_DIR=... CURL_DIR=... SQLITE_DIR=... BUSYBOX_DIR=... OPENSSL_DIR=... LIBXML2_DIR=...
#   export OPENSSH_DIR=... FFMPEG_DIR=... COREUTILS_DIR=...
#   bash scripts/build_full_45_stages_pipeline.sh
#
# 仅 augment（已有 45 个无 SFT 的 jsonl）：
#   bash scripts/build_full_45_stages_pipeline.sh --skip-binskel-build
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_BINSKEL=0
for arg in "$@"; do
  if [[ "$arg" == "--skip-binskel-build" ]]; then
    SKIP_BINSKEL=1
  fi
done

OUT_ROOT="${OUT_ROOT:?请设置 OUT_ROOT（如 /mnt/e/structlift_datasets）}"
FULL_SFT_DIR="${FULL_SFT_DIR:-${OUT_ROOT}/full_sft_45}"
TOKENIZER="${TOKENIZER:-deepseek-ai/deepseek-coder-6.7b-base}"
TOKENIZER_OUT="${TOKENIZER_OUT:-${OUT_ROOT}/tokenizer_struclift_fill}"
# 与 rebuild_loss_datasets_full.sh 的 SFT_MAX_SEQ_LEN 对齐：若设置则传给 augment_all_45_to_full_sft.py --max-seq-len
SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-}"
: "${FULL_STAGES_JSONL:=1}"
: "${RUN_SEPARATE_AUGMENT:=1}"

echo "ROOT=$ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "FULL_STAGES_JSONL=$FULL_STAGES_JSONL"
echo "RUN_SEPARATE_AUGMENT=$RUN_SEPARATE_AUGMENT"
echo "SKIP_BINSKEL=$SKIP_BINSKEL"
echo ""

ROOTS_JSON="${OUT_ROOT}/.full_dataset_45_roots.generated.json"
python3 - <<PY
import json, os
keys = [
    ("zlib", "ZLIB_DIR"),
    ("curl", "CURL_DIR"),
    ("sqlite", "SQLITE_DIR"),
    ("busybox", "BUSYBOX_DIR"),
    ("openssl", "OPENSSL_DIR"),
    ("libxml2", "LIBXML2_DIR"),
    ("openssh", "OPENSSH_DIR"),
    ("ffmpeg", "FFMPEG_DIR"),
    ("coreutils", "COREUTILS_DIR"),
]
out = {}
missing = []
for name, envk in keys:
    v = os.environ.get(envk, "").strip()
    if not v:
        missing.append(envk)
    else:
        out[name] = v
if missing:
    raise SystemExit(
        "缺少环境变量: " + ", ".join(missing) + "（9 个 *_DIR 均需设置）"
    )
path = "${ROOTS_JSON}"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("已写入", path)
PY

if [[ "$SKIP_BINSKEL" -eq 0 ]]; then
  echo "========== 步骤 1: 构建 45 个对齐 jsonl（binskel_*；SFT 见步骤 2 除非已内联 full-stages）=========="
  export OUT_ROOT
  export FULL_STAGES_JSONL
  bash "${ROOT}/scripts/build_binskel_9proj_5opt.sh"
else
  echo "========== 步骤 1: 跳过（--skip-binskel-build） =========="
fi

if [[ "$SKIP_BINSKEL" -eq 1 ]] || [[ "${RUN_SEPARATE_AUGMENT}" == "1" ]]; then
  echo ""
  echo "========== 步骤 2: augment_binskel_sft（经 augment_all_45_to_full_sft.py，与 rebuild_loss_datasets_full.sh 同源算法）=========="
  _AUG=(python3 "${ROOT}/scripts/augment_all_45_to_full_sft.py" \
    --binskel-dir "${OUT_ROOT}" \
    --out-dir "${FULL_SFT_DIR}" \
    --roots-json "${ROOTS_JSON}" \
    --tokenizer "${TOKENIZER}" \
    --save-tokenizer-dir "${TOKENIZER_OUT}")
  if [[ -n "${SFT_MAX_SEQ_LEN:-}" ]]; then
    _AUG+=(--max-seq-len "${SFT_MAX_SEQ_LEN}")
  fi
  "${_AUG[@]}"
  echo "完整（旧式两阶段）输出: ${FULL_SFT_DIR}"
  echo "Tokenizer: ${TOKENIZER_OUT}"
else
  echo ""
  echo "========== 步骤 2: 跳过单独 augment（RUN_SEPARATE_AUGMENT=0）=========="
  echo "请确认各 build_*_binskel.sh 已 source _binskel_md_extra 且 FULL_STAGES_JSONL=1 已真正写入 SFT；"
  echo "否则 OUT_ROOT 下 jsonl 可能缺少 sft_input_ids，Stage3/4 不可用。"
fi

echo ""
echo "完成。8:1:1 划分（可选）: python3 scripts/split_binskel_45_datasets_811.py --in-dir ${OUT_ROOT} --out-dir ${OUT_ROOT}/split_811 --seed 42"
echo ""

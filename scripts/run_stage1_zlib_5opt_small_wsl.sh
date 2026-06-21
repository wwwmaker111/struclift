#!/usr/bin/env bash
# zlib 五个优化等级混合 + 小样本 Stage1（不划 8:1:1，先烟测）。
# --max-samples 在代码里是「每个 jsonl 各自上限」，五档合计约 5 * MAX_SAMPLES 条。
#
# 用法（WSL）:
#   cd /mnt/e/structlift
#   bash scripts/run_stage1_zlib_5opt_small_wsl.sh
#
# 环境变量（可选）:
#   DATASET_DIR   默认 /mnt/e/structlift_datasets
#   MAX_SAMPLES   每档最多条数，默认 64（五档合计约 320 条）
#   EPOCHS        默认 3
#   BATCH_SIZE    默认 4
#   SAVE_DIR      默认 $ROOT/checkpoints/stage1_zlib_5opt_small
#   PYTHON        默认 python3

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATASET_DIR="${DATASET_DIR:-/mnt/e/structlift_datasets}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/stage1_zlib_5opt_small}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PY="${PYTHON:-python3}"

if [ -f "$HOME/struclift_wsl/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$HOME/struclift_wsl/.venv/bin/activate"
  PY="${PYTHON:-python}"
fi

missing=0
for opt in o0 o1 o2 o3 os; do
  f="$DATASET_DIR/binskel_zlib_${opt}.jsonl"
  if [ ! -f "$f" ]; then
    echo "缺少: $f" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "请先准备 zlib 五档 jsonl（例如 scripts/build_zlib_binskel.sh）。" >&2
  exit 1
fi

echo "DATASET_DIR=$DATASET_DIR  MAX_SAMPLES(每档)=$MAX_SAMPLES  EPOCHS=$EPOCHS  SAVE_DIR=$SAVE_DIR"
"$PY" scripts/train_stage1_binskel.py \
  --jsonl \
    "$DATASET_DIR/binskel_zlib_o0.jsonl" \
    "$DATASET_DIR/binskel_zlib_o1.jsonl" \
    "$DATASET_DIR/binskel_zlib_o2.jsonl" \
    "$DATASET_DIR/binskel_zlib_o3.jsonl" \
    "$DATASET_DIR/binskel_zlib_os.jsonl" \
  --max-samples "$MAX_SAMPLES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --save-dir "$SAVE_DIR"

echo "完成: $SAVE_DIR/best_stage1.pt"

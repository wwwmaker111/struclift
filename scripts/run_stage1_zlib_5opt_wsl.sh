#!/usr/bin/env bash
# zlib 五个优化等级（o0/o1/o2/o3/os）混合 Stage1 — 默认使用每档 jsonl 内全部有效样本。
#
# WSL 用法:
#   cd /mnt/e/structlift
#   bash scripts/run_stage1_zlib_5opt_wsl.sh
#
# 环境变量（可选）:
#   DATASET_DIR    jsonl 目录，默认 /mnt/e/structlift_datasets
#   EPOCHS         默认 30（可看曲线；要更充分可设 100）
#   BATCH_SIZE     默认 4（CPU/内存紧可 2；有 GPU 可 8–16）
#   SAVE_DIR       默认 $ROOT/checkpoints/stage1_zlib_5opt_mixed
#   MAX_SAMPLES    若设置则每档最多该条数（调试用）；不设则全量
#   NUM_WORKERS    默认 0
#   PYTHON         解释器，默认 python3

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATASET_DIR="${DATASET_DIR:-/mnt/e/structlift_datasets}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/stage1_zlib_5opt_mixed}"
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

echo "DATASET_DIR=$DATASET_DIR  EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  SAVE_DIR=$SAVE_DIR"
if [ -n "${MAX_SAMPLES:-}" ]; then
  echo "MAX_SAMPLES(每档上限)=$MAX_SAMPLES"
fi

cmd=(
  "$PY" scripts/train_stage1_binskel.py
  --jsonl
    "$DATASET_DIR/binskel_zlib_o0.jsonl"
    "$DATASET_DIR/binskel_zlib_o1.jsonl"
    "$DATASET_DIR/binskel_zlib_o2.jsonl"
    "$DATASET_DIR/binskel_zlib_o3.jsonl"
    "$DATASET_DIR/binskel_zlib_os.jsonl"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --save-dir "$SAVE_DIR"
)

if [ -n "${MAX_SAMPLES:-}" ]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi

"${cmd[@]}"

echo "完成: $SAVE_DIR/best_stage1.pt 与 final_stage1.pt"

#!/usr/bin/env bash
# 250 条训练、验证最多 50 条、20 epoch（可按环境变量改路径与条数）
#
# 用法（在工程根目录，Linux/服务器）:
#   chmod +x scripts/stage2_250_50_20.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/stage2_250_50_20.sh
#
# 覆盖示例:
#   TRAIN_MAX=250 VAL_MAX=50 EPOCHS=20 JSONL_TRAIN=/path/o0.jsonl SAVE_DIR=./ckpt_250 \
#   CUDA_VISIBLE_DEVICES=1 ./scripts/stage2_250_50_20.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_MAX="${TRAIN_MAX:-250}"
VAL_MAX="${VAL_MAX:-50}"
EPOCHS="${EPOCHS:-20}"

JSONL_TRAIN="${JSONL_TRAIN:-/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl}"
INIT_CKPT="${INIT_CKPT:-/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/stage2_250_50_20}"
VAL_JSONL="${VAL_JSONL:-/data/chaoni/WQM/datasets/AB_val.jsonl}"
LOG_DIR="${LOG_DIR:-/data/chaoni/WQM/logs}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_NAME="stage2_${TRAIN_MAX}train_${VAL_MAX}val_${EPOCHS}ep_${TS}.log"
LOG_FILE="${LOG_DIR}/${LOG_NAME}"
mkdir -p "$LOG_DIR" "$(dirname "$SAVE_DIR")" 2>/dev/null || true

echo "ROOT=$ROOT"
echo "训练: $JSONL_TRAIN  --max-samples $TRAIN_MAX"
echo "验证: $VAL_JSONL  --val-max-samples $VAL_MAX"
echo "epoch: $EPOCHS  保存: $SAVE_DIR  init: $INIT_CKPT"
echo "日志: $LOG_FILE  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

exec "$PYTHON" scripts/train_stage2_binskel.py \
  --jsonl "$JSONL_TRAIN" \
  --init-from "$INIT_CKPT" \
  --save-dir "$SAVE_DIR" \
  --max-samples "$TRAIN_MAX" \
  --val-jsonl "$VAL_JSONL" \
  --val-max-samples "$VAL_MAX" \
  --epochs "$EPOCHS" \
  --batch-size 4 \
  --num-workers 0 \
  --prefetch-factor 2 \
  --device "$DEVICE" \
  --no-fp16 \
  2>&1 | tee "$LOG_FILE"

#!/usr/bin/env bash
# 小试跑：20 条训练样本、10 个 epoch，观察 train/val loss 曲线（使用当前 config，含默认 s2_fp16=False）。
#
# 用法（在工程根目录）:
#   chmod +x scripts/stage2_mini_20x10.sh
#   CUDA_VISIBLE_DEVICES=1 ./scripts/stage2_mini_20x10.sh
#
# 或覆盖环境变量:
#   JSONL_TRAIN=/path/o0.jsonl INIT_CKPT=/path/stage1.pt SAVE_DIR=./ckpt_test CUDA_VISIBLE_DEVICES=0 ./scripts/stage2_mini_20x10.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# —— 按服务器实际路径改；也可用环境变量覆盖 ——
JSONL_TRAIN="${JSONL_TRAIN:-/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl}"
INIT_CKPT="${INIT_CKPT:-/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/stage2_mini_20x10}"
VAL_JSONL="${VAL_JSONL:-/data/chaoni/WQM/datasets/AB_val.jsonl}"
# 验证集最多读多少条（加速 mini 跑；设为空或极大则等同全量）
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-200}"
LOG_DIR="${LOG_DIR:-/data/chaoni/WQM/logs}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
# 未设置时用 0 号卡
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage2_mini_20x10_${TS}.log"
mkdir -p "$LOG_DIR" "$(dirname "$SAVE_DIR")" 2>/dev/null || true

echo "ROOT=$ROOT"
echo "训练 jsonl: $JSONL_TRAIN (max 20 条)"
echo "ckpt: $INIT_CKPT"
echo "保存: $SAVE_DIR"
echo "验证: $VAL_JSONL (最多 ${VAL_MAX_SAMPLES} 条，加速 val)"
echo "日志: $LOG_FILE"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

exec "$PYTHON" scripts/train_stage2_binskel.py \
  --jsonl "$JSONL_TRAIN" \
  --init-from "$INIT_CKPT" \
  --save-dir "$SAVE_DIR" \
  --max-samples 20 \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 0 \
  --prefetch-factor 2 \
  --device "$DEVICE" \
  --val-jsonl "$VAL_JSONL" \
  --val-max-samples "$VAL_MAX_SAMPLES" \
  2>&1 | tee "$LOG_FILE"

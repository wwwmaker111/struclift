#!/usr/bin/env bash
# zlib 五个优化等级分别跑 Stage1（每次只训一个 jsonl，避免拼在一起时间过长被中断）
#
# 用法（WSL）:
#   cd /mnt/e/structlift
#   bash scripts/run_stage1_zlib_each_opt.sh
#
# 可选环境变量:
#   DATASET_DIR   jsonl 目录（默认 /mnt/e/structlift_datasets）
#   EPOCHS        每个等级 epoch 数（默认 10）
#   BATCH_SIZE    默认 8
#   NUM_WORKERS   默认 0
#   PYTHON        解释器（默认激活 venv 后不用设）

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATASET_DIR="${DATASET_DIR:-/mnt/e/structlift_datasets}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [ -f "$HOME/struclift_wsl/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$HOME/struclift_wsl/.venv/bin/activate"
fi

PY="${PYTHON:-python}"

for opt in o0 o1 o2 o3 os; do
  JSONL="$DATASET_DIR/binskel_zlib_${opt}.jsonl"
  if [ ! -f "$JSONL" ]; then
    echo "跳过（文件不存在）: $JSONL"
    continue
  fi
  echo ""
  echo "========== Stage1 zlib ${opt} =========="
  "$PY" scripts/train_stage1_binskel.py \
    --jsonl "$JSONL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --save-dir "$ROOT/checkpoints/stage1_zlib_${opt}"
  echo "完成: checkpoints/stage1_zlib_${opt}/"
done

echo ""
echo "全部阶段结束。"

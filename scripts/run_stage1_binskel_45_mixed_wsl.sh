#!/usr/bin/env bash
# 45 个 binskel（各档 train）混合训练 → 单一 Stage1 编码器（WSL）。
# 依赖：已运行 split，生成 train_jsonl_paths.txt（见 run_split_binskel_45_811.sh）。
#
# 用法:
#   cd /mnt/e/structlift && bash scripts/run_stage1_binskel_45_mixed_wsl.sh
#   bash scripts/run_stage1_binskel_45_mixed_wsl.sh /path/to/train_jsonl_paths.txt /path/to/save_dir
#   EXTRA_ARGS="--batch-size 4 --epochs 10" bash scripts/run_stage1_binskel_45_mixed_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TRAIN_LIST="${1:-/mnt/e/structlift_datasets/split_811_seed42/train_jsonl_paths.txt}"
SAVE_DIR="${2:-$ROOT/checkpoints/stage1_binskel_45_mixed}"
PY="${PYTHON:-python3}"

if [[ ! -f "$TRAIN_LIST" ]]; then
  echo "找不到训练列表: $TRAIN_LIST" >&2
  echo "请先: bash scripts/run_split_binskel_45_811.sh [IN_DIR] [OUT_DIR] [SEED]" >&2
  exit 1
fi

echo "jsonl-list: $TRAIN_LIST"
echo "save-dir:   $SAVE_DIR"
$PY scripts/train_stage1_binskel.py \
  --jsonl-list "$TRAIN_LIST" \
  --save-dir "$SAVE_DIR" \
  ${EXTRA_ARGS:-}

echo "完成: $SAVE_DIR/best_stage1.pt 与 final_stage1.pt"

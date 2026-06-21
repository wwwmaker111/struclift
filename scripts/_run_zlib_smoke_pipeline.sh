#!/usr/bin/env bash
# 使用 structlift_datasets 下 zlib O0 jsonl 跑 Stage1–4 冒烟（可调 max_samples）
set -euo pipefail
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate

JSONL="/mnt/e/structlift_datasets/binskel_zlib_o0.jsonl"
MAXS="${MAX_SAMPLES:-32}"
EPOCH1="${S1_EPOCHS:-1}"
EPOCH2="${S2_EPOCHS:-1}"
EPOCH3="${S3_EPOCHS:-1}"
EPOCH4="${S4_EPOCHS:-1}"
S3MAX="${S3_MAX_SAMPLES:-4}"
S4MAX="${S4_MAX_SAMPLES:-2}"
LLM="${LLM:-deepseek-ai/deepseek-coder-6.7b-base}"

echo "=== Stage1 jsonl=$JSONL max_samples=$MAXS ==="
python3 scripts/train_stage1_binskel.py \
  --jsonl "$JSONL" \
  --epochs "$EPOCH1" --batch-size 2 --max-samples "$MAXS" \
  --save-dir checkpoints/zlib_pipeline_s1

echo "=== Stage2 ==="
python3 scripts/train_stage2_binskel.py \
  --jsonl "$JSONL" \
  --init-from checkpoints/zlib_pipeline_s1/best_stage1.pt \
  --epochs "$EPOCH2" --batch-size 2 --max-samples "$MAXS" \
  --save-dir checkpoints/zlib_pipeline_s2

echo "=== Stage3 (needs LLM + SFT jsonl) ==="
python3 scripts/train_stage3_binskel.py \
  --jsonl "$JSONL" \
  --init-from checkpoints/zlib_pipeline_s2/best_stage2.pt \
  --llm "$LLM" \
  --epochs "$EPOCH3" --batch-size 1 --max-samples "$S3MAX" \
  --save-dir checkpoints/zlib_pipeline_s3

echo "=== Stage4 ==="
python3 scripts/train_stage4_binskel.py \
  --jsonl "$JSONL" \
  --init-from checkpoints/zlib_pipeline_s3/best_stage3.pt \
  --llm "$LLM" \
  --max-samples "$S4MAX" --max-new-tokens 128 \
  --save-dir checkpoints/zlib_pipeline_s4

echo "=== done ==="

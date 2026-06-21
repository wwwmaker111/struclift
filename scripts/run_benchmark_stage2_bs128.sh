#!/usr/bin/env bash
# Stage2 单卡 micro-benchmark：固定 batch_size=128，输出与 benchmark_stage2_cache.py 一致（含 mean total）。
#
# 「三轮平均」指脚本内部的 **3 个 measure step**（不含 warmup）。公式：
#   need_batches = n_warmup + n_measure
#   n_samples    = need_batches * batch_size
#   默认 warmup=1、measure=3、bs=128 → n_samples = 4 * 128 = **512**
#
# 若你强行只要 **256 条** jsonl：`N_SAMPLES=256` → 只有 2 个 batch → measure 最多 **1**（做不了三步平均）。
#
# 用法::
#
#   cd /data/chaoni/WQM/model_code/structlift
#   chmod +x scripts/run_benchmark_stage2_bs128.sh
#   ./scripts/run_benchmark_stage2_bs128.sh
#
#   MODE=no-cache ./scripts/run_benchmark_stage2_bs128.sh
#   N_SAMPLES=256 ./scripts/run_benchmark_stage2_bs128.sh   # 仅 1 个 measure step
#
set -euo pipefail

ROOT="${ROOT:-/data/chaoni/WQM/model_code/structlift}"
PYTHON="${PYTHON:-/data/chaoni/miniconda3/envs/wqm_struc/bin/python3}"
JSONL="${JSONL:-/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl}"
INIT_FROM="${INIT_FROM:-/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt}"
CACHE_DIR="${CACHE_DIR:-/data/chaoni/WQM/cache/stage2_encoder}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MODE="${MODE:-cache}"
BATCH_SIZE=128
N_WARMUP="${N_WARMUP:-1}"
# 默认 512 → measure = (512//128) - 1 = 3
N_SAMPLES="${N_SAMPLES:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SHARD_LRU="${SHARD_LRU:-4}"

cd "$ROOT"

extra=(--shard-lru "$SHARD_LRU")
if [[ "$MODE" == "both" || "$MODE" == "cache" ]]; then
  extra+=(--cache-dir "$CACHE_DIR")
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MODE=$MODE  batch=$BATCH_SIZE  n_samples=$N_SAMPLES  warmup=$N_WARMUP"
echo "(measure steps ≈ max(1, (n_samples//batch) - warmup))"

exec env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  "$PYTHON" scripts/benchmark_stage2_cache.py \
  --mode "$MODE" \
  --jsonl "$JSONL" \
  --init-from "$INIT_FROM" \
  --batch-size "$BATCH_SIZE" \
  --n-samples "$N_SAMPLES" \
  --n-warmup-batches "$N_WARMUP" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  "${extra[@]}"

#!/usr/bin/env bash
# Stage2 **DDP 双卡（或多卡）** micro-benchmark：``torchrun`` + ``benchmark_stage2_ddp.py``。
# 与 ``train_stage2_binskel.py`` 一致：每卡 batch_size、手写 ``all_reduce`` 对齐梯度（非 nn.DDP）。
#
# 用法（服务器）::
#
#   cd /data/chaoni/WQM/model_code/structlift
#   chmod +x scripts/run_benchmark_stage2_gpu12.sh
#   ./scripts/run_benchmark_stage2_gpu12.sh
#
# 指定可见 GPU（默认物理 1,2）::
#
#   CUDA_VISIBLE_DEVICES=0,7 ./scripts/run_benchmark_stage2_gpu12.sh
#
# 仅 no-cache、或改进程数::
#
#   MODE=no-cache NPROC=2 ./scripts/run_benchmark_stage2_gpu12.sh
#
# jsonl **总条数**与单卡 benchmark 完全一致（各 rank 分片，每 rank batch 数可能变少）::
#
#   DDP_GLOBAL_DATASET=1 ./scripts/run_benchmark_stage2_gpu12.sh
#
# scp 示例::
#
#   scp scripts/run_benchmark_stage2_gpu12.sh scripts/benchmark_stage2_ddp.py user@host:/path/to/structlift/scripts/
#
set -euo pipefail

ROOT="${ROOT:-/data/chaoni/WQM/model_code/structlift}"
PYTHON="${PYTHON:-/data/chaoni/miniconda3/envs/wqm_struc/bin/python3}"
JSONL="${JSONL:-/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl}"
INIT_FROM="${INIT_FROM:-/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt}"
CACHE_DIR="${CACHE_DIR:-/data/chaoni/WQM/cache/stage2_encoder}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
NPROC="${NPROC:-2}"

MODE="${MODE:-both}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-256}"
N_WARMUP="${N_WARMUP:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SHARD_LRU="${SHARD_LRU:-4}"
LOG_DIR="${LOG_DIR:-}"
TS="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"

extra_py_args=()
if [[ "$MODE" == "both" || "$MODE" == "cache" ]]; then
  extra_py_args+=(--cache-dir "$CACHE_DIR" --shard-lru "$SHARD_LRU")
fi
if [[ "${DDP_GLOBAL_DATASET:-}" == "1" ]]; then
  extra_py_args+=(--ddp-global-dataset)
fi

cmd=(
  env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC"
  scripts/benchmark_stage2_ddp.py
  --mode "$MODE"
  --jsonl "$JSONL"
  --batch-size "$BATCH_SIZE"
  --n-samples "$N_SAMPLES"
  --n-warmup-batches "$N_WARMUP"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
)
if [[ -n "$INIT_FROM" ]]; then
  cmd+=(--init-from "$INIT_FROM")
fi
cmd+=("${extra_py_args[@]}")

echo "ROOT=$ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NPROC=$NPROC"
echo "PYTHON=$PYTHON"
echo "MODE=$MODE  batch(per GPU)=$BATCH_SIZE  n_samples=$N_SAMPLES  warmup_batches=$N_WARMUP"
echo "JSONL=$JSONL"
echo "INIT_FROM=${INIT_FROM:-<none>}"
echo "CACHE_DIR=$CACHE_DIR"
echo "DDP_GLOBAL_DATASET=${DDP_GLOBAL_DATASET:-0}"

if [[ -n "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR"
  log_tag="benchmark_stage2_ddp_${MODE}_${TS}"
  {
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >"${LOG_DIR}/${log_tag}.cmd.txt"
  "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${log_tag}.log"
else
  "${cmd[@]}"
fi

echo ""
echo "Done. rank0 日志里的 mean total 即近似「训练一步」墙钟（含 NCCL all_reduce）。"

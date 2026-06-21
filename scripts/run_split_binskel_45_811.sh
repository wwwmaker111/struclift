#!/usr/bin/env bash
# 调用 split_binskel_45_datasets_811.py，默认路径适配 WSL。
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/mnt/e/structlift_datasets}"
OUT="${2:-/mnt/e/structlift_datasets/split_811_seed42}"
SEED="${3:-42}"
python3 scripts/split_binskel_45_datasets_811.py --in-dir "$IN" --out-dir "$OUT" --seed "$SEED"
echo "清单: $OUT/split_811_manifest.tsv"
echo "训练集路径列表: $OUT/train_jsonl_paths.txt"

#!/bin/bash
# 汇总 20/24 个 binskel 数据集：样本数 + 对齐质量
# 6 项目 × 4 优化级 (O0-O3) = 24
# 用法: bash scripts/summary_all_20_datasets.sh [输出目录]

OUT="${1:-$HOME/structlift_datasets}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

PROJECTS="zlib sqlite busybox openssl coreutils libxml2"

echo "=============================================="
echo "  structlift 数据集汇总"
echo "  目录: $OUT"
echo "=============================================="
echo ""

total_samples=0
total_datasets=0

printf "%-22s %10s\n" "数据集" "样本数"
printf "%s\n" "$(printf '=%.0s' {1..40})"

for proj in $PROJECTS; do
  for opt in 0 1 2 3; do
    f="$OUT/binskel_${proj}_o${opt}.jsonl"
    name="binskel_${proj}_o${opt}"
    if [ ! -f "$f" ]; then
      printf "%-22s %10s\n" "$name" "-"
      continue
    fi
    n=$(wc -l < "$f" 2>/dev/null || echo 0)
    total_samples=$((total_samples + n))
    total_datasets=$((total_datasets + 1))
    printf "%-22s %10d\n" "$name" "$n"
  done
done

printf "%s\n" "$(printf '=%.0s' {1..40})"
echo ""
echo "总计: $total_datasets 个数据集, $total_samples 条样本"
echo ""

echo "=============================================="
echo "  O0 对齐质量 (有 alignment_matrix 的数据集)"
echo "=============================================="
for proj in $PROJECTS; do
  f="$OUT/binskel_${proj}_o0.jsonl"
  if [ -f "$f" ]; then
    echo ""
    echo "--- $proj O0 ($(wc -l < "$f") 样本) ---"
    python3 "$WORKDIR/scripts/validate_alignment.py" "$f" 2>/dev/null | tail -15
  fi
done 2>/dev/null || true

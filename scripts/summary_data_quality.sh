#!/bin/bash
# 汇总当前 binskel 数据效果：样本数 + O0 对齐质量
# 用法: summary_data_quality.sh [输出目录] [O0的jsonl路径]
# 例:  bash scripts/summary_data_quality.sh /home/wuqiongmin/structlift_datasets

OUT="${1:-$HOME/structlift_datasets}"
O0_JSONL="${2:-$OUT/binskel_zlib_o0.jsonl}"
WORKDIR="$(dirname "$(dirname "$(realpath "$0")")")"

echo "=============================================="
echo "  数据量汇总"
echo "=============================================="
for f in "$OUT"/binskel_zlib_o*.jsonl; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  echo "  $(basename "$f"): $n 条"
done
echo ""

if [ -f "$O0_JSONL" ]; then
  echo "=============================================="
  echo "  O0 对齐质量 (MD 第七步)"
  echo "=============================================="
  python "$WORKDIR/scripts/validate_alignment.py" "$O0_JSONL"
else
  echo "未找到 $O0_JSONL，跳过质量校验"
fi

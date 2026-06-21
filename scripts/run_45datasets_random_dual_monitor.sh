#!/usr/bin/env bash
# 9 项目 × 5 档（O0–O3 + Os）= 45 个 binskel jsonl：对各文件随机抽样，做「结构 + 语义」对齐监测。
#
# 依赖：与 build_binskel 相同（仓库根、Python3、tree-sitter 等），见 monitor_bb_alignment_dual.py。
#
# 与构建脚本对应关系：
#   - 批量生成 45 个 jsonl：scripts/rebuild_binskel_all_9.sh（需先各项目已编译 ELF+IR）
#   - 单项目 zlib：scripts/compile_zlib.sh + scripts/build_zlib_binskel.sh
#   - 数据集统计：scripts/binskel_alignment_batch_stats.py
#
# 用法（在 structlift 仓库根）：
#   export DATA_DIR="$HOME/structlift_datasets"    # jsonl 所在目录
#   bash scripts/run_45datasets_random_dual_monitor.sh
#
# 可调：
#   export RANDOM_PER_FILE=5   # 每个数据集随机抽几条（默认 5）
#   export SEED=42
#   export OUT_CSV="$PWD/bb_dual_random_45x5.csv"
#   export BY_BB=1             # 非空则加 --by-bb：CSV 为 **每个基本块一行**（非每函数一行）
#

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$HOME/structlift_datasets}"
RANDOM_PER_FILE="${RANDOM_PER_FILE:-5}"
SEED="${SEED:-42}"
OUT_CSV="${OUT_CSV:-$ROOT/bb_dual_random_${RANDOM_PER_FILE}each.csv}"

echo "[1/2] 45 数据集随机抽样 + 结构/语义双监测 -> $OUT_CSV"
EXTRA=()
if [ -n "${BY_BB:-}" ]; then
  EXTRA+=(--by-bb)
  echo "  （BY_BB 已设：输出为基本块粒度）"
fi
python3 "$ROOT/scripts/monitor_bb_alignment_dual.py" \
  --scan \
  --all-default-paths \
  --out-dir "$DATA_DIR" \
  --opt-levels "0,1,2,3,os" \
  --random-per-file "$RANDOM_PER_FILE" \
  --seed "$SEED" \
  "${EXTRA[@]}" \
  --csv > "$OUT_CSV"

echo "[2/2] 完成。行数（含表头）: $(wc -l < "$OUT_CSV")"
if [ -n "${BY_BB:-}" ]; then
  echo "  BY_BB 模式：行数 ≈ 各抽样函数内含 BB 之和 + 表头（远大于 45×每文件条数）。"
else
  echo "  预期约: $(( 45 * RANDOM_PER_FILE + 1 ))（45 文件 × 每文件 $RANDOM_PER_FILE 条 + 表头）若某 jsonl 缺失则更少。"
fi

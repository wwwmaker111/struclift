#!/bin/bash
# 完整流水线：编译 → 提取 BB↔源码行号对齐
# 规则：
#   1. 编译必须用 -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fstandalone-debug
#   2. DWARF .debug_line 中只用 is_stmt=True 的记录（语句边界），过滤噪声

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# SQLite 路径（可按需修改）
ELF="${ELF:-$HOME/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000/sqlite3}"
SRC="${SRC:-$(dirname "$ELF")}"
OUT="${OUT:-$HOME/structlift_datasets/sqlite_O0_binskel_full.jsonl}"

echo "=== Step 1: 确认 ELF 已用正确参数编译 ==="
if [ ! -f "$ELF" ]; then
  echo "ELF 不存在: $ELF"
  echo "请先执行 compile_for_alignment.sh 或 configure+make（使用 CFLAGS）"
  exit 1
fi
file "$ELF"

echo ""
echo "=== Step 2: 提取 BB↔源码对齐（is_stmt=True）==="
python build_binskel_dataset.py \
  --elf "$ELF" \
  --src "$SRC" \
  --out "$OUT"

echo ""
echo "完成: $OUT"

#!/bin/bash
# 按对齐要求编译：必须用这些参数，少一个都会增加对齐难度
# -O0 -g3 -fstandalone-debug 确保 DWARF 完整
# -fno-inline/-fno-unroll/-fno-vectorize 避免优化引入噪声
# is_stmt=True 的记录依赖上述参数才能准确

set -e

FLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"

if [ $# -lt 2 ]; then
  echo "用法: $0 <source.c> <output_prefix> [extra_sources...]"
  echo "例:   $0 sqlite3.c sqlite3 shell.c"
  echo "产出: <output_prefix>.bin 和 <output_prefix>.ll"
  exit 1
fi

SRC="$1"
PREFIX="$2"
shift 2
EXTRA="$@"

echo "[1/2] 编译二进制 (含 DWARF): ${PREFIX}.bin"
clang $FLAGS "$SRC" $EXTRA -o "${PREFIX}.bin" -ldl -lpthread -lm -lz 2>/dev/null || \
clang $FLAGS "$SRC" $EXTRA -o "${PREFIX}.bin" -ldl -lpthread

echo "[2/2] 产出 LLVM IR: ${PREFIX}.ll"
clang $FLAGS -emit-llvm -S "$SRC" -o "${PREFIX}.ll"

echo "完成: ${PREFIX}.bin, ${PREFIX}.ll"

#!/usr/bin/env bash
# 在 WSL 中从仓库根目录调用：bash scripts/_rebuild_zlib_o0_dataset.sh
# 先按 MD 用 -g3 重链 zlib_o0（当前 tree 里若缺 DWARF 则无法产出 149 条语句档）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZDIR="${ZDIR:-$ROOT/zlib_build}"
cd "$ZDIR"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
LIB_SRCS="adler32.c crc32.c deflate.c infback.c inffast.c inflate.c inftrees.c trees.c zutil.c compress.c uncompr.c gzclose.c gzlib.c gzread.c gzwrite.c"
MINIGZIP_SRC="test/minigzip.c"

if [ ! -f "$ZDIR/configure" ]; then
  echo "ERROR: 不是 zlib 源码目录: $ZDIR"
  exit 1
fi

echo "[1/2] 重编译 zlib_o0 + zlib_o0.ll (clang -g3) ..."
export CC=clang
make clean 2>/dev/null || true
make CFLAGS="$MD_CFLAGS" LDFLAGS="-g" minigzip
cp -f minigzip zlib_o0

rm -f *.bc zlib_o0.bc 2>/dev/null || true
for c in $LIB_SRCS; do
  clang $MD_CFLAGS -emit-llvm -c -I. "$c" -o "${c%.c}.bc"
done
clang $MD_CFLAGS -emit-llvm -c -I. "$MINIGZIP_SRC" -o minigzip.bc
llvm-link *.bc -o zlib_o0.bc
llvm-dis zlib_o0.bc -o zlib_o0.ll
rm -f *.bc zlib_o0.bc 2>/dev/null || true

if readelf -S zlib_o0 2>/dev/null | grep -qE '\.debug_line'; then
  echo "  OK: zlib_o0 含 .debug_line"
else
  echo "  WARN: zlib_o0 仍无 .debug_line，数据集可能极少或为 0"
fi

echo "[2/2] 构建 binskel_zlib_o0.jsonl ..."
cd "$ROOT"
python3 build_binskel_dataset_md.py \
  --elf "$ZDIR/zlib_o0" \
  --src "$ZDIR" \
  --out "$ROOT/binskel_zlib_o0.jsonl" \
  --llvm-ir "$ZDIR/zlib_o0.ll" \
  --opt o0 \
  --workers 8 \
  --num-opcodes 1024 \
  --src-vocab-size 32000

echo "行数:"
wc -l "$ROOT/binskel_zlib_o0.jsonl"

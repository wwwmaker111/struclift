#!/usr/bin/env bash
# 仅在已有 zlib 源码树中编译 -Os，产出 zlib_os、zlib_os.ll（不修改 O0–O3 已有产物）
# 用法: bash scripts/os_only/compile_os_zlib.sh <zlib源码根目录>
set -euo pipefail
STRIP=:
export STRIP
DIR="${1:?缺少 zlib 源码根目录参数}"
[ -f "$DIR/configure" ] || { echo "不是 zlib 源码根: $DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
LIB_SRCS="adler32.c crc32.c deflate.c infback.c inffast.c inflate.c inftrees.c trees.c zutil.c compress.c uncompr.c gzclose.c gzlib.c gzread.c gzwrite.c"
MINIGZIP_SRC="test/minigzip.c"

cd "$DIR"
export CC=clang
make clean 2>/dev/null || true
make CFLAGS="$OS_CFLAGS" LDFLAGS="-g" STRIP="$STRIP" minigzip
cp -f minigzip zlib_os
echo "    -> $DIR/zlib_os"

rm -f *.bc zlib_os.bc 2>/dev/null || true
for c in $LIB_SRCS; do
  clang $OS_CFLAGS -emit-llvm -c -I. "$c" -o "${c%.c}.bc"
done
clang $OS_CFLAGS -emit-llvm -c -I. "$MINIGZIP_SRC" -o minigzip.bc
llvm-link *.bc -o zlib_os.bc
llvm-dis zlib_os.bc -o zlib_os.ll
rm -f *.bc zlib_os.bc 2>/dev/null || true
echo "    -> $DIR/zlib_os.ll"
echo "完成 Os: zlib_os, zlib_os.ll"

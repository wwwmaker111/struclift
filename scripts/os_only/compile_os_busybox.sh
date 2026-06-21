#!/usr/bin/env bash
# 仅在已有 BusyBox 源码树中编译 -Os，产出 busybox_os、busybox_os.ll
# 用法: bash scripts/os_only/compile_os_busybox.sh <busybox源码根目录>
set -euo pipefail
DIR="${1:?用法: $0 <busybox源码目录>}"
[ -f "$DIR/Makefile" ] || { echo "不是 busybox 源码根: $DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)
BB_INC="-std=gnu99 -Iinclude -Ilibbb -I. -include include/autoconf.h -D_GNU_SOURCE -DNDEBUG -funsigned-char -w"

cd "$DIR"
make clean 2>/dev/null || true
make -j"$NPROC" CC=clang EXTRA_CFLAGS="$OS_CFLAGS"
cp -f busybox_unstripped busybox_os
echo "    -> $DIR/busybox_os"

BC_DIR="$DIR/.bc_os_only"
rm -rf "$BC_DIR"
mkdir -p "$BC_DIR"
find . -name "*.o" \
  -not -name "built-in.o" \
  -not -name "*.mod.o" \
  -not -path "./scripts/*" \
  -not -path "./.bc_*/*" \
  | sort > /tmp/bb_ofiles_os.txt
while IFS= read -r ofile; do
  cfile="${ofile%.o}.c"
  if [ -f "$cfile" ]; then
    bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
    bn="${bn%.c}.bc"
    cfile_dir=$(dirname "$cfile")
    clang $OS_CFLAGS $BB_INC -I"$cfile_dir" -emit-llvm -c "$cfile" -o "$BC_DIR/$bn" 2>/dev/null || true
  fi
done < /tmp/bb_ofiles_os.txt
rm -f /tmp/bb_ofiles_os.txt
n_bc=$(find "$BC_DIR" -name "*.bc" 2>/dev/null | wc -l)
if [ "$n_bc" -gt 0 ]; then
  llvm-link "$BC_DIR"/*.bc -o "$DIR/busybox_os.bc" 2>/dev/null
  if [ -f "$DIR/busybox_os.bc" ]; then
    llvm-dis "$DIR/busybox_os.bc" -o "$DIR/busybox_os.ll"
    rm -f "$DIR/busybox_os.bc"
    echo "    -> $DIR/busybox_os.ll"
  fi
fi
rm -rf "$BC_DIR"
echo "完成 Os: busybox_os, busybox_os.ll"

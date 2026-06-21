#!/usr/bin/env bash
# 仅在已有 OpenSSL 源码树中编译 -Os，产出 openssl_os、ir_os/（不修改 O0–O3）
# 用法: bash scripts/os_only/compile_os_openssl.sh <openssl源码根目录>
set -euo pipefail
DIR="${1:?用法: $0 <openssl源码目录>}"
[ -f "$DIR/Configure" ] || { echo "不是 OpenSSL 源码根: $DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)
SSL_INC="-Iinclude -I. -Icrypto -Iproviders -Iproviders/common/include -Iproviders/implementations/include"
SSL_DEFS="-DOPENSSL_BUILDING_OPENSSL -DNDEBUG"

compile_openssl_os() {
  local cflags="$OS_CFLAGS"
  echo "清理并 Configure (Os) ..."
  cd "$DIR"
  make clean 2>/dev/null || true
  make distclean 2>/dev/null || true
  ./Configure linux-x86_64 \
    no-shared no-tests no-fips \
    --prefix="$DIR/install_os" \
    CC=clang \
    CFLAGS="$cflags" \
    2>&1 | tail -3
  make -j"$NPROC" build_libs 2>&1 | tail -3
  if [ -f libcrypto.a ] && [ -f libssl.a ]; then
    local TMPDIR="$DIR/.link_os"
    rm -rf "$TMPDIR"
    mkdir -p "$TMPDIR"
    cd "$TMPDIR"
    ar x "$DIR/libcrypto.a"
    ar x "$DIR/libssl.a"
    clang $cflags -nostdlib -r *.o -o "$DIR/openssl_os" 2>/dev/null || \
      ld -r *.o -o "$DIR/openssl_os" 2>/dev/null || true
    cd "$DIR"
    rm -rf "$TMPDIR"
    [ -f "$DIR/openssl_os" ] && echo "    -> $DIR/openssl_os"
  fi
}

cd "$DIR"
compile_openssl_os

IR_SRC_DIRS=()
for d in crypto ssl providers engines; do
  [ -d "$d" ] && IR_SRC_DIRS+=("$d")
done

IR_DIR="$DIR/ir_os"
rm -rf "$IR_DIR"
mkdir -p "$IR_DIR"
n_ir=0
while IFS= read -r -d '' cfile; do
  bn=$(echo "$cfile" | tr '/' '_')
  bn="${bn%.c}.ll"
  cfile_dir=$(dirname "$cfile")
  if clang $OS_CFLAGS $SSL_INC $SSL_DEFS -I"$cfile_dir" -emit-llvm -S "$cfile" -o "$IR_DIR/$bn" 2>/dev/null; then
    n_ir=$((n_ir+1))
  fi
done < <(find "${IR_SRC_DIRS[@]}" -name "*.c" -not -path "*/test/*" -not -path "*/fuzz/*" -print0 2>/dev/null)
echo "    -> $IR_DIR/ ($n_ir 个 .ll)"
echo "完成 Os: openssl_os, ir_os/"

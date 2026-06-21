#!/bin/bash
set -e
SSL_VERSION=3.3.1
DIR=$HOME/struclift_wsl/datasets/openssl/openssl-${SSL_VERSION}
MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
cd "$DIR"

echo '=== [1/2] O0 Configure + make ==='
make clean 2>/dev/null || true
make distclean 2>/dev/null || true
./Configure linux-x86_64 no-shared no-tests no-fips \
  --prefix="$DIR/install_o0" \
  CC=clang CFLAGS="$MD_CFLAGS" 2>&1 | tail -3
make -j$(nproc) build_libs 2>&1 | tail -3

if [ ! -f "include/openssl/configuration.h" ]; then
  echo "ERROR: configuration.h 未生成"
  exit 1
fi
echo "configuration.h OK"

if [ -f libcrypto.a ] && [ -f libssl.a ]; then
  echo "链接 O0 ELF ..."
  TMPDIR="$DIR/.link_o0"
  rm -rf "$TMPDIR"
  mkdir -p "$TMPDIR"
  cd "$TMPDIR"
  ar x "$DIR/libcrypto.a"
  ar x "$DIR/libssl.a"
  clang $MD_CFLAGS -nostdlib -r *.o -o "$DIR/openssl_o0" 2>/dev/null || \
  ld -r *.o -o "$DIR/openssl_o0" 2>/dev/null || true
  cd "$DIR"
  rm -rf "$TMPDIR"
  ls -lh "$DIR/openssl_o0"
fi

echo '=== [2/2] 逐文件 .ll 生成 ==='
IR_DIR="$DIR/ir_o0"
rm -rf "$IR_DIR"
mkdir -p "$IR_DIR"

SSL_INC="-Iinclude -I. -Icrypto -Iproviders -Iproviders/common/include -Iproviders/implementations/include"
SSL_DEFS="-DOPENSSL_BUILDING_OPENSSL -DNDEBUG"

IR_SRC_DIRS=()
for d in crypto ssl providers engines; do
  [ -d "$d" ] && IR_SRC_DIRS+=("$d")
done

n_ok=0
n_fail=0
fail_list=""

while IFS= read -r -d '' cfile; do
  bn=$(echo "$cfile" | tr '/' '_')
  bn="${bn%.c}.ll"
  cfile_dir=$(dirname "$cfile")
  if clang $MD_CFLAGS $SSL_INC $SSL_DEFS -I"$cfile_dir" -emit-llvm -S "$cfile" -o "$IR_DIR/$bn" 2>/dev/null; then
    n_ok=$((n_ok+1))
  else
    n_fail=$((n_fail+1))
    fail_list="$fail_list $cfile"
  fi
done < <(find "${IR_SRC_DIRS[@]}" -name '*.c' -not -path '*/test/*' -not -path '*/fuzz/*' -print0 2>/dev/null)

echo "逐文件 .ll: 成功 $n_ok, 失败 $n_fail"
echo "-> $IR_DIR/ ($n_ok 个 .ll)"
if [ "$n_fail" -gt 0 ]; then
  echo "失败文件(前20):"
  echo "$fail_list" | tr ' ' '\n' | head -20 | while read -r f; do
    [ -n "$f" ] && echo "  $f"
  done
fi

n_funcs=$(grep -r '^define ' "$IR_DIR/" 2>/dev/null | wc -l)
echo "逐文件 IR 函数总数: $n_funcs"
echo "=== 完成 ==="

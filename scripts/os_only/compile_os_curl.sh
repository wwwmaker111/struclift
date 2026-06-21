#!/usr/bin/env bash
# 仅在已有 curl 源码树中编译 -Os，产出 curl_os、curl_os.ll
# 用法: bash scripts/os_only/compile_os_curl.sh <curl源码根目录>
set -euo pipefail
DIR="${1:?用法: $0 <curl源码目录>}"
[ -f "$DIR/configure" ] || { echo "不是 curl 源码根: $DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
CURL_EXTRA_LDFLAGS="${LDFLAGS:-} -g"
INC="-I. -Iinclude -Ilib -Isrc"

cd "$DIR"
export CC=clang
curl_make() {
  local _cflags="$1"
  shift
  make CFLAGS="$_cflags" LDFLAGS="$CURL_EXTRA_LDFLAGS" "$@"
}

emit_curl_os_ll() {
  local F="$1"
  local BC_DIR="$DIR/.bc_curl_os"
  rm -rf "$BC_DIR"
  mkdir -p "$BC_DIR"
  STRCONST_H="$BC_DIR/curl_strconst.h"
  printf '%s\n' '/* generated */' '#define STRCONST(x) (x,(size_t)(sizeof(x)-1))' > "$STRCONST_H"
  BC_CFLAGS="$F -DHAVE_CONFIG_H"
  BC_LIB_CFLAGS="$BC_CFLAGS -DBUILDING_LIBCURL -include $STRCONST_H"
  for c in lib/*.c; do
    [ -f "$c" ] || continue
    bn=$(echo "$c" | tr '/' '_' | sed 's/\.c$/.bc/')
    clang $BC_LIB_CFLAGS $INC -emit-llvm -c "$c" -o "$BC_DIR/$bn" || true
  done
  for c in src/*.c; do
    [ -f "$c" ] || continue
    bn=$(echo "$c" | tr '/' '_' | sed 's/\.c$/.bc/')
    clang $BC_CFLAGS $INC -emit-llvm -c "$c" -o "$BC_DIR/$bn" || true
  done
  n_bc=$(find "$BC_DIR" -name "*.bc" 2>/dev/null | wc -l)
  if [ "$n_bc" -gt 0 ]; then
    llvm-link "$BC_DIR"/*.bc -o "$DIR/curl_os.bc"
    llvm-dis "$DIR/curl_os.bc" -o "$DIR/curl_os.ll"
    rm -rf "$BC_DIR" "$DIR/curl_os.bc"
    echo "    -> $DIR/curl_os.ll"
  else
    echo "WARN: curl_os.ll 未生成"
    rm -rf "$BC_DIR"
  fi
}

make clean 2>/dev/null || true
curl_make "$OS_CFLAGS"
CURL_REAL=""
[ -f src/.libs/curl ] && CURL_REAL="src/.libs/curl"
[ -z "$CURL_REAL" ] && [ -f src/curl ] && file src/curl 2>/dev/null | grep -q ELF && CURL_REAL="src/curl"
[ -z "$CURL_REAL" ] && [ -f src/curl ] && CURL_REAL="src/curl"
[ -f "$CURL_REAL" ] && cp -f "$CURL_REAL" curl_os || { echo "未找到 curl 二进制"; exit 1; }
echo "    -> $DIR/curl_os"
emit_curl_os_ll "$OS_CFLAGS"
echo "完成 Os: curl_os, curl_os.ll"

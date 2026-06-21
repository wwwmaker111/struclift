#!/usr/bin/env bash
# 仅在已有 libxml2 源码树中编译 -Os，产出 xmllint_os、ir_os/
# 用法: bash scripts/os_only/compile_os_libxml2.sh <libxml2源码根目录>
set -euo pipefail
WORK_DIR="${1:?用法: $0 <libxml2源码目录>}"
[ -f "$WORK_DIR/configure" ] || { echo "不是 libxml2 源码根: $WORK_DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)
XML_INC="-I. -Iinclude -Iinclude/libxml"

# 与 compile_libxml2.sh 一致：产物目录 = 若 USE_TMP 则需用户传两参数；默认同树
OUT_DIR="${2:-$WORK_DIR}"
mkdir -p "$OUT_DIR"

get_cfile() {
  local ofile="$1"
  echo "$ofile" | sed 's|/.libs/|/|g; s|\.o$|.c|'
}

emit_ir_dir() {
  local cflags="$1"
  local irdir="$2"
  rm -rf "$irdir"
  mkdir -p "$irdir"
  find . -name "*.o" -not -path "./.bc_o*/*" -not -path "*/.deps/*" 2>/dev/null | sort > /tmp/xml_ofiles_os.txt
  n_ok=0
  while IFS= read -r ofile; do
    cfile=$(get_cfile "$ofile")
    [ -f "$cfile" ] || continue
    bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
    bn="${bn%.c}.ll"
    cdir=$(dirname "$cfile")
    if clang $cflags $XML_INC -I"$cdir" -emit-llvm -S "$cfile" -o "$irdir/$bn" 2>/dev/null; then
      n_ok=$((n_ok+1))
    fi
  done < /tmp/xml_ofiles_os.txt
  rm -f /tmp/xml_ofiles_os.txt
  echo "    -> $irdir/ ($n_ok 个 .ll)"
}

cd "$WORK_DIR"
make clean 2>/dev/null || true
./configure CC=clang CFLAGS="$OS_CFLAGS" --without-python --without-readline 2>&1 | tail -2
make -j"$NPROC" 2>&1 | tail -3
XMLLINT_BIN=""
[ -f .libs/xmllint ] && XMLLINT_BIN=".libs/xmllint"
[ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && file xmllint 2>/dev/null | grep -q ELF && XMLLINT_BIN="xmllint"
[ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && XMLLINT_BIN="xmllint"
[ -n "$XMLLINT_BIN" ] && cp -f "$XMLLINT_BIN" "$OUT_DIR/xmllint_os" && echo "    -> $OUT_DIR/xmllint_os"
emit_ir_dir "$OS_CFLAGS" "$OUT_DIR/ir_os"
echo "完成 Os: xmllint_os, ir_os/"

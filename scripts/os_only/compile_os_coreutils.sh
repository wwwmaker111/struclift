#!/usr/bin/env bash
# 仅在已有 coreutils 构建树中编译 -Os，产出 coreutils_os、ir_os/（与 compile_coreutils 同目录约定）
# 用法: bash scripts/os_only/compile_os_coreutils.sh <源码/构建目录> [产物目录]
# 若 O0–O3 在 datasets/.../coreutils-9.4/ 下，通常两参数相同或只传一个。
set -euo pipefail
WORK_DIR="${1:?用法: $0 <coreutils 构建目录> [产物目录]}"
OUT_DIR="${2:-$WORK_DIR}"
[ -f "$WORK_DIR/configure" ] || { echo "目录无 configure: $WORK_DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)
CU_INC="-I. -I./lib -Ilib -I./lib -Isrc -I./src"
CU_LIB_FLAGS="-Wno-cast-qual -Wno-conversion -Wno-float-equal -Wno-sign-compare -Wno-undef -Wno-unused-function -Wno-unused-parameter -Wno-float-conversion -Wimplicit-fallthrough -Wno-pedantic -Wno-sign-conversion -Wno-type-limits"

get_cfile() {
  local ofile="$1"
  local base=$(basename "$ofile" .o)
  local dir=$(dirname "$ofile")
  if [[ "$base" == libcoreutils_a-* ]]; then
    echo "${dir}/${base#libcoreutils_a-}.c"
  elif [[ "$base" == coreutils-* ]]; then
    echo "${dir}/${base#coreutils-}.c"
  else
    echo "${dir}/${base}.c"
  fi
}

emit_ir_dir() {
  local cflags="$1"
  local irdir="$2"
  rm -rf "$irdir"
  mkdir -p "$irdir"
  find lib src -name "*.o" -not -path "*/.deps/*" 2>/dev/null | sort > /tmp/cu_ofiles_os.txt
  n_ok=0
  while IFS= read -r ofile; do
    cfile=$(get_cfile "$ofile")
    [ -f "$cfile" ] || continue
    bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
    bn="${bn%.c}.ll"
    cdir=$(dirname "$cfile")
    extra=""
    [[ "$cfile" == lib/* ]] && extra="$CU_LIB_FLAGS"
    [[ "$cfile" == src/coreutils.c ]] && extra="-DSINGLE_BINARY"
    if clang $cflags $CU_INC $extra -I"$cdir" -emit-llvm -S "$cfile" -o "$irdir/$bn" 2>/dev/null; then
      n_ok=$((n_ok+1))
    fi
  done < /tmp/cu_ofiles_os.txt
  rm -f /tmp/cu_ofiles_os.txt
  echo "    -> $irdir/ ($n_ok 个 .ll)"
}

CC_BUILD="${CC_BUILD:-clang}"
export gl_cv_cc_undeclared_builtin_works=no
mkdir -p "$OUT_DIR"

cd "$WORK_DIR"
make clean 2>/dev/null || true
./configure --enable-single-binary CC="$CC_BUILD" CFLAGS="$OS_CFLAGS" 2>&1 | tail -2
make -j"$NPROC" 2>&1 | tail -3
BIN=$(find . -name "coreutils" -type f -path "*/src/*" 2>/dev/null | head -1)
[ -z "$BIN" ] && BIN=$(find . -name "coreutils" -type f 2>/dev/null | head -1)
[ -n "$BIN" ] && cp -f "$BIN" "$OUT_DIR/coreutils_os" && echo "    -> $OUT_DIR/coreutils_os"
emit_ir_dir "$OS_CFLAGS" "$OUT_DIR/ir_os"
echo "完成 Os: coreutils_os, ir_os/"

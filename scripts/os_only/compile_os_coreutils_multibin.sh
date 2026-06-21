#!/usr/bin/env bash
# 仅在已有 coreutils 源码树中编译「多产物」-Os（不启用 single-binary），
# 产出 multibin_os/、ir_os/（与 compile_coreutils_multibin.sh 最后一档相同）。
# O0–O3 已由 multibin 编过时，可用本脚本单独补 Os，避免全流程重跑。
#
# 用法:
#   bash scripts/os_only/compile_os_coreutils_multibin.sh <coreutils 源码根目录> [产物写入目录，默认同上]
#
set -euo pipefail
STRIP=:
export STRIP
WORK_DIR="${1:?缺少 coreutils 源码根目录}"
OUT_DIR="${2:-$WORK_DIR}"
[ -f "$WORK_DIR/configure" ] || { echo "无 configure: $WORK_DIR"; exit 1; }

OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)
CU_INC="-I. -I./lib -Ilib -I./lib -Isrc -I./src"
CU_LIB_FLAGS="-Wno-cast-qual -Wno-conversion -Wno-float-equal -Wno-sign-compare -Wno-undef -Wno-unused-function -Wno-unused-parameter -Wno-float-conversion -Wimplicit-fallthrough -Wno-pedantic -Wno-sign-conversion -Wno-type-limits"

get_cfile() {
  local ofile="$1"
  local base
  base=$(basename "$ofile" .o)
  local dir
  dir=$(dirname "$ofile")
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
  find lib src -name "*.o" -not -path "*/.deps/*" 2>/dev/null | sort > /tmp/cu_ofiles_cu_os_mb.txt
  n_ok=0
  while IFS= read -r ofile; do
    cfile=$(get_cfile "$ofile")
    [ -f "$cfile" ] || continue
    bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
    bn="${bn%.c}.ll"
    cdir=$(dirname "$cfile")
    extra=""
    [[ "$cfile" == lib/* ]] && extra="$CU_LIB_FLAGS"
    if clang $cflags $CU_INC $extra -I"$cdir" -emit-llvm -S "$cfile" -o "$irdir/$bn" 2>/dev/null; then
      n_ok=$((n_ok+1))
    fi
  done < /tmp/cu_ofiles_cu_os_mb.txt
  rm -f /tmp/cu_ofiles_cu_os_mb.txt
  echo "    ir_os: $n_ok 个 .ll -> $irdir"
}

copy_src_elfs() {
  local dest="$1"
  rm -rf "$dest"
  mkdir -p "$dest"
  local n=0
  for f in src/*; do
    [ -f "$f" ] || continue
    [ -x "$f" ] || continue
    file "$f" 2>/dev/null | grep -q ELF || continue
    cp -f "$f" "$dest/$(basename "$f")"
    n=$((n+1))
  done
  echo "    multibin_os: $n 个 ELF -> $dest"
}

CC_BUILD="${CC_BUILD:-clang}"
export gl_cv_cc_undeclared_builtin_works=no
mkdir -p "$OUT_DIR"

cd "$WORK_DIR"
echo "[multibin Os] CC=$CC_BUILD"
make clean 2>/dev/null || true
./configure CC="$CC_BUILD" CFLAGS="$OS_CFLAGS" 2>&1 | tail -3
make -j"$NPROC" STRIP="$STRIP" 2>&1 | tail -5

copy_src_elfs "$OUT_DIR/multibin_os"
emit_ir_dir "$OS_CFLAGS" "$OUT_DIR/ir_os"
echo "完成: $OUT_DIR/multibin_os/ + $OUT_DIR/ir_os/"

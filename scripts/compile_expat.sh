#!/usr/bin/env bash
# expat: static xmlwf ELF + ir_o0..ir_o3
# Usage: compile_expat.sh [base_dir]
# Env: EXPAT_VERSION=2.6.4

set -eu
VER="${EXPAT_VERSION:-2.6.4}"
# R_2_6_4 tag style for GitHub
TAG="R_${VER//./_}"
BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/expat}"
URL="https://github.com/libexpat/libexpat/releases/download/${TAG}/expat-${VER}.tar.xz"
TARBALL="/tmp/expat-${VER}.tar.xz"
DIR="${BASE_DIR}/expat-${VER}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"
if [ ! -f "$TARBALL" ]; then
  wget -q -O "$TARBALL" "$URL" || curl -fsSL -o "$TARBALL" "$URL"
fi
if [ ! -d "$DIR" ]; then
  tar -xJf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local flags="$1"
  local ird="$2"
  rm -rf "$ird"
  mkdir -p "$ird"
  for c in lib/*.c; do
    [ -f "$c" ] || continue
    bn=$(basename "$c" .c)
    clang $flags -Ilib -I. -DHAVE_EXPAT_CONFIG_H -emit-llvm -S "$c" -o "$ird/${bn}.ll" 2>/dev/null || true
  done
  [ -f xmlwf/xmlwf.c ] && clang $flags -Ilib -I. -Ixmlwf -emit-llvm -S xmlwf/xmlwf.c -o "$ird/xmlwf.ll" 2>/dev/null || true
}

build_one() {
  local tag="$1"
  local flags="$2"
  local out="$3"
  echo "=== expat $tag ==="
  make distclean 2>/dev/null || true
  CC=clang CFLAGS="$flags" ./configure --disable-shared
  # 新版 expat 为顶层单一 Makefile，无目标 xmlwf/xmlwf；默认 all 会编 lib+xmlwf
  make -j"$NPROC"
  XW=""
  for cand in xmlwf/.libs/xmlwf xmlwf/xmlwf; do
    if [ -f "$cand" ]; then XW="$cand"; break; fi
  done
  [ -n "$XW" ] || { echo "ERROR: xmlwf not found (try: find . -name xmlwf -type f)"; exit 1; }
  cp -f "$XW" "$out"
  echo "    -> $out"
  emit_ir "$flags" "$DIR/ir_${tag}"
}

echo "[expat] O0..."
build_one "o0" "$MD_CFLAGS" "$DIR/expat_o0"
echo "[expat] O1..."
build_one "o1" "$O1_CFLAGS" "$DIR/expat_o1"
echo "[expat] O2..."
build_one "o2" "$O2_CFLAGS" "$DIR/expat_o2"
echo "[expat] O3..."
build_one "o3" "$O3_CFLAGS" "$DIR/expat_o3"
echo "Done: $DIR/expat_o0..o3"

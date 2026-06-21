#!/usr/bin/env bash
# libpng: static pngtest ELF + ir_o0..ir_o3 (per-file .ll), full MD binskel pipeline
# Usage: compile_libpng.sh [base_dir]
# Env: LIBPNG_VERSION=1.6.47

set -eu
VER="${LIBPNG_VERSION:-1.6.47}"
BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/libpng}"
URL="https://downloads.sourceforge.net/project/libpng/libpng16/${VER}/libpng-${VER}.tar.gz"
TARBALL="/tmp/libpng-${VER}.tar.gz"
DIR="${BASE_DIR}/libpng-${VER}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

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
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local flags="$1"
  local ird="$2"
  rm -rf "$ird"
  mkdir -p "$ird"
  # Core lib sources (top-level png*.c) + pngtest
  for c in png*.c; do
    [ -f "$c" ] || continue
    clang $flags -I. -emit-llvm -S "$c" -o "$ird/${c%.c}.ll" 2>/dev/null || true
  done
  [ -f pngtest.c ] && clang $flags -I. -emit-llvm -S pngtest.c -o "$ird/pngtest.ll" 2>/dev/null || true
}

build_one() {
  local tag="$1"
  local flags="$2"
  local out="$3"
  echo "=== libpng $tag ==="
  make distclean 2>/dev/null || true
  CC=clang CFLAGS="$flags" ./configure --disable-shared --enable-static
  make -j"$NPROC" pngtest
  # Executable name may be pngtest or .libs/pngtest
  if [ -f pngtest ]; then
    cp -f pngtest "$out"
  elif [ -f .libs/pngtest ]; then
    cp -f .libs/pngtest "$out"
  else
    echo "ERROR: pngtest not built"
    exit 1
  fi
  echo "    -> $out"
  emit_ir "$flags" "$DIR/ir_${tag}"
}

echo "[libpng] O0..."
build_one "o0" "$MD_CFLAGS" "$DIR/libpng_o0"
echo "[libpng] O1..."
build_one "o1" "$O1_CFLAGS" "$DIR/libpng_o1"
echo "[libpng] O2..."
build_one "o2" "$O2_CFLAGS" "$DIR/libpng_o2"
echo "[libpng] O3..."
build_one "o3" "$O3_CFLAGS" "$DIR/libpng_o3"
echo "Done: $DIR/libpng_o0..o3, ir_o0..ir_o3"

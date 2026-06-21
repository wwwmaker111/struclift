#!/usr/bin/env bash
# pcre2: static pcre2grep ELF + ir_o0..ir_o3
# Usage: compile_pcre2.sh [base_dir]
# Env: PCRE2_VERSION=10.44

set -eu
VER="${PCRE2_VERSION:-10.44}"
BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/pcre2}"
URL="https://github.com/PCRE2Project/pcre2/releases/download/pcre2-${VER}/pcre2-${VER}.tar.bz2"
TARBALL="/tmp/pcre2-${VER}.tar.bz2"
DIR="${BASE_DIR}/pcre2-${VER}"

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
  tar -xjf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local flags="$1"
  local ird="$2"
  rm -rf "$ird"
  mkdir -p "$ird"
  # 在 make 之后调用：含生成的 pcre2_chartables.c 等；跳过 JIT 测试与 fuzz
  for c in src/*.c; do
    [ -f "$c" ] || continue
    bn=$(basename "$c" .c)
    case "$bn" in
      pcre2_jit_test|pcre2fuzz*) continue ;;
    esac
    clang $flags -Isrc -I. -DHAVE_CONFIG_H -emit-llvm -S "$c" -o "$ird/${bn}.ll" 2>/dev/null || true
  done
}

build_one() {
  local tag="$1"
  local flags="$2"
  local out="$3"
  echo "=== pcre2 $tag ==="
  make distclean 2>/dev/null || true
  CC=clang CFLAGS="$flags" ./configure --disable-shared --enable-pcre2-8 --disable-jit
  make -j"$NPROC" pcre2grep
  # libtool：真实 ELF 多在 src/.libs/，顶层 .libs/ 常不存在
  PG=""
  for cand in src/.libs/pcre2grep .libs/pcre2grep src/pcre2grep; do
    if [ -f "$cand" ] && file -b "$cand" 2>/dev/null | grep -q ELF; then
      PG="$cand"
      break
    fi
  done
  if [ -z "$PG" ]; then
    while IFS= read -r f; do
      [ -f "$f" ] || continue
      file -b "$f" 2>/dev/null | grep -q ELF || continue
      PG="$f"
      break
    done < <(find . -maxdepth 4 -name pcre2grep -type f 2>/dev/null)
  fi
  [ -n "$PG" ] && [ -f "$PG" ] || { echo "ERROR: pcre2grep ELF not found (see src/.libs)"; exit 1; }
  cp -f "$PG" "$out"
  echo "    -> $out"
  emit_ir "$flags" "$DIR/ir_${tag}"
}

echo "[pcre2] O0..."
build_one "o0" "$MD_CFLAGS" "$DIR/pcre2_o0"
echo "[pcre2] O1..."
build_one "o1" "$O1_CFLAGS" "$DIR/pcre2_o1"
echo "[pcre2] O2..."
build_one "o2" "$O2_CFLAGS" "$DIR/pcre2_o2"
echo "[pcre2] O3..."
build_one "o3" "$O3_CFLAGS" "$DIR/pcre2_o3"
echo "Done: $DIR/pcre2_o0..o3"

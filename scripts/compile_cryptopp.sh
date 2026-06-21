#!/usr/bin/env bash
# Crypto++: O0–O3 relocatable ELF + per-TU .ll (bear + compile_commands, or cpp fallback)
#
# Requires: clang++, python3, ar, bear (recommended)
# Usage: compile_cryptopp.sh [base_dir]
# Env: CRYPTOPP_VERSION=8_9_0  (Git tag style)

set -eu

BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/cryptopp}"
CRYPTOPP_TAG="${CRYPTOPP_TAG:-CRYPTOPP_8_9_0}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_URL="https://github.com/weidai11/cryptopp/archive/refs/tags/${CRYPTOPP_TAG}.zip"
DIR=""

MD_CXXFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CXXFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CXXFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CXXFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"

DIR=$(find "$BASE_DIR" -maxdepth 1 -type d -name 'cryptopp-*' 2>/dev/null | head -1)
if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
  echo "[1/3] Download Crypto++ ${CRYPTOPP_TAG} ..."
  TMPZ="/tmp/cryptopp-${CRYPTOPP_TAG}.zip"
  wget -q -O "$TMPZ" "$ZIP_URL" || curl -sL -o "$TMPZ" "$ZIP_URL"
  unzip -q "$TMPZ" -d "$BASE_DIR"
  DIR=$(find "$BASE_DIR" -maxdepth 1 -type d -name 'cryptopp-*' | head -1)
fi
[ -n "$DIR" ] && [ -d "$DIR" ] || { echo "ERROR: Crypto++ source directory not found under $BASE_DIR"; exit 1; }

cd "$DIR"

merge_static_to_reloc() {
  local archive_path="$1"
  local out_elf="$2"
  local TMP
  TMP=$(mktemp -d)
  (
    cd "$TMP" || exit 1
    ar x "$archive_path"
    shopt -s nullglob
    objs=( *.o )
    if [ "${#objs[@]}" -eq 0 ]; then
      echo "ERROR: ar x produced no .o files"
      exit 1
    fi
    clang++ -nostdlib -r "${objs[@]}" -o "$out_elf" 2>/dev/null || ld -r "${objs[@]}" -o "$out_elf"
  )
  rm -rf "$TMP"
}

emit_ir_ccjson() {
  local root="$1"
  local ir_dir="$2"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$WORKDIR/scripts/emit_ir_from_compile_commands.py" "$root" "$ir_dir"
}

emit_ir_fallback_cpp() {
  local cxxflags="$1"
  local ir_dir="$2"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  local ok=0
  for f in *.cpp; do
    [ -f "$f" ] || continue
    case "$f" in
      test.cpp|bench*.cpp|datatest.cpp|validat*.cpp|adhoc*.cpp) continue ;;
    esac
    local out="$ir_dir/${f%.cpp}.ll"
    if clang++ $cxxflags -emit-llvm -S "$f" -o "$out" 2>/dev/null; then
      ok=$((ok + 1))
    fi
  done
  echo "emit_ir_fallback_cpp: ok=$ok -> $ir_dir"
}

build_one() {
  local tag="$1"
  local cxxflags="$2"

  echo "=== Crypto++ ${tag} ==="
  make clean 2>/dev/null || true
  rm -f compile_commands.json libcryptopp.a

  if command -v bear >/dev/null 2>&1; then
    bear -- make -j"$NPROC" libcryptopp.a CXX=clang++ CXXFLAGS="$cxxflags"
  else
    make -j"$NPROC" libcryptopp.a CXX=clang++ CXXFLAGS="$cxxflags"
  fi

  if [ ! -f libcryptopp.a ]; then
    echo "ERROR: libcryptopp.a not built"
    exit 1
  fi

  merge_static_to_reloc "$(pwd)/libcryptopp.a" "$DIR/cryptopp_${tag}"
  echo "    -> $DIR/cryptopp_${tag}"

  if [ -f compile_commands.json ]; then
    emit_ir_ccjson "$DIR" "$DIR/ir_${tag}"
  else
    echo "    (no compile_commands.json — using flat .cpp IR fallback)"
    emit_ir_fallback_cpp "$cxxflags" "$DIR/ir_${tag}"
  fi
}

echo "[2/3] O0 ..."
build_one "o0" "$MD_CXXFLAGS"

echo "[3/3] O1 / O2 / O3 ..."
build_one "o1" "$O1_CXXFLAGS"
build_one "o2" "$O2_CXXFLAGS"
build_one "o3" "$O3_CXXFLAGS"

echo "Done."
echo "  ELF: $DIR/cryptopp_o0 .. cryptopp_o3"
echo "  IR:  $DIR/ir_o0 .. ir_o3"

#!/usr/bin/env bash
# Botan: O0–O3 relocatable ELF (merged .o from static lib) + per-TU IR via compile_commands.json
#
# Requires: clang++, python3
# For compile_commands.json (IR): either `bear` OR `compiledb` (pip install compiledb)
# Usage: compile_botan.sh [base_dir]
# Env: BOTAN_VERSION=3.6.1

set -eu

BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/botan}"
BOTAN_VERSION="${BOTAN_VERSION:-3.6.1}"
TARBALL_NAME="Botan-${BOTAN_VERSION}"
ARCHIVE="${TARBALL_NAME}.tar.xz"
DOWNLOAD_URL="https://botan.randombit.net/releases/${ARCHIVE}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${BASE_DIR}/${TARBALL_NAME}"

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

MD_CXXFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CXXFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CXXFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CXXFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -d "$DIR" ]; then
  echo "[1/3] Download Botan ${BOTAN_VERSION} ..."
  if [ ! -f "/tmp/${ARCHIVE}" ]; then
    wget -q -O "/tmp/${ARCHIVE}" "$DOWNLOAD_URL" || curl -sL -o "/tmp/${ARCHIVE}" "$DOWNLOAD_URL"
  fi
  tar -xJf "/tmp/${ARCHIVE}" -C "$BASE_DIR"
fi

cd "$DIR"

# Need compile_commands.json: system `bear`, or `compiledb` / `python3 -m compiledb`
run_make_with_compile_commands() {
  if command -v bear >/dev/null 2>&1; then
    echo "  (bear) recording compile_commands.json ..."
    bear -- make -j"$NPROC"
    return 0
  fi
  if command -v compiledb >/dev/null 2>&1; then
    echo "  (compiledb) recording compile_commands.json ..."
    compiledb make -j"$NPROC"
    return 0
  fi
  if python3 -m compiledb --help >/dev/null 2>&1; then
    echo "  (python3 -m compiledb) recording compile_commands.json ..."
    python3 -m compiledb make -j"$NPROC"
    return 0
  fi
  echo "ERROR: need bear OR compiledb to generate compile_commands.json for IR."
  echo "  Option A: sudo apt install bear"
  echo "  Option B: pip install compiledb   # then ensure ~/.local/bin on PATH, or use venv"
  echo "            pip install compiledb && python3 -m compiledb make -j\$(nproc)  # also works"
  return 1
}

emit_ir() {
  local build_root="$1"
  local ir_dir="$2"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$WORKDIR/scripts/emit_ir_from_compile_commands.py" "$build_root" "$ir_dir"
}

build_one() {
  local tag="$1"
  local cxxflags="$2"
  local out_elf_name="$3"

  echo "=== Botan ${tag} ==="
  make clean 2>/dev/null || true
  rm -f compile_commands.json

  ./configure.py \
    --cc-bin=clang++ \
    --os=linux \
    --disable-shared \
    --cxxflags="$cxxflags"

  run_make_with_compile_commands || exit 1

  local ARCHIVE
  ARCHIVE=$(ls -1 libbotan*.a 2>/dev/null | head -1)
  if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: libbotan*.a not found after build"
    exit 1
  fi
  echo "    static lib: $ARCHIVE"

  merge_static_to_reloc "$(pwd)/$ARCHIVE" "$DIR/botan_${out_elf_name}"
  echo "    -> $DIR/botan_${out_elf_name}"

  if [ -f compile_commands.json ]; then
    emit_ir "$DIR" "$DIR/ir_${tag}"
  else
    echo "ERROR: compile_commands.json missing (bear/compiledb did not produce it?)"
    exit 1
  fi
}

echo "[2/3] O0 ..."
build_one "o0" "$MD_CXXFLAGS" "o0"

echo "[3/3] O1 / O2 / O3 ..."
build_one "o1" "$O1_CXXFLAGS" "o1"
build_one "o2" "$O2_CXXFLAGS" "o2"
build_one "o3" "$O3_CXXFLAGS" "o3"

echo "Done."
echo "  ELF: $DIR/botan_o0 .. botan_o3"
echo "  IR:  $DIR/ir_o0 .. ir_o3"

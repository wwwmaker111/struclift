#!/usr/bin/env bash
# BoringSSL: O0–O3 ELF (bssl) + per-TU LLVM IR via compile_commands.json
# NOTE: 默认一键流水线已不再包含本项目；需要时请单独运行本脚本。
# Same obtain pattern as zlib/coreutils: wget|curl + tar.gz (no git).
#
# Requires: cmake, ninja, clang/clang++, python3, tar
# Usage: compile_boringssl.sh [datasets_parent]
#   Source ends up at: <datasets_parent>/boringssl
#
# Env:
#   BORINGSSL_DIR=/abs/path           — override destination directory
#   BORINGSSL_DOWNLOAD_URL=...        — full URL to source .tar.gz (optional)
#   BORINGSSL_BRANCH=main             — GitHub branch for archive (default: main; avoid stale master)
#   BORINGSSL_GHPROXY_PREFIX=...      — prepend to GitHub URL (optional)

set -eu

BASE_DIR="${1:-${HOME}/struclift_wsl/datasets}"
DIR="${BORINGSSL_DIR:-${BASE_DIR}/boringssl}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

_check_tools() {
  local missing=""
  for c in cmake ninja clang clang++ python3; do
    command -v "$c" >/dev/null 2>&1 || missing="$missing $c"
  done
  if [ -n "$missing" ]; then
    echo "ERROR: missing command(s):$missing"
    echo "  Debian/Ubuntu: sudo apt update && sudo apt install -y cmake ninja-build clang python3"
    echo "  (ninja is required for: cmake -G Ninja)"
    exit 1
  fi
}
_check_tools

# Upstream requires these; stale/incomplete "master" archives often lack them.
_boringssl_tree_ok() {
  [ -f "$DIR/CMakeLists.txt" ] && [ -f "$DIR/cmake/go.cmake" ] && [ -f "$DIR/gen/sources.cmake" ]
}

download_and_extract() {
  local url="$1"
  local tb="$2"
  echo "  URL: $url"
  wget -q -O "$tb" "$url" || curl -fsSL -o "$tb" "$url" || return 1
  rm -rf "$DIR"
  tar -xzf "$tb" -C "$BASE_DIR" || return 1
  local extracted
  extracted=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -name 'boringssl-*' | head -1)
  if [ -z "$extracted" ] || [ ! -d "$extracted" ]; then
    echo "ERROR: expected boringssl-* under $BASE_DIR after extract"
    return 1
  fi
  mv "$extracted" "$DIR"
  if ! _boringssl_tree_ok; then
    echo "ERROR: archive missing cmake/go.cmake or gen/sources.cmake (wrong branch or corrupt). Remove $TB and retry."
    rm -rf "$DIR"
    return 1
  fi
  echo "  -> $DIR"
  return 0
}

mkdir -p "$BASE_DIR"

if _boringssl_tree_ok; then
  echo "[1/4] using existing BoringSSL: $DIR"
else
  if [ -d "$DIR" ]; then
    echo "WARN: $DIR exists but is incomplete (Google moved default branch to main; old master zips may lack cmake/gen). Removing."
    rm -rf "$DIR"
  fi
  echo "[1/4] download + extract BoringSSL -> $DIR"
  BR="${BORINGSSL_BRANCH:-main}"
  OK=0
  if [ -n "${BORINGSSL_DOWNLOAD_URL:-}" ]; then
    U="${BORINGSSL_GHPROXY_PREFIX:-}${BORINGSSL_DOWNLOAD_URL}"
    TB="/tmp/boringssl-custom.tar.gz"
    download_and_extract "$U" "$TB" && OK=1 || true
  else
    # Prefer user's branch first, then main, then master (master may be stale on GitHub).
    BR_ORDER=("$BR" main master)
    for br in "${BR_ORDER[@]}"; do
      case "$br" in main|master) ;; *) continue ;; esac
      TB="/tmp/boringssl-${br}.tar.gz"
      U="https://github.com/google/boringssl/archive/refs/heads/${br}.tar.gz"
      U="${BORINGSSL_GHPROXY_PREFIX:-}${U}"
      if [ -f "$TB" ]; then
        echo "  reuse cached $TB"
        if rm -rf "$DIR" && tar -xzf "$TB" -C "$BASE_DIR"; then
          extracted=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -name 'boringssl-*' | head -1)
          if [ -n "$extracted" ]; then
            mv "$extracted" "$DIR"
            if _boringssl_tree_ok; then
              echo "  -> $DIR"
              OK=1
              break
            fi
            echo "  cached $TB is incomplete; delete it with: rm -f $TB"
            rm -rf "$DIR"
          fi
        fi
      fi
      if download_and_extract "$U" "$TB"; then
        OK=1
        break
      fi
    done
  fi
  if [ "$OK" != 1 ]; then
    echo "ERROR: could not download/extract a complete BoringSSL tree."
    echo "  rm -f /tmp/boringssl-master.tar.gz   # if you cached a stale master archive"
    echo "  Or: export BORINGSSL_DOWNLOAD_URL=<full URL to main.tar.gz>"
    exit 1
  fi
fi

emit_ir() {
  local build_dir="$1"
  local ir_dir="$2"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$WORKDIR/scripts/emit_ir_from_compile_commands.py" "$build_dir" "$ir_dir"
}

build_level() {
  local tag="$1"
  local flags="$2"
  local out_elf="$3"
  local BDIR="$DIR/build_${tag}"

  echo "=== BoringSSL ${tag} (flags: ${flags:0:48}...) ==="
  rm -rf "$BDIR"
  cmake -S "$DIR" -B "$BDIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="$flags" \
    -DCMAKE_CXX_FLAGS="$flags" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_SHARED_LIBS=OFF

  cmake --build "$BDIR" -j"$NPROC" --target bssl

  if [ -f "$BDIR/bssl" ]; then
    cp -f "$BDIR/bssl" "$DIR/$out_elf"
    echo "    -> $DIR/$out_elf"
  else
    echo "ERROR: missing $BDIR/bssl"
    exit 1
  fi

  emit_ir "$BDIR" "$DIR/ir_${tag}"
}

echo "[2/4] O0 ..."
build_level "o0" "$MD_CFLAGS" "boringssl_o0"

echo "[3/4] O1 ..."
build_level "o1" "$O1_CFLAGS" "boringssl_o1"

echo "[4/4] O2 / O3 ..."
build_level "o2" "$O2_CFLAGS" "boringssl_o2"
build_level "o3" "$O3_CFLAGS" "boringssl_o3"

echo "Done."
echo "  ELF:  $DIR/boringssl_o0 .. boringssl_o3"
echo "  IR:   $DIR/ir_o0 .. $DIR/ir_o3"
echo "  Src:  $DIR (--src for build_binskel)"

#!/usr/bin/env bash
# mbedtls: O0–O3 合并静态库为单个可重定位 ELF + compile_commands → ir_oN/
# Usage: compile_mbedtls.sh [base_dir]
# Env: MBEDTLS_VERSION=3.6.2
#
# 依赖: git, cmake, ninja, clang, python3, ar
#
# 注意：mbedtls 3.6+ 依赖子模块 framework/；GitHub「Source code」tar.gz 不含子模块，
# 会导致 CMake 报 framework/CMakeLists.txt not found。必须用带 submodule 的源码。

set -eu
VER="${MBEDTLS_VERSION:-3.6.2}"
BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/mbedtls}"
DIR="${BASE_DIR}/mbedtls-${VER}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
GIT_URL="${MBEDTLS_GIT_URL:-https://github.com/Mbed-TLS/mbedtls.git}"
GIT_TAG="${MBEDTLS_GIT_TAG:-v${VER}}"
# 部分 WSL/发行版用 GnuTLS 编译 git，但全局配置了 sslBackend=openssl，会报：
#   Unsupported SSL backend 'openssl'. Supported: gnutls
GIT_SSL_BACKEND="${MBEDTLS_GIT_SSL_BACKEND:-gnutls}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)

for c in git cmake ninja clang python3 ar; do
  command -v "$c" >/dev/null 2>&1 || { echo "ERROR: need $c in PATH"; exit 1; }
done

mkdir -p "$BASE_DIR"

_fetch_git() {
  echo "[mbedtls] git clone --depth 1 --recurse-submodules $GIT_TAG -> $DIR (sslBackend=$GIT_SSL_BACKEND)"
  rm -rf "$DIR"
  if ! git -c "http.sslBackend=${GIT_SSL_BACKEND}" clone --depth 1 --branch "$GIT_TAG" \
    --recurse-submodules "$GIT_URL" "$DIR"; then
    echo "ERROR: git clone 失败。若仍报 SSL backend，可执行:" >&2
    echo "  git config --global http.sslBackend gnutls" >&2
    echo "或设环境变量: MBEDTLS_GIT_SSL_BACKEND=gnutls（默认已是 gnutls）" >&2
    exit 1
  fi
}

if [ -d "$DIR" ] && [ ! -f "$DIR/framework/CMakeLists.txt" ]; then
  echo "WARN: $DIR 缺少 framework/（常见于仅用 tar.gz 解压），将删除后改用 git clone"
  rm -rf "$DIR"
fi

if [ ! -d "$DIR" ]; then
  _fetch_git
fi

if [ ! -f "$DIR/framework/CMakeLists.txt" ]; then
  echo "ERROR: 仍缺少 $DIR/framework/CMakeLists.txt（检查网络或 MBEDTLS_GIT_URL / MBEDTLS_GIT_TAG）"
  exit 1
fi

emit_ir() {
  local build_dir="$1"
  local ir_dir="$2"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$WORKDIR/scripts/emit_ir_from_compile_commands.py" "$build_dir" "$ir_dir"
}

merge_mbedtls_libs() {
  local bdir="$1"
  local out_elf="$2"
  local tmp rc
  tmp=$(mktemp -d)
  # 勿用 trap RETURN + $tmp：与 set -u 组合时，RETURN 触发时 local tmp 可能已失效 → unbound variable
  (
    cd "$tmp" || exit 1
    for a in "$bdir/library/libmbedcrypto.a" "$bdir/library/libmbedx509.a" "$bdir/library/libmbedtls.a"; do
      [ -f "$a" ] || { echo "ERROR: missing $a"; exit 1; }
      sub=$(mktemp -d)
      (cd "$sub" && ar x "$a")
      shopt -s nullglob
      for o in "$sub"/*.o; do mv -f "$o" .; done
      shopt -u nullglob
      rmdir "$sub" 2>/dev/null || true
    done
    shopt -s nullglob
    objs=(./*.o)
    shopt -u nullglob
    [ "${#objs[@]}" -gt 0 ] || { echo "ERROR: no .o from static libs"; exit 1; }
    clang -nostdlib -r "${objs[@]}" -o "$out_elf"
  )
  rc=$?
  rm -rf "$tmp"
  return "$rc"
}

build_one() {
  local tag="$1"
  local flags="$2"
  local out_elf="$3"
  local BDIR="$DIR/build_${tag}"

  echo "=== mbedtls $tag ==="
  rm -rf "$BDIR"
  cmake -S "$DIR" -B "$BDIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_C_FLAGS="$flags" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_TESTING=OFF \
    -DENABLE_PROGRAMS=OFF

  cmake --build "$BDIR" -j"$NPROC" --target mbedcrypto mbedx509 mbedtls

  merge_mbedtls_libs "$BDIR" "$DIR/$out_elf"
  echo "    -> $DIR/$out_elf"

  emit_ir "$BDIR" "$DIR/ir_${tag}"
}

echo "[mbedtls] O0..."
build_one "o0" "$MD_CFLAGS" "mbedtls_o0"
echo "[mbedtls] O1..."
build_one "o1" "$O1_CFLAGS" "mbedtls_o1"
echo "[mbedtls] O2..."
build_one "o2" "$O2_CFLAGS" "mbedtls_o2"
echo "[mbedtls] O3..."
build_one "o3" "$O3_CFLAGS" "mbedtls_o3"
echo "Done: $DIR/mbedtls_o0..o3, ir_o0..ir_o3"

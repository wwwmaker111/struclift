#!/usr/bin/env bash
# Lua：下载、O0–O3 四档编译，产出与 zlib/OpenSSL 一致形态，供完整 binskel（DWARF + ir_oN/ + tree-sitter）。
# 产出（在 lua-${VER}/ 目录下）:
#   lua_o0 … lua_o3  — 解释器可执行文件（ELF + DWARF）
#   ir_o0 … ir_o3/   — src 下各 .c 对应的 .ll
#
# 用法: bash scripts/compile_lua.sh [base_dir]
# 例:   bash scripts/compile_lua.sh ~/struclift_wsl/datasets/lua
# 依赖: clang、llvm（发 IR）、make、wget 或 curl

set -euo pipefail

LUA_VERSION="${LUA_VERSION:-5.4.7}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/lua}"
DOWNLOAD_URL="https://www.lua.org/ftp/lua-${LUA_VERSION}.tar.gz"
TARBALL="/tmp/lua-${LUA_VERSION}.tar.gz"
DIR="${BASE_DIR}/lua-${LUA_VERSION}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "[1/4] 下载 lua-${LUA_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -fsSL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/4] 解压 ..."
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local flags="$1"
  local tag="$2"
  local ir_dir="$DIR/ir_${tag}"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  local n_ok=0 n_fail=0
  local c bn
  for c in "$DIR"/src/*.c; do
    [ -f "$c" ] || continue
    bn=$(basename "${c%.c}.ll")
    if clang $flags -I"$DIR/src" -emit-llvm -S "$c" -o "$ir_dir/$bn" 2>/dev/null; then
      n_ok=$((n_ok + 1))
    else
      n_fail=$((n_fail + 1))
      echo "    IR fail: $c"
    fi
  done
  echo "    -> $ir_dir ($n_ok ok, $n_fail fail)"
}

build_tier() {
  local tag="$1"
  local flags="$2"

  echo "  === lua ${tag} ==="
  make clean >/dev/null 2>&1 || true
  # generic：官方发布包顶层目标；必须清空 MYOPTS（src/Makefile 默认 MYOPTS=-O2，会盖过 MYCFLAGS 里的 -O0 等）
  make -j"$NPROC" generic \
    CC=clang \
    MYOPTS= \
    MYCFLAGS="$flags" \
    MYLDFLAGS="-g"
  cp -f "$DIR/src/lua" "$DIR/lua_${tag}"
  echo "    -> $DIR/lua_${tag}"
  emit_ir "$flags" "$tag"
}

echo "[3/4] O0 / O1 / O2 / O3 ..."
build_tier "o0" "$MD_CFLAGS"
build_tier "o1" "$O1_CFLAGS"
build_tier "o2" "$O2_CFLAGS"
build_tier "o3" "$O3_CFLAGS"

echo "[4/4] 完成"
echo "  ELF: $DIR/lua_o0 … lua_o3"
echo "  IR:  $DIR/ir_o0/ … ir_o3/"
echo "  源码根: $DIR （--src 指向此目录）"
if command -v readelf >/dev/null 2>&1; then
  echo "  DWARF 自检 (.debug_line):"
  for b in lua_o0 lua_o1 lua_o2 lua_o3; do
    [ -f "$DIR/$b" ] || continue
    if readelf -S "$DIR/$b" 2>/dev/null | grep -qE '\.debug_line'; then
      echo "    OK $b"
    else
      echo "    FAIL $b 缺少 .debug_line"
    fi
  done
fi

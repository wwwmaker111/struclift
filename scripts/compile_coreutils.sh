#!/bin/bash
# 下载、编译 coreutils (single-binary)：O0-O3 + IR，产出 coreutils_o0..o3 及 ir_oN/
# 用法: bash scripts/compile_coreutils.sh [base_dir]
# 与 OpenSSL 一致：使用逐文件 .ll 目录，避免 llvm-link 重命名
# 若 base_dir 在 /mnt/* (WSL 挂载 Windows)，clang/gcc 可能无法创建可执行文件，会复制到 /tmp 构建
set -e
COREUTILS_VERSION="${COREUTILS_VERSION:-9.4}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/coreutils}"
DOWNLOAD_URL="https://ftp.gnu.org/gnu/coreutils/coreutils-${COREUTILS_VERSION}.tar.xz"
TARBALL="/tmp/coreutils-${COREUTILS_VERSION}.tar.xz"
DIR="${BASE_DIR}/coreutils-${COREUTILS_VERSION}"
# WSL 在 /mnt/* 上编译常失败，可复制到 /tmp 构建；设 USE_TMP_BUILD=0 可禁用
if [[ "$(realpath "$BASE_DIR" 2>/dev/null || echo "$BASE_DIR")" == /mnt/* ]] && [[ "${USE_TMP_BUILD:-1}" != 0 ]]; then
  BUILD_DIR="/tmp/structlift_coreutils_build"
  USE_TMP_BUILD=1
else
  BUILD_DIR="$DIR"
  USE_TMP_BUILD=0
fi

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)
# 与 make V=1 一致
CU_INC="-I. -I./lib -Ilib -I./lib -Isrc -I./src"
CU_LIB_FLAGS="-Wno-cast-qual -Wno-conversion -Wno-float-equal -Wno-sign-compare -Wno-undef -Wno-unused-function -Wno-unused-parameter -Wno-float-conversion -Wimplicit-fallthrough -Wno-pedantic -Wno-sign-conversion -Wno-type-limits"

mkdir -p "$BASE_DIR"
if [ ! -f "$TARBALL" ]; then
  echo "[1/5] 下载 coreutils-${COREUTILS_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -sL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ "$USE_TMP_BUILD" = 1 ]; then
  echo "[1b/5] 复制到 /tmp 构建（避免 /mnt 上 GCC 无法创建可执行文件）..."
  rm -rf "$BUILD_DIR"
  mkdir -p "$(dirname "$BUILD_DIR")"
  tar -xJf "$TARBALL" -C "$(dirname "$BUILD_DIR")"
  WORK_DIR="$(dirname "$BUILD_DIR")/coreutils-${COREUTILS_VERSION}"
  [ -d "$WORK_DIR" ] || { echo "解压失败"; exit 1; }
else
  if [ ! -d "$DIR" ]; then
    echo "[2/5] 解压 ..."
    tar -xJf "$TARBALL" -C "$BASE_DIR"
  fi
  WORK_DIR="$DIR"
fi
cd "$WORK_DIR"

# 若之前 configure 失败，清理残留
make distclean 2>/dev/null || true

# .o -> .c 映射：lib/libcoreutils_a-foo.o -> lib/foo.c, src/coreutils-foo.o -> src/foo.c
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
  local opt="$1"
  local cflags="$2"
  local irdir="$3"
  rm -rf "$irdir"
  mkdir -p "$irdir"
  find lib src -name "*.o" -not -path "*/.deps/*" 2>/dev/null | sort > /tmp/cu_ofiles.txt
  n_ok=0
  n_fail=0
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
    else
      n_fail=$((n_fail+1))
    fi
  done < /tmp/cu_ofiles.txt
  rm -f /tmp/cu_ofiles.txt
  echo "    逐文件 .ll: 成功 $n_ok, 失败 $n_fail"
  echo "    -> $irdir/ ($(ls "$irdir"/*.ll 2>/dev/null | wc -l) 个 .ll)"
}

# 使用 clang；gnulib 的 undeclared builtin 检测需绕过
CC_BUILD="${CC_BUILD:-clang}"
# 绕过 gnulib "cannot make clang report undeclared builtins"（eggert 建议）
export gl_cv_cc_undeclared_builtin_works=no
echo "[2b/5] 配置 (single-binary, CC=$CC_BUILD) ..."
./configure --enable-single-binary CC="$CC_BUILD" 2>&1 | tail -3

echo "[3/5] O0 (按 MD) ..."
make clean 2>/dev/null || true
./configure --enable-single-binary CC="$CC_BUILD" CFLAGS="$MD_CFLAGS" 2>&1 | tail -2
make -j"$NPROC" 2>&1 | tail -3
BIN=$(find . -name "coreutils" -type f -path "*/src/*" 2>/dev/null | head -1)
[ -z "$BIN" ] && BIN=$(find . -name "coreutils" -type f 2>/dev/null | head -1)
mkdir -p "$DIR"
[ -n "$BIN" ] && cp -f "$BIN" "$DIR/coreutils_o0" && echo "    -> $DIR/coreutils_o0"
echo "    [.ll] 生成 ir_o0/ ..."
emit_ir_dir "0" "$MD_CFLAGS" "$DIR/ir_o0"

echo "[4/5] O1 / O2 / O3 (ELF + ir_oN/) ..."
for opt in 1 2 3; do
  case $opt in 1) F="$O1_CFLAGS" ;; 2) F="$O2_CFLAGS" ;; 3) F="$O3_CFLAGS" ;; esac
  make clean 2>/dev/null || true
  ./configure --enable-single-binary CC="$CC_BUILD" CFLAGS="$F" 2>&1 | tail -2
  make -j"$NPROC" 2>&1 | tail -3
  BIN=$(find . -name "coreutils" -type f -path "*/src/*" 2>/dev/null | head -1)
  [ -z "$BIN" ] && BIN=$(find . -name "coreutils" -type f 2>/dev/null | head -1)
  [ -n "$BIN" ] && cp -f "$BIN" "$DIR/coreutils_o$opt" && echo "    -> $DIR/coreutils_o$opt"
  echo "    [.ll] 生成 ir_o$opt/ ..."
  emit_ir_dir "$opt" "$F" "$DIR/ir_o$opt"
done

if [ "$USE_TMP_BUILD" = 1 ]; then
  rm -rf "$(dirname "$BUILD_DIR")/coreutils-${COREUTILS_VERSION}"
fi
echo "[5/5] 完成"
echo "  O0-O3: $DIR/coreutils_o0 .. coreutils_o3"
echo "  IR:    $DIR/ir_o0/ .. ir_o3/"
echo "  源码:  $DIR"

#!/usr/bin/env bash
# GNU coreutils：多产物（每个工具独立 ELF），与 compile_coreutils.sh 的单二进制版并行存在。
# 产出（在 $DIR 下）:
#   multibin_o0..multibin_o3、multibin_os — 各档位下 src/ 内所有 ELF
#   ir_o0 … ir_o3、ir_os               — 由 lib+src 各 .c 生成 .ll 目录
# 共 5 档：O0、O1、O2、O3、Os（与全项目 binskel 约定一致）
#
# 用法:
#   bash scripts/compile_coreutils_multibin.sh [base_dir]
#   # 若在 /mnt 上编译失败，可改用本地盘:
#   USE_TMP_BUILD=0 bash scripts/compile_coreutils_multibin.sh ~/datasets/coreutils
#
set -euo pipefail
STRIP=:
export STRIP
COREUTILS_VERSION="${COREUTILS_VERSION:-9.4}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/coreutils}"
DOWNLOAD_URL="https://ftp.gnu.org/gnu/coreutils/coreutils-${COREUTILS_VERSION}.tar.xz"
TARBALL="/tmp/coreutils-${COREUTILS_VERSION}.tar.xz"
DIR="${BASE_DIR}/coreutils-${COREUTILS_VERSION}"

if [[ "$(realpath "$BASE_DIR" 2>/dev/null || echo "$BASE_DIR")" == /mnt/* ]] && [[ "${USE_TMP_BUILD:-1}" != 0 ]]; then
  echo "提示: 在 /mnt 上多产物编译易失败；建议: USE_TMP_BUILD=0 且 base_dir 在 \$HOME 下，或挂载策略允许创建可执行文件。"
fi

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"
OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)
CU_INC="-I. -I./lib -Ilib -I./lib -Isrc -I./src"
CU_LIB_FLAGS="-Wno-cast-qual -Wno-conversion -Wno-float-equal -Wno-sign-compare -Wno-undef -Wno-unused-function -Wno-unused-parameter -Wno-float-conversion -Wimplicit-fallthrough -Wno-pedantic -Wno-sign-conversion -Wno-type-limits"

mkdir -p "$BASE_DIR"
if [ ! -f "$TARBALL" ]; then
  echo "[下载] coreutils-${COREUTILS_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -sL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[解压] -> $DIR"
  tar -xJf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

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
  find lib src -name "*.o" -not -path "*/.deps/*" 2>/dev/null | sort > /tmp/cu_ofiles_multibin.txt
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
    if clang $cflags $CU_INC $extra -I"$cdir" -emit-llvm -S "$cfile" -o "$irdir/$bn" 2>/dev/null; then
      n_ok=$((n_ok+1))
    else
      n_fail=$((n_fail+1))
    fi
  done < /tmp/cu_ofiles_multibin.txt
  rm -f /tmp/cu_ofiles_multibin.txt
  echo "    逐文件 .ll: 成功 $n_ok, 失败 $n_fail -> $irdir"
}

copy_src_elfs() {
  local dest="$1"
  rm -rf "$dest"
  mkdir -p "$dest"
  local n=0
  for f in src/*; do
    [ -f "$f" ] || continue
    [ -x "$f" ] || continue
    if ! file "$f" 2>/dev/null | grep -q ELF; then
      continue
    fi
    cp -f "$f" "$dest/$(basename "$f")"
    n=$((n+1))
  done
  echo "    已收集 ELF: $n 个 -> $dest"
}

CC_BUILD="${CC_BUILD:-clang}"
export gl_cv_cc_undeclared_builtin_works=no

echo "[多产物] 不使用 --enable-single-binary；CC=$CC_BUILD"
make distclean 2>/dev/null || true

build_one() {
  local tag="$1"
  local cflags="$2"
  echo ""
  echo "===== O$tag ====="
  make clean 2>/dev/null || true
  ./configure CC="$CC_BUILD" CFLAGS="$cflags" 2>&1 | tail -3
  make -j"$NPROC" STRIP="$STRIP" 2>&1 | tail -5
  copy_src_elfs "$DIR/multibin_o$tag"
  emit_ir_dir "$cflags" "$DIR/ir_o$tag"
}

build_one 0 "$MD_CFLAGS"
build_one 1 "$O1_CFLAGS"
build_one 2 "$O2_CFLAGS"
build_one 3 "$O3_CFLAGS"

echo ""
echo "===== Os ====="
make clean 2>/dev/null || true
./configure CC="$CC_BUILD" CFLAGS="$OS_CFLAGS" 2>&1 | tail -3
make -j"$NPROC" STRIP="$STRIP" 2>&1 | tail -5
copy_src_elfs "$DIR/multibin_os"
emit_ir_dir "$OS_CFLAGS" "$DIR/ir_os"

echo ""
echo "[完成] 源码树: $DIR"
echo "  ELF: $DIR/multibin_o{0..3}/ + $DIR/multibin_os/"
echo "  IR:  $DIR/ir_o{0..3}/ + $DIR/ir_os/"

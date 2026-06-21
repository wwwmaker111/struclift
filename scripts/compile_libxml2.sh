#!/bin/bash
# 下载、编译 libxml2 (xmllint)：O0-O3 + IR，产出 xmllint_o0..o3 及 ir_oN/
# 用法: bash scripts/compile_libxml2.sh [base_dir]
# 与 OpenSSL 一致：使用逐文件 .ll 目录
# 若 base_dir 在 /mnt/* (WSL 挂载)，会复制到 /tmp 构建
set -e
LIBXML2_VERSION="${LIBXML2_VERSION:-2.12.7}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/libxml2}"
DOWNLOAD_URL="https://download.gnome.org/sources/libxml2/2.12/libxml2-${LIBXML2_VERSION}.tar.xz"
TARBALL="/tmp/libxml2-${LIBXML2_VERSION}.tar.xz"
DIR="${BASE_DIR}/libxml2-${LIBXML2_VERSION}"
if [[ "$(realpath "$BASE_DIR" 2>/dev/null || echo "$BASE_DIR")" == /mnt/* ]] && [[ "${USE_TMP_BUILD:-1}" != 0 ]]; then
  BUILD_PARENT="/tmp"
  USE_TMP_BUILD=1
else
  USE_TMP_BUILD=0
fi

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)
XML_INC="-I. -Iinclude -Iinclude/libxml"

mkdir -p "$BASE_DIR"
if [ ! -f "$TARBALL" ]; then
  echo "[1/5] 下载 libxml2-${LIBXML2_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -sL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ "$USE_TMP_BUILD" = 1 ]; then
  echo "[1b/5] 复制到 /tmp 构建（避免 /mnt 上编译失败）..."
  rm -rf "$BUILD_PARENT/libxml2-${LIBXML2_VERSION}"
  tar -xJf "$TARBALL" -C "$BUILD_PARENT"
  WORK_DIR="$BUILD_PARENT/libxml2-${LIBXML2_VERSION}"
else
  if [ ! -d "$DIR" ]; then
    echo "[2/5] 解压 ..."
    tar -xJf "$TARBALL" -C "$BASE_DIR"
  fi
  WORK_DIR="$DIR"
fi
mkdir -p "$DIR"
cd "$WORK_DIR"

# .o -> .c：支持 .libs/ 与直接 .o
get_cfile() {
  local ofile="$1"
  local cfile=$(echo "$ofile" | sed 's|/.libs/|/|g; s|\.o$|.c|')
  echo "$cfile"
}

emit_ir_dir() {
  local opt="$1"
  local cflags="$2"
  local irdir="$3"
  rm -rf "$irdir"
  mkdir -p "$irdir"
  find . -name "*.o" -not -path "./.bc_o*/*" -not -path "*/.deps/*" 2>/dev/null | sort > /tmp/xml_ofiles.txt
  n_ok=0
  n_fail=0
  while IFS= read -r ofile; do
    cfile=$(get_cfile "$ofile")
    [ -f "$cfile" ] || continue
    bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
    bn="${bn%.c}.ll"
    cdir=$(dirname "$cfile")
    if clang $cflags $XML_INC -I"$cdir" -emit-llvm -S "$cfile" -o "$irdir/$bn" 2>/dev/null; then
      n_ok=$((n_ok+1))
    else
      n_fail=$((n_fail+1))
    fi
  done < /tmp/xml_ofiles.txt
  rm -f /tmp/xml_ofiles.txt
  echo "    逐文件 .ll: 成功 $n_ok, 失败 $n_fail"
  echo "    -> $irdir/ ($(ls "$irdir"/*.ll 2>/dev/null | wc -l) 个 .ll)"
}

echo "[2b/5] 配置 ..."
./configure CC=clang --without-python --without-readline 2>&1 | tail -3

echo "[3/5] O0 (按 MD) ..."
make clean 2>/dev/null || true
./configure CC=clang CFLAGS="$MD_CFLAGS" --without-python --without-readline 2>&1 | tail -2
make -j"$NPROC" 2>&1 | tail -3
# libtool 时真实 ELF 在 .libs/xmllint，根目录 xmllint 是包装脚本
XMLLINT_BIN=""
[ -f .libs/xmllint ] && XMLLINT_BIN=".libs/xmllint"
[ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && file xmllint 2>/dev/null | grep -q ELF && XMLLINT_BIN="xmllint"
[ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && XMLLINT_BIN="xmllint"
[ -n "$XMLLINT_BIN" ] && cp -f "$XMLLINT_BIN" "$DIR/xmllint_o0" && echo "    -> $DIR/xmllint_o0"
echo "    [.ll] 生成 ir_o0/ ..."
emit_ir_dir "0" "$MD_CFLAGS" "$DIR/ir_o0"

echo "[4/5] O1 / O2 / O3 (ELF + ir_oN/) ..."
for opt in 1 2 3; do
  case $opt in 1) F="$O1_CFLAGS" ;; 2) F="$O2_CFLAGS" ;; 3) F="$O3_CFLAGS" ;; esac
  make clean 2>/dev/null || true
  ./configure CC=clang CFLAGS="$F" --without-python --without-readline 2>&1 | tail -2
  make -j"$NPROC" 2>&1 | tail -3
  XMLLINT_BIN=""
  [ -f .libs/xmllint ] && XMLLINT_BIN=".libs/xmllint"
  [ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && file xmllint 2>/dev/null | grep -q ELF && XMLLINT_BIN="xmllint"
  [ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && XMLLINT_BIN="xmllint"
  [ -n "$XMLLINT_BIN" ] && cp -f "$XMLLINT_BIN" "$DIR/xmllint_o$opt" && echo "    -> $DIR/xmllint_o$opt"
  echo "    [.ll] 生成 ir_o$opt/ ..."
  emit_ir_dir "$opt" "$F" "$DIR/ir_o$opt"
done

if [ "$USE_TMP_BUILD" = 1 ]; then
  rm -rf "$WORK_DIR"
fi
echo "[5/5] 完成"
echo "  O0-O3: $DIR/xmllint_o0 .. xmllint_o3"
echo "  IR:    $DIR/ir_o0/ .. ir_o3/"
echo "  源码:  $DIR"

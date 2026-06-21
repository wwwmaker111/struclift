#!/bin/sh
# 自包含：复制到 ~ 后执行，避免从 /mnt 读取导致 I/O 错误
# 用法: cp /mnt/e/structlift/scripts/run_libxml2_from_home.sh ~/ && chmod +x ~/run_libxml2_from_home.sh && ~/run_libxml2_from_home.sh
# 输出目录默认 /mnt/e/structlift_src_datasets/libxml2，可传参覆盖
set -e
BASE_DIR="${1:-/mnt/e/structlift_src_datasets/libxml2}"
LIBXML2_VERSION="${LIBXML2_VERSION:-2.12.7}"
DOWNLOAD_URL="https://download.gnome.org/sources/libxml2/2.12/libxml2-${LIBXML2_VERSION}.tar.xz"
TARBALL="/tmp/libxml2-${LIBXML2_VERSION}.tar.xz"
DIR="${BASE_DIR}/libxml2-${LIBXML2_VERSION}"
BUILD_PARENT="/tmp"
WORK_DIR="$BUILD_PARENT/libxml2-${LIBXML2_VERSION}"

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
echo "[1b/5] 在 /tmp 构建 ..."
rm -rf "$WORK_DIR"
tar -xJf "$TARBALL" -C "$BUILD_PARENT"
mkdir -p "$DIR"
cd "$WORK_DIR"

get_cfile() {
  echo "$1" | sed 's|/.libs/|/|g; s|\.o$|.c|'
}

emit_ir_dir() {
  opt="$1"
  cflags="$2"
  irdir="$3"
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
XMLLINT_BIN=""
[ -f .libs/xmllint ] && XMLLINT_BIN=".libs/xmllint"
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
  [ -z "$XMLLINT_BIN" ] && [ -f xmllint ] && XMLLINT_BIN="xmllint"
  [ -n "$XMLLINT_BIN" ] && cp -f "$XMLLINT_BIN" "$DIR/xmllint_o$opt" && echo "    -> $DIR/xmllint_o$opt"
  echo "    [.ll] 生成 ir_o$opt/ ..."
  emit_ir_dir "$opt" "$F" "$DIR/ir_o$opt"
done

rm -rf "$WORK_DIR"
echo "[5/5] 完成"
echo "  O0-O3: $DIR/xmllint_o0 .. xmllint_o3"
echo "  IR:    $DIR/ir_o0/ .. ir_o3/"
echo "  源码:  $DIR"

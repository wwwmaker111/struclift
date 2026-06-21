#!/bin/bash
# 下载并编译 zlib：O0 严格按 BB_Source_Alignment_Pipeline.md，O1/O2/O3 用于 Module A 训练数据
# 产出: zlib_o0, zlib_o0.ll (MD 对齐用), zlib_o1, zlib_o2, zlib_o3 (仅 ELF，--skip-alignment 用)

set -e
ZLIB_VERSION="${ZLIB_VERSION:-1.3.1}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/zlib}"
DOWNLOAD_URL="https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz"
TARBALL="/tmp/zlib-${ZLIB_VERSION}.tar.gz"
DIR="${BASE_DIR}/zlib-${ZLIB_VERSION}"

# MD 文档第一步：O0 必须的编译选项（一字不差）
MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"

# O1/O2/O3 保留调试信息，便于行级对齐
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "[1/5] 下载 zlib-${ZLIB_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -sL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/5] 解压 ..."
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

# zlib 必须运行 configure 才会生成带 minigzip 目标的 Makefile（解压后的 Makefile 可能是占位）
echo "[2b/5] 配置 (configure) ..."
export CC=clang
./configure

# 所有参与链接的 .c（与 Makefile 一致，用于生成单一 .ll）
LIB_SRCS="adler32.c crc32.c deflate.c infback.c inffast.c inflate.c inftrees.c trees.c zutil.c compress.c uncompr.c gzclose.c gzlib.c gzread.c gzwrite.c"
MINIGZIP_SRC="test/minigzip.c"

echo "[3/5] O0 (按 MD) + 产出 IR ..."
export CC=clang
export CFLAGS="$MD_CFLAGS"
make clean 2>/dev/null || true
make minigzip
cp -f minigzip zlib_o0

# 产出 zlib_o0.ll：各 .c 编译为 .bc 再 llvm-link（所有 .bc 放在当前目录）
rm -f *.bc zlib_o0.bc 2>/dev/null || true
for c in $LIB_SRCS; do
  clang $MD_CFLAGS -emit-llvm -c -I. "$c" -o "${c%.c}.bc"
done
clang $MD_CFLAGS -emit-llvm -c -I. "$MINIGZIP_SRC" -o minigzip.bc
llvm-link *.bc -o zlib_o0.bc
llvm-dis zlib_o0.bc -o zlib_o0.ll
rm -f *.bc zlib_o0.bc 2>/dev/null || true
echo "    -> $DIR/zlib_o0, $DIR/zlib_o0.ll"

echo "[4/5] O1 / O2 / O3 (ELF + IR) ..."
for opt in 1 2 3; do
  case $opt in
    1) FLAGS="$O1_CFLAGS" ;;
    2) FLAGS="$O2_CFLAGS" ;;
    3) FLAGS="$O3_CFLAGS" ;;
  esac
  make clean 2>/dev/null || true
  make CFLAGS="$FLAGS" LDFLAGS="-g" STRIP="$STRIP" minigzip
  cp -f minigzip "zlib_o$opt"
  echo "    -> $DIR/zlib_o$opt"

  # 产出 zlib_oN.ll（与 O0 相同流程，仅优化级别不同）
  rm -f *.bc "zlib_o${opt}.bc" 2>/dev/null || true
  for c in $LIB_SRCS; do
    clang $FLAGS -emit-llvm -c -I. "$c" -o "${c%.c}.bc"
  done
  clang $FLAGS -emit-llvm -c -I. "$MINIGZIP_SRC" -o minigzip.bc
  llvm-link *.bc -o "zlib_o${opt}.bc"
  llvm-dis "zlib_o${opt}.bc" -o "zlib_o${opt}.ll"
  rm -f *.bc "zlib_o${opt}.bc" 2>/dev/null || true
  echo "    -> $DIR/zlib_o${opt}.ll"
done

echo "[5/5] 完成"
echo "  O0 (MD): $DIR/zlib_o0  $DIR/zlib_o0.ll"
echo "  O1/O2/O3: $DIR/zlib_o1 $DIR/zlib_o1.ll  $DIR/zlib_o2 $DIR/zlib_o2.ll  $DIR/zlib_o3 $DIR/zlib_o3.ll"
echo "  源码目录: $DIR (--src 指向此目录)"

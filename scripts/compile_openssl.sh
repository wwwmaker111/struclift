#!/bin/bash
# OpenSSL O0：编译二进制 + 生成逐文件 LLVM IR（避免 llvm-link 重命名静态函数）
# 产出:
#   openssl_o0          — O0 ELF（DWARF 信号源）
#   ir_o0/              — 逐 .c 文件的 .ll（IR 信号源，保留原始静态函数名）
#   openssl_o1/o2/o3    — 高优化级别 ELF（Module A 用）
#
# 用法: compile_openssl.sh [base_dir]

set -e
SSL_VERSION="${SSL_VERSION:-3.3.1}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/openssl}"
DOWNLOAD_URL="https://www.openssl.org/source/openssl-${SSL_VERSION}.tar.gz"
TARBALL="/tmp/openssl-${SSL_VERSION}.tar.gz"
DIR="${BASE_DIR}/openssl-${SSL_VERSION}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "[1/6] 下载 openssl-${SSL_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -sL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/6] 解压 ..."
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

# ═══════════════════════════════════════════════════════════
# 编译函数：Configure + make + 链接为单一 ELF
# ═══════════════════════════════════════════════════════════
compile_openssl() {
  local opt_name="$1"
  local cflags="$2"
  local out_name="$3"

  echo "    清理 ..."
  make clean 2>/dev/null || true
  make distclean 2>/dev/null || true

  echo "    Configure (CC=clang, $opt_name) ..."
  ./Configure linux-x86_64 \
    no-shared no-tests no-fips \
    --prefix="$DIR/install_${opt_name}" \
    CC=clang \
    CFLAGS="$cflags" \
    2>&1 | tail -3

  echo "    编译 ..."
  make -j"$NPROC" build_libs 2>&1 | tail -3

  if [ -f libcrypto.a ] && [ -f libssl.a ]; then
    echo "    链接为单一 ELF ..."
    local TMPDIR="$DIR/.link_${opt_name}"
    rm -rf "$TMPDIR"
    mkdir -p "$TMPDIR"
    cd "$TMPDIR"
    ar x "$DIR/libcrypto.a"
    ar x "$DIR/libssl.a"
    clang $cflags -nostdlib -r *.o -o "$DIR/$out_name" 2>/dev/null || \
    ld -r *.o -o "$DIR/$out_name" 2>/dev/null || true
    cd "$DIR"
    rm -rf "$TMPDIR"
    if [ -f "$DIR/$out_name" ]; then
      echo "    -> $DIR/$out_name ($(du -h "$DIR/$out_name" | cut -f1))"
    else
      echo "    警告: 链接失败"
    fi
  else
    echo "    警告: libcrypto.a 或 libssl.a 未找到"
  fi
}

# ═══════════════════════════════════════════════════════════
# [3/6] O0 二进制
# ═══════════════════════════════════════════════════════════
echo "[3/6] O0 (按 MD) ..."
compile_openssl "o0" "$MD_CFLAGS" "openssl_o0"

# ═══════════════════════════════════════════════════════════
# [4/6] O0 逐文件 LLVM IR
#   关键：在 Configure+make 之后、distclean 之前执行
#   这样 configuration.h 等生成头文件都存在
#   使用 -emit-llvm -S 直接生成 .ll，不经过 llvm-link
#   避免静态函数被 llvm-link 重命名（foo -> foo.1）
# ═══════════════════════════════════════════════════════════
echo "[4/6] O0 逐文件 .ll (LLVM IR) ..."
IR_DIR="$DIR/ir_o0"
rm -rf "$IR_DIR"
mkdir -p "$IR_DIR"

SSL_INC="-Iinclude -I. -Icrypto -Iproviders -Iproviders/common/include -Iproviders/implementations/include"
SSL_DEFS="-DOPENSSL_BUILDING_OPENSSL -DNDEBUG"

IR_SRC_DIRS=()
for d in crypto ssl providers engines; do
  [ -d "$d" ] && IR_SRC_DIRS+=("$d")
done

n_ok=0
n_fail=0
fail_list=""

while IFS= read -r -d '' cfile; do
  bn=$(echo "$cfile" | tr '/' '_')
  bn="${bn%.c}.ll"
  cfile_dir=$(dirname "$cfile")
  if clang $MD_CFLAGS $SSL_INC $SSL_DEFS -I"$cfile_dir" -emit-llvm -S "$cfile" -o "$IR_DIR/$bn" 2>/dev/null; then
    n_ok=$((n_ok+1))
  else
    n_fail=$((n_fail+1))
    fail_list="$fail_list $cfile"
  fi
done < <(find "${IR_SRC_DIRS[@]}" -name "*.c" -not -path "*/test/*" -not -path "*/fuzz/*" -print0 2>/dev/null)

echo "    逐文件 .ll: 成功 $n_ok, 失败 $n_fail"
echo "    -> $IR_DIR/ ($n_ok 个 .ll)"
if [ "$n_fail" -gt 0 ]; then
  echo "    失败文件(前10):"
  echo "$fail_list" | tr ' ' '\n' | head -10 | while read -r f; do
    [ -n "$f" ] && echo "      $f"
  done
fi

# 统计逐文件 IR 的函数总数
n_funcs=$(grep -r "^define " "$IR_DIR/" 2>/dev/null | wc -l)
echo "    逐文件 IR 函数总数: $n_funcs"

# ═══════════════════════════════════════════════════════════
# [5/6] O1 / O2 / O3 (ELF + 逐文件 IR)
# ═══════════════════════════════════════════════════════════
echo "[5/6] O1 / O2 / O3 ..."
for opt in 1 2 3; do
  case $opt in
    1) F="$O1_CFLAGS"; N="o1" ;;
    2) F="$O2_CFLAGS"; N="o2" ;;
    3) F="$O3_CFLAGS"; N="o3" ;;
  esac
  echo "  === $N ==="
  compile_openssl "$N" "$F" "openssl_$N"

  # 产出 ir_oN/ 逐文件 .ll（在 Configure+make 之后立即执行）
  IR_DIR="$DIR/ir_$N"
  rm -rf "$IR_DIR"
  mkdir -p "$IR_DIR"
  n_ir=0
  while IFS= read -r -d '' cfile; do
    bn=$(echo "$cfile" | tr '/' '_')
    bn="${bn%.c}.ll"
    cfile_dir=$(dirname "$cfile")
    if clang $F $SSL_INC $SSL_DEFS -I"$cfile_dir" -emit-llvm -S "$cfile" -o "$IR_DIR/$bn" 2>/dev/null; then
      n_ir=$((n_ir+1))
    fi
  done < <(find "${IR_SRC_DIRS[@]}" -name "*.c" -not -path "*/test/*" -not -path "*/fuzz/*" -print0 2>/dev/null)
  echo "    -> $IR_DIR/ ($n_ir 个 .ll)"
done

echo "[6/6] 完成"
echo "  O0 ELF:     $DIR/openssl_o0"
echo "  O0 IR 目录: $DIR/ir_o0/ ($n_ok 个 .ll)"
echo "  O1/O2/O3:   $DIR/openssl_o1 $DIR/ir_o1/  $DIR/openssl_o2 $DIR/ir_o2/  $DIR/openssl_o3 $DIR/ir_o3/"
echo "  源码目录:   $DIR (--src 指向此目录)"

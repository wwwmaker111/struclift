#!/bin/bash
# 下载并编译 BusyBox：O0 按 BB_Source_Alignment_Pipeline.md，O1/O2/O3 用于 Module A
# 产出: busybox_o0, busybox_o0.ll, busybox_o1, busybox_o2, busybox_o3
# BusyBox 用 Kbuild (defconfig)，编译为单一静态二进制

set -e
BB_VERSION="${BB_VERSION:-1.36.1}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/busybox}"
DOWNLOAD_URL="https://busybox.net/downloads/busybox-${BB_VERSION}.tar.bz2"
TARBALL="/tmp/busybox-${BB_VERSION}.tar.bz2"
DIR="${BASE_DIR}/busybox-${BB_VERSION}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "[1/5] 下载 busybox-${BB_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -sL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/5] 解压 ..."
  tar -xjf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

echo "[2b/5] 配置 (defconfig + 静态编译) ..."
make defconfig
# 确保静态编译、禁用 PIE（更干净的反汇编）
sed -i 's/# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config 2>/dev/null || true
sed -i 's/CONFIG_PIE=y/# CONFIG_PIE is not set/' .config 2>/dev/null || true
yes "" | make oldconfig 2>/dev/null || true

echo "[3/5] O0 (按 MD) ..."
make clean 2>/dev/null || true
make -j"$NPROC" CC=clang EXTRA_CFLAGS="$MD_CFLAGS"
cp -f busybox_unstripped busybox_o0
echo "    -> $DIR/busybox_o0"

echo "    [.ll] 编译所有 .c 为 .bc（BusyBox 文件多，约 2-5 分钟）..."
BC_DIR="$DIR/.bc_o0"
rm -rf "$BC_DIR"
mkdir -p "$BC_DIR"

# BusyBox Kbuild 实际编译参数（从 make V=1 提取的关键标志）
BB_INC="-std=gnu99 -Iinclude -Ilibbb -I. -include include/autoconf.h -D_GNU_SOURCE -DNDEBUG -funsigned-char -w"

# 找到所有实际编译过的 .c（有对应 .o 的文件）
find . -name "*.o" \
  -not -name "built-in.o" \
  -not -name "*.mod.o" \
  -not -path "./scripts/*" \
  -not -path "./.bc_o0/*" \
  | sort > /tmp/bb_ofiles.txt

n_ok=0
n_fail=0
while IFS= read -r ofile; do
  cfile="${ofile%.o}.c"
  if [ -f "$cfile" ]; then
    bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
    bn="${bn%.c}.bc"
    cfile_dir=$(dirname "$cfile")
    if clang $MD_CFLAGS $BB_INC -I"$cfile_dir" -emit-llvm -c "$cfile" -o "$BC_DIR/$bn" 2>/dev/null; then
      n_ok=$((n_ok+1))
    else
      n_fail=$((n_fail+1))
    fi
  fi
done < /tmp/bb_ofiles.txt
rm -f /tmp/bb_ofiles.txt
echo "    .bc 编译: 成功 $n_ok, 失败 $n_fail"

n_bc=$(find "$BC_DIR" -name "*.bc" 2>/dev/null | wc -l)
if [ "$n_bc" -gt 0 ]; then
  llvm-link "$BC_DIR"/*.bc -o "$DIR/busybox_o0.bc" 2>/dev/null
  if [ -f "$DIR/busybox_o0.bc" ]; then
    llvm-dis "$DIR/busybox_o0.bc" -o "$DIR/busybox_o0.ll"
    rm -f "$DIR/busybox_o0.bc"
    echo "    -> $DIR/busybox_o0.ll ($n_bc 个 .bc 已链接)"
  else
    echo "    警告: llvm-link 失败，O0 将仅用 DWARF"
  fi
else
  echo "    警告: 未生成 .bc，跳过 busybox_o0.ll（O0 将仅用 DWARF）"
fi
rm -rf "$BC_DIR"

echo "[4/5] O1 / O2 / O3 (ELF + IR) ..."
for opt in 1 2 3; do
  case $opt in 1) F="$O1_CFLAGS" ;; 2) F="$O2_CFLAGS" ;; 3) F="$O3_CFLAGS" ;; esac
  make clean 2>/dev/null || true
  make -j"$NPROC" CC=clang EXTRA_CFLAGS="$F"
  cp -f busybox_unstripped "busybox_o$opt"
  echo "    -> $DIR/busybox_o$opt"

  # 产出 busybox_oN.ll（与 O0 相同流程，仅优化级别不同）
  BC_DIR="$DIR/.bc_o${opt}"
  rm -rf "$BC_DIR"
  mkdir -p "$BC_DIR"
  find . -name "*.o" \
    -not -name "built-in.o" \
    -not -name "*.mod.o" \
    -not -path "./scripts/*" \
    -not -path "./.bc_o*/*" \
    | sort > /tmp/bb_ofiles.txt
  n_ok=0
  n_fail=0
  while IFS= read -r ofile; do
    cfile="${ofile%.o}.c"
    if [ -f "$cfile" ]; then
      bn=$(echo "$cfile" | sed 's|^\./||' | tr '/' '_')
      bn="${bn%.c}.bc"
      cfile_dir=$(dirname "$cfile")
      if clang $F $BB_INC -I"$cfile_dir" -emit-llvm -c "$cfile" -o "$BC_DIR/$bn" 2>/dev/null; then
        n_ok=$((n_ok+1))
      else
        n_fail=$((n_fail+1))
      fi
    fi
  done < /tmp/bb_ofiles.txt
  rm -f /tmp/bb_ofiles.txt
  n_bc=$(find "$BC_DIR" -name "*.bc" 2>/dev/null | wc -l)
  if [ "$n_bc" -gt 0 ]; then
    llvm-link "$BC_DIR"/*.bc -o "$DIR/busybox_o${opt}.bc" 2>/dev/null
    if [ -f "$DIR/busybox_o${opt}.bc" ]; then
      llvm-dis "$DIR/busybox_o${opt}.bc" -o "$DIR/busybox_o${opt}.ll"
      rm -f "$DIR/busybox_o${opt}.bc"
      echo "    -> $DIR/busybox_o${opt}.ll ($n_bc 个 .bc 已链接)"
    fi
  fi
  rm -rf "$BC_DIR"
done

echo "[5/5] 完成"
echo "  O0: $DIR/busybox_o0  $DIR/busybox_o0.ll"
echo "  O1/O2/O3: $DIR/busybox_o1 $DIR/busybox_o1.ll  $DIR/busybox_o2 $DIR/busybox_o2.ll  $DIR/busybox_o3 $DIR/busybox_o3.ll"
echo "  源码目录: $DIR (--src 指向此目录)"

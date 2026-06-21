#!/bin/bash
# coreutils + libxml2: 下载、编译 O0-O3 + IR，构建全部数据集（完整流水线，与 zlib/busybox/openssl 一致）
# 用法: bash scripts/compile_and_regen_coreutils_libxml2.sh
set -e
cd "$(dirname "$0")/.." || cd /mnt/e/structlift
[ -n "$VIRTUAL_ENV" ] || source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true
OUT="${OUT:-$HOME/structlift_datasets}"

# 解析路径：支持 ~/struclift_wsl 或 /mnt/e/structlift_src_datasets
resolve_dir() {
  local pattern="$1"
  for base in "$HOME/struclift_wsl/datasets" "/mnt/e/structlift_src_datasets"; do
    [ -d "$base" ] || continue
    local d=$(ls -d ${base}/${pattern} 2>/dev/null | head -1)
    [ -n "$d" ] && [ -d "$d" ] && echo "$d" && return
  done
  echo ""
}

# 确定源码/输出目录（优先 Linux 原生路径 ~/struclift_wsl，避免 /mnt 上 GCC 失败）
if [ -d "$HOME/struclift_wsl/datasets" ]; then
  BASE_COREUTILS="$HOME/struclift_wsl/datasets/coreutils"
  BASE_LIBXML2="$HOME/struclift_wsl/datasets/libxml2"
else
  BASE_COREUTILS="/mnt/e/structlift_src_datasets/coreutils"
  BASE_LIBXML2="/mnt/e/structlift_src_datasets/libxml2"
fi

echo "===== [1/4] coreutils 编译 ====="
bash scripts/compile_coreutils.sh "$BASE_COREUTILS"

echo ""
echo "===== [2/4] coreutils 数据集 O0-O3 ====="
CDIR=$(resolve_dir "coreutils/coreutils-*")
if [ -n "$CDIR" ]; then
  for opt in 0 1 2 3; do
    elf="$CDIR/coreutils_o$opt"
    irdir="$CDIR/ir_o$opt"
    out="$OUT/binskel_coreutils_o$opt.jsonl"
    [ -f "$elf" ] || { echo "  跳过 O$opt: $elf 不存在"; continue; }
    echo "  coreutils O$opt ..."
    ir_args=""
    [ -d "$irdir" ] && ir_args="--llvm-ir-dir $irdir"
    python build_binskel_dataset_md.py --elf "$elf" --src "$CDIR" --out "$out" --opt "O$opt" $ir_args 2>&1 | tail -3
    [ -f "$out" ] && echo "    完成: $out ($(wc -l < "$out") 样本)"
  done
else
  echo "  跳过: 未找到 coreutils 目录"
fi

echo ""
echo "===== [3/4] libxml2 编译 ====="
bash scripts/compile_libxml2.sh "$BASE_LIBXML2"

echo ""
echo "===== [4/4] libxml2 数据集 O0-O3 ====="
XDIR=$(resolve_dir "libxml2/libxml2-*")
if [ -n "$XDIR" ]; then
  for opt in 0 1 2 3; do
    elf="$XDIR/xmllint_o$opt"
    irdir="$XDIR/ir_o$opt"
    out="$OUT/binskel_libxml2_o$opt.jsonl"
    [ -f "$elf" ] || { echo "  跳过 O$opt: $elf 不存在"; continue; }
    echo "  libxml2 O$opt ..."
    ir_args=""
    [ -d "$irdir" ] && ir_args="--llvm-ir-dir $irdir"
    python build_binskel_dataset_md.py --elf "$elf" --src "$XDIR" --out "$out" --opt "O$opt" $ir_args 2>&1 | tail -3
    [ -f "$out" ] && echo "    完成: $out ($(wc -l < "$out") 样本)"
  done
else
  echo "  跳过: 未找到 libxml2 目录"
fi

echo ""
echo "===== coreutils + libxml2 全部完成 $(date) ====="

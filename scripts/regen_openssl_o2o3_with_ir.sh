#!/bin/bash
# 在 compile_openssl 完成后，用 ir_o2/ ir_o3/ 重新构建 OpenSSL O2、O3 数据集（含 IR）
# 用法: bash scripts/regen_openssl_o2o3_with_ir.sh
set -e
cd "$(dirname "$0")/.."
[ -n "$VIRTUAL_ENV" ] || source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true
OUT="${OUT:-$HOME/structlift_datasets}"
SDIR=~/struclift_wsl/datasets/openssl/openssl-3.3.1

for opt in 2 3; do
  irdir="$SDIR/ir_o$opt"
  elf="$SDIR/openssl_o$opt"
  outfile="$OUT/binskel_openssl_o$opt.jsonl"
  echo ""
  echo "===== openssl O$opt (完整对齐 + IR) ====="
  if [ ! -f "$elf" ]; then
    echo "  跳过: $elf 不存在"
    continue
  fi
  if [ ! -d "$irdir" ]; then
    echo "  跳过: $irdir 不存在，请先运行 compile_openssl.sh"
    continue
  fi
  python build_binskel_dataset_md.py \
    --elf "$elf" \
    --src "$SDIR" \
    --out "$outfile" \
    --opt "O$opt" \
    --llvm-ir-dir "$irdir"
  echo "  完成: $outfile"
  [ -f "$outfile" ] && echo "  样本数: $(wc -l < "$outfile")"
done
echo ""
echo "===== OpenSSL O2/O3 重建完成 ====="

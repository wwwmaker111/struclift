#!/bin/bash
# 等待 OpenSSL ir_o2、ir_o3 就绪后：regen OpenSSL O2/O3，再编译+构建 coreutils、libxml2
# 用法: bash scripts/wait_openssl_then_regen_and_coreutils_libxml2.sh
# 在 OpenSSL compile 进行中时运行，本脚本会轮询等待 ir 完成
set -e
cd "$(dirname "$0")/.." || cd /mnt/e/structlift
[ -n "$VIRTUAL_ENV" ] || source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true

SDIR=~/struclift_wsl/datasets/openssl/openssl-3.3.1
LOG="$PWD/scripts/wait_openssl_regen_coreutils_libxml2_log.txt"

echo "===== 开始 $(date) =====" | tee "$LOG"

echo "[1/3] 等待 OpenSSL ir_o2、ir_o3 就绪 ..." | tee -a "$LOG"
while true; do
  if [ -d "$SDIR/ir_o2" ] && [ -d "$SDIR/ir_o3" ]; then
    echo "  ir_o2、ir_o3 已就绪" | tee -a "$LOG"
    break
  fi
  echo "  $(date +%H:%M:%S) 等待中 ..." | tee -a "$LOG"
  sleep 60
done

echo "" | tee -a "$LOG"
echo "[2/3] OpenSSL O2/O3 用 IR 重建数据集 =====" | tee -a "$LOG"
bash scripts/regen_openssl_o2o3_with_ir.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[3/3] coreutils + libxml2 编译与数据集构建 =====" | tee -a "$LOG"
bash scripts/compile_and_regen_coreutils_libxml2.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== 全部完成 $(date) =====" | tee -a "$LOG"

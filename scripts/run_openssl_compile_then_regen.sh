#!/bin/bash
# OpenSSL: 编译 O0-O3 + IR，完成后自动 regen O2/O3（含 IR）
# 用法: bash scripts/run_openssl_compile_then_regen.sh
set -e
cd "$(dirname "$0")/.." || cd /mnt/e/structlift
[ -n "$VIRTUAL_ENV" ] || source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true

echo "===== [1/2] OpenSSL 编译 O0-O3 + IR ====="
bash scripts/compile_openssl.sh

echo ""
echo "===== [2/2] OpenSSL O2/O3 用 IR 重建数据集 ====="
bash scripts/regen_openssl_o2o3_with_ir.sh

echo ""
echo "===== OpenSSL 全部完成 $(date) ====="

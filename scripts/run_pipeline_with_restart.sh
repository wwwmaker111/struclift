#!/bin/bash
# 后台运行：等待 OpenSSL → regen O2/O3 → coreutils + libxml2
# 意外退出时自动重启
set -o pipefail
cd "$(dirname "$0")/.." || cd /mnt/e/structlift
LOG="/mnt/e/structlift/scripts/pipeline_restart_log.txt"

while true; do
  echo "===== 启动 $(date) =====" | tee -a "$LOG"
  if bash scripts/wait_openssl_then_regen_and_coreutils_libxml2.sh 2>&1 | tee -a "$LOG"; then
    echo "===== 正常完成 $(date) =====" | tee -a "$LOG"
    break
  fi
  echo "===== 意外退出，10 秒后重启 $(date) =====" | tee -a "$LOG"
  sleep 10
done

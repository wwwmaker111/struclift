#!/bin/bash
# 完整重建 O0 + O1-O3 四个项目，输出到 regen_full_log.txt
set -e
cd "$(dirname "$0")/.." || cd /mnt/e/structlift
LOG="$PWD/scripts/regen_full_log.txt"
echo "===== 开始 $(date) =====" | tee "$LOG"
echo "" | tee -a "$LOG"

echo "===== [1/2] O0 重建 =====" | tee -a "$LOG"
bash scripts/regen_sqlite_openssl_busybox_o0.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== [2/2] O1-O3 重建 =====" | tee -a "$LOG"
bash scripts/regen_o1o3_four_projects.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== 全部完成 $(date) =====" | tee -a "$LOG"

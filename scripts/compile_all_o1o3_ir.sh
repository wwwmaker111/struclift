#!/bin/bash
# 后台编译四个项目，生成 O1-O3 的 IR（zlib_oN.ll, sqlite_oN.ll, busybox_oN.ll, ir_o1/ir_o2/ir_o3）
# 用法: bash scripts/compile_all_o1o3_ir.sh
set -e
cd "$(dirname "$0")/.."
LOG="$PWD/scripts/compile_o1o3_ir_log.txt"
echo "===== 开始 $(date) =====" | tee "$LOG"

echo "" | tee -a "$LOG"
echo "===== [1/4] zlib O1-O3 IR =====" | tee -a "$LOG"
bash scripts/compile_zlib.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== [2/4] SQLite O1-O3 IR =====" | tee -a "$LOG"
bash scripts/compile_sqlite.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== [3/4] BusyBox O1-O3 IR =====" | tee -a "$LOG"
bash scripts/compile_busybox.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== [4/4] OpenSSL O1-O3 IR =====" | tee -a "$LOG"
bash scripts/compile_openssl.sh 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== 全部完成 $(date) =====" | tee -a "$LOG"

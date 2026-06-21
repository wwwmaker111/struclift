#!/bin/bash
# 除 zlib 外五个项目：完整流水线重新编译 + 生成 binskel O0–O3（DWARF+IR+tree-sitter）
# 项目: curl, sqlite, busybox, openssl, libxml2
#
# 用法:
#   bash scripts/regen_five_projects_full_pipeline.sh
# 环境变量:
#   WORKDIR   默认: 脚本所在仓库根
#   DATA      默认: $HOME/struclift_wsl/datasets
#   OUT       默认: $HOME/structlift_datasets
#   VENV      默认: $HOME/struclift_wsl/.venv/bin/activate
#   SKIP_COMPILE=1  仅构建 binskel（假定 ELF/IR 已存在）
#
# 不使用 pipefail：部分 sh/旧 bash 不支持；CRLF 会导致 "pipefail\r" 报 invalid option
set -eu

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA="${DATA:-$HOME/struclift_wsl/datasets}"
OUT="${OUT:-$HOME/structlift_datasets}"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/regen_five_projects_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "regen_five_projects_full_pipeline"
echo "WORKDIR=$WORKDIR DATA=$DATA OUT=$OUT"
echo "LOG=$LOG"
echo "SKIP_COMPILE=${SKIP_COMPILE:-0}"
echo "========================================"

cd "$WORKDIR"
if [ -f "$VENV" ]; then
  # shellcheck source=/dev/null
  source "$VENV"
else
  echo "警告: 未找到 VENV=$VENV，请自行激活 Python 环境"
fi

CR="${DATA}/curl/curl-8.7.1"
SQLITE_DIR=$(ls -d "$DATA"/sqlite/sqlite-* 2>/dev/null | head -1 || true)
BR="${DATA}/busybox/busybox-1.36.1"
OR="${DATA}/openssl/openssl-3.3.1"
# libxml2 目录名随版本
XR=$(ls -d "$DATA"/libxml2/libxml2-* 2>/dev/null | head -1 || true)

die() { echo "ERROR: $*"; exit 1; }

if [ "${SKIP_COMPILE:-0}" != 1 ]; then
  echo ""
  echo "########## [1/5] compile curl ##########"
  bash "$WORKDIR/scripts/compile_curl.sh" "$DATA/curl" || die "compile_curl failed"

  echo ""
  echo "########## [2/5] compile sqlite ##########"
  bash "$WORKDIR/scripts/compile_sqlite.sh" "$DATA/sqlite" || die "compile_sqlite failed"

  echo ""
  echo "########## [3/5] compile busybox ##########"
  bash "$WORKDIR/scripts/compile_busybox.sh" "$DATA/busybox" || die "compile_busybox failed"

  echo ""
  echo "########## [4/5] compile openssl ##########"
  bash "$WORKDIR/scripts/compile_openssl.sh" "$DATA/openssl" || die "compile_openssl failed"

  echo ""
  echo "########## [5/5] compile libxml2 ##########"
  bash "$WORKDIR/scripts/compile_libxml2.sh" "$DATA/libxml2" || die "compile_libxml2 failed"
else
  echo "SKIP_COMPILE=1，跳过编译步骤"
fi

SQLITE_DIR=$(ls -d "$DATA"/sqlite/sqlite-* 2>/dev/null | head -1 || true)
XR=$(ls -d "$DATA"/libxml2/libxml2-* 2>/dev/null | head -1 || true)

[ -d "$CR" ] || die "curl 目录不存在: $CR"
[ -n "$SQLITE_DIR" ] && [ -f "$SQLITE_DIR/sqlite_o0" ] || die "sqlite 目录或 sqlite_o0 不存在"
[ -d "$BR" ] || die "busybox 目录不存在: $BR"
[ -d "$OR" ] || die "openssl 目录不存在: $OR"
[ -n "$XR" ] && [ -f "$XR/xmllint_o0" ] || die "libxml2 目录或 xmllint_o0 不存在: $XR"

mkdir -p "$OUT"

echo ""
echo "########## build binskel: curl ##########"
bash "$WORKDIR/scripts/build_curl_binskel.sh" "$CR" "$OUT" "$WORKDIR"

echo ""
echo "########## build binskel: sqlite ##########"
bash "$WORKDIR/scripts/build_sqlite_binskel.sh" "$SQLITE_DIR" "$OUT" "$WORKDIR"

echo ""
echo "########## build binskel: busybox ##########"
bash "$WORKDIR/scripts/build_busybox_binskel.sh" "$BR" "$OUT" "$WORKDIR"

echo ""
echo "########## build binskel: openssl ##########"
bash "$WORKDIR/scripts/build_openssl_binskel.sh" "$OR" "$OUT" "$WORKDIR"

echo ""
echo "########## build binskel: libxml2 ##########"
bash "$WORKDIR/scripts/build_libxml2_binskel.sh" "$XR" "$OUT" "$WORKDIR"

echo ""
echo "########## 数据集统计 ##########"
python "$WORKDIR/scripts/binskel_dataset_stats.py" \
  "$OUT/binskel_curl_o0.jsonl" \
  "$OUT/binskel_curl_o1.jsonl" \
  "$OUT/binskel_curl_o2.jsonl" \
  "$OUT/binskel_curl_o3.jsonl" \
  "$OUT/binskel_sqlite_o0.jsonl" \
  "$OUT/binskel_sqlite_o1.jsonl" \
  "$OUT/binskel_sqlite_o2.jsonl" \
  "$OUT/binskel_sqlite_o3.jsonl" \
  "$OUT/binskel_busybox_o0.jsonl" \
  "$OUT/binskel_busybox_o1.jsonl" \
  "$OUT/binskel_busybox_o2.jsonl" \
  "$OUT/binskel_busybox_o3.jsonl" \
  "$OUT/binskel_openssl_o0.jsonl" \
  "$OUT/binskel_openssl_o1.jsonl" \
  "$OUT/binskel_openssl_o2.jsonl" \
  "$OUT/binskel_openssl_o3.jsonl" \
  "$OUT/binskel_libxml2_o0.jsonl" \
  "$OUT/binskel_libxml2_o1.jsonl" \
  "$OUT/binskel_libxml2_o2.jsonl" \
  "$OUT/binskel_libxml2_o3.jsonl" \
  --markdown || true

echo ""
echo "全部完成。日志: $LOG"

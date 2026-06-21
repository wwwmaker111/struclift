#!/usr/bin/env bash
#
# 九项目仅生成 binskel_*_os.jsonl（需先完成 compile_os_*）。在仓库根执行：
#
#   export OUT_ROOT=~/structlift_datasets
#   export ZLIB_DIR=... CURL_DIR=... （与 compile_os_all_9.sh 相同变量名）
#   export COREUTILS_DIR=...   # coreutils 产物目录（含 coreutils_os、ir_os）
#
#   bash scripts/os_only/build_binskel_os_all_9.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:?请设置 OUT_ROOT（jsonl 输出目录）}"
S="$ROOT/scripts/os_only"

run_one() {
  local name="$1" script="$2" vname="$3"
  local dir="${!vname:-}"
  if [[ -z "${dir}" ]]; then
    echo "[跳过] $name：未设置 $vname"
    return 0
  fi
  if [[ ! -d "$dir" ]]; then
    echo "[跳过] $name：目录不存在 $dir"
    return 0
  fi
  echo ""
  echo "########## binskel Os: $name ##########"
  bash "$S/$script" "$dir" "$OUT_ROOT" "$ROOT"
}

run_one zlib      build_zlib_os_binskel.sh      ZLIB_DIR
run_one curl      build_curl_os_binskel.sh      CURL_DIR
run_one sqlite    build_sqlite_os_binskel.sh    SQLITE_DIR
run_one busybox   build_busybox_os_binskel.sh   BUSYBOX_DIR
run_one openssl   build_openssl_os_binskel.sh   OPENSSL_DIR
run_one openssh   build_openssh_os_binskel.sh   OPENSSH_DIR
run_one libxml2   build_libxml2_os_binskel.sh   LIBXML2_DIR
run_one ffmpeg    build_ffmpeg_os_binskel.sh    FFMPEG_DIR
run_one coreutils build_coreutils_os_binskel.sh COREUTILS_DIR

echo ""
echo "全部 Os binskel 调度结束。输出: $OUT_ROOT/binskel_*_os.jsonl"

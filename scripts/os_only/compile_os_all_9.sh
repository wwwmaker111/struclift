#!/usr/bin/env bash
#
# 九项目仅编译 -Os 档（O0–O3 已有时使用）。在仓库根执行，先 export 各项目源码目录：
#
#   export ZLIB_DIR=~/struclift_wsl/datasets/zlib/zlib-1.3.1
#   export CURL_DIR=~/struclift_wsl/datasets/curl/curl-8.7.1
#   export SQLITE_DIR=~/struclift_wsl/datasets/sqlite/sqlite-3450100
#   export BUSYBOX_DIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
#   export OPENSSL_DIR=~/struclift_wsl/datasets/openssl/openssl-3.3.1
#   export OPENSSH_DIR=~/struclift_wsl/datasets/openssh/openssh-9.8p1
#   export LIBXML2_DIR=~/struclift_wsl/datasets/libxml2/libxml2-2.12.7
#   export FFMPEG_DIR=~/struclift_wsl/datasets/ffmpeg/ffmpeg-7.1
#   export COREUTILS_BUILD_DIR=~/struclift_wsl/datasets/coreutils/coreutils-9.4
#   # 若 coreutils 产物与源码同目录，可只设 COREUTILS_BUILD_DIR；否则再设 COREUTILS_OUT_DIR
#   # export COREUTILS_OUT_DIR=...
#
#   bash scripts/os_only/compile_os_all_9.sh
#
# 未设置的变量对应项目会跳过。
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S="$ROOT/scripts/os_only"

run() {
  local name="$1" script="$2" dir="${3:-}"
  if [[ -z "${dir}" ]]; then
    echo "[跳过] $name：未设置对应 *_DIR"
    return 0
  fi
  if [[ ! -d "$dir" ]]; then
    echo "[跳过] $name：目录不存在 $dir"
    return 0
  fi
  echo ""
  echo "########## compile Os: $name ##########"
  bash "$S/$script" "$dir"
}

run zlib     compile_os_zlib.sh       "${ZLIB_DIR:-}"
run curl     compile_os_curl.sh       "${CURL_DIR:-}"
run sqlite   compile_os_sqlite.sh     "${SQLITE_DIR:-}"
run busybox  compile_os_busybox.sh    "${BUSYBOX_DIR:-}"
run openssl  compile_os_openssl.sh    "${OPENSSL_DIR:-}"
run openssh  compile_os_openssh.sh    "${OPENSSH_DIR:-}"
# libxml2：第二参数为产物目录时 export LIBXML2_OUT_DIR
if [[ -n "${LIBXML2_DIR:-}" ]]; then
  if [[ -n "${LIBXML2_OUT_DIR:-}" ]]; then
    bash "$S/compile_os_libxml2.sh" "$LIBXML2_DIR" "$LIBXML2_OUT_DIR"
  else
    bash "$S/compile_os_libxml2.sh" "$LIBXML2_DIR"
  fi
else
  echo "[跳过] libxml2：未设置 LIBXML2_DIR"
fi
run ffmpeg   compile_os_ffmpeg.sh     "${FFMPEG_DIR:-}"

if [[ -n "${COREUTILS_BUILD_DIR:-}" ]]; then
  echo ""
  echo "########## compile Os: coreutils ##########"
  bash "$S/compile_os_coreutils.sh" "$COREUTILS_BUILD_DIR" "${COREUTILS_OUT_DIR:-$COREUTILS_BUILD_DIR}"
elif [[ -n "${COREUTILS_DIR:-}" ]]; then
  echo ""
  echo "########## compile Os: coreutils ##########"
  bash "$S/compile_os_coreutils.sh" "$COREUTILS_DIR" "${COREUTILS_OUT_DIR:-$COREUTILS_DIR}"
else
  echo "[跳过] coreutils：未设置 COREUTILS_BUILD_DIR 或 COREUTILS_DIR"
fi

echo ""
echo "全部 Os 编译调度结束。"

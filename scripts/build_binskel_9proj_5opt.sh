#!/usr/bin/env bash
#
# 九项目 × 五档（O0–O3 + Os）= 45 个 jsonl 的调度脚本。
#
# 其中 **coreutils / ffmpeg / openssh** 使用 multibin 合并脚本（与 stats 脚本命名一致）；
# 其余六项目为单 ELF 每档 + Os 单独脚本。
#
# 前置：各项目已按仓库 scripts/compile_*.sh / compile_*_multibin.sh 完成编译（含 .ll 或 ir_* / multibin_*）。
#
# 用法（在 structlift 仓库根执行，推荐 WSL/Linux bash）:
#
#   export OUT_ROOT="$HOME/structlift_datasets"
#   export ZLIB_DIR="$HOME/struclift_wsl/datasets/zlib/zlib-1.3.1"
#   export CURL_DIR="..."
#   export SQLITE_DIR="..."
#   export BUSYBOX_DIR="..."
#   export OPENSSL_DIR="..."
#   export OPENSSH_DIR="..."      # multibin
#   export LIBXML2_DIR="..."
#   export FFMPEG_DIR="..."       # multibin
#   export COREUTILS_DIR="..."    # multibin
#   # 可选：去掉 jsonl 里 binary/source_file 的机器相关前缀
#   # export STRIP_PATH_PREFIX="/home/yourname/"
#
#   bash scripts/build_binskel_9proj_5opt.sh
#
# 若某 *_DIR 未设置，对应项目会跳过。
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:?请设置 OUT_ROOT（jsonl 输出目录，如 ~/structlift_datasets）}"

run_or_skip() {
  local name="$1" script_path="$2" dir_var="$3"
  local dir="${!dir_var:-}"
  if [[ -z "${dir}" ]]; then
    echo "[跳过] ${name}：未设置 ${dir_var}"
    return 0
  fi
  if [[ ! -d "${dir}" ]]; then
    echo "[跳过] ${name}：目录不存在 ${dir}"
    return 0
  fi
  echo ""
  echo "###################################################################"
  echo "# ${name}"
  echo "###################################################################"
  bash "${ROOT}/${script_path}" "${dir}" "${OUT_ROOT}" "${ROOT}"
}

echo "OUT_ROOT=${OUT_ROOT}"
echo "STRIP_PATH_PREFIX=${STRIP_PATH_PREFIX:-<未设置>}"
echo ""

# 六项目：单 ELF × O0–O3 + Os（共 5 文件/项目）
run_or_skip "zlib"     "scripts/build_zlib_binskel.sh"       ZLIB_DIR
run_or_skip "zlib Os"  "scripts/os_only/build_zlib_os_binskel.sh" ZLIB_DIR

run_or_skip "curl"     "scripts/build_curl_binskel.sh"       CURL_DIR
run_or_skip "curl Os"  "scripts/os_only/build_curl_os_binskel.sh" CURL_DIR

run_or_skip "sqlite"   "scripts/build_sqlite_binskel.sh"     SQLITE_DIR
run_or_skip "sqlite Os" "scripts/os_only/build_sqlite_os_binskel.sh" SQLITE_DIR

run_or_skip "busybox"  "scripts/build_busybox_binskel.sh"    BUSYBOX_DIR
run_or_skip "busybox Os" "scripts/os_only/build_busybox_os_binskel.sh" BUSYBOX_DIR

run_or_skip "openssl"  "scripts/build_openssl_binskel.sh"    OPENSSL_DIR
run_or_skip "openssl Os" "scripts/os_only/build_openssl_os_binskel.sh" OPENSSL_DIR

run_or_skip "libxml2"  "scripts/build_libxml2_binskel.sh"    LIBXML2_DIR
run_or_skip "libxml2 Os" "scripts/os_only/build_libxml2_os_binskel.sh" LIBXML2_DIR

# 三项目：multibin（每档一条合并 jsonl，共 5 文件/项目）
run_or_skip "openssh (multibin)"  "scripts/build_openssh_multibin_binskel.sh"  OPENSSH_DIR
run_or_skip "ffmpeg (multibin)"   "scripts/build_ffmpeg_multibin_binskel.sh"   FFMPEG_DIR
run_or_skip "coreutils (multibin)" "scripts/build_coreutils_multibin_binskel.sh" COREUTILS_DIR

echo ""
echo "调度结束。输出目录: ${OUT_ROOT}"
echo "期望 45 个文件（名称见 docs/BUILD_BINSKEL_9x5_FULL.md）。"

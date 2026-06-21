#!/usr/bin/env bash
#
# 按 scripts/binskel_alignment_batch_stats.py 默认「8 项目」顺序批量重导 binskel jsonl：
#   zlib → curl → sqlite → busybox → openssl → openssh → libxml2 → ffmpeg
#
# 与论文 BinSkel-Train 表格相比：多 openssh、ffmpeg；无 coreutils（coreutils 见 compile_and_regen_coreutils_libxml2.sh）。
#
# 用法（在已编译好各项目 O0–O3 + IR 的前提下）：
#
#   export OUT_ROOT=~/structlift_datasets
#   export ZLIB_DIR=~/struclift_wsl/datasets/zlib/zlib-1.3.1
#   export CURL_DIR=~/struclift_wsl/datasets/curl/curl-8.7.1
#   # sqlite：务必与 scripts/compile_sqlite.sh 一致，目录名为 sqlite-<版本号>（amalgamation），
#   # 不是 sqlite-autoconf-*；默认编译为 sqlite-3450100，以你机器上 ls ~/struclift_wsl/datasets/sqlite/ 为准。
#   export SQLITE_DIR=~/struclift_wsl/datasets/sqlite/sqlite-3450100
#   export BUSYBOX_DIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
#   # 未编译的项目不要填占位路径；不设该变量会「跳过」该项，不会报错。
#   export OPENSSL_DIR=...   # 内含 openssl_o*、ir_o*
#   export OPENSSH_DIR=...   # 内含 openssh_o*、ir_o*
#   export LIBXML2_DIR=...    # 内含 xmllint_o*、ir_o*
#   export FFMPEG_DIR=...     # 内含 ffmpeg_o*、ir_o*
#
#   bash scripts/rebuild_binskel_all_8.sh
#
# 已从 zlib/curl 跑完、只想续跑后面几项时：
#   export START_AT=sqlite   # 或 busybox / openssl / openssh / libxml2 / ffmpeg
#   bash scripts/rebuild_binskel_all_8.sh
#
# 若出现「cd: /path/to/structlift」：检查 shell 里是否误 export 了文档占位路径，
# 在仓库根执行，或 unset 相关变量后重试。
#
# 可选：与训练一致时修改
#   export NUM_OPCODES=1024 SRC_VOCAB=32000 WORKERS=8
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:?请设置 OUT_ROOT（jsonl 输出目录）}"

NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
export NUM_OPCODES SRC_VOCAB WORKERS

echo "词表: num_opcodes=$NUM_OPCODES src_vocab_size=$SRC_VOCAB workers=$WORKERS"
echo "输出目录: $OUT_ROOT"
mkdir -p "$OUT_ROOT"

# 从某一步骤起跑，避免重复 zlib/curl。合法值：zlib curl sqlite busybox openssl openssh libxml2 ffmpeg
START_AT="${START_AT:-zlib}"
_valid_steps=(zlib curl sqlite busybox openssl openssh libxml2 ffmpeg)
_ok=
for _s in "${_valid_steps[@]}"; do [[ "$_s" == "$START_AT" ]] && _ok=1 && break; done
[[ -n "$_ok" ]] || { echo "无效 START_AT='$START_AT'，应为: ${_valid_steps[*]}"; exit 1; }
[[ "$START_AT" == zlib ]] || echo "从步骤「$START_AT」起续跑（之前的步骤会跳过）"
_step_ok=0
should_run_step() {
  local name="$1"
  if [[ "$START_AT" == "$name" ]]; then
    _step_ok=1
    return 0
  fi
  [[ "$_step_ok" -eq 1 ]]
}

run_one() {
  local name="$1" script="$2" vname="$3"
  local dir="${!vname:-}"
  if [[ -z "${dir}" ]]; then
    echo "[跳过] $name：未设置环境变量 $vname"
    return 0
  fi
  if [[ ! -d "$dir" ]]; then
    echo "[跳过] $name：目录不存在 $dir"
    return 0
  fi
  echo ""
  echo "########## $name ##########"
  bash "$ROOT/scripts/$script" "$dir" "$OUT_ROOT" "$ROOT"
}

should_run_step zlib     && run_one "zlib"     "build_zlib_binskel.sh"    "ZLIB_DIR"
should_run_step curl     && run_one "curl"     "build_curl_binskel.sh"    "CURL_DIR"
should_run_step sqlite   && run_one "sqlite"   "build_sqlite_binskel.sh"  "SQLITE_DIR"
should_run_step busybox  && run_one "busybox"  "build_busybox_binskel.sh" "BUSYBOX_DIR"
should_run_step openssl  && run_one "openssl"  "build_openssl_binskel.sh" "OPENSSL_DIR"
should_run_step openssh  && run_one "openssh"  "build_openssh_binskel.sh" "OPENSSH_DIR"
should_run_step libxml2  && run_one "libxml2"  "build_libxml2_binskel.sh" "LIBXML2_DIR"
should_run_step ffmpeg   && run_one "ffmpeg"   "build_ffmpeg_binskel.sh"  "FFMPEG_DIR"

echo ""
echo "全部调度结束。输出应在: $OUT_ROOT/binskel_*_o*.jsonl"

#!/usr/bin/env bash
# 后台续跑 4 项：openssl → openssh → libxml2 → ffmpeg（仅构建 binskel jsonl，不编译源码）
# 依赖：各项目已按 scripts/compile_*.sh 编好 O0–O3 + IR，目录默认与 compile 脚本一致。
#
# 用法（在仓库根）:
#   export OUT_ROOT=~/structlift_datasets   # 可选
#   bash scripts/rebuild_binskel_openssl_to_ffmpeg_bg.sh
#
# 前台跑（便于当场看报错）:
#   FOREGROUND=1 bash scripts/rebuild_binskel_openssl_to_ffmpeg_bg.sh
#
# 自定义版本/路径（与 compile_openssl.sh 等一致）:
#   export SSL_VERSION=3.3.1 OPENSSL_DIR=...
#   export OPENSSH_VERSION=9.8p1 OPENSSH_DIR=...
#   export LIBXML2_VERSION=2.12.7 LIBXML2_DIR=...
#   export FFMPEG_VERSION=7.1 FFMPEG_DIR=...
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$HOME/structlift_datasets}"
NOHUP_LOG="${NOHUP_LOG:-$OUT_ROOT/rebuild_binskel_openssl_to_ffmpeg.nohup.log}"

SSL_VERSION="${SSL_VERSION:-3.3.1}"
OPENSSH_VERSION="${OPENSSH_VERSION:-9.8p1}"
LIBXML2_VERSION="${LIBXML2_VERSION:-2.12.7}"
FFMPEG_VERSION="${FFMPEG_VERSION:-7.1}"

export OUT_ROOT
export START_AT=openssl
export OPENSSL_DIR="${OPENSSL_DIR:-$HOME/struclift_wsl/datasets/openssl/openssl-${SSL_VERSION}}"
export OPENSSH_DIR="${OPENSSH_DIR:-$HOME/struclift_wsl/datasets/openssh/openssh-${OPENSSH_VERSION}}"
export LIBXML2_DIR="${LIBXML2_DIR:-$HOME/struclift_wsl/datasets/libxml2/libxml2-${LIBXML2_VERSION}}"
export FFMPEG_DIR="${FFMPEG_DIR:-$HOME/struclift_wsl/datasets/ffmpeg/ffmpeg-${FFMPEG_VERSION}}"

mkdir -p "$OUT_ROOT"
mkdir -p "$(dirname "$NOHUP_LOG")"

_preflight() {
  local miss=0
  need_elf_ir() {
    local name="$1" dir="$2" elf="$3"
    if [[ ! -d "$dir" ]]; then
      echo "[缺失] $name 目录不存在: $dir"
      return 1
    fi
    if [[ ! -f "$dir/$elf" ]]; then
      echo "[缺失] $name 未找到 $dir/$elf（请先运行 scripts/compile_${name}.sh）"
      return 1
    fi
    if [[ ! -d "$dir/ir_o0" ]]; then
      echo "[缺失] $name 缺少 $dir/ir_o0（完整流水线需要 IR 目录）"
      return 1
    fi
    return 0
  }
  need_elf() {
    local name="$1" dir="$2" elf="$3"
    if [[ ! -d "$dir" ]]; then
      echo "[缺失] $name 目录不存在: $dir"
      return 1
    fi
    if [[ ! -f "$dir/$elf" ]]; then
      echo "[缺失] $name 未找到 $dir/$elf（请先运行 scripts/compile_libxml2.sh）"
      return 1
    fi
    return 0
  }

  need_elf_ir openssl "$OPENSSL_DIR" openssl_o0 || miss=1
  need_elf_ir openssh "$OPENSSH_DIR" openssh_o0 || miss=1
  need_elf libxml2 "$LIBXML2_DIR" xmllint_o0 || miss=1
  need_elf_ir ffmpeg "$FFMPEG_DIR" ffmpeg_o0 || miss=1

  if [[ "$miss" -ne 0 ]]; then
    echo "预检失败：补全编译后再执行本脚本。"
    exit 1
  fi
}

_preflight

run_fg() {
  cd "$ROOT"
  bash scripts/rebuild_binskel_all_8.sh
}

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  echo "前台运行 rebuild_binskel_all_8.sh START_AT=openssl ..."
  run_fg 2>&1 | tee -a "$NOHUP_LOG"
  exit 0
fi

echo "日志: $NOHUP_LOG"
echo "已预检 4 项 ELF 存在，后台启动（顺序 openssl→openssh→libxml2→ffmpeg）..."

nohup env \
  OUT_ROOT="$OUT_ROOT" \
  START_AT=openssl \
  NUM_OPCODES="${NUM_OPCODES:-1024}" \
  SRC_VOCAB="${SRC_VOCAB:-32000}" \
  WORKERS="${WORKERS:-8}" \
  OPENSSL_DIR="$OPENSSL_DIR" \
  OPENSSH_DIR="$OPENSSH_DIR" \
  LIBXML2_DIR="$LIBXML2_DIR" \
  FFMPEG_DIR="$FFMPEG_DIR" \
  bash "$ROOT/scripts/rebuild_binskel_all_8.sh" >>"$NOHUP_LOG" 2>&1 &

echo $! >"${NOHUP_LOG%.log}.pid"
echo "PID=$(cat "${NOHUP_LOG%.log}.pid")"
echo "查看进度: tail -f $NOHUP_LOG"

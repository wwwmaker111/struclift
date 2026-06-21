#!/usr/bin/env bash
# =============================================================================
# 【「删除步骤」独立版】mbedtls 源码目录异常时：先删再 git clone
#
# 与 scripts/compile_mbedtls.sh 第 46–49 行逻辑一致：
#   若仅用 GitHub「Source code」tar 解压，会缺少 framework/ 子模块，
#   必须先 rm -rf 该目录，再用带 --recurse-submodules 的 git clone。
#
# 本脚本用于：你希望**强制**清空 mbedtls-${VER} 后重新拉取（例如修坏目录、换版本）。
#
# 用法：
#   bash scripts/force_mbedtls_git_reclone.sh [BASE_DIR]
#   BASE_DIR 默认：$HOME/struclift_wsl/datasets/mbedtls
# 环境变量：MBEDTLS_VERSION（默认 3.6.2），与 compile_mbedtls.sh 一致。
# =============================================================================
set -euo pipefail
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
VER="${MBEDTLS_VERSION:-3.6.2}"
BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/mbedtls}"
DIR="${BASE_DIR}/mbedtls-${VER}"

echo "[force_mbedtls_git_reclone] 将删除: $DIR"
rm -rf "$DIR"
echo "[force_mbedtls_git_reclone] 调用 compile_mbedtls.sh ..."
exec bash "$WORKDIR/scripts/compile_mbedtls.sh" "$BASE_DIR"

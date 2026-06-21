#!/bin/bash
# curl 数据集一键流水线：编译 O0–O3（ELF + DWARF + .ll）→ 四档完整 binskel 对齐 + 校验
#
# 用法:
#   bash scripts/run_curl_binskel_pipeline.sh
#   WORKDIR=/mnt/e/structlift CURL_BASE=~/struclift_wsl/datasets/curl bash scripts/run_curl_binskel_pipeline.sh
#
# 环境变量（均可选，有合理默认）:
#   WORKDIR     structlift 仓库根目录（默认：本脚本所在仓库根）
#   CURL_BASE   curl 数据集父目录，其下应有 curl-8.7.1（默认: $HOME/struclift_wsl/datasets/curl）
#   OUT_DIR     jsonl 输出目录（默认: $HOME/structlift_datasets）
#   VENV        Python venv 的 activate 路径（默认: $HOME/struclift_wsl/.venv/bin/activate）
#   SKIP_COMPILE 若设为 1，跳过 compile，仅构建 binskel（需已有 curl_o0..curl_o3）

set -euo pipefail

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "$_SCRIPTS_DIR/.." && pwd)}"
CURL_BASE="${CURL_BASE:-$HOME/struclift_wsl/datasets/curl}"
OUT_DIR="${OUT_DIR:-$HOME/structlift_datasets}"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
CURL_VERSION="${CURL_VERSION:-8.7.1}"
CDIR="$CURL_BASE/curl-$CURL_VERSION"
SKIP_COMPILE="${SKIP_COMPILE:-0}"

[ -d "$WORKDIR" ] || { echo "WORKDIR 不存在: $WORKDIR"; exit 1; }
[ -f "$WORKDIR/scripts/compile_curl.sh" ] || { echo "未找到 compile_curl.sh，请检查 WORKDIR=$WORKDIR"; exit 1; }
[ -f "$WORKDIR/build_binskel_dataset_md.py" ] || { echo "未找到 build_binskel_dataset_md.py: $WORKDIR"; exit 1; }

if [ -f "$VENV" ]; then
  # shellcheck source=/dev/null
  source "$VENV"
else
  echo "警告: 未找到 VENV=$VENV ，使用当前环境的 python"
fi

cd "$WORKDIR"
mkdir -p "$OUT_DIR"

if [ "$SKIP_COMPILE" != "1" ]; then
  echo "========== [1/2] 编译 curl O0–O3（含 DWARF + .ll）=========="
  bash "$WORKDIR/scripts/compile_curl.sh" "$CURL_BASE"
else
  echo "========== [1/2] 跳过编译 (SKIP_COMPILE=1) =========="
fi

echo "========== [2/2] 构建 binskel O0–O3（完整对齐）=========="
bash "$WORKDIR/scripts/build_curl_binskel.sh" "$CDIR" "$OUT_DIR" "$WORKDIR"

echo "========== 全部结束 =========="
echo "  ELF/源码: $CDIR"
echo "  输出:     $OUT_DIR/binskel_curl_o0.jsonl … binskel_curl_o3.jsonl"

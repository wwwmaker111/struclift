#!/usr/bin/env bash
# =============================================================================
# 【历史「总控脚本」入口恢复】
#
# 早期仓库曾有一个独立的一键总控 shell，后合并为
#   scripts/rebuild_loss_datasets_full.sh
# 本文件为**稳定别名**：行为与后者完全一致，便于旧文档/习惯路径继续可用。
#
# 用法（与 rebuild_loss_datasets_full.sh 相同）：
#   chmod +x scripts/total_control_rebuild_loss_datasets.sh
#   export WORKDIR=/path/to/structlift
#   export DATA_ROOT=$HOME/struclift_wsl/datasets
#   export OUT_DIR=$HOME/structlift_datasets
#   ./scripts/total_control_rebuild_loss_datasets.sh
#
# 详见 scripts/rebuild_loss_datasets_full.sh 文件头注释。
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/rebuild_loss_datasets_full.sh" "$@"

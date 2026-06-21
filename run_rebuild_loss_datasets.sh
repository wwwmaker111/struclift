#!/usr/bin/env bash
# 仓库根目录便捷入口：一键调用「训练损失数据集」总控（zlib/openssl/sqlite/coreutils 等）
# 与 scripts/rebuild_loss_datasets_full.sh 等价；也可改用 scripts/total_control_rebuild_loss_datasets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/scripts/rebuild_loss_datasets_full.sh" "$@"

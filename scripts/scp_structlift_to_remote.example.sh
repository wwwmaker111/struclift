#!/usr/bin/env bash
# 将本仓库同步到远程训练机（在 Windows 上可用 Git Bash / WSL 执行）。
# 使用前编辑 REMOTE 与可选 RSYNC_EXCLUDE；首次运行前: chmod +x scripts/scp_structlift_to_remote.example.sh
#
# 仅上传代码用 rsync 比 scp 递归目录更稳；没有 rsync 可用下方「纯 scp」段落。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# —— 改成你的用户与主机、远程工程目录 ——
REMOTE="user@node4"                    # 例: chaoni@192.168.1.10
REMOTE_DIR="/data/chaoni/structlift"   # 远程 clone 或已有目录

# rsync: 推整个仓库，排除大目录与本地缓存
RSYNC_EXCLUDE=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.venv'
  --exclude 'checkpoints'
)

echo "==> rsync 本地 -> ${REMOTE}:${REMOTE_DIR}"
rsync -avz --delete "${RSYNC_EXCLUDE[@]}" \
  "$ROOT/" "${REMOTE}:${REMOTE_DIR}/"

echo "在远程执行训练（可改为 nohup/tmux）:"
cat <<'CMD'
  cd /data/chaoni/structlift && \
  TRAIN_MAX=250 VAL_MAX=50 EPOCHS=20 \
  JSONL_TRAIN=/data/chaoni/WQM/datasets/AB_train_o0_2048.jsonl \
  VAL_JSONL=/data/chaoni/WQM/datasets/AB_val.jsonl \
  INIT_CKPT=/data/chaoni/WQM/checkpoints/stage1_gpu0_7_tune/best_stage1_snapshot.pt \
  ./scripts/stage2_250_50_20.sh
CMD

# —— 若无 rsync，可用 scp 打包上传（在本地执行） ——
#   cd /e/structlift
#   tar --exclude='.git' -czf /tmp/structlift_code.tgz struclift scripts
#   scp /tmp/structlift_code.tgz user@node4:/data/chaoni/
# 远程: mkdir -p /data/chaoni/structlift && tar -xzf /data/chaoni/structlift_code.tgz -C /data/chaoni/structlift

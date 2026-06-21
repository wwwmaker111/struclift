#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true
OUT="${OUT:-$HOME/structlift_datasets}"
BDIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1

echo "===== BusyBox O0 ====="
[ -f "$BDIR/busybox_o0" ] || { echo "跳过: $BDIR/busybox_o0 不存在"; exit 0; }
python build_binskel_dataset_md.py \
  --elf "$BDIR/busybox_o0" \
  --src "$BDIR" \
  --llvm-ir "$BDIR/busybox_o0.ll" \
  --out "$OUT/binskel_busybox_o0.jsonl" \
  --opt O0 \
  --workers 8
echo "完成: $OUT/binskel_busybox_o0.jsonl"
wc -l < "$OUT/binskel_busybox_o0.jsonl" | xargs echo "样本数:"

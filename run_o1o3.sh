#!/bin/bash
set -e
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate

SDIR=~/struclift_wsl/datasets/sqlite/sqlite-3450100
BDIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
OUT=~/structlift_datasets

echo "===== SQLite O1（完整流水线）====="
time python build_binskel_dataset_md.py \
  --elf "$SDIR/sqlite_o1" --src "$SDIR" \
  --llvm-ir "$SDIR/sqlite_o1.ll" \
  --out "$OUT/binskel_sqlite_o1.jsonl" --opt O1

echo ""
echo "===== SQLite O2（完整流水线）====="
time python build_binskel_dataset_md.py \
  --elf "$SDIR/sqlite_o2" --src "$SDIR" \
  --llvm-ir "$SDIR/sqlite_o2.ll" \
  --out "$OUT/binskel_sqlite_o2.jsonl" --opt O2

echo ""
echo "===== SQLite O3（完整流水线）====="
time python build_binskel_dataset_md.py \
  --elf "$SDIR/sqlite_o3" --src "$SDIR" \
  --llvm-ir "$SDIR/sqlite_o3.ll" \
  --out "$OUT/binskel_sqlite_o3.jsonl" --opt O3

echo ""
echo "===== BusyBox O1（完整流水线）====="
time python build_binskel_dataset_md.py \
  --elf "$BDIR/busybox_o1" --src "$BDIR" \
  --llvm-ir "$BDIR/busybox_o1.ll" \
  --out "$OUT/binskel_busybox_o1.jsonl" --opt O1

echo ""
echo "===== BusyBox O2（完整流水线）====="
time python build_binskel_dataset_md.py \
  --elf "$BDIR/busybox_o2" --src "$BDIR" \
  --llvm-ir "$BDIR/busybox_o2.ll" \
  --out "$OUT/binskel_busybox_o2.jsonl" --opt O2

echo ""
echo "===== BusyBox O3（完整流水线）====="
time python build_binskel_dataset_md.py \
  --elf "$BDIR/busybox_o3" --src "$BDIR" \
  --llvm-ir "$BDIR/busybox_o3.ll" \
  --out "$OUT/binskel_busybox_o3.jsonl" --opt O3

echo ""
echo "===== 验证所有新数据集 ====="
for f in "$OUT"/binskel_sqlite_o1.jsonl "$OUT"/binskel_sqlite_o2.jsonl "$OUT"/binskel_sqlite_o3.jsonl \
         "$OUT"/binskel_busybox_o1.jsonl "$OUT"/binskel_busybox_o2.jsonl "$OUT"/binskel_busybox_o3.jsonl; do
  echo ""
  echo "--- $(basename $f) ---"
  python scripts/validate_alignment.py "$f"
done

echo ""
echo "===== 全部完成 ====="

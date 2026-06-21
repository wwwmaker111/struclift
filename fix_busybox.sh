#!/bin/bash
BDIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1

echo "=== 检查 BusyBox O1-O3 是否有 DWARF ==="
for f in busybox_o1 busybox_o2 busybox_o3; do
  if [ -f "$BDIR/$f" ]; then
    has_dwarf=$(readelf -S "$BDIR/$f" 2>/dev/null | grep -c debug_info || echo 0)
    size=$(ls -lh "$BDIR/$f" | awk '{print $5}')
    echo "  $f: $size, .debug_info 段: $has_dwarf"
  fi
done

echo ""
echo "=== 检查是否有 unstripped 版本可用 ==="
ls -lh "$BDIR"/busybox_unstripped 2>/dev/null || echo "  busybox_unstripped 不存在"

echo ""
echo "=== 需要重新编译 BusyBox O1-O3 ==="
echo "  进入源码目录重新编译..."
cd "$BDIR"

NPROC=$(nproc 2>/dev/null || echo 4)
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

for opt in 1 2 3; do
  case $opt in 1) F="$O1_CFLAGS" ;; 2) F="$O2_CFLAGS" ;; 3) F="$O3_CFLAGS" ;; esac
  echo ""
  echo "--- 编译 O${opt} ---"
  make clean 2>/dev/null || true
  make -j"$NPROC" CC=clang EXTRA_CFLAGS="$F" 2>&1 | tail -3
  if [ -f busybox_unstripped ]; then
    cp -f busybox_unstripped "busybox_o${opt}"
    has_dwarf=$(readelf -S "busybox_o${opt}" 2>/dev/null | grep -c debug_info || echo 0)
    echo "  -> busybox_o${opt} ($(ls -lh busybox_o${opt} | awk '{print $5}'), debug_info: $has_dwarf)"
  else
    echo "  警告: busybox_unstripped 不存在!"
  fi
done

echo ""
echo "=== 重新生成 BusyBox O1-O3 数据集 ==="
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate
OUT=~/structlift_datasets

for opt in 1 2 3; do
  echo ""
  echo "===== BusyBox O${opt} ====="
  IR_ARGS=()
  [ -f "$BDIR/busybox_o${opt}.ll" ] && IR_ARGS=(--llvm-ir "$BDIR/busybox_o${opt}.ll")
  [ ${#IR_ARGS[@]} -eq 0 ] && echo "  警告: 缺少 busybox_o${opt}.ll，将主要依赖 DWARF"
  time python build_binskel_dataset_md.py \
    --elf "$BDIR/busybox_o${opt}" \
    --src "$BDIR" \
    "${IR_ARGS[@]}" \
    --out "$OUT/binskel_busybox_o${opt}.jsonl" \
    --opt "O${opt}"
done

echo ""
echo "===== 验证 ====="
for opt in 1 2 3; do
  echo "--- BusyBox O${opt} ---"
  python scripts/validate_alignment.py "$OUT/binskel_busybox_o${opt}.jsonl"
  echo ""
done

echo "===== 全部完成 ====="

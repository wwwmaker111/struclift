#!/bin/bash
# 重建 zlib、SQLite、BusyBox、OpenSSL 四个项目 O0 数据集（含 pattern_labels 细化、行级置信度修复、tree-sitter 优化）
# 用法: bash scripts/regen_sqlite_openssl_busybox_o0.sh
set -e
cd "$(dirname "$0")/.."
[ -n "$VIRTUAL_ENV" ] || source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true
OUT="${OUT:-$HOME/structlift_datasets}"

# run_o0 name elf src ir_arg [extra_args]
# ir_arg: 单文件用 "file:PATH"，OpenSSL 逐文件目录用 "dir:BASE"（--llvm-ir-dir BASE/ir_o0）
run_o0() {
  local name="$1"
  local elf="$2"
  local src="$3"
  local ir_arg="$4"
  local extra_args="${5:-}"
  echo ""
  echo "===== $name O0 (pattern_labels 完善) ====="
  [ -f "$elf" ] || { echo "  跳过: $elf 不存在"; return 0; }
  local ir_params=""
  if [[ "$ir_arg" == file:* ]]; then
    local ll="${ir_arg#file:}"
    [ -f "$ll" ] && ir_params="--llvm-ir $ll" || echo "  警告: $ll 不存在，将仅用 DWARF"
  elif [[ "$ir_arg" == dir:* ]]; then
    local base="${ir_arg#dir:}"
    local irdir="${base}/ir_o0"
    [ -d "$irdir" ] && ir_params="--llvm-ir-dir $irdir" || echo "  警告: $irdir 不存在，将仅用 DWARF"
  fi
  python build_binskel_dataset_md.py \
    --elf "$elf" \
    --src "$src" \
    --out "$OUT/binskel_${name}_o0.jsonl" \
    --opt O0 \
    $ir_params $extra_args
  echo "  完成: $OUT/binskel_${name}_o0.jsonl"
  [ -f "$OUT/binskel_${name}_o0.jsonl" ] && wc -l < "$OUT/binskel_${name}_o0.jsonl" | xargs echo "  样本数:"
}

# 1. Zlib
ZDIR=~/struclift_wsl/datasets/zlib/zlib-1.3.1
if [ -f "$ZDIR/zlib_o0" ]; then
  run_o0 zlib "$ZDIR/zlib_o0" "$ZDIR" "file:$ZDIR/zlib_o0.ll"
else
  echo ""
  echo "===== Zlib 跳过 (未找到 $ZDIR/zlib_o0) ====="
fi

# 2. SQLite
SQLITE_DIR=$(ls -d ~/struclift_wsl/datasets/sqlite/sqlite-* 2>/dev/null | head -1)
if [ -n "$SQLITE_DIR" ] && [ -f "$SQLITE_DIR/sqlite_o0" ]; then
  run_o0 sqlite \
    "$SQLITE_DIR/sqlite_o0" \
    "$SQLITE_DIR" \
    "file:$SQLITE_DIR/sqlite_o0.ll"
else
  echo ""
  echo "===== SQLite 跳过 (未找到 $SQLITE_DIR/sqlite_o0) ====="
fi

# 2. BusyBox
BDIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
if [ -f "$BDIR/busybox_o0" ]; then
  run_o0 busybox \
    "$BDIR/busybox_o0" \
    "$BDIR" \
    "file:$BDIR/busybox_o0.ll"
else
  echo ""
  echo "===== BusyBox 跳过 (未找到 $BDIR/busybox_o0) ====="
fi

# 3. OpenSSL（compile 产出 ir_o0/ 逐文件目录，与 O1-O3 一致）
SDIR=~/struclift_wsl/datasets/openssl/openssl-3.3.1
if [ -f "$SDIR/openssl_o0" ]; then
  run_o0 openssl \
    "$SDIR/openssl_o0" \
    "$SDIR" \
    "dir:$SDIR"
else
  echo ""
  echo "===== OpenSSL 跳过 (未找到 $SDIR/openssl_o0) ====="
fi

echo ""
echo "===== 全部完成 ====="
for f in "$OUT"/binskel_zlib_o0.jsonl "$OUT"/binskel_sqlite_o0.jsonl "$OUT"/binskel_busybox_o0.jsonl "$OUT"/binskel_openssl_o0.jsonl; do
  [ -f "$f" ] && echo "  $(basename $f): $(wc -l < "$f") 样本"
done

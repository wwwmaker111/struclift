#!/bin/bash
# 四项目 zlib/curl/sqlite/busybox 的 O1–O3：完整流水线（DWARF+IR+tree-sitter），无 --skip-alignment
set -e
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate

SDIR=~/struclift_wsl/datasets/sqlite/sqlite-3450100
CDIR=~/struclift_wsl/datasets/curl/curl-8.7.1
ZDIR=~/struclift_wsl/datasets/zlib/zlib-1.3.1
BDIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
OUT=~/structlift_datasets

run_one() {
  local PROJ="$1"
  local ELF_BASE="$2"
  local SRC="$3"
  local IR_SPEC="$4"
  for OPT in 1 2 3; do
    ELF="${ELF_BASE}_o${OPT}"
    OUTF="$OUT/binskel_${PROJ}_o${OPT}.jsonl"
    if [ ! -f "$ELF" ]; then
      echo "跳过 $PROJ O${OPT}: $ELF 不存在"
      continue
    fi
    IR_ARGS=()
    if [[ "$IR_SPEC" == file:* ]]; then
      IB="${IR_SPEC#file:}"
      LL="${IB}_o${OPT}.ll"
      if [ -f "$LL" ]; then
        IR_ARGS=(--llvm-ir "$LL")
      else
        echo "  警告: 缺少 $LL，仅 DWARF"
      fi
    fi
    echo "===== $PROJ O${OPT}（完整流水线）====="
    time python build_binskel_dataset_md.py \
      --elf "$ELF" --src "$SRC" \
      "${IR_ARGS[@]}" \
      --out "$OUTF" --opt "O${OPT}"
    echo ""
  done
}

run_one sqlite  "$SDIR/sqlite"  "$SDIR" "file:$SDIR/sqlite"
run_one curl    "$CDIR/curl"    "$CDIR" "file:$CDIR/curl"
run_one zlib    "$ZDIR/zlib"    "$ZDIR" "file:$ZDIR/zlib"
run_one busybox "$BDIR/busybox" "$BDIR" "file:$BDIR/busybox"

echo "===== 验证 ====="
for f in "$OUT"/binskel_*_o[123].jsonl; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  python scripts/validate_alignment.py "$f"
  echo ""
done

echo "===== 全部完成 ====="

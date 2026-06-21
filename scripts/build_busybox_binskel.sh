#!/bin/bash
# BusyBox O0–O3：四档均为完整基本块→源码语句对齐（按 MD，DWARF+IR+tree-sitter），无 --skip-alignment
# 用法: build_busybox_binskel.sh <busybox_dir> <out_dir> [structlift_workdir]

set -e
BDIR="${1:?usage: $0 <busybox_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <busybox_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(dirname "$(dirname "$(realpath "$0")")")}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$BDIR" ] || { echo "busybox 目录不存在: $BDIR"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

for n in 0 1 2 3; do
  [ -f "$BDIR/busybox_o$n" ] || { echo "缺少 $BDIR/busybox_o$n，请先运行 scripts/compile_busybox.sh"; exit 1; }
done

cd "$WORKDIR"
[ -n "$VENV" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

for n in 0 1 2 3; do
  O="O$n"
  echo "=== BusyBox $O: 基本块→源码语句对齐（完整流水线，按 MD）==="
  LLVM_IR_ARG=()
  if [ -f "$BDIR/busybox_o$n.ll" ]; then
    LLVM_IR_ARG=(--llvm-ir "$BDIR/busybox_o$n.ll")
  else
    echo "  警告: 未找到 busybox_o$n.ll，该档将主要依赖 DWARF"
  fi
  python "$PY" \
    --elf   "$BDIR/busybox_o$n" \
    --src   "$BDIR" \
    --out   "$OUT/binskel_busybox_o$n.jsonl" \
    "${LLVM_IR_ARG[@]}" \
    --opt   "$O" \
    "${MD_EXTRA[@]}"
done

echo "=== 校验 O0–O3 ==="
for n in 0 1 2 3; do
  j="$OUT/binskel_busybox_o$n.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done
echo "完成: $OUT/binskel_busybox_o[0-3].jsonl"

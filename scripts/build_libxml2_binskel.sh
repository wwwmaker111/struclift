#!/bin/bash
# libxml2 O0-O3: 四档均为完整基本块->源码语句对齐（按 MD，DWARF+IR+tree-sitter），无 --skip-alignment
# 用法: build_libxml2_binskel.sh <libxml2_dir> <out_dir> [structlift_workdir]

set -e
XDIR="${1:?usage: $0 <libxml2_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <libxml2_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(dirname "$(dirname "$(realpath "$0")")")}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$XDIR" ] || { echo "libxml2 目录不存在: $XDIR"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

# xmllint 的主源码在 shell/xmllint.c；优先使用 shell/ 作为 --src，避免 DWARF 仅给基名时无法命中源码
SRC_DIR="$XDIR"
if [ -f "$XDIR/shell/xmllint.c" ]; then
  SRC_DIR="$XDIR/shell"
fi
[ -f "$SRC_DIR/xmllint.c" ] || {
  echo "错误: 未找到源码文件 $SRC_DIR/xmllint.c"
  echo "请先重新运行: bash scripts/compile_libxml2.sh <base_dir>  (确保产物目录包含源码树)"
  exit 1
}

for n in 0 1 2 3; do
  [ -f "$XDIR/xmllint_o$n" ] || { echo "缺少 $XDIR/xmllint_o$n，请先运行 scripts/compile_libxml2.sh"; exit 1; }
  if [ ! -d "$XDIR/ir_o$n" ]; then
    echo "警告: 未找到 $XDIR/ir_o$n，该档将主要依赖 DWARF，置信度可能偏低。"
  fi
done
# Os 可选：若未重新编译可跳过

cd "$WORKDIR"
[ -n "$VENV" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

for n in 0 1 2 3; do
  O="O$n"
  echo "=== $O: libxml2 基本块->源码语句对齐（完整流水线，按 MD）==="
  echo "  使用源码目录: $SRC_DIR"
  IR_ARG=()
  [ -d "$XDIR/ir_o$n" ] && IR_ARG=(--llvm-ir-dir "$XDIR/ir_o$n")
  python "$PY" \
    --elf "$XDIR/xmllint_o$n" \
    --src "$SRC_DIR" \
    --out "$OUT/binskel_libxml2_o$n.jsonl" \
    "${IR_ARG[@]}" \
    --opt "$O" \
    "${MD_EXTRA[@]}"
done

echo "=== 校验 O0-O3（validate_alignment）==="
for n in 0 1 2 3; do
  j="$OUT/binskel_libxml2_o$n.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成:"
for n in 0 1 2 3; do
  echo "  $OUT/binskel_libxml2_o$n.jsonl"
done

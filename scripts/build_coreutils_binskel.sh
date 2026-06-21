#!/usr/bin/env bash
# GNU coreutils（single-binary）O0–O3：DWARF + ir_oN/（逐文件 .ll）+ tree-sitter
# 需先: bash scripts/compile_coreutils.sh [base_dir]
# 用法: build_coreutils_binskel.sh <coreutils_src_dir> <out_dir> [structlift_workdir]
# 例:   build_coreutils_binskel.sh ~/struclift_wsl/datasets/coreutils/coreutils-9.4 ~/structlift_datasets /mnt/e/structlift

set -euo pipefail
CDIR="${1:?usage: $0 <coreutils_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <coreutils_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$CDIR" ] || { echo "coreutils 目录不存在: $CDIR"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

for n in 0 1 2 3; do
  [ -f "$CDIR/coreutils_o$n" ] || { echo "缺少 $CDIR/coreutils_o$n，请先运行 scripts/compile_coreutils.sh"; exit 1; }
done

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== coreutils O0 ====="
[ -d "$CDIR/ir_o0" ] || { echo "缺少 $CDIR/ir_o0"; exit 1; }
time python "$PY" \
  --elf "$CDIR/coreutils_o0" \
  --src "$CDIR" \
  --llvm-ir-dir "$CDIR/ir_o0" \
  --out "$OUT/binskel_coreutils_o0.jsonl" \
  --opt O0 \
  "${MD_EXTRA[@]}"
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_coreutils_o0.jsonl" || true

for OPT in 1 2 3; do
  echo ""
  echo "===== coreutils O$OPT ====="
  IR_ARG=()
  [ -d "$CDIR/ir_o$OPT" ] && IR_ARG=(--llvm-ir-dir "$CDIR/ir_o$OPT")
  [ ${#IR_ARG[@]} -eq 0 ] && echo "  警告: 缺少 $CDIR/ir_o$OPT，将主要依赖 DWARF"
  time python "$PY" \
    --elf "$CDIR/coreutils_o$OPT" \
    --src "$CDIR" \
    --out "$OUT/binskel_coreutils_o$OPT.jsonl" \
    --opt "O$OPT" \
    "${IR_ARG[@]}" \
    "${MD_EXTRA[@]}"
done

echo ""
echo "===== 校验 O1–O3 ====="
for OPT in 1 2 3; do
  j="$OUT/binskel_coreutils_o$OPT.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成: $OUT/binskel_coreutils_o*.jsonl"

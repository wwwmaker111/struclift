#!/usr/bin/env bash
# musl O0–O3：完整对齐（DWARF + ir_oN/ + tree-sitter），与 build_openssl_binskel.sh 相同用法。
# 用法: build_musl_binskel.sh <musl_src_dir> <out_dir> [structlift_workdir]
# 需先运行: bash scripts/compile_musl.sh

set -euo pipefail

SDIR="${1:?usage: $0 <musl_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <musl_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"

[ -d "$SDIR" ] || { echo "目录不存在: $SDIR"; exit 1; }
[ -f "$SDIR/musl_o0" ] || { echo "请先运行 scripts/compile_musl.sh"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== musl O0（完整流水线）====="
[ -d "$SDIR/ir_o0" ] || { echo "缺少 $SDIR/ir_o0"; exit 1; }
time python "$PY" \
  --elf "$SDIR/musl_o0" \
  --src "$SDIR" \
  --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_musl_o0.jsonl" \
  --opt O0

python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_musl_o0.jsonl" || true

for OPT in 1 2 3; do
  echo ""
  echo "===== musl O${OPT} ====="
  [ -f "$SDIR/musl_o${OPT}" ] || { echo "  跳过: 缺少 musl_o${OPT}"; continue; }
  IR_ARG=()
  [ -d "$SDIR/ir_o${OPT}" ] && IR_ARG=(--llvm-ir-dir "$SDIR/ir_o${OPT}")
  [ ${#IR_ARG[@]} -eq 0 ] && echo "  警告: 缺少 ir_o${OPT}，将主要依赖 DWARF"
  time python "$PY" \
    --elf "$SDIR/musl_o${OPT}" \
    --src "$SDIR" \
    --out "$OUT/binskel_musl_o${OPT}.jsonl" \
    --opt "O${OPT}" \
    "${IR_ARG[@]}"
done

echo ""
for OPT in 1 2 3; do
  j="$OUT/binskel_musl_o${OPT}.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成: $OUT/binskel_musl_o*.jsonl"

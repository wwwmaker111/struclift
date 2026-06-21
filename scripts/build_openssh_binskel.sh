#!/usr/bin/env bash
# OpenSSH portable（ssh 客户端）O0–O3：完整对齐（DWARF + ir_oN/ + tree-sitter）
# 用法: build_openssh_binskel.sh <openssh_src_dir> <out_dir> [structlift_workdir]
# 需先: bash scripts/compile_openssh.sh

set -euo pipefail

SDIR="${1:?usage: $0 <openssh_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <openssh_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$SDIR" ] || { echo "目录不存在: $SDIR"; exit 1; }
[ -f "$SDIR/openssh_o0" ] || { echo "请先运行 scripts/compile_openssh.sh"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== OpenSSH ssh O0（完整流水线）====="
[ -d "$SDIR/ir_o0" ] || { echo "缺少 $SDIR/ir_o0"; exit 1; }
time python "$PY" \
  --elf "$SDIR/openssh_o0" \
  --src "$SDIR" \
  --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_openssh_o0.jsonl" \
  --opt O0 \
  "${MD_EXTRA[@]}"

python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_openssh_o0.jsonl" || true

for OPT in 1 2 3; do
  echo ""
  echo "===== OpenSSH ssh O${OPT} ====="
  [ -f "$SDIR/openssh_o${OPT}" ] || { echo "  跳过: 缺少 openssh_o${OPT}"; continue; }
  IR_ARG=()
  [ -d "$SDIR/ir_o${OPT}" ] && IR_ARG=(--llvm-ir-dir "$SDIR/ir_o${OPT}")
  [ ${#IR_ARG[@]} -eq 0 ] && { echo "  错误: 缺少 $SDIR/ir_o${OPT}"; exit 1; }
  time python "$PY" \
    --elf "$SDIR/openssh_o${OPT}" \
    --src "$SDIR" \
    --out "$OUT/binskel_openssh_o${OPT}.jsonl" \
    --opt "O${OPT}" \
    "${IR_ARG[@]}" \
    "${MD_EXTRA[@]}"
done

echo ""
for OPT in 1 2 3; do
  j="$OUT/binskel_openssh_o${OPT}.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成: $OUT/binskel_openssh_o0.jsonl … binskel_openssh_o3.jsonl"

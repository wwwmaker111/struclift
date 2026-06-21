#!/bin/bash
# OpenSSL O0–O3：完整对齐（DWARF + ir_oN/ + tree-sitter）
# 用法: build_openssl_binskel.sh <openssl_dir> <out_dir> [structlift_workdir]

set -e
SDIR="${1:?usage: $0 <openssl_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <openssl_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(dirname "$(dirname "$(realpath "$0")")")}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$SDIR" ] || { echo "OpenSSL 目录不存在: $SDIR"; exit 1; }
[ -f "$SDIR/openssl_o0" ] || { echo "请先运行 scripts/compile_openssl.sh"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

cd "$WORKDIR"
[ -n "$VENV" ] && [ -f "$VENV" ] && source "$VENV" || true
# shellcheck source=/dev/null
source "$WORKDIR/scripts/_binskel_sft_augment.sh"
mkdir -p "$OUT"

echo "===== OpenSSL O0（完整流水线）====="
[ -d "$SDIR/ir_o0" ] || { echo "缺少 $SDIR/ir_o0"; exit 1; }
time python "$PY" \
  --elf "$SDIR/openssl_o0" \
  --src "$SDIR" \
  --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_openssl_o0.jsonl" \
  --opt O0 \
  "${MD_EXTRA[@]}"

binskel_augment_sft_jsonl "$OUT/binskel_openssl_o0.jsonl" "$SDIR"

echo ""
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_openssl_o0.jsonl" || true

for OPT in 1 2 3; do
  echo ""
  echo "===== OpenSSL O${OPT}（完整流水线）====="
  [ -f "$SDIR/openssl_o${OPT}" ] || { echo "  跳过: 缺少 openssl_o${OPT}"; continue; }
  IR_ARG=()
  [ -d "$SDIR/ir_o${OPT}" ] && IR_ARG=(--llvm-ir-dir "$SDIR/ir_o${OPT}")
  [ ${#IR_ARG[@]} -eq 0 ] && echo "  警告: 缺少 $SDIR/ir_o${OPT}，将主要依赖 DWARF"
  time python "$PY" \
    --elf "$SDIR/openssl_o${OPT}" \
    --src "$SDIR" \
    --out "$OUT/binskel_openssl_o${OPT}.jsonl" \
    --opt "O${OPT}" \
    "${IR_ARG[@]}" \
    "${MD_EXTRA[@]}"
  binskel_augment_sft_jsonl "$OUT/binskel_openssl_o${OPT}.jsonl" "$SDIR"
done

echo ""
echo "===== 校验 O1–O3 ====="
for OPT in 1 2 3; do
  j="$OUT/binskel_openssl_o${OPT}.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成: $OUT/binskel_openssl_o*.jsonl"

#!/usr/bin/env bash
set -e
XDIR="${1:?用法: $0 <libxml2_dir> <out_dir> [workdir]}"
OUT="${2:?}"
WORKDIR="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")
SRC_DIR="$XDIR"
[ -f "$XDIR/shell/xmllint.c" ] && SRC_DIR="$XDIR/shell"
[ -f "$XDIR/xmllint_os" ] || { echo "缺少 xmllint_os"; exit 1; }
IR_ARG=()
[ -d "$XDIR/ir_os" ] && IR_ARG=(--llvm-ir-dir "$XDIR/ir_os")
cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"
python "$PY" --elf "$XDIR/xmllint_os" --src "$SRC_DIR" --out "$OUT/binskel_libxml2_os.jsonl" \
  --opt Os "${IR_ARG[@]}" "${MD_EXTRA[@]}"
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_libxml2_os.jsonl" || true
echo "完成: $OUT/binskel_libxml2_os.jsonl"

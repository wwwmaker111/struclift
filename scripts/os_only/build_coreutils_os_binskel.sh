#!/usr/bin/env bash
set -euo pipefail
CDIR="${1:?用法: $0 <coreutils_dir> <out_dir> [workdir]}"
OUT="${2:?}"
WORKDIR="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")
[ -f "$CDIR/coreutils_os" ] || { echo "缺少 coreutils_os"; exit 1; }
IR_ARG=()
[ -d "$CDIR/ir_os" ] && IR_ARG=(--llvm-ir-dir "$CDIR/ir_os")
cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"
python "$PY" --elf "$CDIR/coreutils_os" --src "$CDIR" --out "$OUT/binskel_coreutils_os.jsonl" \
  --opt Os "${IR_ARG[@]}" "${MD_EXTRA[@]}"
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_coreutils_os.jsonl" || true
echo "完成: $OUT/binskel_coreutils_os.jsonl"

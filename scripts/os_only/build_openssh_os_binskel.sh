#!/usr/bin/env bash
set -euo pipefail
SDIR="${1:?用法: $0 <openssh_dir> <out_dir> [workdir]}"
OUT="${2:?}"
WORKDIR="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")
[ -f "$SDIR/openssh_os" ] || { echo "缺少 openssh_os"; exit 1; }
[ -d "$SDIR/ir_os" ] || { echo "缺少 ir_os"; exit 1; }
cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"
python "$PY" --elf "$SDIR/openssh_os" --src "$SDIR" --llvm-ir-dir "$SDIR/ir_os" \
  --out "$OUT/binskel_openssh_os.jsonl" --opt Os "${MD_EXTRA[@]}"
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_openssh_os.jsonl" || true
echo "完成: $OUT/binskel_openssh_os.jsonl"

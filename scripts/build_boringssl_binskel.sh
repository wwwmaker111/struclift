#!/usr/bin/env bash
# BoringSSL O0–O3: full alignment (DWARF + ir_oN/ + tree-sitter)
# Usage: build_boringssl_binskel.sh <boringssl_src_dir> <out_dir> [structlift_workdir]

set -eu
SDIR="${1:?usage: $0 <boringssl_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <boringssl_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"

[ -d "$SDIR" ] || { echo "Directory not found: $SDIR"; exit 1; }
[ -f "$SDIR/boringssl_o0" ] || { echo "Run scripts/compile_boringssl.sh first"; exit 1; }
[ -f "$PY" ] || { echo "Missing $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== BoringSSL O0 (full pipeline) ====="
[ -d "$SDIR/ir_o0" ] || { echo "Missing $SDIR/ir_o0"; exit 1; }
time python "$PY" \
  --elf "$SDIR/boringssl_o0" \
  --src "$SDIR" \
  --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_boringssl_o0.jsonl" \
  --opt O0

python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_boringssl_o0.jsonl" || true

for OPT in 1 2 3; do
  echo ""
  echo "===== BoringSSL O${OPT} (full pipeline) ====="
  [ -f "$SDIR/boringssl_o${OPT}" ] || { echo "  skip: missing boringssl_o${OPT}"; continue; }
  IR_ARG=()
  [ -d "$SDIR/ir_o${OPT}" ] && IR_ARG=(--llvm-ir-dir "$SDIR/ir_o${OPT}")
  [ ${#IR_ARG[@]} -eq 0 ] && echo "  warn: missing $SDIR/ir_o${OPT}, DWARF only"
  time python "$PY" \
    --elf "$SDIR/boringssl_o${OPT}" \
    --src "$SDIR" \
    --out "$OUT/binskel_boringssl_o${OPT}.jsonl" \
    --opt "O${OPT}" \
    "${IR_ARG[@]}"
done

echo ""
for OPT in 1 2 3; do
  j="$OUT/binskel_boringssl_o${OPT}.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "Done: $OUT/binskel_boringssl_o*.jsonl"

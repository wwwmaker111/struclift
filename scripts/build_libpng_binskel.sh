#!/usr/bin/env bash
# libpng O0-O3: DWARF + ir_oN/ + tree-sitter (same as openssl)
# Usage: build_libpng_binskel.sh <libpng_src_dir> <out_dir> [structlift_workdir]

set -eu
SDIR="${1:?usage: $0 <libpng_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <libpng_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"

[ -f "$SDIR/libpng_o0" ] || { echo "Run scripts/compile_libpng.sh first"; exit 1; }
[ -f "$PY" ] || { echo "Missing $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== libpng O0 ====="
[ -d "$SDIR/ir_o0" ] || { echo "Missing $SDIR/ir_o0"; exit 1; }
python "$PY" --elf "$SDIR/libpng_o0" --src "$SDIR" --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_libpng_o0.jsonl" --opt O0
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_libpng_o0.jsonl" || true

for n in 1 2 3; do
  echo "===== libpng O$n ====="
  [ -f "$SDIR/libpng_o$n" ] || continue
  IR=( )
  [ -d "$SDIR/ir_o$n" ] && IR=(--llvm-ir-dir "$SDIR/ir_o$n")
  python "$PY" --elf "$SDIR/libpng_o$n" --src "$SDIR" --out "$OUT/binskel_libpng_o$n.jsonl" \
    --opt "O$n" "${IR[@]}"
done
for n in 1 2 3; do
  [ -f "$OUT/binskel_libpng_o$n.jsonl" ] && python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_libpng_o$n.jsonl" || true
done
echo "Done: $OUT/binskel_libpng_o*.jsonl"

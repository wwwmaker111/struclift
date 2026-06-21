#!/usr/bin/env bash
# expat O0–O3: DWARF + ir_oN/ + tree-sitter（与 libpng 一致）
# Usage: build_expat_binskel.sh <expat_src_dir> <out_dir> [structlift_workdir]

set -eu
SDIR="${1:?usage: $0 <expat_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <expat_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"

[ -f "$SDIR/expat_o0" ] || { echo "Run scripts/compile_expat.sh first"; exit 1; }
[ -f "$PY" ] || { echo "Missing $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== expat O0 ====="
[ -d "$SDIR/ir_o0" ] || { echo "Missing $SDIR/ir_o0"; exit 1; }
python "$PY" --elf "$SDIR/expat_o0" --src "$SDIR" --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_expat_o0.jsonl" --opt O0
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_expat_o0.jsonl" || true

for n in 1 2 3; do
  echo "===== expat O$n ====="
  [ -f "$SDIR/expat_o$n" ] || continue
  IR=( )
  [ -d "$SDIR/ir_o$n" ] && IR=(--llvm-ir-dir "$SDIR/ir_o$n")
  python "$PY" --elf "$SDIR/expat_o$n" --src "$SDIR" --out "$OUT/binskel_expat_o$n.jsonl" \
    --opt "O$n" "${IR[@]}"
done
for n in 1 2 3; do
  [ -f "$OUT/binskel_expat_o$n.jsonl" ] && python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_expat_o$n.jsonl" || true
done
echo "Done: $OUT/binskel_expat_o*.jsonl"

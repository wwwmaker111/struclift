#!/usr/bin/env bash
# FFmpeg O0–O3: DWARF + ir_oN/ + tree-sitter
# Usage: build_ffmpeg_binskel.sh <ffmpeg_src_dir> <out_dir> [structlift_workdir]

set -eu
SDIR="${1:?usage: $0 <ffmpeg_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <ffmpeg_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -f "$SDIR/ffmpeg_o0" ] || { echo "请先运行: bash scripts/compile_ffmpeg.sh"; exit 1; }
[ -f "$PY" ] || { echo "Missing $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== ffmpeg O0 ====="
[ -d "$SDIR/ir_o0" ] || { echo "Missing $SDIR/ir_o0"; exit 1; }
echo "  ELF: $SDIR/ffmpeg_o0  IR: $SDIR/ir_o0"
python "$PY" --elf "$SDIR/ffmpeg_o0" --src "$SDIR" --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_ffmpeg_o0.jsonl" --opt O0 "${MD_EXTRA[@]}"
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_ffmpeg_o0.jsonl" || true

for n in 1 2 3; do
  echo "===== ffmpeg O$n ====="
  [ -f "$SDIR/ffmpeg_o$n" ] || { echo "  跳过: 缺少 $SDIR/ffmpeg_o$n"; continue; }
  IR=( )
  if [ -d "$SDIR/ir_o$n" ]; then
    IR=(--llvm-ir-dir "$SDIR/ir_o$n")
    echo "  ELF: $SDIR/ffmpeg_o$n  IR: $SDIR/ir_o$n"
  else
    echo "  警告: 无 $SDIR/ir_o$n，本档仅 DWARF（与 O0 完整流水线不一致）"
    echo "  ELF: $SDIR/ffmpeg_o$n"
  fi
  python "$PY" --elf "$SDIR/ffmpeg_o$n" --src "$SDIR" --out "$OUT/binskel_ffmpeg_o$n.jsonl" \
    --opt "O$n" "${IR[@]}" "${MD_EXTRA[@]}"
done
for n in 1 2 3; do
  [ -f "$OUT/binskel_ffmpeg_o$n.jsonl" ] && python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_ffmpeg_o$n.jsonl" || true
done
echo "Done: $OUT/binskel_ffmpeg_o*.jsonl"

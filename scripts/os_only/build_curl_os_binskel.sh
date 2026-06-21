#!/usr/bin/env bash
set -e
CDIR="${1:?用法: $0 <curl_dir> <out_dir> [workdir]}"
OUT="${2:?}"
WORKDIR="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")
[ -f "$CDIR/curl_os" ] && [ -f "$CDIR/curl_os.ll" ] || { echo "缺少 curl_os/curl_os.ll"; exit 1; }
cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"
python "$PY" --elf "$CDIR/curl_os" --src "$CDIR" --out "$OUT/binskel_curl_os.jsonl" \
  --llvm-ir "$CDIR/curl_os.ll" --opt Os "${MD_EXTRA[@]}"
python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_curl_os.jsonl" || true
echo "完成: $OUT/binskel_curl_os.jsonl"

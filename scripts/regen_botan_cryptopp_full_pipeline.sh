#!/usr/bin/env bash
# Botan + Crypto++ only: compile O0–O3 (ELF + ir_*) then binskel JSONL (full MD alignment).
# (BoringSSL is not part of this pipeline.)
#
# Usage (WSL/Linux):
#   cd /mnt/e/structlift
#   export OUT_DIR=~/structlift_datasets
#   bash scripts/regen_botan_cryptopp_full_pipeline.sh [datasets_parent]

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_PARENT="${1:-${HOME}/struclift_wsl/datasets}"
OUT_DIR="${OUT_DIR:-${HOME}/structlift_datasets}"

echo "========== 1/4 Botan compile =========="
bash "$ROOT/scripts/compile_botan.sh" "${BASE_PARENT}/botan"

echo "========== 2/4 Botan binskel =========="
BOTAN_DIR=$(find "${BASE_PARENT}/botan" -maxdepth 1 -type d -name 'Botan-*' | head -1)
[ -n "$BOTAN_DIR" ] || { echo "ERROR: Botan-* not found under ${BASE_PARENT}/botan"; exit 1; }
bash "$ROOT/scripts/build_botan_binskel.sh" "$BOTAN_DIR" "$OUT_DIR" "$ROOT"

echo "========== 3/4 Crypto++ compile =========="
bash "$ROOT/scripts/compile_cryptopp.sh" "${BASE_PARENT}/cryptopp"

echo "========== 4/4 Crypto++ binskel =========="
CRYPTOPP_DIR=$(find "${BASE_PARENT}/cryptopp" -maxdepth 1 -type d -name 'cryptopp-*' | head -1)
[ -n "$CRYPTOPP_DIR" ] || { echo "ERROR: cryptopp-* not found under ${BASE_PARENT}/cryptopp"; exit 1; }
bash "$ROOT/scripts/build_cryptopp_binskel.sh" "$CRYPTOPP_DIR" "$OUT_DIR" "$ROOT"

echo ""
echo "All done. JSONL under: $OUT_DIR"
ls -la "$OUT_DIR"/binskel_botan_o*.jsonl "$OUT_DIR"/binskel_cryptopp_o*.jsonl 2>/dev/null || true

python3 "$ROOT/scripts/binskel_dataset_stats.py" \
  "$OUT_DIR"/binskel_botan_o0.jsonl \
  "$OUT_DIR"/binskel_cryptopp_o0.jsonl 2>/dev/null || true

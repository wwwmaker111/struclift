#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true
OUT="${OUT:-$HOME/structlift_datasets}"
SDIR=~/struclift_wsl/datasets/openssl/openssl-3.3.1

echo "===== OpenSSL O0 ====="
[ -f "$SDIR/openssl_o0" ] || { echo "跳过: $SDIR/openssl_o0 不存在"; exit 0; }
python build_binskel_dataset_md.py \
  --elf "$SDIR/openssl_o0" \
  --src "$SDIR" \
  --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_openssl_o0.jsonl" \
  --opt O0
echo "完成: $OUT/binskel_openssl_o0.jsonl"
wc -l < "$OUT/binskel_openssl_o0.jsonl" | xargs echo "样本数:"

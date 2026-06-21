#!/bin/bash
# 导出多个样本到单个文件供人工核查
# 用法: ./export_batch_for_check.sh [output_file]

OUT="${1:-/tmp/manual_check_samples.txt}"
PY="$(dirname "$0")/export_sample_for_manual_check.py"
D="~/structlift_datasets"

echo "导出到 $OUT"
echo "======== 样本 1: sqlite3MisuseError ========" > "$OUT"
python3 "$PY" "$D/binskel_sqlite_o0.jsonl" sqlite3MisuseError >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "======== 样本 2: gz_reset (zlib) ========" >> "$OUT"
python3 "$PY" "$D/binskel_zlib_o0.jsonl" gz_reset >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "======== 样本 3: closeUnixFile (sqlite) ========" >> "$OUT"
python3 "$PY" "$D/binskel_sqlite_o0.jsonl" closeUnixFile >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "======== 样本 4: tftp_progress_init (busybox) ========" >> "$OUT"
python3 "$PY" "$D/binskel_busybox_o0.jsonl" tftp_progress_init >> "$OUT" 2>&1

echo "" >> "$OUT"
echo "======== 样本 5: cipher_generic_init_internal (openssl, index 0) ========" >> "$OUT"
python3 "$PY" "$D/binskel_openssl_o0.jsonl" --index 0 >> "$OUT" 2>&1

echo "完成: $OUT"
wc -l "$OUT"

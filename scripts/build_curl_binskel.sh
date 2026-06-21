#!/bin/bash
# curl O0–O3：四档均为完整基本块→源码语句对齐（MD：DWARF+IR+tree-sitter），无 --skip-alignment
# 实际行为由仓库根目录 build_binskel_dataset_md.py 决定（默认跳过行级回退、写 src_encoding_tier 等）。
#
# 用法: build_curl_binskel.sh <curl_dir> <out_dir> [structlift_workdir]
# 例:   build_curl_binskel.sh ~/struclift_wsl/datasets/curl/curl-8.7.1 ~/structlift_datasets /mnt/e/structlift

set -e
CDIR="${1:?usage: $0 <curl_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <curl_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(dirname "$(dirname "$(realpath "$0")")")}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$CDIR" ] || { echo "curl 目录不存在: $CDIR"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

for n in 0 1 2 3; do
  [ -f "$CDIR/curl_o$n" ] || { echo "缺少 $CDIR/curl_o$n，请先运行 scripts/compile_curl.sh"; exit 1; }
  if [ ! -f "$CDIR/curl_o$n.ll" ]; then
    echo "警告: 未找到 curl_o$n.ll，该档将主要依赖 DWARF，置信度可能偏低。"
  fi
done

cd "$WORKDIR"
[ -n "$VENV" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

for n in 0 1 2 3; do
  O="O$n"
  echo "=== $O: 基本块→源码语句对齐（完整流水线，按 MD）==="
  LLVM_IR_ARG=()
  if [ -f "$CDIR/curl_o$n.ll" ]; then
    LLVM_IR_ARG=(--llvm-ir "$CDIR/curl_o$n.ll")
  fi
  python "$PY" \
    --elf "$CDIR/curl_o$n" \
    --src "$CDIR" \
    --out "$OUT/binskel_curl_o$n.jsonl" \
    "${LLVM_IR_ARG[@]}" \
    --opt "$O" \
    "${MD_EXTRA[@]}"
done

echo "=== 校验 O0–O3（validate_alignment）==="
for n in 0 1 2 3; do
  j="$OUT/binskel_curl_o$n.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成:"
for n in 0 1 2 3; do
  echo "  $OUT/binskel_curl_o$n.jsonl"
done

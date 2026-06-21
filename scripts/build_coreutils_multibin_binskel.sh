#!/usr/bin/env bash
# 对 compile_coreutils_multibin.sh 产出的每个独立 ELF 跑 build_binskel，再按档位合并为一条 jsonl。
# 合并后的样本含不同 \"binary\" 字段；同一 func_name 可出现在多个程序中，训练时应用 (binary, func_name) 去重或兼留。
#
# 共 5 档：O0–O3 + Os → binskel_coreutils_o{0..3}_multibin.jsonl、binskel_coreutils_os_multibin.jsonl
#
# 用法:
#   bash scripts/build_coreutils_multibin_binskel.sh <coreutils_src_dir> <out_dir> [structlift_workdir]
# 例:
#   bash scripts/build_coreutils_multibin_binskel.sh \
#     ~/struclift_wsl/datasets/coreutils/coreutils-9.4 \
#     ~/structlift_datasets \
#     /mnt/e/structlift
#
set -euo pipefail
CDIR="${1:?usage: $0 <coreutils_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <coreutils_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$CDIR" ] || { echo "目录不存在: $CDIR"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
# shellcheck source=/dev/null
source "$WORKDIR/scripts/_binskel_sft_augment.sh"
mkdir -p "$OUT"

run_opt() {
  local n="$1"
  local opt_u="O$n"
  local bindir="$CDIR/multibin_o$n"
  local ir="$CDIR/ir_o$n"
  [ -d "$bindir" ] || { echo "缺少 $bindir，先运行 compile_coreutils_multibin.sh"; exit 1; }
  [ -d "$ir" ] || { echo "缺少 $ir"; exit 1; }

  local part="$OUT/.coreutils_multibin_parts_o$n"
  rm -rf "$part"
  mkdir -p "$part"

  shopt -s nullglob
  local bins=( "$bindir"/* )
  shopt -u nullglob
  if [ ${#bins[@]} -eq 0 ]; then
    echo "无 ELF: $bindir"
    exit 1
  fi

  echo ""
  echo "===== multibin $opt_u: ${#bins[@]} 个 ELF ====="
  local i=0
  for elf in "${bins[@]}"; do
    [ -f "$elf" ] || continue
    i=$((i+1))
    local bn
    bn=$(basename "$elf")
    echo "  ($i/${#bins[@]}) $bn"
    python "$PY" \
      --elf "$elf" \
      --src "$CDIR" \
      --llvm-ir-dir "$ir" \
      --out "$part/${bn}.jsonl" \
      --opt "$opt_u" \
      "${MD_EXTRA[@]}" || { echo "  警告: $bn 失败，跳过"; rm -f "$part/${bn}.jsonl"; continue; }
  done

  local merged="$OUT/binskel_coreutils_o${n}_multibin.jsonl"
  : > "$merged"
  for j in "$part"/*.jsonl; do
    [ -f "$j" ] || continue
    cat "$j" >> "$merged"
  done
  local lines
  lines=$(wc -l < "$merged" | tr -d ' ')
  echo ""
  echo "合并: $merged （共 $lines 行）"
  binskel_augment_sft_jsonl "$merged" "$CDIR"
  python "$WORKDIR/scripts/validate_alignment.py" "$merged" || true
}

run_multibin_os() {
  local bindir="$CDIR/multibin_os"
  local ir="$CDIR/ir_os"
  [ -d "$bindir" ] || { echo "缺少 $bindir，先运行 compile_coreutils_multibin.sh（含 Os）或 os_only/compile_os_coreutils_multibin.sh"; exit 1; }
  [ -d "$ir" ] || { echo "缺少 $ir"; exit 1; }

  local part="$OUT/.coreutils_multibin_parts_os"
  rm -rf "$part"
  mkdir -p "$part"

  shopt -s nullglob
  local bins=( "$bindir"/* )
  shopt -u nullglob
  if [ ${#bins[@]} -eq 0 ]; then
    echo "无 ELF: $bindir"
    exit 1
  fi

  echo ""
  echo "===== multibin Os: ${#bins[@]} 个 ELF ====="
  local i=0
  for elf in "${bins[@]}"; do
    [ -f "$elf" ] || continue
    i=$((i+1))
    local bn
    bn=$(basename "$elf")
    echo "  ($i/${#bins[@]}) $bn"
    python "$PY" \
      --elf "$elf" \
      --src "$CDIR" \
      --llvm-ir-dir "$ir" \
      --out "$part/${bn}.jsonl" \
      --opt Os \
      "${MD_EXTRA[@]}" || { echo "  警告: $bn 失败，跳过"; rm -f "$part/${bn}.jsonl"; continue; }
  done

  local merged="$OUT/binskel_coreutils_os_multibin.jsonl"
  : > "$merged"
  for j in "$part"/*.jsonl; do
    [ -f "$j" ] || continue
    cat "$j" >> "$merged"
  done
  local lines
  lines=$(wc -l < "$merged" | tr -d ' ')
  echo ""
  echo "合并: $merged （共 $lines 行）"
  binskel_augment_sft_jsonl "$merged" "$CDIR"
  python "$WORKDIR/scripts/validate_alignment.py" "$merged" || true
}

for n in 0 1 2 3; do
  run_opt "$n"
done
run_multibin_os

echo ""
echo "完成。输出:"
echo "  $OUT/binskel_coreutils_o0_multibin.jsonl … o3_multibin.jsonl"
echo "  $OUT/binskel_coreutils_os_multibin.jsonl"

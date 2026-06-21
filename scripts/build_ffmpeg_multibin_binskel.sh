#!/usr/bin/env bash
# FFmpeg multibin：对每个 multibin_* 内 ELF 跑 build_binskel，合并 5 份 jsonl。
#
# 用法:
#   bash scripts/build_ffmpeg_multibin_binskel.sh <ffmpeg_src_dir> <out_dir> [structlift_workdir]
#
# WSL + /mnt/e 若出现「line 7: pipefail」或「line 31: $'{\r'」：脚本被存成 CRLF。
#   sed -i 's/\r$//' scripts/build_ffmpeg_multibin_binskel.sh
# 或不经改文件直接跑（stdin 去 \r）:
#   bash <(sed 's/\r$//' scripts/build_ffmpeg_multibin_binskel.sh) <ffmpeg_dir> <out_dir> [workdir]
#
# 须 bash（数组、shopt）；勿用 sh。
if [ -z "${BASH_VERSION:-}" ]; then
  echo "请使用 bash 运行: bash $0 ..." >&2
  exit 1
fi
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

[ -d "$SDIR" ] || { echo "目录不存在: $SDIR"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

run_tier() {
  local tag="$1"
  local opt_arg="$2"
  local bindir="$SDIR/multibin_${tag}"
  local ir="$SDIR/ir_${tag}"
  [ -d "$bindir" ] || { echo "缺少 $bindir，先运行 scripts/compile_ffmpeg_multibin.sh"; exit 1; }
  [ -d "$ir" ] || { echo "缺少 $ir"; exit 1; }

  local part="$OUT/.ffmpeg_multibin_parts_el_${tag}"
  rm -rf "$part"
  mkdir -p "$part"

  shopt -s nullglob
  local bins=( "$bindir"/* )
  shopt -u nullglob
  [ ${#bins[@]} -gt 0 ] || { echo "无 ELF: $bindir"; exit 1; }

  echo ""
  echo "===== ffmpeg multibin $opt_arg (${#bins[@]} 个) ====="
  local i=0
  for elf in "${bins[@]}"; do
    [ -f "$elf" ] || continue
    i=$((i+1))
    local bn
    bn=$(basename "$elf")
    echo "  ($i/${#bins[@]}) $bn"
    python "$PY" \
      --elf "$elf" \
      --src "$SDIR" \
      --llvm-ir-dir "$ir" \
      --out "$part/${bn}.jsonl" \
      --opt "$opt_arg" \
      "${MD_EXTRA[@]}" || { echo "  警告: $bn 跳过"; rm -f "$part/${bn}.jsonl"; continue; }
  done

  local outj
  if [ "$tag" = "os" ]; then
    outj="$OUT/binskel_ffmpeg_os_multibin.jsonl"
  else
    outj="$OUT/binskel_ffmpeg_${tag}_multibin.jsonl"
  fi
  : > "$outj"
  for j in "$part"/*.jsonl; do
    [ -f "$j" ] || continue
    cat "$j" >> "$outj"
  done
  local lines
  lines=$(wc -l < "$outj" | tr -d ' ')
  echo "合并: $outj （$lines 行）"
  python "$WORKDIR/scripts/validate_alignment.py" "$outj" || true
}

run_tier "o0" "O0"
run_tier "o1" "O1"
run_tier "o2" "O2"
run_tier "o3" "O3"
run_tier "os" "Os"

echo ""
echo "完成: binskel_ffmpeg_o{0..3}_multibin.jsonl + binskel_ffmpeg_os_multibin.jsonl"

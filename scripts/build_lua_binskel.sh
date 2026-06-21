#!/usr/bin/env bash
# Lua O0–O3：完整对齐（DWARF + ir_oN/ + tree-sitter）
# 用法: build_lua_binskel.sh <lua_src_dir> <out_dir> [structlift_workdir]
# 需先: bash scripts/compile_lua.sh

set -euo pipefail

SDIR="${1:?usage: $0 <lua_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <lua_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="$WORKDIR/build_binskel_dataset_md.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"

[ -d "$SDIR" ] || { echo "目录不存在: $SDIR"; exit 1; }
[ -f "$SDIR/lua_o0" ] || { echo "请先运行 scripts/compile_lua.sh"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

cd "$WORKDIR"
[ -n "${VENV:-}" ] && [ -f "$VENV" ] && source "$VENV" || true
mkdir -p "$OUT"

echo "===== Lua O0（完整流水线：DWARF + IR + tree-sitter）====="
[ -d "$SDIR/ir_o0" ] || { echo "缺少 $SDIR/ir_o0"; exit 1; }
time python "$PY" \
  --elf "$SDIR/lua_o0" \
  --src "$SDIR" \
  --llvm-ir-dir "$SDIR/ir_o0" \
  --out "$OUT/binskel_lua_o0.jsonl" \
  --opt O0

python "$WORKDIR/scripts/validate_alignment.py" "$OUT/binskel_lua_o0.jsonl" || true

for OPT in 1 2 3; do
  echo ""
  echo "===== Lua O${OPT}（完整流水线）====="
  [ -f "$SDIR/lua_o${OPT}" ] || { echo "  跳过: 缺少 lua_o${OPT}"; continue; }
  IR_ARG=()
  [ -d "$SDIR/ir_o${OPT}" ] && IR_ARG=(--llvm-ir-dir "$SDIR/ir_o${OPT}")
  [ ${#IR_ARG[@]} -eq 0 ] && { echo "  错误: 缺少 $SDIR/ir_o${OPT}，完整对齐需要 IR 目录"; exit 1; }
  time python "$PY" \
    --elf "$SDIR/lua_o${OPT}" \
    --src "$SDIR" \
    --out "$OUT/binskel_lua_o${OPT}.jsonl" \
    --opt "O${OPT}" \
    "${IR_ARG[@]}"
done

echo ""
echo "===== 校验 O1–O3 ====="
for OPT in 1 2 3; do
  j="$OUT/binskel_lua_o${OPT}.jsonl"
  [ -f "$j" ] && python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done

echo "完成: $OUT/binskel_lua_o0.jsonl … binskel_lua_o3.jsonl"

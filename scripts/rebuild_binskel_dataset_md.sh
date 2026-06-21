#!/usr/bin/env bash
#
# 使用 build_binskel_dataset_md.py 完整重导 jsonl（含稳定 opcode/token id、src_edge_type）
#
# 必填环境变量：
#   ELF   - 带 DWARF 的 ELF 路径
#   SRC   - 源码树根目录（与构建该 ELF 时一致）
#   OUT   - 输出 .jsonl 路径
#
# 可选环境变量：
#   LLVM_IR      - 单个链接后的 .ll（与 --llvm-ir 一致）
#   LLVM_IR_DIR  - 按源文件拆分的 .ll 目录（与 --llvm-ir-dir 一致）
#   OPT          - 写入样本的优化档标签（默认 O0）
#   WORKERS      - 预解析源码并行度（默认 8）
#   NUM_OPCODES  - 与训练 VocabConfig.num_opcodes 一致（默认 1024）
#   SRC_VOCAB    - 与 SourceCFGEncoder.src_vocab_size 一致（默认 32000）
#   LIMIT        - 若设置正整数，等价于 --limit
#   EXTRA_ARGS   - 附加参数，例如 '--skip-alignment' 或 '--keep-line-fallback'
#
# 示例（BusyBox O0，WSL/Linux）：
#   export ELF=/path/to/busybox-1.36.1/busybox_o0
#   export SRC=/path/to/busybox-1.36.1
#   export OUT=/path/to/binskel_busybox_o0.jsonl
#   export LLVM_IR=/path/to/busybox_o0.ll   # 可选
#   bash scripts/rebuild_binskel_dataset_md.sh
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${ELF:?请设置 ELF（导出变量 ELF=...）}"
: "${SRC:?请设置 SRC（导出变量 SRC=...）}"
: "${OUT:?请设置 OUT（导出变量 OUT=...）}"

OPT="${OPT:-O0}"
WORKERS="${WORKERS:-8}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"

PY="$ROOT/build_binskel_dataset_md.py"
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }

CMD=(
  python "$PY"
  --elf "$ELF"
  --src "$SRC"
  --out "$OUT"
  --opt "$OPT"
  --workers "$WORKERS"
  --num-opcodes "$NUM_OPCODES"
  --src-vocab-size "$SRC_VOCAB"
)

if [[ -n "${LLVM_IR:-}" ]]; then
  CMD+=(--llvm-ir "$LLVM_IR")
fi
if [[ -n "${LLVM_IR_DIR:-}" ]]; then
  CMD+=(--llvm-ir-dir "$LLVM_IR_DIR")
fi
if [[ -n "${LIMIT:-}" && "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -gt 0 ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi

echo "词表: num_opcodes=$NUM_OPCODES src_vocab_size=$SRC_VOCAB"
echo "运行: ${CMD[*]}"
"${CMD[@]}"

echo "完成: $OUT"

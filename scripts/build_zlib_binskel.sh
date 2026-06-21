#!/bin/bash
# 在 zlib 已编译完成后，构建 binskel 数据集：O0/O1/O2/O3 均完整对齐(MD，含 IR + tree-sitter)
# 生成后默认再跑 augment_binskel_sft.py，jsonl 内含 sft_* / RL 字段（需 transformers）。
#   跳过 SFT：EMIT_SFT_JSONL=0 ； tokenizer / 序列长：SFT_TOKENIZER、SFT_MAX_SEQ_LEN（默认 16384）
# 用法: build_zlib_binskel.sh <zlib_dir> <out_dir> [structlift_workdir]
# 例:   build_zlib_binskel.sh /home/user/datasets/zlib/zlib-1.3.1 /home/user/structlift_datasets /mnt/e/structlift

set -e
ZDIR="${1:?usage: $0 <zlib_dir> <out_dir> [workdir]}"
OUT="${2:?usage: $0 <zlib_dir> <out_dir> [workdir]}"
WORKDIR="${3:-$(dirname "$(dirname "$(realpath "$0")")")}"
PY="$WORKDIR/build_binskel_dataset_md.py"
PRETTY_PY="$WORKDIR/scripts/jsonl_to_pretty_json.py"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")

[ -d "$ZDIR" ] || { echo "zlib 目录不存在: $ZDIR"; exit 1; }
[ -f "$ZDIR/zlib_o0" ] || { echo "请先运行 scripts/compile_zlib.sh 编译 zlib"; exit 1; }
[ -f "$PY" ] || { echo "未找到 $PY"; exit 1; }
[ -f "$PRETTY_PY" ] || { echo "未找到 $PRETTY_PY"; exit 1; }

cd "$WORKDIR"
[ -n "$VENV" ] && [ -f "$VENV" ] && source "$VENV" || true
# shellcheck source=/dev/null
source "$WORKDIR/scripts/_binskel_sft_augment.sh"
mkdir -p "$OUT"

# 若缺少 zlib_o0.ll 则自动生成（否则 O0 无 IR 交叉验证，置信度会很低）
if [ ! -f "$ZDIR/zlib_o0.ll" ]; then
  echo "未找到 zlib_o0.ll，正在生成（需 clang/llvm-link/llvm-dis）..."
  MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN -I."
  LIB_SRCS="adler32.c crc32.c deflate.c infback.c inffast.c inflate.c inftrees.c trees.c zutil.c compress.c uncompr.c gzclose.c gzlib.c gzread.c gzwrite.c"
  ( cd "$ZDIR" && \
    rm -f *.bc zlib_o0.bc 2>/dev/null; \
    for c in $LIB_SRCS; do clang $MD_CFLAGS -emit-llvm -c "$c" -o "${c%.c}.bc" || exit 1; done; \
    clang $MD_CFLAGS -emit-llvm -c -I. test/minigzip.c -o minigzip.bc && \
    llvm-link *.bc -o zlib_o0.bc && llvm-dis zlib_o0.bc -o zlib_o0.ll && \
    rm -f *.bc zlib_o0.bc 2>/dev/null; \
  ) && echo "  已生成 $ZDIR/zlib_o0.ll" || echo "  生成失败，O0 将仅用 DWARF（置信度偏低）"
fi

echo "=== O0: 基本块→源码语句对齐（按 MD，含 IR + tree-sitter）==="
python "$PY" \
  --elf   "$ZDIR/zlib_o0" \
  --src   "$ZDIR" \
  --out   "$OUT/binskel_zlib_o0.jsonl" \
  --llvm-ir "$ZDIR/zlib_o0.ll" \
  --opt   o0 \
  "${MD_EXTRA[@]}"
binskel_augment_sft_jsonl "$OUT/binskel_zlib_o0.jsonl" "$ZDIR"
python "$PRETTY_PY" "$OUT/binskel_zlib_o0.jsonl" "$OUT/binskel_zlib_o0.pretty.json"

echo "=== O1/O2/O3: 完整对齐（按 MD，含 IR + tree-sitter）==="
for opt in 1 2 3; do
  [ -f "$ZDIR/zlib_o$opt" ] || { echo "  跳过 O$opt: 缺少 zlib_o$opt"; continue; }
  [ -f "$ZDIR/zlib_o${opt}.ll" ] || { echo "  跳过 O$opt: 缺少 zlib_o${opt}.ll，请先运行 compile_zlib.sh"; continue; }
  echo "  O$opt ..."
  python "$PY" \
    --elf     "$ZDIR/zlib_o$opt" \
    --src     "$ZDIR" \
    --out     "$OUT/binskel_zlib_o$opt.jsonl" \
    --llvm-ir "$ZDIR/zlib_o${opt}.ll" \
    --opt     "o$opt" \
    "${MD_EXTRA[@]}"
  binskel_augment_sft_jsonl "$OUT/binskel_zlib_o$opt.jsonl" "$ZDIR"
  python "$PRETTY_PY" "$OUT/binskel_zlib_o$opt.jsonl" "$OUT/binskel_zlib_o$opt.pretty.json"
done

echo "=== 校验 O0–O3 对齐质量（validate_alignment）==="
for n in 0 1 2 3; do
  j="$OUT/binskel_zlib_o$n.jsonl"
  [ -f "$j" ] || continue
  echo ""
  echo "--- O$n ---"
  python "$WORKDIR/scripts/validate_alignment.py" "$j" || true
done
echo ""
echo "完成: $OUT/binskel_zlib_o*.jsonl"

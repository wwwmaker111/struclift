#!/bin/bash
# 构建 zlib、SQLite、BusyBox、OpenSSL 四个项目 O1/O2/O3 数据集（与 O0 相同：完整对齐 + IR）
# 用法: bash scripts/regen_o1o3_four_projects.sh
# 需先运行 compile_*_o1o3 或完整 compile_* 以生成 O1-O3 的 .ll / ir_oN/
set -e
cd "$(dirname "$0")/.."
[ -n "$VIRTUAL_ENV" ] || source ~/struclift_wsl/.venv/bin/activate 2>/dev/null || true
OUT="${OUT:-$HOME/structlift_datasets}"

# run_o1o3 name elf_base src [ir_spec]
# ir_spec: "file:BASE" 用 --llvm-ir BASE_oN.ll；"dir:BASE" 用 --llvm-ir-dir BASE/ir_oN/；空则无 IR
run_o1o3() {
  local name="$1"
  local elf_base="$2"
  local src="$3"
  local ir_spec="${4:-}"
  for opt in 1 2 3; do
    local elf="${elf_base}_o${opt}"
    local outfile="$OUT/binskel_${name}_o${opt}.jsonl"
    echo ""
    echo "===== $name O${opt} (完整对齐 + IR，同 O0) ====="
    [ -f "$elf" ] || { echo "  跳过: $elf 不存在"; continue; }
    local ir_args=""
    if [ -n "$ir_spec" ]; then
      if [[ "$ir_spec" == file:* ]]; then
        local base="${ir_spec#file:}"
        local ll="${base}_o${opt}.ll"
        if [ -f "$ll" ]; then
          ir_args="--llvm-ir $ll"
        else
          echo "  警告: $ll 不存在，将仅用 DWARF"
        fi
      elif [[ "$ir_spec" == dir:* ]]; then
        local base="${ir_spec#dir:}"
        local irdir="${base}/ir_o${opt}"
        if [ -d "$irdir" ]; then
          ir_args="--llvm-ir-dir $irdir"
        else
          echo "  警告: $irdir 不存在，将仅用 DWARF"
        fi
      fi
    fi
    python build_binskel_dataset_md.py \
      --elf "$elf" \
      --src "$src" \
      --out "$outfile" \
      --opt "O${opt}" \
      $ir_args
    echo "  完成: $outfile"
    [ -f "$outfile" ] && wc -l < "$outfile" | xargs echo "  样本数:"
  done
}

# 1. Zlib
ZDIR=~/struclift_wsl/datasets/zlib/zlib-1.3.1
if [ -f "$ZDIR/zlib_o1" ] || [ -f "$ZDIR/zlib_o2" ] || [ -f "$ZDIR/zlib_o3" ]; then
  run_o1o3 zlib "$ZDIR/zlib" "$ZDIR" "file:$ZDIR/zlib"
else
  echo ""
  echo "===== Zlib 跳过 (未找到 O1/O2/O3 二进制) ====="
fi

# 2. SQLite
SQLITE_DIR=$(ls -d ~/struclift_wsl/datasets/sqlite/sqlite-* 2>/dev/null | head -1)
if [ -n "$SQLITE_DIR" ] && { [ -f "$SQLITE_DIR/sqlite_o1" ] || [ -f "$SQLITE_DIR/sqlite_o2" ] || [ -f "$SQLITE_DIR/sqlite_o3" ]; }; then
  run_o1o3 sqlite "$SQLITE_DIR/sqlite" "$SQLITE_DIR" "file:$SQLITE_DIR/sqlite"
else
  echo ""
  echo "===== SQLite 跳过 (未找到 O1/O2/O3 二进制) ====="
fi

# 3. BusyBox
BDIR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
if [ -f "$BDIR/busybox_o1" ] || [ -f "$BDIR/busybox_o2" ] || [ -f "$BDIR/busybox_o3" ]; then
  run_o1o3 busybox "$BDIR/busybox" "$BDIR" "file:$BDIR/busybox"
else
  echo ""
  echo "===== BusyBox 跳过 (未找到 O1/O2/O3 二进制) ====="
fi

# 4. OpenSSL（使用 ir_oN/ 逐文件 IR 目录）
SDIR=~/struclift_wsl/datasets/openssl/openssl-3.3.1
if [ -f "$SDIR/openssl_o1" ] || [ -f "$SDIR/openssl_o2" ] || [ -f "$SDIR/openssl_o3" ]; then
  run_o1o3 openssl "$SDIR/openssl" "$SDIR" "dir:$SDIR"
else
  echo ""
  echo "===== OpenSSL 跳过 (未找到 O1/O2/O3 二进制) ====="
fi

echo ""
echo "===== 全部完成 ====="
for proj in zlib sqlite busybox openssl; do
  for opt in 1 2 3; do
    f="$OUT/binskel_${proj}_o${opt}.jsonl"
    [ -f "$f" ] && echo "  $(basename $f): $(wc -l < "$f") 样本"
  done
done

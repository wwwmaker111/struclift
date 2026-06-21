#!/bin/bash
# 五个项目 O0 重新生成（含列压缩、signal_stats、单信号乘数 0.80）
# 并生成 O1-O3（完整流水线：DWARF+IR+tree-sitter，与 O0 一致，无 --skip-alignment）
# 用法: bash run_all_regen.sh [--o0-only]
set -e
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate
OUT=~/structlift_datasets
O0_ONLY=false
[ "$1" = "--o0-only" ] && O0_ONLY=true

# ir_mode: file -> 第4参为 .ll 路径；dir -> 第4参为 ir_o0 目录（OpenSSL 无合并 .ll，仅有 ir_o0/）
run_o0() {
  local name="$1"
  local elf="$2"
  local src="$3"
  local ir_path="$4"
  local ir_mode="${5:-file}"
  echo ""
  echo "===== $name O0 ====="
  [ -f "$elf" ] || { echo "  跳过: $elf 不存在"; return 1; }
  local ir_args=()
  if [ "$ir_mode" = dir ]; then
    [ -d "$ir_path" ] || { echo "  跳过: IR 目录不存在 $ir_path"; return 1; }
    ir_args=(--llvm-ir-dir "$ir_path")
  else
    [ -f "$ir_path" ] || echo "  警告: 缺少 $ir_path，将主要依赖 DWARF"
    [ -f "$ir_path" ] && ir_args=(--llvm-ir "$ir_path")
  fi
  python build_binskel_dataset_md.py \
    --elf "$elf" \
    --src "$src" \
    "${ir_args[@]}" \
    --out "$OUT/binskel_${name}_o0.jsonl" \
    --opt O0
  echo "  验证:"
  python scripts/deep_validate.py "$OUT/binskel_${name}_o0.jsonl" --show 0 2>/dev/null | grep -E "样本数|BB 覆盖率|高置信度|DWARF-IR|顺序一致性"
}

# ir_spec: file:BASE -> --llvm-ir BASE_oN.ll；dir:BASE -> --llvm-ir-dir BASE/ir_oN/
run_o1o3() {
  local name="$1"
  local elf_base="$2"
  local src="$3"
  local ir_spec="${4:-}"
  echo ""
  echo "===== $name O1-O3（完整流水线）====="
  for opt in 1 2 3; do
    [ -f "${elf_base}_o${opt}" ] || { echo "  跳过 O${opt}: 不存在"; continue; }
    local ir_args=()
    if [ -n "$ir_spec" ]; then
      if [[ "$ir_spec" == file:* ]]; then
        local ibase="${ir_spec#file:}"
        local ll="${ibase}_o${opt}.ll"
        if [ -f "$ll" ]; then
          ir_args=(--llvm-ir "$ll")
        else
          echo "  警告 O${opt}: 缺少 $ll，仅 DWARF"
        fi
      elif [[ "$ir_spec" == dir:* ]]; then
        local ibase="${ir_spec#dir:}"
        local irdir="${ibase}/ir_o${opt}"
        if [ -d "$irdir" ]; then
          ir_args=(--llvm-ir-dir "$irdir")
        else
          echo "  警告 O${opt}: 缺少 $irdir，仅 DWARF"
        fi
      fi
    fi
    python build_binskel_dataset_md.py \
      --elf "${elf_base}_o${opt}" \
      --src "$src" \
      --out "$OUT/binskel_${name}_o${opt}.jsonl" \
      --opt "O${opt}" \
      "${ir_args[@]}"
  done
}

# 1. Zlib
run_o0 zlib \
  ~/struclift_wsl/datasets/zlib/zlib-1.3.1/zlib_o0 \
  ~/struclift_wsl/datasets/zlib/zlib-1.3.1 \
  ~/struclift_wsl/datasets/zlib/zlib-1.3.1/zlib_o0.ll

# 2. Curl
run_o0 curl \
  ~/struclift_wsl/datasets/curl/curl-8.7.1/curl_o0 \
  ~/struclift_wsl/datasets/curl/curl-8.7.1 \
  ~/struclift_wsl/datasets/curl/curl-8.7.1/curl_o0.ll

# 3. BusyBox
run_o0 busybox \
  ~/struclift_wsl/datasets/busybox/busybox-1.36.1/busybox_o0 \
  ~/struclift_wsl/datasets/busybox/busybox-1.36.1 \
  ~/struclift_wsl/datasets/busybox/busybox-1.36.1/busybox_o0.ll

# 4. SQLite (amalgamation，列压缩后体积会显著减小)
SQLITE_DIR=$(ls -d ~/struclift_wsl/datasets/sqlite/sqlite-* 2>/dev/null | head -1)
if [ -n "$SQLITE_DIR" ] && [ -f "$SQLITE_DIR/sqlite_o0" ]; then
  run_o0 sqlite \
    "$SQLITE_DIR/sqlite_o0" \
    "$SQLITE_DIR" \
    "$SQLITE_DIR/sqlite_o0.ll"
else
  echo ""
  echo "===== SQLite 跳过 (未找到编译产物) ====="
fi

# 5. OpenSSL（compile_openssl.sh 只产出 ir_o0/，无 openssl_o0.ll）
OR=~/struclift_wsl/datasets/openssl/openssl-3.3.1
run_o0 openssl "$OR/openssl_o0" "$OR" "$OR/ir_o0" dir

if [ "$O0_ONLY" = false ]; then
  ZR=~/struclift_wsl/datasets/zlib/zlib-1.3.1
  CR=~/struclift_wsl/datasets/curl/curl-8.7.1
  BR=~/struclift_wsl/datasets/busybox/busybox-1.36.1
  OR=~/struclift_wsl/datasets/openssl/openssl-3.3.1
  run_o1o3 zlib   "$ZR/zlib"     "$ZR" "file:$ZR/zlib"
  run_o1o3 curl   "$CR/curl"     "$CR" "file:$CR/curl"
  run_o1o3 busybox "$BR/busybox" "$BR" "file:$BR/busybox"
  [ -n "$SQLITE_DIR" ] && run_o1o3 sqlite "$SQLITE_DIR/sqlite" "$SQLITE_DIR" "file:$SQLITE_DIR/sqlite"
  run_o1o3 openssl "$OR/openssl" "$OR" "dir:$OR"
fi

echo ""
echo "===== 全部完成 ====="
ls -lh "$OUT"/binskel_*.jsonl 2>/dev/null | tail -20

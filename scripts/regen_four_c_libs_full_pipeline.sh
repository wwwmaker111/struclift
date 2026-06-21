#!/usr/bin/env bash
# 四个 C 库完整流水线：libpng / expat / pcre2 / mbedtls
# 每库：四档优化 O0–O3 → ELF + ir_oN/ → build_binskel_dataset_md.py → validate_alignment.py
#
# 用法（在 WSL/Linux 下，仓库根目录）:
#   export STRUCTLIFT_ROOT=/path/to/structlift   # 可选，默认为本脚本所在仓库根
#   export DATASET_ROOT=~/struclift_wsl/datasets
#   export BINSKEL_OUT=~/struclift_wsl/datasets/four_c_libs_binskel
#   bash scripts/regen_four_c_libs_full_pipeline.sh
#
# 可选版本环境变量: LIBPNG_VERSION EXPAT_VERSION PCRE2_VERSION MBEDTLS_VERSION
# 依赖: clang make cmake ninja autoconf automake libtool python3 wget|curl
#       tree-sitter 等与 build_binskel_dataset_md.py 相同（建议先 source venv）

set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_AUTO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# 若误设了不存在的 $HOME/structlift 等，避免用到空/错路径（会打成 /scripts/...）
if [ -z "${STRUCTLIFT_ROOT:-}" ] || [ ! -f "$STRUCTLIFT_ROOT/build_binskel_dataset_md.py" ]; then
  if [ -n "${STRUCTLIFT_ROOT:-}" ]; then
    echo "WARN: STRUCTLIFT_ROOT=$STRUCTLIFT_ROOT 无效（缺少 build_binskel_dataset_md.py），改用: $_AUTO_ROOT" >&2
  fi
  STRUCTLIFT_ROOT="$_AUTO_ROOT"
fi
DATASET_ROOT="${DATASET_ROOT:-${HOME}/struclift_wsl/datasets}"
BINSKEL_OUT="${BINSKEL_OUT:-${DATASET_ROOT}/four_c_libs_binskel}"

LIBPNG_V="${LIBPNG_VERSION:-1.6.47}"
EXPAT_V="${EXPAT_VERSION:-2.6.4}"
PCRE2_V="${PCRE2_VERSION:-10.44}"
MBEDTLS_V="${MBEDTLS_VERSION:-3.6.2}"

LIBPNG_DIR="${DATASET_ROOT}/libpng/libpng-${LIBPNG_V}"
EXPAT_DIR="${DATASET_ROOT}/expat/expat-${EXPAT_V}"
PCRE2_DIR="${DATASET_ROOT}/pcre2/pcre2-${PCRE2_V}"
MBEDTLS_DIR="${DATASET_ROOT}/mbedtls/mbedtls-${MBEDTLS_V}"

echo "========== 1/8 libpng: 编译 =========="
bash "$STRUCTLIFT_ROOT/scripts/compile_libpng.sh" "${DATASET_ROOT}/libpng"

echo "========== 2/8 libpng: binskel =========="
mkdir -p "${BINSKEL_OUT}/libpng"
bash "$STRUCTLIFT_ROOT/scripts/build_libpng_binskel.sh" "$LIBPNG_DIR" "${BINSKEL_OUT}/libpng" "$STRUCTLIFT_ROOT"

echo "========== 3/8 expat: 编译 =========="
bash "$STRUCTLIFT_ROOT/scripts/compile_expat.sh" "${DATASET_ROOT}/expat"

echo "========== 4/8 expat: binskel =========="
mkdir -p "${BINSKEL_OUT}/expat"
bash "$STRUCTLIFT_ROOT/scripts/build_expat_binskel.sh" "$EXPAT_DIR" "${BINSKEL_OUT}/expat" "$STRUCTLIFT_ROOT"

echo "========== 5/8 pcre2: 编译 =========="
bash "$STRUCTLIFT_ROOT/scripts/compile_pcre2.sh" "${DATASET_ROOT}/pcre2"

echo "========== 6/8 pcre2: binskel =========="
mkdir -p "${BINSKEL_OUT}/pcre2"
bash "$STRUCTLIFT_ROOT/scripts/build_pcre2_binskel.sh" "$PCRE2_DIR" "${BINSKEL_OUT}/pcre2" "$STRUCTLIFT_ROOT"

echo "========== 7/8 mbedtls: 编译 =========="
bash "$STRUCTLIFT_ROOT/scripts/compile_mbedtls.sh" "${DATASET_ROOT}/mbedtls"

echo "========== 8/8 mbedtls: binskel =========="
mkdir -p "${BINSKEL_OUT}/mbedtls"
bash "$STRUCTLIFT_ROOT/scripts/build_mbedtls_binskel.sh" "$MBEDTLS_DIR" "${BINSKEL_OUT}/mbedtls" "$STRUCTLIFT_ROOT"

echo "全部完成。"
echo "  源码与 ELF: 见 ${DATASET_ROOT}/{libpng,expat,pcre2,mbedtls}/"
echo "  JSONL: 见 ${BINSKEL_OUT}/*/binskel_*_o*.jsonl"

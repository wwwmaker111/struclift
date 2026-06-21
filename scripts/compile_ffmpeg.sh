#!/usr/bin/env bash
# FFmpeg: 静态 ffmpeg 可执行文件 + O0–O3 + compile_commands → ir_oN/（bear）
# Usage: compile_ffmpeg.sh [base_dir]
# Env:
#   FFMPEG_JOBS=16       — make 并行数（默认 = nproc，即「能看到的 CPU 线程数」）
#   FFMPEG_USE_CCACHE=1  — 用 ccache 包一层 clang（第二次起四档重编会快很多，需 apt install ccache）
#   FFMPEG_VERSION=7.1   FFMPEG_EXTRA_CONFIGURE_ARGS="..."
#
# 依赖: clang, make, python3, bear, wget|curl, xz
# 推荐: nasm（无则自动加 --disable-x86asm，体量略减但仍很大）
#
# 加速建议: ① export FFMPEG_JOBS=$(nproc) 或物理核数 ② WSL 下把源码放在 ~/ 的 ext4，避免只在 /mnt/e 上编
# 说明: 首次全量编译常需 30–90+ 分钟；bear 在 make 时生成 compile_commands.json 供 emit_ir。

set -eu
VER="${FFMPEG_VERSION:-7.1}"
BASE_DIR="${1:-${HOME}/struclift_wsl/datasets/ffmpeg}"
URL="https://ffmpeg.org/releases/ffmpeg-${VER}.tar.xz"
TARBALL="/tmp/ffmpeg-${VER}.tar.xz"
DIR="${BASE_DIR}/ffmpeg-${VER}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"
NPROC="${FFMPEG_JOBS:-$(nproc 2>/dev/null || echo 4)}"
# 子目录递归 make 也并行（FFmpeg 大量子 Makefile）
export MAKEFLAGS="-j${NPROC}"

_cc_clang() {
  if [ "${FFMPEG_USE_CCACHE:-0}" = 1 ] && command -v ccache >/dev/null 2>&1; then
    echo "ccache clang"
  else
    echo "clang"
  fi
}
_cxx_clang() {
  if [ "${FFMPEG_USE_CCACHE:-0}" = 1 ] && command -v ccache >/dev/null 2>&1; then
    echo "ccache clang++"
  else
    echo "clang++"
  fi
}

for c in clang make python3 bear xz; do
  command -v "$c" >/dev/null 2>&1 || { echo "ERROR: 需要安装 $c（如: sudo apt install bear nasm clang make xz-utils）"; exit 1; }
done
command -v wget >/dev/null 2>&1 || command -v curl >/dev/null 2>&1 || {
  echo "ERROR: 需要 wget 或 curl"; exit 1
}

mkdir -p "$BASE_DIR"
if [ ! -f "$TARBALL" ]; then
  wget -q -O "$TARBALL" "$URL" || curl -fsSL -o "$TARBALL" "$URL"
fi
if [ ! -d "$DIR" ]; then
  tar -xJf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local src_root="$1"
  local ir_dir="$2"
  local ccj="$src_root/compile_commands.json"
  [ -f "$ccj" ] || { echo "ERROR: 缺少 $ccj（bear 未生成？）"; exit 1; }
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$WORKDIR/scripts/emit_ir_from_compile_commands.py" "$src_root" "$ir_dir"
}

_asm_flags() {
  if [ "${FFMPEG_FORCE_NO_ASM:-0}" = 1 ]; then
    echo "--disable-x86asm --disable-inline-asm"
    return
  fi
  if command -v nasm >/dev/null 2>&1; then
    echo ""
  else
    echo "--disable-x86asm --disable-inline-asm"
  fi
}

build_one() {
  local tag="$1"
  # $2 保留与 MD_CFLAGS/O*_CFLAGS 对应，便于阅读；实际 -O/调试 以 tag 分支为准（见下）
  local _flags_doc="$2"
  local out="$3"
  echo "=== ffmpeg $tag ==="
  make distclean 2>/dev/null || true
  rm -f compile_commands.json

  # FFmpeg 的 configure 会写入 config.mak：默认 OPTFLAGS 常为 -O3，仅靠 export CFLAGS
  # 往往**不会**改变最终编译命令，导致 O0–O3 产出**逐字节相同**的二进制（md5 一致）。
  # 必须用 --optflags 指定 -O 档，其余调试/禁用优化选项用 --extra-cflags。
  local optflags extracf
  case "$tag" in
    o0)
      optflags="-O0"
      extracf="-g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
      ;;
    o1)
      optflags="-O1"
      extracf="-g3 -fno-inline -fno-unroll-loops"
      ;;
    o2)
      optflags="-O2"
      extracf="-g3 -fno-inline -fno-unroll-loops"
      ;;
    o3)
      optflags="-O3"
      extracf="-g3 -fno-inline -fno-unroll-loops"
      ;;
    *)
      echo "ERROR: unknown tag $tag"; exit 1
      ;;
  esac
  export CFLAGS="$optflags $extracf"
  export CXXFLAGS="$optflags $extracf"

  # 注意：不得在函数内使用 set --，否则会覆盖本函数的 $1/$2/$3
  # shellcheck disable=SC2086
  echo "configure + asm: optflags=$optflags extra-cflags=$extracf $(_asm_flags) ${FFMPEG_EXTRA_CONFIGURE_ARGS:-}"
  ./configure \
    --cc="$(_cc_clang)" \
    --cxx="$(_cxx_clang)" \
    --optflags="$optflags" \
    --extra-cflags="$extracf" \
    --disable-shared \
    --enable-static \
    --disable-stripping \
    --disable-doc \
    --disable-ffplay \
    --disable-ffprobe \
    --disable-autodetect \
    $(_asm_flags) \
    ${FFMPEG_EXTRA_CONFIGURE_ARGS:-} \
    || { echo "ERROR: configure 失败，可尝试 FFMPEG_FORCE_NO_ASM=1 或安装依赖后重试"; exit 1; }

  # 全量重编以便 bear 捕获所有编译命令
  bear -- make clean
  bear -- make -j"$NPROC" ffmpeg

  # 默认 make 会 strip ffmpeg，DWARF 被剥光 → binskel 里 dwarf 条数为 0。
  # 优先复制未 strip 的 ffmpeg_g；并已在 configure 加 --disable-stripping。
  _bin=""
  if [ -f ffmpeg_g ] && [ -s ffmpeg_g ]; then
    _bin="ffmpeg_g"
  elif [ -f ffmpeg ] && [ -s ffmpeg ]; then
    _bin="ffmpeg"
  fi
  [ -n "$_bin" ] || { echo "ERROR: 未生成 ffmpeg / ffmpeg_g"; exit 1; }
  cp -f "$_bin" "$DIR/$out"
  chmod +x "$DIR/$out"
  echo "    -> $DIR/$out"
  emit_ir "$DIR" "$DIR/ir_${tag}"
}

echo "[ffmpeg] 并行: make -j${NPROC}（改线程数: export FFMPEG_JOBS=32） USE_CCACHE=${FFMPEG_USE_CCACHE:-0}"
echo "[ffmpeg] 预计耗时较长；版本=${VER} 输出目录=${DIR}"
echo "[ffmpeg] O0..."
build_one "o0" "$MD_CFLAGS" "ffmpeg_o0"
echo "[ffmpeg] O1..."
build_one "o1" "$O1_CFLAGS" "ffmpeg_o1"
echo "[ffmpeg] O2..."
build_one "o2" "$O2_CFLAGS" "ffmpeg_o2"
echo "[ffmpeg] O3..."
build_one "o3" "$O3_CFLAGS" "ffmpeg_o3"
echo "Done: $DIR/ffmpeg_o0..o3, ir_o0..ir_o3"

#!/usr/bin/env bash
# FFmpeg：多产物 ffmpeg + ffprobe（可选 ffplay）× 五档 O0–O3 + Os。
# 单产物版见 compile_ffmpeg.sh（仅 ffmpeg、且默认关 ffprobe/ffplay）。
#
# 环境变量（与 compile_ffmpeg.sh 一致）:
#   FFMPEG_VERSION  FFMPEG_JOBS  FFMPEG_USE_CCACHE  FFMPEG_EXTRA_CONFIGURE_ARGS
#   FFMPEG_FORCE_NO_ASM=1  FFMPEG_MULTIBIN_FFPLAY=1  — 尝试编 ffplay（需 SDL 等）
#
# 用法:
#   bash scripts/compile_ffmpeg_multibin.sh [base_dir]
#
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
OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"

NPROC="${FFMPEG_JOBS:-$(nproc 2>/dev/null || echo 4)}"
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
  command -v "$c" >/dev/null 2>&1 || { echo "ERROR: 需要 $c"; exit 1; }
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
  [ -f "$ccj" ] || { echo "ERROR: 缺少 compile_commands.json"; exit 1; }
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

_ffplay_cfg() {
  if [ "${FFMPEG_MULTIBIN_FFPLAY:-0}" = 1 ]; then
    echo ""
  else
    echo "--disable-ffplay"
  fi
}

build_one() {
  local tag="$1"
  local outdir="$2"
  echo "=== ffmpeg multibin $tag ==="
  make distclean 2>/dev/null || true
  rm -f compile_commands.json

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
    os)
      optflags="-Os"
      extracf="-g3 -fno-inline -fno-unroll-loops"
      ;;
    *)
      echo "ERROR: tag $tag"; exit 1
      ;;
  esac
  export CFLAGS="$optflags $extracf"
  export CXXFLAGS="$optflags $extracf"

  # 开启 ffprobe；ffplay 可选（默认关，避免缺 SDL）
  # shellcheck disable=SC2046
  ./configure \
    --cc="$(_cc_clang)" \
    --cxx="$(_cxx_clang)" \
    --optflags="$optflags" \
    --extra-cflags="$extracf" \
    --disable-shared \
    --enable-static \
    --disable-stripping \
    --disable-doc \
    --disable-autodetect \
    $(_ffplay_cfg) \
    $(_asm_flags) \
    ${FFMPEG_EXTRA_CONFIGURE_ARGS:-} \
    || { echo "ERROR: configure 失败"; exit 1; }

  bear -- make clean
  local maket=("ffmpeg" "ffprobe")
  if [ "${FFMPEG_MULTIBIN_FFPLAY:-0}" = 1 ]; then
    maket+=("ffplay")
  fi
  bear -- make -j"$NPROC" "${maket[@]}"

  rm -rf "$DIR/$outdir"
  mkdir -p "$DIR/$outdir"

  _copy_g() {
    local base="$1"
    if [ -f "${base}_g" ] && [ -s "${base}_g" ]; then
      cp -f "${base}_g" "$DIR/$outdir/$base"
    elif [ -f "$base" ] && [ -s "$base" ]; then
      cp -f "$base" "$DIR/$outdir/$base"
    else
      echo "    警告: 无 $base"
      return 1
    fi
    chmod +x "$DIR/$outdir/$base"
    echo "    <- $outdir/$base"
    return 0
  }

  _copy_g ffmpeg
  _copy_g ffprobe || true
  if [ "${FFMPEG_MULTIBIN_FFPLAY:-0}" = 1 ]; then
    _copy_g ffplay || true
  fi

  emit_ir "$DIR" "$DIR/ir_${tag}"
}

echo "[ffmpeg multibin] 版本=$VER dir=$DIR ffplay=${FFMPEG_MULTIBIN_FFPLAY:-0}"
build_one "o0" "multibin_o0"
build_one "o1" "multibin_o1"
build_one "o2" "multibin_o2"
build_one "o3" "multibin_o3"
build_one "os" "multibin_os"

echo "Done: multibin_o0..o3 multibin_os + ir_o0..ir_os"

#!/usr/bin/env bash
# 仅在已有 FFmpeg 源码树中编译 -Os，产出 ffmpeg_os、ir_os/
# 用法: bash scripts/os_only/compile_os_ffmpeg.sh <ffmpeg源码根目录>
# 环境变量与 compile_ffmpeg.sh 相同：FFMPEG_JOBS、FFMPEG_USE_CCACHE、FFMPEG_EXTRA_CONFIGURE_ARGS 等
set -eu
DIR="${1:?用法: $0 <ffmpeg源码目录>}"
[ -f "$DIR/configure" ] || { echo "不是 FFmpeg 源码根: $DIR"; exit 1; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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

emit_ir() {
  local src_root="$1"
  local ir_dir="$2"
  local ccj="$src_root/compile_commands.json"
  [ -f "$ccj" ] || { echo "ERROR: 缺少 compile_commands.json"; exit 1; }
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$ROOT/scripts/emit_ir_from_compile_commands.py" "$src_root" "$ir_dir"
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

cd "$DIR"
echo "=== ffmpeg Os ==="
make distclean 2>/dev/null || true
rm -f compile_commands.json

optflags="-Os"
extracf="-g3 -fno-inline -fno-unroll-loops"
export CFLAGS="$optflags $extracf"
export CXXFLAGS="$optflags $extracf"

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
  || { echo "ERROR: configure 失败"; exit 1; }

bear -- make clean
bear -- make -j"$NPROC" ffmpeg

_bin=""
if [ -f ffmpeg_g ] && [ -s ffmpeg_g ]; then
  _bin="ffmpeg_g"
elif [ -f ffmpeg ] && [ -s ffmpeg ]; then
  _bin="ffmpeg"
fi
[ -n "$_bin" ] || { echo "ERROR: 未生成 ffmpeg"; exit 1; }
cp -f "$_bin" "$DIR/ffmpeg_os"
chmod +x "$DIR/ffmpeg_os"
echo "    -> $DIR/ffmpeg_os"

emit_ir "$DIR" "$DIR/ir_os"
echo "完成 Os: ffmpeg_os, ir_os/"

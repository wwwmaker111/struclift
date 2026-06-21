#!/usr/bin/env bash
# OpenSSH portable：下载、O0–O3 四档编译客户端 ssh + bear → compile_commands.json → ir_oN/（与 FFmpeg 一致）
# 产出（在 openssh-*p* 源码根目录）:
#   openssh_o0 … openssh_o3  — ssh 可执行文件（ELF + DWARF）
#   ir_o0 … ir_o3/           — emit_ir_from_compile_commands.py 生成的 .ll
#
# 依赖: clang make python3 bear wget|curl
# 系统库（Debian/Ubuntu 示例）: sudo apt install -y libssl-dev zlib1g-dev
# 可选: libpam0g-dev；若不想依赖 PAM，脚本已加 --without-pam
#
# 用法: bash scripts/compile_openssh.sh [base_dir]
# 例:   bash scripts/compile_openssh.sh ~/struclift_wsl/datasets/openssh

set -euo pipefail

OPENSSH_VERSION="${OPENSSH_VERSION:-9.8p1}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/openssh}"
DOWNLOAD_URL="https://cdn.openbsd.org/pub/OpenBSD/OpenSSH/portable/openssh-${OPENSSH_VERSION}.tar.gz"
TARBALL="/tmp/openssh-${OPENSSH_VERSION}.tar.gz"
DIR="${BASE_DIR}/openssh-${OPENSSH_VERSION}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
EMIT_IR="$WORKDIR/scripts/emit_ir_from_compile_commands.py"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"

NPROC=$(nproc 2>/dev/null || echo 4)

for x in clang make python3; do
  command -v "$x" >/dev/null 2>&1 || { echo "ERROR: 需要 $x"; exit 1; }
done
command -v bear >/dev/null 2>&1 || { echo "ERROR: 需要 bear（如: sudo apt install bear），用于生成 compile_commands.json"; exit 1; }
[ -f "$EMIT_IR" ] || { echo "ERROR: 缺少 $EMIT_IR"; exit 1; }

# OpenSSL：configure 会做「链接 -lcrypto」探测；仅有 openssl 命令行工具、没有 libssl-dev 时会报
#   working libcrypto not found
# 这里补全 CPPFLAGS/LDFLAGS，并在 Debian/Ubuntu multiarch 下加 -L/usr/lib/<triplet>。
_openssh_openssl_env() {
  OPENSSH_SSL_CPPFLAGS=""
  OPENSSH_SSL_LDFLAGS=""
  OPENSSH_WITH_SSL_DIR_ARG=()
  if [ -n "${OPENSSH_SSL_DIR:-}" ]; then
    OPENSSH_SSL_CPPFLAGS=""
    OPENSSH_SSL_LDFLAGS=""
    OPENSSH_WITH_SSL_DIR_ARG=(--with-ssl-dir="$OPENSSH_SSL_DIR")
    if [ ! -f "$OPENSSH_SSL_DIR/include/openssl/opensslv.h" ] && [ ! -f "$OPENSSH_SSL_DIR/openssl/opensslv.h" ]; then
      echo "ERROR: OPENSSH_SSL_DIR=$OPENSSH_SSL_DIR 下未找到 openssl/opensslv.h"
      exit 1
    fi
    return 0
  fi
  if [ ! -f /usr/include/openssl/opensslv.h ] && [ ! -f /usr/local/include/openssl/opensslv.h ]; then
    echo "ERROR: 未找到 openssl/opensslv.h。请安装开发包，例如:"
    echo "       sudo apt-get install -y libssl-dev"
    echo "  若 OpenSSL 装在非标准路径，可设置: export OPENSSH_SSL_DIR=/path/to/openssl/prefix"
    exit 1
  fi
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openssl 2>/dev/null; then
    OPENSSH_SSL_CPPFLAGS=$(pkg-config --cflags openssl 2>/dev/null || true)
    OPENSSH_SSL_LDFLAGS=$(pkg-config --libs-only-L openssl 2>/dev/null || true)
    _pfx=$(pkg-config --variable=prefix openssl 2>/dev/null || echo /usr)
    OPENSSH_WITH_SSL_DIR_ARG=(--with-ssl-dir="$_pfx")
  else
    OPENSSH_WITH_SSL_DIR_ARG=(--with-ssl-dir=/usr)
  fi
  if command -v dpkg-architecture >/dev/null 2>&1; then
    _m=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || true)
    if [ -n "$_m" ] && [ -d "/usr/lib/$_m" ]; then
      OPENSSH_SSL_LDFLAGS="-L/usr/lib/$_m $OPENSSH_SSL_LDFLAGS"
    fi
  fi
}
_openssh_openssl_env

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "[1/4] 下载 openssh-${OPENSSH_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -fsSL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/4] 解压 ..."
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local ir_dir="$DIR/ir_$1"
  local ccj="$DIR/compile_commands.json"
  [ -f "$ccj" ] || { echo "ERROR: 缺少 $ccj"; exit 1; }
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$EMIT_IR" "$DIR" "$ir_dir"
}

build_tier() {
  local tag="$1"
  local flags="$2"

  echo "  === openssh ${tag} ==="
  if [ -f Makefile ]; then
    make distclean 2>/dev/null || true
  fi
  rm -f "$DIR/compile_commands.json"

  echo "    configure ..."
  # 仅编 ssh 客户端，减小 bear 捕获范围；--without-pam 减少无 PAM 环境下的失败
  # shellcheck disable=SC2086
  CC=clang \
  CFLAGS="$flags" \
  CPPFLAGS="$OPENSSH_SSL_CPPFLAGS ${CPPFLAGS:-}" \
  LDFLAGS="$OPENSSH_SSL_LDFLAGS -g ${LDFLAGS:-}" \
    ./configure \
    --disable-strip \
    --without-pam \
    --without-selinux \
    --without-audit \
    "${OPENSSH_WITH_SSL_DIR_ARG[@]}" \
    ${OPENSSH_EXTRA_CONFIGURE:-} \
    || {
      echo "ERROR: configure 仍失败时请看源码目录下 config.log 末尾。"
      echo "  常见处理: sudo apt-get install -y libssl-dev zlib1g-dev"
      echo "  自编译 OpenSSL: export OPENSSH_SSL_DIR=/opt/openssl3 后重跑本脚本"
      exit 1
    }

  echo "    bear + make ssh ..."
  bear -- make clean
  bear -- make -j"$NPROC" ssh

  [ -f "$DIR/ssh" ] || { echo "ERROR: 未生成 $DIR/ssh"; exit 1; }
  cp -f "$DIR/ssh" "$DIR/openssh_${tag}"
  chmod +x "$DIR/openssh_${tag}"
  echo "    -> $DIR/openssh_${tag}"

  emit_ir "$tag"
}

echo "[3/4] O0 / O1 / O2 / O3 ..."
build_tier "o0" "$MD_CFLAGS"
build_tier "o1" "$O1_CFLAGS"
build_tier "o2" "$O2_CFLAGS"
build_tier "o3" "$O3_CFLAGS"

echo "[4/4] 完成"
echo "  ELF: $DIR/openssh_o0 … openssh_o3"
echo "  IR:  $DIR/ir_o0/ … ir_o3/"
echo "  源码根: $DIR （--src 指向此目录）"

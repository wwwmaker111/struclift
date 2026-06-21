#!/usr/bin/env bash
# 仅在已有 OpenSSH portable 源码树中编译 -Os，产出 openssh_os、ir_os/
# 用法: bash scripts/os_only/compile_os_openssh.sh <openssh源码根目录>
# 依赖: 与 compile_openssh.sh 相同（bear、libssl-dev 等）；可 export OPENSSH_SSL_DIR
set -euo pipefail
DIR="${1:?用法: $0 <openssh源码目录>}"
[ -f "$DIR/configure" ] || { echo "不是 openssh 源码根: $DIR"; exit 1; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EMIT_IR="$ROOT/scripts/emit_ir_from_compile_commands.py"
OS_CFLAGS="-Os -g3 -fno-inline -fno-unroll-loops"
NPROC=$(nproc 2>/dev/null || echo 4)

for x in clang make python3 bear; do
  command -v "$x" >/dev/null 2>&1 || { echo "ERROR: 需要 $x"; exit 1; }
done
[ -f "$EMIT_IR" ] || { echo "ERROR: 缺少 $EMIT_IR"; exit 1; }

_openssh_openssl_env() {
  OPENSSH_SSL_CPPFLAGS=""
  OPENSSH_SSL_LDFLAGS=""
  OPENSSH_WITH_SSL_DIR_ARG=()
  if [ -n "${OPENSSH_SSL_DIR:-}" ]; then
    OPENSSH_WITH_SSL_DIR_ARG=(--with-ssl-dir="$OPENSSH_SSL_DIR")
    if [ ! -f "$OPENSSH_SSL_DIR/include/openssl/opensslv.h" ] && [ ! -f "$OPENSSH_SSL_DIR/openssl/opensslv.h" ]; then
      echo "ERROR: OPENSSH_SSL_DIR 下未找到 openssl/opensslv.h"; exit 1
    fi
    return 0
  fi
  if [ ! -f /usr/include/openssl/opensslv.h ] && [ ! -f /usr/local/include/openssl/opensslv.h ]; then
    echo "ERROR: 请安装 libssl-dev 或设置 OPENSSH_SSL_DIR"; exit 1
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

emit_ir() {
  local ir_dir="$DIR/ir_$1"
  local ccj="$DIR/compile_commands.json"
  [ -f "$ccj" ] || { echo "ERROR: 缺少 compile_commands.json"; exit 1; }
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$EMIT_IR" "$DIR" "$ir_dir"
}

cd "$DIR"
echo "=== openssh Os ==="
if [ -f Makefile ]; then
  make distclean 2>/dev/null || true
fi
rm -f "$DIR/compile_commands.json"

CC=clang \
CFLAGS="$OS_CFLAGS" \
CPPFLAGS="$OPENSSH_SSL_CPPFLAGS ${CPPFLAGS:-}" \
LDFLAGS="$OPENSSH_SSL_LDFLAGS -g ${LDFLAGS:-}" \
  ./configure \
  --disable-strip \
  --without-pam \
  --without-selinux \
  --without-audit \
  "${OPENSSH_WITH_SSL_DIR_ARG[@]}" \
  ${OPENSSH_EXTRA_CONFIGURE:-} \
  || { echo "configure 失败，见 config.log"; exit 1; }

bear -- make clean
bear -- make -j"$NPROC" ssh

[ -f "$DIR/ssh" ] || { echo "ERROR: 未生成 ssh"; exit 1; }
cp -f "$DIR/ssh" "$DIR/openssh_os"
chmod +x "$DIR/openssh_os"
echo "    -> $DIR/openssh_os"

emit_ir "os"
echo "完成 Os: openssh_os, ir_os/"

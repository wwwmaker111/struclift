#!/usr/bin/env bash
# OpenSSH portable：多产物（ssh / sshd / scp / sftp / ssh-add）× 五档 O0–O3 + Os。
# 与 compile_openssh.sh 单 ssh 并行；产物目录 multibin_o0…multibin_o3、multibin_os + ir_o0…ir_os。
#
# 用法:
#   bash scripts/compile_openssh_multibin.sh [base_dir]
# 可选:
#   export OPENSSH_MULTIBIN_TARGETS="ssh sshd scp sftp ssh-add"
#   export OPENSSH_VERSION=9.8p1
#   export OPENSSH_SSL_DIR=/path/to/openssl
#
set -euo pipefail

OPENSSH_VERSION="${OPENSSH_VERSION:-9.8p1}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/openssh}"
DOWNLOAD_URL="https://cdn.openbsd.org/pub/OpenBSD/OpenSSH/portable/openssh-${OPENSSH_VERSION}.tar.gz"
TARBALL="/tmp/openssh-${OPENSSH_VERSION}.tar.gz"
DIR="${BASE_DIR}/openssh-${OPENSSH_VERSION}"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
EMIT_IR="$WORKDIR/scripts/emit_ir_from_compile_commands.py"

# 默认多产物（可用环境变量覆盖）
IFS=' ' read -r -a PROGS <<< "${OPENSSH_MULTIBIN_TARGETS:-ssh sshd scp sftp ssh-add}"

MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops"
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
      echo "ERROR: OPENSSH_SSL_DIR 无效"; exit 1
    fi
    return 0
  fi
  if [ ! -f /usr/include/openssl/opensslv.h ] && [ ! -f /usr/local/include/openssl/opensslv.h ]; then
    echo "ERROR: 安装 libssl-dev 或设置 OPENSSH_SSL_DIR"; exit 1
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
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -fsSL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir() {
  local tag="$1"
  local ir_dir="$DIR/ir_${tag}"
  local ccj="$DIR/compile_commands.json"
  [ -f "$ccj" ] || { echo "ERROR: 缺少 compile_commands.json"; exit 1; }
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"
  python3 "$EMIT_IR" "$DIR" "$ir_dir"
}

copy_multibin() {
  local tag="$1"
  local dest="$DIR/multibin_${tag}"
  rm -rf "$dest"
  mkdir -p "$dest"
  local n=0
  for p in "${PROGS[@]}"; do
    if [ -f "$DIR/$p" ] && [ -x "$DIR/$p" ]; then
      cp -f "$DIR/$p" "$dest/$p"
      chmod +x "$dest/$p"
      n=$((n+1))
    else
      echo "    警告: 未生成 $p（可检查 make 目标是否支持）"
    fi
  done
  echo "    multibin_${tag}: $n 个 ELF -> $dest"
}

build_tier() {
  local tag="$1"
  local flags="$2"

  echo "  === openssh multibin ${tag} ==="
  if [ -f Makefile ]; then
    make distclean 2>/dev/null || true
  fi
  rm -f "$DIR/compile_commands.json"

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
    || { echo "ERROR: configure 失败，见 config.log"; exit 1; }

  echo "    bear + make ${PROGS[*]} ..."
  bear -- make clean
  bear -- make -j"$NPROC" "${PROGS[@]}"

  copy_multibin "$tag"
  emit_ir "$tag"
}

echo "[openssh multibin] 程序: ${PROGS[*]}"
build_tier "o0" "$MD_CFLAGS"
build_tier "o1" "$O1_CFLAGS"
build_tier "o2" "$O2_CFLAGS"
build_tier "o3" "$O3_CFLAGS"
build_tier "os" "$OS_CFLAGS"

echo "[完成] $DIR"
echo "  multibin_o0..o3 multibin_os + ir_o0..ir_o3 ir_os"

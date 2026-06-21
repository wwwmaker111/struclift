#!/usr/bin/env bash
# musl libc：下载、解压、O0–O3 四档编译，与 zlib/OpenSSL 脚本同一套 CFLAGS 约定。
# 产出（均在 musl 源码目录下）:
#   musl_o0 … musl_o3  — 由 lib/libc.a 整体 reloc 合并成的单个 ELF(ET_REL)，含 DWARF，供 binskel
#   ir_o0 … ir_o3/     — 逐 .c 的 .ll（与 OpenSSL ir_oN/ 用法相同）
#
# 说明:
#   - musl 默认 Makefile 会加 -Os；必须在 ./configure 时传入含 -O0/-O1/… 的 CFLAGS，
#     configure 检测到 -O* 后会 optimize=no，不再给部分子目录单独 -O3（见官方 configure）。
#   - 合并 libc 使用: ld -r --whole-archive lib/libc.a（避免 ar x 文件过多）。
#   - IR 为独立 clang 编译，少数与 make 不完全一致的 .c 可能失败，脚本会计数跳过。
#
# 用法: bash scripts/compile_musl.sh [base_dir]
# 例:   bash scripts/compile_musl.sh ~/struclift_wsl/datasets/musl
# 依赖: clang, llvm 工具链, make, wget 或 curl, GNU ld（或兼容 --whole-archive 的链接器）

set -euo pipefail

MUSL_VERSION="${MUSL_VERSION:-1.2.5}"
BASE_DIR="${1:-$HOME/struclift_wsl/datasets/musl}"
DOWNLOAD_URL="https://musl.libc.org/releases/musl-${MUSL_VERSION}.tar.gz"
TARBALL="/tmp/musl-${MUSL_VERSION}.tar.gz"
DIR="${BASE_DIR}/musl-${MUSL_VERSION}"

# O1–O3 也带 -fstandalone-debug（与 O0 一致）：Clang 在 -O1+ 下若不加，常见调试节不完整或与 reloc 合并后 readelf 看不到 .debug_line。
_MUSL_DEBUG_EXTRA="-fstandalone-debug"
MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
O1_CFLAGS="-O1 -g3 -fno-inline -fno-unroll-loops $_MUSL_DEBUG_EXTRA"
O2_CFLAGS="-O2 -g3 -fno-inline -fno-unroll-loops $_MUSL_DEBUG_EXTRA"
O3_CFLAGS="-O3 -g3 -fno-inline -fno-unroll-loops $_MUSL_DEBUG_EXTRA"

NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [ ! -f "$TARBALL" ]; then
  echo "[1/5] 下载 musl-${MUSL_VERSION} ..."
  wget -q -O "$TARBALL" "$DOWNLOAD_URL" || curl -fsSL -o "$TARBALL" "$DOWNLOAD_URL"
fi
if [ ! -d "$DIR" ]; then
  echo "[2/5] 解压 ..."
  tar -xzf "$TARBALL" -C "$BASE_DIR"
fi
cd "$DIR"

emit_ir_dir() {
  local flags="$1"
  local tag="$2"
  local ir_dir="$DIR/ir_${tag}"
  rm -rf "$ir_dir"
  mkdir -p "$ir_dir"

  local arch
  arch=$(sed -n 's/^ARCH = //p' config.mak 2>/dev/null | tr -d ' ' | head -1)
  if [ -z "$arch" ]; then
    echo "    警告: 无法从 config.mak 读取 ARCH，跳过 IR"
    return 0
  fi

  # 与 musl Makefile 中 CFLAGS_ALL 的 -I 顺序尽量一致（缺省路径不存在则略过）
  local inc=(
    "-D_XOPEN_SOURCE=700"
    "-I$DIR/arch/$arch"
    "-I$DIR/arch/generic"
  )
  [ -d "$DIR/obj/include" ] && inc+=("-I$DIR/obj/include")
  [ -d "$DIR/obj/src/internal" ] && inc+=("-I$DIR/obj/src/internal")
  inc+=(
    "-I$DIR/src/include"
    "-I$DIR/src/internal"
    "-I$DIR/include"
  )

  local n_ok=0 n_fail=0
  local cfile bn out_ll

  while IFS= read -r -d '' cfile; do
    bn=$(echo "$cfile" | sed "s|^$DIR/||" | tr '/' '_')
    bn="${bn%.c}.ll"
    out_ll="$ir_dir/$bn"
    if clang $flags "${inc[@]}" -std=c99 -ffreestanding -nostdinc -w \
      -emit-llvm -S "$cfile" -o "$out_ll" 2>/dev/null; then
      n_ok=$((n_ok + 1))
    else
      n_fail=$((n_fail + 1))
    fi
  done < <(
    find "$DIR/src" "$DIR/crt" "$DIR/ldso" \
      "$DIR/arch/$arch" "$DIR/arch/generic" \
      -name '*.c' -print0 2>/dev/null
  )

  echo "    IR $ir_dir: 成功 $n_ok, 失败 $n_fail"
}

merge_libc_reloc() {
  local out_elf="$1"
  if ld -r -o "$out_elf" --whole-archive "$DIR/lib/libc.a" --no-whole-archive 2>/dev/null; then
    return 0
  fi
  echo "    警告: ld -r --whole-archive 失败，尝试 ar x + clang -r ..."
  local tmp
  tmp=$(mktemp -d)
  (
    cd "$tmp"
    ar x "$DIR/lib/libc.a"
    objs=$(find . -name '*.o' -print | tr '\n' ' ')
    # shellcheck disable=SC2086
    clang -nostdlib -r $objs -o "$out_elf"
  )
  rm -rf "$tmp"
}

compile_musl_tier() {
  local tag="$1"
  local flags="$2"

  echo "  === musl $tag ==="
  echo "    清理 ..."
  make distclean 2>/dev/null || true
  rm -rf obj lib 2>/dev/null || true

  echo "    configure (CC=clang, CFLAGS 含 -O*) ..."
  # --disable-warnings: 减少 clang 下 -Werror 类失败
  CC=clang ./configure CFLAGS="$flags" --disable-warnings --prefix="$DIR/install_${tag}"

  echo "    make -j$NPROC ..."
  make -j"$NPROC"

  if [ ! -f "$DIR/lib/libc.a" ]; then
    echo "    错误: 未生成 lib/libc.a"
    exit 1
  fi

  echo "    reloc 合并 -> musl_${tag} ..."
  merge_libc_reloc "$DIR/musl_${tag}"
  echo "    -> $DIR/musl_${tag} ($(du -h "$DIR/musl_${tag}" | cut -f1))"

  emit_ir_dir "$flags" "$tag"
}

echo "[3/5] O0 (按 MD 约定) ..."
compile_musl_tier "o0" "$MD_CFLAGS"

echo "[4/5] O1 / O2 / O3 ..."
for opt in 1 2 3; do
  case $opt in
    1) F="$O1_CFLAGS"; N="o1" ;;
    2) F="$O2_CFLAGS"; N="o2" ;;
    3) F="$O3_CFLAGS"; N="o3" ;;
  esac
  compile_musl_tier "$N" "$F"
done

echo "[5/5] 完成"
echo "  ELF:  $DIR/musl_o0 … musl_o3（ET_REL，由 libc.a 合并）"
echo "  IR:   $DIR/ir_o0/ … ir_o3/"
echo "  源码: $DIR （--src 指向此目录）"
if command -v readelf >/dev/null 2>&1; then
  echo "  DWARF 自检（.debug_line / .zdebug_line 等）:"
  for b in musl_o0 musl_o1 musl_o2 musl_o3; do
    [ -f "$DIR/$b" ] || continue
    if readelf -S "$DIR/$b" 2>/dev/null | grep -qE '\.(z)?debug_(line|info|abbrev|str)'; then
      echo "    OK $b"
    else
      echo "    FAIL $b 未见常见 .debug_* / .zdebug_* 节（若已加 -g 仍失败请看 config.log / 换 ld.bfd）"
    fi
  done
fi

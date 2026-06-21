# O0 编译优化等级 — 所有命令行

按 BB_Source_Alignment_Pipeline.md 的 O0 要求，汇总 zlib 与 curl 的**完整 O0 构建命令行**（含生成可执行文件与 `.ll`）。

---

## 1. 通用 O0 编译选项（MD 一致）

```bash
MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
```

- **zlib**：所有 O0 编译均使用上述 `MD_CFLAGS`。
- **curl**：可执行文件 O0 用 `MD_CFLAGS`；生成 `.bc`/.ll 时在此基础上增加宏（见下）。

---

## 2. zlib O0

假设已进入 zlib 源码目录：`cd /path/to/zlib-1.3.1`。

### 2.1 配置

```bash
export CC=clang
./configure
```

### 2.2 生成可执行文件 zlib_o0（make）

```bash
export CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
make clean
make minigzip
cp -f minigzip zlib_o0
```

make 内部对每个 `.c` 的等效命令（由 Makefile 决定，此处为典型形式）：

```bash
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -c -I. <file.c> -o <file.o>
# 最后链接
clang ... -o minigzip
```

### 2.3 生成 zlib_o0.ll（逐文件 .bc → link → dis）

库源列表（与 Makefile 一致）：

```bash
LIB_SRCS="adler32.c crc32.c deflate.c infback.c inffast.c inflate.c inftrees.c trees.c zutil.c compress.c uncompr.c gzclose.c gzlib.c gzread.c gzwrite.c"
MINIGZIP_SRC="test/minigzip.c"
```

对每个库源编译为 `.bc`：

```bash
for c in $LIB_SRCS; do
  clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -emit-llvm -c -I. "$c" -o "${c%.c}.bc"
done
```

minigzip 单独编一个 `.bc`：

```bash
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -emit-llvm -c -I. test/minigzip.c -o minigzip.bc
```

链接并生成 `.ll`：

```bash
llvm-link *.bc -o zlib_o0.bc
llvm-dis zlib_o0.bc -o zlib_o0.ll
rm -f *.bc zlib_o0.bc
```

**单条形式（库 + minigzip）**：

```bash
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -emit-llvm -c -I. adler32.c -o adler32.bc
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -emit-llvm -c -I. crc32.c -o crc32.bc
# ... 其余 LIB_SRCS 同理 ...
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug -emit-llvm -c -I. test/minigzip.c -o minigzip.bc
llvm-link *.bc -o zlib_o0.bc
llvm-dis zlib_o0.bc -o zlib_o0.ll
```

---

## 3. curl O0

假设已进入 curl 源码目录：`cd /path/to/curl-8.7.1`。

### 3.1 配置

```bash
export CC=clang
./configure --disable-shared --without-ssl --without-zlib --disable-ldap --disable-manual
# 若失败可尝试: ./configure --disable-shared
```

### 3.2 生成可执行文件 curl_o0（make）

```bash
export CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
make clean
make
cp -f src/curl curl_o0
```

make 内部对每个 `.c` 的等效命令由 Makefile + configure 决定，形式类似：

```bash
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug <configure 的 -I/-D> -c <file.c> -o <file.o>
# 最后链接得到 src/curl
```

### 3.3 生成 curl_o0.ll（lib/*.c + src/*.c → .bc → link → dis）

公共变量：

```bash
MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
INC="-I. -Iinclude -Ilib -Isrc"
BC_CFLAGS="$MD_CFLAGS -DHAVE_CONFIG_H"
BC_LIB_CFLAGS="$BC_CFLAGS -DBUILDING_LIBCURL -DSTRCONST(x)=(x,(size_t)(sizeof(x)-1))"
BC_DIR=".bc_o0"   # 可改为任意临时目录
mkdir -p "$BC_DIR"
```

**lib 下每个 .c → .bc**（必须用 BC_LIB_CFLAGS）：

```bash
for c in lib/*.c; do
  [ -f "$c" ] || continue
  bn=$(echo "$c" | tr '/' '_' | sed 's/\.c$/.bc/')
  clang $BC_LIB_CFLAGS $INC -emit-llvm -c "$c" -o "$BC_DIR/$bn"
done
```

展开为单条示例（lib 中一个文件）：

```bash
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug \
  -DHAVE_CONFIG_H -DBUILDING_LIBCURL -DSTRCONST(x)=(x,(size_t)(sizeof(x)-1)) \
  -I. -Iinclude -Ilib -Isrc -emit-llvm -c lib/mime.c -o "$BC_DIR/lib_mime.bc"
```

**src 下每个 .c → .bc**（用 BC_CFLAGS，不要 BUILDING_LIBCURL）：

```bash
for c in src/*.c; do
  [ -f "$c" ] || continue
  bn=$(echo "$c" | tr '/' '_' | sed 's/\.c$/.bc/')
  clang $BC_CFLAGS $INC -emit-llvm -c "$c" -o "$BC_DIR/$bn"
done
```

展开为单条示例：

```bash
clang -O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug \
  -DHAVE_CONFIG_H -I. -Iinclude -Ilib -Isrc -emit-llvm -c src/tool_main.c -o "$BC_DIR/src_tool_main.bc"
```

**链接并生成 .ll**：

```bash
llvm-link "$BC_DIR"/*.bc -o curl_o0.bc
llvm-dis curl_o0.bc -o curl_o0.ll
rm -rf "$BC_DIR" curl_o0.bc
```

---

## 4. 汇总表

| 项目 | 可执行文件 O0 | O0 .ll 生成 |
|------|----------------|-------------|
| **zlib** | `CC=clang CFLAGS="$MD_CFLAGS" make minigzip` | 对 `$LIB_SRCS` 和 `test/minigzip.c` 用 `clang $MD_CFLAGS -emit-llvm -c -I. <file> -o <file.bc>`，再 `llvm-link *.bc -o zlib_o0.bc`，`llvm-dis zlib_o0.bc -o zlib_o0.ll` |
| **curl** | `CC=clang CFLAGS="$MD_CFLAGS" make`，产物 `src/curl` → `curl_o0` | lib: `clang $BC_LIB_CFLAGS $INC -emit-llvm -c`；src: `clang $BC_CFLAGS $INC -emit-llvm -c`；再 `llvm-link`、`llvm-dis` 得到 `curl_o0.ll` |

**O0 必须的 CFLAGS（与 MD 一致）**：

```text
-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug
```

curl 生成 .ll 时额外需要：`-DHAVE_CONFIG_H`；对 lib 还需 `-DBUILDING_LIBCURL` 与 `-DSTRCONST(x)=(x,(size_t)(sizeof(x)-1))`（-D 的宏名不能带引号），以及 `-I. -Iinclude -Ilib -Isrc`。

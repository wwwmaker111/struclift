# 仅 curl O0 构建 — 所有命令行（在 curl 源码目录执行）

**必须先进入 curl 源码目录**（那里才有 `lib/`、`src/`），再执行下面命令。  
把你的 curl 目录设为 `CDIR`，然后 `cd "$CDIR"`。

---

## 一键复制：从进入目录到生成 curl_o0 + curl_o0.ll

```bash
# 1) 进入 curl 源码目录（必做，否则 lib/*.c 找不到）
CDIR=/home/wuqiongmin/struclift_wsl/datasets/curl/curl-8.7.1
cd "$CDIR"

# 2) 配置（若已 configure 过可跳过）
export CC=clang
./configure --disable-shared --without-ssl --without-zlib --disable-ldap --disable-manual

# 3) 生成可执行文件 curl_o0
export CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
make clean
make
cp -f src/curl curl_o0

# 4) 生成 curl_o0.ll：变量 + lib/*.c → .bc
MD_CFLAGS="-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug"
INC="-I. -Iinclude -Ilib -Isrc"
BC_CFLAGS="$MD_CFLAGS -DHAVE_CONFIG_H"
BC_LIB_CFLAGS="$BC_CFLAGS -DBUILDING_LIBCURL -DSTRCONST(x)=(x,(size_t)(sizeof(x)-1))"
BC_DIR=".bc_o0"
mkdir -p "$BC_DIR"
rm -f "$BC_DIR"/*.bc

for c in lib/*.c; do
  [ -f "$c" ] || continue
  bn=$(echo "$c" | tr '/' '_' | sed 's/\.c$/.bc/')
  clang $BC_LIB_CFLAGS $INC -emit-llvm -c "$c" -o "$BC_DIR/$bn"
done

# 5) src/*.c → .bc
for c in src/*.c; do
  [ -f "$c" ] || continue
  bn=$(echo "$c" | tr '/' '_' | sed 's/\.c$/.bc/')
  clang $BC_CFLAGS $INC -emit-llvm -c "$c" -o "$BC_DIR/$bn"
done

# 6) 链接并生成 .ll
llvm-link "$BC_DIR"/*.bc -o curl_o0.bc
llvm-dis curl_o0.bc -o curl_o0.ll
rm -rf "$BC_DIR" curl_o0.bc

# 完成：当前目录下应有 curl_o0、curl_o0.ll
ls -la curl_o0 curl_o0.ll
```

---

## 说明

- **报错 `no such file or directory: 'lib/*.c'`**：说明当前目录不是 curl 源码目录，没有 `lib/`。必须先 `cd "$CDIR"`（例如上面的 `CDIR=.../curl-8.7.1`）。
- zlib 已按脚本弄好的话，不用再跑 zlib；上面只做 curl 的 O0 可执行文件 + `curl_o0.ll`。

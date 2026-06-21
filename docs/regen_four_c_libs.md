# 四个 C 库完整流水线（libpng / expat / pcre2 / mbedtls）

与 zlib、OpenSSL 等一致：**O0–O3 四档**、**DWARF**、**逐文件 LLVM IR（`ir_oN/`）**、**`build_binskel_dataset_md.py` + `validate_alignment.py`（含 tree-sitter）**。

## 环境

- **WSL2 / Linux**（推荐；Windows 上请在 WSL 中跑 `.sh`）。
- **仓库路径**：若在 WSL 里用 `/mnt/e/structlift`，请设 `export STRUCTLIFT_ROOT=/mnt/e/structlift`，**不要**照抄示例里的 `$HOME/structlift`（除非你确实 clone 到主目录）。
- **行尾 CRLF**：若出现 `$'\r': command not found` 或 `set: invalid option`，在仓库根执行：  
  `sed -i 's/\r$//' scripts/*.sh`  
  （本仓库 `.gitattributes` 已要求 `*.sh` 使用 LF；从别处拷贝脚本后仍可能带 `\r`。）
- **工具**：`clang`、`make`、`cmake`、`ninja`、`python3`、`wget` 或 `curl`；autotools 项目需 `autoconf` / `automake` / `libtool`（一般已装）。
- **Python**：与主项目相同依赖；可选 `export VENV=~/struclift_wsl/.venv/bin/activate`（`build_*_binskel.sh` 会自动 `source`）。
- 若在 Windows 检出后脚本报错，检查 **LF 行尾**：`sed -i 's/\r$//' scripts/*.sh`。

## 一键总控

在**仓库根目录**（或设置 `STRUCTLIFT_ROOT`）执行：

```bash
export STRUCTLIFT_ROOT="$HOME/structlift"   # 改为你的 structlift 路径
export DATASET_ROOT="$HOME/struclift_wsl/datasets"
export BINSKEL_OUT="$DATASET_ROOT/four_c_libs_binskel"

cd "$STRUCTLIFT_ROOT"
bash scripts/regen_four_c_libs_full_pipeline.sh
```

可选覆盖版本：

```bash
export LIBPNG_VERSION=1.6.47
export EXPAT_VERSION=2.6.4
export PCRE2_VERSION=10.44
export MBEDTLS_VERSION=3.6.2
```

## 分步命令（与总控等价）

假设 `STRUCTLIFT_ROOT`、`DATASET_ROOT`、`BINSKEL_OUT` 已导出，`cd "$STRUCTLIFT_ROOT"`。

### libpng

```bash
bash scripts/compile_libpng.sh "$DATASET_ROOT/libpng"
bash scripts/build_libpng_binskel.sh \
  "$DATASET_ROOT/libpng/libpng-${LIBPNG_VERSION:-1.6.47}" \
  "$BINSKEL_OUT/libpng"
```

### expat

```bash
bash scripts/compile_expat.sh "$DATASET_ROOT/expat"
bash scripts/build_expat_binskel.sh \
  "$DATASET_ROOT/expat/expat-${EXPAT_VERSION:-2.6.4}" \
  "$BINSKEL_OUT/expat"
```

### pcre2

```bash
bash scripts/compile_pcre2.sh "$DATASET_ROOT/pcre2"
bash scripts/build_pcre2_binskel.sh \
  "$DATASET_ROOT/pcre2/pcre2-${PCRE2_VERSION:-10.44}" \
  "$BINSKEL_OUT/pcre2"
```

### mbedtls

```bash
bash scripts/compile_mbedtls.sh "$DATASET_ROOT/mbedtls"
bash scripts/build_mbedtls_binskel.sh \
  "$DATASET_ROOT/mbedtls/mbedtls-${MBEDTLS_VERSION:-3.6.2}" \
  "$BINSKEL_OUT/mbedtls"
```

## 产出路径约定

| 项目    | 解压源码 + ELF + `ir_oN/` |
|--------|---------------------------|
| libpng | `$DATASET_ROOT/libpng/libpng-<ver>/`（`libpng_o0`…`o3`，`ir_o0`…`ir_o3`） |
| expat  | `$DATASET_ROOT/expat/expat-<ver>/`（`expat_o*`） |
| pcre2  | `$DATASET_ROOT/pcre2/pcre2-<ver>/`（`pcre2_o*`） |
| mbedtls| `$DATASET_ROOT/mbedtls/mbedtls-<ver>/`（`mbedtls_o*`，由三静态库合并的可重定位 ELF） |

Binskel JSONL：`$BINSKEL_OUT/<name>/binskel_<name>_o0.jsonl` … `o3.jsonl`。

## 说明

- **O0**：`build_*_binskel.sh` 强制带 `--llvm-ir-dir ir_o0`；O1–O3 若存在对应 `ir_oN/` 也会传入。
- **mbedtls**：**必须用 git**（`compile_mbedtls.sh` 会 `git clone --recurse-submodules`，并对本次 clone 使用 `-c http.sslBackend=gnutls`，避免部分环境下「只支持 gnutls 却配置了 openssl」的报错）。3.6+ 依赖子模块 **`framework/`**，GitHub 自动生成的 **tar.gz 不含子模块**，会触发 `framework/CMakeLists.txt not found`。若你曾用 tar 解压过同目录，脚本会删掉该目录再 clone。可选：`MBEDTLS_GIT_URL`、`MBEDTLS_GIT_TAG`（默认 `v$MBEDTLS_VERSION`）、`MBEDTLS_GIT_SSL_BACKEND`（默认 `gnutls`）。若子模块拉取仍失败，可执行：`git config --global http.sslBackend gnutls`。  
  构建：CMake + Ninja → `compile_commands.json` → `emit_ir_from_compile_commands.py`；ELF 为三静态库解包后 `clang -nostdlib -r` 合并。
- **pcre2**：`--disable-jit`；IR 在 **make 之后** 对 `src/*.c` 生成（含生成源码）。
- **expat**：目标为静态链接的 `xmlwf`；IR 覆盖 `lib/*.c` 与 `xmlwf/xmlwf.c`（若存在）。

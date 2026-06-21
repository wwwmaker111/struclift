# 九项目 × 五优化档（45 个 jsonl）— 完整重建说明

**分三部分、每部分 3 个项目依次执行的命令**：见 [BUILD_BINSKEL_9x5_THREE_PARTS.md](BUILD_BINSKEL_9x5_THREE_PARTS.md)。

本文档与 `scripts/binskel_alignment_batch_stats.py` 中的 **`DEFAULT_PROJECTS`** 一致，并约定其中 **3 个项目使用 multibin 合并输出**（与 `stats_9proj_5opt_prefer_multibin.py` 的选文件规则一致）：

| 项目 | 类型 | O0–O3 + Os 共 5 个文件名模式 |
|------|------|------------------------------|
| zlib | 单 ELF | `binskel_zlib_o{0..3}.jsonl`、`binskel_zlib_os.jsonl` |
| curl | 单 ELF | `binskel_curl_o{0..3}.jsonl`、`binskel_curl_os.jsonl` |
| sqlite | 单 ELF | `binskel_sqlite_o{0..3}.jsonl`、`binskel_sqlite_os.jsonl` |
| busybox | 单 ELF | `binskel_busybox_o{0..3}.jsonl`、`binskel_busybox_os.jsonl` |
| openssl | 单 ELF | `binskel_openssl_o{0..3}.jsonl`、`binskel_openssl_os.jsonl` |
| libxml2 | 单 ELF | `binskel_libxml2_o{0..3}.jsonl`、`binskel_libxml2_os.jsonl` |
| **openssh** | **multibin** | `binskel_openssh_{o0,o1,o2,o3}_multibin.jsonl`、`binskel_openssh_os_multibin.jsonl` |
| **ffmpeg** | **multibin** | `binskel_ffmpeg_{o0,o1,o2,o3}_multibin.jsonl`、`binskel_ffmpeg_os_multibin.jsonl` |
| **coreutils** | **multibin** | `binskel_coreutils_{o0,o1,o2,o3}_multibin.jsonl`、`binskel_coreutils_os_multibin.jsonl` |

**合计：9 × 5 = 45 个文件。**

---

## 数据集里与 AST 相关的字段（`build_binskel_dataset_md.py`）

每条样本在 **语句档**（`src_encoding_tier=stmt`）时包含：

- **`src_ast_types`**：与每条源码语句列对齐的 AST 类型 id（tree-sitter 语句类型映射）。
- **`src_compound_id`**：与 `src_ast_types` **同长度**；表示最内层包住该语句的 `compound_statement` 在**当前列空间**中的下标，无则为 `-1`。行级回退档为全 `-1`。

可选：设置环境变量 **`STRIP_PATH_PREFIX`** 后，所有 `build_*_binskel.sh` 会通过 `scripts/_binskel_md_extra.sh` 传给 `--strip-path-prefix`，用于从写出的 **`binary` / `source_file`** 字符串中去掉机器相关前缀。

---

## 前置条件

1. 在 **structlift 仓库根** 下使用 **bash**（WSL/Linux；勿用未装 bash 的纯 cmd）。
2. 激活与构建脚本一致的 Python 环境（脚本默认 `source ~/struclift_wsl/.venv/bin/activate`，可用环境变量 **`VENV`** 覆盖）。
3. **每个项目**已按仓库内对应 **`scripts/compile_*.sh`** 编译出各档 ELF 与 IR（单 ELF 为 `*_oN` / `*_oN.ll` 或 `ir_oN/`；multibin 为 `multibin_oN/` + `ir_oN/` 等，以各 compile 脚本为准）。

---

## 一键调度（推荐）

在仓库根执行，先设置 **输出目录** 与各项目 **源码+产物根目录**，再运行：

```bash
cd /path/to/structlift
# source ~/struclift_wsl/.venv/bin/activate   # 若需要

export OUT_ROOT="$HOME/structlift_datasets"

export ZLIB_DIR="$HOME/struclift_wsl/datasets/zlib/zlib-1.3.1"
export CURL_DIR="$HOME/struclift_wsl/datasets/curl/curl-8.7.1"
export SQLITE_DIR="$HOME/struclift_wsl/datasets/sqlite/sqlite-3450100"
export BUSYBOX_DIR="$HOME/struclift_wsl/datasets/busybox/busybox-1.36.1"
export OPENSSL_DIR="$HOME/struclift_wsl/datasets/openssl/openssl-3.3.1"
export OPENSSH_DIR="$HOME/struclift_wsl/datasets/openssh/openssh-9.8p1"
export LIBXML2_DIR="$HOME/struclift_wsl/datasets/libxml2/libxml2-2.12.7"
export FFMPEG_DIR="$HOME/struclift_wsl/datasets/ffmpeg/ffmpeg-7.1"
export COREUTILS_DIR="$HOME/struclift_wsl/datasets/coreutils/coreutils-9.4"

# 可选：统一去掉 jsonl 路径前缀（按本机实际改）
# export STRIP_PATH_PREFIX="/home/yourname/"

bash scripts/build_binskel_9proj_5opt.sh
```

**说明：** 未设置的 `*_DIR` 会跳过；路径请改为你机器上的真实目录。

---

## 分项目完整命令（与一键脚本等价）

以下假定：

- `STRUCTLIFT=/path/to/structlift`
- `OUT=$HOME/structlift_datasets`
- 第三个参数为 **structlift 仓库根**（与旧脚本一致）。

### 1. zlib（6 条命令：O0–O3 一条脚本 + Os 一条）

```bash
bash "$STRUCTLIFT/scripts/build_zlib_binskel.sh"       "$ZLIB_DIR" "$OUT" "$STRUCTLIFT"
bash "$STRUCTLIFT/scripts/os_only/build_zlib_os_binskel.sh" "$ZLIB_DIR" "$OUT" "$STRUCTLIFT"
```

### 2. curl

```bash
bash "$STRUCTLIFT/scripts/build_curl_binskel.sh"       "$CURL_DIR" "$OUT" "$STRUCTLIFT"
bash "$STRUCTLIFT/scripts/os_only/build_curl_os_binskel.sh" "$CURL_DIR" "$OUT" "$STRUCTLIFT"
```

### 3. sqlite

```bash
bash "$STRUCTLIFT/scripts/build_sqlite_binskel.sh"     "$SQLITE_DIR" "$OUT" "$STRUCTLIFT"
bash "$STRUCTLIFT/scripts/os_only/build_sqlite_os_binskel.sh" "$SQLITE_DIR" "$OUT" "$STRUCTLIFT"
```

### 4. busybox

```bash
bash "$STRUCTLIFT/scripts/build_busybox_binskel.sh"    "$BUSYBOX_DIR" "$OUT" "$STRUCTLIFT"
bash "$STRUCTLIFT/scripts/os_only/build_busybox_os_binskel.sh" "$BUSYBOX_DIR" "$OUT" "$STRUCTLIFT"
```

### 5. openssl

```bash
bash "$STRUCTLIFT/scripts/build_openssl_binskel.sh"    "$OPENSSL_DIR" "$OUT" "$STRUCTLIFT"
bash "$STRUCTLIFT/scripts/os_only/build_openssl_os_binskel.sh" "$OPENSSL_DIR" "$OUT" "$STRUCTLIFT"
```

### 6. libxml2

```bash
bash "$STRUCTLIFT/scripts/build_libxml2_binskel.sh"    "$LIBXML2_DIR" "$OUT" "$STRUCTLIFT"
bash "$STRUCTLIFT/scripts/os_only/build_libxml2_os_binskel.sh" "$LIBXML2_DIR" "$OUT" "$STRUCTLIFT"
```

### 7. openssh（**multibin**，一条脚本生成 5 个 jsonl）

```bash
bash "$STRUCTLIFT/scripts/build_openssh_multibin_binskel.sh" "$OPENSSH_DIR" "$OUT" "$STRUCTLIFT"
```

### 8. ffmpeg（**multibin**）

```bash
bash "$STRUCTLIFT/scripts/build_ffmpeg_multibin_binskel.sh" "$FFMPEG_DIR" "$OUT" "$STRUCTLIFT"
```

### 9. coreutils（**multibin**）

```bash
bash "$STRUCTLIFT/scripts/build_coreutils_multibin_binskel.sh" "$COREUTILS_DIR" "$OUT" "$STRUCTLIFT"
```

---

## 可选环境变量（所有上述脚本）

| 变量 | 含义 |
|------|------|
| `WORKERS` | 预解析源码并行度（默认 8） |
| `NUM_OPCODES` | 与训练 `VocabConfig.num_opcodes` 一致（默认 1024） |
| `SRC_VOCAB` | 源码词表大小（默认 32000） |
| `STRIP_PATH_PREFIX` | 传给 `build_binskel_dataset_md.py --strip-path-prefix` |
| `VENV` | venv 的 `activate` 路径 |

---

## 校验

对每个生成的 jsonl：

```bash
python scripts/validate_alignment.py "$OUT/binskel_zlib_o0.jsonl"
```

批量抽查可用：`scripts/spotcheck_binskel_dataset_requirements.py`。

---

## 与旧文档的关系

单项目示例见仓库根 `BUILD_BINSKEL_COMMANDS.md`；本文档覆盖 **九项目 × 五档** 的命名与调度。

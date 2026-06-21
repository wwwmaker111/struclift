# 九项目 binskel 分三部分执行（每部分 3 个项目）

在 **WSL/Linux bash** 下操作。请把 **`STRUCTLIFT`**、**各 `*_DIR`** 换成你本机真实路径；不要用 `/path/to` 或 `...` 占位。

**通用约定：**

- `STRUCTLIFT`：structlift 仓库根（该目录下应有 `build_binskel_dataset_md.py` 与 `scripts/`）。
- `OUT`：jsonl 输出目录，例如 `$HOME/structlift_datasets`。
- 第三个参数传 **`$STRUCTLIFT`**（仓库根），与现有脚本一致。
- 可选：`export STRIP_PATH_PREFIX="/home/你的用户名/"`（统一剥离 jsonl 里 `binary`/`source_file` 前缀）。
- 可选：`export WORKERS=16`（加快预解析，视 CPU/内存调整）。

---

## 第一部分：zlib、curl、sqlite（各 5 个 jsonl，共 15 个文件）

```bash
STRUCTLIFT="/mnt/e/structlift"
OUT="$HOME/structlift_datasets"

export ZLIB_DIR="$HOME/struclift_wsl/datasets/zlib/zlib-1.3.1"
export CURL_DIR="$HOME/struclift_wsl/datasets/curl/curl-8.7.1"
export SQLITE_DIR="$HOME/struclift_wsl/datasets/sqlite/sqlite-3450100"

cd "$STRUCTLIFT"
mkdir -p "$OUT"

bash scripts/build_zlib_binskel.sh       "$ZLIB_DIR"    "$OUT" "$STRUCTLIFT"
bash scripts/os_only/build_zlib_os_binskel.sh    "$ZLIB_DIR"    "$OUT" "$STRUCTLIFT"

bash scripts/build_curl_binskel.sh       "$CURL_DIR"    "$OUT" "$STRUCTLIFT"
bash scripts/os_only/build_curl_os_binskel.sh    "$CURL_DIR"    "$OUT" "$STRUCTLIFT"

bash scripts/build_sqlite_binskel.sh     "$SQLITE_DIR"  "$OUT" "$STRUCTLIFT"
bash scripts/os_only/build_sqlite_os_binskel.sh  "$SQLITE_DIR"  "$OUT" "$STRUCTLIFT"
```

**产出文件名：**  
`binskel_zlib_o{0..3}.jsonl`、`binskel_zlib_os.jsonl`；curl、sqlite 同理。

---

## 第二部分：busybox、openssl、libxml2（各 5 个 jsonl，共 15 个文件）

```bash
STRUCTLIFT="/mnt/e/structlift"
OUT="$HOME/structlift_datasets"

export BUSYBOX_DIR="$HOME/struclift_wsl/datasets/busybox/busybox-1.36.1"
export OPENSSL_DIR="$HOME/struclift_wsl/datasets/openssl/openssl-3.3.1"
export LIBXML2_DIR="$HOME/struclift_wsl/datasets/libxml2/libxml2-2.12.7"

cd "$STRUCTLIFT"
mkdir -p "$OUT"

bash scripts/build_busybox_binskel.sh    "$BUSYBOX_DIR"  "$OUT" "$STRUCTLIFT"
bash scripts/os_only/build_busybox_os_binskel.sh "$BUSYBOX_DIR"  "$OUT" "$STRUCTLIFT"

bash scripts/build_openssl_binskel.sh    "$OPENSSL_DIR"  "$OUT" "$STRUCTLIFT"
bash scripts/os_only/build_openssl_os_binskel.sh "$OPENSSL_DIR"  "$OUT" "$STRUCTLIFT"

bash scripts/build_libxml2_binskel.sh     "$LIBXML2_DIR"  "$OUT" "$STRUCTLIFT"
bash scripts/os_only/build_libxml2_os_binskel.sh "$LIBXML2_DIR"  "$OUT" "$STRUCTLIFT"
```

**产出文件名：**  
`binskel_busybox_*`、`binskel_openssl_*`、`binskel_libxml2_*`（各 O0–O3 + os）。

---

## 第三部分：openssh、ffmpeg、coreutils（均为 multibin，各 5 个 jsonl，共 15 个文件）

每个项目 **只跑一条 multibin 脚本**，内部已包含 O0–O3 + Os。

```bash
STRUCTLIFT="/mnt/e/structlift"
OUT="$HOME/structlift_datasets"

export OPENSSH_DIR="$HOME/struclift_wsl/datasets/openssh/openssh-9.8p1"
export FFMPEG_DIR="$HOME/struclift_wsl/datasets/ffmpeg/ffmpeg-7.1"
export COREUTILS_DIR="$HOME/struclift_wsl/datasets/coreutils/coreutils-9.4"

cd "$STRUCTLIFT"
mkdir -p "$OUT"

bash scripts/build_openssh_multibin_binskel.sh   "$OPENSSH_DIR"   "$OUT" "$STRUCTLIFT"
bash scripts/build_ffmpeg_multibin_binskel.sh  "$FFMPEG_DIR"    "$OUT" "$STRUCTLIFT"
bash scripts/build_coreutils_multibin_binskel.sh "$COREUTILS_DIR" "$OUT" "$STRUCTLIFT"
```

**产出文件名：**  
`binskel_openssh_{o0,o1,o2,o3}_multibin.jsonl`、`binskel_openssh_os_multibin.jsonl`；ffmpeg、coreutils 同理。

---

## 三部分合计

共 **45** 个 jsonl（9 项目 × 5 档）。三部分可 **分三天/三台机器** 跑，只要 **`OUT` 相同** 即可写到同一目录。

详见 [BUILD_BINSKEL_9x5_FULL.md](BUILD_BINSKEL_9x5_FULL.md)。

---

## 故障排除：`\r` / `$'\r': command not found`

脚本若在 Windows 下被保存成 **CRLF**，在 WSL 里 bash 会报错。任选其一：

1. **更新仓库**（本仓库已对 `*.sh` 统一为 LF，且 `.gitattributes` 强制 `*.sh` 使用 LF）：在 `/mnt/e/structlift` 执行 `git pull` 或重新同步工作区。  
2. **本地一键去 `\r`**（在 structlift 根目录）：

```bash
find scripts -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
find . -maxdepth 1 -name '*.sh' -print0 | xargs -0 sed -i 's/\r$//'
```

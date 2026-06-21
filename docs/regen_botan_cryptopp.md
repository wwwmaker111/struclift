# Botan 与 Crypto++ 完整流水线（O0–O3 + DWARF + 逐文件 IR + tree-sitter）

与 OpenSSL、zlib 等一致：**四档优化各一份 ELF、`ir_o0`…`ir_o3` 目录**，再跑 `build_binskel_dataset_md.py`（**不**使用 `--skip-alignment`）。

> **说明**：一键流水线**不再包含 BoringSSL**。仓库里仍保留 `scripts/compile_boringssl.sh` / `build_boringssl_binskel.sh` 供需要时**单独**使用。

## 依赖（WSL / Linux）

| 依赖 | Botan | Crypto++ |
|------|-------|----------|
| `clang` / `clang++` | ✓ | ✓ |
| `python3` | ✓ | ✓ |
| `wget` 或 `curl` | ✓ | ✓ |
| `bear` **或** `compiledb` | **二选一**（生成 `compile_commands.json` 以产 IR） | 可选（无则根目录 `.cpp` 启发式 IR） |
| `unzip` | — | ✓（首次解压） |

建议激活项目 venv（`VENV`，与其它 `build_*_binskel.sh` 相同）。

## 路径与 CRLF

- 在仓库根执行，例如：`cd /mnt/e/structlift`。
- 若脚本从 Windows 拷入带 **CRLF**：`sed -i 's/\r$//' scripts/regen_botan_cryptopp_full_pipeline.sh scripts/compile_botan.sh scripts/compile_cryptopp.sh scripts/build_botan_binskel.sh scripts/build_cryptopp_binskel.sh`

## 一键：8 个 JSONL（两项目 × 四档）

```bash
export OUT_DIR=~/structlift_datasets
cd /mnt/e/structlift
bash scripts/regen_botan_cryptopp_full_pipeline.sh ~/struclift_wsl/datasets
```

（旧名 `regen_crypto_three_full_pipeline.sh` 仍会调用同一脚本。）

数据集目录：

- `~/struclift_wsl/datasets/botan/Botan-<版本>/`
- `~/struclift_wsl/datasets/cryptopp/cryptopp-*`

输出：

- `binskel_botan_o{0,1,2,3}.jsonl`
- `binskel_cryptopp_o{0,1,2,3}.jsonl`

## 源码获取

| 项目 | 方式 |
|------|------|
| **Botan** | `https://botan.randombit.net/releases/Botan-x.y.z.tar.xz` + `tar -xJf` |
| **Crypto++** | GitHub tag **`.../archive/refs/tags/CRYPTOPP_x_x_x.zip`** + `unzip` |

## 分步执行

### Botan

**生成 IR 需要 `compile_commands.json`**，任选其一：

- **bear**：`sudo apt install bear`
- **compiledb**（无需 sudo，适合已有 venv）：`pip install compiledb`  
  脚本会依次尝试：`bear` → `compiledb` → `python3 -m compiledb`

```bash
# 可选: pip install compiledb   # 在 (.venv) 里即可
# 可选: export BOTAN_VERSION=3.6.1
bash scripts/compile_botan.sh ~/struclift_wsl/datasets/botan
bash scripts/build_botan_binskel.sh \
  ~/struclift_wsl/datasets/botan/Botan-3.6.1 \
  ~/structlift_datasets \
  /mnt/e/structlift
```

### Crypto++

```bash
# 可选: export CRYPTOPP_TAG=CRYPTOPP_8_9_0
bash scripts/compile_cryptopp.sh ~/struclift_wsl/datasets/cryptopp
bash scripts/build_cryptopp_binskel.sh \
  ~/struclift_wsl/datasets/cryptopp/cryptopp-CRYPTOPP_8_9_0 \
  ~/structlift_datasets \
  /mnt/e/structlift
```

## 统计样本数

```bash
python scripts/binskel_dataset_stats.py \
  ~/structlift_datasets/binskel_botan_o0.jsonl \
  ~/structlift_datasets/binskel_cryptopp_o0.jsonl --markdown
```

## C++ 与 tree-sitter

`build_binskel_dataset_md.py` 使用 **tree-sitter C 文法**；复杂 C++ 可能更多**行级回退**，DWARF + IR 仍按完整流水线执行。

## 相关脚本

| 脚本 | 作用 |
|------|------|
| `scripts/emit_ir_from_compile_commands.py` | 从 `compile_commands.json` 生成 `.ll` |
| `scripts/regen_botan_cryptopp_full_pipeline.sh` | Botan + Crypto++ 编译与四档 binskel |
| `scripts/regen_crypto_three_full_pipeline.sh` | 同上（旧文件名，转发） |

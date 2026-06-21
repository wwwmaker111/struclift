# 五项目完整流水线重生成（不含 zlib）

项目：**curl、sqlite、busybox、openssl、libxml2** — 先 `compile_*.sh`（O0–O3 + IR），再 `build_*_binskel.sh`（DWARF + IR + tree-sitter）。

## 一键脚本

```bash
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate

# 编译 + 构建 + 统计（耗时很长，建议 nohup）
mkdir -p logs
nohup bash scripts/regen_five_projects_full_pipeline.sh >> logs/nohup_regen_five_projects.log 2>&1 &
tail -f logs/nohup_regen_five_projects.log
```

仅构建（假定 ELF/IR 已就绪）：

```bash
SKIP_COMPILE=1 bash scripts/regen_five_projects_full_pipeline.sh
```

环境变量：

| 变量 | 含义 |
|------|------|
| `WORKDIR` | 仓库根，默认脚本推断 |
| `DATA` | 数据集根，默认 `$HOME/struclift_wsl/datasets` |
| `OUT` | jsonl 输出目录，默认 `$HOME/structlift_datasets` |
| `VENV` | venv 的 `activate` 路径 |

## 日志与统计

- 每次运行会在 `logs/regen_five_projects_YYYYMMDD_HHMMSS.log` 写一份完整日志（与 tee 同步）。
- 脚本结束会打印 **Markdown 表**（各 `binskel_*_oN.jsonl` 样本数与大小）。

单独统计：

```bash
python scripts/binskel_dataset_stats.py ~/structlift_datasets/binskel_curl_o0.jsonl --markdown
```

## 修复说明（相对旧版）

- **`scripts/build_openssl_binskel.sh`**：改为与 zlib/curl 相同传参  
  `build_openssl_binskel.sh <openssl_dir> <out_dir> [workdir]`，不再写死 `/mnt/e/structlift`。
- **`scripts/regen_five_projects_full_pipeline.sh`**：五项目编译 + 五项目 binskel + 统计。
- **`scripts/binskel_dataset_stats.py`**：统一样本数/大小统计。

## OpenSSL 说明

O0 必须使用 **`--llvm-ir-dir ir_o0/`**（无合并 `openssl_o0.ll`），与 `compile_openssl.sh` 一致。

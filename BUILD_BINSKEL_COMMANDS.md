# Binskel 数据集构建 — 完整命令行

环境与路径（按你的实际环境替换）：

- 工作目录：`cd /mnt/e/structlift`
- 激活环境：`source ~/struclift_wsl/.venv/bin/activate`
- ELF 与源码目录：`/home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000/`
- 输出目录：`/home/wuqiongmin/structlift_datasets/`

---

## 1. O0（含 IR 对齐，Module B）

需要：`sqlite3_o0`、`sqlite3.c`、`sqlite3.ll` 同目录。

```bash
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate

python build_binskel_dataset_md.py \
  /home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000/sqlite3_o0 \
  /home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl \
  --ir /home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000/sqlite3.ll \
  --opt o0
```

（若用 MD 文档编译的 IR：`--ir .../sqlite3_md.ll`，输出可改为 `binskel_sqlite_o0_md.jsonl`。）

---

## 2. O1 / O2 / O3（完整流水线，与 O0 一致）

对每档传入对应 **`sqlite_oN.ll`**（或你目录下的合并 IR），**不要**使用 `--skip-alignment`。

**O1：**

```bash
python build_binskel_dataset_md.py \
  --elf /home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000/sqlite_o1 \
  --src /home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000 \
  --llvm-ir /home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000/sqlite_o1.ll \
  --out /home/wuqiongmin/structlift_datasets/binskel_sqlite_o1.jsonl \
  --opt O1
```

**O2 / O3：** 将 `o1` / `sqlite_o1` / `sqlite_o1.ll` 换成对应档位即可。

---

## 3. O3（若有 sqlite_o3）

```bash
python build_binskel_dataset_md.py \
  --elf "$BASE/sqlite_o3" \
  --src "$BASE" \
  --llvm-ir "$BASE/sqlite_o3.ll" \
  --out "$OUT/binskel_sqlite_o3.jsonl" \
  --opt O3
```

---

## 4. 校验（validate）

对每个生成的 jsonl 做一次校验（BB 覆盖率、置信度、顺序、控制流一致性）：

```bash
# O0
python scripts/validate_alignment.py \
  /home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl

# O1
python scripts/validate_alignment.py \
  /home/wuqiongmin/structlift_datasets/binskel_sqlite_o1.jsonl

# O2
python scripts/validate_alignment.py \
  /home/wuqiongmin/structlift_datasets/binskel_sqlite_o2.jsonl
```

---

## 5. 合并多优化级别（可选）

若需要把 O0/O1/O2 合并成一份数据集（例如按函数名去重或按 opt 区分）：

```bash
# 简单按行合并（不去重）
cat /home/wuqiongmin/structlift_datasets/binskel_sqlite_o0.jsonl \
    /home/wuqiongmin/structlift_datasets/binskel_sqlite_o1.jsonl \
    /home/wuqiongmin/structlift_datasets/binskel_sqlite_o2.jsonl \
  > /home/wuqiongmin/structlift_datasets/binskel_sqlite_merged.jsonl
```

如需按函数名或 (func_name, opt) 去重，可写小脚本过滤 jsonl。

---

## 6. 常用可选参数

- `-v`：打开 DEBUG 日志（每函数打印）。
- `--workers N`：当前为预留参数，多进程未实现，保持默认 1 即可。
- `--llvm-ir <path>`：各优化等级均建议提供对应合并 `.ll`，与 MD 流水线一致。

---

## 7. 一键顺序执行（复制整段）

```bash
cd /mnt/e/structlift
source ~/struclift_wsl/.venv/bin/activate
BASE=/home/wuqiongmin/struclift_wsl/datasets/sqlite/sqlite-autoconf-3520000
OUT=/home/wuqiongmin/structlift_datasets

# O0–O3：完整流水线（推荐直接用 scripts/build_sqlite_binskel.sh）
# 以下为等价手写示例（路径按你本机 amalgamation 命名调整：sqlite_o0 或 sqlite3_o0）
python build_binskel_dataset_md.py --elf "$BASE/sqlite_o0" --src "$BASE" --llvm-ir "$BASE/sqlite_o0.ll" --out "$OUT/binskel_sqlite_o0.jsonl" --opt O0
for n in 1 2 3; do
  python build_binskel_dataset_md.py --elf "$BASE/sqlite_o$n" --src "$BASE" --llvm-ir "$BASE/sqlite_o$n.ll" --out "$OUT/binskel_sqlite_o$n.jsonl" --opt "O$n"
done

# 校验
for n in 0 1 2 3; do python scripts/validate_alignment.py "$OUT/binskel_sqlite_o$n.jsonl"; done
```

**可选加速**：若仅需行级、可接受降质，可显式加 `--skip-alignment`（默认仓库脚本已统一走完整流水线）。

# Zlib 数据集流水线：下载、编译、对齐、构建（完整命令行）

- **O0 / O1 / O2 / O3**：均按 `BB_Source_Alignment_Pipeline.md` 做**完整**基本块→源码**语句**对齐（DWARF + IR + tree-sitter），**不**使用 `--skip-alignment`。
- `compile_zlib.sh` 为每档产出对应 `.ll`；`build_zlib_binskel.sh` 对四档均传入 `--llvm-ir`。

---

## 环境（按你的机器改）

```bash
WORKDIR=/mnt/e/structlift
VENV=~/struclift_wsl/.venv/bin/activate
# zlib 下载与编译目录（建议 WSL 下）
ZLIB_BASE=/home/wuqiongmin/struclift_wsl/datasets/zlib
OUT_DIR=/home/wuqiongmin/structlift_datasets
```

---

## 前置依赖

- **clang**（编译）
- **llvm-link、llvm-dis**（生成各档合并 `.ll`）

## 一、下载并编译 zlib（四档均含 DWARF + 对应 IR）

编译选项：O0 严格按 MD；O1/O2/O3 为 `-O1/-O2/-O3 -g3 ...`，详见 `scripts/compile_zlib.sh`。

```bash
cd "$WORKDIR"
source "$VENV"

bash "$WORKDIR/scripts/compile_zlib.sh" "$ZLIB_BASE"
```

产出（在 `$ZLIB_BASE/zlib-1.3.1/` 下）：

- `zlib_o0` … `zlib_o3` — 各优化等级 ELF
- `zlib_o0.ll` … `zlib_o3.ll` — 各档合并 IR，供 binskel 完整流水线

---

## 二、构建 binskel 数据集（四档完整对齐）

### 2.1 一键脚本（推荐）

```bash
cd "$WORKDIR"
source "$VENV"

ZLIB_DIR="$ZLIB_BASE/zlib-1.3.1"
bash "$WORKDIR/scripts/build_zlib_binskel.sh" "$ZLIB_DIR" "$OUT_DIR" "$WORKDIR"
```

### 2.2 逐条命令（与脚本等价）

**O0：**

```bash
python build_binskel_dataset_md.py \
  --elf   "$ZLIB_DIR/zlib_o0" \
  --src   "$ZLIB_DIR" \
  --out   "$OUT_DIR/binskel_zlib_o0.jsonl" \
  --llvm-ir "$ZLIB_DIR/zlib_o0.ll" \
  --opt   o0
```

**O1 / O2 / O3：**（同样完整流水线，传入对应 `.ll`，**不要**加 `--skip-alignment`）

```bash
for n in 1 2 3; do
  python build_binskel_dataset_md.py \
    --elf   "$ZLIB_DIR/zlib_o$n" \
    --src   "$ZLIB_DIR" \
    --out   "$OUT_DIR/binskel_zlib_o$n.jsonl" \
    --llvm-ir "$ZLIB_DIR/zlib_o$n.ll" \
    --opt   "o$n"
done
```

---

## 三、校验对齐质量

```bash
python scripts/validate_alignment.py "$OUT_DIR/binskel_zlib_o0.jsonl"
# 可选：对 O1–O3 同样运行 validate_alignment
```

---

## 四、与 MD / StructLift 的对应关系

| 步骤 | 说明 |
|------|------|
| **编译** | `compile_zlib.sh` 产出各档 ELF + `zlib_oN.ll`。 |
| **DWARF** | `get_bb_source_lines` 优先 is_stmt（MD 第二步）。 |
| **IR** | 解析对应 `zlib_oN.ll` 的 !dbg，BB↔IR BB（MD 第三、五步）。 |
| **语句映射** | tree-sitter + `map_lines_to_statement`（MD 第四步）；**四档均不加** `--skip-alignment`。 |
| **M_gt / W** | 按 MD 第六步。 |
| **Module A** | O0–O3 的 jsonl 均可作训练数据，对齐口径一致。 |

---

## 五、一键完整流水线（复制整段执行）

```bash
WORKDIR=/mnt/e/structlift
VENV=~/struclift_wsl/.venv/bin/activate
ZLIB_BASE=/home/wuqiongmin/struclift_wsl/datasets/zlib
OUT_DIR=/home/wuqiongmin/structlift_datasets
ZLIB_DIR="$ZLIB_BASE/zlib-1.3.1"

cd "$WORKDIR"
source "$VENV"

bash "$WORKDIR/scripts/compile_zlib.sh" "$ZLIB_BASE"
bash "$WORKDIR/scripts/build_zlib_binskel.sh" "$ZLIB_DIR" "$OUT_DIR" "$WORKDIR"
```

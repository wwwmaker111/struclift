# zlib 是否已编译完成 + 数据集格式与效果

## 一、zlib 是否已经编译完？

**判断方式**：看 zlib 源码目录下是否有这些文件（路径以你的 `ZLIB_DIR` 为准，例如 `$ZLIB_BASE/zlib-1.3.1`）：

| 文件 | 含义 |
|------|------|
| `zlib_o0` | O0 可执行文件（minigzip），**必须有** |
| `zlib_o0.ll` | O0 单一 LLVM IR，用于对齐时 DWARF+IR 交叉验证，**建议有**（无则 O0 置信度会偏低） |
| `zlib_o1.ll` … `zlib_o3.ll` | 各档合并 IR，完整流水线对齐用，**建议有**（与 `compile_zlib.sh` 一致） |
| `zlib_o1`、`zlib_o2`、`zlib_o3` | O1/O2/O3 可执行文件，完整对齐用，**可选** |

**一键检查**（把 `ZLIB_DIR` 换成你的实际路径）：

```bash
ZLIB_DIR=/home/wuqiongmin/struclift_wsl/datasets/zlib/zlib-1.3.1
for f in zlib_o0 zlib_o0.ll zlib_o1 zlib_o1.ll zlib_o2 zlib_o2.ll zlib_o3 zlib_o3.ll; do
  [ -f "$ZLIB_DIR/$f" ] && echo "有: $f" || echo "无: $f"
done
```

- 若 **有 `zlib_o0`**：可以认为 **zlib 已编译完成**，能跑 `build_zlib_binskel.sh`。
- 若还有 **`zlib_o0.ll`**：O0 对齐会走「DWARF + IR + tree-sitter」完整流程，置信度正常。
- 若 **没有 `zlib_o0`**：需要先执行 `scripts/compile_zlib.sh` 再构建数据集。

---

## 二、zlib 数据集格式（binskel JSONL）

构建脚本产出的是 **每行一条 JSON** 的 `.jsonl` 文件，例如：

- `binskel_zlib_o0.jsonl` — O0，**完整对齐**（基本块↔源码语句 + 置信度 W）
- `binskel_zlib_o1.jsonl`、`binskel_zlib_o2.jsonl`、`binskel_zlib_o3.jsonl` — O1/O2/O3，与 O0 **相同完整流水线**（DWARF + IR + tree-sitter，无 `--skip-alignment`）

**每条样本（一行 JSON）的字段**：

| 字段 | 含义 |
|------|------|
| `binary` | 使用的 ELF 路径（如 `.../zlib_o0`） |
| `func_name` | 函数名 |
| `opt_level` | 优化等级，如 `"0"`、`"1"` |
| `bin_opcodes` | 基本块内指令 opcode 序列 |
| `bin_operand_types` / `bin_operand_values` | 操作数类型/值 |
| `bin_block_lengths` | 每个基本块指令数 |
| `bin_edge_index` | 控制流边 [源块下标列表, 目标块下标列表] |
| `bin_edge_type` | 边类型（如 0=条件 1=无条件等） |
| `bin_struct_features` | 二进制侧结构特征（供模型用） |
| `src_token_ids` | 源码 token 序列（按语句展平） |
| `src_stmt_lengths` | 每条语句的 token 数 |
| `src_edge_index` | 源码控制流边（无 AST 时常为 [[], []]） |
| `src_ast_types` | 语句 AST 类型 |
| **`block_src_lines`** | 每个基本块对应的源码行号列表（每块一行号列表） |
| **`block_src_confidence`** | 每块每行的置信度（与 block_src_lines 对应） |
| **`alignment_matrix`** | 基本块↔源码语句对齐矩阵 M_gt（行=BB，列=语句） |
| **`alignment_confidence`** | 对齐置信度矩阵 W（与 M_gt 同形状） |
| `pattern_labels` | 模式标签（若启用） |
| `source_file` | 主源码文件路径 |
| `source_lines` | 涉及到的源码行号（展平） |

对齐质量校验主要看：`alignment_matrix`、`alignment_confidence`、`block_src_lines`、`block_src_confidence`。

---

## 三、zlib 数据集效果（质量指标）

用 **O0 的 jsonl** 做校验（MD 第七步）：

```bash
python scripts/validate_alignment.py "$OUT_DIR/binskel_zlib_o0.jsonl"
```

脚本会输出 **4 类指标**（对每条样本算完再求平均）：

1. **BB 覆盖率**：有对齐的 BB 占比，期望 ≥70%。
2. **平均置信度**：所有 (BB, 语句) 非零置信度的平均值。
3. **高/低置信度占比**：置信度 >0.7 为高，<0.3 为低。
4. **顺序一致性**：BB 与源码语句顺序是否合理，期望 ≥80%。
5. **控制流一致性**：条件边两端 BB 是否对应不同源码语句，期望较高。

**文档中的参考结论**（见 `docs/MD_IMPLEMENTATION_VERIFICATION.md`）：

- zlib 这类 **小项目、多 .c 文件** 下，**高置信度约 41.7%、平均置信度约 0.54** 是合理范围。
- MD 里 75–85% 高置信度是针对「严格编译 + 大单文件」的预期；zlib 体量小、多文件，比例略低属正常。

**汇总数据量**（可选）：

```bash
bash scripts/summary_data_quality.sh "$OUT_DIR" "$OUT_DIR/binskel_zlib_o0.jsonl"
```

会列出各 `binskel_zlib_o*.jsonl` 的条数，并对 O0 跑上述质量校验。

---

## 四、小结

| 问题 | 答案 |
|------|------|
| zlib 是否已编译完？ | 看 `ZLIB_DIR` 下是否有 `zlib_o0`（必有）、`zlib_o0.ll`（建议有）、`zlib_o1`～`zlib_o3`（可选）。 |
| 数据集格式？ | 每行一条 JSON 的 `.jsonl`，每条含二进制特征、源码 token、**block_src_lines / alignment_matrix / alignment_confidence** 等（见上表）。 |
| 效果怎么样？ | 用 `validate_alignment.py` 看覆盖率、平均/高/低置信度、顺序与控制流一致性；zlib 约 41.7% 高置信、平均 0.54 在文档预期内，属正常。 |

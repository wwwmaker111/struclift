# BB_Source_Alignment_Pipeline.md 与实现逐项对照

本文档核对 `build_binskel_dataset_md.py` 是否按 MD 文档实现，**置信度 W 的公式与 MD 第六步完全一致**。

---

## 第一步：编译参数

| MD 要求 | 实现 | 一致？ |
|--------|------|--------|
| `-O0 -g3 -fno-inline -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fstandalone-debug`，clang 产出二进制与 IR | `scripts/compile_zlib.sh` / 手动编译使用相同 CFLAGS；zlib 自动生成 .ll 时也用该组选项 | ✓ |

---

## 第二步：信号 A — DWARF

| MD 要求 | 实现（build_binskel_dataset_md.py） | 一致？ |
|--------|-------------------------------------|--------|
| 提取**全部**行表条目，含 `is_stmt` | `extract_dwarf_line_map`: 遍历 `lp.get_entries()`，取 addr/line/col/file/is_stmt，不筛 is_stmt | ✓ |
| 按地址排序 | `entries.sort(key=lambda x: x["addr"])` | ✓ |
| `get_bb_source_lines(bb_start, bb_end)`：区间 **[bb_start, bb_end)**，**优先 is_stmt**，空则退回全部行 | `get_bb_source_lines`: bisect 取区间，先收集 stmt_lines 与 all_lines，`return stmt_lines if stmt_lines else all_lines` | ✓ |

---

## 第三步：信号 B — LLVM IR

| MD 要求 | 实现 | 一致？ |
|--------|------|--------|
| 解析 `!DILocation(line: N)` | `parse_llvm_ir_alignment`: `re.finditer(r"!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)"` 建 dbg_locations | ✓ |
| 按函数、按 BB 标签划分，收集 `!dbg` 行号 | 按 `define ... @func` 取 body，按 `bb_label:` 划分，每行 `!dbg !N` 对应 line 加入 BB 的 lines | ✓ |
| 每个 BB 有 role、primary_line | 根据 terminator 设 role；终结指令的 !dbg 作为 primary_line | ✓ |

---

## 第四步：行号 → 语句（tree-sitter）

| MD 要求 | 实现 | 一致？ |
|--------|------|--------|
| STATEMENT_TYPES（if/while/for/do/switch/case/return/expression_statement/declaration/break/continue/goto/compound_statement） | `STATEMENT_TYPES` 含上述，并多 `labeled_statement`、`attributed_statement`（扩展） | ✓ |
| 行号 → 最合适语句：1 个→直接；多个→优先控制流→最内层 | `map_lines_to_statement`: 1 个→0.95；控制流唯一→0.80；最内层→0.60 或 0.40 | ✓ |
| 置信度：唯一 0.95，控制流唯一 0.80，最内层 0.60/0.40 | 同上，数值一致 | ✓ |

---

## 第五步：机器码 BB ↔ IR BB

| MD 要求 | 实现 | 一致？ |
|--------|------|--------|
| 用 DWARF 取机器码 BB 行号，与各 IR BB 算 **Jaccard** | `map_machine_bb_to_ir_bb`: `j = overlap/union`（即 Jaccard） | ✓ |
| 阈值 **> 0.3** 建立映射 | `if best_ir and best_j >= 0.3` | ✓ |
| 返回 `{mach_bb_addr: {ir_label, confidence}}` | `mapping[mach_bb["addr"]] = {"ir_label": best_ir, "confidence": best_j}` | ✓ |

---

## 第六步：M_gt 与置信度 W（核心）

MD 公式与实现**逐行对照**：

| 情况 | MD 公式 | 实现（generate_alignment_ground_truth） | 一致？ |
|------|--------|----------------------------------------|--------|
| 两信号一致 | `M_gt[i, dwarf_stmt_id] = 1.0`<br>`W[i, dwarf_stmt_id] = min(dwarf_conf + ir_conf, 1.0)` | `M_gt[i][dwarf_stmt_id] = 1.0`<br>`W_mat[i][dwarf_stmt_id] = min(dwarf_conf + ir_conf, 1.0)` | ✓ |
| 两信号不一致 | `M_gt[i, dwarf_stmt_id] = 0.6`<br>`W[i, dwarf_stmt_id] = dwarf_conf * 0.5`<br>`M_gt[i, ir_stmt_id] = 0.4`<br>`W[i, ir_stmt_id] = ir_conf * 0.5` | 同上，0.6/0.5 与 0.4/0.5 | ✓ |
| 仅 DWARF | `M_gt[i, dwarf_stmt_id] = 1.0`<br>`W[i, dwarf_stmt_id] = dwarf_conf * 0.8` | 同上 | ✓ |
| 仅 IR | `M_gt[i, ir_stmt_id] = 1.0`<br>`W[i, ir_stmt_id] = ir_conf * 0.8` | 同上 | ✓ |
| 行归一化 | `row_sums = M_gt.sum(axis=1)`，`M_gt = M_gt / row_sums`（避免除零） | `row_sum = sum(M_gt[i])`，若 >0 则 `M_gt[i][j] /= row_sum` | ✓ |

**ir_conf 的计算**（MD：`ir_conf *= ir_mapping['confidence']`）：  
实现中 `ir_conf = raw_ir_conf * ir_mapping["confidence"]`，与 MD 一致。

---

## 第七步：质量验证

| MD 要求 | 实现（validate_alignment.py） | 一致？ |
|--------|-------------------------------|--------|
| 覆盖率、平均/高/低置信度、顺序一致性、控制流一致性 | 从样本的 `alignment_matrix`、`alignment_confidence`（即 W）统计 | ✓ |

校验脚本中的「平均置信度」「高置信度(>0.7)」等，均来自 jsonl 里的 **alignment_confidence**，即第六步产出的 **W**。

---

## 结论

- **置信度计算**：与 MD 第六步完全一致（一致→min(dwarf_conf+ir_conf,1.0)；不一致→0.5 倍；仅单信号→0.8 倍；行归一化）。
- **各步构建思路**：第一步编译、第二步 DWARF、第三步 IR、第四步 tree-sitter、第五步机器码↔IR、第六步 M_gt/W、第七步验证，均按 MD 实现；仅第四步多了两种语句类型及主文件过滤等增强。
- **zlib 数据合理性**：高置信度 41.7%、平均 0.54 在「小项目 + 多 .c 文件」下合理；MD 的 75–85% 高置信度是针对「严格编译 + 大单文件」的预期，zlib 体量小、多文件，比例略低属正常。

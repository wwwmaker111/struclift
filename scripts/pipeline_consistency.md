# O0 与 O1-O3 流水线一致性说明

## 对齐后的流水线对比

| 项目 | O0 | O1 | O2 | O3 |
|------|-----|-----|-----|-----|
| **zlib** | --elf zlib_o0<br>--llvm-ir zlib_o0.ll | --elf zlib_o1<br>--llvm-ir zlib_o1.ll | --elf zlib_o2<br>--llvm-ir zlib_o2.ll | --elf zlib_o3<br>--llvm-ir zlib_o3.ll |
| **SQLite** | --elf sqlite_o0<br>--llvm-ir sqlite_o0.ll | --elf sqlite_o1<br>--llvm-ir sqlite_o1.ll | ... | ... |
| **BusyBox** | --elf busybox_o0<br>--llvm-ir busybox_o0.ll | --elf busybox_o1<br>--llvm-ir busybox_o1.ll | ... | ... |
| **OpenSSL** | --elf openssl_o0<br>--llvm-ir-dir ir_o0/ | --elf openssl_o1<br>--llvm-ir-dir ir_o1/ | --llvm-ir-dir ir_o2/ | --llvm-ir-dir ir_o3/ |

## 共同点

1. **无 --skip-alignment**：均使用完整 tree-sitter 语句级对齐
2. **均有 IR**：zlib/sqlite/busybox 用单文件 .ll，OpenSSL 用逐文件 ir_oN/ 目录
3. **同一 build 脚本**：`build_binskel_dataset_md.py`，参数结构一致
4. **IR 交叉验证**：DWARF + IR 共同参与 `generate_alignment_ground_truth` 和 `cross_validate`

## 差异（仅优化级别）

- **--opt**：O0 / O1 / O2 / O3
- **ELF 与 IR**：来自不同优化级别的编译产物

## 修复记录

- **O0 OpenSSL**：原先使用 `--llvm-ir openssl_o0.ll`，但 compile 只产出 `ir_o0/`，已改为 `--llvm-ir-dir ir_o0/`，与 O1-O3 一致

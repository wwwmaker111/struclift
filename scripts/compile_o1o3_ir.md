# O1-O3 IR 支持说明

## 修改内容

为使 O1-O3 数据集与 O0 一样具备 IR 交叉验证（提高可信度），已修改：

### 1. 编译脚本（需重新编译以生成 IR）

| 脚本 | 新增产出 |
|------|----------|
| `compile_zlib.sh` | zlib_o1.ll, zlib_o2.ll, zlib_o3.ll |
| `compile_sqlite.sh` | sqlite_o1.ll, sqlite_o2.ll, sqlite_o3.ll |
| `compile_busybox.sh` | busybox_o1.ll, busybox_o2.ll, busybox_o3.ll |
| `compile_openssl.sh` | ir_o1/, ir_o2/, ir_o3/ 目录 |

### 2. 构建脚本

`regen_o1o3_four_projects.sh` 已更新为自动传入 IR：
- zlib/sqlite/busybox: `--llvm-ir <path>_oN.ll`
- openssl: `--llvm-ir-dir <path>/ir_oN/`

若 IR 文件不存在，会打印警告并退化为仅用 DWARF（与之前行为一致）。

## 使用步骤

1. **重新编译四个项目**（生成 O1-O3 的 .ll / ir_oN/）：
   ```bash
   bash scripts/compile_zlib.sh
   bash scripts/compile_sqlite.sh
   bash scripts/compile_busybox.sh
   bash scripts/compile_openssl.sh
   ```

2. **构建 O1-O3 数据集**：
   ```bash
   bash scripts/regen_o1o3_four_projects.sh
   ```

## 注意事项

- BusyBox、OpenSSL 编译耗时较长（含 IR 生成）
- 若已有 O1-O3 ELF 但无 IR，需重新执行对应 compile 脚本以生成 IR

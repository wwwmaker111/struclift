# 构建监控

## 当前状态

### O0 ✅ 已完成
| 项目 | 样本数 | tree-sitter | IR |
|------|--------|-------------|-----|
| zlib | 149 | 100% | ✅ |
| sqlite | 2869 | 100% | ✅ |
| busybox | 4277 | 99.93% | ✅ |
| openssl | 16389 | 99.5% | ✅ ir_o0/ 1080 个 .ll |

### O1-O3 🔄 进行中
- zlib O1/O2/O3 ✅（完整流水线：`zlib_oN` + `zlib_oN.ll`）
- sqlite O1 🔄 进行中
- busybox、openssl 待构建

**说明**：O1-O3 的 .ll / ir_oN 需先运行 `compile_*` 脚本生成，当前退化为仅 DWARF

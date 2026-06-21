# FFmpeg 完整流水线（O0–O3 + DWARF + 逐文件 IR + binskel）

与 OpenSSL / libpng 等一致：四档优化、**bear → compile_commands.json → emit_ir**、**build_binskel_dataset_md.py**、**validate_alignment.py**。

## 行尾 CRLF（在 WSL 报 `$'\r': command not found` 时）

在仓库根执行：

```bash
sed -i 's/\r$//' scripts/compile_ffmpeg.sh scripts/build_ffmpeg_binskel.sh
```

（从 Windows/编辑器保存后常需做一次。）

## 依赖（WSL / Debian 系示例）

```bash
sudo apt update
sudo apt install -y clang make python3 bear nasm xz-utils wget curl \
  pkg-config
```

无 **nasm** 时脚本会自动加 `--disable-x86asm`（仍可编， SIMD 路径略少）。

## 一键：编译 + 数据集

```bash
export STRUCTLIFT_ROOT=/mnt/e/structlift          # 改为你的仓库根
export DATASET_ROOT="$HOME/struclift_wsl/datasets"
export OUT="$DATASET_ROOT/ffmpeg_binskel"

# 1) 编译（耗时长，默认 ffmpeg 7.1）
bash "$STRUCTLIFT_ROOT/scripts/compile_ffmpeg.sh" "$DATASET_ROOT/ffmpeg"

# 2) binskel（版本号与目录需与上一步一致）
bash "$STRUCTLIFT_ROOT/scripts/build_ffmpeg_binskel.sh" \
  "$DATASET_ROOT/ffmpeg/ffmpeg-${FFMPEG_VERSION:-7.1}" \
  "$OUT" \
  "$STRUCTLIFT_ROOT"
```

## 加速编译（多核与其它）

1. **多核并行（默认已开）**  
   脚本使用 **`make -j$(nproc)`**，并设置 **`MAKEFLAGS=-jN`**，子目录递归编译也会并行。  
   若 `nproc` 不准或想手动拉满：
   ```bash
   export FFMPEG_JOBS=32   # 改成你的逻辑 CPU 数或略少，避免内存爆）
   ```

2. **ccache（四档各编一遍时很值）**  
   ```bash
   sudo apt install -y ccache
   export FFMPEG_USE_CCACHE=1
   ```
   第一档仍要全编；**O1/O2/O3 重 configure 时大量 .o 可命中缓存**，整体能省不少时间。

3. **别把源码放在 `/mnt/e` 上编**  
   WSL 下 **ext4（例如 `$HOME/...`）** 比 **9p 的 `/mnt/e`** 快很多；可把 `BASE_DIR` 指到 `~/build/ffmpeg` 等。

4. **换不了的事实**  
   仍要 **4 次完整 configure + make + emit_ir**；要再省时间只能 **少编几种码流**（自行加 `FFMPEG_EXTRA_CONFIGURE_ARGS` 裁剪 configure），但 **样本量会下降**。

## 可选环境变量

| 变量 | 含义 |
|------|------|
| `FFMPEG_VERSION` | 源码版本，默认 `7.1` |
| `FFMPEG_JOBS` | 并行 make 线程数（默认 `nproc`） |
| `FFMPEG_USE_CCACHE=1` | 使用 ccache 包装 clang/clang++ |
| `FFMPEG_EXTRA_CONFIGURE_ARGS` | 追加传给 `./configure` 的参数 |
| `FFMPEG_FORCE_NO_ASM=1` | 强制关 x86 汇编（无 nasm 或排错时） |
| `VENV` | Python venv 的 `activate` 路径 |

## 产出

- 源码与 ELF：`$DATASET_ROOT/ffmpeg/ffmpeg-<ver>/ffmpeg_o0` … `ffmpeg_o3`
- IR 目录：`ir_o0` … `ir_o3/`
- JSONL：`$OUT/binskel_ffmpeg_o0.jsonl` … `o3.jsonl`

**四档对齐效果一键对比**（与 `validate_alignment.py` 同口径）：

```bash
cd "$STRUCTLIFT_ROOT"
python3 scripts/stats_ffmpeg_binskel_alignment.py --dir "$OUT"
# 跳过 MD5（大文件略快）: 加 --no-md5
```

## 与其它项目对比：为什么 FFmpeg 的「汇总表」有时看起来更「平」

1. **`compile_ffmpeg.sh` 在 O1–O3 仍带 `-fno-inline -fno-unroll-loops`**（与 zlib 脚本一致），有意削弱跨档的**结构剧变**；而像 SQLite / BusyBox 等工程在更高优化下 CFG、内联、IR 与 DWARF 的错位往往更大，表上 **O0 vs O1** 会拉得更开。  
2. **FFmpeg 样本量极大**，全量平均置信度、高置信比例等会 **大数定律下更稳定**，再叠加表格只印 2～3 位小数时，容易和「四档一样」混淆。  
3. **请用更高精度 + 相对 O0 的差分** 看 FFmpeg，例如（jsonl 若在单独目录要加 `--search-dir`）：
   ```bash
   python3 scripts/binskel_alignment_batch_stats.py \
     --out-dir ~/structlift_datasets \
     --search-dir "$HOME/struclift_wsl/datasets/ffmpeg_binskel"
   ```
   表中 **平均置信** 为 4 位小数，并有 **Δ均信 / Δ高%（相对同项目 O0）**；若 md5 四档 jsonl 已不同，而 Δ 仍接近 0，多半属 **编译策略 + 平均效应**，而非「四档共用同一数据集」。

## 说明

- **O0–O3 的 `ffmpeg_on` 若 `md5sum` 完全相同**：多为历史问题——仅 `export CFLAGS` 时 FFmpeg 的 `configure` 仍可能用默认 **OPTFLAGS（如 -O3）** 写进 `config.mak`，四档实际编出同一份二进制。当前 `compile_ffmpeg.sh` 已改为 **`--optflags` + `--extra-cflags`** 强制分档；发现 md5 相同后请 **删 `ffmpeg_o*` 并重跑** `compile_ffmpeg.sh`，再 `md5sum` 核对。
- **DWARF 为 0**：若曾用**默认 strip** 的 `ffmpeg`，调试信息会被剥掉。脚本已加 **`--disable-stripping`** 并优先复制 **`ffmpeg_g`**（未 strip）。若你早前编过旧脚本产物，需 **重新跑 `compile_ffmpeg.sh`** 再跑 `build_ffmpeg_binskel.sh`。
- **首次全量** `compile_ffmpeg.sh` 常见 **30–90+ 分钟**（视 CPU 与是否启用汇编而定）。
- 若 `bear` 过旧，请用发行版自带 **bear 3.x**；生成失败时检查源码根下是否有 `compile_commands.json`。
- 行尾 CRLF 若在 Windows 编辑过脚本，在 WSL 执行：`sed -i 's/\r$//' scripts/compile_ffmpeg.sh scripts/build_ffmpeg_binskel.sh`。

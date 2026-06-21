#!/usr/bin/env bash
# =============================================================================
# 重建「训练损失所需」的 binskel 数据集（对齐 + 可选 Stage3/4 SFT 字段）
#
# 与「以前删掉的总控脚本」/ 45 档总线 的关系
#   - 本脚本是对仓库内 **现存的** compile_*.sh / build_*_binskel.sh / augment_binskel_sft.py
#     的**编排**。
#   - **SFT/RL 张量算法**（sft_input_ids、sft_labels、slot_types、emit-rl 三字段）与
#     scripts/augment_all_45_to_full_sft.py 子进程 **完全一致**：同一 `augment_binskel_sft.py`、
#     同默认 tokenizer（SFT_TOKENIZER）、同 `--emit-rl-fields`（EMIT_RL_FIELDS=1）、
#     **未**传 `--mask-prompt-labels` / `--mask-non-fill-labels` / `--max-seq-len`（与 45 脚本一致，长度策略见 augment 内 `_effective_sft_max_len`）。
#   - **另一路径** `FULL_STAGES_JSONL=1` + `_binskel_md_extra.sh` 在 build_binskel_dataset_md 内联 SFT：
#     当前默认各 build_*_binskel.sh **未** source 该片段，故本脚本采用的仍是 **augment 路径**（与 augment_all_45 一致）。
#   - **稳定别名（恢复入口名）**：scripts/total_control_rebuild_loss_datasets.sh（exec 本文件）
#   - **仓库根便捷**：./run_rebuild_loss_datasets.sh
#   - **mbedtls「先删再 clone」独立步骤**：scripts/force_mbedtls_git_reclone.sh（与 compile_mbedtls.sh 内删目录逻辑一致）
#   - 每条样本的字段由 **build_binskel_dataset_md.py** + **augment_binskel_sft.py** 决定；
#     你贴的样例里 ``binary`` 含 ``.../multibin_o0/...`` → 对应 **coreutils multibin**
#     （compile_coreutils_multibin + build_coreutils_multibin_binskel），
#     **不是** 单 ELF 的 ``coreutils_os`` / ``os_only/build_coreutils_os_binskel.sh``。
#
# 流程概览
#   1) [可选] 编译：zlib / openssl / sqlite / coreutils **multibin**（O0–O3+Os）；可选再编 coreutils_os 单档
#   2) build_*_binskel.sh → jsonl（含 alignment_matrix、pattern_labels、src_* …）
#   3) [可选] augment_binskel_sft.py → sft_input_ids / sft_labels / slot_types（+ 可选 bin_pattern_dist 等 RL 字段）
#
# 依赖（Ubuntu/WSL）
#   sudo apt-get install -y build-essential clang llvm wget curl unzip xz-utils
#   pip install -r requirements.txt
#
# 用法
#   chmod +x scripts/rebuild_loss_datasets_full.sh
#   export WORKDIR=/path/to/structlift
#   export DATA_ROOT=$HOME/struclift_wsl/datasets
#   export OUT_DIR=$HOME/structlift_datasets
#   ./scripts/rebuild_loss_datasets_full.sh
#
# 仅重建 jsonl、不重新编译（ELF 已存在）：
#   SKIP_COMPILE=1 ./scripts/rebuild_loss_datasets_full.sh
#
# 跳过 SFT 增补（只做对齐 jsonl，供 Stage1/2）：
#   SKIP_AUGMENT_SFT=1 ./scripts/rebuild_loss_datasets_full.sh
#
# coreutils 源码不在默认路径时（例如编译在 /tmp）：
#   export CU_BUILD_DIR=/tmp/structlift_coreutils_build/coreutils-9.4
# =============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$HOME/struclift_wsl/datasets}"
OUT_DIR="${OUT_DIR:-$HOME/structlift_datasets}"
VENV="${VENV:-$WORKDIR/.venv/bin/activate}"

SKIP_COMPILE="${SKIP_COMPILE:-0}"
SKIP_AUGMENT_SFT="${SKIP_AUGMENT_SFT:-0}"
DO_ZLIB="${DO_ZLIB:-1}"
DO_OPENSSL="${DO_OPENSSL:-1}"
DO_SQLITE="${DO_SQLITE:-1}"
# 与你成功样本一致：coreutils 用 multibin_o0…（每档目录下多个 ELF，再合并 jsonl）
DO_COREUTILS_MULTIBIN="${DO_COREUTILS_MULTIBIN:-1}"
# 单二进制 coreutils_os（os_only），与 multibin 不同；默认关，需要时 DO_COREUTILS_OS=1
DO_COREUTILS_OS="${DO_COREUTILS_OS:-0}"

# augment_binskel_sft 使用（与 train_stage3、augment_all_45_to_full_sft.py 默认一致）
SFT_TOKENIZER="${SFT_TOKENIZER:-deepseek-ai/deepseek-coder-6.7b-base}"
EMIT_RL_FIELDS="${EMIT_RL_FIELDS:-1}"
# 若设置（例如 16384），则传给 augment --max-seq-len；未设置则与 augment_all_45 行为一致（不显式传参）
SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-}"
# 默认只监督 FILL 槽位真值（并屏蔽 prompt）；设 0 可恢复旧行为（整段 labels）。
SFT_MASK_PROMPT_LABELS="${SFT_MASK_PROMPT_LABELS:-1}"
SFT_MASK_NON_FILL_LABELS="${SFT_MASK_NON_FILL_LABELS:-1}"

log() { echo "[rebuild-loss] $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { log "缺少命令: $1"; exit 1; }
}

activate_venv() {
  if [[ -f "$VENV" ]]; then
    # shellcheck source=/dev/null
    source "$VENV"
  fi
}

run_augment() {
  local inp="$1"
  local out="$2"
  local src_root="$3"
  [[ -f "$inp" ]] || { log "跳过 augment（无文件）: $inp"; return 0; }
  log "SFT 增补: $inp -> $out"
  local _ms=()
  local _ml=()
  [[ -n "${SFT_MAX_SEQ_LEN:-}" ]] && _ms=(--max-seq-len "$SFT_MAX_SEQ_LEN")
  [[ "${SFT_MASK_PROMPT_LABELS}" == "1" ]] && _ml+=(--mask-prompt-labels)
  [[ "${SFT_MASK_NON_FILL_LABELS}" == "1" ]] && _ml+=(--mask-non-fill-labels)
  python "$WORKDIR/scripts/augment_binskel_sft.py" \
    --input "$inp" \
    --output "$out" \
    --tokenizer "$SFT_TOKENIZER" \
    --source-root "$src_root" \
    ${EMIT_RL_FIELDS:+--emit-rl-fields} \
    "${_ml[@]}" \
    "${_ms[@]}"
}

main() {
  need_cmd bash
  need_cmd python3
  [[ -d "$WORKDIR" ]] || { log "WORKDIR 不存在: $WORKDIR"; exit 1; }
  [[ -f "$WORKDIR/build_binskel_dataset_md.py" ]] || { log "缺少 build_binskel_dataset_md.py"; exit 1; }

  mkdir -p "$OUT_DIR"
  activate_venv

  if [[ "$SKIP_COMPILE" != "1" ]]; then
    need_cmd clang
    command -v llvm-link >/dev/null 2>&1 || { log "缺少 llvm-link，请: sudo apt-get install -y llvm"; exit 1; }
    command -v llvm-dis >/dev/null 2>&1 || { log "缺少 llvm-dis"; exit 1; }
  fi

  # --- zlib ---
  if [[ "$DO_ZLIB" == "1" ]]; then
    ZLIB_DIR="$DATA_ROOT/zlib/zlib-${ZLIB_VERSION:-1.3.1}"
    if [[ "$SKIP_COMPILE" != "1" ]]; then
      log "编译 zlib ..."
      bash "$WORKDIR/scripts/compile_zlib.sh" "$DATA_ROOT/zlib"
    fi
    log "构建 binskel: zlib O0–O3 ..."
    bash "$WORKDIR/scripts/build_zlib_binskel.sh" "$ZLIB_DIR" "$OUT_DIR" "$WORKDIR"
    if [[ "$SKIP_AUGMENT_SFT" != "1" ]]; then
      for n in 0 1 2 3; do
        f="$OUT_DIR/binskel_zlib_o${n}.jsonl"
        [[ -f "$f" ]] || continue
        run_augment "$f" "$OUT_DIR/binskel_zlib_o${n}_sft.jsonl" "$ZLIB_DIR"
      done
    fi
  fi

  # --- openssl ---
  if [[ "$DO_OPENSSL" == "1" ]]; then
    SSL_VER="${SSL_VERSION:-3.3.1}"
    SSL_DIR="$DATA_ROOT/openssl/openssl-${SSL_VER}"
    if [[ "$SKIP_COMPILE" != "1" ]]; then
      log "编译 openssl ..."
      bash "$WORKDIR/scripts/compile_openssl.sh" "$DATA_ROOT/openssl"
    fi
    log "构建 binskel: openssl O0–O3 ..."
    bash "$WORKDIR/scripts/build_openssl_binskel.sh" "$SSL_DIR" "$OUT_DIR" "$WORKDIR"
    if [[ "$SKIP_AUGMENT_SFT" != "1" ]]; then
      for n in 0 1 2 3; do
        f="$OUT_DIR/binskel_openssl_o${n}.jsonl"
        [[ -f "$f" ]] || continue
        run_augment "$f" "$OUT_DIR/binskel_openssl_o${n}_sft.jsonl" "$SSL_DIR"
      done
    fi
  fi

  # --- sqlite ---
  if [[ "$DO_SQLITE" == "1" ]]; then
    SQLITE_VER="${SQLITE_VERSION:-3450100}"
    SQLITE_DIR="$DATA_ROOT/sqlite/sqlite-${SQLITE_VER}"
    if [[ "$SKIP_COMPILE" != "1" ]]; then
      log "编译 sqlite ..."
      bash "$WORKDIR/scripts/compile_sqlite.sh" "$DATA_ROOT/sqlite"
    fi
    log "构建 binskel: sqlite O0–O3 ..."
    bash "$WORKDIR/scripts/build_sqlite_binskel.sh" "$SQLITE_DIR" "$OUT_DIR" "$WORKDIR"
    if [[ "$SKIP_AUGMENT_SFT" != "1" ]]; then
      for n in 0 1 2 3; do
        f="$OUT_DIR/binskel_sqlite_o${n}.jsonl"
        [[ -f "$f" ]] || continue
        run_augment "$f" "$OUT_DIR/binskel_sqlite_o${n}_sft.jsonl" "$SQLITE_DIR"
      done
    fi
  fi

  # --- coreutils MULTIBIN（O0–O3 + Os，与样例 binary 路径 multibin_o* 一致）---
  if [[ "$DO_COREUTILS_MULTIBIN" == "1" ]]; then
    CU_VER="${COREUTILS_VERSION:-9.4}"
    CU_DIR="${CU_BUILD_DIR:-$DATA_ROOT/coreutils/coreutils-${CU_VER}}"
    if [[ "$SKIP_COMPILE" != "1" ]]; then
      log "编译 coreutils multibin（产出 multibin_o0..o3、multibin_os、ir_o*）..."
      bash "$WORKDIR/scripts/compile_coreutils_multibin.sh" "$DATA_ROOT/coreutils"
    fi
    log "构建 binskel: coreutils multibin 合并 jsonl ..."
    bash "$WORKDIR/scripts/build_coreutils_multibin_binskel.sh" "$CU_DIR" "$OUT_DIR" "$WORKDIR"
    if [[ "$SKIP_AUGMENT_SFT" != "1" ]]; then
      for n in 0 1 2 3; do
        f="$OUT_DIR/binskel_coreutils_o${n}_multibin.jsonl"
        [[ -f "$f" ]] || continue
        run_augment "$f" "$OUT_DIR/binskel_coreutils_o${n}_multibin_sft.jsonl" "$CU_DIR"
      done
      f="$OUT_DIR/binskel_coreutils_os_multibin.jsonl"
      [[ -f "$f" ]] && run_augment "$f" "$OUT_DIR/binskel_coreutils_os_multibin_sft.jsonl" "$CU_DIR" || true
    fi
  fi

  # --- coreutils Os（单 ELF，仅当你需要 os_only 那条线时打开 DO_COREUTILS_OS=1）---
  if [[ "$DO_COREUTILS_OS" == "1" ]]; then
    CU_VER="${COREUTILS_VERSION:-9.4}"
    # WSL 在 /mnt 编译时 compile_coreutils 可能用 /tmp/structlift_coreutils_build/coreutils-* ，可设 CU_BUILD_DIR
    CU_DIR="${CU_BUILD_DIR:-$DATA_ROOT/coreutils/coreutils-${CU_VER}}"
    if [[ "$SKIP_COMPILE" != "1" ]]; then
      log "编译 coreutils（标准 O0–O3，供源码树；WSL 在 /mnt 下可能自动用 /tmp 构建）..."
      bash "$WORKDIR/scripts/compile_coreutils.sh" "$DATA_ROOT/coreutils"
    fi
    if [[ ! -f "$CU_DIR/coreutils_os" ]]; then
      log "编译 coreutils Os（产出 coreutils_os + ir_os）..."
      bash "$WORKDIR/scripts/os_only/compile_os_coreutils.sh" "$CU_DIR" "$CU_DIR"
    else
      log "已存在 coreutils_os，跳过 compile_os_coreutils"
    fi
    log "构建 binskel: coreutils Os ..."
    bash "$WORKDIR/scripts/os_only/build_coreutils_os_binskel.sh" "$CU_DIR" "$OUT_DIR" "$WORKDIR"
    if [[ "$SKIP_AUGMENT_SFT" != "1" ]]; then
      f="$OUT_DIR/binskel_coreutils_os.jsonl"
      [[ -f "$f" ]] && run_augment "$f" "$OUT_DIR/binskel_coreutils_os_sft.jsonl" "$CU_DIR" || true
    fi
  fi

  log "完成。输出目录: $OUT_DIR"
  log "对齐: binskel_*；coreutils multibin: binskel_coreutils_o{0..3}_multibin.jsonl"
  log "含 SFT/RL（Stage3/4）: *_sft.jsonl 或 *_multibin_sft.jsonl"
}

main "$@"

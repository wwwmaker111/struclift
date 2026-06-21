#!/usr/bin/env bash
#
# 分三组构建 binskel，且每条 jsonl 直接包含 Stage1–4 训练字段（SFT+RL）。
# 依赖：仓库根执行；已编译各项目；FULL_STAGES_JSONL=1 时必须已安装 transformers：
#   source ~/struclift_wsl/.venv/bin/activate && pip install transformers
#
# 原理：export FULL_STAGES_JSONL=1 → scripts/_binskel_md_extra.sh 会给
#       build_binskel_dataset_md.py 追加 --full-stages-jsonl（生成时内联 SFT，无需再跑 augment）
#
# 下面 *_DIR 与 scripts/compile_sqlite.sh（sqlite-3450100）、compile_coreutils.sh、
# rebuild_binskel_all_9.sh 头注释、docs/BUILD_BINSKEL_9x5_THREE_PARTS.md 一致（$HOME/struclift_wsl/datasets）。
# 若你本机目录名不同，只改对应 export，不要改脚本逻辑。
#
set -euo pipefail

# ── 公共（三组保持一致）──────────────────────────────────────────────────
# export STRUCTLIFT=/mnt/e/structlift
# export OUT_ROOT=/mnt/e/structlift_datasets
# export FULL_STAGES_JSONL=1
# export SFT_TOKENIZER=deepseek-ai/deepseek-coder-6.7b-base
# # 可选：从 jsonl 路径里剥前缀
# # export STRIP_PATH_PREFIX=/home/wuqiongmin/

# ═══════════════════════════════════════════════════════════════════════════
# 第一部分：仅 coreutils（multibin）
# ═══════════════════════════════════════════════════════════════════════════
# cd "$STRUCTLIFT"
# export OUT_ROOT=/mnt/e/structlift_datasets
# export FULL_STAGES_JSONL=1
# export COREUTILS_DIR="$HOME/struclift_wsl/datasets/coreutils/coreutils-9.4"
# bash scripts/build_coreutils_multibin_binskel.sh "$COREUTILS_DIR" "$OUT_ROOT" "$STRUCTLIFT"

# ═══════════════════════════════════════════════════════════════════════════
# 第二部分：ffmpeg（multibin）+ openssl + zlib + sqlite（各含 O0–O3 + Os 脚本）
# ═══════════════════════════════════════════════════════════════════════════
# cd "$STRUCTLIFT"
# export OUT_ROOT=/mnt/e/structlift_datasets
# export FULL_STAGES_JSONL=1
# export FFMPEG_DIR="$HOME/struclift_wsl/datasets/ffmpeg/ffmpeg-7.1"
# export OPENSSL_DIR="$HOME/struclift_wsl/datasets/openssl/openssl-3.3.1"
# export ZLIB_DIR="$HOME/struclift_wsl/datasets/zlib/zlib-1.3.1"
# export SQLITE_DIR="$HOME/struclift_wsl/datasets/sqlite/sqlite-3450100"
#
# bash scripts/build_ffmpeg_multibin_binskel.sh  "$FFMPEG_DIR"  "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/build_openssl_binskel.sh         "$OPENSSL_DIR" "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/os_only/build_openssl_os_binskel.sh "$OPENSSL_DIR" "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/build_zlib_binskel.sh              "$ZLIB_DIR"    "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/os_only/build_zlib_os_binskel.sh   "$ZLIB_DIR"    "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/build_sqlite_binskel.sh            "$SQLITE_DIR"  "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/os_only/build_sqlite_os_binskel.sh "$SQLITE_DIR"  "$OUT_ROOT" "$STRUCTLIFT"

# ═══════════════════════════════════════════════════════════════════════════
# 第三部分：curl + libxml2 + busybox + openssh（multibin）
# ═══════════════════════════════════════════════════════════════════════════
# cd "$STRUCTLIFT"
# export OUT_ROOT=/mnt/e/structlift_datasets
# export FULL_STAGES_JSONL=1
# export CURL_DIR="$HOME/struclift_wsl/datasets/curl/curl-8.7.1"
# export LIBXML2_DIR="$HOME/struclift_wsl/datasets/libxml2/libxml2-2.12.7"
# export BUSYBOX_DIR="$HOME/struclift_wsl/datasets/busybox/busybox-1.36.1"
# export OPENSSH_DIR="$HOME/struclift_wsl/datasets/openssh/openssh-9.8p1"
#
# bash scripts/build_curl_binskel.sh              "$CURL_DIR"    "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/os_only/build_curl_os_binskel.sh   "$CURL_DIR"    "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/build_libxml2_binskel.sh           "$LIBXML2_DIR" "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/os_only/build_libxml2_os_binskel.sh "$LIBXML2_DIR" "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/build_busybox_binskel.sh           "$BUSYBOX_DIR" "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/os_only/build_busybox_os_binskel.sh "$BUSYBOX_DIR" "$OUT_ROOT" "$STRUCTLIFT"
# bash scripts/build_openssh_multibin_binskel.sh  "$OPENSSH_DIR" "$OUT_ROOT" "$STRUCTLIFT"

# 取消注释并运行其中一部分即可。

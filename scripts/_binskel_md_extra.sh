#!/usr/bin/env bash
# 由 build_*_binskel.sh source。在 source 之前请已设置 WORKDIR（仓库根）、
# WORKERS / NUM_OPCODES / SRC_VOCAB（可选）。
#
# 可选环境变量 STRIP_PATH_PREFIX：
#   （曾计划）传给 build_binskel_dataset_md.py --strip-path-prefix；
#   当前 Python 端未实现该项；跨机器请改用：
#     scripts/normalize_binskel_source_paths.py   或
#     scripts/augment_binskel_sft.py --rewrite-source-prefix OLD NEW
#
# shellcheck shell=bash
: "${WORKERS:=8}"
: "${NUM_OPCODES:=1024}"
: "${SRC_VOCAB:=32000}"
# 设为 1 时：build_binskel_dataset_md 写出 Stage1–4 全部字段（SFT+RL），无需再跑 augment_binskel_sft
: "${FULL_STAGES_JSONL:=0}"
: "${SFT_TOKENIZER:=deepseek-ai/deepseek-coder-6.7b-base}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")
if [[ -n "${STRIP_PATH_PREFIX:-}" ]]; then
  MD_EXTRA+=(--strip-path-prefix "$STRIP_PATH_PREFIX")
fi
if [[ "${FULL_STAGES_JSONL}" == "1" ]]; then
  MD_EXTRA+=(--full-stages-jsonl --sft-tokenizer "$SFT_TOKENIZER")
  # 与 build_*_binskel.sh 一致：先激活 VENV（若存在），再检查依赖，避免跑几百个 ELF 才报错
  _binskel_venv="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
  if [[ -n "${_binskel_venv}" && -f "${_binskel_venv}" ]]; then
    # shellcheck source=/dev/null
    source "${_binskel_venv}"
  fi
  _py=python
  command -v python >/dev/null 2>&1 || _py=python3
  if ! "$_py" -c "import transformers" 2>/dev/null; then
    echo "错误: FULL_STAGES_JSONL=1 需要 Python 包 transformers（用于 SFT tokenizer）。" >&2
    echo "  当前: $(command -v "$_py" 2>/dev/null || echo 未找到 python/python3)" >&2
    echo "  安装: pip install transformers" >&2
    [[ -f "${_binskel_venv}" ]] && echo "  例: source ${_binskel_venv} && pip install transformers" >&2
    exit 1
  fi
fi

#!/usr/bin/env bash
# 扫描本机常见 conda envs 目录，检查是否适合 StrucLift 模块 C（Stage3）及 DeepSeek-V4。
# 用法: bash scripts/scan_conda_for_module_c.sh
#
# 说明（看输出自己判断）:
#   - 一般 Stage3（6.7B 等）: cuda_available=True，有 transformers，有 peft 更佳
#   - DeepSeek-V4-Flash(FP8 权重): 需 float8_e8m0fnu=True 且 deepseek_v4_registered=True

set -uo pipefail

PY_SNIPPET='
import sys
def line(k, v):
    print(f"  {k}: {v}")

try:
    import torch
    line("torch", torch.__version__)
    line("cuda_available", torch.cuda.is_available())
    line("float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None) is not None)
except Exception as e:
    line("torch", f"ERROR {e}")

try:
    import transformers
    line("transformers", transformers.__version__)
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        line("deepseek_v4_registered", "deepseek_v4" in CONFIG_MAPPING)
    except Exception as e:
        line("deepseek_v4_registered", f"check_failed: {e}")
except Exception as e:
    line("transformers", f"ERROR {e}")

try:
    import peft
    line("peft", peft.__version__)
except Exception:
    line("peft", "NOT_INSTALLED")
'

roots=(
  "/data/chaoni/miniconda3/envs"
  "/home/cn/miniconda3/envs"
  "/home/cn/miniforge3/envs"
)

echo "==================================================================="
echo " StrucLift 模块 C / Stage3  —  conda 环境探测"
echo " （torch · CUDA · transformers · deepseek_v4 · peft）"
echo "==================================================================="
echo

for root in "${roots[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "[跳过] 目录不存在: $root"
    continue
  fi
  for envdir in "$root"/*; do
    [[ -d "$envdir" ]] || continue
    py="$envdir/bin/python"
    if [[ ! -x "$py" ]]; then
      continue
    fi
    echo "-------------------------------------------------------------------"
    echo "PYTHON: $py"
    "$py" -c "$PY_SNIPPET" 2>&1 || echo "  (该 python 执行失败)"
  done
done

echo
echo "-------------------------------------------------------------------"
echo "怎么读:"
echo "  • DeepSeek-V4-Flash（你当前的 FP8 权重）: float8_e8m0fnu=True 且 deepseek_v4_registered=True"
echo "  • 普通 HF LLM Stage3（如 deepseek-coder-6.7b）: cuda_available=True + transformers 已安装即可试"
echo "  • peft=NOT_INSTALLED: 仍可跑 Stage3，但多为冻结 LLM；建议 pip install peft"
echo "==================================================================="

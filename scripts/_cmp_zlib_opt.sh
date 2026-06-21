#!/usr/bin/env bash
# 对比 zlib binskel O0–O3 + Os 的 validate_alignment 摘要
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/struclift_wsl/.venv/bin/activate}"
DATA="${1:-$HOME/structlift_datasets}"
# shellcheck disable=SC1090
[ -f "$VENV" ] && source "$VENV" || true
for f in o0 o1 o2 o3 os; do
  j="$DATA/binskel_zlib_${f}.jsonl"
  echo "===== zlib_${f} ($j) ====="
  python "$ROOT/scripts/validate_alignment.py" "$j" | tail -12
  echo
done

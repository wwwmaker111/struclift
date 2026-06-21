#!/usr/bin/env bash
set -euo pipefail

cd /data/chaoni/WQM/model_code/structlift

export PYTHON="${PYTHON:-/data/chaoni/miniconda3/envs/DeepseekV4_env/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export DATA_DIR="${DATA_DIR:-/data/chaoni/WQM/datasets/ab_srcfix_patternfix_final_named}"
export A_TRAIN="${A_TRAIN:-$DATA_DIR/A_train_repaired_2048.jsonl}"
export A_VAL="${A_VAL:-$DATA_DIR/A_val_repaired.jsonl}"
export A_CKPT="${A_CKPT:-/data/chaoni/WQM/checkpoints/AB_repairedA_from_fullStage1_oldB_merged_20260530_081248/AB_repairedA_oldB_model_only.pt}"

export OUT_ROOT="${OUT_ROOT:-/data/chaoni/WQM/audits/bskel_smoke100_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

echo "===== B_skel smoke100 ====="
echo "OUT_ROOT=$OUT_ROOT"
echo "PYTHON=$PYTHON"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "A_TRAIN=$A_TRAIN"
echo "A_VAL=$A_VAL"
echo "A_CKPT=$A_CKPT"

"$PYTHON" -m py_compile \
  struclift/models/module_b_skel.py \
  scripts/export_cst_supervision_from_mgt.py \
  scripts/train_module_b_skel.py

echo
echo "===== export train teacher 100 ====="
"$PYTHON" -u scripts/export_cst_supervision_from_mgt.py \
  --jsonl "$A_TRAIN" \
  --out-jsonl "$OUT_ROOT/train_first100_cst_teacher.jsonl" \
  --out-txt "$OUT_ROOT/train_first100_cst_teacher_preview.txt" \
  --num-examples 100 \
  --limit 3000 \
  --source-control-filter bb-start-line \
  --min-source-controls 1 \
  --preview-examples 100 \
  --skip-errors \
  2>&1 | tee "$OUT_ROOT/export_train_teacher.log"

echo
echo "===== export val teacher 100 ====="
"$PYTHON" -u scripts/export_cst_supervision_from_mgt.py \
  --jsonl "$A_VAL" \
  --out-jsonl "$OUT_ROOT/val_first100_cst_teacher.jsonl" \
  --out-txt "$OUT_ROOT/val_first100_cst_teacher_preview.txt" \
  --num-examples 100 \
  --limit 3000 \
  --source-control-filter bb-start-line \
  --min-source-controls 1 \
  --preview-examples 100 \
  --skip-errors \
  2>&1 | tee "$OUT_ROOT/export_val_teacher.log"

echo
echo "===== train B_skel smoke ====="
"$PYTHON" -u scripts/train_module_b_skel.py \
  --train-jsonl "$A_TRAIN" \
  --train-teacher-jsonl "$OUT_ROOT/train_first100_cst_teacher.jsonl" \
  --val-jsonl "$A_VAL" \
  --val-teacher-jsonl "$OUT_ROOT/val_first100_cst_teacher.jsonl" \
  --a-ckpt "$A_CKPT" \
  --save-dir "$OUT_ROOT/bskel_train" \
  --device cuda \
  --epochs 5 \
  --batch-size 8 \
  --max-train-examples 100 \
  --max-val-examples 100 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --ctrl-weight 1.0 \
  --role-weight 1.0 \
  --slot-weight 0.5 \
  --parent-weight 1.0 \
  --tree-weight 0.1 \
  --region-weight 0.5 \
  --same-slot-weight 0.5 \
  --slot-ot-weight 0.3 \
  --role-pair-weight 0.3 \
  --log-every 5 \
  --export-predictions 100 \
  2>&1 | tee "$OUT_ROOT/train_bskel_smoke.log"

echo
echo "===== output check ====="
find "$OUT_ROOT" -maxdepth 2 -type f -printf "%p %k KB\n" | sort
echo
echo "OUT_ROOT=$OUT_ROOT"
echo "teacher_train=$OUT_ROOT/train_first100_cst_teacher_preview.txt"
echo "teacher_val=$OUT_ROOT/val_first100_cst_teacher_preview.txt"
echo "prediction_preview=$OUT_ROOT/bskel_train/bskel_v1_prediction_preview.txt"

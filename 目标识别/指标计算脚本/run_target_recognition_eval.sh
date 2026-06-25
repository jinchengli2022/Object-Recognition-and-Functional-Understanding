#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/amadeus/anaconda3/envs/sam3d-objects/bin/python}"
DEVICE="${DEVICE:-0}"
PRED_DIR="$BASE_DIR/results/recomputed_target_predictions"
EVAL_DIR="$BASE_DIR/results/recomputed_target_eval"
"$PYTHON_BIN" "$BASE_DIR/指标计算脚本/predict_yolo_seg_from_manifest.py" \
  --weights "$BASE_DIR/best_model/target_yolo_seg_best.pt" \
  --gt-manifest "$BASE_DIR/test_set/manifest.tsv" \
  --out-dir "$PRED_DIR" \
  --device "$DEVICE" \
  --conf 0.05 \
  --imgsz 768
"$PYTHON_BIN" "$BASE_DIR/指标计算脚本/evaluate_target_recognition_miou.py" \
  --gt-manifest "$BASE_DIR/test_set/manifest.tsv" \
  --pred-manifest "$PRED_DIR/predictions.tsv" \
  --out-dir "$EVAL_DIR"
"$PYTHON_BIN" "$BASE_DIR/指标计算脚本/compute_threshold_accuracy.py" \
  --per-instance "$EVAL_DIR/per_instance_results.tsv" \
  --out-json "$EVAL_DIR/threshold_accuracy_summary.json"

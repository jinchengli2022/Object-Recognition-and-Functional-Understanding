#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_DEVICE="${TARGET_DEVICE:-0}"
FUNCTION_DEVICE="${FUNCTION_DEVICE:-cuda:0}"
TARGET_VISUALIZE_LIMIT="${TARGET_VISUALIZE_LIMIT:-0}"
FUNCTION_VISUALIZE_MAX="${FUNCTION_VISUALIZE_MAX:-0}"
VISUALIZATION_RESPONSE_THRESHOLD="${VISUALIZATION_RESPONSE_THRESHOLD:-0.477221595}"

SRC_DIR="$BASE_DIR/src"
MODEL_DIR="$BASE_DIR/model"
DATASET_DIR="$BASE_DIR/dataset"
RESULTS_DIR="$BASE_DIR/results"
TARGET_PRED_DIR="$RESULTS_DIR/target_recognition/predictions"
TARGET_EVAL_DIR="$RESULTS_DIR/target_recognition/evaluation"
FUNCTION_EVAL_DIR="$RESULTS_DIR/functional_understanding"
VISUALIZATION_DIR="$RESULTS_DIR/visualizations"
VISUALIZATION_HISTORY_DIR="$RESULTS_DIR/visualization_history"

# 将第三方库的运行时配置和缓存限制在测试包的 results/ 中。
export YOLO_CONFIG_DIR="$RESULTS_DIR/runtime_config/Ultralytics"
export HF_HOME="$RESULTS_DIR/runtime_cache/huggingface"
export TRANSFORMERS_CACHE="$RESULTS_DIR/runtime_cache/huggingface/transformers"
export MPLCONFIGDIR="$RESULTS_DIR/runtime_config/matplotlib"

# 强制功能理解使用包内 RoBERTa，运行时不访问外网或外部模型缓存。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

# 将上一轮可视化非破坏性移入历史目录，确保本轮的 obj_XXX
# 子目录只在对应物品完成时逐个出现。
if [[ -d "$VISUALIZATION_DIR" ]] \
  && [[ -n "$(find "$VISUALIZATION_DIR" -mindepth 1 -print -quit)" ]]; then
  VISUALIZATION_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$VISUALIZATION_HISTORY_DIR"
  mv \
    "$VISUALIZATION_DIR" \
    "$VISUALIZATION_HISTORY_DIR/$VISUALIZATION_RUN_ID"
fi

mkdir -p \
  "$YOLO_CONFIG_DIR" \
  "$HF_HOME" \
  "$MPLCONFIGDIR" \
  "$TARGET_PRED_DIR" \
  "$TARGET_EVAL_DIR" \
  "$FUNCTION_EVAL_DIR" \
  "$VISUALIZATION_DIR"
cd "$BASE_DIR"

"$PYTHON_BIN" - <<'PY'
import cv2
import matplotlib
import numpy
import scipy
import torch
import transformers
import ultralytics
import yaml
from PIL import Image

print("Python 环境检查通过")
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"Ultralytics: {ultralytics.__version__}")
PY

echo
echo "[1/2] 按物品运行目标识别与功能理解联合评测..."
"$PYTHON_BIN" "$SRC_DIR/evaluate_objectwise.py" \
  --data-root "$DATASET_DIR" \
  --target-weights "$MODEL_DIR/target_recognition/target_yolo_seg_best.pt" \
  --functional-config "$SRC_DIR/functional_understanding/config/evaluation.yaml" \
  --functional-checkpoint "$MODEL_DIR/functional_understanding/best_affordance_model.pt" \
  --target-out-dir "$TARGET_PRED_DIR" \
  --functional-out-dir "$FUNCTION_EVAL_DIR" \
  --target-visualize-dir "$VISUALIZATION_DIR" \
  --functional-visualize-dir "$VISUALIZATION_DIR" \
  --target-device "$TARGET_DEVICE" \
  --functional-device "$FUNCTION_DEVICE" \
  --conf 0.05 \
  --imgsz 768 \
  --target-visualize-limit "$TARGET_VISUALIZE_LIMIT" \
  --functional-visualize-max "$FUNCTION_VISUALIZE_MAX" \
  --visualization-response-threshold "$VISUALIZATION_RESPONSE_THRESHOLD" \
  --target-correct 857 \
  --chosen-distance-threshold 0.02 \
  --distance-thresholds "0.02,0.03,0.04,0.05,0.06,0.08,0.10" \
  > /dev/null

echo
echo "[2/2] 计算并汇总最终准确率..."
"$PYTHON_BIN" "$SRC_DIR/target_recognition/scripts/evaluate_target_recognition_miou.py" \
  --gt-manifest "$DATASET_DIR/target_recognition_manifest.tsv" \
  --pred-manifest "$TARGET_PRED_DIR/predictions.tsv" \
  --out-dir "$TARGET_EVAL_DIR" \
  > /dev/null

"$PYTHON_BIN" "$SRC_DIR/target_recognition/scripts/compute_threshold_accuracy.py" \
  --per-instance "$TARGET_EVAL_DIR/per_instance_results.tsv" \
  --out-json "$TARGET_EVAL_DIR/threshold_accuracy_summary.json" \
  > /dev/null

"$PYTHON_BIN" "$SRC_DIR/summarize_results.py"

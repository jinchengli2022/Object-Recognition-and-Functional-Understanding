#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/amadeus/anaconda3/envs/sam3d-objects/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
cd "$BASE_DIR/code"
"$PYTHON_BIN" scripts/evaluate_affordance_model_functional.py \
  --data-root "$BASE_DIR/test_set" \
  --config "$BASE_DIR/code/config/train_affordance_model.yaml" \
  --ckpt "$BASE_DIR/best_model/best_affordance_model.pt" \
  --out-dir "$BASE_DIR/results/recomputed_functional_eval" \
  --split test \
  --device "$DEVICE" \
  --target-correct 857 \
  --chosen-distance-threshold 0.02 \
  --distance-thresholds "0.02,0.03,0.04,0.05,0.06,0.08,0.10"

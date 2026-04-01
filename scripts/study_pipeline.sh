#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_DIR="${1:-results/study_pipeline}"
DEVICE="${2:-cuda}"
SEEDS="${3:-42,123,456}"
NUM_ROUNDS="${4:-100}"

mkdir -p "$OUTPUT_DIR"

python scripts/run_five_stages.py \
  --stages 1 \
  --device "$DEVICE" \
  --seeds "$SEEDS" \
  --num_rounds 50 \
  --output_dir "$OUTPUT_DIR/phase1"

python scripts/run_five_stages.py \
  --stages all \
  --device "$DEVICE" \
  --seeds "$SEEDS" \
  --num_rounds "$NUM_ROUNDS" \
  --output_dir "$OUTPUT_DIR/five_stages"

python scripts/analyze_results.py select-best \
  --results_root "$OUTPUT_DIR" \
  --output_dir "$OUTPUT_DIR/summary"

python scripts/analyze_results.py plot \
  --best_runs "$OUTPUT_DIR/summary/best_runs.json" \
  --output_dir "$OUTPUT_DIR/summary/plots" \
  --format pdf

echo "study pipeline completed: $OUTPUT_DIR"

#!/bin/bash
set -e
echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"
echo "Running data processing..."
python3 src/01_data_processing.py
echo "Running model training..."
python3 src/02_train.py
echo "Running consensus evaluation..."
python3 src/03_consensus_eval.py
echo "Pipeline finished successfully."
echo "Launching Gradio interface..."
python3 src/04_inference.py
echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
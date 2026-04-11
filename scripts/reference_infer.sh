#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m medvlm_classifier.eval.reference_infer \
  --model_id "llava-hf/llava-1.5-7b-hf" \
  --data_path "examples/reference_like_eval.jsonl" \
  --class_path "examples/reference_like_classes.json" \
  --split "valid" \
  --output_path "outputs/reference_style_preds.jsonl" \
  --batch_size 1 \
  --including_label \
  --n_labels 2 \
  --resume

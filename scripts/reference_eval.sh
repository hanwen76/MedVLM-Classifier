#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m medvlm_classifier.eval.reference_metrics \
  --pred_jsonl "outputs/reference_style_preds.jsonl" \
  --output_json "outputs/reference_style_metrics.json"

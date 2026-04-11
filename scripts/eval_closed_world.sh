#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python eval.py \
  --model_name_or_path "outputs/medvlm-classifier" \
  --annotation "examples/medical_cls_val.json" \
  --image_root "examples/images" \
  --batch_size 1 \
  --max_new_tokens 32 \
  --output_path "outputs/eval_closed_world.json"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python train.py \
  --model_name_or_path "llava-hf/llava-1.5-7b-hf" \
  --classification_ann "examples/medical_cls_train.json" \
  --classification_image_root "examples/images" \
  --instruction_jsonl "examples/instruction_tune.jsonl" \
  --instruction_image_root "examples/images" \
  --ratio "2:1" \
  --batch_size 2 \
  --epochs 1 \
  --output_dir "outputs/medvlm-classifier"

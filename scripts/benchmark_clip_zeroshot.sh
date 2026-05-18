#!/usr/bin/env bash
set -euo pipefail

python scripts/benchmark_clip_zeroshot.py \
  --clip_model "openai/clip-vit-large-patch14" \
  --data_json "data/isic2019/test.json" \
  --image_root "data/isic2019/raw" \
  --labels_json "data/isic2019/labels.json" \
  --prompt_template "a dermoscopic image of {label}" \
  --batch_size 32 \
  --output_dir "outputs/isic2019_clip_zeroshot"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/benchmark_medmnist_zeroshot.py \
  --model_ids "llava-hf/llava-1.5-7b-hf,/home/zhanghanwen/models/Qwen2-VL-2B-Instruct" \
  --data_json "data/medmnist/pathmnist/test.json" \
  --image_root "data/medmnist/pathmnist/images" \
  --labels_json "data/medmnist/pathmnist/labels.json" \
  --batch_size 1 \
  --mode choice \
  --language en \
  --output_dir "outputs/medmnist_zeroshot"

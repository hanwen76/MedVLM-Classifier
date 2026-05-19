#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_clip_prompt_tuning.py \
  --clip_model "openai/clip-vit-base-patch32" \
  --train_json "/tmp/medvlm_tb_dataset/train.json" \
  --test_json "/tmp/medvlm_tb_dataset/test.json" \
  --image_root "/mnt/diskB/zhw/A_DATASET/TB_dataset" \
  --labels_json "/tmp/medvlm_tb_dataset/labels.json" \
  --label_prompt_template "{label} chest x-ray" \
  --n_ctx 8 \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --output_dir "/tmp/medvlm_tb_dataset/clip_prompt_tuning"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-llava-hf/llava-1.5-7b-hf}"
INSTRUCTION_JSONL="${INSTRUCTION_JSONL:-examples/instruction_tune.jsonl}"
INSTRUCTION_IMAGE_ROOT="${INSTRUCTION_IMAGE_ROOT:-examples/images}"
ISIC_ROOT="${ISIC_ROOT:-data/isic2019}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/isic2019}"
RATIO="${RATIO:-2:1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
VIRTUAL_SIZE="${VIRTUAL_SIZE:-12000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
LOG_EVERY="${LOG_EVERY:-20}"
SEED="${SEED:-42}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) MODEL_NAME_OR_PATH="$2"; shift 2;;
    --instruction_jsonl) INSTRUCTION_JSONL="$2"; shift 2;;
    --instruction_image_root) INSTRUCTION_IMAGE_ROOT="$2"; shift 2;;
    --isic_root) ISIC_ROOT="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --ratio) RATIO="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --max_length) MAX_LENGTH="$2"; shift 2;;
    --virtual_size) VIRTUAL_SIZE="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --log_every) LOG_EVERY="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --extra_train_args) EXTRA_TRAIN_ARGS="$2"; shift 2;;
    --extra_eval_args) EXTRA_EVAL_ARGS="$2"; shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

TRAIN_JSON="${ISIC_ROOT}/train.json"
TEST_JSON="${ISIC_ROOT}/test.json"
IMAGE_ROOT="${ISIC_ROOT}/raw"
MODEL_OUT="${OUTPUT_DIR}/checkpoint"
EVAL_OUT="${OUTPUT_DIR}/eval_closed_world.json"

python train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --classification_ann "${TRAIN_JSON}" \
  --classification_image_root "${IMAGE_ROOT}" \
  --instruction_jsonl "${INSTRUCTION_JSONL}" \
  --instruction_image_root "${INSTRUCTION_IMAGE_ROOT}" \
  --output_dir "${MODEL_OUT}" \
  --ratio "${RATIO}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --max_length "${MAX_LENGTH}" \
  --virtual_size "${VIRTUAL_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --log_every "${LOG_EVERY}" \
  --seed "${SEED}" \
  ${EXTRA_TRAIN_ARGS}

python eval.py \
  --model_name_or_path "${MODEL_OUT}" \
  --annotation "${TEST_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --batch_size 1 \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --log_every "${LOG_EVERY}" \
  --output_path "${EVAL_OUT}" \
  ${EXTRA_EVAL_ARGS}

echo "[done] result=${EVAL_OUT}"

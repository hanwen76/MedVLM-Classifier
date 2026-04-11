#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Usage:
# bash scripts/run_medmnist_multi_dataset.sh \
#   --model_name_or_path /home/zhanghanwen/models/llava-1.5-7b-hf \
#   --instruction_jsonl /home/zhanghanwen/datasets/llava-med-zh-instruct-60k/instruction_zh_60k.jsonl \
#   --instruction_image_root /home/zhanghanwen/datasets/llava-med-zh-instruct-60k/images \
#   --datasets pathmnist,dermamnist,bloodmnist \
#   --medmnist_root data/medmnist \
#   --output_root outputs/medmnist_multiset

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-llava-hf/llava-1.5-7b-hf}"
INSTRUCTION_JSONL="${INSTRUCTION_JSONL:-examples/instruction_tune.jsonl}"
INSTRUCTION_IMAGE_ROOT="${INSTRUCTION_IMAGE_ROOT:-examples/images}"
DATASETS="${DATASETS:-pathmnist,dermamnist}"
MEDMNIST_ROOT="${MEDMNIST_ROOT:-data/medmnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/medmnist_multiset}"
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
    --datasets) DATASETS="$2"; shift 2;;
    --medmnist_root) MEDMNIST_ROOT="$2"; shift 2;;
    --output_root) OUTPUT_ROOT="$2"; shift 2;;
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

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${INSTRUCTION_JSONL}" ]]; then
  echo "[error] instruction jsonl not found: ${INSTRUCTION_JSONL}"
  exit 1
fi

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"

for DATASET in "${DATASET_LIST[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -z "${DATASET}" ]] && continue

  TRAIN_JSON="${MEDMNIST_ROOT}/${DATASET}/train.json"
  TEST_JSON="${MEDMNIST_ROOT}/${DATASET}/test.json"
  IMAGE_ROOT="${MEDMNIST_ROOT}/${DATASET}/images"

  if [[ ! -f "${TRAIN_JSON}" ]]; then
    echo "[skip] missing train annotation: ${TRAIN_JSON}"
    continue
  fi
  if [[ ! -f "${TEST_JSON}" ]]; then
    echo "[skip] missing test annotation: ${TEST_JSON}"
    continue
  fi
  if [[ ! -d "${IMAGE_ROOT}" ]]; then
    echo "[skip] missing image root: ${IMAGE_ROOT}"
    continue
  fi

  RUN_DIR="${OUTPUT_ROOT}/${DATASET}"
  MODEL_OUT="${RUN_DIR}/checkpoint"
  EVAL_OUT="${RUN_DIR}/eval_closed_world.json"
  mkdir -p "${RUN_DIR}"

  echo "[run] dataset=${DATASET}"
  echo "[run] train_json=${TRAIN_JSON}"
  echo "[run] test_json=${TEST_JSON}"
  echo "[run] model_out=${MODEL_OUT}"

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

  echo "[done] dataset=${DATASET} result=${EVAL_OUT}"
done

echo "[all done] outputs under ${OUTPUT_ROOT}"

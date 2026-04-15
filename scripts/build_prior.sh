#!/usr/bin/env bash
set -euo pipefail

python build_prior.py \
  --dataset "${DATASET:-MVTec-AD}" \
  --data_path "${DATA_PATH:-/path/to/dataset}" \
  --encoder "${ENCODER:-dinov2reg_vit_base_14}" \
  --INP_num "${INP_NUM:-6}" \
  --K "${K:-50}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --input_size "${INPUT_SIZE:-448}" \
  --crop_size "${CROP_SIZE:-392}" \
  --output_path "${OUTPUT_PATH:-checkpoints/prior_bank.pt}" \
  ${MODEL_PATH:+--model_path "$MODEL_PATH"}

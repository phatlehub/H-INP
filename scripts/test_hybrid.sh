#!/usr/bin/env bash
set -euo pipefail

python test.py \
  --config "${CONFIG_PATH:-configs/hybrid_config.yaml}" \
  --dataset "${DATASET:-MVTec-AD}" \
  --data_path "${DATA_PATH:-/path/to/dataset}" \
  --save_dir "${SAVE_DIR:-./saved_results}" \
  --save_name "${SAVE_NAME:-INP-Former-Single-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6}" \
  --mode "${MODE:-hybrid}" \
  --prior_path "${PRIOR_PATH:-checkpoints/prior_bank.pt}"

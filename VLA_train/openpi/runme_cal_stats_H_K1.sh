#!/usr/bin/env bash
# H K1 (Scale joint, K=1, 5-tier) — compute norm stats for pi05_scale_H_K1_lora_v1
# Must be run AFTER runme_gen_finetune_data_H_K1.sh.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_scale_H_K1_lora_v1

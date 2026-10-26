#!/usr/bin/env bash
# H M4 (Scale joint, K=7, N=150) — pi0.5 LoRA finetune on perm/Scale_H_M4
# Prereq:
#   1) runme_gen_finetune_data_H_M4.sh — produces LeRobot dataset (35 missions × 150 demos = 5250)
#   2) runme_cal_stats_H_M4.sh         — produces norm stats
#   3) Config entry pi05_scale_H_M4_lora_v1 in VLAs/openpi/src/openpi/training/config.py
# Base ckpt: gs://openpi-assets/checkpoints/pi05_base/params

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_scale_H_M4_lora_v1 \
  --exp-name=pi05_scale_H_M4_run1 \
  --overwrite

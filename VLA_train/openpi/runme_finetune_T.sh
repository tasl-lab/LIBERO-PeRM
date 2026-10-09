#!/usr/bin/env bash
# T (SingPref-DataTest) — pi0.5 LoRA finetune on perm/SingPref_DataTest (R33 data, marker-free)
# Prereq:
#   1) runme_gen_finetune_data_T.sh — produces LeRobot dataset
#   2) runme_cal_stats_T.sh         — produces norm stats
# Base ckpt: gs://openpi-assets/checkpoints/pi05_base/params (NOT pi05_libero)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
# 覆盖 .bashrc 里指向 /data4 的旧路径（data4 已无空间）
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_singpref_data_test_lora_v1 \
  --exp-name=pi05_data_test_run1 \
  --overwrite

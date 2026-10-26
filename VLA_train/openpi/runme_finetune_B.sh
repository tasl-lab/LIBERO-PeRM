#!/usr/bin/env bash
# B (SingPref-LoRA) — pi0.5 LoRA finetune on perm/SingPref_InDomain
# Prereq:
#   1) runme_gen_finetune_data_B.sh — produces LeRobot dataset
#   2) runme_cal_stats_B.sh         — produces norm stats
# Base ckpt: gs://openpi-assets/checkpoints/pi05_libero/params (same as D 0-shot)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
# 覆盖 .bashrc 里指向 /data4 的旧路径（data4 已无空间）
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_libero_perm_singpref_indomain \
  --exp-name=pi05_singpref_indomain_run1 \
  --overwrite

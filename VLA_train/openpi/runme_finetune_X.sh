#!/usr/bin/env bash
# X (pi0 + SingPref v5) — pi0 LoRA finetune on perm/SingPref (40 missions)
# Mirrors U's recipe (45k steps, peak_lr=5e-5, warmup=10k) but pi0 architecture.
# Prereq:
#   1) LeRobot dataset already at /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/SingPref/   (built by U's runme_gen_finetune_data_U.sh — REUSE)
#   2) runme_cal_stats_X.sh run first (norm_stats NOT compatible with U's because of extra_delta_transform=True)
# Base ckpt: gs://openpi-assets/checkpoints/pi0_base/params

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi0_singpref_lora_v1 \
  --exp-name=pi0_singpref_run1 \
  --overwrite

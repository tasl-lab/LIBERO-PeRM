#!/usr/bin/env bash
# PrefCompose (RQ4 a/b) — generate LeRobot finetune data.
# Tasksuite: VLA_data/PrefCompose/tasksuite.yaml (12 missions × 100 demos = 1200)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/PrefCompose/
# All 12 hdf5 in /data3/zhixu/LIBERO-PeRM/VLA_data/PrefCompose/ (gen'd by LIBERO data_gen 2026-05-06).

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/PrefCompose \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/PrefCompose/tasksuite.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/PrefCompose

#!/usr/bin/env bash
# H L3 (Scale joint, K=5, N=90, 5-tier × 5 layout × 90 demos = 2250) — generate LeRobot finetune data
# Tasksuite: tasksuite_L3.yaml (25 missions, n_demos=90, all train_eval)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/Scale_H_L3/

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/Scale_H_L3 \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/PrefScale/tasksuite_L3.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/PrefScale

#!/usr/bin/env bash
# H M4 (Scale joint, K=7, N=150, 5-tier × 7 layout × 150 demos = 5250) — generate LeRobot finetune data
# Tasksuite: tasksuite_M4.yaml (35 missions, n_demos=150, all train_eval)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/Scale_H_M4/

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/Scale_H_M4 \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/PrefScale/tasksuite_M4.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/PrefScale

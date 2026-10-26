#!/usr/bin/env bash
# H B0 (Scale, no-pref baseline, K=1, N=50, slot 0-1) — generate LeRobot finetune data
# Suite: PrefScale, single mission 0-1_032 (50 demos, no pref)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/Scale_H_B0/

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/Scale_H_B0 \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/PrefScale/tasksuite.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/PrefScale \
  --missions 0-1_032

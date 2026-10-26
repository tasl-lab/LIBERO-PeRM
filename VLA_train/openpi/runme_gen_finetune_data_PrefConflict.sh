#!/usr/bin/env bash
# PrefConflict (RQ5) — generate LeRobot finetune data.
# Tasksuite: VLA_data/PrefConflict/tasksuite.yaml (16 missions × 100 demos = 1600)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/PrefConflict/
# All 16 hdf5 already physically in /data3/zhixu/LIBERO-PeRM/VLA_data/PrefConflict/
#   - 9 gen'd by LIBERO data_gen (4 pp P2 + 3 pt _c + 2 pp5 push/drag)
#   - 7 mv'd from /data4/zhixu/SinglePref/ (4 pp P1 + 3 pt source)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/PrefConflict \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/PrefConflict/tasksuite.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/PrefConflict

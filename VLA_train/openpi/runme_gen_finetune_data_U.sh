#!/usr/bin/env bash
# U (SingPref v5) — generate LeRobot finetune data from v5 SingPref HDF5
# Suite: single-pref (40 missions; tasks + prefs read from
#        /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml in anchor mode
#        — "Task: <X>. Preference: <Y>." format).
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/SingPref/

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/SingPref \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref

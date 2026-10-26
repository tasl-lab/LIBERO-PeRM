#!/usr/bin/env bash
# Y (SingPref-mix joint LoRA) — generate combined LeRobot from LIBERO RLDS + perm HDF5
# Phase 1: LIBERO RLDS (4 suites, 1693 episodes) wrapped with `Task: <inst>.`
# Phase 2: perm/SingPref HDF5 (40 missions × 100 demos = 4200 episodes) `Task:...Preference:...`
# Output: VLA_data/finetune_data/perm/SingPref-mix/  (~5900 episodes)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_libero_rlds_perm_hdf5_to_lerobot_mix.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/SingPref-mix \
  --libero-data-dir /data3/zhixu/datasets/openvla_libero_rlds \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref

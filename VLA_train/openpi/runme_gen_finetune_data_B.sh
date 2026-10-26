#!/usr/bin/env bash
# B (SingPref-LoRA) — generate LeRobot finetune data from P_Bench SingPref HDF5
# Suite: single-pref-in-domain (38 missions)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/SingPref_InDomain/
#
# lerobot 只装在 openpi 的 uv env 里，所以必须 cd 到 openpi 目录用 uv run。

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
# 覆盖 .bashrc 里指向 /data4 的旧路径（data4 已无空间）
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref.py \
  --output_dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo_id perm/SingPref_InDomain \
  --suite single-pref-in-domain

#!/usr/bin/env bash
# T (SingPref-DataTest) — generate LeRobot finetune data from R33 SingPrefDataTest HDF5
# Suite: single-pref-data-test (18 missions; tasks + prefs read from
#        /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePrefDataTest/tasksuite.yaml)
# Output: /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/SingPref_DataTest/
#
# lerobot 只装在 openpi 的 uv env 里，所以必须 cd 到 openpi 目录用 uv run。

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
# 覆盖 .bashrc 里指向 /data4 的旧路径（data4 已无空间）
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref_anchor.py \
  --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
  --repo-id perm/SingPref_DataTest \
  --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePrefDataTest/tasksuite.yaml \
  --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePrefDataTest

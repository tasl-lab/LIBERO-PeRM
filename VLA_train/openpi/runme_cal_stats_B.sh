#!/usr/bin/env bash
# B (SingPref-LoRA) — compute norm stats for pi05_libero_perm_singpref_indomain
# Must be run AFTER runme_gen_finetune_data_B.sh (needs the LeRobot dataset on disk).

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
# 覆盖 .bashrc 里指向 /data4 的旧路径（data4 已无空间）
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_libero_perm_singpref_indomain

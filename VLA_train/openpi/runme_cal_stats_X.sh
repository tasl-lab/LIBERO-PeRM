#!/usr/bin/env bash
# X (pi0 + SingPref v5) — compute norm stats for pi0_singpref_lora_v1.
# pi0 uses extra_delta_transform=True (vs pi05's False), so norm stats are NOT
# compatible with U's — must recompute even though LeRobot dataset is reused.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_singpref_lora_v1

#!/usr/bin/env bash
# V (SingPref v5 + pi05_libero) — pi0.5 LoRA finetune on perm/SingPref (40 missions)
# starting from official pi05_libero ckpt (LIBERO-tuned), unlike U which starts from pi05_base.
# Prereq:
#   1) LeRobot dataset already at /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm/SingPref/   (built by U's runme_gen_finetune_data_U.sh — REUSE, no need to regenerate)
#   2) norm_stats already at  /data3/zhixu/LIBERO-PeRM/VLAs/openpi/assets/pi05_libero_singpref_lora_v1/perm/SingPref/norm_stats.json
#      → cp from U: cp -a assets/pi05_singpref_lora_v1/perm/SingPref assets/pi05_libero_singpref_lora_v1/perm/
# Base ckpt: gs://openpi-assets/checkpoints/pi05_libero/params

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export HF_LEROBOT_HOME=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data
export HF_HOME=/data3/zhixu/.cache/huggingface
export HF_DATASETS_CACHE=/data3/zhixu/.cache/huggingface/datasets

cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_libero_singpref_lora_v1 \
  --exp-name=pi05_libero_singpref_run1 \
  --overwrite

#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# GR00T N1.6 fine-tuning on P_Bench SinglePref data.
#
# v2 recipe (2026-04-19): reproduce official 0xAnkitSingh/GR00T-N1.6-LIBERO
# recipe (batch=16, steps=40k, LR=1e-4) + OpenVLA no_noops filter, to avoid
# gripper collapse observed in v1 (batch=320 + no filter).
# See docs/groot_n16_gripper_collapse.md for analysis.
#
# Prerequisites:
#   1. Data converted with no-op filter + gripper binarization:
#      python VLA_train/post_training_script/convert_perm_hdf5_to_groot.py \
#          --hdf5_dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref \
#          --output_dir /data3/zhixu/LIBERO-PeRM/VLA_data/groot_lerobot/perm_singlepref_v2 \
#          --binarize_gripper \
#          --filter_noops
#
#   2. Base model downloaded:
#      hf download nvidia/GR00T-N1.6-3B \
#          --local-dir /data3/zhixu/LIBERO-PeRM/checkpoints/groot_n16_base/
#
# Usage:
#   bash VLA_train/groot/runme_finetune_groot.sh
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  — default: 4,5,6,7
#   NUM_GPUS              — default: 4
#   MAX_STEPS             — default: 40000  (was 20000 in v1)
#   GLOBAL_BATCH_SIZE     — default: 16     (was 320 in v1 → caused collapse)
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-4}"
# v2: match official 0xAnkitSingh/GR00T-N1.6-LIBERO recipe (40k steps, batch 16, LR 1e-4).
# v1 used batch=320 which averaged out sparse gripper-event gradients → collapse to zero.
export MAX_STEPS="${MAX_STEPS:-40000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export SAVE_STEPS="${SAVE_STEPS:-2000}"

# Use LIBERO-finetuned checkpoint as base (0xAnkitSingh/GR00T-N1.6-LIBERO, SR 97.9%)
# so the model already knows LIBERO tasks and only needs to learn preference following.
BASE_MODEL="/data3/zhixu/LIBERO-PeRM/checkpoints/groot_n16_libero"
DATASET="/data3/zhixu/LIBERO-PeRM/VLA_data/groot_lerobot/perm_singlepref_v2"
OUTPUT_DIR="/data3/zhixu/LIBERO-PeRM/checkpoints/groot_n16_perm_singlepref_v2_noops"
EMBODIMENT_TAG="LIBERO_PANDA"

echo "──────────────────────────────────────────────────────────────"
echo "  GR00T N1.6 Fine-tuning — P_Bench SinglePref"
echo "──────────────────────────────────────────────────────────────"
echo "  Base model      : ${BASE_MODEL}"
echo "  Dataset          : ${DATASET}"
echo "  Output           : ${OUTPUT_DIR}"
echo "  Embodiment       : ${EMBODIMENT_TAG}"
echo "  GPUs             : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})"
echo "  Max steps        : ${MAX_STEPS}"
echo "  Batch size       : ${GLOBAL_BATCH_SIZE}"
echo "──────────────────────────────────────────────────────────────"

export UV_PROJECT_ENVIRONMENT=/data1/vla-reasoning/virtual_envs/groot
cd /data3/zhixu/LIBERO-PeRM/VLAs/groot_repo

uv run bash examples/finetune.sh \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --output-dir "${OUTPUT_DIR}" \
    --state-dropout-prob 0.8

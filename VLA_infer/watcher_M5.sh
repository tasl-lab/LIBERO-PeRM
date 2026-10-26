#!/usr/bin/env bash
# Polls GPU 5 mem usage; when free (<2 GB) launches chain_M5.sh.
set -uo pipefail
GPU=5
INFER=/data3/zhixu/LIBERO-PeRM/VLA_infer

echo "[watcher] polling GPU $GPU every 60s; will launch chain_M5.sh when free…"
while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    if [[ "$used" -lt 2048 ]]; then
        echo "[watcher $(date +%H:%M:%S)] GPU $GPU free (${used} MiB) → launching chain"
        bash $INFER/chain_M5.sh
        exit $?
    fi
    sleep 60
done

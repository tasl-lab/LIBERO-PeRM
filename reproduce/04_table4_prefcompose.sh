#!/usr/bin/env bash
# Table 4 — LIBERO-PrefCompose: composition (train PA, PB → eval PA+PB) and
# decomposition (train PA+PB, PA → eval PB), π0.5 LoRA.
#
#   bash 04_table4_prefcompose.sh train   # gen LeRobot data → norm stats → LoRA finetune (PrefCompose config)
#   bash 04_table4_prefcompose.sh eval    # 16 missions (4 layouts × {nopref, A, B, AB}) × 4 trials
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"; source "${HERE}/lib.sh"

case "${1:-eval}" in
  train) train_exp PrefCompose ;;
  eval)  run_eval "${PERM_PROFILE}/LoRA/Pi05/PrefCompose_step21k_4trial.yaml" ;;
  *) echo "usage: $0 {train|eval}" >&2; exit 1 ;;
esac
log_section "Table 4 done — Policy_output/lora_ft/pref-compose/PrefCompose_step21k_4trial/*/evalresult.xlsx"

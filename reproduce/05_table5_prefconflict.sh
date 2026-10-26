#!/usr/bin/env bash
# Table 5 — LIBERO-PrefConflict, π0.5 LoRA (single-pref demos only; conflicts are never trained jointly).
#   preference–preference (PP): in-domain single-pref prompts vs. joint "P1 + P2" prompts  → SR/SA + Pref-Dom.
#   task–preference       (PT): "TA + PB" prompts → Follow task / Follow pref / Incomplete
#
#   bash 05_table5_prefconflict.sh train      # gen LeRobot data → norm stats → LoRA finetune (PrefConflict config)
#   bash 05_table5_prefconflict.sh eval       # PP in-domain, PP conflict (set A + set B), PT conflict
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"; source "${HERE}/lib.sh"
P="${PERM_PROFILE}/LoRA/Pi05"

case "${1:-eval}" in
  train) train_exp PrefConflict ;;
  eval)
    run_eval "${P}/PrefConflict_indomain_setA_4trial.yaml"   # PP in-domain columns (each pref alone)
    run_eval "${P}/PrefConflict_PP_4trial.yaml"              # PP conflict, pair set A
    run_eval "${P}/PrefConflict_PP_setB_4trial.yaml"         # PP conflict, pair set B
    run_eval "${P}/PrefConflict_PT_4trial.yaml"              # task–preference conflict
    ;;
  *) echo "usage: $0 {train|eval}" >&2; exit 1 ;;
esac
log_section "Table 5 done — Policy_output/lora_ft/prefconflict/*/evalresult.xlsx"

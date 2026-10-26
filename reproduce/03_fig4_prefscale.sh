#!/usr/bin/env bash
# Figure 4 — LIBERO-PrefScale data scaling (π0.5 LoRA), in-domain / unseen-layout / unseen-layout+different-task.
#
# Scale cells (K = #layouts per task, N = demos per mission):
#   M1  250 demos  K=1 N=50   (ckpt shared with K1 config)     M2  750 demos  K=3 N=50
#   M3  2.25K      K=3 N=150  (= "L2" in the paper)            M4  5.25K      K=7 N=150
#   L1  2.25K      K=2 N=225        L3  2.25K  K=5 N=90        L4  2.3K  K=3 pref+matched no-pref
#
#   bash 03_fig4_prefscale.sh train            # gen LeRobot data → norm stats → LoRA finetune, all cells
#   bash 03_fig4_prefscale.sh eval             # in-domain + layout-OOD + 4 pref-OOD evals, all cells
#   bash 03_fig4_prefscale.sh eval M2 L3       # subset of cells
#   CELLS="M1 M2" bash 03_fig4_prefscale.sh train
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"; source "${HERE}/lib.sh"
P="${PERM_PROFILE}/LoRA/Pi05"

MODE="${1:-eval}"; shift || true
CELLS="${*:-${CELLS:-M1 M2 M3 M4 L1 L3 L4}}"

# training config name per cell (M1 reuses the K1 checkpoint)
train_name() { case "$1" in M1) echo H_K1 ;; *) echo "H_$1" ;; esac; }
# in-domain eval runfile per cell (naming differs slightly between cells)
indomain_yaml() { echo "${P}/H_$1_4trial.yaml"; }

case "${MODE}" in
  train)
    for c in ${CELLS}; do train_exp "$(train_name "$c")"; done ;;
  eval)
    for c in ${CELLS}; do
      run_eval "$(indomain_yaml "$c")"                       # blue  : in-domain
      run_eval "${P}/H_${c}_layout_ood.yaml"                 # red   : unseen layout
      for pref in speed left red multi; do                   # green : unseen layout + different task
        run_eval "${P}/H_${c}_pref_ood_${pref}.yaml"
      done
    done ;;
  *) echo "usage: $0 {train|eval} [cells...]" >&2; exit 1 ;;
esac
log_section "Figure 4 ${MODE} done — evalresult.xlsx under Policy_output/lora_ft/liberoscale{,_layout_ood,_pref_ood}/"

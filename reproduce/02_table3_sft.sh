#!/usr/bin/env bash
# Table 3 — fine-tuned preference learning on LIBERO-SinglePref (+ LIBERO-ComprePref column), all from π0.5 base.
#   π0.5-LoRA      (U)  LoRA on SinglePref demos only                         → No-Pref column is "–"
#   π0.5-Mix-LoRA  (Y)  LoRA on LIBERO RLDS + SinglePref mixed in one run
#   π0.5-Seq-LoRA  (V)  LoRA on SinglePref starting from the LIBERO-tuned pi05_libero ckpt (LIBERO → SinglePref)
#   π0.5-LoRA*          (unseen layout) — no runfile / output exists in this repo; not covered here, see manifest.yaml
#
#   bash 02_table3_sft.sh train [U|Y|V ...]     # gen LeRobot data → norm stats → LoRA finetune
#   bash 02_table3_sft.sh eval  [U|Y|V ...]     # SinglePref eval + ComprePref (S1) eval for each row
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"; source "${HERE}/lib.sh"
P="${PERM_PROFILE}/LoRA/Pi05"

MODE="${1:-eval}"; shift || true
ROWS="${*:-U Y V}"

case "${MODE}" in
  train)
    for r in ${ROWS}; do
      # V reuses U's LeRobot dataset + norm stats (see header of runme_finetune_V.sh) → only the finetune stage exists
      train_exp "$r"
    done ;;
  eval)
    for r in ${ROWS}; do
      case "$r" in
        U) run_eval "${P}/U_pi05_singpref_step45k.yaml"                # 40 missions, max_steps default
           run_eval "${P}/U_singlepref_rerun.yaml"                      # 25 long-horizon missions at max_steps 1250;
                                                                        # paper cell = per-mission max of the two sweeps
           run_eval "${P}/Comp/Comp_S1_pi05_singpref_lora_step45k.yaml" ;;
        Y) run_eval "${P}/Y_pi05_singpref_mix_step45k.yaml"
           run_eval "${P}/Comp/Comp_S1_pi05_singpref_mix_lora_step45k.yaml" ;;
        V) run_eval "${P}/V_pi05_libero_singpref_step45k.yaml"
           run_eval "${P}/Comp/Comp_S1_pi05_libero_singpref_lora_step45k.yaml" ;;
        *) echo "unknown row $r (U|Y|V)" >&2 ;;
      esac
    done ;;
  *) echo "usage: $0 {train|eval} [rows...]" >&2; exit 1 ;;
esac
log_section "Table 3 ${MODE} done — Policy_output/lora_ft/{single-pref,comprehensive}/*/evalresult.xlsx"

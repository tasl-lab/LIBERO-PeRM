#!/usr/bin/env bash
# Table 2 — zero-shot SR/SA on LIBERO-SinglePref for 6 VLAs (No-Pref = `base`, All-Pref = `pref`).
#
#   bash 01_table2_zeroshot.sh                 # all six policies, sequentially
#   bash 01_table2_zeroshot.sh pi05 groot      # subset: pi05 pi0 groot smolvla univla oft
#   DRY_RUN=1 bash 01_table2_zeroshot.sh       # print the commands only
#
# Each cell = policy server + rollout client; SR + SA land in
#   Policy_output/0shot/single-pref/<experiment_name>/<server.name>/evalresult.xlsx
# UniVLA and OpenVLA-OFT ship one checkpoint / unnorm key per LIBERO suite, so they
# are run 4x and merged per mission with best-of-4 (aggregate_best_of_n.py).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"; source "${HERE}/lib.sh"
CFG="${HERE}/configs/table2_zeroshot"
OUT="${PERM_ROOT}/Policy_output/0shot/single-pref"

SEL=("$@")
sel() { [[ ${#SEL[@]} -eq 0 ]] && return 0; local x; for x in "${SEL[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

sel pi05    && run_eval "${CFG}/pi05_official_singpref.yaml"       # π0.5  (pi05_libero, openpi)
sel pi0     && run_eval "${CFG}/pi0_official_singpref.yaml"        # π0    (pi0_libero,  openpi)
sel groot   && run_eval "${CFG}/groot16.yaml"                      # GR00T-N1.6
sel smolvla && run_eval "${CFG}/smolvla.yaml"                      # SmolVLA
if sel univla; then                                                 # UniVLA: 4 suite ckpts → best-of-4
  for k in spatial object goal 10; do run_eval "${CFG}/univla_${k}.yaml"; done
  _run python "${HERE}/aggregate_best_of_n.py" "${OUT}"/univla_{spatial,object,goal,10}_full_v2_repro \
       --out "${PERM_LOG_ROOT}/table2_univla_best_of_4.md"
fi
if sel oft; then                                                    # OpenVLA-OFT 4-in-1: 4 unnorm keys → best-of-4
  for k in spatial object goal 10; do run_eval "${HERE}/configs/table2_oft_4pass/oft_4in1_${k}.yaml"; done
  _run python "${HERE}/aggregate_best_of_n.py" "${OUT}"/oft_4in1_{spatial,object,goal,10}_repro \
       --out "${PERM_LOG_ROOT}/table2_oft_best_of_4.md"
fi
log_section "Table 2 done — evalresult.xlsx under ${OUT}/*_repro/, best-of-4 tables in ${PERM_LOG_ROOT}"

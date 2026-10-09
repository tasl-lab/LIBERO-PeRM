#!/usr/bin/env bash
# Everything in the paper, in order. Training cells run one after another (each finetune script
# owns its GPUs via CUDA_VISIBLE_DEVICES inside VLA_train/openpi/runme_finetune_*.sh).
#   bash run_all.sh            # train + eval
#   bash run_all.sh eval       # eval only (checkpoints already trained / downloaded)
#   DRY_RUN=1 bash run_all.sh  # list the commands
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"

bash "${HERE}/01_table2_zeroshot.sh"
for s in 02_table3_sft.sh 03_fig4_prefscale.sh 04_table4_prefcompose.sh 05_table5_prefconflict.sh; do
  [[ "${MODE}" == "all" ]] && bash "${HERE}/${s}" train
  bash "${HERE}/${s}" eval
done

#!/bin/bash
set -e

echo "===== Layout 3_1 - Mission 16 ====="
CFG="cfgs/layout3_1.json"

mid="16"
python main.py \
  --naive_llm_plan 0 \
  --run_layout_gen 0 \
  --high_pref_gpt_plan 0 \
  --low_pref_gpt_plan 0 \
  --generate_evaluation_data 0 \
  --enable_collision_check 1 \
  --batch_size 5 \
  --batch_start 0 \
  --mis_id "$mid" \
  --cfg "$CFG" \
  --manually_set_offset 1 \
  --check_init_layout 0 \
  --gen_in_batch 1 \
  --overwrite_old 1 \
  --use_forcibly 1 || true

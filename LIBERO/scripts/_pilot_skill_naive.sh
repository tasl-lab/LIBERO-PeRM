#!/usr/bin/env bash
# skill-pref naive 试点：3 个代表性家族，各 10 条，init 与 pref 逐条配对
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl
MIDS="${*:-2-1_077 2-12_009 2-14_109}"
for MID in $MIDS; do
  echo "=== $MID ==="
  $PY main.py --mission_id "$MID" --tasksuite SinglePref \
    --check_init_layout 0 --run_layout_gen 0 --generate_evaluation_data 0 \
    --gen_in_batch 1 --naive_llm_plan 0 --run_naive_plan 0 \
    --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 \
    --gen_naive_anchor --force_naive_plan 1 \
    --pair_init_from "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/Mission_${MID}.hdf5" \
    --batch_size 10 --max_attempts 40 --overwrite_old 1 --use_forcibly 1 \
    2>&1 | grep -aE "success rate|Traceback|Error|error:|EOFError" | tail -20
done
echo "=== SKILL NAIVE PILOT DONE ==="

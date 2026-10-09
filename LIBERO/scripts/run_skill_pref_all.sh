#!/usr/bin/env bash
# 31 个 series_2 mission 的 pref + naive demo 全量生成。
#
#   phase 1  已有 pref 的 9 个 -> 补 naive（init 与 pref 逐条配对）
#   phase 2  缺 pref 的 19 个  -> 生成 pref（多跑一些，给门槛留筛选余量）
#   phase 3  这 19 个          -> 补 naive
#
# 并行 4 路。每个 mission 一份日志在 /tmp/skillgen/。
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl
LOGD=/tmp/skillgen; mkdir -p "$LOGD"
JOBS=${JOBS:-4}

COMMON="--tasksuite SinglePref --check_init_layout 0 --run_layout_gen 0
        --generate_evaluation_data 0 --gen_in_batch 1 --naive_llm_plan 0
        --run_naive_plan 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0
        --overwrite_old 1 --use_forcibly 1"

HAVE_PREF="2-1_019 2-2_019 2-2_077 2-3_077 2-4_119 2-5_116 2-6_119 2-11_118 2-13_001"
NEED_PREF="2-1_001 2-3_075 2-4_112 2-5_112 2-6_042 2-10_015 2-10_065 2-11_062 2-11_116
           2-12_085 2-12_093 2-13_034 2-14_068 2-7_ood1 2-7_ood2 2-8_ood3 2-8_ood4
           2-9_ood3 2-9_ood4"

gen_pref () {   # $1 = mission id, $2 = batch size
  $PY main.py --mission_id "$1" $COMMON --batch_size "$2" --max_attempts $(( $2 * 4 )) \
    > "$LOGD/pref_$1.log" 2>&1
  echo "pref  $1 -> $(grep -ao 'success rate: [0-9]*/[0-9]*' "$LOGD/pref_$1.log" | tail -1)"
}

gen_naive () {  # $1 = mission id
  local SRC=/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/Mission_$1.hdf5
  [ -f "$SRC" ] || { echo "naive $1 -> SKIP (no pref hdf5)"; return; }
  $PY main.py --mission_id "$1" $COMMON --gen_naive_anchor --force_naive_plan 1 \
    --pair_init_from "$SRC" --batch_size 10 --max_attempts 40 \
    > "$LOGD/naive_$1.log" 2>&1
  echo "naive $1 -> $(grep -ao 'success rate: [0-9]*/[0-9]*' "$LOGD/naive_$1.log" | tail -1)"
}

run_pool () {   # $1 = fn name, rest = ids
  local fn=$1; shift
  local n=0
  for m in "$@"; do
    "$fn" "$m" "${EXTRA:-}" &
    n=$((n+1))
    if [ $((n % JOBS)) -eq 0 ]; then wait; fi
  done
  wait
}

echo "===== phase 1: naive for the $(echo $HAVE_PREF | wc -w) missions that already have pref"
run_pool gen_naive $HAVE_PREF

echo "===== phase 2: pref for the $(echo $NEED_PREF | wc -w) missing missions (30 each)"
EXTRA=30 run_pool gen_pref $NEED_PREF

echo "===== phase 3: naive for those $(echo $NEED_PREF | wc -w)"
run_pool gen_naive $NEED_PREF

echo "===== ALL SKILL GEN DONE ====="

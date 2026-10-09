#!/usr/bin/env bash
# 3 系列 pref demo 生成（task_plan，pref bddl 目标）：默认 30 条成功 / 最多 60 次。
# 用法: JOBS=6 bash scripts/run_series3_pref.sh 3-3_084_1 3-4_058_1 ...   日志 /tmp/s3gen/pref_<id>.log
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
V=/data3/zhixu/LIBERO-PeRM/VLA_data
ARGS="--tasksuite SinglePref --naive_llm_plan 0 --run_layout_gen 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 --generate_evaluation_data 0 --verbose --enable_collision_check 1 --batch_start 0 --manually_set_offset 1 --check_init_layout 0 --gen_in_batch 1 --overwrite_old 1 --use_forcibly 1"
JOBS=${JOBS:-6}; n=0
for m in "$@"; do
  $PY main.py --mission_id "$m" $ARGS --batch_size "${PREF_TARGET:-30}" --max_attempts "${MAX_ATT:-60}" > /tmp/s3gen/pref_$m.log 2>&1 &
  n=$((n+1)); [ $((n % JOBS)) -eq 0 ] && wait
done
wait
for m in "$@"; do
  c=$($PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$V/SinglePref/Mission_$m.hdf5" 2>/dev/null || echo 0)
  echo "$m pref $c  $(grep -a 'success rate' /tmp/s3gen/pref_$m.log | tail -n 1 | sed 's/.*success rate/SR/')"
done
echo "===== S3 PREF DONE ====="

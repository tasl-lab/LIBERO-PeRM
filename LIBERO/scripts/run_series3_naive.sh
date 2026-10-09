#!/usr/bin/env bash
# 3 系列 naive 生成：按 pref demo 的 init 逐条配对跑 naive_plan，前 30 个 init 里凑 10 条成功。
# 用法: JOBS=6 bash scripts/run_series3_naive.sh 3-1_004_1 3-1_006_1 ...
# 产物: VLA_data/SinglePref_naive/Mission_<id>.hdf5，日志 /tmp/s3gen/naive_<id>.log
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
V=/data3/zhixu/LIBERO-PeRM/VLA_data
COMMON="--tasksuite SinglePref --check_init_layout 0 --run_layout_gen 0 --generate_evaluation_data 0 --gen_in_batch 1 --naive_llm_plan 0 --run_naive_plan 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 --overwrite_old 1 --use_forcibly 1"
JOBS=${JOBS:-6}; n=0
for m in "$@"; do
  $PY main.py --mission_id "$m" $COMMON --gen_naive_anchor --force_naive_plan 1 \
    --pair_init_from "$V/SinglePref/Mission_$m.hdf5" --batch_size "${NAIVE_TARGET:-10}" --max_attempts "${MAX_ATT:-30}" \
    > /tmp/s3gen/naive_$m.log 2>&1 &
  n=$((n+1)); [ $((n % JOBS)) -eq 0 ] && wait
done
wait
for m in "$@"; do
  c=$($PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$V/SinglePref_naive/Mission_$m.hdf5" 2>/dev/null || echo 0)
  a=$(grep -ac 'paired-init\] batch' /tmp/s3gen/naive_$m.log)
  echo "$m naive $c/$a"
done
echo "===== S3 NAIVE DONE ====="

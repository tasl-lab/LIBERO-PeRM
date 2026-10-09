#!/usr/bin/env bash
# batch 2：Conflict 8 个新 mission（排最前）+ Single 13 个新挂 + Single 4 个 naive 修复 → pref 30 → naive 配对 → 按 family 收尾。
# 用法: nohup bash scripts/run_single_batch2.sh > /tmp/s3gen/batch2.log 2>&1 &
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
V=/data3/zhixu/LIBERO-PeRM/VLA_data; HR6=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S6
mkdir -p /tmp/s3gen /tmp/skillgen
CONFLICT_PP="1-1_062 1-5_104 2-1_034 2-1_075 3-15_023_1"
CONFLICT_PT="3-1_004_1c 3-9_063_1c 3-8_070_1c"
SINGLE_NEW="1-1_034 1-3_065 1-4_109 1-8_071 2-2_070 2-4_058 2-6_068 2-10_071 2-13_077 2-14_036 3-1_115_1 3-13_060_2 3-14_060_3"
FIX_NAIVE="1-5_034 3-6_036_1 3-8_070_1 3-12_036_2"
MOTION="1-1_062 1-5_104 1-1_034 1-3_065 1-4_109 1-8_071 1-5_034"
SKILL="2-1_034 2-1_075 2-2_070 2-4_058 2-6_068 2-10_071 2-13_077 2-14_036"
PLAN="3-15_023_1 3-1_115_1 3-13_060_2 3-14_060_3 3-6_036_1 3-8_070_1 3-12_036_2"
echo "[batch2] $(date) pref gen"
JOBS=${JOBS:-6} bash scripts/run_series3_pref.sh $CONFLICT_PP $CONFLICT_PT $SINGLE_NEW
echo "[batch2] $(date) naive gen"
JOBS=${JOBS:-6} bash scripts/run_series3_naive.sh $CONFLICT_PP $SINGLE_NEW $FIX_NAIVE
echo "[batch2] $(date) finish"
for m in $MOTION; do bash scripts/finish_motion_mission.sh "$m"; done
for m in $SKILL;  do bash scripts/finish_skill_mission_v2.sh "$m"; done
for m in $PLAN;   do bash scripts/finish_plan_mission.sh "$m"; done
echo "[batch2] $(date) pt _c: SR + pref-only video"
mkdir -p "$HR6/all"
for m in $CONFLICT_PT; do
  c=$($PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$V/SinglePref/Mission_$m.hdf5" 2>/dev/null || echo 0)
  a=$(grep -ac "HDF5 saved" /tmp/s3gen/pref_$m.log)
  $PY scripts/render_demo_video.py --missions "$m" --demo demo_0 --demo_dir "$V/SinglePref" --out "$HR6/all" > /tmp/s3gen/video_$m.log 2>&1 && v=video || v=novideo
  echo "$m _c OK  pref $c/$a  $v"
done
echo "===== BATCH2 DONE $(date) ====="

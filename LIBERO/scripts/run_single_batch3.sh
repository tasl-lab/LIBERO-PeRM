#!/usr/bin/env bash
# batch 3：Single 4 个换方案 + 2-1_001 审阅对 + Compose 8 个新单 pref + 15 ∅ + 15 A+B（后两类只看 SR + 单排视频）
# 用法: nohup bash scripts/run_single_batch3.sh > /tmp/s3gen/batch3.log 2>&1 &
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
V=/data3/zhixu/LIBERO-PeRM/VLA_data; HR4=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S4
mkdir -p /tmp/s3gen /tmp/skillgen
SINGLE_FIX="1-4_109 1-8_071 2-13_070 2-14_072"
COMPOSE_SINGLE="1-1_112 1-2_118 1-4_001 1-7_065 1-5_007 1-7_058 1-3_114 3-3_077_1"
COMBO=""; for x in 4-3_112 4-7_062 4-7_118 4-8_034 4-8_001 4-9_015 4-9_065 4-10_070 4-10_007 4-11_104 4-11_058 4-12_107 4-12_114 4-13_019 4-13_077; do COMBO="$COMBO ${x}_combo_nopref ${x}_combo_AB"; done
MOTION="1-4_109 1-8_071 1-1_112 1-2_118 1-4_001 1-7_065 1-5_007 1-7_058 1-3_114"
SKILL="2-13_070 2-14_072 2-1_001"
PLAN="3-3_077_1"
echo "[batch3] $(date) pref gen"
JOBS=${JOBS:-4} bash scripts/run_series3_pref.sh $SINGLE_FIX $COMPOSE_SINGLE $COMBO
echo "[batch3] $(date) naive gen"
JOBS=${JOBS:-4} bash scripts/run_series3_naive.sh $SINGLE_FIX $COMPOSE_SINGLE 2-1_001
echo "[batch3] $(date) finish"
for m in $MOTION; do bash scripts/finish_motion_mission.sh "$m"; done
for m in $SKILL;  do bash scripts/finish_skill_mission_v2.sh "$m"; done
for m in $PLAN;   do bash scripts/finish_plan_mission.sh "$m"; done
echo "[batch3] $(date) combos: SR + single-row video"
mkdir -p "$HR4/all"
for m in $COMBO; do
  c=$($PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$V/SinglePref/Mission_$m.hdf5" 2>/dev/null || echo 0)
  a=$(grep -ac "HDF5 saved" /tmp/s3gen/pref_$m.log)
  $PY scripts/render_demo_video.py --missions "$m" --demo demo_0 --demo_dir "$V/SinglePref" --out "$HR4/all" > /tmp/s3gen/video_$m.log 2>&1 && v=video || v=novideo
  echo "$m combo OK  pref $c/$a  $v"
done
echo "===== BATCH3 DONE $(date) ====="

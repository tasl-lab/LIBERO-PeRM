#!/usr/bin/env bash
# 只重渲染 human-review/all 的对比视频（配对文件不动）。用法：JOBS=4 bash scripts/rerender_skill_videos.sh <mid ...>
cd /data3/zhixu/LIBERO-PeRM/LIBERO; PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python; export MUJOCO_GL=egl PYTHONUNBUFFERED=1
V=/data3/zhixu/LIBERO-PeRM/VLA_data; HR=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2; JOBS=${JOBS:-4}; n=0
for m in "$@"; do
  if [ -f "$V/SinglePref_naive_paired/Mission_$m.hdf5" ]; then
    $PY scripts/render_demo_video.py --missions "$m" --demo demo_0 --demo_dir "$V/SinglePref_paired" --naive_dir "$V/SinglePref_naive_paired" --out "$HR/all" > /tmp/skillgen/render_$m.log 2>&1 &
  else
    $PY scripts/render_demo_video.py --missions "$m" --demo demo_0 --demo_dir "$V/SinglePref_paired" --out "$HR/all" > /tmp/skillgen/render_$m.log 2>&1 &
  fi
  n=$((n+1)); [ $((n % JOBS)) -eq 0 ] && wait
done; wait; echo "RERENDER DONE"

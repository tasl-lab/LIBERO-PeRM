#!/usr/bin/env bash
# Single 第 3 mission batch 1：14 个现成目录 → pref 30（已有 bulk 的 4 个不重跑）→ naive 按 init 配对 → 收尾（motion 5 / profile 8；2-12_009 早已收尾）。
# 用法: nohup bash scripts/run_single_third_batch1.sh > /tmp/s3gen/batch1.log 2>&1 &
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
mkdir -p /tmp/s3gen
PREF_NEW="1-2_062 1-5_034 1-6_067 1-8_049 3-2_097_1 3-3_019_1 3-4_081_1 3-6_036_1 3-8_070_1 3-12_036_2"
NAIVE_ALL="1-2_062 1-5_034 1-6_067 1-7_114 1-8_049 3-2_097_1 3-3_019_1 3-4_081_1 3-5_120_1 3-6_036_1 3-8_070_1 3-12_036_2 3-15_120_1"
MOTION="1-2_062 1-5_034 1-6_067 1-7_114 1-8_049"
PROFILE="3-2_097_1 3-3_019_1 3-4_081_1 3-5_120_1 3-6_036_1 3-8_070_1 3-12_036_2 3-15_120_1"
echo "[batch1] $(date) pref gen: $PREF_NEW"
JOBS=${JOBS:-5} bash scripts/run_series3_pref.sh $PREF_NEW
echo "[batch1] $(date) naive gen: $NAIVE_ALL"
JOBS=${JOBS:-5} bash scripts/run_series3_naive.sh $NAIVE_ALL
echo "[batch1] $(date) finish"
for m in $MOTION;  do bash scripts/finish_motion_mission.sh "$m"; done
for m in $PROFILE; do bash scripts/finish_plan_mission.sh "$m"; done
echo "===== BATCH1 DONE $(date) ====="

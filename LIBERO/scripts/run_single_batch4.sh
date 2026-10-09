#!/usr/bin/env bash
# batch 4（修复轮）：1-8_107 / 1-4_070 换方案；3-1_115_1 / 3-12_036_2 改了 bddl 重生成；1-5_104 只补 naive；2-1_075 / 2-2_070 重收尾（yaml 修过）。
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
mkdir -p /tmp/s3gen /tmp/skillgen
until grep -q "BATCH2 DONE" /tmp/s3gen/batch2.log; do sleep 30; done
echo "[batch4] $(date) pref gen"
JOBS=${JOBS:-3} bash scripts/run_series3_pref.sh 1-8_107 1-4_070 3-1_115_1 3-12_036_2
echo "[batch4] $(date) naive gen"
JOBS=${JOBS:-3} bash scripts/run_series3_naive.sh 1-8_107 1-4_070 3-1_115_1 3-12_036_2 1-5_104
echo "[batch4] $(date) finish"
for m in 1-8_107 1-4_070 1-5_104; do bash scripts/finish_motion_mission.sh "$m"; done
SKIP_GEN=1 bash scripts/finish_skill_mission_v2.sh 2-1_075
bash scripts/finish_skill_mission_v2.sh 2-2_070
for m in 3-1_115_1 3-12_036_2; do bash scripts/finish_plan_mission.sh "$m"; done
echo "===== BATCH4 DONE $(date) ====="

#!/usr/bin/env bash
# 批量收尾：bash scripts/run_skill_pref_finish.sh 2-1_019 2-2_019 ...   （JOBS 并行，默认 3）
cd /data3/zhixu/LIBERO-PeRM/LIBERO
JOBS=${JOBS:-3}; n=0
for m in "$@"; do
  bash scripts/finish_skill_mission_v2.sh "$m" > /tmp/skillgen/finish_$m.out 2>&1 &
  n=$((n+1)); [ $((n % JOBS)) -eq 0 ] && wait
done
wait
for m in "$@"; do tail -1 /tmp/skillgen/finish_$m.out; done
echo "===== FINISH DONE ====="

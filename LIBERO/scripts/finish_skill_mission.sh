#!/usr/bin/env bash
# 一个 series_2 mission 的收尾：gate pref → 从 gated pref 逐条配对重跑 naive → 双锚 + cohen_d → 对比视频。
# 前置：VLA_data/SinglePref/Mission_<id>.hdf5（≥30 条）和 VLA_data/SinglePref_naive/Mission_<id>.hdf5（gate 要 naive 分布）。
# 产物：VLA_data/SinglePref_gated/Mission_<id>.hdf5（10 条）、SinglePref_naive 覆盖成配对的 10 条、
#       mission_info.yaml 的 low_pref_calibration、human-review/all/<id>_pref_vs_nopref.mp4、/tmp/skillgen/finish_<id>.json
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1   # unbuffered: a GL teardown crash at exit otherwise drops the last ~8KB of log
MID=$1; K=${K:-3}; SCAN=${SCAN:-30}
V=/data3/zhixu/LIBERO-PeRM/VLA_data
HR=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2
LOGD=/tmp/skillgen; OUT=$LOGD/finish_$MID.log; [ -f "$OUT" ] && mv -f "$OUT" "$OUT.prev"; : > "$OUT"
J=$LOGD/finish_$MID.json
COMMON="--tasksuite SinglePref --check_init_layout 0 --run_layout_gen 0 --generate_evaluation_data 0 --gen_in_batch 1 --naive_llm_plan 0 --run_naive_plan 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 --overwrite_old 1 --use_forcibly 1"
fail () { echo "{\"mission\":\"$MID\",\"stage\":\"$1\",\"ok\":false,\"note\":\"$2\"}" > "$J"; echo "$MID FAIL@$1: $2"; exit 1; }

ndemos () { $PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$1" 2>/dev/null || echo 0; }
GATE=""; NSR=""
if [ "${SKIP_GEN:-0}" = "1" ] && [ "$(ndemos $V/SinglePref_gated/Mission_$MID.hdf5)" = "10" ] && [ "$(ndemos $V/SinglePref_naive/Mission_$MID.hdf5)" = "10" ]; then
  echo "== [$MID] gate + paired naive already there (SKIP_GEN=1)" | tee -a "$OUT"
  GATE=$(grep -ah "pref demos pass" "$LOGD/finish_$MID.log.prev" 2>/dev/null | tail -1 | grep -ao "[0-9]*/[0-9]* pref demos pass" | cut -d' ' -f1)
  NSR="10/$(grep -ac 'paired-init\] batch' "$LOGD/finish_$MID.log.prev" 2>/dev/null)"
else
  echo "== [$MID] 1/4 gate (k=$K, scan=$SCAN)" | tee -a "$OUT"
  rm -f "$V/SinglePref_gated/Mission_$MID.hdf5"
  $PY scripts/gate_pref_demos.py --missions "$MID" --scan "$SCAN" --target 10 --cap 10 --k "$K" --apply >> "$OUT" 2>&1
  GATE=$(grep -a "pref demos pass" "$OUT" | tail -1 | grep -ao "[0-9]*/[0-9]* pref demos pass" | cut -d' ' -f1)
  [ "$(ndemos $V/SinglePref_gated/Mission_$MID.hdf5)" = "10" ] || fail gate "only $GATE pass"

  echo "== [$MID] 2/4 naive paired to gated pref" | tee -a "$OUT"
  $PY main.py --mission_id "$MID" $COMMON --gen_naive_anchor --force_naive_plan 1 \
    --pair_init_from "$V/SinglePref_gated/Mission_$MID.hdf5" --batch_size 10 --max_attempts 40 >> "$OUT" 2>&1
  # the log's last "success rate" line is unreliable (see PYTHONUNBUFFERED note); trust the hdf5 + one [paired-init] line per attempt
  NSR="$(ndemos $V/SinglePref_naive/Mission_$MID.hdf5)/$(grep -ac 'paired-init\] batch' "$OUT")"
  [ "${NSR%%/*}" = "10" ] || fail naive "naive SR $NSR"
fi

echo "== [$MID] 3/4 anchors + cohen_d" | tee -a "$OUT"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_gated" --naive_dir "$V/SinglePref_naive" \
  --missions "$MID" --n_demos 10 --apply >> "$OUT" 2>&1 || fail calibrate "calibrate error"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_gated" --naive_dir "$V/SinglePref_naive" \
  --missions "$MID" --n_demos 10 --table --csv "$LOGD/table_$MID.csv" >> "$OUT" 2>&1

echo "== [$MID] 4/4 video" | tee -a "$OUT"
mkdir -p "$HR/all"
$PY scripts/render_demo_video.py --missions "$MID" --demo demo_0 --demo_dir "$V/SinglePref_gated" \
  --naive_dir "$V/SinglePref_naive" --out "$HR/all" >> "$OUT" 2>&1 || fail video "render error"

echo "{\"mission\":\"$MID\",\"ok\":true,\"gate\":\"$GATE\",\"naive_sr\":\"$NSR\",\"k\":$K}" > "$J"
echo "$MID OK  gate=$GATE naive=$NSR"

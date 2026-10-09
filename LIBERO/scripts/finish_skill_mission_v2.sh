#!/usr/bin/env bash
# 一个 series_2 mission 的收尾 v2：gate pref（留全部幸存者）→ naive 按 gated init 逐条配对（失败的 init 跳过）
# → 只保留 naive 成功的前 10 个 init，pref/naive 各写 10 条 paired 文件 → 双锚 + cohen_d → 对比视频。
# 前置：VLA_data/SinglePref/Mission_<id>.hdf5（≥30 条）+ VLA_data/SinglePref_naive/Mission_<id>.hdf5（gate 要 naive 分布）。
# 产物：VLA_data/SinglePref_paired/ 与 SinglePref_naive_paired/ 各 10 条（demo_i ↔ demo_i 同一 init）、
#       mission_info.yaml 的 low_pref_calibration、human-review/all/<id>_pref_vs_nopref.mp4、/tmp/skillgen/finish_<id>.json
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
MID=$1; K=${K:-3}; SCAN=${SCAN:-30}
V=/data3/zhixu/LIBERO-PeRM/VLA_data
HR=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2
LOGD=/tmp/skillgen; OUT=$LOGD/finish_$MID.log; [ -f "$OUT" ] && mv -f "$OUT" "$OUT.prev"; : > "$OUT"
J=$LOGD/finish_$MID.json
COMMON="--tasksuite SinglePref --check_init_layout 0 --run_layout_gen 0 --generate_evaluation_data 0 --gen_in_batch 1 --naive_llm_plan 0 --run_naive_plan 0 --high_pref_gpt_plan 0 --low_pref_gpt_plan 0 --overwrite_old 1 --use_forcibly 1"
fail () { echo "{\"mission\":\"$MID\",\"stage\":\"$1\",\"ok\":false,\"note\":\"$2\"}" > "$J"; echo "$MID FAIL@$1: $2"; exit 1; }
ndemos () { $PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$1" 2>/dev/null || echo 0; }
GATE=""; NSR=""
if [ "${SKIP_GEN:-0}" = "1" ] && [ "$(ndemos $V/SinglePref_paired/Mission_$MID.hdf5)" = "10" ] && [ "$(ndemos $V/SinglePref_naive_paired/Mission_$MID.hdf5)" = "10" ]; then
  echo "== [$MID] paired sets already there (SKIP_GEN=1)" | tee -a "$OUT"
  GATE=$(grep -ah "pref demos pass" "$OUT.prev" 2>/dev/null | tail -1 | grep -ao "[0-9]*/[0-9]* pref demos pass" | cut -d' ' -f1)
  NSR="10/$(grep -ac 'paired-init\] batch' "$OUT.prev" 2>/dev/null)"
else
  echo "== [$MID] 1/5 gate (k=$K, scan=$SCAN, keep all survivors)" | tee -a "$OUT"
  rm -f "$V/SinglePref_gated/Mission_$MID.hdf5"
  $PY scripts/gate_pref_demos.py --missions "$MID" --scan "$SCAN" --target 10 --cap 30 --k "$K" --apply >> "$OUT" 2>&1
  GATE=$(grep -a "pref demos pass" "$OUT" | tail -1 | grep -ao "[0-9]*/[0-9]* pref demos pass" | cut -d' ' -f1)
  NG=$(ndemos "$V/SinglePref_gated/Mission_$MID.hdf5"); [ "$NG" -ge 10 ] 2>/dev/null || fail gate "only $GATE pass"

  echo "== [$MID] 2/5 naive on the $NG gated inits (stop at 10 successes)" | tee -a "$OUT"
  $PY main.py --mission_id "$MID" $COMMON --gen_naive_anchor --force_naive_plan 1 \
    --pair_init_from "$V/SinglePref_gated/Mission_$MID.hdf5" --batch_size "${NAIVE_TARGET:-10}" --max_attempts "$NG" >> "$OUT" 2>&1
  NSR="$(ndemos $V/SinglePref_naive/Mission_$MID.hdf5)/$(grep -ac 'paired-init\] batch' "$OUT")"
  [ "${NSR%%/*}" -ge 10 ] 2>/dev/null || fail naive "naive SR $NSR"

  echo "== [$MID] 3/5 pair: keep the inits where naive succeeded" | tee -a "$OUT"
  $PY - "$MID" "$V" "$OUT" >> "$OUT" 2>&1 <<'PYEOF' || fail pair "pairing error"
import sys, re, os, h5py
mid, V, log = sys.argv[1:4]
txt = open(log, errors="ignore").read()
ids = sorted({int(m) for m in re.findall(r"HDF5 saved: \S+/HDF5/rollout_(\d+)\.hdf5", txt)})
# drop gated pref demos whose discrete verdict is 0 (e.g. a lay_down that left the
# bottle upright): only possible for families whose criterion needs no anchors yet
try:
    sys.path.insert(0, "scripts"); import calibrate_low_pref as cal, yaml
    y = yaml.safe_load(open(cal._mission_info_path(mid))) or {}
    prefs = cal._build_prefs(y.get("low_prefs") or [])
    if prefs and prefs[0].criterion_threshold is not None:
        out = cal._raw_lists_for_dir(prefs, f"{V}/SinglePref_gated/Mission_{mid}.hdf5", 100)
        sats = out.get(prefs[0].preference_type + "__sat", [])
        bad = [i for i, v in enumerate(sats) if v != 100.0]
        if bad:
            print(f"[pair] {mid}: pref demos failing the criterion, dropped: {bad}")
            ids = [i for i in ids if i not in bad]
except Exception as e:
    print(f"[pair] {mid}: criterion filter skipped ({e})")
assert len(ids) >= 10, f"only {len(ids)} usable pairs"
def copy_subset(src, dst, keys):
    tmp = dst + ".tmp"
    if os.path.exists(tmp): os.remove(tmp)
    with h5py.File(src, "r") as fs, h5py.File(tmp, "w") as fd:
        g = fd.create_group("data")
        for k, v in fs["data"].attrs.items(): g.attrs[k] = v
        for i, k in enumerate(keys): fs.copy(fs[f"data/{k}"], g, name=f"demo_{i}")
    os.replace(tmp, dst)
G = f"{V}/SinglePref_gated/Mission_{mid}.hdf5"; N = f"{V}/SinglePref_naive/Mission_{mid}.hdf5"
with h5py.File(N) as f: nk = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
succ_ids = sorted({int(m) for m in re.findall(r"HDF5 saved: \S+/HDF5/rollout_(\d+)\.hdf5", txt)})
assert len(nk) == len(succ_ids), f"naive demos {len(nk)} != success ids {len(succ_ids)}"
sel = ids[:10]
nk = [nk[succ_ids.index(i)] for i in sel]   # naive demos in the same init order as the kept pref demos
os.makedirs(f"{V}/SinglePref_paired", exist_ok=True); os.makedirs(f"{V}/SinglePref_naive_paired", exist_ok=True)
copy_subset(G, f"{V}/SinglePref_paired/Mission_{mid}.hdf5", [f"demo_{i}" for i in sel])
copy_subset(N, f"{V}/SinglePref_naive_paired/Mission_{mid}.hdf5", nk[:10])
print(f"[pair] {mid}: gated inits used by naive = {sel}; wrote 10 + 10 paired demos")
PYEOF
fi

echo "== [$MID] 4/5 anchors + cohen_d (paired sets)" | tee -a "$OUT"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_paired" --naive_dir "$V/SinglePref_naive_paired" \
  --missions "$MID" --n_demos 10 --apply >> "$OUT" 2>&1 || fail calibrate "calibrate error"
$PY scripts/wire_skill_criterion.py --missions "$MID" >> "$OUT" 2>&1 || fail criterion "wire error"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_paired" --naive_dir "$V/SinglePref_naive_paired" \
  --missions "$MID" --n_demos 10 --table --csv "$LOGD/table_$MID.csv" >> "$OUT" 2>&1

echo "== [$MID] 5/5 video" | tee -a "$OUT"
mkdir -p "$HR/all"
$PY scripts/render_demo_video.py --missions "$MID" --demo demo_0 --demo_dir "$V/SinglePref_paired" \
  --naive_dir "$V/SinglePref_naive_paired" --out "$HR/all" >> "$OUT" 2>&1 || fail video "render error"

echo "{\"mission\":\"$MID\",\"ok\":true,\"gate\":\"$GATE\",\"naive_sr\":\"$NSR\",\"k\":$K}" > "$J"
echo "$MID OK  gate=$GATE naive=$NSR"

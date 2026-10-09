#!/usr/bin/env bash
# 1 系列（motion）第三 mission 收尾：naive 已按 pref init 配对生成（run_series3_naive.sh）→ 取 naive 成功的前 10 个 init 写 paired → 双锚 → 对比视频到 human-review/S1
# → 只保留 naive 成功 且 pref 判据=100 的 init，前 10 对写 paired 文件 → 锚 + 判据率 → 对比视频。
# 用法: bash scripts/finish_motion_mission.sh 1-2_062      （日志 /tmp/s3gen/finish_<id>.log，结果 json 同目录）
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
MID=$1
V=/data3/zhixu/LIBERO-PeRM/VLA_data
HR=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S1
LOGD=/tmp/s3gen; OUT=$LOGD/finish_$MID.log; : > "$OUT"; J=$LOGD/finish_$MID.json
NLOG=${NLOG:-$LOGD/naive_$MID.log}
fail () { echo "{\"mission\":\"$MID\",\"stage\":\"$1\",\"ok\":false,\"note\":\"$2\"}" > "$J"; echo "$MID FAIL@$1: $2"; exit 1; }

echo "== [$MID] 1/3 pair (naive successes x pref criterion)" | tee -a "$OUT"
$PY - "$MID" "$V" "$NLOG" >> "$OUT" 2>&1 <<'PYEOF' || fail pair "$(tail -n 1 $OUT)"
import sys, re, os, h5py
mid, V, log = sys.argv[1:4]
sys.path.insert(0, "scripts"); import calibrate_low_pref as cal, yaml
txt = open(log, errors="ignore").read()
succ_ids = sorted({int(m) for m in re.findall(r"HDF5 saved: \S+/HDF5/rollout_(\d+)\.hdf5", txt)})
P = f"{V}/SinglePref/Mission_{mid}.hdf5"; N = f"{V}/SinglePref_naive/Mission_{mid}.hdf5"
with h5py.File(N) as f: nk = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
assert len(nk) == len(succ_ids), f"naive demos {len(nk)} != success ids {len(succ_ids)} in {log}"
y = yaml.safe_load(open(cal._mission_info_path(mid))) or {}
prefs = cal._build_prefs(y.get("low_prefs") or [])
scan = max(succ_ids) + 1 if succ_ids else 0
out = cal._raw_lists_for_dir(prefs, P, scan)
sats = out.get(prefs[0].preference_type + "__sat", [])
bad = []  # motion metrics are continuous: no discrete criterion gate
if bad: print(f"[pair] {mid}: pref demos failing the criterion on naive-success inits, dropped: {bad}")
ids = [i for i in succ_ids if i not in bad]
assert len(ids) >= 10, f"only {len(ids)} usable pairs (naive successes {len(succ_ids)})"
def copy_subset(src, dst, keys):
    tmp = dst + ".tmp"
    if os.path.exists(tmp): os.remove(tmp)
    with h5py.File(src, "r") as fs, h5py.File(tmp, "w") as fd:
        g = fd.create_group("data")
        for k, v in fs["data"].attrs.items(): g.attrs[k] = v
        for i, k in enumerate(keys): fs.copy(fs[f"data/{k}"], g, name=f"demo_{i}")
    os.replace(tmp, dst)
sel = ids[:10]; nsel = [nk[succ_ids.index(i)] for i in sel]
os.makedirs(f"{V}/SinglePref_paired", exist_ok=True); os.makedirs(f"{V}/SinglePref_naive_paired", exist_ok=True)
copy_subset(P, f"{V}/SinglePref_paired/Mission_{mid}.hdf5", [f"demo_{i}" for i in sel])
copy_subset(N, f"{V}/SinglePref_naive_paired/Mission_{mid}.hdf5", nsel)
n_att = len(re.findall(r"paired-init\] batch", txt))
print(f"[pair] {mid}: inits {sel}; naive SR {len(succ_ids)}/{n_att}; wrote 10 + 10 paired demos")
PYEOF
PAIR=$(grep -a "^\[pair\] $MID: inits" "$OUT" | tail -n 1)

echo "== [$MID] 2/3 anchors + verdict rates (paired sets)" | tee -a "$OUT"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_paired" --naive_dir "$V/SinglePref_naive_paired" \
  --missions "$MID" --n_demos 10 --apply >> "$OUT" 2>&1 || fail calibrate "calibrate error"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_paired" --naive_dir "$V/SinglePref_naive_paired" \
  --missions "$MID" --n_demos 10 --table --csv "$LOGD/table_$MID.csv" >> "$OUT" 2>&1

echo "== [$MID] 3/3 video + contact sheet" | tee -a "$OUT"
mkdir -p "$HR/all"
$PY scripts/render_demo_video.py --missions "$MID" --demo demo_0 --demo_dir "$V/SinglePref_paired" \
  --naive_dir "$V/SinglePref_naive_paired" --out "$HR/all" >> "$OUT" 2>&1 || fail video "render error"
$PY scripts/skill_contact_sheet.py --missions "$MID" --out "$HR/sheets" >> "$OUT" 2>&1 || echo "[sheet] failed (non-fatal)" >> "$OUT"
SAT=$(tail -n 1 "$LOGD/table_$MID.csv" | cut -d, -f7,8)
echo "{\"mission\":\"$MID\",\"ok\":true,\"pair\":\"${PAIR//\"/}\",\"sat_pref_naive\":\"$SAT\"}" > "$J"
echo "$MID OK  sat(pref,naive)=$SAT  $PAIR"

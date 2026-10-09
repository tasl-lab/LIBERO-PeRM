#!/usr/bin/env bash
# pref-only 收尾（没有可用 naive 的 mission，如 2-7/2-8/2-9 OOD）：按离散判据 gate pref demo → 取前 10 条写 paired
# → 单锚校准 + 判据写进 yaml → 单排视频。产物同 v2（SinglePref_paired、finish_<id>.json/.csv、human-review/all/<id>_demo_0.mp4）。
set -uo pipefail
cd /data3/zhixu/LIBERO-PeRM/LIBERO
PY=/home/zli538/miniconda3/envs/libero_DataGen/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
MID=$1; SCAN=${SCAN:-30}
V=/data3/zhixu/LIBERO-PeRM/VLA_data
HR=/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2
LOGD=/tmp/skillgen; OUT=$LOGD/finish_$MID.log; [ -f "$OUT" ] && mv -f "$OUT" "$OUT.prev"; : > "$OUT"
J=$LOGD/finish_$MID.json
fail () { echo "{\"mission\":\"$MID\",\"stage\":\"$1\",\"ok\":false,\"note\":\"$2\"}" > "$J"; echo "$MID FAIL@$1: $2"; exit 1; }

echo "== [$MID] 1/3 gate pref demos by the discrete criterion (pref-only)" | tee -a "$OUT"
$PY scripts/wire_skill_criterion.py --missions "$MID" >> "$OUT" 2>&1
$PY - "$MID" "$V" "$SCAN" "$LOGD" >> "$OUT" 2>&1 <<'PYEOF' || fail gate "criterion gate failed (see log)"
import sys, os, csv, h5py, numpy as np, yaml
sys.path.insert(0, "scripts"); import calibrate_low_pref as cal
mid, V, scan, logd = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
info = cal._mission_info_path(mid); data = yaml.safe_load(open(info)) or {}
prefs = cal._build_prefs(data.get("low_prefs") or [])
assert prefs, "no low_prefs wired"
tobj = [str(s[1]) for s in (data.get("task_plan") or []) if isinstance(s, (list, tuple)) and len(s) >= 2]
for p in prefs:
    if hasattr(p, "set_task_context"): p.set_task_context(tobj)
    if hasattr(p, "set_task_plan"): p.set_task_plan(data.get("task_plan"))
src = f"{V}/SinglePref/Mission_{mid}.hdf5"
out = cal._raw_lists_for_dir(prefs, src, scan)
t = prefs[0].preference_type; raws = out.get(t, []); sats = out.get(t + "__sat", [])
keep = [i for i, s in enumerate(sats) if s == 100.0]
print(f"{mid} [{t}] criterion: {'raw >=' if prefs[0].criterion_direction > 0 else 'raw <='} {prefs[0].criterion_threshold:.4f}  ->  {len(keep)}/{len(sats)} pref demos satisfy")
print(f"    raws: {np.round(raws, 4).tolist()}")
assert len(keep) >= 10, f"only {len(keep)} satisfy"
with h5py.File(src) as f: keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:scan]
sel = [keys[i] for i in keep[:10]]
os.makedirs(f"{V}/SinglePref_paired", exist_ok=True)
dst = f"{V}/SinglePref_paired/Mission_{mid}.hdf5"; tmp = dst + ".tmp"
if os.path.exists(tmp): os.remove(tmp)
with h5py.File(src, "r") as fs, h5py.File(tmp, "w") as fd:
    g = fd.create_group("data")
    for k, v in fs["data"].attrs.items(): g.attrs[k] = v
    for i, k in enumerate(sel): fs.copy(fs[f"data/{k}"], g, name=f"demo_{i}")
os.replace(tmp, dst)
kept = [raws[i] for i in keep[:10]]
with open(f"{logd}/table_{mid}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mission", "pref_type", "mean_pref", "mean_naive", "gap", "cohen_d", "sat_pref", "sat_naive", "n_pref", "n_naive"])
    w.writeheader(); w.writerow({"mission": mid, "pref_type": t, "mean_pref": float(np.mean(kept)), "mean_naive": "", "gap": "", "cohen_d": "", "sat_pref": 100.0, "sat_naive": "", "n_pref": 10, "n_naive": 0})
open(f"{logd}/gate_{mid}.txt", "w").write(f"{len(keep)}/{len(sats)}\n")
print(f"[pair] {mid}: pref-only, kept demos {sel}")
PYEOF
GATE=$(cat "$LOGD/gate_$MID.txt")

echo "== [$MID] 2/3 single-anchor calibration (anchor only; the score is the discrete criterion)" | tee -a "$OUT"
$PY scripts/calibrate_low_pref.py --suite single-pref --pref_dir "$V/SinglePref_paired" --missions "$MID" --n_demos 10 --apply >> "$OUT" 2>&1 || fail calibrate "calibrate error"
$PY scripts/wire_skill_criterion.py --missions "$MID" >> "$OUT" 2>&1

echo "== [$MID] 3/3 video (pref row only)" | tee -a "$OUT"
mkdir -p "$HR/all"
$PY scripts/render_demo_video.py --missions "$MID" --demo demo_0 --demo_dir "$V/SinglePref_paired" --out "$HR/all" >> "$OUT" 2>&1 || fail video "render error"
echo "{\"mission\":\"$MID\",\"ok\":true,\"gate\":\"$GATE\",\"naive_sr\":\"pref-only\",\"k\":0}" > "$J"
echo "$MID OK  gate=$GATE (pref-only, no naive)"

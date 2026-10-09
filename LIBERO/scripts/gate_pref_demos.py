"""Keep only the pref demos that actually satisfy their own preference.

A rollout is stored when the BDDL goal fires (`dones=1`). For series-2 that is
not enough: `lay_down` reaches "bottle is on the cabinet" whether or not the
bottle ended up on its side, so 4/10 of the 2-14_109 pilot pref demos scored
object_tilt ~0.07 rad (upright) next to 6 that scored 0.80-1.08. Those demos
are task-successes but preference-failures, and they drag the anchor and the
comparison video with them.

Gate: replay the naive demos to get the score-0 distribution, then require each
pref demo to sit at least `k` robust sigma beyond it, in whichever direction the
preference points:

    N     = median(naive raw)
    sigma = 1.4826 * MAD(naive raw)        (robust sd; floored, see SIGMA_FLOOR)
    dir   = sign(median(pref raw) - N)
    keep  <=>  dir * (raw - (N + dir * k * sigma))  >= 0

Writes a filtered copy holding the first `--target` survivors, so downstream
calibration / rendering see a clean pref set.

Usage:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python scripts/gate_pref_demos.py --missions 2-14_109 [--apply] [--k 3]
Dry-run by default (prints per-demo raws + how many survive).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys

import h5py
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LIBERO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_ROOT)

_spec = importlib.util.spec_from_file_location("_cal", os.path.join(_HERE, "calibrate_low_pref.py"))
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

PREF_DIR = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref"
NAIVE_DIR = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref_naive"
# A naive set can be near-deterministic (MAD -> 0); without a floor the gate
# would collapse onto the naive median and admit anything a hair beyond it.
SIGMA_FLOOR_FRAC = 0.05  # of |median(pref) - N|


def _raws(mission_id: str, hdf5: str, n: int):
    info = cal._mission_info_path(mission_id)
    data = yaml.safe_load(open(info)) or {}
    prefs = cal._build_prefs(data.get("low_prefs") or [])
    if not prefs:
        return None, None
    tobj = [str(s[1]) for s in (data.get("task_plan") or [])
            if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]]
    for p in prefs:
        if hasattr(p, "set_task_context"):
            p.set_task_context(tobj)
        if hasattr(p, "set_task_plan"):
            p.set_task_plan(data.get("task_plan"))
    out = cal._raw_lists_for_dir(prefs, hdf5, n)
    t = prefs[0].preference_type
    return t, [float(v) for v in out.get(t, [])]


def _copy_subset(src: str, dst: str, keys) -> None:
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with h5py.File(src, "r") as fs, h5py.File(tmp, "w") as fd:
        g = fd.create_group("data")
        for k, v in fs["data"].attrs.items():
            g.attrs[k] = v
        for i, k in enumerate(keys):
            fs.copy(fs[f"data/{k}"], g, name=f"demo_{i}")
    os.replace(tmp, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", required=True)
    ap.add_argument("--pref_dir", default=PREF_DIR)
    ap.add_argument("--naive_dir", default=NAIVE_DIR)
    ap.add_argument("--scan", type=int, default=100, help="pref demos to replay")
    ap.add_argument("--target", type=int, default=10, help="minimum survivors required")
    ap.add_argument("--cap", type=int, default=100, help="max survivors to write out")
    ap.add_argument("--k", type=float, default=3.0, help="robust sigmas beyond naive")
    ap.add_argument("--out_dir", default="/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref_gated",
                    help="where the filtered pref demos are written")
    ap.add_argument("--apply", action="store_true", help="write the filtered copy")
    a = ap.parse_args()

    for mid in a.missions.split(","):
        ph = cal._hdf5_in_dir(a.pref_dir, mid)
        nh = cal._hdf5_in_dir(a.naive_dir, mid)
        if ph is None or nh is None:
            print(f"  [skip] {mid}: pref={ph} naive={nh}")
            continue
        t, nv = _raws(mid, nh, 100)
        _, pv = _raws(mid, ph, a.scan)
        if not t or not nv or not pv:
            print(f"  [skip] {mid}: no metric wired or replay failed")
            continue
        nv, pv = np.asarray(nv), np.asarray(pv)
        N = float(np.median(nv))
        mad = float(np.median(np.abs(nv - N))) * 1.4826
        direction = 1.0 if float(np.median(pv)) >= N else -1.0
        sigma = max(mad, SIGMA_FLOOR_FRAC * abs(float(np.median(pv)) - N))
        gate = N + direction * a.k * sigma
        keep = np.nonzero(direction * (pv - gate) >= 0)[0]
        print(f"{mid}  [{t}]  N={N:+.4f} sigma={sigma:.4f} dir={'+' if direction > 0 else '-'} "
              f"gate={gate:+.4f}  ->  {len(keep)}/{len(pv)} pref demos pass")
        print(f"    pref raws: {np.round(pv, 4).tolist()}")
        if len(keep) < a.target:
            print(f"    !! only {len(keep)} pass, need {a.target} — generate more pref rollouts")
        if a.apply and len(keep) >= a.target:
            with h5py.File(ph, "r") as f:
                ks = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[:a.scan]
            sel = [ks[i] for i in keep[:a.cap]]
            # A filtered COPY, never in-place: VLA_data/SinglePref is the training
            # corpus and dropping 77% of a mission's demos there is unrecoverable.
            os.makedirs(a.out_dir, exist_ok=True)
            dst = os.path.join(a.out_dir, os.path.basename(ph))
            _copy_subset(ph, dst, sel)
            print(f"    wrote {len(sel)} demos -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

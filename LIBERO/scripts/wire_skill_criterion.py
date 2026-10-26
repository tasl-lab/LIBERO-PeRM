#!/usr/bin/env python
"""Write each series-2 mission's discrete criterion into its low_prefs params.

Skill preferences are scored at their node as 100 / 0:
    100  <=>  criterion_direction * (raw - criterion_threshold) >= 0
Most families carry a physical default in Preference/low_level/skill_pose.py
(30 deg tilt, 1 cm lift, 5 cm drop, 30 deg rotation). grasp_offset has none:
its sign convention depends on the object and the skill pair, so threshold =
midpoint of the paired pref / naive anchors and direction = sign(P - N), read
from the `low_pref_calibration` block calibrate_low_pref.py --apply wrote.

The values are written explicitly for every family so the yaml documents the
criterion that was used. Text-level edit: comments in the yaml survive.

Usage: cd LIBERO && python scripts/wire_skill_criterion.py --missions 2-1_019[,...] [--dry]
"""
import argparse, math, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, LIBERO); sys.path.insert(0, HERE)
import calibrate_low_pref as cal  # noqa: E402
from Preference.low_level import skill_pose as sp  # noqa: E402

CLASS = {"grasp_offset": sp.GraspOffsetPreference, "grasp_tilt": sp.GraspTiltPreference,
         "place_rotation": sp.PlaceRotationPreference, "object_tilt": sp.ObjectTiltPreference,
         "release_drop": sp.ReleaseDropPreference, "object_lift": sp.ObjectLiftPreference}


def criterion_for(mission_id, data):
    lp = (data.get("low_prefs") or [None])[0]
    if not lp or lp.get("type") not in CLASS:
        return None, "no skill low_pref"
    t = lp["type"]; cls = CLASS[t]
    if cls.CRITERION_THRESHOLD is not None:
        return (cls.CRITERION_THRESHOLD, cls.CRITERION_DIRECTION), "class default"
    if t == "grasp_offset" and str((lp.get("params") or {}).get("axis", "")).lower() == "handle":
        return (sp.GraspOffsetPreference.HANDLE_CRITERION_M, 1), "handle-projection default (>= 2.5 cm toward the handle)"
    cal_blk = (data.get("low_pref_calibration") or {}).get(t) or {}
    P, N = cal_blk.get("pref_anchor"), cal_blk.get("naive_anchor")
    if P is None or N is None:
        return None, f"{t} needs pref/naive anchors first (calibrate --apply with --naive_dir)"
    if abs(P - N) < 1e-4:
        return None, f"anchors too close (P={P}, N={N})"
    return ((P + N) / 2.0, 1 if P > N else -1), f"midpoint of anchors P={P:+.4f} N={N:+.4f}"


def write_params(text, threshold, direction):
    """Insert / replace criterion_* lines inside the first low_prefs params block."""
    text = re.sub(r"^    criterion_(threshold|direction):.*\n", "", text, flags=re.M)
    m = re.search(r"^low_prefs:\n- type: .*\n  params:\n((?:    .*\n)+)", text, flags=re.M)
    if not m:
        raise RuntimeError("low_prefs params block not found")
    ins = f"    criterion_threshold: {threshold:.6f}\n    criterion_direction: {direction}\n"
    return text[:m.end(1)] + ins + text[m.end(1):]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--missions", required=True); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for mid in a.missions.split(","):
        info = cal._mission_info_path(mid)
        if not info or not os.path.exists(info):
            print(f"  [skip] {mid}: no mission_info"); continue
        data = yaml.safe_load(open(info)) or {}
        crit, why = criterion_for(mid, data)
        if crit is None:
            print(f"  [skip] {mid}: {why}"); continue
        thr, d = crit
        print(f"  {mid}: {data['low_prefs'][0]['type']}  100 <=> {'raw >=' if d > 0 else 'raw <='} {thr:.4f}   ({why})")
        if not a.dry:
            # read fully BEFORE opening for write (open(..,"w") truncates first)
            text = open(info).read()
            new_text = write_params(text, thr, d)
            with open(info, "w") as f:
                f.write(new_text)


if __name__ == "__main__":
    main()

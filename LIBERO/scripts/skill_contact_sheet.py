#!/usr/bin/env python
"""Contact sheet for the VLM double-check of a skill-pref mission.

One tile per paired demo at the skill's NODE step (first step where the
discrete verdict exists): agentview | eye-in-hand, label = demo / raw / verdict.
Rows 1-2 = the 10 pref demos, rows 3-4 = the 10 naive demos (pref-only
missions: 2 rows). Also writes <out>/<mid>.json with node / raw / verdict per
demo so the judge's calls can be scored against the metric.

Usage: cd LIBERO && python scripts/skill_contact_sheet.py --missions 2-1_019,... [--out DIR]
"""
import argparse, json, os, sys
import numpy as np, cv2, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, LIBERO); sys.path.insert(0, HERE)
import calibrate_low_pref as cal          # noqa: E402
import render_demo_video as rdv           # noqa: E402

V = "/data3/zhixu/LIBERO-PeRM/VLA_data"
OUT_DEFAULT = "/data3/zhixu/PeRM_data/demos/SinglePref-demo/human-review/S2/sheets"
T = 256   # tile image size


def node_info(mid, h5, demo):
    sat, raw, labels, t_succ, n, _ = rdv.replay_satisfaction(mid, h5, demo)
    if not sat:
        return None
    k = list(sat.keys())[0]
    s = np.asarray(sat[k], dtype=float); r = np.asarray(raw[k], dtype=float)
    idx = np.nonzero(~np.isnan(s))[0]
    node = int(idx[0]) if len(idx) else None
    verdict = float(s[idx[-1]]) if len(idx) else None
    ridx = np.nonzero(~np.isnan(r))[0]
    raw_final = float(r[ridx[-1]]) if len(ridx) else None
    # frame to SHOW the judge = the instant the preference is visible, not the
    # step where the verdict settles: grasp / release instant, or the peak of
    # a running-max metric (highest lift, largest tilt)
    if k in ("object_lift", "object_tilt", "ObjectLiftPreference", "ObjectTiltPreference") and len(ridx):
        show = int(ridx[np.argmax(r[ridx])])
    elif k in ("place_rotation", "PlaceRotationPreference"):
        show = node
    else:
        show = int(ridx[0]) if len(ridx) else node
    return {"pref_type": k, "node": node, "show": show, "verdict": verdict, "raw": raw_final, "n": int(n), "t_success": t_succ}


def tile(h5, demo, info, title):
    with h5py.File(h5, "r") as f:
        g = f[f"data/{demo}"]; n = g["obs/agentview_rgb"].shape[0]
        t = min(info["show"] if info and info.get("show") is not None else n - 1, n - 1)
        a = g["obs/agentview_rgb"][t]; e = g["obs/eye_in_hand_rgb"][t] if "eye_in_hand_rgb" in g["obs"] else a
    a = cv2.resize(np.ascontiguousarray(a), (T, T)); e = cv2.resize(np.ascontiguousarray(e), (T, T))
    im = np.hstack([a, e])
    bar = np.zeros((22, im.shape[1], 3), np.uint8)
    v = "?" if not info or info["verdict"] is None else ("SAT" if info["verdict"] >= 100 else "no")
    rawtxt = "-" if not info or info["raw"] is None else f"{info['raw']:+.3f}"
    cv2.putText(bar, f"{title} {demo}  t={t}  raw={rawtxt}  metric:{v}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 255, 80) if v == "SAT" else ((80, 80, 255) if v == "no" else (200, 200, 200)), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def sheet(mid, out_dir):
    ph = f"{V}/SinglePref_paired/Mission_{mid}.hdf5"; nh = f"{V}/SinglePref_naive_paired/Mission_{mid}.hdf5"
    if not os.path.exists(ph):
        print(f"  {mid}: no paired pref"); return
    rec = {"mission": mid, "pref": [], "naive": []}
    rows = []
    for tag, h5 in (("PREF", ph), ("NAIVE", nh if os.path.exists(nh) else None)):
        if h5 is None:
            continue
        with h5py.File(h5, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:10]
        tiles = []
        for d in demos:
            info = node_info(mid, h5, d)
            rec["pref" if tag == "PREF" else "naive"].append({"demo": d, **(info or {})})
            tiles.append(tile(h5, d, info, tag))
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        for i in range(0, len(tiles), 5):
            rows.append(np.hstack(tiles[i:i + 5]))
    # header: what the judge should look for
    import yaml
    y = yaml.safe_load(open(cal._mission_info_path(mid))) or {}
    lp = (y.get("low_prefs") or [{}])[0]; prm = lp.get("params", {})
    thr, dr = prm.get("criterion_threshold"), prm.get("criterion_direction", 1)
    head = np.zeros((30, rows[0].shape[1], 3), np.uint8)
    txt = (f"{mid}  pref='{lp.get('description','')}'  metric={lp.get('type')}  criterion: "
           f"{'raw >= ' if (dr or 1) > 0 else 'raw <= '}{thr if thr is not None else '?'}   (top rows = pref demos, bottom rows = no-pref demos)")
    cv2.putText(head, txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.vstack([head] + rows)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{mid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    json.dump(rec, open(os.path.join(out_dir, f"{mid}.json"), "w"), indent=1)
    vp = [x.get("verdict") for x in rec["pref"]]; vn = [x.get("verdict") for x in rec["naive"]]
    print(f"  {mid}: pref SAT {sum(1 for v in vp if v == 100)}/{len(vp)}  naive SAT {sum(1 for v in vn if v == 100)}/{len(vn)}  -> {out_dir}/{mid}.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--missions", required=True); ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args()
    for mid in a.missions.split(","):
        try:
            sheet(mid, a.out)
        except Exception as e:
            print(f"  {mid}: ERROR {e}")


if __name__ == "__main__":
    main()

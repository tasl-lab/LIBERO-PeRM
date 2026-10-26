#!/usr/bin/env python
"""Measure candidate motion-tier raw quantities straight from demo HDF5 obs
(no sim replay). Part of the SA metric rework (plan A1 "先量再动").

For every motion mission, compares GT-pref demos vs naive demos on:
  T_succ          steps until first dones==1 (post-success tail excluded)
  T_total         recorded steps (what the current SpeedRange raw uses)
  v_move          mean EE speed over moving steps (|dp|>1mm), m/step
  v_carry         mean EE speed inside the carry window
  acc_peak/acc_p95  peak / P95 of |dv| over [0, T_succ)
  z_max/z_min     carry-window EE z, relative to z at carry start (m)
  straight        transport-segment (carry start -> final descent) path/chord - 1
  rot_step        mean per-step rotation angle on lateral steps (rad)
  rot_net         total rotation angle accumulated over the carry window (rad)
  rot_total       total per-step rotation over [0, T_succ) (rad)
  rot_lat_frac    share of rot_total that happens on laterally-moving steps
  cur_reward      mean of the per-step low_pref_rewards stored in the file
                  (= what the current online chain would score)

Carry window = first gripper-closed step -> first re-open after it
(finger gap < GAP_CLOSED). Falls back to [0, T_succ) when no close is found.

Usage:
  cd LIBERO && python scripts/measure_motion_raw.py \
      --pref_dir /data4/zhixu/SinglePref --naive_dir VLA_data/SinglePref_naive \
      [--missions 1-1_004,1-7_104] [--n_demos 100] [--out table.md] [--csv raw.csv]
"""
import argparse, glob, os, sys, math
import numpy as np
import h5py

GAP_CLOSED = 0.07     # finger gap (m); open ~0.08
MOVE_EPS = 1e-3       # m/step
LATERAL_EPS = 1e-3


def rel_angle(e1, e2):
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler("XYZ", e1).inv() * R.from_euler("XYZ", e2)
    return float(np.linalg.norm(r.as_rotvec()))


def carry_window(gap, t_succ):
    closed = np.where(gap[:t_succ] < GAP_CLOSED)[0]
    if len(closed) == 0:
        return 0, t_succ
    s = int(closed[0])
    reopen = np.where(gap[s:t_succ] >= GAP_CLOSED)[0]
    e = int(s + reopen[0]) if len(reopen) else t_succ
    return s, max(e, s + 2)


def measure_demo(g):
    ee = g["obs/ee_pos"][:]
    ori = g["obs/ee_ori"][:]
    gs = g["obs/gripper_states"][:]
    dones = g["dones"][:]
    T = len(ee)
    succ = np.where(dones == 1)[0]
    t_succ = int(succ[0]) + 1 if len(succ) else T
    gap = np.abs(gs[:, 0] - gs[:, 1])
    s, e = carry_window(gap, t_succ)

    dp = np.diff(ee[:t_succ], axis=0)
    sp = np.linalg.norm(dp, axis=1)
    moving = sp > MOVE_EPS
    dv = np.diff(dp, axis=0)
    acc = np.linalg.norm(dv, axis=1)

    seg = ee[s:e]
    # transport segment = carry start -> start of the final descent
    # (last step whose z is within 1 cm of the carry-window z max)
    zmax = seg[:, 2].max()
    t_desc = int(np.where(seg[:, 2] >= zmax - 0.01)[0][-1])
    tr = seg[: t_desc + 1]
    path = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))) if len(tr) > 1 else 0.0
    chord = float(np.linalg.norm(tr[-1] - tr[0])) if len(tr) > 1 else 0.0
    straight = path / chord - 1 if chord > 1e-3 else float("nan")

    dxy = np.linalg.norm(np.diff(ee[s:e, :2], axis=0), axis=1)
    dz = np.abs(np.diff(ee[s:e, 2]))
    lateral = np.where(dxy > LATERAL_EPS)[0]
    # lateral-dominated runs (|dxy| > 2|dz|), >=5 steps: per-run path/chord-1, length-weighted
    lat_dom = (dxy > LATERAL_EPS) & (dxy > 2 * dz)
    runs, i = [], 0
    while i < len(lat_dom):
        if lat_dom[i]:
            j = i
            while j < len(lat_dom) and lat_dom[j]:
                j += 1
            if j - i >= 5:
                runs.append((i, j))
            i = j
        else:
            i += 1
    num = den = 0.0
    for (i0, j0) in runs:
        pts = seg[i0:j0 + 1]
        pl = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        ch = float(np.linalg.norm(pts[-1] - pts[0]))
        if ch > 1e-3:
            num += (pl / ch - 1) * pl
            den += pl
    straight_lat = num / den if den > 0 else float("nan")
    rot_steps = [rel_angle(ori[s + i], ori[s + i + 1]) for i in lateral]
    rot_step = float(np.mean(rot_steps)) if rot_steps else 0.0
    rot_net = rel_angle(ori[s], ori[e - 1]) if e - 1 > s else 0.0
    # whole-episode (pre-success) rotation: share that happens while moving laterally
    from scipy.spatial.transform import Rotation as R
    rots = R.from_euler("XYZ", ori[:t_succ])
    ang = np.linalg.norm((rots[:-1].inv() * rots[1:]).as_rotvec(), axis=1)
    lat_all = np.linalg.norm(dp[:, :2], axis=1) > LATERAL_EPS
    rot_total = float(ang.sum())
    rot_lat_frac = float(ang[lat_all].sum() / rot_total) if rot_total > 1e-6 else float("nan")

    out = dict(
        T_succ=t_succ, T_total=T,
        v_move=float(sp[moving].mean()) if moving.any() else 0.0,
        v_carry=float(np.linalg.norm(np.diff(seg, axis=0), axis=1).mean()) if len(seg) > 1 else 0.0,
        acc_peak=float(acc.max()) if len(acc) else 0.0,
        acc_p95=float(np.percentile(acc, 95)) if len(acc) else 0.0,
        z_max=float(seg[:, 2].max() - seg[0, 2]),
        z_min=float(seg[:, 2].min() - seg[0, 2]),
        straight=straight, straight_lat=straight_lat, rot_step=rot_step, rot_net=rot_net,
        rot_total=rot_total, rot_lat_frac=rot_lat_frac,
        carry_len=e - s,
    )
    if "low_pref_rewards" in g:
        lr = g["low_pref_rewards"]
        vals = []
        if isinstance(lr, h5py.Dataset):
            vals.append(float(np.mean(lr[:])))
        else:
            for k in lr:
                vals.append(float(np.mean(lr[k][:])))
        out["cur_reward"] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_dir(d, mid, n):
    p = os.path.join(d, f"Mission_{mid}.hdf5")
    if not os.path.exists(p):
        return []
    rows = []
    with h5py.File(p, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:n]
        for k in keys:
            rows.append(measure_demo(f["data"][k]))
    return rows


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 1:
        return float("nan")
    sa = a.std(ddof=1)
    sb = b.std(ddof=1) if len(b) > 1 else sa
    pooled = math.sqrt(((len(a) - 1) * sa**2 + (max(len(b) - 1, 1)) * sb**2) / max(len(a) + len(b) - 2, 1))
    return float((np.median(a) - np.median(b)) / pooled) if pooled > 1e-12 else float("inf")


def auroc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
    return float(max(gt, 1 - gt))


QUANTS = ["T_succ", "T_total", "v_move", "v_carry", "acc_peak", "acc_p95",
          "z_max", "z_min", "straight", "straight_lat", "rot_step", "rot_net", "rot_total", "rot_lat_frac", "cur_reward"]
RELEVANT = {
    "1-1": ["T_succ", "T_total", "v_move", "v_carry"],
    "1-2": ["T_succ", "T_total", "v_move", "v_carry"],
    "1-3": ["acc_peak", "acc_p95"],
    "1-4": [],  # avoid: needs object pos from sim state; not measurable from obs
    "1-5": ["z_max", "z_min"],
    "1-6": ["straight", "straight_lat"],
    "1-7": ["z_max", "z_min"],
    "1-8": ["rot_step", "rot_net", "rot_total", "rot_lat_frac"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref_dir", default="/data4/zhixu/SinglePref")
    ap.add_argument("--naive_dir", default="VLA_data/SinglePref_naive")
    ap.add_argument("--missions", default=None)
    ap.add_argument("--n_demos", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.missions:
        mids = a.missions.split(",")
    else:
        mids = sorted(os.path.basename(p)[8:-5] for p in glob.glob(os.path.join(a.pref_dir, "Mission_1-*.hdf5")))

    lines, csv_rows = [], []
    for mid in mids:
        fam = mid.split("_")[0]
        pref = load_dir(a.pref_dir, mid, a.n_demos)
        naive = load_dir(a.naive_dir, mid, a.n_demos)
        if not pref:
            lines.append(f"\n## {mid}: no pref demos\n"); continue
        lines.append(f"\n## {mid}  (pref n={len(pref)}, naive n={len(naive)})  relevant: {RELEVANT.get(fam, [])}\n")
        lines.append("| quantity | pref median | pref IQR | naive median | naive IQR | cohen_d | AUROC |")
        lines.append("|---|---|---|---|---|---|---|")
        for q in QUANTS:
            pa = np.array([r.get(q, np.nan) for r in pref], float)
            na = np.array([r.get(q, np.nan) for r in naive], float) if naive else np.array([])
            pm, pq = np.nanmedian(pa), (np.nanpercentile(pa, 75) - np.nanpercentile(pa, 25))
            nm = np.nanmedian(na) if len(na) else float("nan")
            nq = (np.nanpercentile(na, 75) - np.nanpercentile(na, 25)) if len(na) > 1 else 0.0
            d = cohen_d(pa, na) if len(na) else float("nan")
            au = auroc(pa, na) if len(na) else float("nan")
            mark = " **" if q in RELEVANT.get(fam, []) else ""
            lines.append(f"| {q}{mark} | {pm:.4g} | {pq:.3g} | {nm:.4g} | {nq:.3g} | {d:.2f} | {au:.2f} |")
            csv_rows.append((mid, q, pm, pq, nm, nq, d, au, len(pa), len(na)))
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write("# Motion raw quantities from demo obs (pref vs naive)\n" + text + "\n")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["mission", "quantity", "pref_med", "pref_iqr", "naive_med", "naive_iqr", "cohen_d", "auroc", "n_pref", "n_naive"])
            w.writerows(csv_rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Score demos with the reworked motion metrics: pref demos should get high
satisfaction AND succeed, naive (no-pref) demos should score low.

Per mission:
  1. replay pref + naive demos through the mission's motion prefs (same
     sim-replay path as calibrate_low_pref; raws cached to JSON so re-runs
     with more naive demos only replay what is new),
  2. dual anchors from the EVEN-indexed demos (pref_anchor = median pref raw,
     naive_anchor = median naive raw),
  3. satisfaction evaluated on the ODD-indexed demos (held out), plus SR from
     HDF5 `dones`, cohen_d / AUROC on all raws.

Usage:
  cd LIBERO && python scripts/score_motion_demos.py \
      --pref_dir /data4/zhixu/SinglePref --naive_dir VLA_data/SinglePref_naive \
      [--missions 1-1_004,...] [--n_pref 100] [--n_naive 10] \
      [--cache logs/raw_cache.json] [--out table.md] [--csv table.csv]
"""
import argparse, json, math, os, sys, time
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, LIBERO)
sys.path.insert(0, HERE)

import calibrate_low_pref as cal  # noqa: E402

MOTION_MISSIONS = ["1-1_004", "1-1_006", "1-2_113", "1-3_107", "1-3_109",
                   "1-4_034", "1-5_114", "1-6_105", "1-7_104", "1-8_114"]


def _successes(hdf5_path, n):
    out = []
    with h5py.File(hdf5_path, "r") as f:
        for k in list(f["data"].keys())[:n]:
            d = f[f"data/{k}/dones"][:] if f"data/{k}/dones" in f else np.zeros(1)
            out.append(bool(np.any(d == 1)))
    return out


def _cache_key(path, n):
    st = os.stat(path)
    return f"{os.path.realpath(path)}|{int(st.st_mtime)}|{st.st_size}|{n}"


def replay_dir(prefs, path, n, cache):
    """{pref_type: [raw per demo]} + [success per demo], cached."""
    key = _cache_key(path, n)
    if key in cache:
        return cache[key]["raws"], cache[key]["succ"]
    t0 = time.time()
    raws = cal._raw_lists_for_dir(prefs, path, n)
    succ = _successes(path, n)
    cache[key] = {"raws": raws, "succ": succ, "seconds": round(time.time() - t0, 1)}
    return raws, succ


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else float("inf")


def auroc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
    return float(max(gt, 1 - gt))


def sat(r, pref_anchor, naive_anchor):
    denom = pref_anchor - naive_anchor
    if abs(denom) < 1e-9:
        return float("nan")
    return float(min(max((r - naive_anchor) / denom, 0.0), 1.0) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref_dir", default="/data4/zhixu/SinglePref")
    ap.add_argument("--naive_dir", default=os.path.join(LIBERO, "..", "VLA_data", "SinglePref_naive"))
    ap.add_argument("--missions", default=",".join(MOTION_MISSIONS))
    ap.add_argument("--n_pref", type=int, default=100)
    ap.add_argument("--n_naive", type=int, default=10)
    ap.add_argument("--cache", default=os.path.join(LIBERO, "logs", "motion_raw_cache.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pref_only", action="store_true", help="replay/cache pref demos only")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    from PolicyEval.low_pref_reward import load_motion_prefs

    rows, lines = [], []
    hdr = ("| mission | pref | n_pref/n_naive | pref raw med | naive raw med | cohen_d | AUROC | "
           "SR_pref | sat_pref mean / min / %>=70 | SA_pref | sat_naive mean / max / %<=30 |")
    lines += [hdr, "|" + "---|" * 11]
    for mid in a.missions.split(","):
        info = cal._mission_info_path(mid)
        pref_h5 = cal._hdf5_in_dir(a.pref_dir, mid)
        naive_h5 = cal._hdf5_in_dir(a.naive_dir, mid)
        if info is None or pref_h5 is None:
            lines.append(f"| {mid} | – | missing pref demos | | | | | | | | |"); continue
        prefs = load_motion_prefs(info)
        if not prefs:
            continue
        praw, psucc = replay_dir(prefs, pref_h5, a.n_pref, cache)
        json.dump(cache, open(a.cache, "w"))
        if a.pref_only or naive_h5 is None:
            for pt, vals in praw.items():
                lines.append(f"| {mid} | {pt} | {len(vals)}/0 | {np.median(vals):.4g} | – | | | "
                             f"{np.mean(psucc):.2f} | (no naive) | | |")
            continue
        nraw, nsucc = replay_dir(prefs, naive_h5, a.n_naive, cache)
        json.dump(cache, open(a.cache, "w"))
        for pt in praw:
            p_all, n_all = np.asarray(praw[pt], float), np.asarray(nraw.get(pt, []), float)
            if len(n_all) == 0:
                continue
            p_cal, p_ev = p_all[0::2], p_all[1::2]
            n_cal, n_ev = (n_all[0::2], n_all[1::2]) if len(n_all) >= 2 else (n_all, n_all)
            pa, na = float(np.median(p_cal)), float(np.median(n_cal))
            s_p = np.array([sat(r, pa, na) for r in p_ev])
            s_n = np.array([sat(r, pa, na) for r in n_ev])
            succ_ev = np.asarray(psucc, bool)[1::2][: len(s_p)]
            sa_p = float(np.mean(np.where(succ_ev, s_p, 0.0))) if len(s_p) else float("nan")
            row = dict(mission=mid, pref_type=pt, n_pref=len(p_all), n_naive=len(n_all),
                       pref_raw_med=float(np.median(p_all)), naive_raw_med=float(np.median(n_all)),
                       pref_anchor=pa, naive_anchor=na, cohen_d=cohen_d(p_all, n_all), auroc=auroc(p_all, n_all),
                       sr_pref=float(np.mean(psucc)), sr_naive=float(np.mean(nsucc)),
                       sat_pref_mean=float(np.nanmean(s_p)), sat_pref_min=float(np.nanmin(s_p)),
                       sat_pref_ge70=float(np.mean(s_p >= 70)), sa_pref=sa_p,
                       sat_naive_mean=float(np.nanmean(s_n)), sat_naive_max=float(np.nanmax(s_n)),
                       sat_naive_le30=float(np.mean(s_n <= 30)))
            rows.append(row)
            lines.append(f"| {mid} | {pt} | {row['n_pref']}/{row['n_naive']} | {row['pref_raw_med']:.4g} | "
                         f"{row['naive_raw_med']:.4g} | {row['cohen_d']:.2f} | {row['auroc']:.2f} | {row['sr_pref']:.2f} | "
                         f"{row['sat_pref_mean']:.0f} / {row['sat_pref_min']:.0f} / {100*row['sat_pref_ge70']:.0f}% | {row['sa_pref']:.0f} | "
                         f"{row['sat_naive_mean']:.0f} / {row['sat_naive_max']:.0f} / {100*row['sat_naive_le30']:.0f}% |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write("# Motion metric scoring: pref vs naive demos (held-out halves)\n\n" + text + "\n")
    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

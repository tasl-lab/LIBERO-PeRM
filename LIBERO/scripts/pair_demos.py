#!/usr/bin/env python
"""Write the init-paired 10 + 10 demo files for one mission.

The no-preference run replays the with-preference demos' initial states one by one
(`main.py --gen_naive_anchor --pair_init_from`), so its log tells which inits it solved.
Keep the first N inits where the no-preference run succeeded and, for metrics with a
discrete criterion, the with-preference demo meets it; copy both sides in the same order:

    VLA_data/SinglePref_paired/Mission_<id>.hdf5        demo_i  <->  demo_i
    VLA_data/SinglePref_naive_paired/Mission_<id>.hdf5

Usage:  python scripts/pair_demos.py <mission_id> <naive_log> [--n 10]
"""
import argparse, json, os, re, sys
import h5py, yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, LIBERO); sys.path.insert(0, HERE)
V = os.path.join(os.path.dirname(LIBERO), "VLA_data")


def copy_subset(src, dst, keys):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mission"); ap.add_argument("naive_log"); ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    import calibrate_low_pref as cal
    mid = a.mission
    txt = open(a.naive_log, errors="ignore").read()
    succ = sorted({int(m) for m in re.findall(r"HDF5 saved: \S+/HDF5/rollout_(\d+)\.hdf5", txt)})
    P, N = f"{V}/SinglePref/Mission_{mid}.hdf5", f"{V}/SinglePref_naive/Mission_{mid}.hdf5"
    with h5py.File(N, "r") as f:
        nk = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    assert len(nk) == len(succ), f"no-pref demos {len(nk)} != success ids {len(succ)} in the log"
    y = yaml.safe_load(open(cal._mission_info_path(mid))) or {}
    prefs = cal._build_prefs(y.get("low_prefs") or [])
    out = cal._raw_lists_for_dir(prefs, P, max(succ) + 1 if succ else 0)
    bad = set()
    for p in prefs:
        if getattr(p, "criterion_threshold", None) is None:
            continue
        sats = out.get(p.preference_type + "__sat", [])
        bad |= {i for i, v in enumerate(sats) if v != 100.0}
    ids = [i for i in succ if i not in bad]
    res = dict(mission=mid, naive_success=len(succ), naive_attempts=len(re.findall(r"paired-init\] batch", txt)),
               pref_fail_criterion=sorted(bad & set(succ)), usable=len(ids))
    if len(ids) >= a.n:
        sel = ids[:a.n]
        os.makedirs(f"{V}/SinglePref_paired", exist_ok=True); os.makedirs(f"{V}/SinglePref_naive_paired", exist_ok=True)
        copy_subset(P, f"{V}/SinglePref_paired/Mission_{mid}.hdf5", [f"demo_{i}" for i in sel])
        copy_subset(N, f"{V}/SinglePref_naive_paired/Mission_{mid}.hdf5", [nk[succ.index(i)] for i in sel])
        res.update(ok=True, inits=sel)
    else:
        res.update(ok=False)
    print("PAIR " + json.dumps(res))
    sys.exit(0 if res["ok"] else 2)

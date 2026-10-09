"""Sweep task_plan_progress R1/R2 across all Series 3 missions in VLA_data.

For each mission, replay GT-pref demos through the metric twice:
  - plan_key=task_plan  → R1 anchor (should be ~1.0)
  - plan_key=naive_plan → R2 contrast (should be < 1.0)

Reports per-mission table + aggregate Cohen's d.

Run:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python -m Preference.sweep_task_plan_progress --n_demos 10
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob
from typing import Dict, List, Tuple

import h5py
import numpy as np
import yaml
from robosuite.utils.binding_utils import MjSim, MjSimState

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LIBERO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_ROOT)

from Preference.eval_task_plan_progress_on_demos import (  # noqa: E402
    _build_obj_body_id,
    _wrap_env,
    _load_metric,
    _replay_demo,
)


VLA_DATA_DIR = "/data4/zhixu/SinglePref"
DATASET_DIR = "/data3/zhixu/LIBERO-PeRM/LIBERO/dataset"


def _mission_info_path(mission_id: str) -> str:
    """3-1_004_1 → series_3/perm_3_1/libero_004_1/mission_info.yaml"""
    m = re.match(r"^(\d+)-(\d+)_(.+)$", mission_id)
    if not m:
        raise ValueError(mission_id)
    series, sub, lib = m.group(1), m.group(2), m.group(3)
    return os.path.join(
        DATASET_DIR, f"series_{series}", f"perm_{series}_{sub}", f"libero_{lib}", "mission_info.yaml"
    )


def _eval_mission(hdf5_path: str, info_path: str, plan_key: str, n_demos: int) -> Dict:
    metric = _load_metric(info_path, plan_key=plan_key)
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())[:n_demos]
        xml = f[f"data/{demos[0]}"].attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8")
        sim = MjSim.from_xml_string(xml)
        obj_body_id = _build_obj_body_id(sim, list(metric.expected_set))
        env = _wrap_env(sim, obj_body_id)
        scores: List[float] = []
        sats: List[float] = []
        mapped: List[float] = []
        spurious = 0
        for demo_key in demos:
            states = np.asarray(f[f"data/{demo_key}/states"][:])
            try:
                r = _replay_demo(sim, states, env, metric)
            except Exception:
                continue
            scores.append(r["score"])
            sats.append(float(metric.satisfaction(r["score"])))
            mapped.append(float(metric.mapped_satisfaction(r["score"])))
            if r["raw"]["spurious_first_obj"]:
                spurious += 1
    arr = np.asarray(scores) if scores else np.array([0.0])
    return {
        "n": len(scores),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "sat": float(np.mean(sats)) if sats else 0.0,
        "mapped": float(np.mean(mapped)) if mapped else 0.0,
        "spurious": spurious,
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math_pooled_std(a, b)
    return float((a.mean() - b.mean()) / s) if s > 1e-9 else float("inf")


def math_pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va = a.var(ddof=1) if na > 1 else 0.0
    vb = b.var(ddof=1) if nb > 1 else 0.0
    return float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_demos", type=int, default=10)
    ap.add_argument("--prefix", default="Mission_3-", help="HDF5 filename prefix to sweep")
    args = ap.parse_args()

    paths = sorted(glob(os.path.join(VLA_DATA_DIR, f"{args.prefix}*.hdf5")))
    print(f"Found {len(paths)} mission(s) under prefix '{args.prefix}'")

    rows: List[Tuple[str, Dict, Dict]] = []
    pref_scores_all: List[float] = []
    naive_scores_all: List[float] = []

    for p in paths:
        mid = os.path.basename(p).replace("Mission_", "").replace(".hdf5", "")
        info = _mission_info_path(mid)
        if not os.path.exists(info):
            print(f"  [skip] {mid}  (no mission_info.yaml)")
            continue
        try:
            r_pref = _eval_mission(p, info, "task_plan", args.n_demos)
            r_naive = _eval_mission(p, info, "naive_plan", args.n_demos)
        except Exception as e:
            print(f"  [err]  {mid}  {e}")
            continue
        rows.append((mid, r_pref, r_naive))
        # Sample-level for Cohen's d (use mean per demo as a proxy: just append mean repeatedly).
        pref_scores_all.extend([r_pref["mean"]] * r_pref["n"])
        naive_scores_all.extend([r_naive["mean"]] * r_naive["n"])

    print(f"\n{'mission':<14} {'R1 score':>9} {'sat':>6} {'mapped':>7}  | {'R2 score':>9} {'sat':>6} {'mapped':>7}")
    print("-" * 78)
    pref_means: List[float] = []
    naive_means: List[float] = []
    for mid, p, n in rows:
        pref_means.append(p["mean"])
        naive_means.append(n["mean"])
        print(f"{mid:<14} {p['mean']:>9.3f} {p['sat']:>6.1f} {p['mapped']:>7.2f}  | {n['mean']:>9.3f} {n['sat']:>6.1f} {n['mapped']:>7.2f}")
    pa = np.asarray(pref_means)
    na = np.asarray(naive_means)
    print("-" * 78)
    print(f"{'AGGREGATE':<14} {pa.mean():>8.3f} {pa.std():>8.3f}          | {na.mean():>8.3f} {na.std():>8.3f} {(pa.mean()-na.mean()):>7.3f}")
    d = _cohens_d(pa, na)
    print(f"\nMission-level Cohen's d (R1 vs R2): {d:.2f}  (target ≥ 0.5)")
    print(f"R1 missions with mean ≥ 0.95 : {int((pa >= 0.95).sum())}/{len(pa)}")
    print(f"R2 missions with mean ≤ 0.80 : {int((na <= 0.80).sum())}/{len(na)}")


if __name__ == "__main__":
    main()

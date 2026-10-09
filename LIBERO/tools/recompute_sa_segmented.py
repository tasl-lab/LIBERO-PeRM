#!/usr/bin/env python3
"""Offline recompute Series-1 SA with subtask-aware metric scopes.

Phases derived from each hdf5 trajectory:
  pre_pick_moveto  = [0, gripper_close_step)
  post_pick_moveto = (gripper_close_step, first_sustained_descent_after_close)
  all_moveto       = pre_pick_moveto ∪ post_pick_moveto

Metric quantities (per trajectory):
  speed_range      : total step count                      (global)
  max_acc          : peak ||Δv|| in ee_pos                 (global)
  rotation         : total Δθ summed over phase            (pre_pick_moveto)
  straight_path    : path_len / displacement               (post_pick_moveto)
  height_offset    : peak z relative to first-frame z      (post_pick_moveto)
  avoid_to_object  : NOT computed offline (needs obj poses)

SA = clip(quantity / demo_mean_quantity, 0, 1) * 100
For metrics where "less = better" (speed_range under speed_up),
SA = clip(demo_mean / actual, 0, 1) * 100.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import yaml


# Mission → metric (from mission_info.yaml low_prefs[0].type)
MISSION_METRIC = {
    "1-1_004": ("speed_range", "less"),
    "1-1_006": ("speed_range", "less"),
    "1-2_113": ("speed_range", "more"),  # Slow down
    "1-3_107": ("max_acc", "less"),       # Avoid sudden acceleration
    "1-3_109": ("max_acc", "less"),
    "1-4_034": ("avoid_to_object", "more"),
    "1-5_114": ("height_offset", "more"),
    "1-6_105": ("straight_path", "more"),
    "1-7_104": ("height_offset", "more"),
    "1-8_114": ("rotation", "more"),
}

# 1-2 description "Slow down" → direction=more (more steps = better).
# 1-1 "Speed up" → direction=less (fewer steps = better).


# ─── phase detection ──────────────────────────────────────────────

def detect_phases(grip: np.ndarray, z: np.ndarray, dz_thresh: float = 1e-3, k: int = 3) -> Dict[str, Tuple[int, int]]:
    """Return phase ranges as (start_inclusive, end_exclusive)."""
    T = len(grip)
    # gripper_close: first step where gripper sum drops below baseline by margin.
    # gripper_states is finger separation; closing → smaller value.
    # Use median of first 10 steps as "open baseline".
    base = float(np.median(grip[:max(10, T // 20)]))
    close_idx = None
    margin = 0.3 * base if base > 1e-3 else 0.005
    for t in range(T):
        if grip[t] < base - margin:
            close_idx = t
            break
    if close_idx is None or close_idx >= T - 1:
        # Fallback: close at midpoint
        close_idx = T // 2

    # First sustained descent after close: k consecutive Δz < -dz_thresh.
    descent_idx = None
    for t in range(close_idx + 1, T - k):
        if all((z[t + i] - z[t + i - 1]) < -dz_thresh for i in range(k)):
            descent_idx = t
            break
    if descent_idx is None:
        descent_idx = T

    return {
        "pre_pick_moveto": (0, close_idx),
        "post_pick_moveto": (close_idx, descent_idx),
        "all_moveto": (0, descent_idx),
    }


# ─── per-metric quantity ──────────────────────────────────────────

def q_speed_range(ee_pos, ee_ori, grip, ph):
    return float(len(ee_pos))  # total step count


def q_max_acc(ee_pos, ee_ori, grip, ph):
    if len(ee_pos) < 3:
        return 0.0
    v = np.diff(ee_pos, axis=0)
    a = np.diff(v, axis=0)
    return float(np.max(np.linalg.norm(a, axis=1)))


def q_rotation(ee_pos, ee_ori, grip, ph):
    s, e = ph["pre_pick_moveto"]
    if e - s < 2:
        return 0.0
    o = ee_ori[s:e]
    # Sum |Δaxis-angle| step-to-step. ee_ori is (T,3) Euler-like or axis-angle.
    # Use L2 of differences as a proxy for angular travel.
    d = np.diff(o, axis=0)
    # wrap to [-pi, pi] per axis
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.linalg.norm(d, axis=1)))


def q_straight_path(ee_pos, ee_ori, grip, ph):
    s, e = ph["post_pick_moveto"]
    if e - s < 2:
        return 0.0
    seg = ee_pos[s:e]
    diffs = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    path = float(np.sum(diffs))
    disp = float(np.linalg.norm(seg[-1] - seg[0]))
    if disp < 1e-4:
        return 1.0
    # Lower path/disp = better (closer to 1 = straight). Return inefficiency = path/disp.
    return path / disp


def q_height_offset(ee_pos, ee_ori, grip, ph):
    s, e = ph["post_pick_moveto"]
    if e - s < 1:
        return 0.0
    seg_z = ee_pos[s:e, 2]
    base_z = float(ee_pos[0, 2])
    return float(np.max(seg_z) - base_z)


METRIC_FN = {
    "speed_range": q_speed_range,
    "max_acc": q_max_acc,
    "rotation": q_rotation,
    "straight_path": q_straight_path,
    "height_offset": q_height_offset,
}


def compute_quantity(metric: str, hdf5_grp) -> float:
    ee_pos = hdf5_grp["obs/ee_pos"][:]
    ee_ori = hdf5_grp["obs/ee_ori"][:]
    grip = hdf5_grp["obs/gripper_states"][:]
    grip_scalar = grip[:, 0] if grip.ndim == 2 else grip
    z = ee_pos[:, 2]
    ph = detect_phases(grip_scalar, z)
    fn = METRIC_FN.get(metric)
    if fn is None:
        return float("nan")
    return fn(ee_pos, ee_ori, grip_scalar, ph)


# ─── normalize to SA ──────────────────────────────────────────────

def to_sa(q: float, anchor: float, direction: str, metric: str) -> float:
    """SA in [0, 100]. direction: 'less'=actual≤anchor good, 'more'=actual≥anchor good."""
    if anchor <= 0 or not np.isfinite(q):
        return 0.0
    if metric == "straight_path":
        # Lower inefficiency (path/disp closer to 1) = better. Treat as 'less'.
        # SA = clip(anchor / q, 0, 1) * 100 where anchor = demo path/disp.
        return float(min(max(anchor / max(q, 1e-6), 0.0), 1.0) * 100.0)
    if direction == "less":
        return float(min(max(anchor / max(q, 1e-6), 0.0), 1.0) * 100.0)
    # 'more': larger q = better
    return float(min(max(q / anchor, 0.0), 1.0) * 100.0)


# ─── main ─────────────────────────────────────────────────────────

def scan_dir(root: Path, mission_filter: List[str]) -> Dict[str, List[float]]:
    """Return {mission_id: [quantity per traj]}."""
    out: Dict[str, List[float]] = {}
    for mid in mission_filter:
        metric, _ = MISSION_METRIC[mid]
        hpath = root / f"Mission_{mid}.hdf5" if root.name == "SinglePref" else None
        # Demo hdf5 layout: data/demo_<i>/...
        # Rollout hdf5 layout (per mission dir): data/demo_0/... but we have one per ep.
        if hpath and hpath.exists():
            with h5py.File(hpath, "r") as f:
                qs = []
                for d in f["data"].keys():
                    if not d.startswith("demo_"):
                        continue
                    try:
                        qs.append(compute_quantity(metric, f[f"data/{d}"]))
                    except Exception:
                        pass
                out[mid] = qs
        else:
            # rollout root: <root>/<mid>/HDF5/Ep<N>_*.hdf5
            mdir = root / mid
            qs = []
            if mdir.exists():
                for h5f in sorted((mdir / "HDF5").glob("Ep*.hdf5")):
                    try:
                        with h5py.File(h5f, "r") as f:
                            qs.append(compute_quantity(metric, f["data/demo_0"]))
                    except Exception:
                        pass
            out[mid] = qs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root", default="/data4/zhixu/SinglePref")
    ap.add_argument("--rollout-root",
                    default="/data3/zhixu/LIBERO-PeRM/Policy_output/lora_ft/single-pref/U_pi05_singpref_step45k_full_2trial_skill/pi05_singpref_run1_step44999/pref")
    args = ap.parse_args()

    missions = list(MISSION_METRIC.keys())
    demo_q = scan_dir(Path(args.demo_root), missions)
    roll_q = scan_dir(Path(args.rollout_root), missions)

    print(f"{'mission':12} {'metric':18} {'demo_n':>7} {'demo_q':>10} {'roll_n':>7} {'roll_q':>10} {'demo_SA':>8} {'roll_SA':>8}")
    for mid in missions:
        metric, direction = MISSION_METRIC[mid]
        dq = demo_q.get(mid, [])
        rq = roll_q.get(mid, [])
        if not dq or not rq:
            print(f"{mid:12} {metric:18} {len(dq):>7} {'-':>10} {len(rq):>7} {'-':>10} {'-':>8} {'-':>8}")
            continue
        dq_mean = float(np.mean(dq))
        rq_mean = float(np.mean(rq))
        # demo SA: should be 100 by construction (anchor = demo mean → ratio = 1)
        demo_sas = [to_sa(q, dq_mean, direction, metric) for q in dq]
        roll_sas = [to_sa(q, dq_mean, direction, metric) for q in rq]
        print(f"{mid:12} {metric:18} {len(dq):>7} {dq_mean:>10.3f} {len(rq):>7} {rq_mean:>10.3f} {np.mean(demo_sas):>8.1f} {np.mean(roll_sas):>8.1f}")


if __name__ == "__main__":
    main()

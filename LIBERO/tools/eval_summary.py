#!/usr/bin/env python3
"""Post-eval SA aggregator.

Auto-invoked at end of run_client.sh. Computes per-mission:
  SR  (from rollout_summary.yaml)
  raw SA (success-only):
    series 1 → segmented low-pref metric quantity vs demo anchor
    series 2 → profile_pref_reward.task_plan_progress.satisfaction (placeholder)
    series 3 → profile_pref_reward.task_plan_progress.satisfaction (real profile pref)

Per-series + grand-total:
  SR mean, raw SA mean (missing missions skipped), final SA = SR × raw_SA.

Writes xlsx to <exp_dir>/<server_name>/evalresult.xlsx (or _1/_2/... if exists).
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import yaml

# ─── Series-1 segmented recompute (replicated from recompute_sa_segmented.py) ─

WINDOW_HALF = 24

# missions that have demo hdf5 + which low-pref metric type they use.
# Detected on the fly from mission_info.yaml; keyed by base mission_id (1-N_LLL).
def load_low_pref_type(mission_info_yaml: Path) -> str | None:
    if not mission_info_yaml.exists():
        return None
    with open(mission_info_yaml) as f:
        d = yaml.safe_load(f) or {}
    lp = d.get("low_prefs") or []
    if not lp:
        return None
    return lp[0].get("type")


def detect_phases(grip: np.ndarray, z: np.ndarray, dz_thresh=1e-3, k=3):
    base = float(np.median(grip[: max(10, len(grip) // 20)]))
    margin = 0.3 * base if base > 1e-3 else 0.005
    close_idx = next((t for t in range(len(grip)) if grip[t] < base - margin), len(grip) // 2)
    descent_idx = next(
        (t for t in range(close_idx + 1, len(z) - k)
         if all((z[t + i] - z[t + i - 1]) < -dz_thresh for i in range(k))),
        len(z),
    )
    return {
        "pre_pick_moveto": (0, close_idx),
        "post_pick_moveto": (close_idx, descent_idx),
    }


IMPLEMENTED_LOW_METRICS = {"speed_range", "max_acc", "rotation", "straight_path", "height_offset"}


def quantity_for_metric(metric: str, ee_pos, ee_ori, ph) -> float:
    if metric == "speed_range":
        return float(len(ee_pos))
    if metric == "max_acc":
        if len(ee_pos) < 3:
            return 0.0
        v = np.diff(ee_pos, axis=0)
        a = np.diff(v, axis=0)
        return float(np.max(np.linalg.norm(a, axis=1)))
    if metric == "rotation":
        s, e = ph["pre_pick_moveto"]
        if e - s < 2:
            return 0.0
        d = np.diff(ee_ori[s:e], axis=0)
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return float(np.sum(np.linalg.norm(d, axis=1)))
    if metric == "straight_path":
        s, e = ph["post_pick_moveto"]
        if e - s < 2:
            return 0.0
        seg = ee_pos[s:e]
        diffs = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        path = float(np.sum(diffs))
        disp = float(np.linalg.norm(seg[-1] - seg[0]))
        if disp < 1e-4:
            return 1.0
        return path / disp
    if metric == "height_offset":
        s, e = ph["post_pick_moveto"]
        if e - s < 1:
            return 0.0
        return float(np.max(ee_pos[s:e, 2]) - ee_pos[0, 2])
    return float("nan")


def compute_traj_quantity(metric: str, hdf5_grp) -> float:
    ee_pos = hdf5_grp["obs/ee_pos"][:]
    ee_ori = hdf5_grp["obs/ee_ori"][:]
    grip_states = hdf5_grp["obs/gripper_states"][:]
    grip = grip_states[:, 0] if grip_states.ndim == 2 else grip_states
    z = ee_pos[:, 2]
    ph = detect_phases(grip, z)
    return quantity_for_metric(metric, ee_pos, ee_ori, ph)


def to_sa(q: float, anchor: float, metric: str, direction: str) -> float:
    """SA in [0, 100]. direction: 'less' means actual≤anchor=better."""
    if anchor <= 0 or not np.isfinite(q):
        return 0.0
    if metric in ("straight_path", "speed_range") and direction == "less":
        return float(min(max(anchor / max(q, 1e-6), 0.0), 1.0) * 100.0)
    if direction == "less":
        return float(min(max(anchor / max(q, 1e-6), 0.0), 1.0) * 100.0)
    return float(min(max(q / anchor, 0.0), 1.0) * 100.0)


# Direction map: how to interpret quantity vs demo.
# 'less' = smaller quantity is better; 'more' = larger is better.
DIRECTION_BY_DESC = {
    "Speed up": "less",       # fewer steps
    "Slow down": "more",      # more steps
    "Avoid sudden acceleration": "less",
    "Move straight": "less",  # path/disp closer to 1
    "Lift higher": "more",
    "Move close to ground": "less",
    "Rotate when moving": "more",
}


def direction_for(low_pref_desc: str) -> str:
    return DIRECTION_BY_DESC.get(low_pref_desc.strip(), "more")


# ─── demo anchor cache ───────────────────────────────────────────────────
# Search multiple roots so SinglePref + PrefScale (Series 5) demos both work.
DEMO_ROOTS = (
    Path("/data4/zhixu/SinglePref"),
    Path("/data4/zhixu/VLA_data/PrefScale"),
)


def demo_anchor(mission_id_no_pref: str, metric: str) -> float | None:
    """Mean quantity over demos. Strips '_1','_2','_nopref' suffixes when looking up demo file."""
    candidates = [mission_id_no_pref]
    # series 3 / scale tiers: 1-1_004_1 → try 1-1_004 too
    if "_" in mission_id_no_pref:
        base = "_".join(mission_id_no_pref.split("_")[:-1])
        if base != mission_id_no_pref:
            candidates.append(base)
    paths = [root / f"Mission_{c}.hdf5" for root in DEMO_ROOTS for c in candidates]
    for path in paths:
        if path.exists():
            try:
                with h5py.File(path, "r") as f:
                    qs = []
                    for d in f["data"].keys():
                        if not d.startswith("demo_"):
                            continue
                        try:
                            qs.append(compute_traj_quantity(metric, f[f"data/{d}"]))
                        except Exception:
                            pass
                    if qs:
                        return float(np.mean(qs))
            except Exception:
                pass
    return None


def rollout_quantity(mdir: Path, metric: str, success_only: bool) -> List[float]:
    """Return per-ep quantities from rollout HDF5 dir."""
    qs = []
    h5dir = mdir / "HDF5"
    if not h5dir.exists():
        return qs
    pat = "*_success.hdf5" if success_only else "*.hdf5"
    for h5f in sorted(h5dir.glob(pat)):
        if not success_only and "fail" in str(h5f):
            continue
        try:
            with h5py.File(h5f, "r") as f:
                qs.append(compute_traj_quantity(metric, f["data/demo_0"]))
        except Exception:
            pass
    return qs


# ─── main aggregation ────────────────────────────────────────────────────

DATASET_ROOT = Path("/data3/zhixu/LIBERO-PeRM/LIBERO/dataset")


def _series_dir_for(mid: str) -> Path | None:
    """Map mission_id → series_X/perm_X_Y/libero_ZZZ[_suffix] dir."""
    if "-" not in mid:
        return None
    series, rest = mid.split("-", 1)
    sub_idx, libero_part = rest.split("_", 1)
    perm = DATASET_ROOT / f"series_{series}" / f"perm_{series}_{sub_idx}"
    cand = perm / f"libero_{libero_part}"
    if cand.exists():
        return cand
    return None


def per_mission_row(mdir: Path) -> Dict | None:
    sf = mdir / "rollout_summary.yaml"
    if not sf.exists():
        return None
    d = yaml.safe_load(sf.read_text())
    mid = mdir.name
    series = mid.split("-")[0]
    sr = float(d.get("sr", 0.0))
    eps = d.get("episodes", [])
    n_total = len(eps)
    n_succ = sum(1 for e in eps if e.get("success"))

    raw_sa = None
    metric_used = ""
    note = ""

    if series == "1":
        sd = _series_dir_for(mid)
        metric = load_low_pref_type(sd / "mission_info.yaml") if sd else None
        if metric is None:
            note = "no low_prefs"
        elif metric not in IMPLEMENTED_LOW_METRICS:
            metric_used = metric
            note = "metric not implemented"
        else:
            metric_used = metric
            anchor = demo_anchor(mid, metric)
            if anchor is None:
                note = "demo missing"
            else:
                # direction
                with open(sd / "mission_info.yaml") as f:
                    desc = ((yaml.safe_load(f) or {}).get("low_prefs") or [{}])[0].get("description", "")
                direction = direction_for(desc)
                qs = rollout_quantity(mdir, metric, success_only=True)
                if qs:
                    raw_sa = float(np.mean([to_sa(q, anchor, metric, direction) for q in qs]))
                else:
                    note = "no succ ep"
    else:
        # series 2/3: prefer task_plan_compliance.satisfaction, fall back to the
        # legacy task_plan_progress key (older rollout summaries). success-only.
        metric_used = "task_plan_compliance"
        sas = [
            (
                (e.get("profile_pref_reward") or {}).get("task_plan_compliance")
                or (e.get("profile_pref_reward") or {}).get("task_plan_progress", {})
            ).get("satisfaction")
            for e in eps if e.get("success")
        ]
        sas = [v for v in sas if v is not None]
        if sas:
            raw_sa = float(np.mean(sas))
        else:
            note = "no succ ep"

    return {
        "mission_id": mid,
        "series": series,
        "tier": {"1": "low", "2": "skill", "3": "profile"}.get(series, "?"),
        "metric": metric_used,
        "sr": sr,
        "raw_sa": raw_sa,
        "n_succ": n_succ,
        "n_total": n_total,
        "note": note,
    }


def find_rollout_root(runfile: Path) -> Path:
    cfg = yaml.safe_load(runfile.read_text())
    out = Path("/data3/zhixu/LIBERO-PeRM/Policy_output") / cfg["exp_type"] / cfg["task_suite"] / cfg["experiment_name"] / cfg["server"]["name"]
    return out


def next_xlsx_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    parent = base.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runfile", help="path to runfile yaml")
    ap.add_argument("--rollout-root", help="override rollout root")
    ap.add_argument("--xlsx", help="override output xlsx path")
    args = ap.parse_args()

    if args.rollout_root:
        rollout_root = Path(args.rollout_root)
    elif args.runfile:
        rollout_root = find_rollout_root(Path(args.runfile))
    else:
        sys.exit("--runfile or --rollout-root required")

    print(f"[eval_summary] rollout root: {rollout_root}")

    rows = []
    for cond_dir in sorted(rollout_root.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name not in ("base", "pref"):
            continue
        for mdir in sorted(cond_dir.iterdir()):
            if not mdir.is_dir():
                continue
            row = per_mission_row(mdir)
            if row is None:
                continue
            row["condition"] = cond_dir.name
            rows.append(row)

    if not rows:
        print("[eval_summary] no missions found")
        return

    df = pd.DataFrame(rows).sort_values(["condition", "series", "mission_id"]).reset_index(drop=True)

    # raw_sa → mapped_sa (3-bin step):
    #   [0,50) → 25,  [50,90) → 75,  [90,100] → 100
    def _map_sa(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if v >= 90:
            return 100.0
        if v >= 50:
            return 75.0
        return 25.0
    df["mapped_sa"] = df["raw_sa"].apply(_map_sa)

    # rollups: per (condition, series) and grand
    rollups = []
    for (cond, ser), g in df.groupby(["condition", "series"]):
        sr_mean = g["sr"].mean()
        sa_valid = g["raw_sa"].dropna()
        raw_sa_mean = float(sa_valid.mean()) if len(sa_valid) else None
        mapped_valid = g["mapped_sa"].dropna()
        mapped_sa_mean = float(mapped_valid.mean()) if len(mapped_valid) else None
        final_sa = (sr_mean * raw_sa_mean) if raw_sa_mean is not None else None
        final_mapped_sa = (sr_mean * mapped_sa_mean) if mapped_sa_mean is not None else None
        rollups.append({
            "condition": cond, "series": ser, "n": len(g), "n_sa_valid": len(sa_valid),
            "sr": sr_mean, "raw_sa": raw_sa_mean, "mapped_sa": mapped_sa_mean,
            "final_sa": final_sa, "final_mapped_sa": final_mapped_sa,
        })
    for cond, g in df.groupby("condition"):
        sr_mean = g["sr"].mean()
        sa_valid = g["raw_sa"].dropna()
        raw_sa_mean = float(sa_valid.mean()) if len(sa_valid) else None
        mapped_valid = g["mapped_sa"].dropna()
        mapped_sa_mean = float(mapped_valid.mean()) if len(mapped_valid) else None
        final_sa = (sr_mean * raw_sa_mean) if raw_sa_mean is not None else None
        final_mapped_sa = (sr_mean * mapped_sa_mean) if mapped_sa_mean is not None else None
        rollups.append({
            "condition": cond, "series": "ALL", "n": len(g), "n_sa_valid": len(sa_valid),
            "sr": sr_mean, "raw_sa": raw_sa_mean, "mapped_sa": mapped_sa_mean,
            "final_sa": final_sa, "final_mapped_sa": final_mapped_sa,
        })
    rdf = pd.DataFrame(rollups)

    out_xlsx = Path(args.xlsx) if args.xlsx else (rollout_root / "evalresult.xlsx")
    out_xlsx = next_xlsx_path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="per_mission")
        rdf.to_excel(w, index=False, sheet_name="rollup")

    print(f"[eval_summary] wrote {out_xlsx}")
    print()
    print(rdf.to_string(index=False))


if __name__ == "__main__":
    main()

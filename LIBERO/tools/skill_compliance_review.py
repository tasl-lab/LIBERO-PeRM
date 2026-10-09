#!/usr/bin/env python3
"""Skill-compliance frame extractor for series-2 / 3 missions.

For each rollout HDF5:
  1. Find trigger frame t* = min(first_touch_target, gripper_open)
     - Approximated as gripper_close index for now (robust signal in hdf5).
     - Refinement: also detect gripper_open after close as candidate.
  2. Extract [t*-24, t*+24) (48 frames), downsample 4:1 → 12 frames
  3. Tile into 3 PNG mosaics (2x2 each)
  4. Write to <rollout_dir>/skill_review/<mission>_Ep<N>_panel{0,1,2}.png
"""
import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from PIL import Image


WINDOW_HALF = 24
DOWNSAMPLE = 4
TILE_GRID = (2, 2)
PANEL_FRAMES = TILE_GRID[0] * TILE_GRID[1]


def gripper_event_indices(grip: np.ndarray) -> Tuple[int, int]:
    """Return (close_idx, open_idx) using finger separation.

    grip: (T, 2). Open ≈ [+0.04, -0.04] → separation ≈ 0.08.
    Close → separation drops near 0.
    """
    sep = grip[:, 0] - grip[:, 1]
    base = float(np.median(sep[: max(10, len(sep) // 20)]))
    thresh_close = 0.5 * base
    thresh_open = 0.85 * base
    T = len(sep)
    close_idx = next((t for t in range(T) if sep[t] < thresh_close), T // 2)
    open_idx = next((t for t in range(close_idx + 1, T) if sep[t] > thresh_open), T)
    return close_idx, open_idx


def trigger_frame(grip: np.ndarray) -> int:
    close_idx, open_idx = gripper_event_indices(grip)
    return min(close_idx, open_idx)


def extract_panels(rgb: np.ndarray, t_star: int, out_dir: Path, prefix: str) -> int:
    T = len(rgb)
    s = max(0, t_star - WINDOW_HALF)
    e = min(T, t_star + WINDOW_HALF)
    window = rgb[s:e]
    # Resample to exactly 48 frames if window short
    if len(window) < 2 * WINDOW_HALF:
        idx = np.linspace(0, len(window) - 1, 2 * WINDOW_HALF).astype(int)
        window = window[idx]
    sampled = window[::DOWNSAMPLE][: 12]  # 12 frames
    # 3 panels × 4 frames each
    out_dir.mkdir(parents=True, exist_ok=True)
    n_panels = 0
    for p in range(3):
        chunk = sampled[p * PANEL_FRAMES : (p + 1) * PANEL_FRAMES]
        if len(chunk) < PANEL_FRAMES:
            break
        H, W = chunk[0].shape[:2]
        canvas = np.zeros((H * TILE_GRID[0], W * TILE_GRID[1], 3), dtype=np.uint8)
        for k, img in enumerate(chunk):
            r, c = divmod(k, TILE_GRID[1])
            canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
        Image.fromarray(canvas).save(out_dir / f"{prefix}_panel{p}.png")
        n_panels += 1
    return n_panels


def process_mission(mission_dir: Path, out_root: Path) -> dict:
    h5_dir = mission_dir / "HDF5"
    if not h5_dir.exists():
        return {}
    out = {}
    for h5f in sorted(h5_dir.glob("Ep*.hdf5")):
        ep_name = h5f.stem  # e.g. Ep0_success
        prefix = f"{mission_dir.name}_{ep_name}"
        try:
            with h5py.File(h5f, "r") as f:
                grip = f["data/demo_0/obs/gripper_states"][:]
                rgb = f["data/demo_0/obs/agentview_rgb"][:]
            t_star = trigger_frame(grip)
            n = extract_panels(rgb, t_star, out_root / mission_dir.name, prefix)
            out[ep_name] = {"t_star": int(t_star), "panels": n}
        except Exception as e:
            out[ep_name] = {"error": str(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-root",
                    default="/data3/zhixu/LIBERO-PeRM/Policy_output/lora_ft/single-pref/U_pi05_singpref_step45k_full_2trial_skill/pi05_singpref_run1_step44999/pref")
    ap.add_argument("--out-root", default=None,
                    help="default = <rollout-root>/../skill_review")
    ap.add_argument("--missions", nargs="*", default=None,
                    help="specific mission ids; default = all 2-* missions")
    args = ap.parse_args()

    rollout_root = Path(args.rollout_root)
    out_root = Path(args.out_root) if args.out_root else rollout_root.parent / "skill_review"
    out_root.mkdir(parents=True, exist_ok=True)

    mlist = args.missions or sorted([d.name for d in rollout_root.iterdir() if d.is_dir() and d.name.startswith("2-")])
    print(f"Processing {len(mlist)} missions → {out_root}")
    for mid in mlist:
        info = process_mission(rollout_root / mid, out_root)
        for ep, res in info.items():
            print(f"  {mid}/{ep}: {res}")


if __name__ == "__main__":
    main()

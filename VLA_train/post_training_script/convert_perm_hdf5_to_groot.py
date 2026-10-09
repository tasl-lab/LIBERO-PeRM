"""Convert P_Bench SinglePref HDF5 data to GR00T-flavored LeRobot v2 format.

Input:  /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/Mission_*.hdf5
Output: GR00T LeRobot v2 dataset with parquet + mp4 + meta/

Usage:
    python convert_perm_hdf5_to_groot.py \
        --hdf5_dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref \
        --output_dir /data3/zhixu/LIBERO-PeRM/VLA_data/groot_lerobot/perm_singlepref \
        --dataset_root /home/vla-reasoning/proj/P_Bench/LIBERO/dataset \
        --max_demos_per_mission 50

The script:
1. Parses mission_id from each HDF5 filename (e.g. Mission_1-1_004 → perm_1_1/libero_004)
2. Reads mission_info.yaml to compose pref language instruction
3. Converts each demo to a parquet episode + two MP4 video streams
4. Writes meta/ files (info.json, tasks.jsonl, episodes.jsonl, modality.json)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching GROOT LIBERO_PANDA config
# ─────────────────────────────────────────────────────────────────────────────

FPS = 10
CHUNK_SIZE = 1000
VIDEO_CODEC = "mp4v"

# LIBERO_PANDA modality.json (from examples/LIBERO/modality.json)
MODALITY_JSON = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Mission ID parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_mission_id(hdf5_name: str) -> str:
    """'Mission_1-1_004.hdf5' → '1-1_004'"""
    m = re.match(r"^Mission_(.+)\.hdf5$", hdf5_name)
    if not m:
        raise ValueError(f"Cannot parse mission_id from {hdf5_name}")
    return m.group(1)


def mission_id_to_dir(mission_id: str) -> str:
    """'1-1_004' → 'perm_1_1/libero_004'"""
    m = re.match(r"^(\d+-\d+)_(.+)$", mission_id)
    if not m:
        raise ValueError(f"Invalid mission_id: {mission_id}")
    return f"perm_{m.group(1).replace('-', '_')}/libero_{m.group(2)}"


def compose_pref_language(mission_info: dict) -> str:
    """Compose preference language: task + high_pref + low_pref."""
    inst = (mission_info.get("language_instructions") or "").strip()
    high = (mission_info.get("high_pref_lan") or "").strip()
    low = (mission_info.get("low_pref_lan") or "").strip()
    parts = [p for p in (inst, high, low) if p]
    return " ".join(p if p.endswith(".") else p + "." for p in parts)


# ─────────────────────────────────────────────────────────────────────────────
# No-op filter (ported from OpenVLA's regenerate_libero_dataset.py)
#
# Filters out frames where BOTH (1) ee pose action norm < threshold AND
# (2) gripper action unchanged from previous step. Gripper-flip frames
# are preserved even if ee is stationary — critical for keeping grasp/release
# event signals in the training distribution.
#
# Rationale: GR00T N1.6 uses `use_relative_action=True`; heuristic demos with
# many idle frames cause ~89% of 16-step chunks to have Δaction = 0 on the
# gripper axis, so the diffusion head learns to output constant 0 (gripper
# never actuates). See docs/groot_n16_gripper_collapse.md for derivation.
# ─────────────────────────────────────────────────────────────────────────────


def is_noop(action: np.ndarray, prev_action: np.ndarray | None, threshold: float = 1e-4) -> bool:
    """Return True iff (ee action ≈ 0) AND (gripper unchanged)."""
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    return (
        float(np.linalg.norm(action[:-1])) < threshold
        and float(action[-1]) == float(prev_action[-1])
    )


def filter_noop_indices(actions: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Return indices of frames to keep (non-noop). Assumes gripper already binarized."""
    keep = []
    prev = None
    for t in range(len(actions)):
        if not is_noop(actions[t], prev, threshold):
            keep.append(t)
        prev = actions[t]
    return np.asarray(keep, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Video writing
# ─────────────────────────────────────────────────────────────────────────────


def write_video_mp4(frames: np.ndarray, path: str, fps: int = FPS):
    """Write (T, H, W, 3) uint8 RGB frames to MP4. Flip BGR for cv2."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for t in range(frames.shape[0]):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Parquet writing
# ─────────────────────────────────────────────────────────────────────────────


def build_parquet_table(
    states: np.ndarray,     # (T, 8) float32
    actions: np.ndarray,    # (T, 7) float32
    task_index: int,
    episode_index: int,
    global_offset: int,
) -> pa.Table:
    """Build a pyarrow Table for one episode."""
    T = states.shape[0]
    timestamps = np.arange(T, dtype=np.float32) / FPS

    data = {
        "observation.state": [states[t].tolist() for t in range(T)],
        "action": [actions[t].tolist() for t in range(T)],
        "timestamp": timestamps.tolist(),
        "annotation.human.action.task_description": [task_index] * T,
        "task_index": [task_index] * T,
        "episode_index": [episode_index] * T,
        "index": list(range(global_offset, global_offset + T)),
        "frame_index": list(range(T)),
        "next.done": [False] * (T - 1) + [True],
        "next.reward": [0.0] * (T - 1) + [1.0],
    }
    return pa.table(data)


# ─────────────────────────────────────────────────────────────────────────────
# Stats computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_stats(all_states: list, all_actions: list) -> dict:
    """Compute min/max/mean/std for state and action arrays."""
    states = np.concatenate(all_states, axis=0)  # (N, 8)
    actions = np.concatenate(all_actions, axis=0)  # (N, 7)

    def stat_dict(arr):
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }

    return {
        "observation.state": stat_dict(states),
        "action": stat_dict(actions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5_dir", required=True, help="Directory with Mission_*.hdf5 files")
    parser.add_argument("--output_dir", required=True, help="Output GR00T LeRobot v2 dataset directory")
    parser.add_argument("--dataset_root", default="/home/vla-reasoning/proj/P_Bench/LIBERO/dataset",
                        help="Root of P_Bench dataset for mission_info.yaml lookup")
    parser.add_argument("--max_demos_per_mission", type=int, default=50,
                        help="Max demos to convert per HDF5 file")
    parser.add_argument("--language_mode", choices=["pref", "base"], default="pref",
                        help="'pref' = task+preference language, 'base' = task only")
    parser.add_argument("--binarize_gripper", action="store_true", default=False,
                        help="Binarize gripper action to {-1, +1}. Required if training GROOT, "
                             "which expects binary gripper distribution (continuous values cause "
                             "diffusion model to collapse to mean ~0 → gripper never actuates).")
    parser.add_argument("--filter_noops", action="store_true", default=False,
                        help="Drop frames where ee action norm < threshold AND gripper unchanged "
                             "(OpenVLA-style no-op filter). Must be combined with --binarize_gripper "
                             "so gripper equality is robust to floating-point noise. Strongly "
                             "recommended to avoid gripper collapse in relative-action mode.")
    parser.add_argument("--noop_threshold", type=float, default=1e-4,
                        help="L2 norm threshold on ee action (first 6 dims) below which a frame is "
                             "considered a no-op candidate.")
    args = parser.parse_args()

    if args.filter_noops and not args.binarize_gripper:
        print("WARNING: --filter_noops without --binarize_gripper will rarely filter anything "
              "because continuous gripper values almost never compare equal. Enable binarization.")

    hdf5_dir = Path(args.hdf5_dir)
    output_dir = Path(args.output_dir)

    # Clean output
    if output_dir.exists():
        print(f"Output dir exists, removing: {output_dir}")
        shutil.rmtree(output_dir)

    # Gather HDF5 files — supports both flat and nested (perm_*/libero_*/HDF5/Mission_*.hdf5) layouts
    hdf5_files = sorted(hdf5_dir.glob("Mission_*.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(hdf5_dir.rglob("Mission_*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files in {hdf5_dir}")

    # Pass 1: collect tasks (language instructions) and episode metadata
    tasks = []  # list of unique task strings
    task_to_index = {}
    episodes_meta = []  # [{episode_index, tasks, length}]
    episode_index = 0
    global_frame_index = 0
    total_frames = 0

    # Accumulators for stats
    all_states = []
    all_actions = []
    total_frames_before_filter = 0
    total_frames_after_filter = 0

    for hdf5_path in hdf5_files:
        mission_id = parse_mission_id(hdf5_path.name)
        mission_dir_rel = mission_id_to_dir(mission_id)
        mission_dir = Path(args.dataset_root) / mission_dir_rel
        mission_info_path = mission_dir / "mission_info.yaml"

        if not mission_info_path.exists():
            print(f"  SKIP {mission_id}: mission_info.yaml not found at {mission_info_path}")
            continue

        with open(mission_info_path) as f:
            mission_info = yaml.safe_load(f)

        if args.language_mode == "pref":
            lang = compose_pref_language(mission_info)
        else:
            lang = (mission_info.get("language_instructions") or "").strip()

        if not lang:
            print(f"  SKIP {mission_id}: empty language instruction")
            continue

        # Register task
        if lang not in task_to_index:
            task_to_index[lang] = len(tasks)
            tasks.append(lang)
        task_idx = task_to_index[lang]

        print(f"Processing {hdf5_path.name} → mission {mission_id}, task_idx={task_idx}")
        print(f"  Language: {lang[:80]}...")

        with h5py.File(hdf5_path, "r") as F:
            demo_keys = sorted(F["data"].keys(), key=lambda k: int(k.split("_")[1]))
            n_demos = min(len(demo_keys), args.max_demos_per_mission)

            for demo_i, demo_key in enumerate(demo_keys[:n_demos]):
                demo = F["data"][demo_key]
                actions_raw = demo["actions"][()]          # (T, 7)
                ee_states = demo["obs"]["ee_states"][()]   # (T, 6)
                gripper_states = demo["obs"]["gripper_states"][()]  # (T, 2)
                agentview = demo["obs"]["agentview_rgb"][()]  # (T, 256, 256, 3)
                wrist_view = demo["obs"]["eye_in_hand_rgb"][()]  # (T, 256, 256, 3)

                T = actions_raw.shape[0]

                # Build state: (T, 8) = ee_states(6) + gripper(2)
                states = np.concatenate([ee_states, gripper_states], axis=-1).astype(np.float32)
                actions = actions_raw.astype(np.float32)

                # Binarize gripper action if requested (last dim)
                # P_Bench heuristic skills produce continuous gripper ∈ [-1,+1],
                # which makes GROOT's diffusion head collapse to predicting mean ≈ 0.
                # Binarizing to {-1, +1} matches LIBERO env convention and training distribution.
                if args.binarize_gripper:
                    actions[:, 6] = np.where(actions[:, 6] >= 0, 1.0, -1.0)

                # Apply OpenVLA-style no-op filter after binarization so that gripper
                # equality check is robust. Subset states/actions/videos to kept frames.
                T_before = T
                total_frames_before_filter += T_before
                if args.filter_noops:
                    keep_idx = filter_noop_indices(actions, args.noop_threshold)
                    if len(keep_idx) < 2:
                        print(f"    WARN demo {demo_key}: filtered to {len(keep_idx)} frames, skipping")
                        continue
                    actions = actions[keep_idx]
                    states = states[keep_idx]
                    agentview = agentview[keep_idx]
                    wrist_view = wrist_view[keep_idx]
                    T = actions.shape[0]
                total_frames_after_filter += T

                # Collect for stats
                all_states.append(states)
                all_actions.append(actions)

                chunk_idx = episode_index // CHUNK_SIZE

                # ── Write parquet ──
                parquet_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
                parquet_dir.mkdir(parents=True, exist_ok=True)
                table = build_parquet_table(states, actions, task_idx, episode_index, global_frame_index)
                pq.write_table(table, parquet_dir / f"episode_{episode_index:06d}.parquet")

                # ── Write videos ──
                vid_chunk_dir = output_dir / "videos" / f"chunk-{chunk_idx:03d}"
                write_video_mp4(
                    agentview,
                    str(vid_chunk_dir / "observation.images.image" / f"episode_{episode_index:06d}.mp4"),
                )
                write_video_mp4(
                    wrist_view,
                    str(vid_chunk_dir / "observation.images.wrist_image" / f"episode_{episode_index:06d}.mp4"),
                )

                episodes_meta.append({
                    "episode_index": episode_index,
                    "tasks": [lang],
                    "length": T,
                })

                global_frame_index += T
                total_frames += T
                episode_index += 1

                if (demo_i + 1) % 10 == 0:
                    print(f"    {demo_i + 1}/{n_demos} demos done")

        print(f"  {n_demos} demos converted, total episodes so far: {episode_index}")

    # ── Write meta files ──
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # modality.json (LIBERO_PANDA format)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(MODALITY_JSON, f, indent=4)

    # info.json
    total_chunks = (episode_index - 1) // CHUNK_SIZE + 1 if episode_index > 0 else 0
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                "shape": [7],
            },
            "observation.state": {
                "dtype": "float32",
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_0", "gripper_1"],
                "shape": [8],
            },
            "observation.images.image": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": VIDEO_CODEC,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": VIDEO_CODEC,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_chunks": total_chunks,
        "total_videos": episode_index * 2,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # stats.json
    if all_states:
        stats = compute_stats(all_states, all_actions)
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Episodes: {episode_index}")
    print(f"  Frames:   {total_frames}")
    print(f"  Tasks:    {len(tasks)}")
    if args.filter_noops and total_frames_before_filter > 0:
        kept_pct = 100.0 * total_frames_after_filter / total_frames_before_filter
        print(f"  No-op filter: {total_frames_before_filter} → {total_frames_after_filter} frames "
              f"({kept_pct:.1f}% kept, {100 - kept_pct:.1f}% filtered)")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

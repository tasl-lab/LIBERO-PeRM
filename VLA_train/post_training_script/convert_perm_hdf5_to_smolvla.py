"""Convert P_Bench SinglePref HDF5 data to LeRobot v2 format for SmolVLA finetuning.

Input:  /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/Mission_*.hdf5
Output: LeRobot v2 dataset with parquet + mp4 + meta/

Usage:
    python convert_perm_hdf5_to_smolvla.py \
        --hdf5_dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref \
        --output_dir /data3/zhixu/LIBERO-PeRM/VLA_data/lerobot_smolvla/perm_singlepref \
        --dataset_root /home/vla-reasoning/proj/P_Bench/LIBERO/dataset \
        --max_demos_per_mission 50

SmolVLA (HuggingFaceVLA/smolvla_libero) expects:
    observation.images.image   ← agentview_rgb  (3, 256, 256)
    observation.images.image2  ← eye_in_hand_rgb (3, 256, 256)  [renamed from GROOT's "wrist_image"]
    observation.state          ← ee_pos(3) + ee_axisangle(3) + gripper_qpos(2) = 8D
    action                     ← 7D (pos dxyz + axis-angle drxyz + gripper)

SmolVLA uses absolute action + diffusion chunk (no relative-action collapse risk),
so the no_noops filter is OFF by default. Turn on via --filter_noops if desired
(matches OpenVLA's filter used for GROOT N1.6 finetuning).
"""

from __future__ import annotations

import argparse
import json
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FPS = 10
CHUNK_SIZE = 1000
VIDEO_CODEC = "mp4v"

# Video key renaming: P_Bench HDF5 obs key → lerobot feature key
VIDEO_KEYS = {
    "agentview_rgb": "observation.images.image",
    "eye_in_hand_rgb": "observation.images.image2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Mission ID parsing (shared with GROOT converter)
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
    """task + high_pref + low_pref (sentences joined with periods)."""
    inst = (mission_info.get("language_instructions") or "").strip()
    high = (mission_info.get("high_pref_lan") or "").strip()
    low = (mission_info.get("low_pref_lan") or "").strip()
    parts = [p for p in (inst, high, low) if p]
    return " ".join(p if p.endswith(".") else p + "." for p in parts)


# ─────────────────────────────────────────────────────────────────────────────
# No-op filter (ported from OpenVLA, optional)
# ─────────────────────────────────────────────────────────────────────────────


def is_noop(action: np.ndarray, prev_action: np.ndarray | None, threshold: float = 1e-4) -> bool:
    """True iff (ee action norm < threshold) AND (gripper unchanged from prev)."""
    if prev_action is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    return (
        float(np.linalg.norm(action[:-1])) < threshold
        and float(action[-1]) == float(prev_action[-1])
    )


def filter_noop_indices(actions: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Return indices of frames to keep (non-noop). Assumes gripper binarized."""
    keep = []
    prev = None
    for t in range(len(actions)):
        if not is_noop(actions[t], prev, threshold):
            keep.append(t)
        prev = actions[t]
    return np.asarray(keep, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Video writing (resize to 256×256 to match smolvla_libero config)
# ─────────────────────────────────────────────────────────────────────────────


def write_video_mp4(frames: np.ndarray, path: str, fps: int = FPS, size: int = 256):
    """Write (T, H, W, 3) uint8 RGB frames to MP4. Resizes each frame to size×size."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(path, fourcc, fps, (size, size))
    for t in range(frames.shape[0]):
        frame = frames[t]
        if frame.shape[:2] != (size, size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
    T = states.shape[0]
    timestamps = np.arange(T, dtype=np.float32) / FPS

    data = {
        "observation.state": [states[t].tolist() for t in range(T)],
        "action": [actions[t].tolist() for t in range(T)],
        "timestamp": timestamps.tolist(),
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
    """Min/max/mean/std for observation.state and action arrays (per-dim)."""
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

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
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5_dir", required=True, help="Directory with Mission_*.hdf5 files")
    parser.add_argument("--output_dir", required=True, help="Output LeRobot v2 dataset directory")
    parser.add_argument(
        "--dataset_root",
        default="/home/vla-reasoning/proj/P_Bench/LIBERO/dataset",
        help="Root of P_Bench dataset for mission_info.yaml lookup",
    )
    parser.add_argument(
        "--max_demos_per_mission", type=int, default=50,
        help="Max demos per HDF5 file",
    )
    parser.add_argument(
        "--language_mode", choices=["pref", "base"], default="pref",
        help="'pref' = task + preference language, 'base' = task only",
    )
    parser.add_argument(
        "--binarize_gripper", action="store_true", default=False,
        help="Binarize gripper action to {-1, +1}. Matches LIBERO env convention and makes "
             "--filter_noops meaningful (gripper equality becomes robust).",
    )
    parser.add_argument(
        "--filter_noops", action="store_true", default=False,
        help="Drop frames where ee action norm < threshold AND gripper unchanged. Not required "
             "for SmolVLA (absolute action, diffusion-chunked) but available for parity with GROOT.",
    )
    parser.add_argument(
        "--noop_threshold", type=float, default=1e-4,
        help="L2 norm threshold on ee action (first 6 dims)",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Output video frame size (SmolVLA config expects 256×256)",
    )
    args = parser.parse_args()

    if args.filter_noops and not args.binarize_gripper:
        print(
            "WARNING: --filter_noops without --binarize_gripper rarely filters anything "
            "(continuous gripper values almost never compare equal). Enabling binarization recommended.",
        )

    hdf5_dir = Path(args.hdf5_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        print(f"Output dir exists, removing: {output_dir}")
        shutil.rmtree(output_dir)

    hdf5_files = sorted(hdf5_dir.glob("Mission_*.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(hdf5_dir.rglob("Mission_*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files in {hdf5_dir}")

    tasks = []
    task_to_index = {}
    episodes_meta = []
    episode_index = 0
    global_frame_index = 0
    total_frames = 0
    total_frames_before_filter = 0
    total_frames_after_filter = 0

    all_states = []
    all_actions = []

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
                agentview = demo["obs"]["agentview_rgb"][()]  # (T, 256, 256, 3) uint8
                wrist_view = demo["obs"]["eye_in_hand_rgb"][()]  # (T, 256, 256, 3) uint8

                T_before = actions_raw.shape[0]

                states = np.concatenate([ee_states, gripper_states], axis=-1).astype(np.float32)
                actions = actions_raw.astype(np.float32)

                if args.binarize_gripper:
                    actions[:, 6] = np.where(actions[:, 6] >= 0, 1.0, -1.0)

                total_frames_before_filter += T_before
                if args.filter_noops:
                    keep = filter_noop_indices(actions, args.noop_threshold)
                    if len(keep) < 2:
                        print(f"    WARN demo {demo_key}: filtered to {len(keep)} frames, skipping")
                        continue
                    actions = actions[keep]
                    states = states[keep]
                    agentview = agentview[keep]
                    wrist_view = wrist_view[keep]
                T = actions.shape[0]
                total_frames_after_filter += T

                all_states.append(states)
                all_actions.append(actions)

                chunk_idx = episode_index // CHUNK_SIZE

                # Parquet
                parquet_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
                parquet_dir.mkdir(parents=True, exist_ok=True)
                table = build_parquet_table(states, actions, task_idx, episode_index, global_frame_index)
                pq.write_table(table, parquet_dir / f"episode_{episode_index:06d}.parquet")

                # Videos — lerobot feature key → dir
                vid_chunk_dir = output_dir / "videos" / f"chunk-{chunk_idx:03d}"
                write_video_mp4(
                    agentview,
                    str(vid_chunk_dir / VIDEO_KEYS["agentview_rgb"] / f"episode_{episode_index:06d}.mp4"),
                    size=args.image_size,
                )
                write_video_mp4(
                    wrist_view,
                    str(vid_chunk_dir / VIDEO_KEYS["eye_in_hand_rgb"] / f"episode_{episode_index:06d}.mp4"),
                    size=args.image_size,
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

    # ── Meta files ──
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    total_chunks = (episode_index - 1) // CHUNK_SIZE + 1 if episode_index > 0 else 0
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": episode_index * 2,
        "total_chunks": total_chunks,
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
                "shape": [args.image_size, args.image_size, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": args.image_size,
                    "video.width": args.image_size,
                    "video.codec": VIDEO_CODEC,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.image2": {
                "dtype": "video",
                "shape": [args.image_size, args.image_size, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": args.image_size,
                    "video.width": args.image_size,
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
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

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

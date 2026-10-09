"""
Convert (LIBERO RLDS + perm SingPref HDF5) → single LeRobot dataset for joint-mix LoRA.

Phase 1: LIBERO RLDS — 4 suites (libero_10/goal/object/spatial _no_noops), reads
         `step["observation"]["state"]` (8D), `step["action"]` (7D), wraps language
         instruction with anchor prefix `Task: <inst>.` for prompt-style consistency
         with perm's `Task: ... Preference: ...`.

Phase 2: perm SingPref HDF5 — N missions per --tasksuite-yaml. State = concat(
         ee_states[6], gripper_states[2]). Prompt = `Task: <task>. Preference: <pref>.`.

Both phases write to the same `LeRobotDataset` object (LeRobot API auto-manages
global episode_idx / chunk allocation / meta files).

Output:
  $output_dir/$repo_id/   (default repo_id = perm/SingPref-mix)
  + $repo_id/mission_mapping.json   (perm-side mapping; libero phase tracked only by counts)

Usage:
  cd /data3/zhixu/LIBERO-PeRM/VLAs/openpi
  uv run python /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/convert_libero_rlds_perm_hdf5_to_lerobot_mix.py \
    --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
    --repo-id perm/SingPref-mix \
    --libero-data-dir /data3/zhixu/datasets/openvla_libero_rlds \
    --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml \
    --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref
"""

import json
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds
import tyro
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROBOT_TYPE = "panda"
LIBERO_DATA_DIR = "/data3/zhixu/datasets/openvla_libero_rlds"
LIBERO_SUITES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]
HDF5_ROOT = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref"
TASKSUITE_YAML = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml"


def load_tasksuite(path: str) -> dict:
    with open(path) as f:
        ts = yaml.safe_load(f)
    for key in ("pairings", "tasks", "preferences"):
        if key not in ts:
            raise KeyError(f"{path}: missing top-level key {key!r}")
    return ts


def build_mission_ids(pairings: dict) -> list[str]:
    out = []
    for series_key, libero_ids in pairings.items():
        for lid in libero_ids:
            out.append(f"{series_key}_{lid}")
    return out


def compose_perm_prompt(task: str, pref: str) -> str:
    """Format: 'Task: <task>. Preference: <pref>.'"""
    task = (task or "").strip().rstrip(".")
    pref = (pref or "").strip().rstrip(".")
    if not task and not pref:
        return ""
    if not pref:
        return f"Task: {task}."
    if not task:
        return f"Preference: {pref}."
    return f"Task: {task}. Preference: {pref}."


def compose_libero_prompt(language_instruction: str) -> str:
    """LIBERO has no preference; wrap with `Task:` anchor only.
    Strips trailing period to keep format symmetric with perm's task segment."""
    inst = (language_instruction or "").strip().rstrip(".")
    return f"Task: {inst}." if inst else ""


def ingest_libero_phase(dataset: LeRobotDataset, libero_data_dir: str) -> int:
    """Phase 1: LIBERO RLDS → LeRobot. Returns episode count added."""
    n_added = 0
    sanity_printed = False
    for suite in LIBERO_SUITES:
        print(f"[LIBERO/{suite}] tfds.load …")
        raw = tfds.load(suite, data_dir=libero_data_dir, split="train")
        suite_count = 0
        for episode in raw:
            for step in episode["steps"].as_numpy_iterator():
                state = step["observation"]["state"].astype(np.float32)
                action = step["action"].astype(np.float32)
                if not sanity_printed:
                    print(
                        f"  SANITY[LIBERO {suite}] state.shape={state.shape} dtype={state.dtype} "
                        f"first={state.tolist()}"
                    )
                    print(f"  SANITY[LIBERO {suite}] action.shape={action.shape} first={action.tolist()}")
                    sanity_printed = True
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": state,
                        "actions": action,
                        "task": compose_libero_prompt(step["language_instruction"].decode()),
                    }
                )
            dataset.save_episode()
            suite_count += 1
        print(f"[LIBERO/{suite}] {suite_count} episodes")
        n_added += suite_count
    return n_added


def ingest_perm_phase(
    dataset: LeRobotDataset,
    tasksuite_yaml: str,
    hdf5_base_dir: str,
    base_episode_idx: int,
) -> tuple[int, dict]:
    """Phase 2: perm HDF5 → LeRobot. Returns (episode count added, mission_mapping)."""
    ts = load_tasksuite(tasksuite_yaml)
    mission_ids = build_mission_ids(ts["pairings"])
    print(f"[perm] tasksuite={tasksuite_yaml} → {len(mission_ids)} missions")

    missing_task = [m for m in mission_ids if m not in ts["tasks"]]
    missing_pref = [m for m in mission_ids if m not in ts["preferences"]]
    if missing_task or missing_pref:
        raise KeyError(
            f"tasksuite.yaml missing entries — tasks: {missing_task}, prefs: {missing_pref}"
        )

    mission_mapping: dict = {}
    sanity_printed = False
    n_added = 0
    global_episode_idx = base_episode_idx

    for mid in mission_ids:
        hdf5_path = Path(hdf5_base_dir) / f"Mission_{mid}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

        command = compose_perm_prompt(ts["tasks"][mid], ts["preferences"][mid])
        if not command:
            raise ValueError(f"Empty prompt for {mid}")
        print(f"[perm/{mid}] {command}")
        mission_demos = []

        with h5py.File(hdf5_path, "r") as F:
            demo_keys = sorted(F["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            for demo_key in demo_keys:
                demo = F["data"][demo_key]
                actions = demo["actions"][()]
                ee_states = demo["obs"]["ee_states"][()]
                gripper_states = demo["obs"]["gripper_states"][()]
                images = demo["obs"]["agentview_rgb"][()]
                wrist_images = demo["obs"]["eye_in_hand_rgb"][()]

                T = actions.shape[0]
                assert ee_states.shape == (T, 6), f"{mid}/{demo_key}: ee_states {ee_states.shape}"
                assert gripper_states.shape == (T, 2), (
                    f"{mid}/{demo_key}: gripper_states {gripper_states.shape}"
                )

                for t in range(T):
                    state = np.concatenate(
                        [ee_states[t], gripper_states[t]], axis=-1
                    ).astype(np.float32)
                    if not sanity_printed:
                        print(
                            f"  SANITY[perm {mid}/{demo_key}] state.shape={state.shape} "
                            f"first={state.tolist()}"
                        )
                        print(
                            f"  SANITY[perm {mid}/{demo_key}] action.shape={actions[t].shape} "
                            f"first={actions[t].tolist()}"
                        )
                        sanity_printed = True
                    dataset.add_frame(
                        {
                            "image": images[t],
                            "wrist_image": wrist_images[t],
                            "state": state,
                            "actions": actions[t].astype(np.float32),
                            "task": command,
                        }
                    )
                dataset.save_episode()

                chunk_id = global_episode_idx // 1000
                mission_demos.append(
                    {
                        "demo_key": demo_key,
                        "chunk_id": chunk_id,
                        "global_episode_idx": global_episode_idx,
                    }
                )
                global_episode_idx += 1
                n_added += 1

        mission_mapping[f"mission_{mid}"] = {
            "demos": mission_demos,
            "total_demos": len(mission_demos),
            "prompt": command,
        }

    return n_added, mission_mapping


def main(
    *,
    output_dir: str,
    repo_id: str = "perm/SingPref-mix",
    libero_data_dir: str = LIBERO_DATA_DIR,
    tasksuite_yaml: str = TASKSUITE_YAML,
    hdf5_base_dir: str = HDF5_ROOT,
):
    output_root = Path(output_dir) / repo_id
    if output_root.exists():
        raise FileExistsError(
            f"Output dir already exists: {output_root} — remove or pick a new repo_id"
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type=ROBOT_TYPE,
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # ---- Phase 1: LIBERO RLDS ----
    n_libero = ingest_libero_phase(dataset, libero_data_dir)
    print(f"=== LIBERO phase done: {n_libero} episodes ===")

    # ---- Phase 2: perm SingPref HDF5 ----
    n_perm, mission_mapping = ingest_perm_phase(
        dataset, tasksuite_yaml, hdf5_base_dir, base_episode_idx=n_libero
    )
    print(f"=== perm phase done: {n_perm} episodes ===")

    # ---- Write mission_mapping (perm-side only) ----
    mapping_file = output_root / "mission_mapping.json"
    with mapping_file.open("w") as f:
        json.dump(
            {
                "tasksuite_yaml": tasksuite_yaml,
                "libero_suites": LIBERO_SUITES,
                "n_libero_episodes": n_libero,
                "n_perm_missions": len(mission_mapping),
                "n_perm_episodes": n_perm,
                "total_episodes": n_libero + n_perm,
                "missions": mission_mapping,
            },
            f,
            indent=2,
        )
    print(f"Mapping saved: {mapping_file}")
    print(
        f"OK — {n_libero + n_perm} total episodes "
        f"(LIBERO={n_libero}, perm={n_perm}) at {output_root}"
    )


if __name__ == "__main__":
    tyro.cli(main)

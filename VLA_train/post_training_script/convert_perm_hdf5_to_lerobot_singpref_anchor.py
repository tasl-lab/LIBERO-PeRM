"""
Convert P_Bench SingPref HDF5 → LeRobot with anchor-prompt format.

Active conversion script for U / T / V experiments (anchor mode):
  - Mission list AND prompt strings are BOTH read from a tasksuite.yaml supplied via
    --tasksuite-yaml (e.g. VLA_data/SinglePref/tasksuite.yaml).
    NOT from per-mission mission_info.yaml; NOT from LIBERO/configs/task_suites.yaml.
  - Prompt format: "Task: <task>. Preference: <pref>." — anchors let the model
    address task/pref as separate segments.
  - tasksuite.yaml is the single source of truth for prompts in these experiments;
    edit it to paraphrase pref text without touching mission_info.yaml.

The flat-prompt variant `convert_perm_hdf5_to_lerobot_singpref.py` (reads
mission_info.yaml directly) is deprecated; only B used it (B done).

Usage:
  python convert_perm_hdf5_to_lerobot_singpref_anchor.py \
    --output-dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
    --repo-id perm/SingPref \
    --tasksuite-yaml /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml \
    --hdf5-base-dir /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref
"""

import json
from pathlib import Path

import h5py
import numpy as np
import tyro
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROBOT_TYPE = "panda"
HDF5_ROOT = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref"
TASKSUITE_YAML = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml"


def load_tasksuite(path: str = TASKSUITE_YAML) -> dict:
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


def compose_prompt(task: str, pref: str) -> str:
    """Format: 'Task: <task>. Preference: <pref>.' — anchors let the model
    treat task and pref as separately addressable segments."""
    task = (task or "").strip().rstrip(".")
    pref = (pref or "").strip().rstrip(".")
    if not task and not pref:
        return ""
    if not pref:
        return f"Task: {task}."
    if not task:
        return f"Preference: {pref}."
    return f"Task: {task}. Preference: {pref}."


def main(
    *,
    output_dir: str,
    repo_id: str,
    tasksuite_yaml: str = TASKSUITE_YAML,
    hdf5_base_dir: str = HDF5_ROOT,
    missions: str | None = None,
):
    """Filter logic (priority):
      1) If --missions a,b,c → use exactly that subset (must exist in pairings).
      2) Else if tasksuite has `role` map → default to all role=='train_eval' missions
         (skip role=='eval_only', which are held-out for layout-OOD eval).
      3) Else → all missions in tasksuite.
    """
    ts = load_tasksuite(tasksuite_yaml)
    all_ids = build_mission_ids(ts["pairings"])
    if missions:
        wanted = {m.strip() for m in missions.split(",") if m.strip()}
        unknown = wanted - set(all_ids)
        if unknown:
            raise KeyError(f"--missions contains ids not in tasksuite: {unknown}")
        mission_ids = [m for m in all_ids if m in wanted]
        print(f"tasksuite={tasksuite_yaml}: --missions filter → {len(mission_ids)}/{len(all_ids)}: {sorted(mission_ids)}")
    elif ts.get("role"):
        roles = ts["role"]
        mission_ids = [m for m in all_ids if roles.get(m, "train_eval") == "train_eval"]
        print(f"tasksuite={tasksuite_yaml}: role=train_eval default → {len(mission_ids)}/{len(all_ids)}: {sorted(mission_ids)}")
    else:
        mission_ids = all_ids
        print(f"tasksuite={tasksuite_yaml}: {len(mission_ids)} missions (all)")

    missing_task = [m for m in mission_ids if m not in ts["tasks"]]
    missing_pref = [m for m in mission_ids if m not in ts["preferences"]]
    if missing_task or missing_pref:
        raise KeyError(
            f"tasksuite.yaml missing entries — tasks: {missing_task}, prefs: {missing_pref}"
        )

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

    mission_mapping: dict = {}
    global_episode_idx = 0

    for mid in mission_ids:
        hdf5_path = Path(hdf5_base_dir) / f"Mission_{mid}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

        command = compose_prompt(ts["tasks"][mid], ts["preferences"][mid])
        if not command:
            raise ValueError(f"Empty prompt for {mid}")
        print(f"[{mid}] {command}")
        mission_demos = []

        # n_demos per mission from tasksuite (default 100 if not specified)
        n_demos = int((ts.get("n_demos") or {}).get(mid, 100))

        with h5py.File(hdf5_path, "r") as F:
            demo_keys = sorted(F["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            available = len(demo_keys)
            if available < n_demos:
                raise ValueError(
                    f"{mid}: requested n_demos={n_demos} but HDF5 has only {available}"
                )
            demo_keys = demo_keys[:n_demos]
            print(f"  → using {len(demo_keys)}/{available} demos")
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
                    state = np.concatenate([ee_states[t], gripper_states[t]], axis=-1).astype(
                        np.float32
                    )
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

        mission_mapping[f"mission_{mid}"] = {
            "demos": mission_demos,
            "total_demos": len(mission_demos),
            "prompt": command,
        }

    mapping_file = output_root / "mission_mapping.json"
    with mapping_file.open("w") as f:
        json.dump(
            {
                "tasksuite_yaml": tasksuite_yaml,
                "n_missions": len(mission_ids),
                "total_episodes": global_episode_idx,
                "missions": mission_mapping,
            },
            f,
            indent=2,
        )
    print(f"Mapping saved: {mapping_file}")
    print(f"OK — {global_episode_idx} episodes at {output_root}")


if __name__ == "__main__":
    tyro.cli(main)

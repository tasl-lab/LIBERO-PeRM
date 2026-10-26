"""
[DEPRECATED 2026-04-29] Flat-prompt variant. Only B used it (B done).
Active script for U / T / V is `convert_perm_hdf5_to_lerobot_singpref_anchor.py`
(reads tasks/prefs from tasksuite.yaml, anchor format "Task: ... Preference: ...").

Convert P_Bench SingPref HDF5 → LeRobot format for pi0.5 LoRA finetune (Experiment B).

Adapted from the RSS_version script but:
  - Reads mission list from LIBERO/configs/task_suites.yaml (suite key arg)
  - Reads per-mission language from dataset/perm_*/libero_*/mission_info.yaml
  - Builds the `task` string with the SAME logic as PolicyEval eval.py
    `compose_prompt(condition='pref')` — so training & eval prompts match
  - Takes ALL demos per mission (no 100 cap)
  - Same 8D state (ee_states + gripper_states) / 7D action as RSS

Usage:
  python convert_perm_hdf5_to_lerobot_singpref.py \
    --output_dir /data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data \
    --repo_id perm/SingPref_InDomain \
    --suite single-pref-in-domain
"""

import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import tyro
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROBOT_TYPE = "panda"
DATASET_ROOT = "/data3/zhixu/LIBERO-PeRM/LIBERO/dataset"
HDF5_ROOT = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref"
SUITES_YAML = "/data3/zhixu/LIBERO-PeRM/LIBERO/configs/task_suites.yaml"


def parse_mission_id(mission_id: str):
    m = re.match(r"^(\d+-\d+)_(.+)$", mission_id)
    if not m:
        raise ValueError(f"Invalid mission_id: {mission_id!r}")
    return f"perm_{m.group(1).replace('-', '_')}", f"libero_{m.group(2)}"


def compose_pref_prompt(mission_info: dict) -> str:
    inst = (mission_info.get("language_instructions") or "").strip()
    high = (mission_info.get("high_pref_lan") or "").strip()
    low = (mission_info.get("low_pref_lan") or "").strip()
    parts = [p for p in (inst, high, low) if p]
    return " ".join(p if p.endswith(".") else p + "." for p in parts)


def load_mission_lang(mission_id: str) -> str:
    perm, libero = parse_mission_id(mission_id)
    series = "series_" + re.match(r"perm_(\d+)_\d+", perm).group(1)
    mi_path = Path(DATASET_ROOT) / series / perm / libero / "mission_info.yaml"
    if not mi_path.exists():
        raise FileNotFoundError(f"mission_info.yaml missing: {mi_path}")
    with mi_path.open() as f:
        mission_info = yaml.safe_load(f)
    # Match PolicyEval.eval.compose_prompt(..., 'pref') byte-for-byte — no .lower(),
    # no extra .strip(): the composer already strips each part.
    prompt = compose_pref_prompt(mission_info)
    if not prompt:
        raise ValueError(f"Empty pref prompt for {mission_id} ({mi_path})")
    return prompt


def load_suite_missions(suite_name: str):
    with open(SUITES_YAML) as f:
        suites = yaml.safe_load(f)
    entry = suites.get(suite_name)
    if entry is None:
        raise KeyError(f"Suite {suite_name!r} not in {SUITES_YAML}")
    missions = entry["missions"] if isinstance(entry, dict) else entry
    return list(missions)


def main(
    *,
    output_dir: str,
    repo_id: str,
    suite: str = "single-pref-in-domain",
    hdf5_base_dir: str = HDF5_ROOT,
):
    mission_ids = load_suite_missions(suite)
    print(f"Suite {suite!r}: {len(mission_ids)} missions")

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

        command = load_mission_lang(mid)
        print(f"[{mid}] {command}")
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
                "suite": suite,
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

"""tfds builder: perm HDF5 -> RLDS for OpenVLA-OFT finetune.

Schema per step:
  observation/image           (256, 256, 3) uint8   agentview_rgb
  observation/wrist_image     (256, 256, 3) uint8   eye_in_hand_rgb
  observation/state           (8,) float32          ee_states[6] + gripper_states[2]
  action                      (7,) float32
  language_instruction        string                "Task: ... Preference: ...."
  is_first / is_last / is_terminal  bool
  reward / discount           float32 (placeholders)

Usage (rlds_env):
  cd /data3/zhixu/LIBERO-PeRM/VLA_train/post_training_script/perm_rlds
  conda activate rlds_env
  PBENCH_HDF5_DIR=/data3/zhixu/LIBERO-PeRM/VLA_data/PrefCompose \
  PBENCH_TASKSUITE=/data3/zhixu/LIBERO-PeRM/VLA_data/PrefCompose/tasksuite.yaml \
  PBENCH_RLDS_OUT=/data3/zhixu/LIBERO-PeRM/VLA_data/finetune_data/perm_rlds \
    tfds build --data_dir=$PBENCH_RLDS_OUT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds
import yaml


HDF5_DIR = os.environ.get("PBENCH_HDF5_DIR", "/data4/zhixu/SinglePref")
TASKSUITE_YAML = os.environ.get(
    "PBENCH_TASKSUITE", "/data4/zhixu/SinglePref/tasksuite.yaml"
)
# Comma-separated mission_id whitelist; empty => all missions in tasksuite.
MISSION_FILTER = [
    s for s in os.environ.get("PBENCH_MISSION_FILTER", "").split(",") if s
]
# 0 => no cap.
MAX_DEMOS_PER_MISSION = int(os.environ.get("PBENCH_MAX_DEMOS_PER_MISSION", "0"))


def _compose_prompt(task: str, pref: str) -> str:
    task = (task or "").strip().rstrip(".")
    pref = (pref or "").strip().rstrip(".")
    if not task and not pref:
        return ""
    if not pref:
        return f"Task: {task}."
    if not task:
        return f"Preference: {pref}."
    return f"Task: {task}. Preference: {pref}."


def _load_tasksuite(path: str) -> dict:
    with open(path) as f:
        ts = yaml.safe_load(f)
    for k in ("pairings", "tasks", "preferences"):
        if k not in ts:
            raise KeyError(f"{path}: missing top-level key {k!r}")
    return ts


def _build_mission_list(ts: dict) -> list[str]:
    out: list[str] = []
    for series_key, suffixes in ts["pairings"].items():
        for sfx in suffixes:
            out.append(f"{series_key}_{sfx}")
    return out


class Pbench(tfds.core.GeneratorBasedBuilder):
    """Pbench preference-conditioned manipulation dataset (HDF5 -> RLDS)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release"}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Agentview RGB",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Eye-in-hand RGB",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="ee_states[6] + gripper_states[2]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="delta-EE 6 + gripper 1",
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "mission_id": tfds.features.Text(),
                            "demo_key": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterator[Tuple[str, dict]]:
        ts = _load_tasksuite(TASKSUITE_YAML)
        mission_ids = _build_mission_list(ts)
        if MISSION_FILTER:
            kept = [m for m in mission_ids if m in MISSION_FILTER]
            print(f"[filter] {len(kept)}/{len(mission_ids)} missions after whitelist")
            mission_ids = kept

        for mid in mission_ids:
            hdf5_path = Path(HDF5_DIR) / f"Mission_{mid}.hdf5"
            if not hdf5_path.exists():
                print(f"[skip] missing {hdf5_path}")
                continue

            prompt = _compose_prompt(
                ts["tasks"].get(mid, ""), ts["preferences"].get(mid, "")
            )
            if not prompt:
                print(f"[skip] empty prompt for {mid}")
                continue

            with h5py.File(hdf5_path, "r") as F:
                demo_keys = sorted(
                    F["data"].keys(), key=lambda k: int(k.split("_")[-1])
                )
                if MAX_DEMOS_PER_MISSION > 0:
                    demo_keys = demo_keys[:MAX_DEMOS_PER_MISSION]
                for demo_key in demo_keys:
                    demo = F["data"][demo_key]
                    actions = demo["actions"][()].astype(np.float32)
                    ee_states = demo["obs"]["ee_states"][()].astype(np.float32)
                    gripper_states = demo["obs"]["gripper_states"][()].astype(
                        np.float32
                    )
                    images = demo["obs"]["agentview_rgb"][()]
                    wrist_images = demo["obs"]["eye_in_hand_rgb"][()]

                    T = actions.shape[0]
                    if not (
                        ee_states.shape == (T, 6)
                        and gripper_states.shape == (T, 2)
                        and images.shape == (T, 256, 256, 3)
                        and wrist_images.shape == (T, 256, 256, 3)
                    ):
                        raise ValueError(
                            f"{mid}/{demo_key}: shape mismatch "
                            f"act={actions.shape} ee={ee_states.shape} "
                            f"grip={gripper_states.shape} img={images.shape} "
                            f"wrist={wrist_images.shape}"
                        )

                    state_seq = np.concatenate([ee_states, gripper_states], axis=-1)

                    steps = []
                    for t in range(T):
                        steps.append(
                            {
                                "observation": {
                                    "image": images[t],
                                    "wrist_image": wrist_images[t],
                                    "state": state_seq[t],
                                },
                                "action": actions[t],
                                "discount": np.float32(1.0),
                                "reward": np.float32(1.0 if t == T - 1 else 0.0),
                                "is_first": t == 0,
                                "is_last": t == T - 1,
                                "is_terminal": t == T - 1,
                                "language_instruction": prompt,
                            }
                        )

                    ep_id = f"{mid}__{demo_key}"
                    yield ep_id, {
                        "steps": steps,
                        "episode_metadata": {
                            "file_path": str(hdf5_path),
                            "mission_id": mid,
                            "demo_key": demo_key,
                        },
                    }

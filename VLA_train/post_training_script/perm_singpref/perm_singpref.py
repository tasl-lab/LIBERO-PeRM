"""TFDS GeneratorBasedBuilder for the P-Bench SinglePref task suite, RLDS layout.

Source data:  VLA_data/SinglePref/Mission_<mid>.hdf5  (40 missions × 100 demos each)
Output:       <data_dir>/perm_singpref/1.0.0/      (sharded TFRecord)

Schema mirrors openvla/modified_libero_rlds so the existing OXE/RLDS pipeline
(`prismatic.vla.datasets.rlds.*`) can consume it with the same standardisation
transform as the LIBERO datasets — see `libero_dataset_transform`.

Per-step payload:
  observation/image         (256, 256, 3) uint8   — agentview, rotated 180°
  observation/wrist_image   (256, 256, 3) uint8   — eye-in-hand, rotated 180°
  observation/state         (8,)          float32 — ee_states(6) ⊕ gripper_states(2)
  action                    (7,)          float32 — 6 delta xyz/rpy ⊕ gripper{-1,+1}
  is_first / is_last / is_terminal / discount / reward
  language_instruction      str — `Task: <task>. Preference: <pref>.` from tasksuite.yaml

Build:
  cd <repo>/VLA_train/post_training_script/perm_singpref_rlds
  tfds build --data_dir <repo>/VLA_data/finetune_data
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HDF5_DIR = _REPO_ROOT / "VLA_data" / "SinglePref"
_DEFAULT_TASKSUITE_YAML = _DEFAULT_HDF5_DIR / "tasksuite.yaml"

HDF5_DIR = Path(os.environ.get("PBENCH_HDF5_DIR", _DEFAULT_HDF5_DIR))
TASKSUITE_YAML = Path(os.environ.get("PBENCH_TASKSUITE_YAML", _DEFAULT_TASKSUITE_YAML))


def _compose_prompt(task: str, pref: str) -> str:
    """`Task: <task>. Preference: <pref>.` (matches converter convention)."""
    task = (task or "").strip().rstrip(".")
    pref = (pref or "").strip().rstrip(".")
    if not task and not pref:
        return ""
    if not pref:
        return f"Task: {task}."
    if not task:
        return f"Preference: {pref}."
    return f"Task: {task}. Preference: {pref}."


def _load_tasksuite() -> dict:
    with open(TASKSUITE_YAML, "r") as f:
        return yaml.safe_load(f) or {}


def _list_missions(ts: dict) -> list:
    """All 40 missions present in tasksuite.yaml AND with a Mission_<mid>.hdf5 on disk."""
    libero_inst = ts.get("libero_instructions", {}) or {}
    have = []
    for mid in libero_inst.keys():
        path = HDF5_DIR / f"Mission_{mid}.hdf5"
        if path.is_file():
            have.append(mid)
    have.sort(key=_mission_sort_key)
    return have


def _mission_sort_key(mid: str) -> Tuple[int, int, str]:
    s, *rest = mid.split("_", 1)
    a, b = s.split("-")
    return int(a), int(b), rest[0] if rest else ""


class PbenchSingpref(tfds.core.GeneratorBasedBuilder):
    """P-Bench SinglePref demos (40 missions × 100 demos)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release: 40 missions × 100 demos (3999 total)."}

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
                                        doc="Agentview RGB, rotated 180° (LIBERO RLDS convention).",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Eye-in-hand RGB, rotated 180°.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="ee_states(6) ⊕ gripper_states(2).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="6 delta xyz/rpy ⊕ gripper {-1=open,+1=close}.",
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
                            "mission_id": tfds.features.Text(),
                            "demo_key": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterator[Tuple[str, dict]]:
        ts = _load_tasksuite()
        tasks = ts.get("tasks", {}) or {}
        prefs = ts.get("preferences", {}) or {}
        missions = _list_missions(ts)
        if not missions:
            raise RuntimeError(
                f"No Mission_*.hdf5 found under {HDF5_DIR}. "
                f"Set PBENCH_HDF5_DIR if data is elsewhere."
            )

        for mid in missions:
            prompt = _compose_prompt(tasks.get(mid, ""), prefs.get(mid, ""))
            if not prompt:
                # Fall back to the LIBERO base instruction if pref/task entries are missing.
                prompt = (ts.get("libero_instructions", {}).get(mid) or "").strip()
            hdf5_path = HDF5_DIR / f"Mission_{mid}.hdf5"

            with h5py.File(str(hdf5_path), "r") as F:
                demo_keys = sorted(
                    F["data"].keys(),
                    key=lambda k: int(re.findall(r"\d+", k)[0]),
                )
                for demo_key in demo_keys:
                    demo = F["data"][demo_key]
                    actions = np.asarray(demo["actions"][()], dtype=np.float32)
                    ee_states = np.asarray(demo["obs"]["ee_states"][()], dtype=np.float32)
                    gripper_states = np.asarray(
                        demo["obs"]["gripper_states"][()], dtype=np.float32
                    )
                    images = np.asarray(demo["obs"]["agentview_rgb"][()], dtype=np.uint8)
                    wrist_images = np.asarray(
                        demo["obs"]["eye_in_hand_rgb"][()], dtype=np.uint8
                    )

                    T = actions.shape[0]
                    if not (
                        ee_states.shape == (T, 6)
                        and gripper_states.shape == (T, 2)
                        and images.shape == (T, 256, 256, 3)
                        and wrist_images.shape == (T, 256, 256, 3)
                    ):
                        raise ValueError(
                            f"Shape mismatch in {mid}/{demo_key}: T={T}, "
                            f"ee_states={ee_states.shape}, "
                            f"gripper_states={gripper_states.shape}, "
                            f"image={images.shape}, wrist_image={wrist_images.shape}"
                        )

                    # Rotate 180° to match LIBERO RLDS preprocessing convention.
                    images = images[:, ::-1, ::-1, :]
                    wrist_images = wrist_images[:, ::-1, ::-1, :]

                    state = np.concatenate([ee_states, gripper_states], axis=-1).astype(
                        np.float32
                    )

                    steps = []
                    for t in range(T):
                        steps.append(
                            {
                                "observation": {
                                    "image": np.ascontiguousarray(images[t]),
                                    "wrist_image": np.ascontiguousarray(wrist_images[t]),
                                    "state": state[t],
                                },
                                "action": actions[t],
                                "discount": np.float32(1.0),
                                "reward": np.float32(1.0 if t == T - 1 else 0.0),
                                "is_first": bool(t == 0),
                                "is_last": bool(t == T - 1),
                                "is_terminal": bool(t == T - 1),
                                "language_instruction": prompt,
                            }
                        )

                    yield (
                        f"{mid}/{demo_key}",
                        {
                            "steps": steps,
                            "episode_metadata": {
                                "mission_id": mid,
                                "demo_key": demo_key,
                            },
                        },
                    )

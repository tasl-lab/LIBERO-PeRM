"""Write rollout HDF5s in the P_Bench schema consumed by TaskEval.hdf5_player.

Schema:
    data/demo_0
        attrs: model_file (str), num_samples (int), init_state (flat state)
        actions, states, rewards, dones, robot_states       (datasets)
        obs/{agentview_rgb, eye_in_hand_rgb, ee_states, ee_pos, ee_ori,
             gripper_states, joint_states}                  (datasets)
        low_pref_rewards/{pref_type: [step, step, ...]}     (group of datasets)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import h5py
import numpy as np


def _compressed(grp, name, data):
    if data is None or (hasattr(data, "__len__") and len(data) == 0):
        return
    arr = np.stack(data, axis=0) if isinstance(data, list) else np.array(data)
    chunks = (1,) + arr.shape[1:] if arr.ndim > 1 else True
    grp.create_dataset(
        name, data=arr, chunks=chunks, shuffle=True, compression="gzip", compression_opts=1
    )


class RolloutRecorder:
    """Accumulates rollout data in memory then writes a single HDF5 on flush."""

    def __init__(self):
        self.actions: List = []
        self.states: List = []
        self.rewards: List = []
        self.dones: List = []
        self.robot_states: List = []
        self.obs: Dict[str, List] = {
            "agentview_rgb": [],
            "eye_in_hand_rgb": [],
            "ee_states": [],
            "ee_pos": [],
            "ee_ori": [],
            "gripper_states": [],
            "joint_states": [],
        }
        self.low_pref_rewards: Dict[str, List[float]] = {}
        self.model_xml: Optional[str] = None

    def set_model_xml(self, xml: str) -> None:
        self.model_xml = xml

    def append(
        self,
        raw_obs: Dict,
        action,
        reward: float,
        done: bool,
        sim_state,
        low_pref_step: Optional[Dict[str, float]] = None,
    ) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.states.append(np.asarray(sim_state, dtype=np.float64))
        self.rewards.append(float(reward))
        self.dones.append(1 if done else 0)

        # robot_state (9D): eef_pos(3) + axisangle(3) + gripper(2 or 1 padded)
        rs = _robot_state_vector(raw_obs)
        self.robot_states.append(rs)

        if "robot0_gripper_qpos" in raw_obs:
            self.obs["gripper_states"].append(np.asarray(raw_obs["robot0_gripper_qpos"]))
        if "robot0_joint_pos" in raw_obs:
            self.obs["joint_states"].append(np.asarray(raw_obs["robot0_joint_pos"]))
        if "robot0_eef_pos" in raw_obs and "robot0_eef_quat" in raw_obs:
            eef_pos = np.asarray(raw_obs["robot0_eef_pos"])
            eef_ori = _quat2axisangle(raw_obs["robot0_eef_quat"])
            self.obs["ee_pos"].append(eef_pos)
            self.obs["ee_ori"].append(eef_ori)
            self.obs["ee_states"].append(np.concatenate([eef_pos, eef_ori]))

        if "agentview_image" in raw_obs:
            self.obs["agentview_rgb"].append(_flip_rgb(raw_obs["agentview_image"]))
        if "robot0_eye_in_hand_image" in raw_obs:
            self.obs["eye_in_hand_rgb"].append(_flip_rgb(raw_obs["robot0_eye_in_hand_image"]))

        if low_pref_step:
            for k, v in low_pref_step.items():
                self.low_pref_rewards.setdefault(k, []).append(float(v))

    def write(self, hdf5_path: str, demo_idx: int = 0) -> None:
        with h5py.File(hdf5_path, "w") as f:
            data_grp = f.create_group("data")
            demo = data_grp.create_group(f"demo_{demo_idx}")
            obs_grp = demo.create_group("obs")

            for k, v in self.obs.items():
                _compressed(obs_grp, k, v)
            _compressed(demo, "actions", self.actions)
            _compressed(demo, "states", self.states)
            _compressed(demo, "rewards", self.rewards)
            _compressed(demo, "dones", self.dones)
            _compressed(demo, "robot_states", self.robot_states)

            lp_grp = demo.create_group("low_pref_rewards")
            for k, v in self.low_pref_rewards.items():
                _compressed(lp_grp, k, v)

            demo.attrs["num_samples"] = len(self.actions)
            if self.model_xml is not None:
                demo.attrs["model_file"] = self.model_xml
            if self.states:
                demo.attrs["init_state"] = np.asarray(self.states[0], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _flip_rgb(img):
    """LIBERO returns images upside down relative to the viewer; flip to match
    what the P_Bench HDF5 schema stores (and what VLM_Evaluator expects)."""
    arr = np.asarray(img)
    return np.ascontiguousarray(arr[::-1, :, :])


def _quat2axisangle(quat):
    import math

    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if den < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _robot_state_vector(raw_obs: Dict) -> np.ndarray:
    eef_pos = np.asarray(raw_obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    if "robot0_eef_quat" in raw_obs:
        eef_ori = _quat2axisangle(raw_obs["robot0_eef_quat"]).astype(np.float32)
    else:
        eef_ori = np.zeros(3, dtype=np.float32)
    grip = np.asarray(raw_obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
    vec = np.concatenate([eef_pos, eef_ori, grip])
    if vec.shape[0] < 9:
        vec = np.concatenate([vec, np.zeros(9 - vec.shape[0], dtype=np.float32)])
    return vec[:9]

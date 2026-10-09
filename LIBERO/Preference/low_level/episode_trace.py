"""Per-episode observable trace shared by every motion-tier metric.

Design (SA metric rework, plan A0):
- One raw per preference = one episode-level scalar computed at the END of
  the episode from recorded observables (EE position / orientation, gripper
  finger separation), so segment-aware definitions (carry window, lateral
  moves, steps-until-success) are possible.
- The same recorder is driven online (PolicyEval.LowPrefAccumulator, one
  `record` per env step) and offline (scripts/calibrate_low_pref replay of
  HDF5 `states`), so there is exactly one implementation of every raw.
- `mark_success()` freezes `t_end` at the first success step: post-success
  frames (eval keeps stepping 21 extra frames; demos keep a short tail) never
  pollute a raw.

Readers below use only `env.sim` (works for the real robosuite env and for
the calibrate replay's thin `_wrap_env` shim) and degrade to zeros instead
of raising, mirroring the previous helpers in trajectory.py.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from .transport_window import carry_span, detect_transport_window, lateral_move_mask

DEFAULT_EEF = "gripper0_eef"
DEFAULT_FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
LATERAL_STEP_EPS = 1e-3   # m/step; per-step |dxy| above this = laterally moving


def _eef_name(env: Any) -> str:
    try:
        return getattr(env.robots[0], "eef_name", DEFAULT_EEF)
    except Exception:
        return DEFAULT_EEF


def read_ee_pos(env: Any) -> np.ndarray:
    try:
        return np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(_eef_name(env))],
                          dtype=np.float64).copy()
    except Exception:
        return np.zeros(3)


def read_ee_quat(env: Any) -> np.ndarray:
    """EE orientation as wxyz quaternion."""
    try:
        return np.asarray(env.sim.data.body_xquat[env.sim.model.body_name2id(_eef_name(env))],
                          dtype=np.float64).copy()
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0])


def read_gripper(env: Any) -> np.ndarray:
    """Two finger joint positions (same convention as obs `robot0_gripper_qpos`)."""
    names = None
    try:
        names = list(env.robots[0].gripper.joints)
    except Exception:
        names = list(DEFAULT_FINGER_JOINTS)
    try:
        return np.array([float(env.sim.data.get_joint_qpos(n)) for n in names[:2]])
    except Exception:
        return np.zeros(2)


def read_base_z(env: Any) -> float:
    try:
        return float(env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")][2])
    except Exception:
        return 0.0


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Relative rotation angle (rad) between two unit quaternions."""
    d = abs(float(np.dot(q1, q2)))
    return float(2.0 * np.arccos(min(1.0, max(-1.0, d))))


class EpisodeTrace:
    """Records (ee_pos, ee_quat, gripper) per step; freezes at first success."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ee: List[np.ndarray] = []
        self._quat: List[np.ndarray] = []
        self._grip: List[np.ndarray] = []
        self.t_success: int | None = None

    def record(self, env: Any) -> None:
        self._ee.append(read_ee_pos(env))
        self._quat.append(read_ee_quat(env))
        self._grip.append(read_gripper(env))

    def record_arrays(self, ee_pos, ee_quat=None, gripper=None) -> None:
        """Append one step from explicit arrays (tests / offline obs replay)."""
        self._ee.append(np.asarray(ee_pos, dtype=np.float64))
        self._quat.append(np.asarray(ee_quat if ee_quat is not None else [1, 0, 0, 0], dtype=np.float64))
        self._grip.append(np.asarray(gripper if gripper is not None else [0.04, -0.04], dtype=np.float64))

    def mark_success(self) -> None:
        """Call on the step where the task first succeeds (after `record`)."""
        if self.t_success is None:
            self.t_success = max(len(self._ee), 1)

    # ---- views -------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self._ee)

    @property
    def t_end(self) -> int:
        return self.t_success if self.t_success is not None else self.n

    @property
    def ee(self) -> np.ndarray:
        return np.asarray(self._ee[: self.t_end]).reshape(-1, 3)

    @property
    def quat(self) -> np.ndarray:
        return np.asarray(self._quat[: self.t_end]).reshape(-1, 4)

    @property
    def grip(self) -> np.ndarray:
        return np.asarray(self._grip[: self.t_end]).reshape(-1, 2)

    # ---- segments ----------------------------------------------------
    def step_speeds(self) -> np.ndarray:
        """|Δp| per step (m/step), length n-1."""
        ee = self.ee
        return np.linalg.norm(np.diff(ee, axis=0), axis=1) if len(ee) > 1 else np.zeros(0)

    def step_accels(self) -> np.ndarray:
        """|Δ²p| per step (m/step²), length n-2."""
        ee = self.ee
        if len(ee) < 3:
            return np.zeros(0)
        return np.linalg.norm(np.diff(np.diff(ee, axis=0), axis=0), axis=1)

    def lateral_step_mask(self, eps: float = LATERAL_STEP_EPS) -> np.ndarray:
        """Per-transition mask (length n-1): |Δxy| > eps."""
        ee = self.ee
        if len(ee) < 2:
            return np.zeros(0, dtype=bool)
        return np.linalg.norm(np.diff(ee[:, :2], axis=0), axis=1) > eps

    def lateral_move_mask(self) -> np.ndarray:
        """Per-step mask of sustained lateral-move segments (approach + carry)."""
        return lateral_move_mask(self.ee) if self.n >= 2 else np.zeros(self.n, dtype=bool)

    def carry_window(self) -> tuple[int, int] | None:
        """[carry start, start of final descent] (inclusive) or None.

        Start = first gripper-closed step (grasp); end = where the final
        placing descent begins (from transport_window's descent detector),
        so a pref demo that lifts diagonally while translating keeps its
        whole diagonal, while the naive lift-move-lower L keeps lift+move.
        """
        if self.n < 3:
            return None
        span = carry_span(self.grip)
        if span is None:
            return None
        c0, c1 = span
        z = self.ee[c0: c1 + 1, 2]
        # final descent starts after the last step near the carry z-max
        # (1 cm band). Robust to diagonal lifts, where transport_window's
        # "lift stops rising" rule would only fire at the end of the move.
        t_desc = c0 + int(np.flatnonzero(z >= z.max() - self.DESCENT_BAND_M)[-1])
        return c0, max(t_desc, c0 + 1)

    DESCENT_BAND_M: float = 0.01

    def transport_mask(self) -> np.ndarray:
        """Per-step mask of the carry (transport) window, all-False if none."""
        mask = np.zeros(self.n, dtype=bool)
        if self.n >= 3:
            win = detect_transport_window(self.ee, self.grip)
            if win is not None:
                mask[win[0]: win[1] + 1] = True
        return mask[: self.t_end]

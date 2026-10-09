"""Skill-level geometric metric: release height of a held object above a target.

Designed for `throw` and similar dynamic skills where the visual difference
between throw and put is the *vertical clearance* at release. Looks at the
moment the held object detaches from the gripper (sudden EE→obj distance
jump) and records ee_z − target_z at that instant. Score is a band in
[min_height, max_height] (typical: 0.10m and 0.40m).

Yaml:
    low_prefs:
      - type: release_height
        params:
          held_obj: alphabet_soup_1
          target_obj: wooden_tray_1
          min_height: 0.10
          max_height: 0.40

Registers as `release_height` via BaseMotionPreference auto-registry.
"""

from __future__ import annotations

from typing import Any, Set

import numpy as np

from ..motion_preference import BaseMotionPreference


class ReleaseHeightPreference(BaseMotionPreference):
    """Reward when held_obj is released ≥ min_height above target_obj.

    `level = "skill"` — runs every step in sim, no VLM call.
    """

    saturation = 1.0
    level = "skill"

    def __init__(
        self,
        held_obj: str,
        target_obj: str,
        min_height: float = 0.10,
        max_height: float = 0.40,
        detach_thresh: float = 0.10,
        attach_thresh: float = 0.05,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.held_obj = held_obj
        self.target_obj = target_obj
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.detach_thresh = float(detach_thresh)
        self.attach_thresh = float(attach_thresh)
        self.type = "ReleaseHeightPreference"
        self._reset_state()

    def _reset_state(self) -> None:
        self._prev_ee_to_held: float | None = None
        self._released_clearance: float | None = None
        self._was_attached: bool = False

    def reset(self, env: Any) -> None:
        self._reset_state()

    def _ee_pos(self, env: Any) -> np.ndarray:
        eef = getattr(env.robots[0], "eef_name", "gripper0_eef")
        bid = env.sim.model.body_name2id(eef)
        return np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32)

    def _obj_pos(self, env: Any, name: str) -> np.ndarray:
        bid = env.obj_body_id.get(name)
        if bid is None:
            # Fallback to MuJoCo body lookup; may raise.
            bid = env.sim.model.body_name2id(name)
        return np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32)

    def get_reward(self, env: Any) -> float:
        try:
            ee = self._ee_pos(env)
            held = self._obj_pos(env, self.held_obj)
            target = self._obj_pos(env, self.target_obj)
        except Exception:
            return 0.0
        dist = float(np.linalg.norm(ee - held))
        # Track attach state: held obj close to EE means gripper holds it.
        if dist < self.attach_thresh:
            self._was_attached = True
        # Detect release: obj was attached, then distance jumps above detach_thresh.
        if (
            self._was_attached
            and self._released_clearance is None
            and self._prev_ee_to_held is not None
            and self._prev_ee_to_held < self.attach_thresh
            and dist > self.detach_thresh
        ):
            clearance = float(ee[2] - target[2])
            self._released_clearance = clearance
        self._prev_ee_to_held = dist
        if self._released_clearance is None:
            return 0.0
        h = self._released_clearance
        if h <= 0:
            return 0.0
        # Linear band: 0 below min_height, 1 at max_height (or above).
        if h >= self.max_height:
            return 1.0
        if h <= self.min_height:
            return float(h / self.min_height) * 0.5  # 0..0.5 below min anchor
        return 0.5 + 0.5 * (h - self.min_height) / (self.max_height - self.min_height)

    def raw(self) -> dict:
        return {
            "released_clearance_m": self._released_clearance,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.held_obj, self.target_obj}

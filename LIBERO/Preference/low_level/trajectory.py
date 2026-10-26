"""Trajectory-shape preferences: straight path, height offset, rotation.

All three record an EpisodeTrace and compute the raw at episode end:

  StraightPathPreference  raw = path/chord − 1 over the carry segment
                          [grasp, start of final descent]. A pref demo lifts
                          diagonally while translating (→ ~0.05); the naive
                          lift-move-lower L gives ~0.4. The old whole-episode
                          running ratio collapsed because the end point comes
                          back to the start height (chord ≈ 0).
  HeightOffsetPreference  raw = max EE z over sustained lateral-move steps
                          (approach + carry; gen applies path_height_offset to
                          every move_to) relative to robot base z. "Move
                          higher" → larger; "Close to ground" → smaller; the
                          direction comes from the dual anchors.
  RotationPreference      raw = share of the episode's total wrist rotation
                          that happens on laterally-moving steps. "Rotate
                          when moving" pre-rotates during the approach move
                          (~0.87); the naive plan rotates in place above the
                          object (~0.27). The previous per-step mean over
                          the carry window missed the approach and diluted
                          the signal to noise level.
"""

import numpy as np
from typing import Any, Set
from .motion_preference import BaseMotionPreference
from .episode_trace import EpisodeTrace, quat_angle, read_base_z


class StraightPathPreference(BaseMotionPreference):
    saturation = 1.0
    INEFFICIENCY_HEADROOM: float = 10.0

    def __init__(
        self,
        max_deviation: float = 0.05,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        if max_deviation <= 0:
            raise ValueError("max_deviation must be positive")
        self.max_deviation = float(max_deviation)
        self.type = "StraightPathPreference"
        self.trace = EpisodeTrace()

    def reset(self, env: Any) -> None:
        self.trace.reset()

    def mark_success(self) -> None:
        self.trace.mark_success()

    @staticmethod
    def _inefficiency(pts: np.ndarray) -> float | None:
        if len(pts) < 2:
            return None
        path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        if chord < 1e-3:
            return None
        return path / chord - 1.0

    def get_reward(self, env: Any) -> float:
        self.trace.record(env)
        raw = self.final_raw()
        if raw is None:
            return 1.0
        return float(max(0.0, 1.0 - raw / (self.max_deviation * self.INEFFICIENCY_HEADROOM)))

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        """path/chord − 1 over [grasp, final-descent start]; whole episode
        (up to success) when no carry is detected."""
        ee = self.trace.ee
        if len(ee) < 2:
            return None
        win = self.trace.carry_window()
        pts = ee[win[0]: win[1] + 1] if win is not None else ee
        return self._inefficiency(pts)

    @property
    def objects_of_interest(self) -> Set[str]:
        return set()


class HeightOffsetPreference(BaseMotionPreference):
    saturation = 1.0
    DEFAULT_SAFE_LIFT_M: float = 0.2

    def __init__(
        self,
        height_offset: float = 0.10,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.height_offset = float(height_offset)
        self.type = "HeightOffsetPreference"
        self.trace = EpisodeTrace()
        self._base_z: float | None = None

    def reset(self, env: Any) -> None:
        self.trace.reset()
        self._base_z = None

    def mark_success(self) -> None:
        self.trace.mark_success()

    def get_reward(self, env: Any) -> float:
        if self._base_z is None:
            self._base_z = read_base_z(env)
        self.trace.record(env)
        raw = self.final_raw()
        if raw is None or self.height_offset == 0.0:
            return 1.0
        observed = raw - self.DEFAULT_SAFE_LIFT_M
        err = abs(observed - self.height_offset)
        return float(max(0.0, 1.0 - err / max(abs(self.height_offset), 1e-3)))

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        """Max EE z over sustained lateral-move steps, relative to base z."""
        ee = self.trace.ee
        if len(ee) == 0:
            return None
        mask = self.trace.lateral_move_mask()
        z = ee[mask, 2] if mask.any() else ee[:, 2]
        return float(z.max() - (self._base_z or 0.0))

    @property
    def objects_of_interest(self) -> Set[str]:
        return set()


class RotationPreference(BaseMotionPreference):
    saturation = 1.0
    # rad; below this the share is ill-conditioned (a few milliradians of noise
    # can read as "100% of rotation happened while translating") — report 0
    # instead. 0.1 rad ~ 6 deg; real episodes end at 0.20 (naive) to 1.76 (pref),
    # so only the near-zero opening of the running curve is affected.
    MIN_TOTAL_ROT: float = 0.1

    def __init__(
        self,
        angle_per_step: float = 0.1,
        lateral_thresh: float = 0.001,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.angle_per_step = float(angle_per_step)
        self.lateral_thresh = float(lateral_thresh)
        self.type = "RotationPreference"
        self.trace = EpisodeTrace()

    def reset(self, env: Any) -> None:
        self.trace.reset()

    def mark_success(self) -> None:
        self.trace.mark_success()

    def _split(self) -> tuple[float, float]:
        q = self.trace.quat
        if len(q) < 2:
            return 0.0, 0.0
        ang = np.array([quat_angle(q[i], q[i + 1]) for i in range(len(q) - 1)])
        lat = self.trace.lateral_step_mask(self.lateral_thresh)
        return float(ang[lat].sum()), float(ang.sum())

    def get_reward(self, env: Any) -> float:
        self.trace.record(env)
        raw = self.final_raw()
        return float(raw) if raw is not None else 0.0

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        """Absolute EE rotation (rad) accumulated while translating laterally.

        The earlier ratio form (on_lat / total) normalised the signal away: a
        naive run that barely rotates still spends most of its few milliradians
        during transit, so the share sat near the pref value (1-8_061: pref
        0.775 vs naive 0.743, cohen_d 1.67, raw ranges overlapping). The
        absolute in-transit angle is what the preference actually asks for
        ("turn the object while you carry it") and separates cleanly
        (061: 0.72 vs 0.15 rad, d=35.9; 114: 1.53 vs 0.41 rad, d=98.9, no
        overlap in either). Direction/normalisation is handled downstream by
        the dual anchors.
        """
        if self.trace.n < 2:
            return None
        on_lat, _total = self._split()
        return float(on_lat)

    @property
    def objects_of_interest(self) -> Set[str]:
        return set()

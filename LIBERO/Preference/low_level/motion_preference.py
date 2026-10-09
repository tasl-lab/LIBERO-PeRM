"""Base class + auto-registry for low-level motion preferences.

Every concrete subclass is auto-registered under snake_case(class_name - 'Preference')
into BaseMotionPreference._registry.

Contract:
- `reset(env)` is called once at the start of each episode (no-op by default).
- `get_reward(env) -> float [0, 1]` is called every step.
- `saturation` (class attr) normalizes episode-mean reward to [0, 100] via satisfaction().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set
import re


def _snake_name(cls_name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cls_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower().replace("_preference", "")


class BaseMotionPreference(ABC):
    saturation: float = 1.0
    # Preference tier: "low" = motion-level (speed/acc/distance/...);
    # "skill" = skill-level compliance (pick_left_edge / pick_handle / ...);
    # "profile" = user-profile / task_plan compliance.
    # Drives summary bucketing in PolicyEval/low_pref_reward.LowPrefAccumulator.
    level: str = "low"
    _registry: Dict[str, type] = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not getattr(cls, "__abstractmethods__", None):
            BaseMotionPreference._registry[_snake_name(cls.__name__)] = cls

    def __init__(self, pref_description: str = "", pos_description: str = ""):
        self.pref_description = pref_description
        self.pos_description = pos_description
        self.type = None
        # Per-mission demo anchors (raw reward on demos, via final_raw()).
        # Dual-anchor: naive_anchor = score-0 baseline, pref_anchor = score-100
        # target. satisfaction = clip((R - naive)/(pref - naive), 0, 1)*100;
        # sign(pref - naive) auto-encodes direction. naive=None → legacy single
        # clip(R/pref); both None → class-level `saturation`.
        self.pref_anchor: float | None = None
        self.naive_anchor: float | None = None

    def reset(self, env: Any) -> None:
        """Called once at episode start. Subclasses with per-step state override this."""
        pass

    @abstractmethod
    def get_reward(self, env: Any) -> float:
        ...

    @property
    @abstractmethod
    def objects_of_interest(self) -> Set[str]:
        ...

    @property
    def preference_type(self) -> str:
        return _snake_name(self.__class__.__name__)

    # 4-bucket mapping (default): satisfaction → {0.00, 0.25, 0.75, 1.00}.
    # Override `mapping_buckets` in subclasses if a different schedule fits.
    # Tuples are (lower_bound_inclusive, mapped_value); evaluated top-down.
    mapping_buckets = (
        (95.0, 1.00),
        (70.0, 0.75),
        (40.0, 0.25),
        (0.0,  0.00),
    )

    _ANCHOR_EPS = 1e-9  # |pref - naive| below this → degenerate, satisfaction=None

    def satisfaction(self, mean_reward: float) -> float | None:
        """Normalize episode raw reward (R = final_raw) to [0, 100], or None.

        Priority:
          1. Dual-anchor (pref + naive both set):
             clip((R - naive)/(pref - naive), 0, 1)*100. sign(pref - naive)
             auto-encodes direction (speed_up vs slow_down). |pref - naive| < eps
             → None (degenerate: metric can't discriminate).
          2. Single pref_anchor (legacy): clip(R / pref_anchor, 0, 1)*100.
          3. saturation (legacy class constant) > raw clip.
        """
        if self.pref_anchor is not None and self.naive_anchor is not None:
            denom = self.pref_anchor - self.naive_anchor
            if abs(denom) < self._ANCHOR_EPS:
                return None
            return float(min(max((mean_reward - self.naive_anchor) / denom, 0.0), 1.0) * 100.0)
        if self.pref_anchor is not None and self.pref_anchor > 0:
            return float(min(max(mean_reward / self.pref_anchor, 0.0), 1.0) * 100.0)
        s = self.saturation
        if s is None or s <= 0:
            return float(min(max(mean_reward, 0.0), 1.0) * 100.0)
        return float(min(mean_reward / s, 1.0) * 100.0)

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        """Override when the calibration anchor needs an episode-final scalar
        instead of mean per-step reward (e.g. SpeedRange's total step count).
        Default: return mean_reward as-is."""
        return mean_reward

    def mapped_satisfaction(self, mean_reward: float) -> float | None:
        """4-bucket mapping: satisfaction → {1.00, 0.75, 0.25, 0.00}, or None
        when satisfaction is invalid (degenerate dual-anchor).
        Buckets are descending lower-bounds; first match wins."""
        s = self.satisfaction(mean_reward)
        if s is None:
            return None
        for lo, val in self.mapping_buckets:
            if s >= lo:
                return float(val)
        return 0.0

    def __str__(self) -> str:
        return f"MotionPreference(type={self.preference_type}, desc='{self.pref_description}')"

    def obj_position(self) -> Dict[str, Any]:
        return {}

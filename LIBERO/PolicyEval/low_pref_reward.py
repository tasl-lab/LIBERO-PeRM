"""Load motion preferences from mission_info.yaml and compute per-step rewards.

Reuses Preference.pref_gen.create_motion_preference so the reward curves match
what the P_Bench dataset generation pipeline used.

Each preference has:
  - `reset(env)` called once at episode start
  - `get_reward(env) -> float in [0, 1]` called every step
  - `saturation` class attribute (used to normalize mean reward → satisfaction %)

Per-episode aggregation keeps per-step rewards for HDF5 logging; `summary()`
produces `{pref_type: {mean_reward, satisfaction, n_steps}}` via the class's
own `pref.satisfaction(mean_reward)` method (no external saturation dict).
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_LIBERO_ROOT = None


def _ensure_libero_on_path():
    global _LIBERO_ROOT
    if _LIBERO_ROOT is not None:
        return
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    libero_root = os.path.abspath(os.path.join(here, ".."))
    if libero_root not in sys.path:
        sys.path.insert(0, libero_root)
    _LIBERO_ROOT = libero_root


def load_motion_prefs(mission_info_yaml_path: str) -> List[Any]:
    """Read `low_prefs` from mission_info.yaml and build motion preference objects.

    If the yaml has a `low_pref_calibration: {pref_type: pref_anchor}` block
    (written by scripts/calibrate_low_pref.py from GT-pref demos), inject the
    anchor into each preference so `satisfaction()` reports score relative
    to the demo (D4RL-style normalised return).
    """
    _ensure_libero_on_path()
    from Preference.pref_gen import create_motion_preference  # type: ignore

    with open(mission_info_yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    entries = data.get("low_prefs") or []
    calibration = data.get("low_pref_calibration") or {}
    # task_plan objects → transport gating for prefs that support it (avoid).
    task_objects = [
        str(s[1]) for s in (data.get("task_plan") or [])
        if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]
    ]
    prefs: List[Any] = []
    for item in entries:
        p = create_motion_preference(
            preference_type=item["type"],
            preference_parameters=item["params"],
            pref_description=item.get("description", ""),
            pos_description=item.get("pos_description"),
        )
        if p is None:
            continue
        if hasattr(p, "set_task_context"):
            p.set_task_context(task_objects)
        if hasattr(p, "set_task_plan"):
            p.set_task_plan(data.get("task_plan"))
        cal = calibration.get(p.preference_type)
        if isinstance(cal, dict):
            pref_anchor, naive_anchor = cal.get("pref_anchor"), cal.get("naive_anchor")
            # per-subtask 锚（speed 家族：每 subtask 耗时的双锚列表）
            for attr in ("subtask_pref_anchor", "subtask_naive_anchor"):
                v = cal.get(attr)
                if isinstance(v, (list, tuple)) and hasattr(p, attr):
                    try:
                        setattr(p, attr, [float(x) for x in v])
                    except (TypeError, ValueError):
                        pass
        else:
            pref_anchor, naive_anchor = cal, None  # legacy: bare float = pref_anchor
        if pref_anchor is not None:
            try:
                p.pref_anchor = float(pref_anchor)
            except (TypeError, ValueError):
                pass
        if naive_anchor is not None:
            try:
                p.naive_anchor = float(naive_anchor)
            except (TypeError, ValueError):
                pass
        prefs.append(p)
    return prefs


class LowPrefAccumulator:
    """Per-episode accumulator. One instance per rollout episode."""

    def __init__(self, motion_prefs: List[Any]):
        self.prefs = motion_prefs
        self.step_rewards: Dict[str, List[float]] = {p.preference_type: [] for p in motion_prefs}
        self._reset_done = False
        self._step = 0

    def step(self, env: Any, action: Any, success: bool = False) -> Dict[str, float]:
        """Append one timestep; returns per-pref reward this step (for HDF5 logging).

        `success=True` on the step where the task first succeeds freezes every
        pref's episode trace there (post-success frames are still logged but
        never enter a raw)."""
        if not self._reset_done:
            for p in self.prefs:
                try:
                    p.reset(env)
                except Exception:
                    pass
            self._reset_done = True
        self._step += 1
        out: Dict[str, float] = {}
        for p in self.prefs:
            t = p.preference_type
            try:
                r = float(p.get_reward(env))
            except Exception:
                r = math.nan
            self.step_rewards[t].append(r)
            out[t] = r
        if success:
            for p in self.prefs:
                if hasattr(p, "mark_success"):
                    try:
                        p.mark_success()
                    except Exception:
                        pass
        return out

    def summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return {level: {pref_type: {mean_reward, satisfaction, n_steps}}}.

        level ∈ {"low", "skill", "profile"} (read from each pref's class
        attr, default "low"). Each tier is reported in its own bucket so
        rollout_summary.yaml can split them into separate top-level keys.
        """
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for p in self.prefs:
            t = p.preference_type
            level = getattr(p, "level", "low")
            bucket = result.setdefault(level, {})
            rews = [r for r in self.step_rewards.get(t, []) if not math.isnan(r)]
            if not rews:
                bucket[t] = {"mean_reward": None, "satisfaction": None, "n_steps": 0}
                continue
            mean_r = float(np.mean(rews))
            # Score against final_raw (identity for most metrics; SpeedRange →
            # total step count) so the base dual-anchor formula applies uniformly.
            raw_val = p.final_raw(mean_r)
            if raw_val is None:
                raw_val = mean_r
            sat = p.satisfaction(raw_val)
            mapped = p.mapped_satisfaction(raw_val)
            bucket[t] = {
                "mean_reward": mean_r,
                "satisfaction": float(sat) if sat is not None else None,
                "mapped": float(mapped) if mapped is not None else None,
                "n_steps": len(rews),
            }
        return result

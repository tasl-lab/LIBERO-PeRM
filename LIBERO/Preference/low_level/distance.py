"""
Given a env, return the distance to certain object\area at that time step
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, Tuple

from Utility.tool_func import to_sim_env_name
from .motion_preference import BaseMotionPreference


def _resolve_from(env, spec):
	if isinstance(spec, str):
		if spec in ("gripper", "eef"):
			try:
				eef_name = getattr(env.robots[0], "eef_name", "gripper0_eef")
				return env.sim.data.body_xpos[env.sim.model.body_name2id(eef_name)]
			except Exception:
				# Fallback: try as site name
				return env.sim.data.get_site_xpos(eef_name)
	# Default: use spec as a position
	return np.array(spec, dtype=float)


def _resolve_to(env, spec):
	# Dict spec with explicit type
    if isinstance(spec, dict):
        type_ = spec.get("type")
        name = spec.get("name")
        name=to_sim_env_name(name)
        if type_ == "site":
            return env.sim.data.get_site_xpos(name)
        elif type_ in ("body", "object"):
            try:
                bid = env.sim.model.body_name2id(name)
                return env.sim.data.body_xpos[bid]
            except Exception:
                # Fallback to logical object id mapping if available
                bid = env.obj_body_id[name]
                return env.sim.data.body_xpos[bid]
        else:
            raise ValueError(f"Unsupported to.type '{type_}'")
    # String: try site, then body, then obj_body_id
    name = spec
    try:
        return env.sim.data.get_site_xpos(name)
    except Exception:
        pass
    try:
        bid = env.sim.model.body_name2id(name)
        return env.sim.data.body_xpos[bid]
    except Exception:
        # logical object mapping
        bid = env.obj_body_id[name]
        return env.sim.data.body_xpos[bid]


# TODO, please define required parameters first
def distance_to_object():
    """
    Calculates a reward based on the distance between the gripper and a target object.
    The reward is shaped by a metric ('less_than' or 'greater_than') and a threshold.
    """
    def reward(env, params):
        target_name = params.get("target")
        metric = params.get("metric", "less_than")


        threshold = params.get("threshold", 0.05)

        if not target_name:
            raise ValueError("params['target'] is required for distance_to_object")

        # Resolve gripper and target positions

        gripper_pos = _resolve_from(env, "gripper")
        special_obj=["ground_1"]
        if target_name not in special_obj:
            target_pos = _resolve_to(env, {"type": "object", "name": target_name})
        else:
            target_pos=gripper_pos.copy()
            target_pos [2] = 0.0

        
        if gripper_pos is None or target_pos is None:
            return 0.0 # Cannot compute reward if positions are invalid

        # Calculate Euclidean distance
        dist = np.linalg.norm(np.asarray(gripper_pos, dtype=float) - np.asarray(target_pos, dtype=float))

        # Calculate reward based on the metric
        if metric == "less_than":
            # Reward is 1.0 if distance is less than threshold, falls off otherwise
            if dist < threshold:
                return 1.0
            return 1.0 / (1.0 + (dist - threshold))
        elif metric == "greater_than":
            # Reward is 1.0 if distance is greater than threshold, falls off otherwise
            if dist > threshold:
                return 1.0
            return dist / threshold
        else:
            # Default to simple inverse distance for backward compatibility or unknown metrics
            return 1.0 / (1.0 + dist)

    return reward


# "Keep away from X" 讲的是绕开, 不是从上方飞过。EE 在搬运段一律抬到 base+0.2 m 以上,
# 3D 距离里 z 差就占了 0.15-0.25 m, 把水平绕行淹没掉 (1-4_034: 3D ratio 1.06x, 水平 1.36x)。
AVOID_HORIZONTAL_ONLY = True


def _planar(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.linalg.norm((a - b)[:2] if AVOID_HORIZONTAL_ONLY else (a - b)))


def _gripper_object_distance(env: Any, target_name: str) -> float | None:
    """Gripper→object distance in meters (horizontal when AVOID_HORIZONTAL_ONLY);
    None if unresolvable. Mirrors the position resolution inside distance_to_object()."""
    try:
        gripper_pos = _resolve_from(env, "gripper")
        if target_name == "ground_1":
            target_pos = np.asarray(gripper_pos, dtype=float).copy()
            target_pos[2] = 0.0
        else:
            target_pos = _resolve_to(env, {"type": "object", "name": target_name})
    except Exception:
        return None
    if gripper_pos is None or target_pos is None:
        return None
    return _planar(gripper_pos, target_pos)


class ApproachToObjectPreference(BaseMotionPreference):
    saturation = 0.2325

    def __init__(
        self,
        target_obj: str,
        threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_obj = target_obj
        self.threshold = threshold
        self._reward_func = distance_to_object()
        self.type="ApproachToObjectPreference"

    def get_reward(self, env: Any) -> float:
        params = {
            "target": self.target_obj,
            "metric": "less_than",
            "threshold": self.threshold,
        }
        return self._reward_func(env, params)

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.target_obj}



class AvoidToObjectPreference(BaseMotionPreference):
    saturation = 0.2325
    # A step is "transport" when the EE is farther than this from every task
    # object — approaches mandated by the task itself (reach the bowl, reach
    # the drawer) are excluded so the avoid raw only reflects path choice.
    TRANSPORT_TASK_ZONE_M: float = 0.15

    def __init__(
        self,
        target_obj: str,
        threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_obj = target_obj
        self.threshold = threshold
        self._reward_func = distance_to_object()
        self.type="AvoidToObjectPreference"
        # Task-plan objects (set via set_task_context): transport gating
        # fallback + candidates for "the object currently carried".
        self.task_objects: list[str] = []
        from .episode_trace import EpisodeTrace
        self.trace = EpisodeTrace()
        # Per-step closest approach to the avoid target: EE, and the carried
        # task object (a task object within HELD_ZONE_M of the EE), if any.
        self._d_ee: list[float] = []
        self._d_obj: list[float] = []
        self._zone_ok: list[bool] = []   # legacy task-zone transport gate

    HELD_ZONE_M: float = 0.06   # task object this close to the EE counts as carried

    def set_task_context(self, task_objects) -> None:
        """Inject task_plan objects (from mission_info) for transport gating."""
        self.task_objects = [o for o in (task_objects or []) if o and o != self.target_obj]

    def reset(self, env: Any) -> None:
        self.trace.reset()
        self._d_ee, self._d_obj, self._zone_ok = [], [], []

    def mark_success(self) -> None:
        self.trace.mark_success()

    def _in_transport(self, env: Any) -> bool:
        """Legacy gate: EE outside every task object's mandated-approach zone."""
        for obj in self.task_objects:
            d = _gripper_object_distance(env, obj)
            if d is not None and d <= self.TRANSPORT_TASK_ZONE_M:
                return False
        return True

    def _carried_object_distance(self, env: Any) -> float:
        """Distance from the carried task object (if any) to the avoid target."""
        best = float("inf")
        for obj in self.task_objects:
            d_ee = _gripper_object_distance(env, obj)
            if d_ee is None or d_ee > self.HELD_ZONE_M:
                continue
            try:
                p_obj = _resolve_to(env, {"type": "object", "name": obj})
                p_tgt = _resolve_to(env, {"type": "object", "name": self.target_obj})
                d = _planar(p_obj, p_tgt)
                best = min(best, d)
            except Exception:
                continue
        return best

    def get_reward(self, env: Any) -> float:
        self.trace.record(env)
        dist = _gripper_object_distance(env, self.target_obj)
        self._d_ee.append(dist if dist is not None else float("inf"))
        self._d_obj.append(self._carried_object_distance(env))
        self._zone_ok.append(self._in_transport(env))
        params = {
            "target": self.target_obj,
            "metric": "greater_than",
            "threshold": self.threshold,
        }
        return self._reward_func(env, params)

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        """Raw = MAX of min(EE→target, carried_obj→target) over the carry
        segment [grasp, final-descent start]: how far the transport swings
        away from the avoid target.

        Why max, not min: in the SinglePref layouts the grasp point and the
        place point are both closer to the obstacle than any clearance the
        detour can achieve (1-4_034: bowl 0.23 m, cabinet 0.17 m from the
        bottle), so the min is pinned at the endpoints for pref and naive
        alike (0.20 vs 0.21). The excursion separates them (0.37 vs 0.31,
        AUROC 0.98 on 100 vs 10 demos). Fallbacks: legacy task-zone
        transport gate → ungated minimum (no carry detected)."""
        n = self.trace.t_end
        if n == 0:
            return mean_reward
        d = np.minimum(np.asarray(self._d_ee[:n]), np.asarray(self._d_obj[:n]))
        finite = np.isfinite(d)
        if not finite.any():
            return mean_reward
        win = self.trace.carry_window()
        if win is not None:
            c0, c1 = win[0], min(win[1], n - 1)
            seg_ok = finite[c0:c1 + 1]
            if seg_ok.any():
                return float(d[c0:c1 + 1][seg_ok].max())
        zone = np.asarray(self._zone_ok[:n], dtype=bool)
        if (zone & finite).any():
            return float(d[zone & finite].min())
        return float(d[finite].min())

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.target_obj}

"""Profile-tier discrete metric: did the rollout take the PREFERRED branch where
task_plan and naive_plan diverge?

Series-3 preferences ("prefer green objects", "upper drawer for food", "serve
double portion", ...) do not change how a skill is executed; they change WHICH
object is manipulated or WHERE it is placed. The two plans in mission_info.yaml
differ in exactly that branch, so the metric watches only the branch and gives
one discrete verdict at the node where the choice becomes physical:

  mode "object"  (pick objects differ)  node = the first time any candidate
       object (pref-only or naive-only pick) rises LIFT_THRESH_M above its
       start height. raw = 1 if that object is a pref-only pick, 0 if it is a
       naive-only pick. If the episode ends without a candidate lift
       (`mark_episode_end`), node = end, raw = 0. A "serve double portion" plan
       has a pref-only pick and no naive-only pick, so the same rule reads:
       100 iff the extra object was lifted.
  mode "target"  (pick objects equal, put targets differ)  node = release of
       the carried object (gripper gap reopens after the grasp, see
       skill_pose._GraspEventTrace) or episode end. raw = 1 if the carried
       object is then closer to a pref-only target region than to any
       naive-only one, else 0.

satisfaction = 100 if raw >= 0.5 else 0, None before the node, held after —
the same interface as the skill-tier metrics (skill_pose._SkillPosePreference)
so calibrate_low_pref / render_demo_video / skill_contact_sheet reuse as-is.

Build it with `load_plan_choice(mission_info_yaml)` or declare it in yaml
(scripts/wire_plan_choice.py writes the params from the two plans):

    low_prefs:
    - type: plan_choice
      params: {mode: object, pref_objs: [green_cream_cheese_1], naive_objs: [cream_cheese_1]}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from Preference.low_level.motion_preference import BaseMotionPreference
from Preference.low_level.skill_pose import _GraspEventTrace, _body_pos, LIFT_THRESH_M

_LEVEL_SITE = {"_cabinet_top": "_top_region", "_cabinet_middle": "_middle_region",
               "_cabinet_bottom": "_bottom_region", "_top_side": "_top_side"}


def _region_pos(env: Any, name: str) -> Optional[np.ndarray]:
    """World position of a put target: cabinet level -> its `<base>_<level>_region`
    site (the drawer bodies all sit at the cabinet origin), else the body."""
    sim = env.sim
    for suf, site_suf in _LEVEL_SITE.items():
        if name.endswith(suf):
            site = name[: -len(suf)] + site_suf
            try:
                return np.asarray(sim.data.get_site_xpos(site), dtype=np.float64).copy()
            except Exception:
                break
    p = _body_pos(env, name)
    return None if not np.any(p) else p


def plan_branch(task_plan: Sequence, naive_plan: Sequence) -> Optional[Dict[str, Any]]:
    """Diff the two plans -> plan_choice params (None when the plans do not diverge)."""
    def steps(plan):
        return [(str(s[0]), str(s[1])) for s in (plan or []) if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]]
    P, N = steps(task_plan), steps(naive_plan)
    pick = lambda st: [o for sk, o in st if sk.startswith("pick")]
    put = lambda st: [o for sk, o in st if sk.startswith("put")]
    pp, npk = pick(P), pick(N)
    pref_objs = [o for o in dict.fromkeys(pp) if o not in npk]
    naive_objs = [o for o in dict.fromkeys(npk) if o not in pp]
    if pref_objs or naive_objs:
        return {"mode": "object", "pref_objs": pref_objs, "naive_objs": naive_objs}
    pt, nt = put(P), put(N)
    pref_t = [o for o in dict.fromkeys(pt) if o not in nt]
    naive_t = [o for o in dict.fromkeys(nt) if o not in pt]
    if pref_t or naive_t:
        carried = pp[0] if pp else None
        return {"mode": "target", "carried": carried, "pref_targets": pref_t, "naive_targets": naive_t}
    return None


class PlanChoicePreference(BaseMotionPreference):
    level = "profile"
    CRITERION_THRESHOLD = 0.5
    CRITERION_DIRECTION = 1

    def __init__(self, mode: str, pref_objs: Sequence[str] = (), naive_objs: Sequence[str] = (),
                 carried: Optional[str] = None, pref_targets: Sequence[str] = (),
                 naive_targets: Sequence[str] = (), pref_description: str = "",
                 pos_description: str = "", criterion_threshold: float | None = None,
                 criterion_direction: int | None = None) -> None:
        super().__init__(pref_description, pos_description)
        self.mode = str(mode).lower()
        if self.mode not in ("object", "target"):
            raise ValueError(f"mode must be object/target, got {mode!r}")
        self.pref_objs = [str(o) for o in pref_objs]
        self.naive_objs = [str(o) for o in naive_objs]
        self.carried = str(carried) if carried else None
        self.pref_targets = [str(o) for o in pref_targets]
        self.naive_targets = [str(o) for o in naive_targets]
        if self.mode == "object" and not (self.pref_objs or self.naive_objs):
            raise ValueError("object mode needs pref_objs and/or naive_objs")
        if self.mode == "target" and not (self.carried and (self.pref_targets or self.naive_targets)):
            raise ValueError("target mode needs carried + pref_targets/naive_targets")
        self.criterion_threshold = (float(criterion_threshold) if criterion_threshold is not None
                                    else self.CRITERION_THRESHOLD)
        self.criterion_direction = (int(criterion_direction) if criterion_direction is not None
                                    else self.CRITERION_DIRECTION)
        self.type = "PlanChoicePreference"
        self._reset_state()

    # ─── episode state ───────────────────────────────────────────────
    def _reset_state(self) -> None:
        self._n = 0
        self._z0: Dict[str, float] = {}
        self._node: Optional[int] = None      # step of the verdict
        self._chosen: Optional[str] = None    # object lifted first / region nearest at release
        self._raw: Optional[float] = None
        self._episode_done = False
        self._success_marked = False
        self.trace = _GraspEventTrace()

    def reset(self, env: Any) -> None:
        self._reset_state()

    def mark_success(self) -> None:
        self._success_marked = True

    def mark_episode_end(self) -> None:
        self._episode_done = True
        self._settle_at_end()

    def _settle_at_end(self) -> None:
        if self._node is not None or self._n == 0:
            return
        # object mode: no candidate lift -> the branch was not taken (0).
        # target mode: no release registered -> judge where the object ended up.
        self._node = self._n - 1
        if self.mode == "target":
            self._chosen = self._nearest_target_cached
            self._raw = 1.0 if self._chosen in self.pref_targets else 0.0
        else:
            self._raw = 0.0

    # ─── per-step ─────────────────────────────────────────────────────
    @property
    def _candidates(self) -> List[str]:
        return self.pref_objs + self.naive_objs

    def get_reward(self, env: Any) -> float:
        self._n += 1
        t = self._n - 1
        if self.mode == "object":
            for o in self._candidates:
                z = float(_body_pos(env, o)[2])
                if o not in self._z0:
                    self._z0[o] = z
                if self._node is None and z - self._z0[o] > LIFT_THRESH_M:
                    self._node, self._chosen = t, o
                    self._raw = 1.0 if o in self.pref_objs else 0.0
                    break
        else:
            self.trace.record(env, self.carried)
            self._nearest_target_cached = self._nearest_target(env)
            if self._node is None:
                tr = self.trace.t_release()
                if tr is not None:
                    self._node = t
                    self._chosen = self._nearest_target_cached
                    self._raw = 1.0 if self._chosen in self.pref_targets else 0.0
        return float(self._raw) if self._raw is not None else 0.0

    _nearest_target_cached: Optional[str] = None

    def _nearest_target(self, env: Any) -> Optional[str]:
        p = _body_pos(env, self.carried)
        best, bd = None, float("inf")
        for tname in self.pref_targets + self.naive_targets:
            q = _region_pos(env, tname)
            if q is None:
                continue
            d = float(np.linalg.norm(p - q))
            if d < bd:
                best, bd = tname, d
        return best

    # ─── verdict ──────────────────────────────────────────────────────
    def node_step(self) -> Optional[int]:
        return self._node

    def node_reached(self) -> bool:
        return self._node is not None

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        return None if self._raw is None else float(self._raw)

    def satisfaction(self, mean_reward: float | None) -> float | None:
        raw = self.final_raw()
        if raw is None:
            return None
        return 100.0 if self.criterion_direction * (raw - self.criterion_threshold) >= 0 else 0.0

    def raw(self) -> Dict[str, Any]:
        return {"mode": self.mode, "node": self._node, "chosen": self._chosen, "raw": self._raw}

    @property
    def objects_of_interest(self) -> Set[str]:
        s = set(self._candidates) | set(self.pref_targets) | set(self.naive_targets)
        if self.carried:
            s.add(self.carried)
        return s


def load_plan_choice(mission_info_yaml: str, pref_description: str = "") -> Optional[PlanChoicePreference]:
    import yaml
    with open(mission_info_yaml) as f:
        d = yaml.safe_load(f) or {}
    params = plan_branch(d.get("task_plan"), d.get("naive_plan"))
    if params is None:
        return None
    return PlanChoicePreference(pref_description=pref_description or str(d.get("high_pref_lan", "")), **params)

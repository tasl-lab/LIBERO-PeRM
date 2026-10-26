"""User-profile-level metric: monitor whether a rollout follows the task_plan
object sequence declared in mission_info.yaml.

Two independent observation streams drive the metric:

  1. ``touched_seq`` — ordered list of objects the gripper has approached.
     Detected per step by ``dist(eef, obj) < TOUCH_DIST_M``. Object is appended
     the first time the threshold is met (no duplicates).
  2. ``manipulated_set`` — set of objects whose pose at any timestep differs
     from the initial pose by more than ``move_thresh_m`` (2cm) or
     ``rot_thresh_rad`` (5°). Updated each step.

Score combines progress (did each expected object get touched / moved?) and
order-match (did touches happen in the planned sequence?).

  progress_i ∈ {0.0, 0.5, 1.0}:
      0.0  if obj_i ∉ touched_seq and ∉ manipulated_set
      0.5  if obj_i in exactly one of {touched_seq, manipulated_set}
      1.0  if obj_i in both

  order_match = 1.0 if touched_seq is a subsequence of expected_seq_unique
              = LCS(touched_seq, expected_seq_unique) / N otherwise

  score = 0.6 · mean(progress_i) + 0.4 · order_match

Spurious-first detection:
  ``spurious_first_obj`` = the first object outside ``expected_set`` that
  enters ``manipulated_set`` *before* any expected object — flags "policy
  manipulated the wrong thing first".

Registration:
  Auto-registered into ``BaseMotionPreference._registry`` under the snake-case
  key ``task_plan_progress``. Distinct from low-level metrics: it does not
  read ``low_prefs`` from yaml — call ``load_task_plan_progress`` to build it
  directly from ``task_plan``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from Preference.low_level.motion_preference import BaseMotionPreference


TOUCH_DIST_M = 0.05
MOVE_THRESH_M = 0.05  # 5cm, 高于 settle / transient bump 噪声
ROT_THRESH_RAD = math.radians(15.0)  # 15°, 同上

# Region/sub-body suffixes that share the parent body's geoms (cabinets, drawers,
# regions). Stripping these gives the base name used to find geoms.
_REGION_SUFFIXES = (
    "_cabinet_top", "_cabinet_middle", "_cabinet_bottom",
    "_top_side", "_top_region", "_middle_region", "_bottom_region",
    "_right_contain_region", "_left_contain_region",
    "_front_contain_region", "_back_contain_region", "_contain_region",
    "_button", "_cook_region", "_base",
)


def _strip_region_suffix(name: str) -> str:
    for s in _REGION_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def _is_subsequence(actual: List[str], expected: List[str]) -> bool:
    j = 0
    for x in actual:
        if j < len(expected) and x == expected[j]:
            j += 1
    return j == len(expected)


class TaskPlanProgressPreference(BaseMotionPreference):
    """Monitor task_plan execution against gripper-touch and object-motion."""

    saturation = 1.0
    level = "profile"  # tier: user-profile compliance

    def __init__(
        self,
        task_plan: List[Tuple[str, str]],
        pref_description: str = "",
        pos_description: str = "",
        touch_dist_m: float = TOUCH_DIST_M,
        move_thresh_m: float = MOVE_THRESH_M,
        rot_thresh_rad: float = ROT_THRESH_RAD,
    ):
        super().__init__(pref_description, pos_description)
        # Keep object-only sequence; preserve duplicates for completeness reporting.
        self.expected_seq: List[str] = [
            step[1] for step in task_plan if len(step) >= 2 and step[1]
        ]
        self.expected_seq_unique: List[str] = list(dict.fromkeys(self.expected_seq))
        self.expected_set: Set[str] = set(self.expected_seq)
        self.touch_dist_m = float(touch_dist_m)
        self.move_thresh_m = float(move_thresh_m)
        self.rot_thresh_rad = float(rot_thresh_rad)
        self.type = "TaskPlanProgressPreference"

        self._reset_state()

    def _reset_state(self) -> None:
        self._init_pos: Dict[str, np.ndarray] = {}
        self._init_quat: Dict[str, np.ndarray] = {}
        self._touched_seq: List[str] = []
        self._touched_set: Set[str] = set()
        self._manipulated_set: Set[str] = set()
        self._spurious_first_obj: Optional[str] = None
        self._step_count: int = 0
        # Geom id caches; populated lazily on first reset(env).
        self._obj_geom_ids: Dict[str, Set[int]] = {}
        self._gripper_geom_ids: Set[int] = set()

    def reset(self, env: Any) -> None:
        self._reset_state()
        for obj in self._candidate_obj_names(env):
            try:
                self._init_pos[obj] = self._obj_pos(env, obj).copy()
                self._init_quat[obj] = self._obj_quat(env, obj).copy()
            except Exception:
                pass
        self._build_geom_caches(env)

    def get_reward(self, env: Any) -> float:
        self._step_count += 1
        # Touched: gripper-contact OR contact with another expected obj OR
        # close spatial proximity to gripper / another expected obj.
        # Proximity fallback handles cases where mesh-collision contact misses
        # legitimate "object placed in container" interactions.
        contacts_for = self._scan_contacts(env)
        eef_pos = self._eef_pos(env)
        for obj in self.expected_set:
            if obj in self._touched_set:
                continue
            triggered = False
            other_set = contacts_for.get(obj)
            if other_set and ("__gripper__" in other_set or (other_set & self.expected_set)):
                triggered = True
            if not triggered:
                # Proximity check: gripper close OR any other expected obj close.
                try:
                    p = self._obj_pos(env, obj)
                except Exception:
                    p = None
                if p is not None:
                    if eef_pos is not None and float(np.linalg.norm(p - eef_pos)) <= self.touch_dist_m:
                        triggered = True
                    else:
                        for other in self.expected_set:
                            if other == obj:
                                continue
                            try:
                                q = self._obj_pos(env, other)
                            except Exception:
                                continue
                            # "placed on/in" semantics: tight XY alignment
                            # (≤0.10m) AND vertical separation ≤0.25m. Catches
                            # OJ inside basket while rejecting neighbouring
                            # plates that sit beside the target.
                            xy = float(np.linalg.norm((p - q)[:2]))
                            dz = float(abs((p - q)[2]))
                            if xy <= 0.10 and dz <= 0.25:
                                triggered = True
                                break
            if triggered:
                self._touched_set.add(obj)
                self._touched_seq.append(obj)
        # Spurious-first: gripper actively contacted an unexpected obj before
        # any expected obj entered touched_set. Filter out aliases — bodies
        # whose geom-id set overlaps any expected obj's (e.g. cabinet base
        # shares geoms with cabinet_middle).
        # Manipulated: scan all tracked objs for pose delta.
        for obj, p0 in self._init_pos.items():
            if obj in self._manipulated_set:
                continue
            try:
                p = self._obj_pos(env, obj)
                moved = float(np.linalg.norm(p - p0)) > self.move_thresh_m
            except Exception:
                moved = False
            if not moved:
                q0 = self._init_quat.get(obj)
                if q0 is not None:
                    try:
                        q = self._obj_quat(env, obj)
                        dot = abs(float(np.dot(q0, q)))
                        dot = min(1.0, max(-1.0, dot))
                        moved = (2.0 * float(np.arccos(dot))) > self.rot_thresh_rad
                    except Exception:
                        moved = False
            if moved:
                self._manipulated_set.add(obj)
        # Spurious-first: gripper actively moved an unexpected obj before any
        # expected obj entered touched_set. Requires (a) gripper-contact AND
        # (b) obj in manipulated_set (real move, not brush-past). Filters out
        # alias bodies that share a base name with an expected obj.
        if self._spurious_first_obj is None and not self._touched_set:
            expected_bases: Set[str] = {_strip_region_suffix(eo) for eo in self.expected_set}
            for obj, neighbours in contacts_for.items():
                if obj in self.expected_set:
                    continue
                if "__gripper__" not in neighbours:
                    continue
                if _strip_region_suffix(obj) in expected_bases:
                    continue
                if obj not in self._manipulated_set:
                    continue
                self._spurious_first_obj = obj
                break
        return float(self.compute_score())

    def satisfaction(self, mean_reward: float) -> float:
        """Override: progress is cumulative, so report final compute_score
        rather than the running-mean. Ignores the input mean_reward."""
        return float(min(self.compute_score(), 1.0) * 100.0)

    def compute_score(self) -> float:
        """Satisfaction-based score: each expected obj counted iff it was
        touched (gripper / fellow-expected-obj contact / proximity) OR moved.
        Order is not enforced — re-grasps and detours are allowed; only the
        final coverage matters. order_match / spurious_first_obj are still
        reported in raw() as diagnostics."""
        if not self.expected_seq_unique:
            return 1.0
        progress_per = self._progress_per_subtask()
        return sum(progress_per) / len(progress_per)

    def _progress_per_subtask(self) -> List[float]:
        """Per-expected-obj progress:
        1.0 if obj was contacted (gripper or other expected obj) OR manipulated;
        0.0 otherwise.
        Contact-scan is itself a strong signal — fixtures (basket / cabinet)
        receive contact via the placed object even when their body barely moves.
        """
        out: List[float] = []
        for obj in self.expected_seq_unique:
            in_t = obj in self._touched_set
            in_m = obj in self._manipulated_set
            out.append(1.0 if (in_t or in_m) else 0.0)
        return out

    def _order_score(self) -> float:
        if not self.expected_seq_unique:
            return 1.0
        if _is_subsequence(self._touched_seq, self.expected_seq_unique):
            return 1.0
        return _lcs_len(self._touched_seq, self.expected_seq_unique) / float(
            len(self.expected_seq_unique)
        )

    def raw(self) -> Dict[str, Any]:
        return {
            "expected_seq": list(self.expected_seq),
            "expected_seq_unique": list(self.expected_seq_unique),
            "touched_seq": list(self._touched_seq),
            "manipulated_set": sorted(self._manipulated_set),
            "manipulated_unexpected": sorted(self._manipulated_set - self.expected_set),
            "progress_per_subtask": self._progress_per_subtask(),
            "order_match": float(self._order_score()),
            "spurious_first_obj": self._spurious_first_obj,
            "n_steps": int(self._step_count),
        }

    @property
    def objects_of_interest(self) -> Set[str]:
        return set(self.expected_set)

    # ─── env adapters ───────────────────────────────────────────────────

    def _eef_pos(self, env: Any) -> Optional[np.ndarray]:
        try:
            eef_name = getattr(env.robots[0], "eef_name", "gripper0_eef")
            bid = env.sim.model.body_name2id(eef_name)
            return np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32)
        except Exception:
            try:
                eef_name = getattr(env.robots[0], "eef_name", "gripper0_eef")
                return np.asarray(env.sim.data.get_site_xpos(eef_name), dtype=np.float32)
            except Exception:
                return None

    def _candidate_obj_names(self, env: Any) -> List[str]:
        try:
            return list(env.obj_body_id.keys())
        except Exception:
            return list(self.expected_set)

    def _obj_pos(self, env: Any, obj_name: str) -> np.ndarray:
        bid = env.obj_body_id[obj_name]
        return np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32)

    def _obj_quat(self, env: Any, obj_name: str) -> np.ndarray:
        bid = env.obj_body_id[obj_name]
        return np.asarray(env.sim.data.body_xquat[bid], dtype=np.float32)

    # ─── contact scanning ──────────────────────────────────────────────

    def _build_geom_caches(self, env: Any) -> None:
        """Cache geom-id sets for each tracked obj + the gripper.

        Identifies geoms by name-prefix matching (robosuite convention:
        ``<obj_name>_g<i>`` and gripper geoms contain ``finger`` or ``gripper``).
        """
        try:
            sim = env.sim
            ngeom = sim.model.ngeom
            geom_names: List[Optional[str]] = []
            for gid in range(ngeom):
                try:
                    geom_names.append(sim.model.geom_id2name(gid))
                except Exception:
                    geom_names.append(None)
            tracked_objs = set(self._init_pos.keys()) | self.expected_set
            for obj in tracked_objs:
                base = _strip_region_suffix(obj)
                # Match geoms named "<base>_g..." or "<base>_main..." or just "<base>".
                gids = {
                    gid for gid, nm in enumerate(geom_names)
                    if nm and (nm == base or nm.startswith(base + "_g") or nm.startswith(base + "_main"))
                }
                self._obj_geom_ids[obj] = gids
            self._gripper_geom_ids = {
                gid for gid, nm in enumerate(geom_names)
                if nm and ("finger" in nm.lower() or "gripper" in nm.lower())
            }
        except Exception:
            self._obj_geom_ids = {}
            self._gripper_geom_ids = set()

    def _scan_contacts(self, env: Any) -> Dict[str, Set[str]]:
        """Return {obj_name: {other_name | '__gripper__'}} for current step.

        A single geom may belong to multiple logical obj names (e.g. cabinet
        geoms shared by `wooden_cabinet_1`, `wooden_cabinet_1_base`,
        `wooden_cabinet_1_cabinet_middle`). Reverse map is set-valued so we
        propagate contacts to *all* matching logical names.
        """
        out: Dict[str, Set[str]] = {}
        try:
            sim = env.sim
            ncon = int(sim.data.ncon)
        except Exception:
            return out
        if ncon <= 0 or not self._obj_geom_ids:
            return out
        gid_to_objs: Dict[int, Set[str]] = {}
        for obj, gids in self._obj_geom_ids.items():
            for gid in gids:
                gid_to_objs.setdefault(gid, set()).add(obj)
        gripper = self._gripper_geom_ids
        for i in range(ncon):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            objs1 = gid_to_objs.get(g1, set())
            objs2 = gid_to_objs.get(g2, set())
            in_g1 = g1 in gripper
            in_g2 = g2 in gripper
            for o1 in objs1:
                for o2 in objs2:
                    if o1 == o2:
                        continue
                    out.setdefault(o1, set()).add(o2)
                    out.setdefault(o2, set()).add(o1)
                if in_g2:
                    out.setdefault(o1, set()).add("__gripper__")
            for o2 in objs2:
                if in_g1:
                    out.setdefault(o2, set()).add("__gripper__")
        return out


LIFT_THRESH_M = 0.03  # obj z rise above init to count as "lifted" (< MOVE_THRESH_M)


class TaskPlanCompliancePreference(TaskPlanProgressPreference):
    """User-profile compliance metric — per-subtask 2-stage progress, mean.

    Upgrade over the coarse parent (per-object touched/moved coverage). Walks the
    ``task_plan`` as K ordered subtasks ``(skill, obj)`` and gives each a 2-stage
    partial credit, then averages (mean). Subclasses the parent purely to reuse
    its env-adapters / ``_scan_contacts`` / geom caches — only the scoring changes.

    Per subtask credit ∈ {0.0, 0.5, 1.0}:
      pick X : engaged   = X touched (parent gripper-contact / proximity logic)
               completed = X grasped (``__gripper__`` contact) AND lifted
                           (cur_z − init_z > LIFT_THRESH_M)
      put  Y : carried   = obj of the nearest preceding pick subtask
               engaged   = carried placed-near Y (xy≤0.10 & dz≤0.25)
               completed = carried placed-near Y AND gripper no longer contacts
                           carried (released)
               (carried None → degrade to contact-binary on Y: 0/1)
      other  : contact-binary on obj (engaged==completed==touched, 0/1)

    score = mean over subtasks (empty → 1.0). order_match / spurious_first_obj
    stay in ``raw()`` as diagnostics only (progress-only score). ``satisfaction``
    is inherited (cumulative compute_score×100).
    """

    saturation = 1.0
    level = "profile"

    def __init__(
        self,
        task_plan: List[Tuple[str, str]],
        pref_description: str = "",
        pos_description: str = "",
        touch_dist_m: float = TOUCH_DIST_M,
        move_thresh_m: float = MOVE_THRESH_M,
        rot_thresh_rad: float = ROT_THRESH_RAD,
        lift_thresh_m: float = LIFT_THRESH_M,
    ):
        # Keep full (skill, obj) subtasks — parent keeps objects only.
        self.subtasks: List[Tuple[str, str]] = [
            (str(s[0]), str(s[1]))
            for s in task_plan
            if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]
        ]
        # Static map: each put subtask's "carried" obj = nearest preceding pick obj.
        self._carried: List[Optional[str]] = []
        last_pick: Optional[str] = None
        for skill, obj in self.subtasks:
            if skill == "pick":
                last_pick = obj
                self._carried.append(None)
            elif skill == "put":
                self._carried.append(last_pick)
            else:
                self._carried.append(None)
        self.lift_thresh_m = float(lift_thresh_m)
        super().__init__(
            task_plan, pref_description, pos_description,
            touch_dist_m, move_thresh_m, rot_thresh_rad,
        )
        self.type = "TaskPlanCompliancePreference"

    def _reset_state(self) -> None:
        super()._reset_state()
        n = len(getattr(self, "subtasks", []))
        self._sub_engaged: List[bool] = [False] * n
        self._sub_completed: List[bool] = [False] * n

    def get_reward(self, env: Any) -> float:
        # Reuse parent for touched/manipulated/spurious diagnostics + step count.
        # Its return value (child compute_score, one step stale) is discarded.
        super().get_reward(env)
        contacts_for = self._scan_contacts(env)
        for i, (skill, obj) in enumerate(self.subtasks):
            if skill == "pick":
                self._update_pick(env, i, obj, contacts_for)
            elif skill == "put":
                self._update_put(env, i, obj, contacts_for)
            else:
                self._update_other(i, obj, contacts_for)
        return float(self.compute_score())

    # ─── per-subtask latch updates ─────────────────────────────────────
    def _update_pick(self, env: Any, i: int, obj: str, contacts_for) -> None:
        if obj in self._touched_set:
            self._sub_engaged[i] = True
        if self._sub_completed[i]:
            return
        if "__gripper__" not in contacts_for.get(obj, set()):
            return
        try:
            z_now = float(self._obj_pos(env, obj)[2])
            z0 = float(self._init_pos[obj][2])
        except Exception:
            return
        if (z_now - z0) > self.lift_thresh_m:
            self._sub_completed[i] = True

    def _update_put(self, env: Any, i: int, target: str, contacts_for) -> None:
        carried = self._carried[i]
        if carried is None:
            # Degrade: contact-binary on the target object.
            if target in self._touched_set:
                self._sub_engaged[i] = True
                self._sub_completed[i] = True
            return
        try:
            p = self._obj_pos(env, carried)
            q = self._obj_pos(env, target)
        except Exception:
            return
        # "placed on/in" semantics — inlined from parent get_reward (xy≤0.10, dz≤0.25).
        xy = float(np.linalg.norm((p - q)[:2]))
        dz = float(abs((p - q)[2]))
        placed_near = xy <= 0.10 and dz <= 0.25
        if placed_near:
            self._sub_engaged[i] = True
        if self._sub_completed[i]:
            return
        released = "__gripper__" not in contacts_for.get(carried, set())
        if placed_near and released:
            self._sub_completed[i] = True

    def _update_other(self, i: int, obj: str, contacts_for) -> None:
        if obj in self._touched_set:
            self._sub_engaged[i] = True
            self._sub_completed[i] = True

    # ─── scoring ───────────────────────────────────────────────────────
    def compute_score(self) -> float:
        subtasks = getattr(self, "subtasks", [])
        if not subtasks:
            return 1.0
        total = 0.0
        for i in range(len(subtasks)):
            if self._sub_completed[i]:
                total += 1.0
            elif self._sub_engaged[i]:
                total += 0.5
        return float(total / len(subtasks))

    def _per_subtask_compliance(self) -> List[float]:
        return [
            1.0 if self._sub_completed[i] else (0.5 if self._sub_engaged[i] else 0.0)
            for i in range(len(self.subtasks))
        ]

    def raw(self) -> Dict[str, Any]:
        d = super().raw()
        d["subtasks"] = [list(s) for s in self.subtasks]
        d["carried_objs"] = list(self._carried)
        d["per_subtask_compliance"] = self._per_subtask_compliance()
        return d


def load_task_plan_progress(
    mission_info_yaml_path: str,
) -> Optional[TaskPlanProgressPreference]:
    """Construct a TaskPlanProgressPreference from mission_info.yaml.

    Prefers ``task_plan`` (pref-aware sequence); falls back to ``naive_plan``.
    Returns ``None`` if neither is present.
    """
    with open(mission_info_yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    plan = data.get("task_plan") or data.get("naive_plan") or []
    plan_tuples: List[Tuple[str, str]] = []
    for step in plan:
        if isinstance(step, (list, tuple)) and len(step) >= 2 and step[1]:
            plan_tuples.append((str(step[0]), str(step[1])))
    if not plan_tuples:
        return None
    return TaskPlanProgressPreference(task_plan=plan_tuples)


def load_task_plan_compliance(
    mission_info_yaml_path: str,
) -> Optional[TaskPlanCompliancePreference]:
    """Construct a TaskPlanCompliancePreference from mission_info.yaml.

    Same source-of-truth as ``load_task_plan_progress`` (``task_plan`` →
    ``naive_plan``); returns ``None`` when neither is present (e.g. raw
    LIBERO missions with no plan).
    """
    with open(mission_info_yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    plan = data.get("task_plan") or data.get("naive_plan") or []
    plan_tuples: List[Tuple[str, str]] = []
    for step in plan:
        if isinstance(step, (list, tuple)) and len(step) >= 2 and step[1]:
            plan_tuples.append((str(step[0]), str(step[1])))
    if not plan_tuples:
        return None
    return TaskPlanCompliancePreference(task_plan=plan_tuples)

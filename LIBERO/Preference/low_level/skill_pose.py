"""Skill-tier preference metrics (series 2).

Series-2 preferences live in the PLAN — the pref rollout runs `pick_left_edge`
where the naive runs plain `pick`. `TaskPlanCompliancePreference` cannot tell
them apart: both grasp the same object, so its per-subtask progress score is
identical on both sides. What actually differs is the *pose* the gripper takes
at the grasp / release instant, or whether the object is lifted at all.

Every class here follows the same contract as the motion-tier metrics
(`get_reward` per step, `final_raw()` once at the end), so the existing
dual-anchor calibration, satisfaction mapping and video renderer work
unchanged:

    satisfaction = clip( (raw - N) / (P - N), 0, 1 ) x 100

with P/N the pref/naive demo medians. Direction is encoded by sign(P - N), so
none of these need to know which way "more preferred" points.

Family coverage (see dataset/series_2/*/mission_info.yaml):

  grasp_offset   2-1 pick_left_edge / 2-13 pick_right_edge / 2-2 pick_back_edge
                 / 2-3 pick_handle  — grasp point shifted a few cm along one
                 object axis
  grasp_tilt     2-4 pick_top_angle / 2-5, 2-6 pick_from_side / 2-7
                 pick_horizontal / 2-11 pick_from_front — grasp point roughly
                 unchanged, gripper approach axis tilted 45-90 deg
  place_rotation 2-12 put_with_rotation — yaw swept while descending to place
  object_tilt    2-14 lay_down — object ends on its side instead of upright
  object_lift    2-8 push / 2-9 drag — object slides, never leaves the surface

2-10 throw is covered by the pre-existing `ReleaseHeightPreference`
(Preference/low_level/extra/release_height.py).
"""

from __future__ import annotations

from typing import Any, List, Set

import numpy as np

from .episode_trace import DEFAULT_FINGER_JOINTS
from .motion_preference import BaseMotionPreference

# Finger-gap (joint1 - joint2) thresholds for the Panda gripper: ~0.08 fully
# open, ~0.00 fully closed on a thin object.
GRIP_CLOSED_FRAC = 0.6   # gap below 60% of this episode's max = "closed"
LIFT_THRESH_M = 0.01     # object z above its own start = "picked up"
RELEASE_REOPEN_M = 0.01
GRASP_CLOSE_M = 0.01     # fingers must be >= 1 cm inside the fully-open gap for a lift to count as a grasp  # finger gap this much wider than while holding = "released"


def _body_pos(env: Any, name: str) -> np.ndarray:
    """Object body world position; tolerates the `_main` suffix convention."""
    for n in (f"{name}_main", name):
        try:
            return np.asarray(env.sim.data.get_body_xpos(n), dtype=np.float64).copy()
        except Exception:
            continue
    return np.zeros(3)


def _handle_offset_local(env: Any, name: str) -> np.ndarray | None:
    """Offset (OBJECT frame) from the body origin to the geom farthest from it
    in xy -- the handle on a mug. None when the body / its geoms are missing."""
    sim = env.sim
    bid = None
    for n in (f"{name}_main", name):
        try:
            bid = int(sim.model.body_name2id(n))
            break
        except Exception:
            continue
    if bid is None:
        return None
    origin = np.asarray(sim.data.body_xpos[bid], dtype=np.float64)
    mat = np.asarray(sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
    pts = [np.asarray(sim.data.geom_xpos[g], dtype=np.float64)
           for g in range(int(sim.model.ngeom)) if int(sim.model.geom_bodyid[g]) == bid]
    if not pts:
        return None
    d = np.asarray(pts) - origin
    far = d[int(np.argmax(np.linalg.norm(d[:, :2], axis=1)))]
    return mat.T @ far


def _body_mat(env: Any, name: str) -> np.ndarray:
    """Object body world rotation matrix (3x3), identity if unavailable."""
    for n in (f"{name}_main", name):
        try:
            return np.asarray(env.sim.data.get_body_xmat(n), dtype=np.float64).reshape(3, 3).copy()
        except Exception:
            continue
    return np.eye(3)


class _GraspEventTrace:
    """Per-step recorder for EE pose, gripper gap and one object's pose.

    Derives the two instants every skill metric keys off:
      t_grasp   — the step just before the object first rises LIFT_THRESH_M
                  above its own start height (the gripper is closed on it and
                  sitting exactly at the grasp pose)
      t_release — the first step after t_grasp where the finger gap reopens
    Both are None when the episode never lifts the object (push / drag).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ee: List[np.ndarray] = []
        self.ee_mat: List[np.ndarray] = []
        self.gap: List[float] = []
        self.obj: List[np.ndarray] = []
        self.obj_mat: List[np.ndarray] = []

    def record(self, env: Any, obj_name: str) -> None:
        try:
            eef = getattr(env.robots[0], "eef_name", "gripper0_eef")
        except Exception:
            eef = "gripper0_eef"
        try:
            self.ee.append(np.asarray(env.sim.data.get_body_xpos(eef), dtype=np.float64).copy())
            self.ee_mat.append(
                np.asarray(env.sim.data.get_body_xmat(eef), dtype=np.float64).reshape(3, 3).copy())
        except Exception:
            self.ee.append(np.zeros(3))
            self.ee_mat.append(np.eye(3))
        # The offline replay shim (scripts/calibrate_low_pref._wrap_env) has no
        # `.gripper`, so fall back to the Panda joint names like EpisodeTrace does.
        try:
            names = list(env.robots[0].gripper.joints)[:2]
        except Exception:
            names = list(DEFAULT_FINGER_JOINTS)
        try:
            q = [float(env.sim.data.get_joint_qpos(n)) for n in names]
            self.gap.append(abs(q[0] - q[1]))
        except Exception:
            self.gap.append(0.0)
        self.obj.append(_body_pos(env, obj_name))
        self.obj_mat.append(_body_mat(env, obj_name))

    @property
    def n(self) -> int:
        return len(self.ee)

    def t_lift(self) -> int | None:
        """First step the object sits LIFT_THRESH_M above its start height WITH
        the fingers at least GRASP_CLOSE_M inside their fully-open gap. A bowl
        rim can be levered 1 cm up by a descending open finger long before the
        fingers close on it; that bump is not a grasp."""
        if self.n < 3:
            return None
        z = np.array([p[2] for p in self.obj])
        g = np.asarray(self.gap, dtype=float)
        closing = g < (float(g.max()) - GRASP_CLOSE_M) if g.max() > 1e-9 else np.ones_like(g, dtype=bool)
        rise = np.nonzero((z - z[0] > LIFT_THRESH_M) & closing)[0]
        return int(rise[0]) if len(rise) else None

    def t_grasp(self) -> int | None:
        tl = self.t_lift()
        return max(0, tl - 1) if tl is not None else None

    def t_release(self) -> int | None:
        tg = self.t_grasp()
        if tg is None or self.n < 3:
            return None
        g = np.asarray(self.gap, dtype=float)
        if g.max() <= 1e-9:
            return None
        # "closed" = within RELEASE_REOPEN_M of the gap measured while holding
        # the object (a wide can closes at 60%+ of the open gap, so a fraction
        # of the max gap never registers as closed for it)
        hold = float(np.min(g[tg:tg + 3]))
        closed = g[tg:] < hold + RELEASE_REOPEN_M
        if not closed.any():
            return None
        reopen = np.nonzero(~closed)[0]
        reopen = reopen[reopen > int(np.nonzero(closed)[0][0])]
        return int(tg + reopen[0]) if len(reopen) else None

    def place_window(self) -> tuple[int, int] | None:
        """[start of final descent, release] — or [start, now] before release.

        Falling back to "now" keeps the running curve alive during the descend
        instead of leaving it undefined until the very last frame (the video's
        satisfaction line would otherwise be a single dot at the end).
        """
        tg = self.t_grasp()
        if tg is None:
            return None
        tr = self.t_release()
        end = tr if tr is not None else self.n - 1
        if end - tg < 3:
            return None
        z = np.array([p[2] for p in self.ee[tg:end + 1]])
        t0 = tg + int(np.argmax(z))
        return (t0, end) if end - t0 >= 2 else None


class _SkillPosePreference(BaseMotionPreference):
    """Shared plumbing: record every step, score once at the skill's node.

    Skill preferences are scored like a subtask: nothing until the skill's
    node is reached (t_grasp for grasp-pose metrics, t_release for release
    metrics, task success for whole-episode object metrics), then a DISCRETE
    verdict that is held for the rest of the episode:

        satisfaction = 100  if  criterion_direction * (raw - criterion_threshold) >= 0
                     =   0  otherwise

    Every family has its own criterion: the class defaults below are physical
    values (30 deg tilt, 1 cm lift, ...); `criterion_threshold` /
    `criterion_direction` in the mission yaml's low_prefs params override them
    (grasp_offset has no physical default — its threshold/direction are wired
    per mission from the paired pref/naive demos, see
    scripts/wire_skill_criterion.py). The dual-anchor interpolation of the
    motion tier is deliberately NOT used here.
    """

    saturation = 1.0
    level = "skill"
    CRITERION_THRESHOLD: float | None = None
    CRITERION_DIRECTION: int = 1

    def __init__(self, target_obj: str, pref_description: str = "",
                 pos_description: str = "", criterion_threshold: float | None = None,
                 criterion_direction: int | None = None,
                 target_objs: List[str] | None = None) -> None:
        super().__init__(pref_description, pos_description)
        self.target_obj = str(target_obj)
        self._default_target_obj = self.target_obj
        self.target_objs = list(dict.fromkeys(target_objs or [self.target_obj]))
        if self.target_obj not in self.target_objs:
            raise ValueError('target_objs must include target_obj')
        self._candidate_traces = {}
        self._selected_target = None
        self.trace = _GraspEventTrace()
        self.criterion_threshold = (float(criterion_threshold) if criterion_threshold is not None
                                    else self.CRITERION_THRESHOLD)
        self.criterion_direction = (int(criterion_direction) if criterion_direction is not None
                                    else self.CRITERION_DIRECTION)
        self._success_marked = False
        self._episode_done = False

    def reset(self, env: Any) -> None:
        self.target_obj = self._default_target_obj
        self.trace.reset()
        self._candidate_traces = {name: _GraspEventTrace() for name in self.target_objs} if len(self.target_objs) > 1 else {}
        self._selected_target = None
        self._success_marked = False
        self._episode_done = False

    def mark_success(self) -> None:  # trace is not truncated at success
        self._success_marked = True

    def mark_episode_end(self) -> None:
        """The replay / accumulator has fed the last step: settle-type nodes may score now."""
        self._episode_done = True

    def node_reached(self) -> bool:
        """Grasp-pose metrics: the node is t_grasp, i.e. as soon as raw exists."""
        return self.final_raw() is not None

    def satisfaction(self, mean_reward: float | None) -> float | None:
        raw = mean_reward
        if raw is None or self.criterion_threshold is None or not self.node_reached():
            return None
        return 100.0 if self.criterion_direction * (float(raw) - self.criterion_threshold) >= 0 else 0.0

    def get_reward(self, env: Any) -> float:
        if self._candidate_traces and self._selected_target is None:
            # Composition may change the chosen object. Measure the first actual
            # grasp among the declared candidates, using the same criterion for
            # all conditions; never choose the candidate with the best score.
            reached = []
            for name, trace in self._candidate_traces.items():
                trace.record(env, name)
                grasp_step = trace.t_grasp()
                if grasp_step is not None:
                    reached.append((grasp_step, self.target_objs.index(name), name))
            if reached:
                _, _, self._selected_target = min(reached)
                self.target_obj = self._selected_target
                self.trace = self._candidate_traces[self.target_obj]
            else:
                self.trace = self._candidate_traces[self._default_target_obj]
        else:
            self.trace.record(env, self.target_obj)
        raw = self.final_raw()
        return float(raw) if raw is not None else 0.0

    @property
    def objects_of_interest(self) -> Set[str]:
        return set(self.target_objs)


class GraspOffsetPreference(_SkillPosePreference):
    """raw = grasp point offset from the object centre, in the OBJECT frame.

    Measured at t_grasp and projected onto `axis`. Object frame (not world) so
    the number survives a rotated init layout; `signed=False` reports the
    magnitude, which is what edge-vs-centre grasps differ in when the two
    variants sit on opposite sides (pick_left_edge +5 cm vs pick_right_edge
    -5 cm both read as 0.05 against plain pick's ~0).

    `axis="handle"` (2-3, pick_handle): raw = signed projection of the WORLD
    offset onto the object's centre->handle direction (unit xy vector). The
    handle is the geom of the object body farthest (in xy) from the body
    origin, captured in the OBJECT frame on the first recorded step and
    rotated with the object afterwards, so a mug whose handle points -y (red
    coffee mug) and one whose handle points +y (porcelain mug) both read
    +5 cm for a handle grasp and ~0 / negative for a body grasp. Criterion
    default: >= HANDLE_CRITERION_M (2.5 cm; handle grasps 3.5-5.4 cm, plain
    picks -3.3..+1.6 cm).

    Covers 2-1 / 2-2 / 2-3 / 2-13. `axis`: 'x' | 'y' | 'z' | 'xy' | 'handle'.
    """

    AXES = {"x": 0, "y": 1, "z": 2}
    HANDLE_CRITERION_M = 0.025

    def __init__(self, target_obj: str, axis: str = "y", signed: bool = False,
                 frame: str = "object",
                 pref_description: str = "", pos_description: str = "", **criterion) -> None:
        super().__init__(target_obj, pref_description, pos_description, **criterion)
        self.axis = str(axis).lower()
        if self.axis not in self.AXES and self.axis not in ("xy", "handle"):
            raise ValueError(f"axis must be x/y/z/xy/handle, got {axis!r}")
        self.signed = bool(signed)
        self.frame = str(frame).lower()
        if self.frame not in ("object", "world"):
            raise ValueError(f"frame must be object/world, got {frame!r}")
        self._handle_local: np.ndarray | None = None
        if self.axis == "handle" and criterion.get("criterion_threshold") is None:
            self.criterion_threshold = self.HANDLE_CRITERION_M
            self.criterion_direction = 1
        self.type = "GraspOffsetPreference"

    def reset(self, env: Any) -> None:
        super().reset(env)
        self._handle_local = None

    def get_reward(self, env: Any) -> float:
        previous_target = self.target_obj
        r = super().get_reward(env)
        if self.target_obj != previous_target:
            self._handle_local = None
        if self.axis == "handle" and self._handle_local is None:
            self._handle_local = _handle_offset_local(env, self.target_obj)
        return r

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        t = self.trace.t_grasp()
        if t is None:
            return None
        d = self.trace.ee[t] - self.trace.obj[t]
        if self.axis == "handle":
            if self._handle_local is None:
                return None
            h = self.trace.obj_mat[t] @ self._handle_local
            h = np.array([h[0], h[1], 0.0])
            nrm = float(np.linalg.norm(h))
            if nrm < 1e-6:
                return None
            return float(np.dot(d, h / nrm))
        # The edge skills add their offset in WORLD coordinates on top of the
        # plain grasp point (heuristic.py:852-869), so `frame="world"` measures
        # exactly what the skill changed; `frame="object"` is the right choice
        # when the variant comes from an object-frame grasp_params file.
        if self.frame == "object":
            d = self.trace.obj_mat[t].T @ d
        v = float(np.hypot(d[0], d[1])) if self.axis == "xy" \
            else float(d[self.AXES[self.axis]])
        return v if self.signed else abs(v)


class GraspTiltPreference(_SkillPosePreference):
    """raw = angle (rad) between the gripper approach axis and straight-down.

    Criterion: tilt >= 15 deg (top_angle measures 20-28 deg, front ~37, side /
    horizontal ~90; a plain top-down pick reads 4-7 deg). Node: t_grasp.

    The gripper's local +z points along the fingers; a plain top-down `pick`
    holds it at ~0 rad from world -z. pick_top_angle / pick_from_front tilt it
    ~pi/4, pick_from_side / pick_horizontal ~pi/2. Measured at t_grasp.

    Covers 2-4 / 2-5 / 2-6 / 2-7 / 2-11.
    """

    def __init__(self, target_obj: str, pref_description: str = "",
                 pos_description: str = "", **criterion) -> None:
        super().__init__(target_obj, pref_description, pos_description, **criterion)
        self.type = "GraspTiltPreference"

    CRITERION_THRESHOLD = float(np.deg2rad(15.0))   # pick_top_angle measures 20-28 deg, side ~90; plain pick 4-7
    CRITERION_DIRECTION = 1

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        t = self.trace.t_grasp()
        if t is None:
            return None
        g_z = self.trace.ee_mat[t][:, 2]
        c = float(np.clip(np.dot(g_z, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        return float(np.arccos(c))


class PlaceRotationPreference(_SkillPosePreference):
    """raw = total |dyaw| (rad) of the EE over the final descent to release.

    `put_with_rotation` sweeps -45 deg across its 12 cm descend; plain `put`
    descends with a frozen wrist. Restricting the window to [descent start,
    release] keeps transport-phase wrist drift out of the number.

    Covers 2-12.
    """

    def __init__(self, target_obj: str, pref_description: str = "",
                 pos_description: str = "", **criterion) -> None:
        super().__init__(target_obj, pref_description, pos_description, **criterion)
        self.type = "PlaceRotationPreference"

    CRITERION_THRESHOLD = float(np.deg2rad(30.0))   # put_with_rotation sweeps ~45 deg, plain put ~2 deg
    CRITERION_DIRECTION = 1

    def node_reached(self) -> bool:                  # node = release (end of the descent)
        return self.trace.t_release() is not None

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        win = self.trace.place_window()
        if win is None:
            return None
        t0, t1 = win
        yaw = [float(np.arctan2(m[1, 0], m[0, 0])) for m in self.trace.ee_mat[t0:t1 + 1]]
        d = np.diff(np.unwrap(np.asarray(yaw)))
        return float(np.abs(d).sum())


class ObjectTiltPreference(_SkillPosePreference):
    """raw = MAX angle (rad) between the object's own +z axis and world +z.

    `lay_down` leaves the bottle on its side (~pi/2); `put` sets it upright (~0).
    Taken as the episode max, not the last frame: the reverse sweep that follows
    lay_down can end while the bottle is still settling, and 4/10 pilot demos
    read a spurious 0.0 when scored on the final frame alone. `put` never tilts
    the object at all (pilot naive max 0.023 rad = 1.3 deg), so the max is not
    inflated by transport wobble.

    Covers 2-14.
    """

    def __init__(self, target_obj: str, pref_description: str = "",
                 pos_description: str = "", **criterion) -> None:
        super().__init__(target_obj, pref_description, pos_description, **criterion)
        self.type = "ObjectTiltPreference"

    CRITERION_THRESHOLD = float(np.deg2rad(30.0))   # lay_down leaves 33-64 deg, put ~1-4 deg
    CRITERION_DIRECTION = 1

    def node_reached(self) -> bool:                  # node = task success (object at rest on the goal)
        return self._success_marked

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        if self.trace.n < 2:
            return None
        up = np.array([0.0, 0.0, 1.0])
        # reference = the object's own axis that points up at t=0 (a model's body
        # +z is not necessarily its upright axis: the ketchup mesh reads ~93 deg
        # while standing), so raw = how far that axis has rotated away from up
        up_local = self.trace.obj_mat[0].T @ up
        c = np.clip([float(np.dot(m @ up_local, up)) for m in self.trace.obj_mat], -1.0, 1.0)
        return float(np.arccos(c).max())


class ReleaseDropPreference(_SkillPosePreference):
    """raw = how far the object falls after the gripper opens (m).

    `throw` opens at the move_to cruise height (target_z + 0.20 plus container
    bumps), `put` opens 2 cm above the goal — so the drop distance separates
    them without needing to resolve the target body. Measured as the object's
    height at t_release minus its resting height at the end of the episode.

    Covers 2-10.
    """

    def __init__(self, target_obj: str, pref_description: str = "",
                 pos_description: str = "", **criterion) -> None:
        super().__init__(target_obj, pref_description, pos_description, **criterion)
        self.type = "ReleaseDropPreference"

    # With the release detected relative to the holding gap, a put reads ~0 cm
    # (the object is set down before the fingers open) and a throw 4-10 cm
    # (drawer / tray), so 2 cm separates them with margin.
    CRITERION_THRESHOLD = 0.02
    CRITERION_DIRECTION = 1

    def node_reached(self) -> bool:
        """Node = release, once the object has come to rest (raw needs its resting height)."""
        t = self.trace.t_release()
        if t is None:
            return False
        if self._episode_done or self.trace.n - t >= 20:   # episode over / long past the release: final
            return True
        if self.trace.n - t < 5:
            return False
        z = [p[2] for p in self.trace.obj[-4:]]
        return bool(max(z) - min(z) < 1e-3)

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        t = self.trace.t_release()
        if t is None:
            return None
        return float(self.trace.obj[t][2] - self.trace.obj[-1][2])


class ObjectLiftPreference(_SkillPosePreference):
    """raw = max height the object gains over its own start height (m).

    `push` / `drag` slide the object across the surface — it never leaves it —
    while pick-and-put carries it 20-30 cm up.

    Covers 2-8 / 2-9.
    """

    def __init__(self, target_obj: str, pref_description: str = "",
                 pos_description: str = "", **criterion) -> None:
        super().__init__(target_obj, pref_description, pos_description, **criterion)
        self.type = "ObjectLiftPreference"

    CRITERION_THRESHOLD = 0.05                       # push ~0, drag tilts the basket up to ~3 cm; pick-and-put lifts 10-30 cm
    CRITERION_DIRECTION = -1                         # satisfied when raw <= threshold

    def node_reached(self) -> bool:                  # node = task success (object slid onto the goal)
        return self._success_marked

    def final_raw(self, mean_reward: float | None = None) -> float | None:
        if self.trace.n < 2:
            return None
        z = np.array([p[2] for p in self.trace.obj])
        return float(z.max() - z[0])

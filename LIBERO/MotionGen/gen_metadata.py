"""Gen-time per-demo metadata collector (T3, 2026-07-01).

Rides along the skill-by-skill generation loop in `motion_gen.execute_plan`
and emits everything the pipeline already knows for free — so downstream
consumers (dual-anchor calibrate, Profile compliance, Skill metrics) can read
ground truth instead of re-deriving it from the trajectory offline:

  - execution segments: every (skill, target) with [start, end) step bounds,
    attributed to its owning task_plan subtask
  - per-subtask language (templated from the plan), goal image = end-step
    index into the demo's own obs (no pixel copies)
  - per-subtask pref-critical flags (schema v2): for every applied pref, a
    0/1 of whether the pref must show up in THIS subtask (low prefs via a
    type→scope table, high/skill prefs via naive-vs-task plan diff), plus an
    any-pref summary bit; naive_plan + high_pref description ride along
  - per-segment avg speed / peak Δv / start-end EE pose
  - applied low_prefs (yaml entries + the RESOLVED gen knobs, e.g. the
    actually-sampled speed_scale and avoid waypoints — not recoverable
    post-hoc) and per-pref final_raw (lets calibrate skip sim replay)
  - gripper close/open events + release heights, task objects' initial/final
    poses

Storage: one `meta/` group per demo in the rollout HDF5 (json-string
datasets + schema_version attr). `concatenate_hdf5_demos` deep-copies whole
demo groups, so meta follows the merged Mission_<id>.hdf5 for free.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 4

# ── per-subtask language ────────────────────────────────────────────────
# {obj}/{carried} arrive with their article ("the …"); {prep} comes from the
# target phrase (drawers are "in", surfaces "on").
_SKILL_LANGUAGE = {
    "pick": "pick up {obj}",
    "put": "put {carried} {prep} {obj}",
    "push": "push {obj}",
    "drag": "drag {obj}",
    "throw": "throw {obj}",
    "lay_down": "lay down {carried} on {obj}",
    "open_drawer": "open {obj}",
    "close_drawer": "close {obj}",
    "turn_on_stove": "turn on {obj}",
    "turn_off_stove": "turn off {obj}",
}

# pick_<variant> → trailing adverbial phrase
_PICK_VARIANTS = {
    "left_edge": "by its left edge",
    "right_edge": "by its right edge",
    "back_edge": "by its back edge",
    "from_side": "from the side",
    "from_front": "from the front",
    "top_angle": "from a top angle",
    "with_rotation": "with a rotated grip",
    "horizontal": "with a horizontal grip",
    "handle": "by its handle",
    "z_90": "with the grip rotated 90 degrees",
}
# put_<edge> gets its own sentence shape; other put variants append a phrase
_PUT_EDGE_VARIANTS = {"left_edge": "left", "right_edge": "right", "back_edge": "back"}
_PUT_SUFFIX_VARIANTS = {
    "with_rotation": "with a rotated grip",
    "horizontal": "with a horizontal grip",
    "handle": "by its handle",
}

# BDDL region-suffixed targets → readable phrase + fitting preposition
_REGION_SUFFIXES = {
    "_cabinet_bottom": ("the bottom drawer of {base}", "in"),
    "_cabinet_middle": ("the middle drawer of {base}", "in"),
    "_cabinet_top": ("the top drawer of {base}", "in"),
    "_top_side": ("the top of {base}", "on"),
    "_cook_region": ("the cook region of {base}", "on"),
}


def _obj_phrase(name: Optional[str]) -> str:
    """akita_black_bowl_1 → 'akita black bowl' (instance digits dropped
    wherever they appear, incl. mid-name as in white_cabinet_1_cabinet_top)."""
    if not name:
        return "object"
    toks = [t for t in str(name).split("_") if not t.isdigit()]
    return " ".join(toks) or "object"


def _target_phrase(name: Optional[str]) -> Tuple[str, str]:
    """Target name → ('the …' noun phrase, preposition for put)."""
    if not name:
        return "the object", "on"
    s = str(name)
    for suf, (tpl, prep) in _REGION_SUFFIXES.items():
        if s.endswith(suf):
            return tpl.format(base="the " + _obj_phrase(s[: -len(suf)])), prep
    return "the " + _obj_phrase(s), "on"


def _base_skill(skill: str) -> str:
    """pick_right_edge → pick, put_left_edge → put. Stored subtask language is
    pref-agnostic (2026-07-13 decision): the same instruction must serve pref
    and no-pref data, so variant adverbials never enter the language — but the
    actual plan target (region suffix, blue_plate_1) is kept."""
    for base in ("pick", "put"):
        if skill == base or skill.startswith(base + "_"):
            return base
    return skill


def _subtask_language(skill: str, target: str, carried: Optional[str] = None) -> str:
    obj, prep = _target_phrase(target)
    carried_p = ("the " + _obj_phrase(carried)) if carried else "the object"
    tpl = _SKILL_LANGUAGE.get(skill)
    if tpl is not None:
        return tpl.format(obj=obj, carried=carried_p, prep=prep)
    if skill.startswith("pick"):
        suffix = _PICK_VARIANTS.get(skill[len("pick"):].lstrip("_"))
        return f"pick up {obj}" + (f" {suffix}" if suffix else "")
    if skill.startswith("put"):
        variant = skill[len("put"):].lstrip("_")
        if variant in _PUT_EDGE_VARIANTS:
            return f"put {carried_p} at the {_PUT_EDGE_VARIANTS[variant]} edge of {obj}"
        suffix = _PUT_SUFFIX_VARIANTS.get(variant)
        return f"put {carried_p} {prep} {obj}" + (f" {suffix}" if suffix else "")
    return f"{skill.replace('_', ' ')} {obj}"


def _move_language(target: str) -> str:
    """Language for the move phase of a plan step (schema v3: the approach /
    transport before each skill is its own subtask). Always plain "move to X"
    — when carrying, the object was already named by the preceding pick."""
    obj, _ = _target_phrase(target)
    return f"move to {obj}"


# ── pref-critical: which subtasks must EXPRESS each preference ──────────
# Low-level type (BaseMotionPreference.preference_type) → scope.
#   global    — any step can violate it (step count, accel peak)
#   transport — only while carrying an object (metrics gate on transport)
# "global": every phase of every step. "move": every MOVE phase (approach
# included) — gen shapes all move_to's for these ("Move higher" applies
# path_height_offset to each move_to; avoid rewrites every move_to into
# lift_up+move_to_away; the rotation demo's wrist turn happens on the
# approach move, verified 2026-07-21 on 1-8_114: approach 123° vs naive 4°,
# carry ~10° both). "transport": only the CARRYING move phase — gen shapes
# these only with an object in hand (straight_path knob requires
# picked_target + put-family; apply_approach inserts only when gripper
# closed). Schema v3 wrongly put all five path prefs under "transport".
_PREF_SCOPE = {
    "speed_range": "global",
    "acc_range": "global",
    "avoid_to_object": "move",
    "approach_to_object": "transport",
    "straight_path": "transport",
    "height_offset": "move",
    "rotation": "move",
}
# Skill-tier metrics bind to their skill's subtask.
_PREF_SKILL_SCOPE = {"release_height": ("throw",)}


def _targets_compatible(a: str, b: str) -> bool:
    """Same object, allowing region refinement (white_cabinet_1 matches
    white_cabinet_1_cabinet_bottom — granularity, not a pref rewrite)."""
    a, b = str(a), str(b)
    return a == b or a.startswith(b + "_") or b.startswith(a + "_")


def _plan_diff_indices(plan: List[Tuple[str, str]],
                       naive_plan: List[Tuple[str, str]]) -> set:
    """Task-plan positions NOT matched by an order-preserving alignment (LCS)
    against the naive plan. Match = identical skill + compatible target, so an
    edge/rotation variant (pick vs pick_right_edge), an object substitution,
    or an inserted step all come back as diff — i.e. the pref rewrote them."""
    n, m = len(plan), len(naive_plan)
    if not m:
        return set(range(n))

    def _match(i: int, j: int) -> bool:
        return (plan[i][0] == naive_plan[j][0]
                and _targets_compatible(plan[i][1], naive_plan[j][1]))

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = (1 + dp[i + 1][j + 1]) if _match(i, j) else \
                max(dp[i + 1][j], dp[i][j + 1])
    matched, i, j = set(), 0, 0
    while i < n and j < m:
        if _match(i, j) and dp[i][j] == 1 + dp[i + 1][j + 1]:
            matched.add(i)
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return set(range(n)) - matched


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


class GenMetadataCollector:
    """One instance per rollout episode. Cheap: keeps only segment bounds and
    small scalars during the loop; everything else is derived in finalize()
    from the in-memory hdf5_dict."""

    def __init__(
        self,
        task_plan: List[Tuple[str, str]],
        low_pref_entries: Optional[List[dict]] = None,
        gen_cfg: Any = None,  # LowPrefGenPlan dataclass (or None)
        env: Any = None,
        mission_id: Optional[str] = None,
        naive_plan: Optional[List[Tuple[str, str]]] = None,
        high_pref_desc: Optional[str] = None,
    ):
        self.plan: List[Tuple[str, str]] = [
            (str(s[0]), str(s[1])) for s in (task_plan or [])
            if isinstance(s, (list, tuple)) and len(s) >= 2
        ]
        self.naive_plan: List[Tuple[str, str]] = [
            (str(s[0]), str(s[1])) for s in (naive_plan or [])
            if isinstance(s, (list, tuple)) and len(s) >= 2
        ]
        # pipeline builds a TaskPreference with pref_description="" when the
        # mission has no high-level pref (series 1) — treat empty as absent so
        # plan-diff attribution falls through to the low-pref keys
        self.high_pref_desc = high_pref_desc if high_pref_desc else None
        self.mission_id = mission_id
        self.low_pref_entries = low_pref_entries or []
        self.gen_cfg = gen_cfg
        self._segments: List[dict] = []
        self._plan_ptr = 0
        self._avoid_waypoints: List[list] = []
        # carried obj per plan idx (nearest preceding pick) for put language
        self._carried: List[Optional[str]] = []
        last_pick = None
        for skill, obj in self.plan:
            if skill.startswith("pick"):
                last_pick = obj
                self._carried.append(None)
            else:
                self._carried.append(last_pick)
        # initial poses of the plan's objects (env at episode start)
        self._objects_start = self._read_object_poses(env)

    # ─── during the loop ───────────────────────────────────────────────
    @staticmethod
    def _norm_name(name: Optional[str]) -> Optional[str]:
        """Sim-name normalization so plan entries (original names) match the
        loop-tail `target` (often already to_sim_env_name-converted)."""
        if name is None:
            return None
        try:
            from Utility.tool_func import to_sim_env_name
            return to_sim_env_name(str(name))
        except Exception:
            return str(name)

    def on_segment(self, skill: str, target: Optional[str], end_step: int) -> None:
        """Call once per executed skill segment, AFTER its actions ran.
        end_step = len(hdf5_dict['actions']) at that moment. start is implied
        by the previous segment's end (0 for the first)."""
        start = self._segments[-1]["end"] if self._segments else 0
        if end_step <= start:
            return  # segment produced no recorded steps (e.g. skipped)
        plan_idx = min(self._plan_ptr, max(len(self.plan) - 1, 0))
        self._segments.append({
            "skill": str(skill),
            "target": str(target) if target is not None else None,
            "start": int(start),
            "end": int(end_step),
            "plan_idx": int(plan_idx),
        })
        # consume the owning plan entry when its own skill just ran
        if (self._plan_ptr < len(self.plan)
                and str(skill) == self.plan[self._plan_ptr][0]
                and self._norm_name(target) == self._norm_name(self.plan[self._plan_ptr][1])):
            self._plan_ptr += 1

    def record_avoid_waypoint(self, waypoint) -> None:
        self._avoid_waypoints.append(np.asarray(waypoint, dtype=float).tolist())

    # ─── finalize ──────────────────────────────────────────────────────
    def _read_object_poses(self, env: Any) -> Dict[str, list]:
        poses: Dict[str, list] = {}
        if env is None:
            return poses
        from Utility.tool_func import to_sim_env_name
        for _, obj in self.plan:
            sim_name = to_sim_env_name(obj)
            try:
                pos = env.sim.data.get_body_xpos(sim_name)
                poses[obj] = np.asarray(pos, dtype=float).tolist()
            except Exception:
                continue
        return poses

    def _pref_critical_maps(self) -> Tuple[Dict[str, set], Dict[str, set]]:
        """(move_crit, manip_crit): per applied pref, the plan indices whose
        MOVE phase / MANIPULATION phase must show the pref in behavior.
        Low prefs come from their type's scope — move-scope prefs live in
        every move phase (approach included), transport-scope only in the
        carrying move phase, skill-scope in their skill's manipulation,
        global in both. Plan-diff positions (naive vs task plan) flag BOTH
        phases of the rewritten step (approach direction, target choice and
        the grasp/placement itself are all pref-shaped); they go to the
        high-level pref when one exists, else to the low prefs (edge variants
        in low-pref missions are pref-motivated per 2026-07-02 decision)."""
        n = len(self.plan)
        keys = [e.get("type") for e in self.low_pref_entries if e.get("type")]
        move: Dict[str, set] = {}
        manip: Dict[str, set] = {}
        for k in keys:
            scope = _PREF_SCOPE.get(k)
            skills = _PREF_SKILL_SCOPE.get(k)
            if scope == "global":
                move[k] = set(range(n))
                manip[k] = set(range(n))
            elif scope == "move":
                move[k] = set(range(n))
                manip[k] = set()
            elif scope == "transport":
                move[k] = {i for i in range(n) if self._carried[i] is not None}
                manip[k] = set()
            elif skills:
                move[k] = set()
                manip[k] = {i for i in range(n)
                            if self.plan[i][0].startswith(skills)}
            else:  # unknown type — conservative: every subtask may express it
                move[k] = set(range(n))
                manip[k] = set(range(n))
        diff = (_plan_diff_indices(self.plan, self.naive_plan)
                if self.naive_plan else set())
        if self.high_pref_desc is not None:
            move["task_pref"] = set(diff)
            manip["task_pref"] = set(diff)
        elif diff:
            for k in keys:
                move[k] |= diff
                manip[k] |= diff
        return move, manip

    def _build_subtasks(self, segments: List[dict], ee_pos: np.ndarray,
                        n_steps: int) -> List[dict]:
        """Schema v3: each plan step splits into a MOVE subtask (approach /
        transport — every segment before the step's own skill: lift_up,
        move_to*) and a MANIPULATION subtask (the skill itself onward). Steps
        without recorded move segments emit the manipulation subtask only."""
        move_crit, manip_crit = self._pref_critical_maps()

        def _avg(a: int, b: int) -> float:
            p = ee_pos[a:min(b, len(ee_pos))]
            if len(p) >= 2:
                return float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())
            return 0.0

        subtasks: List[dict] = []
        for idx, (skill, target) in enumerate(self.plan):
            segs = [s for s in segments if s["plan_idx"] == idx]
            if not segs:
                continue
            cut = next((i for i, s in enumerate(segs) if s["skill"] == skill),
                       None)
            move_segs = segs[:cut] if cut is not None else []
            manip_segs = segs[cut:] if cut is not None else segs
            carried = self._carried[idx]
            phases = []
            if move_segs:
                phases.append(("move", "move_to", move_segs,
                               _move_language(target), move_crit))
            phases.append(("manipulate", skill, manip_segs,
                           _subtask_language(_base_skill(skill), target,
                                             carried), manip_crit))
            for phase, sk, ph_segs, lang, crit in phases:
                a = min(s["start"] for s in ph_segs)
                b = max(s["end"] for s in ph_segs)
                by_pref = {k: int(idx in s) for k, s in crit.items()}
                subtasks.append({
                    "plan_idx": idx,
                    "phase": phase,
                    "skill": sk,
                    "target": target,
                    "language": lang,
                    "start": int(a),
                    "end": int(b),
                    "avg_speed": _avg(a, b),
                    "goal_image_step": int(max(min(b, n_steps) - 1, a)),
                    "pref_critical": int(any(by_pref.values())),
                    "pref_critical_by_pref": by_pref,
                })
        return subtasks

    @staticmethod
    def _gripper_events(actions: np.ndarray) -> Tuple[List[int], List[int]]:
        """Sign flips of the gripper command dim: -1→1 = close, 1→-1 = open."""
        if actions.size == 0:
            return [], []
        g = np.sign(actions[:, -1])
        closes = [int(i) for i in range(1, len(g)) if g[i] > 0 >= g[i - 1]]
        opens = [int(i) for i in range(1, len(g)) if g[i] < 0 <= g[i - 1]]
        return closes, opens

    def finalize(self, hdf5_dict: dict, env: Any = None,
                 low_pref2apply: Optional[list] = None) -> Dict[str, Any]:
        n_steps = len(hdf5_dict.get("actions", []))
        ee_pos = np.asarray(hdf5_dict["obs"].get("ee_pos", []), dtype=float)
        ee_ori = np.asarray(hdf5_dict["obs"].get("ee_ori", []), dtype=float)
        actions = np.asarray(hdf5_dict.get("actions", []), dtype=float)

        # clamp segment bounds to what was actually recorded
        segments = [dict(s) for s in self._segments if s["start"] < n_steps]
        if segments:
            segments[-1]["end"] = min(segments[-1]["end"], n_steps)

        def _pose(i: int) -> Optional[list]:
            if len(ee_pos) == 0 or i >= len(ee_pos):
                return None
            p = list(np.asarray(ee_pos[i], dtype=float))
            if len(ee_ori) > i:
                p += list(np.asarray(ee_ori[i], dtype=float))
            return p

        for s in segments:
            a, b = s["start"], min(s["end"], len(ee_pos))
            if b - a >= 2:
                d = np.linalg.norm(np.diff(ee_pos[a:b], axis=0), axis=1)
                s["avg_speed"] = float(d.mean())         # m per control step
                dv = np.diff(np.diff(ee_pos[a:b], axis=0), axis=0)
                s["peak_dv"] = (float(np.linalg.norm(dv, axis=1).max())
                                if len(dv) else 0.0)
            else:
                s["avg_speed"] = 0.0
                s["peak_dv"] = 0.0
            s["ee_start"] = _pose(a)
            s["ee_end"] = _pose(max(b - 1, a))
            s["goal_image_step"] = int(max(b - 1, a))   # index into obs/*_rgb

        subtasks = self._build_subtasks(segments, ee_pos, n_steps)

        closes, opens = self._gripper_events(actions)
        releases = [{"step": i,
                     "height": (float(ee_pos[i][2]) if i < len(ee_pos) else None)}
                    for i in opens]

        final_raw: Dict[str, Optional[float]] = {}
        for p in (low_pref2apply or []):
            try:
                vals = [v for v in hdf5_dict.get("low_pref_rewards", {})
                        .get(p.preference_type, []) if v is not None]
                mean_r = float(np.mean(vals)) if vals else None
                raw = p.final_raw(mean_r)
                final_raw[p.preference_type] = (float(raw) if raw is not None
                                                else None)
            except Exception:
                final_raw[p.preference_type] = None

        gen_cfg_dict = None
        if self.gen_cfg is not None:
            from dataclasses import asdict, is_dataclass
            gen_cfg_dict = (asdict(self.gen_cfg) if is_dataclass(self.gen_cfg)
                            else dict(vars(self.gen_cfg)))

        return {
            "schema_version": SCHEMA_VERSION,
            "mission_id": self.mission_id,
            "task_plan": [list(s) for s in self.plan],
            "naive_plan": [list(s) for s in self.naive_plan],
            "high_pref": ({"description": self.high_pref_desc}
                          if self.high_pref_desc is not None else None),
            "subtasks": subtasks,
            "segments": segments,
            "low_prefs_applied": {
                "entries": self.low_pref_entries,
                "gen_knobs": gen_cfg_dict,
                "avoid_waypoints": self._avoid_waypoints,
            },
            "low_pref_final_raw": final_raw,
            "events": {
                "gripper_close_steps": closes,
                "gripper_open_steps": opens,
                "releases": releases,
            },
            "objects": {
                "start": self._objects_start,
                "end": self._read_object_poses(env),
            },
            "num_steps": int(n_steps),
        }

    # ─── hdf5 ──────────────────────────────────────────────────────────
    def write_to(self, ep_data_grp, meta: Optional[Dict[str, Any]] = None,
                 **finalize_kwargs) -> None:
        """Write the metadata as a `meta/` group of json-string datasets."""
        if meta is None:
            meta = self.finalize(**finalize_kwargs)
        g = ep_data_grp.create_group("meta")
        g.attrs["schema_version"] = SCHEMA_VERSION
        for key, val in meta.items():
            if key == "schema_version":
                continue
            g.create_dataset(key, data=json.dumps(val, default=_json_default))


def read_demo_meta(ep_data_grp) -> Optional[Dict[str, Any]]:
    """Convenience reader: data/demo_i group → dict (None if no meta)."""
    if "meta" not in ep_data_grp:
        return None
    g = ep_data_grp["meta"]
    out: Dict[str, Any] = {"schema_version": int(g.attrs.get("schema_version", 0))}
    for key in g.keys():
        raw = g[key][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        out[key] = json.loads(raw)
    return out

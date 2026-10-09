# VLA_data — Demo HDF5 Format

Format of every `Mission_<mission_id>.hdf5` produced by the data-gen pipeline
(`LIBERO/main.py` / `step1_data_gen.py` → `MotionGen/motion_gen.py`), one file
per mission, one or more demos per file. Applies to all data (re)generated
with **subtask meta schema v3** (2026-07-15 onward). Demos generated before
2026-07 lack the `meta/` group — consumers must fall back to sim replay.

## File layout

```
Mission_<mission_id>.hdf5
└── data/
    └── demo_<N>/                       # one successful episode, T steps
        ├── (attrs) init_state          # float64[*] — flattened mujoco sim state at t=0
        ├── (attrs) model_file          # str — full MJCF XML of the scene
        ├── (attrs) num_samples         # int — T
        ├── actions                     # (T, 7)   float64  OSC delta: dpos(3) + dori(3) + gripper(1)
        ├── states                      # (T, D)   float64  full mujoco sim state per step (D scene-dependent)
        ├── robot_states                # (T, 9)   float64
        ├── rewards                     # (T,)     float64
        ├── dones                       # (T,)     int64
        ├── low_pref_rewards/
        │   └── <pref_type>             # (T,)     float64  per-step reward in [0,1], one dataset per applied low pref
        ├── obs/
        │   ├── agentview_rgb           # (T, 256, 256, 3) uint8 — stored upright (already rotated at save time, do NOT flip again)
        │   ├── eye_in_hand_rgb         # (T, 256, 256, 3) uint8
        │   ├── frontview_rgb           # (T, 256, 256, 3) uint8
        │   ├── ee_pos                  # (T, 3)   float64
        │   ├── ee_ori                  # (T, 3)   float64  axis-angle
        │   ├── ee_states               # (T, 6)   float64  ee_pos ++ ee_ori
        │   ├── gripper_states          # (T, 2)   float64
        │   └── joint_states            # (T, 7)   float64
        └── meta/                       # gen-time GT metadata (schema below)
```

## `meta/` group (schema_version = 3)

Group attr `schema_version: 3`. Every dataset is a scalar **JSON string** —
read with `json.loads(demo["meta/<key>"][()])`, or use the helper:

```python
from MotionGen.gen_metadata import read_demo_meta   # returns dict, or None if no meta
meta = read_demo_meta(f["data/demo_0"])
```

| key | content |
|---|---|
| `mission_id` | `"1-2_113"` |
| `num_steps` | `T` (== `actions.shape[0]`) |
| `task_plan` | executed plan `[[skill, target], ...]`, e.g. `[["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "blue_plate_1"]]` |
| `naive_plan` | the no-preference plan for the same mission |
| `subtasks` | **two-level subtask segmentation** — list of dicts, see below |
| `segments` | raw motion segments (finer than subtasks): `{skill, target, start, end, plan_idx, avg_speed, peak_dv, ee_start(6), ee_end(6), goal_image_step}`; includes internal skills like `lift_up` |
| `low_prefs_applied` | `{entries: [{type, description}], gen_knobs: {speed_scale, acc_max, path_height_offset, prefer_straight_path, prefer_rotation_while_moving, obj_to_approach, approach_distance, obj_to_avoid, avoid_min_threshold, avoid_max_threshold}, avoid_waypoints: [[x,y,z], ...]}` — the knobs actually used at gen time |
| `high_pref` | `{description}` for skill/profile (high-level) prefs, else `null` |
| `low_pref_final_raw` | `{pref_type: float}` — gen-time episode-level raw metric per applied low pref (verified ≈ eval-side replay value) |
| `events` | `{gripper_close_steps: [t...], gripper_open_steps: [t...], releases: [{step, height}]}` |
| `objects` | `{start: {obj_name: [x,y,z]}, end: {...}}` — poses of the plan's objects at episode start/end |

### `subtasks` entries

Each plan step is split into a **move** phase (approach/transport) and a
**manipulate** phase, so a 2-step pick-and-place plan yields 4 subtasks
(a plan step with no move segment yields only its manipulate subtask):

```json
{
  "plan_idx": 1,                      // index into task_plan
  "phase": "move",                    // "move" | "manipulate"
  "skill": "move_to",                 // manipulate keeps the plan skill name (e.g. "pick_left_edge")
  "target": "basket_1",
  "language": "move to the basket",   // pref-agnostic (see below)
  "start": 143, "end": 341,           // step range [start, end); end == next subtask's start; last end == num_steps
  "avg_speed": 0.0046,                // mean per-step EE displacement (m)
  "goal_image_step": 340,             // frame index representing this subtask's goal (== end-1)
  "pref_critical": 1,                 // 1 if any pref judges this subtask critical
  "pref_critical_by_pref": {"speed_range": 1}   // per-pref breakdown
}
```

Language rules (pref-agnostic, decided 2026-07-15): skill variants collapse to
their base verb (`pick_right_edge` → "pick up …", `put_left_edge` → "put …"),
but the plan's actual spatial/object target is kept ("bottom drawer of the
white cabinet", "blue plate"). Move subtasks read "move to X" without naming
the carried object (the preceding pick already named it); put language names
the carried object taken from the nearest preceding pick in the plan.

Crit semantics: global low prefs (speed/acc/straight/height/rotation/avoid)
mark both phases of every step by their own applicability; transport-scoped
prefs mark only carrying move phases; high prefs (skill/profile) mark the
phases of the plan step(s) that differ from `naive_plan`.

## Notes

- `Utility/concatenate_hdf5.py` merges per-rollout files into `Mission_*.hdf5`
  and copies the whole `meta/` group verbatim.
- Human-readable dump + per-subtask contact sheet:
  `python scripts/view_subtask_meta.py <file.hdf5>` → `<dir>/subtask_meta_view/`
  (one column per subtask start frame + final frame).
- Sample data with all 8 pref types: `VLA_data/MetaSmoke/` (1 demo each).
- `*.hdf5` under `VLA_data/` are gitignored; only docs/config are tracked.

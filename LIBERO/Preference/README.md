# Preference Satisfaction System

P_Bench evaluates VLA personalization along **three independent preference tiers**.
At rollout time each tier auto-emits per-episode satisfaction values and they
are written as separate top-level keys in `rollout_summary.yaml`.

---

## 1. Three Tiers

| Tier | level tag | What it measures | Where it lives |
|---|---|---|---|
| **Low (Motion)** | `level = "low"` | Continuous trajectory-level signals (speed / acc / distance / path / height / rotation). One per-step reward in `[0,1]` driven by `env.sim.data`. | `Preference/low_level/` |
| **Skill** | `level = "skill"` | Skill-specific compliance — pick / put / drawer / throw style. Geometric (e.g. `release_height`) and VLM-based (planned `SkillCompliancePreference`). | `Preference/low_level/release_height.py` (geometric); VLM class WIP |
| **Profile (User-Profile)** | `level = "profile"` | Whether the rollout follows the planned object sequence (`task_plan`). Touch via mujoco contact + manipulated set + LCS order. | `Preference/high_level/task_plan_progress.py` |

All concrete classes inherit `BaseMotionPreference` (`Preference/low_level/motion_preference.py`) and self-register into `BaseMotionPreference._registry`. The tier is identified solely by the `level` class attr.

---

## 2. Per-pref output schema

`LowPrefAccumulator.summary()` (`PolicyEval/low_pref_reward.py`) buckets all
attached prefs by their `level` and emits, per pref:

```yaml
<pref_type>:
  mean_reward: 0.42      # raw signal averaged over episode
  satisfaction: 84.0     # 0–100, anchored to demo (D4RL-style)
  mapped: 0.20           # 0–1, linear in [mapping_low, mapping_high]; <low → 0
  n_steps: 187
```

`PolicyEval/eval.py:run_cell` then writes three sibling keys per episode:

```yaml
low_pref_reward:    {speed_range: {…}, height_offset: {…}, …}
skill_pref_reward:  {release_height: {…}}
profile_pref_reward:{task_plan_progress: {…}}
```

`scripts/aggregate_eval.py` rolls these into `eval_results.xlsx` with columns
`low_pref_reward_mean / skill_pref_reward_mean / profile_pref_reward_mean / pref_satisfaction_overall` (per-episode product across buckets, mean across episodes — coupled-pref semantics).

---

## 3. Anchors & Mapping

### `pref_anchor` (D4RL-style normalisation)

`scripts/calibrate_low_pref.py` replays GT-pref demos through declared
`low_prefs` and writes the median raw mean into `mission_info.yaml`:

```yaml
low_pref_calibration:
  speed_range:    {pref_anchor: 87.0, n_demos: 50}     # step count for SpeedRange
  height_offset:  {pref_anchor: 0.18}
```

At runtime `load_motion_prefs` injects the anchor into each pref instance, and
`BaseMotionPreference.satisfaction` becomes:

```
satisfaction = clip(mean_reward / pref_anchor, 0, 1) × 100
```

Falls back to legacy `saturation` class const when an anchor is missing.

### `mapped_satisfaction` (0–1 strictness window)

Default window `[mapping_low=80, mapping_high=100]`. `mapped` is the linearly
rescaled `[0,1]` value within that window; below `80` → 0, above `100` → 1.
Subclasses override `mapping_low / mapping_high` if a different band fits.

---

## 4. Implemented Preferences

### Low (motion)

| key | physical signal | notes |
|---|---|---|
| `speed_range` | episode total `n_steps` | direction `"less"` (speed_up) / `"more"` (slow_down) |
| `acc_range` | EE peak \|Δv\| | band `[min_acc, max_acc]` |
| `approach_to_object` / `avoid_to_object` | EE → obj distance | less_than / greater_than threshold |
| `straight_path` | path_length / displacement | linear falloff from straight=1 |
| `height_offset` | EE peak_z − (base_z + 0.20m) | band on target offset |
| `rotation` | quat angle change **after EE descent starts** | 0 if any rotation during descent ≥ `max_angle_rad` |

### Skill

| key | physical signal | covers |
|---|---|---|
| `release_height` | `EE_z − target_z` at release moment (gripper-obj distance jump) | `throw` |
| `skill_compliance` (WIP) | VLM dual-screenshot (demo strip vs policy strip) | `pick_left_edge / pick_handle / pick_from_side / put_with_rotation / …` |

### Profile

| key | physical signal | semantics |
|---|---|---|
| `task_plan_progress` | mujoco contact + obj displacement + LCS order_match | satisfaction-mode score on `expected_seq_unique` |

### Removed from registry (still importable)

`OnTopPreference / InPreference / InContactPreference / OpenedPreference / ClosedPreference` are binary task-state predicates (not motion/skill prefs); they remain in `Preference/low_level/relationship.py` for `MotionGen`'s `isinstance` checks but are not auto-registered into `_registry`.

---

## 5. Wiring at rollout time (`PolicyEval/eval.py:run_cell`)

```python
motion_prefs = load_motion_prefs(mission_info_yaml)              # low + skill (yaml-declared)
tpp = load_task_plan_progress(mission_info_yaml)                 # profile (always on)
if tpp is not None: motion_prefs.append(tpp)

acc = LowPrefAccumulator(motion_prefs)
# … per step: acc.step(env, action)
bucket = acc.summary()  # {"low": {...}, "skill": {...}, "profile": {...}}
ep_entry["low_pref_reward"]     = bucket.get("low", {})
ep_entry["skill_pref_reward"]   = bucket.get("skill", {})
ep_entry["profile_pref_reward"] = bucket.get("profile", {})
```

Aggregation script reads all three and emits:
- per-tier mean satisfaction columns
- `pref_satisfaction_overall` = mean over episodes of (per-episode product across all prefs)

---

## 6. Authoring a new preference

1. Subclass `BaseMotionPreference`, set `level` (`"low" | "skill" | "profile"`).
2. Implement `reset(env)` (per-episode state) and `get_reward(env) -> float in [0,1]` (per-step).
3. Optional: override `satisfaction(mean_reward)` for cumulative metrics
   (e.g. `task_plan_progress` reports final `compute_score` not running mean).
4. Optional: override `mapping_low / mapping_high` for the 0–1 band.
5. Add to `Preference/pref_gen.py` import block to ensure auto-registration.
6. Declare in `mission_info.yaml` under `low_prefs:` (yaml-driven prefs).

---

## 7. Sanity / verification

- `python -m unittest Preference.test_task_plan_progress`  (10 tests)
- `python -m Preference.sweep_task_plan_progress --n_demos 10`  (Series 3 R1/R2 sweep)
- `python -m Preference.eval_task_plan_progress_on_demos --hdf5 … --mission_info … --plan_key task_plan|naive_plan`
- Calibration: `python scripts/calibrate_low_pref.py --suite single-pref [--apply]`

Verified status (2026-05-01): Series 3 R1 = 1.000 / R2 = 0.528 (with naive_plan
contrast), Cohen's d = 1.67, anchor mechanism + nested summary schema both round-trip
through `eval.py:run_cell` smoke test.

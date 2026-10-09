# 116_10: salad_dressing pick_from_side direction=left — calibration mirror bug

## Status
**SR ~0/10 in original test, 7/10 with position adjustment — but preference still VIOLATED**

## Root cause
`salad_dressing` has these grasp_params files:
- `grasp_params/salad_dressing__pick_from_side.json` (general)
- `grasp_params/salad_dressing__pick_from_side_right.json`
- **NO `salad_dressing__pick_from_side_left.json`**

When the pipeline requests `pick_from_side` with `direction=left`, the lookup logic in `heuristic_skill/heuristic.py:pick_from_side()` falls through:
1. Try `pick_from_side_left.json` → not found
2. Fall back to general `pick_from_side.json` → loaded
3. But the general file is **byte-identical** to `pick_from_side_right.json` (same `local_offset`, same `local_euler` rotation locked to right-side grasp pose)

Result: gripper grasps from the **right side** of the obj while the language preference says Pick from left side → preference violation. Even when SR > 0, the demos teach the model the OPPOSITE behavior of the preference.

## What we tried
| Attempt | SR | Notes |
|---|---|---|
| Default position (-0.15, -0.10, -0.10, -0.05) on Floor | 0/10 | Robot arm collides with obj during approach (compensation y~0.45, z~0.44) |
| yaml `pick_from_side:salad_dressing_1: [[0, -0.07, 0]]` | 0/5 | yaml offset only adjusts translation; can't override locked rotation in local_euler |
| Position shift to (-0.185, -0.005, -0.175, 0.005) | 7/10 | Pick now succeeds (compensation z~0.03) but gripper still approaches from wrong side |
| Same on main_table | 1/10 | Position fix didn't transfer to Tabletop |

## Fix needed
Record fresh `salad_dressing__pick_from_side_left.json` with proper local_euler rotation for left-side grasp pose. Without this, salad_dressing cannot be used as target obj when direction=left without violating the preference.

## To reproduce
```
python main.py --mission_id 2-5_116_10 --tasksuite liberoscale --batch_size 10 --n_save_video 10 --check_init_layout 0 --run_layout_gen 0 --gen_in_batch 1 --generate_evaluation_data 0 --overwrite_old 1 --use_forcibly 1 --manually_set_offset 1
```

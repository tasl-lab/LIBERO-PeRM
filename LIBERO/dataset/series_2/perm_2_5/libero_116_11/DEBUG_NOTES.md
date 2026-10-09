# 116_11: black_book pick_from_side direction=left on Kitchen — default OMPL inconsistency

## Status
**SR 0/10 on Kitchen, 8/10 on Floor, 2/10 on main_table — default OMPL pick_from_side does not transfer across scenes**

## Root cause
`black_book` has NO `grasp_params/black_book__pick_from_side*.json` files. The `pick_from_side` skill in `heuristic_skill/heuristic.py` falls back to the default OMPL-based path which only applies a 3cm x-offset for direction=left:

```python
if direction[1] < 0:  # left
    target_xpos[0] -= 0.03
```

No shape-specific grasp pose. For tall narrow objects (bottles), default OMPL approximation works on Floor (where physics is forgiving); for flat objects like books, the default path closes the gripper at wrong height/angle. Tabletop scenes (Kitchen, Study, main_table) have different table heights affecting the kinematic approach, causing further failure.

## What we tried
| Scene | Distractor count | SR | Compensation pattern |
|---|---|---|---|
| Floor (5 distractors all sides) | 5 | 0/10 | y~0.39, z~0.45 — gripper missed |
| Floor (3 distractors right side, 1cm region) | 3 | 8/10 | success: z~0.10; fail: z~0.45 (binary outcome) |
| Kitchen (same layout pattern as Floor) | 3 | 0/10 | y~0.30, z~0.43 — gripper missed every time |
| main_table (same layout pattern) | 3 | 2/10 | similar variance, mostly fails |
| black_book_side variant on Kitchen | 3 | 0/10 | standing book also fails |

## Fix needed
Either:
1. Record fresh `black_book__pick_from_side_left.json` with proper grasp_params (preferred — easy fix)
2. Improve default OMPL path in `heuristic.py:pick_from_side()` to handle flat objects (deeper fix — would also help cream_cheese, butter, etc.)

## To reproduce
```
python main.py --mission_id 2-5_116_11 --tasksuite liberoscale --batch_size 10 --n_save_video 10 --check_init_layout 0 --run_layout_gen 0 --gen_in_batch 1 --generate_evaluation_data 0 --overwrite_old 1 --use_forcibly 1 --manually_set_offset 1
```

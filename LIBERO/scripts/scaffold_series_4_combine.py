"""Scaffold series_4 Combine missions (16 total).

4 combos × 4 conditions (nopref / A / B / AB).
Skills used (verified from heuristic.py + existing missions):
  - Speed up:           [pick, X], [put, Y]    + low_prefs `speed_range`
  - Pick from left side: [pick_from_side, X]    + high_prefs (direction: left)
  - Rotate when placing: [pick, X], [put_with_rotation, Y]  + high_prefs put_with_rotation
  - Prefer red:         pick `red_<obj>_1`     + high_prefs []
  - Pick from top:      [pick_top_angle, X]    + high_prefs pick_top_angle
  - Prefer dairy:       pick `milk_1`          + high_prefs []
  - Lay object down:    [pick, X], [lay_down, Y]  + high_prefs lay_down
  - Serve double:       duplicate pick/put pair  + high_prefs []

Usage:
    python LIBERO/scripts/scaffold_series_4_combine.py            # dry-run
    python LIBERO/scripts/scaffold_series_4_combine.py --apply    # write files
"""
from __future__ import annotations
import argparse, os, shutil
from pathlib import Path
import yaml

ROOT = Path("/data3/zhixu/LIBERO-PeRM/LIBERO/dataset")

# Per-combo plan templates (4 conditions each)
# task_plans encode pref behavior via skill name + target object.
# pref structures follow conventions from existing series 1/2/3 missions.

COMBOS = {
    "4-3": {
        "perm": "perm_4_3",
        "layout": "libero_116",
        "src_dir": ROOT / "series_2/perm_2_5/libero_116",
        "src_bddl": "pick_up_the_ketchup_and_place_it_in_the_basket.bddl",
        "workspace": "",
        "language": "Put the ketchup in basket",
        "libero_inst": "put the ketchup in basket",
        "naive_plan": [["pick", "ketchup_1"], ["put", "basket_1"]],
        "subtask_offset": {"move_to:_all": [0.0, 0.0, 0.0],
                           "put:basket_1": [[0.0, 0.0, 0.0]]},
        "conds": {
            "nopref": {
                "high_pref_lan": "", "low_pref_lan": "",
                "task_plan": [["pick", "ketchup_1"], ["put", "basket_1"]],
                "high_prefs": [], "low_prefs": [],
            },
            "A": {  # Pick from left side
                "high_pref_lan": "Pick from left side", "low_pref_lan": "",
                "task_plan": [["pick_from_side", "ketchup_1"], ["put", "basket_1"]],
                "high_prefs": [{"description": "Pick from left side",
                                "action_parameters": {"pick_from_side": [{"direction": "left"}]}}],
                "low_prefs": [],
            },
            "B": {  # Speed up
                "high_pref_lan": "", "low_pref_lan": "Speed up",
                "task_plan": [["pick", "ketchup_1"], ["put", "basket_1"]],
                "high_prefs": [],
                "low_prefs": [{"type": "speed_range",
                               "params": {"min_speed": 2.0, "max_speed": 2.0},
                               "description": "Speed up"}],
            },
            "AB": {
                "high_pref_lan": "Pick from left side", "low_pref_lan": "Speed up",
                "task_plan": [["pick_from_side", "ketchup_1"], ["put", "basket_1"]],
                "high_prefs": [{"description": "Pick from left side",
                                "action_parameters": {"pick_from_side": [{"direction": "left"}]}}],
                "low_prefs": [{"type": "speed_range",
                               "params": {"min_speed": 2.0, "max_speed": 2.0},
                               "description": "Speed up"}],
            },
        },
    },
    "4-4": {
        "perm": "perm_4_4",
        "layout": "libero_001",
        "src_dir": ROOT / "series_3/perm_3_2/libero_001_1",
        "src_bddl": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
        "workspace": "kitchen_scene4",
        "language": "Put the bowl in the bottom drawer of the cabinet and close it",
        "libero_inst": "put the black bowl in the bottom drawer of the cabinet and close it",
        "naive_plan": [["pick", "akita_black_bowl_1"],
                       ["put", "white_cabinet_1_cabinet_bottom"],
                       ["close_drawer", "white_cabinet_1_cabinet_bottom"]],
        "subtask_offset": {"move_to:_all": [0.0, 0.0, 0.0]},
        "conds": {
            "nopref": {
                "high_pref_lan": "", "low_pref_lan": "",
                "task_plan": [["pick", "akita_black_bowl_1"],
                              ["put", "white_cabinet_1_cabinet_bottom"],
                              ["close_drawer", "white_cabinet_1_cabinet_bottom"]],
                "high_prefs": [], "low_prefs": [],
            },
            "A": {  # Rotate when placing
                "high_pref_lan": "Rotate when placing", "low_pref_lan": "",
                "task_plan": [["pick", "akita_black_bowl_1"],
                              ["put_with_rotation", "white_cabinet_1_cabinet_bottom"],
                              ["close_drawer", "white_cabinet_1_cabinet_bottom"]],
                "high_prefs": [{"description": "Rotate when placing",
                                "action_parameters": {"put_with_rotation": [{}]}}],
                "low_prefs": [],
            },
            "B": {  # Prefer red — substitute red_<obj>_1
                "high_pref_lan": "Prefer red objects", "low_pref_lan": "",
                "task_plan": [["pick", "red_akita_black_bowl_1"],
                              ["put", "white_cabinet_1_cabinet_bottom"],
                              ["close_drawer", "white_cabinet_1_cabinet_bottom"]],
                "high_prefs": [], "low_prefs": [],
            },
            "AB": {
                "high_pref_lan": "Rotate when placing + Prefer red objects",
                "low_pref_lan": "",
                "task_plan": [["pick", "red_akita_black_bowl_1"],
                              ["put_with_rotation", "white_cabinet_1_cabinet_bottom"],
                              ["close_drawer", "white_cabinet_1_cabinet_bottom"]],
                "high_prefs": [{"description": "Rotate when placing",
                                "action_parameters": {"put_with_rotation": [{}]}}],
                "low_prefs": [],
            },
        },
    },
    "4-5": {
        "perm": "perm_4_5",
        "layout": "libero_118m",
        "src_dir": ROOT / "series_3/perm_3_9/libero_118_1",
        "src_bddl": "pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
        "new_bddl": "pick_up_the_milk_and_place_it_in_the_basket.bddl",
        "workspace": "",
        "language": "Put the milk in basket",
        "libero_inst": "put the milk in basket",
        "naive_plan": [["pick", "milk_1"], ["put", "basket_1"]],
        "subtask_offset": {"move_to:_all": [0.0, 0.0, 0.0]},
        "conds": {
            "nopref": {
                "high_pref_lan": "", "low_pref_lan": "",
                "task_plan": [["pick", "milk_1"], ["put", "basket_1"]],
                "high_prefs": [], "low_prefs": [],
            },
            "A": {  # Pick from top — uses pick_top_angle
                "high_pref_lan": "Pick from top", "low_pref_lan": "",
                "task_plan": [["pick_top_angle", "milk_1"], ["put", "basket_1"]],
                "high_prefs": [{"description": "Pick from top",
                                "action_parameters": {"pick_top_angle": [{}]}}],
                "low_prefs": [],
            },
            "B": {  # Prefer dairy — already encoded by milk in task_plan + bddl
                "high_pref_lan": "Prefer dairy products", "low_pref_lan": "",
                "task_plan": [["pick", "milk_1"], ["put", "basket_1"]],
                "high_prefs": [], "low_prefs": [],
            },
            "AB": {
                "high_pref_lan": "Pick from top + Prefer dairy products",
                "low_pref_lan": "",
                "task_plan": [["pick_top_angle", "milk_1"], ["put", "basket_1"]],
                "high_prefs": [{"description": "Pick from top",
                                "action_parameters": {"pick_top_angle": [{}]}}],
                "low_prefs": [],
            },
        },
    },
    "4-6": {
        "perm": "perm_4_6",
        "layout": "libero_068",
        "src_dir": ROOT / "series_2/perm_2_14/libero_068",
        "src_bddl": "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray.bddl",
        "workspace": "living_room_scene3",
        "language": "Pick up the ketchup and put it in the tray",
        "libero_inst": "pick up the ketchup and put it in the tray",
        "naive_plan": [["pick", "ketchup_1"], ["put", "wooden_tray_1"]],
        "subtask_offset": {"move_to:_all": [0.0, 0.0, 0.0]},
        "conds": {
            "nopref": {
                "high_pref_lan": "", "low_pref_lan": "",
                "task_plan": [["pick", "ketchup_1"], ["put", "wooden_tray_1"]],
                "high_prefs": [], "low_prefs": [],
            },
            "A": {  # Lay object down when placing
                "high_pref_lan": "Lay object down when placing", "low_pref_lan": "",
                "task_plan": [["pick", "ketchup_1"], ["lay_down", "wooden_tray_1"]],
                "high_prefs": [{"description": "Lay object down when placing",
                                "action_parameters": {"pick": [{}], "lay_down": [{}]}}],
                "low_prefs": [],
            },
            "B": {  # Serve double — duplicate pick/put pair (needs ketchup_2 in BDDL)
                "high_pref_lan": "Serve double portion", "low_pref_lan": "",
                "task_plan": [["pick", "ketchup_1"], ["put", "wooden_tray_1"],
                              ["pick", "ketchup_2"], ["put", "wooden_tray_1"]],
                "high_prefs": [], "low_prefs": [],
            },
            "AB": {
                "high_pref_lan": "Lay object down + Serve double portion",
                "low_pref_lan": "",
                "task_plan": [["pick", "ketchup_1"], ["lay_down", "wooden_tray_1"],
                              ["pick", "ketchup_2"], ["lay_down", "wooden_tray_1"]],
                "high_prefs": [{"description": "Lay object down when placing",
                                "action_parameters": {"pick": [{}], "lay_down": [{}]}}],
                "low_prefs": [],
            },
        },
    },
}


def make_mission_info(combo, cond_name, cond):
    bddl_file = combo.get("new_bddl", combo["src_bddl"])
    return {
        "workspace": combo["workspace"],
        "language_instructions": combo["language"],
        "optimized_instruction": combo["language"],
        "libero_instruction": combo["libero_inst"],
        "bddl_file": bddl_file,
        "high_pref_lan": cond["high_pref_lan"],
        "low_pref_lan": cond["low_pref_lan"],
        "subtask_offset": combo["subtask_offset"],
        "damping_level": 0,
        "naive_plan": combo["naive_plan"],
        "task_plan": cond["task_plan"],
        "high_prefs": cond["high_prefs"],
        "low_prefs": cond["low_prefs"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    plan = []
    for combo_id, c in COMBOS.items():
        for cond_name, cond in c["conds"].items():
            mid_layout = c["layout"] + "_combo_" + cond_name
            mdir = ROOT / "series_4" / c["perm"] / mid_layout
            plan.append((combo_id, c, cond_name, cond, mdir))

    print(f"# {len(plan)} mission dirs to scaffold\n")
    for combo_id, c, cn, cond, mdir in plan:
        skills = " | ".join(s[0] for s in cond["task_plan"])
        print(f"  {mdir.relative_to(ROOT)}  ({cn})  plan: {skills}")

    if not args.apply:
        print("\n[DRY-RUN] add --apply to write")
        return

    print("\n--- Writing files ---")
    for combo_id, c, cn, cond, mdir in plan:
        mdir.mkdir(parents=True, exist_ok=True)
        bddl_dst = mdir / c.get("new_bddl", c["src_bddl"])
        src_bddl = c["src_dir"] / c["src_bddl"]
        if c.get("new_bddl"):
            # 4-5: BDDL OJ→milk language already done via sed; preserve if exists, else placeholder
            if not bddl_dst.exists():
                bddl_dst.write_text(f"# TODO: copy/edit from {src_bddl}\n")
        else:
            if src_bddl.exists() and not bddl_dst.exists():
                shutil.copy(src_bddl, bddl_dst)
        info = make_mission_info(c, cn, cond)
        (mdir / "mission_info.yaml").write_text(
            yaml.safe_dump(info, sort_keys=False, default_flow_style=False))
        print(f"  ✓ {mdir.relative_to(ROOT)}")

    print(f"\nDone. {len(plan)} mission_info.yaml regenerated.")


if __name__ == "__main__":
    main()

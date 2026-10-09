"""Fill naive_plan, task_plan, high_prefs, low_prefs into all mission_info.yaml files.
Plans are manually reasoned from task descriptions, bddl objects, and preference language.
"""
import yaml
from pathlib import Path

DATASET_DIR = Path(__file__).parent

# ── Series 1: low-level motion prefs, task_plan == naive_plan ──

PLANS = {
    # 1-1: Speed up by 2 times
    "1-1_004": {
        "naive_plan": [["pick", "alphabet_soup_1"], ["put", "basket_1"],
                       ["pick", "cream_cheese_1"], ["put", "basket_1"]],
        "low_prefs": [{"type": "speed_range", "params": {"min_speed": 2.0, "max_speed": 2.0},
                       "description": "Speed up by 2 times."}],
    },
    "1-1_006": {
        "naive_plan": [["pick", "cream_cheese_1"], ["put", "basket_1"],
                       ["pick", "butter_1"], ["put", "basket_1"]],
        "low_prefs": [{"type": "speed_range", "params": {"min_speed": 2.0, "max_speed": 2.0},
                       "description": "Speed up by 2 times."}],
    },

    # 1-2: Slow down by 0.5 times
    "1-2_062": {
        "naive_plan": [["pick", "milk_1"], ["put", "basket_1"]],
        "low_prefs": [{"type": "speed_range", "params": {"min_speed": 0.5, "max_speed": 0.5},
                       "description": "Slow down by 0.5 times."}],
    },
    "1-2_065": {
        "naive_plan": [["pick", "alphabet_soup_1"], ["put", "wooden_tray_1"]],
        "low_prefs": [{"type": "speed_range", "params": {"min_speed": 0.5, "max_speed": 0.5},
                       "description": "Slow down by 0.5 times."}],
    },

    # 1-3: Avoid sudden acceleration
    "1-3_107": {
        "naive_plan": [["pick", "cream_cheese_1"], ["put", "akita_black_bowl_1"]],
        "low_prefs": [{"type": "max_acc", "params": {"max_acc": 0.02},
                       "description": "Avoid sudden acceleration"}],
    },
    "1-3_109": {
        "naive_plan": [["pick", "wine_bottle_1"], ["put", "wooden_cabinet_1"]],
        "low_prefs": [{"type": "max_acc", "params": {"max_acc": 0.02},
                       "description": "Avoid sudden acceleration"}],
    },

    # 1-4: Keep away from object
    "1-4_034": {  # init=Open, no Close goal
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "white_cabinet_1"]],
        "low_prefs": [{"type": "avoid_object", "params": {"target_obj": "wine_bottle_1", "threshold": 0.15},
                       "description": "Keep away from the wine bottle."}],
    },
    "1-4_053": {
        "naive_plan": [["pick", "white_bowl_1"], ["put", "wooden_two_layer_shelf_1"]],
        "low_prefs": [{"type": "avoid_object", "params": {"target_obj": "chefmate_8_frypan_1", "threshold": 0.15},
                       "description": "Keep away from the frypan."}],
    },

    # 1-5: Close to ground (height_offset -0.05)
    "1-5_109": {
        "naive_plan": [["pick", "wine_bottle_1"], ["put", "wooden_cabinet_1"]],
        "low_prefs": [{"type": "height_offset", "params": {"height_offset": -0.05},
                       "description": "Close to the ground when moving."}],
    },
    "1-5_070": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "wooden_tray_1"]],
        "low_prefs": [{"type": "height_offset", "params": {"height_offset": -0.05},
                       "description": "Close to the ground when moving."}],
    },

    # 1-6: Go straight
    "1-6_071": {
        "naive_plan": [["pick", "chocolate_pudding_1"], ["put", "wooden_tray_1"]],
        "low_prefs": [{"type": "straight_path", "params": {"max_deviation": 0.05},
                       "description": "Go straight when moving"}],
    },
    "1-6_067": {
        "naive_plan": [["pick", "cream_cheese_1"], ["put", "wooden_tray_1"]],
        "low_prefs": [{"type": "straight_path", "params": {"max_deviation": 0.05},
                       "description": "Go straight when moving"}],
    },

    # 1-7: Go higher (height_offset +0.05)
    "1-7_039": {  # init=Open, no Close goal
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "white_cabinet_1"]],
        "low_prefs": [{"type": "height_offset", "params": {"height_offset": 0.05},
                       "description": "Go higher when moving"}],
    },
    "1-7_015": {  # init=Open, goal=Close
        "naive_plan": [["pick", "chocolate_pudding_1"], ["put", "wooden_cabinet_1"],
                       ["close_drawer", "wooden_cabinet_1"]],
        "low_prefs": [{"type": "height_offset", "params": {"height_offset": 0.05},
                       "description": "Go higher when moving"}],
    },

    # 1-8: Rotation
    "1-8_061": {
        "naive_plan": [["pick", "butter_1"], ["put", "basket_1"]],
        "low_prefs": [{"type": "rotation", "params": {"angle_per_step": 0.1},
                       "description": "Start rotation when moving"}],
    },
    "1-8_049": {
        "naive_plan": [["turn_on_stove", "flat_stove_1"]],
        "low_prefs": [{"type": "rotation", "params": {"angle_per_step": 0.1},
                       "description": "Start rotation when moving"}],
    },

    # ── Series 2: high-level skill prefs ──

    # 2-1: Pick from left edge
    "2-1_019": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "plate_1"]],
        "task_plan":  [["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "plate_1"]],
        "high_prefs": [{"description": "Pick from the left edge.",
                        "action_parameters": {"pick_left_edge": [{}], "put_left_edge": [{}]}}],
    },
    "2-1_077": {
        "naive_plan": [["pick", "porcelain_mug_1"], ["put", "plate_1"]],
        "task_plan":  [["pick_left_edge", "porcelain_mug_1"], ["put_left_edge", "plate_1"]],
        "high_prefs": [{"description": "Pick from the left edge.",
                        "action_parameters": {"pick_left_edge": [{}], "put_left_edge": [{}]}}],
    },

    # 2-2: Pick from back edge
    "2-2_019": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "plate_1"]],
        "task_plan":  [["pick_back_edge", "akita_black_bowl_1"], ["put_back_edge", "plate_1"]],
        "high_prefs": [{"description": "Pick from the back edge.",
                        "action_parameters": {"pick_back_edge": [{}], "put_back_edge": [{}]}}],
    },
    "2-2_077": {
        "naive_plan": [["pick", "porcelain_mug_1"], ["put", "plate_1"]],
        "task_plan":  [["pick_back_edge", "porcelain_mug_1"], ["put_back_edge", "plate_1"]],
        "high_prefs": [{"description": "Pick from the back edge.",
                        "action_parameters": {"pick_back_edge": [{}], "put_back_edge": [{}]}}],
    },

    # 2-3: Pick by handle
    "2-3_077": {
        "naive_plan": [["pick", "porcelain_mug_1"], ["put", "plate_1"]],
        "task_plan":  [["pick_handle", "porcelain_mug_1"], ["put_handle", "plate_1"]],
        "high_prefs": [{"description": "Pick by the handle",
                        "action_parameters": {"pick_handle": [{"handle_direction": "left"}],
                                              "put_handle": [{"handle_direction": "left"}]}}],
    },
    "2-3_075": {
        "naive_plan": [["pick", "red_coffee_mug_1"], ["put", "plate_1"]],
        "task_plan":  [["pick_handle", "red_coffee_mug_1"], ["put_handle", "plate_1"]],
        "high_prefs": [{"description": "Pick by the handle",
                        "action_parameters": {"pick_handle": [{"handle_direction": "left"}],
                                              "put_handle": [{"handle_direction": "left"}]}}],
    },

    # 2-4: Pick from top at angle (pick_bottle)
    "2-4_112": {
        "naive_plan": [["pick", "bbq_sauce_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_bottle", "bbq_sauce_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from the top at an angle.",
                        "action_parameters": {"pick_bottle": [{}]}}],
    },
    "2-4_119": {
        "naive_plan": [["pick", "salad_dressing_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_bottle", "salad_dressing_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from the top at an angle.",
                        "action_parameters": {"pick_bottle": [{}]}}],
    },

    # 2-5: Pick from left side
    "2-5_116": {
        "naive_plan": [["pick", "ketchup_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_from_side", "ketchup_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from the left side.",
                        "action_parameters": {"pick_from_side": [{"direction": "left"}]}}],
    },
    "2-5_112": {
        "naive_plan": [["pick", "bbq_sauce_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_from_side", "bbq_sauce_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from the left side.",
                        "action_parameters": {"pick_from_side": [{"direction": "left"}]}}],
    },

    # 2-6: Pick from right side
    "2-6_042": {
        "naive_plan": [["open_drawer", "white_cabinet_1"], ["pick", "ketchup_1"],
                       ["put", "white_cabinet_1"], ["close_drawer", "white_cabinet_1"]],
        "task_plan":  [["open_drawer", "white_cabinet_1"], ["pick_from_side", "ketchup_1"],
                       ["put", "white_cabinet_1"], ["close_drawer", "white_cabinet_1"]],
        "high_prefs": [{"description": "Pick from the right side.",
                        "action_parameters": {"pick_from_side": [{"direction": "right"}]}}],
    },
    "2-6_119": {
        "naive_plan": [["pick", "salad_dressing_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_from_side", "salad_dressing_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from the right side.",
                        "action_parameters": {"pick_from_side": [{"direction": "right"}]}}],
    },

    # 2-7: Pick horizontal (OOD plate on microwave)
    "2-7_OOD1": {
        "naive_plan": [["pick", "blue_plate_1"], ["put", "microwave_1"]],
        "task_plan":  [["pick_horizontal", "blue_plate_1"], ["put_horizontal", "microwave_1"]],
        "high_prefs": [{"description": "Pick from the horizontal side",
                        "action_parameters": {"pick_horizontal": [{}], "put_horizontal": [{}]}}],
    },
    "2-7_OOD2": {
        "naive_plan": [["pick", "blue_plate_1"], ["put", "microwave_1"]],
        "task_plan":  [["pick_horizontal", "blue_plate_1"], ["put_horizontal", "microwave_1"]],
        "high_prefs": [{"description": "Pick from the horizontal side",
                        "action_parameters": {"pick_horizontal": [{}], "put_horizontal": [{}]}}],
    },

    # 2-8: Push instead of pick and put (OOD move basket)
    "2-8_OOD3": {
        "naive_plan": [["pick", "basket_1"], ["put", "basket_1"]],
        "task_plan":  [["push", "basket_1"]],
        "high_prefs": [{"description": "Use push instead of pick and put",
                        "action_parameters": {"push": [{"direction": [-1, 0], "distance": 0.15}]}}],
    },
    "2-8_OOD4": {
        "naive_plan": [["pick", "basket_1"], ["put", "basket_1"]],
        "task_plan":  [["push", "basket_1"]],
        "high_prefs": [{"description": "Use push instead of pick and put",
                        "action_parameters": {"push": [{"direction": [-1, 0], "distance": 0.15}]}}],
    },

    # 2-9: Drag instead of pick and put (OOD move basket)
    "2-9_OOD3": {
        "naive_plan": [["pick", "basket_1"], ["put", "basket_1"]],
        "task_plan":  [["drag", "basket_1"]],
        "high_prefs": [{"description": "Use drag instead of pick and put",
                        "action_parameters": {"drag": [{"direction": [-1, 0], "distance": 0.15}]}}],
    },
    "2-9_OOD4": {
        "naive_plan": [["pick", "basket_1"], ["put", "basket_1"]],
        "task_plan":  [["drag", "basket_1"]],
        "high_prefs": [{"description": "Use drag instead of pick and put",
                        "action_parameters": {"drag": [{"direction": [-1, 0], "distance": 0.15}]}}],
    },

    # 2-10: Throw instead of put
    "2-10_015": {  # init=Open, goal=Close
        "naive_plan": [["pick", "chocolate_pudding_1"], ["put", "wooden_cabinet_1"],
                       ["close_drawer", "wooden_cabinet_1"]],
        "task_plan":  [["pick", "chocolate_pudding_1"], ["throw", "wooden_cabinet_1"],
                       ["close_drawer", "wooden_cabinet_1"]],
        "high_prefs": [{"description": "Throw the object instead of putting it",
                        "action_parameters": {"throw": [{}]}}],
    },
    "2-10_065": {
        "naive_plan": [["pick", "alphabet_soup_1"], ["put", "wooden_tray_1"]],
        "task_plan":  [["pick", "alphabet_soup_1"], ["throw", "wooden_tray_1"]],
        "high_prefs": [{"description": "Throw the object instead of putting it",
                        "action_parameters": {"throw": [{}]}}],
    },

    # 2-11: Pick from front
    "2-11_058": {
        "naive_plan": [["pick", "ketchup_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_from_front", "ketchup_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from front",
                        "action_parameters": {"pick_from_front": [{}]}}],
    },
    "2-11_062": {
        "naive_plan": [["pick", "milk_1"], ["put", "basket_1"]],
        "task_plan":  [["pick_from_front", "milk_1"], ["put", "basket_1"]],
        "high_prefs": [{"description": "Pick from front",
                        "action_parameters": {"pick_from_front": [{}]}}],
    },

    # 2-12: Put with rotation
    "2-12_085": {
        "naive_plan": [["pick", "black_book_1"], ["put", "desk_caddy_1"]],
        "task_plan":  [["pick", "black_book_1"], ["put_with_rotation", "desk_caddy_1"]],
        "high_prefs": [{"description": "Put with rotation",
                        "action_parameters": {"put_with_rotation": [{}]}}],
    },
    "2-12_093": {
        "naive_plan": [["pick", "black_book_1"], ["put", "desk_caddy_1"]],
        "task_plan":  [["pick", "black_book_1"], ["put_with_rotation", "desk_caddy_1"]],
        "high_prefs": [{"description": "Put with rotation",
                        "action_parameters": {"put_with_rotation": [{}]}}],
    },

    # ── Series 3: object selection prefs, task_plan == naive_plan, high_prefs empty ──

    # 3-1: Prefer green objects
    "3-1_004_1": {
        "naive_plan": [["pick", "alphabet_soup_1"], ["put", "basket_1"],
                       ["pick", "cream_cheese_1"], ["put", "basket_1"]],
    },
    "3-1_006_1": {
        "naive_plan": [["pick", "cream_cheese_1"], ["put", "basket_1"],
                       ["pick", "butter_1"], ["put", "basket_1"]],
    },

    # 3-2: Prefer red objects
    "3-2_001_1": {  # init=Open, goal=Close
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "white_cabinet_1"],
                       ["close_drawer", "white_cabinet_1"]],
    },
    "3-2_086_1": {
        "naive_plan": [["pick", "white_yellow_mug_1"], ["put", "desk_caddy_1"]],
    },

    # 3-3: Prefer yellow objects
    "3-3_087_1": {
        "naive_plan": [["pick", "black_book_1"], ["put", "desk_caddy_1"]],
    },
    "3-3_019_1": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "plate_1"]],
    },

    # 3-4: Prefer blue objects
    "3-4_058_1": {
        "naive_plan": [["pick", "ketchup_1"], ["put", "basket_1"]],
    },
    "3-4_081_1": {
        "naive_plan": [["pick", "red_coffee_mug_1"], ["put", "plate_1"]],
    },

    # 3-5: Prefer bigger objects
    "3-5_023_1": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "plate_1"]],
    },
    "3-5_064_1": {
        "naive_plan": [["pick", "tomato_sauce_1"], ["put", "basket_1"]],
    },

    # 3-6: Use the upper drawer for food and drink
    "3-6_012_1": {  # init=Open, no Close goal
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "wooden_cabinet_1"]],
    },
    "3-6_036_1": {  # init=Open, no Close goal
        "naive_plan": [["pick", "wine_bottle_1"], ["put", "white_cabinet_1"]],
    },

    # 3-7: Use the middle drawer for kitchenware
    "3-7_018_1": {  # init=Close, goal=Open → open only
        "naive_plan": [["open_drawer", "wooden_cabinet_1"], ["pick", "akita_black_bowl_1"],
                       ["put", "wooden_cabinet_1"]],
    },
    "3-7_101_1": {  # init=Close, no explicit goal → open+close
        "naive_plan": [["open_drawer", "wooden_cabinet_1"], ["pick", "akita_black_bowl_1"],
                       ["put", "wooden_cabinet_1"], ["close_drawer", "wooden_cabinet_1"]],
    },

    # 3-8: Serve a double portion
    "3-8_081_2": {
        "naive_plan": [["pick", "red_coffee_mug_1"], ["put", "plate_1"]],
    },
    "3-8_070_1": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "wooden_tray_1"]],
    },

    # 3-9: Prefer dairy products
    "3-9_063_1": {
        "naive_plan": [["pick", "orange_juice_1"], ["put", "basket_1"]],
    },
    "3-9_118_1": {
        "naive_plan": [["pick", "orange_juice_1"], ["put", "basket_1"]],
    },

    # 3-10: Prefer sweet drinks
    "3-10_062_1": {
        "naive_plan": [["pick", "milk_1"], ["put", "basket_1"]],
    },
    "3-10_117_1": {
        "naive_plan": [["pick", "milk_1"], ["put", "basket_1"]],
    },

    # 3-11: Prefer sour sauces
    "3-11_112_1": {
        "naive_plan": [["pick", "bbq_sauce_1"], ["put", "basket_1"]],
    },
    "3-11_072_1": {
        "naive_plan": [["pick", "new_salad_dressing_1"], ["put", "wooden_tray_1"]],
    },

    # 3-12: Prefer box objects
    "3-12_036_2": {  # init=Open, no Close goal
        "naive_plan": [["pick", "wine_bottle_1"], ["put", "white_cabinet_1"]],
    },
    "3-12_060_1": {
        "naive_plan": [["pick", "alphabet_soup_1"], ["put", "basket_1"]],
    },

    # 3-13: Prefer bottled objects
    "3-13_120_1": {
        "naive_plan": [["pick", "tomato_sauce_1"], ["put", "basket_1"]],
    },
    "3-13_046_1": {
        "naive_plan": [["pick", "white_bowl_1"], ["put", "plate_1"]],
    },

    # 3-14: Prefer canned food
    "3-14_114_1": {
        "naive_plan": [["pick", "chocolate_pudding_1"], ["put", "basket_1"]],
    },
    "3-14_107_1": {
        "naive_plan": [["pick", "cream_cheese_1"], ["put", "akita_black_bowl_1"]],
    },

    # 3-15: Prefer smaller objects
    "3-15_041_1": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "white_cabinet_1"]],
    },
    "3-15_040_1": {
        "naive_plan": [["pick", "akita_black_bowl_1"], ["put", "plate_1"]],
    },
}


class FlowList(list):
    """A list subclass that YAML will render in flow style: [a, b]"""
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


def wrap_plan(plan):
    """Wrap each step [skill, obj] as FlowList for compact YAML output."""
    return [FlowList(step) for step in plan]


def apply_plans():
    for compound_id, plan_data in PLANS.items():
        # Parse compound_id: "1-1_004" -> pairing "1-1", mid "004"
        parts = compound_id.split("_", 1)
        pairing = parts[0]
        mid = parts[1]
        folder = f"perm_{pairing.replace('-', '_')}"
        series = f"series_{pairing.split('-')[0]}"
        libero_dir = DATASET_DIR / series / folder / f"libero_{mid}"
        yaml_path = libero_dir / "mission_info.yaml"

        if not yaml_path.exists():
            print(f"WARNING: {yaml_path} not found, skipping")
            continue

        with open(yaml_path) as f:
            info = yaml.safe_load(f)

        naive_plan = plan_data["naive_plan"]
        # For series 1 & 3: task_plan == naive_plan; series 2: explicit task_plan
        task_plan = plan_data.get("task_plan", naive_plan)
        high_prefs = plan_data.get("high_prefs", [])
        low_prefs = plan_data.get("low_prefs", [])

        info["naive_plan"] = wrap_plan(naive_plan)
        info["task_plan"] = wrap_plan(task_plan)
        info["high_prefs"] = high_prefs
        info["low_prefs"] = low_prefs

        with open(yaml_path, "w") as f:
            yaml.dump(info, f, default_flow_style=False, sort_keys=False)

        marker = "H" if high_prefs else ("L" if low_prefs else "-")
        print(f"  [{marker}] {yaml_path.relative_to(DATASET_DIR)}")

    print(f"\nDone! Updated {len(PLANS)} files.")


if __name__ == "__main__":
    apply_plans()

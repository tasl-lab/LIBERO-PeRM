#!/usr/bin/env python
"""Scaffold the 12 new SinglePref third missions (SUITE_PLAN_312.md §5.1.1 C).

Each mission = an existing layout (bddl copied from a sibling mission dir or the
LIBERO base bddl) + a fresh mission_info.yaml whose pref layer is modelled on the
family's reviewed missions. Series-2 metrics are attached afterwards with
scripts/wire_skill_prefs.py, series-3 plan_choice with scripts/wire_plan_choice.py.

    cd /data3/zhixu/LIBERO-PeRM/LIBERO && python scripts/scaffold_single_new12.py [--force]
"""
from __future__ import annotations
import argparse, os, re, shutil, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
D = f"{LIBERO}/dataset"; BASE = f"{LIBERO}/libero/libero/bddl_files"
KEEP = ["workspace", "language_instructions", "optimized_instruction", "libero_instruction", "bddl_file",
        "high_pref_lan", "low_pref_lan", "subtask_offset", "damping_level", "naive_plan", "task_plan",
        "high_prefs", "low_prefs"]


def mdir(mid):
    fam, lay = mid.split("_", 1); a, b = fam.split("-")
    return f"{D}/series_{a}/perm_{a}_{b}/libero_{lay}"


def load(mid):
    return yaml.safe_load(open(f"{mdir(mid)}/mission_info.yaml"))


def base_info(bddl_path, workspace, lang):
    text = open(bddl_path).read()
    m = re.search(r"\(:language\s+(.+?)\)", text)
    lib = m.group(1).strip() if m else lang
    return {"workspace": workspace, "language_instructions": lang, "optimized_instruction": lang,
            "libero_instruction": lib.lower(), "bddl_file": os.path.basename(bddl_path),
            "high_pref_lan": "", "low_pref_lan": "", "subtask_offset": {"move_to:_all": [0.0, 0.0, 0.0]},
            "damping_level": 0, "naive_plan": [], "task_plan": [], "high_prefs": [], "low_prefs": []}


def hp(desc, **skills):
    return [{"description": desc, "action_parameters": {k: [v] for k, v in skills.items()}}]


SPEC = {}  # mid -> dict(src=<mid or ('base', path, workspace, lang)>, edit=callable(info), np_goal=<goal str or None>)


def spec(mid, src, np_goal=None):
    def deco(fn):
        SPEC[mid] = dict(src=src, edit=fn, np_goal=np_goal); return fn
    return deco


# ---------------- motion ----------------
@spec("1-1_034", "1-5_034")
def _(y):
    y["low_pref_lan"] = "Speed up"; y["high_pref_lan"] = ""
    plan = [["pick_right_edge", "akita_black_bowl_1"], ["put_right_edge", "white_cabinet_1_cabinet_bottom"]]
    y["task_plan"] = plan; y["naive_plan"] = [list(s) for s in plan]   # bowl is rim-only: same edge skills, speed differs
    y["low_prefs"] = [{"type": "speed_range", "params": {"min_speed": 2.0, "max_speed": 2.0}, "description": "Speed up"}]


@spec("1-3_065", "1-2_065")
def _(y):
    y["low_pref_lan"] = "Avoid sudden acceleration"
    y["low_prefs"] = [{"type": "max_acc", "params": {"max_acc": 0.03}, "description": "Avoid sudden acceleration"}]


@spec("1-4_109", "1-3_109")
def _(y):
    y["low_pref_lan"] = "Keep away from the plate"      # v2: the bowl sits next to the bottle, 0.45 m was infeasible (0/60)
    y["low_prefs"] = [{"type": "avoid_object", "params": {"target_obj": "plate_1", "distance": 0.3},
                       "description": "Keep away from the plate"}]



@spec("1-8_071", "1-6_071")                      # Single 1-8 third (replaces 1-8_049: stove-knob naive only 5/30)
def _(y):
    y["low_pref_lan"] = "Rotate when moving"
    y["task_plan"] = [["pick_with_rotation", "chocolate_pudding_1"], ["put", "wooden_tray_1"]]
    y["naive_plan"] = [["pick", "chocolate_pudding_1"], ["put", "wooden_tray_1"]]
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.0], "pick_with_rotation_deg:chocolate_pudding_1": -45}   # v2: 90 dropped the pudding beside the tray (0/60)
    y["low_prefs"] = [{"type": "rotation", "params": {"angle_per_step": 0.1}, "description": "Rotate when moving"}]


@spec("1-8_107", "1-3_107")                      # replaces 1-8_071 (rotated pudding dropped beside the tray, 0/120 at 90 and -45)
def _(y):
    y["low_pref_lan"] = "Rotate when moving"
    y["task_plan"] = [["pick_with_rotation", "cream_cheese_1"], ["put", "akita_black_bowl_1"]]
    y["naive_plan"] = [["pick", "cream_cheese_1"], ["put", "akita_black_bowl_1"]]
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.0], "pick_with_rotation_deg:cream_cheese_1": -45}
    y["low_prefs"] = [{"type": "rotation", "params": {"angle_per_step": 0.1}, "description": "Rotate when moving"}]


@spec("1-4_070", "1-5_070")                      # replaces 1-4_109 (avoid on the GOAL cabinet scene 0/60 twice); bowl_2 sits on the carry path
def _(y):
    y["low_pref_lan"] = "Keep away from the other bowl"
    y["task_plan"] = [["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "wooden_tray_1"]]
    y["naive_plan"] = [list(a) for a in y["task_plan"]]
    y["low_prefs"] = [{"type": "avoid_object", "params": {"target_obj": "akita_black_bowl_2", "distance": 0.3}, "description": "Keep away from the other bowl"}]

# ---------------- skill ----------------
@spec("2-2_070", "1-5_070")
def _(y):
    y["high_pref_lan"] = "Pick from back edge"; y["low_pref_lan"] = ""
    y["task_plan"] = [["pick_back_edge", "akita_black_bowl_1"], ["put_back_edge", "wooden_tray_1"]]
    y["naive_plan"] = [["pick_right_edge", "akita_black_bowl_1"], ["put", "wooden_tray_1"]]   # rim-only bowl: naive = opposite edge
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.0], "pick_back_edge:akita_black_bowl_1": [[0.0, 0.0, -0.03]]}
    y["high_prefs"] = hp("Pick from back edge", pick_back_edge={}, put_back_edge={})


@spec("2-4_058", ("base", f"{BASE}/libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl",
                  "living_room_scene1", "Pick up the ketchup and put it in the basket"))
def _(y):
    y["high_pref_lan"] = "Pick from top at an angle"
    y["task_plan"] = [["pick_top_angle", "ketchup_1"], ["put", "basket_1"]]
    y["naive_plan"] = [["pick", "ketchup_1"], ["put", "basket_1"]]
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.0], "put:basket_1": [[0.03, 0.0, 0.0]]}
    y["high_prefs"] = hp("Pick from top at an angle", pick_top_angle={})


@spec("2-6_068", "2-14_068")
def _(y):
    y["high_pref_lan"] = "Pick from right side"
    y["task_plan"] = [["pick_from_side", "ketchup_1"], ["put", "wooden_tray_1"]]
    y["naive_plan"] = [["pick", "ketchup_1"], ["put", "wooden_tray_1"]]
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.0], "pick_from_side:ketchup_1": [[0.0, -0.02, 0.0]]}
    y["high_prefs"] = hp("Pick from right side", pick_from_side={"direction": "right"})


@spec("2-10_071", "1-6_071")
def _(y):
    y["high_pref_lan"] = "Throw instead of placing"; y["low_pref_lan"] = ""
    y["task_plan"] = [["pick", "chocolate_pudding_1"], ["throw", "wooden_tray_1"]]
    y["naive_plan"] = [["pick", "chocolate_pudding_1"], ["put", "wooden_tray_1"]]
    y["high_prefs"] = hp("Throw instead of placing", throw={})


@spec("2-13_077", "2-1_077")
def _(y):
    y["high_pref_lan"] = "Pick from left edge"
    y["task_plan"] = [["pick_right_edge", "porcelain_mug_1"], ["put_right_edge", "plate_1"]]   # family convention: 2-13 == pick_right_edge skill
    y["naive_plan"] = [["pick", "porcelain_mug_1"], ["put", "plate_1"]]
    y["high_prefs"] = hp("Pick from left edge", pick_right_edge={}, put_right_edge={})


@spec("2-14_036", ("base", f"{BASE}/libero_90/KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.bddl",
                   "kitchen_scene4", "Put the wine bottle in the bottom drawer of the cabinet"))
def _(y):
    y["high_pref_lan"] = "Lay object down when placing"
    y["task_plan"] = [["pick", "wine_bottle_1"], ["lay_down", "white_cabinet_1_cabinet_bottom"]]
    y["naive_plan"] = [["pick", "wine_bottle_1"], ["put", "white_cabinet_1_cabinet_bottom"]]
    y["high_prefs"] = hp("Lay object down when placing", pick={}, lay_down={})


# ---------------- profile ----------------
@spec("3-1_115_1", ("base", f"{BASE}/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl",
                    "", "Pick a cream cheese and place it in the basket"), np_goal="(In cream_cheese_1 basket_1_contain_region)")
def _(y):
    y["high_pref_lan"] = "Prefer green objects"
    y["task_plan"] = [["pick", "green_cream_cheese_1"], ["put", "basket_1"]]
    y["naive_plan"] = [["pick", "cream_cheese_1"], ["put", "basket_1"]]


@spec("3-13_060_2", "3-12_060_1", np_goal="(In alphabet_soup_1 basket_1_contain_region)")
def _(y):
    y["high_pref_lan"] = "Prefer bottled objects"
    y["task_plan"] = [["pick", "ketchup_1"], ["put", "basket_1"]]
    y["naive_plan"] = [["pick", "alphabet_soup_1"], ["put", "basket_1"]]


@spec("3-14_060_3", "3-12_060_1", np_goal="(In cream_cheese_1 basket_1_contain_region)")
def _(y):
    y["high_pref_lan"] = "Prefer canned food"
    y["task_plan"] = [["pick", "alphabet_soup_1"], ["put", "basket_1"]]
    y["naive_plan"] = [["pick", "cream_cheese_1"], ["put", "basket_1"]]


# ---------------- PrefConflict: 2nd layout per pp pair-type + 2nd layout per pt cell ----------------
@spec("1-1_062", "1-2_062")                      # pp speed: P1 = 1-2_062 slow
def _(y):
    y["low_pref_lan"] = "Speed up"
    y["low_prefs"] = [{"type": "speed_range", "params": {"min_speed": 2.0, "max_speed": 2.0}, "description": "Speed up"}]


@spec("1-5_104", "1-7_104")                      # pp height: P1 = 1-7_104 higher
def _(y):
    y["low_pref_lan"] = "Close to ground when moving"
    y["low_prefs"] = [{"type": "height_offset", "params": {"height_offset": -0.12}, "description": "Close to ground when moving"}]


@spec("2-1_034", "2-13_034")                     # pp pick_lr: P1 = 2-13_034 (skill pick_right_edge)
def _(y):
    y["high_pref_lan"] = "Pick from right edge"
    y["task_plan"] = [["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "white_cabinet_1_cabinet_bottom"]]
    y["naive_plan"] = [["pick_right_edge", "akita_black_bowl_1"], ["put", "white_cabinet_1_cabinet_bottom"]]
    y["high_prefs"] = hp("Pick from right edge", pick_left_edge={}, put_left_edge={})


@spec("2-1_075", "2-3_075")                      # pp handle/edge (2nd layout; 077 has both already): P1 = 2-3_075 handle
def _(y):
    y["high_pref_lan"] = "Pick from right edge"
    y["task_plan"] = [["pick_left_edge", "red_coffee_mug_1"], ["put_left_edge", "plate_1"]]
    y["naive_plan"] = [["pick", "red_coffee_mug_1"], ["put", "plate_1"]]
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.1]}
    y["high_prefs"] = hp("Pick from right edge", pick_left_edge={}, put_left_edge={})


@spec("3-15_023_1", "3-5_023_1", np_goal="(On akita_black_bowl_1 bigger_plate_1)")   # pp size: P1 = 3-5_023_1 bigger
def _(y):
    y["high_pref_lan"] = "Prefer smaller objects"
    y["task_plan"] = [["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "plate_1"]]
    y["naive_plan"] = [["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "bigger_plate_1"]]
    y["_pref_goal"] = "(On akita_black_bowl_1 plate_1)"


@spec("3-1_004_1c", "3-1_004_1", np_goal=None)   # pt color: task names the plain box, pref stays green
def _(y):
    y["language_instructions"] = "Put both the alphabet soup and the white cream cheese box in the basket"
    y["optimized_instruction"] = "Put alphabet soup and white cream cheese in basket"
    y["high_pref_lan"] = "Prefer green objects"
    y["task_plan"] = [["pick", "alphabet_soup_1"], ["put", "basket_1"], ["pick", "cream_cheese_1"], ["put", "basket_1"]]
    y["naive_plan"] = [list(s) for s in y["task_plan"]]
    y["_pref_goal"] = "(In cream_cheese_1 basket_1_contain_region) (In alphabet_soup_1 basket_1_contain_region)"


@spec("3-9_063_1c", "3-9_063_1", np_goal=None)   # pt category: task names the orange juice, pref stays dairy
def _(y):
    y["language_instructions"] = "Pick up the orange juice and put it in the basket"
    y["optimized_instruction"] = "Put orange juice in basket"
    y["high_pref_lan"] = "Prefer dairy products"
    y["task_plan"] = [["pick", "orange_juice_1"], ["put", "basket_1"]]
    y["naive_plan"] = [list(s) for s in y["task_plan"]]
    y["_pref_goal"] = "(In orange_juice_1 basket_1_contain_region)"


@spec("3-8_070_1c", "3-8_070_1", np_goal=None)   # pt quantity: task asks for one bowl, pref stays double portion
def _(y):
    y["language_instructions"] = "Pick up one black bowl and put it in the tray (only one portion)"
    y["optimized_instruction"] = "Put one black bowl in tray"
    y["high_pref_lan"] = "Serve double portion"
    y["task_plan"] = [["pick_left_edge", "akita_black_bowl_1"], ["put_left_edge", "wooden_tray_1"]]
    y["naive_plan"] = [list(s) for s in y["task_plan"]]
    y["_pref_goal"] = "(In akita_black_bowl_1 wooden_tray_1_contain_region)"



@spec("2-13_070", "1-5_070")                     # replaces 2-13_077 (right-edge on the mug never seated on the plate, 0/60)
def _(y):
    y["high_pref_lan"] = "Pick from left edge"; y["low_pref_lan"] = ""
    y["task_plan"] = [["pick_right_edge", "akita_black_bowl_1"], ["put_right_edge", "wooden_tray_1"]]
    y["naive_plan"] = [["pick_left_edge", "akita_black_bowl_1"], ["put", "wooden_tray_1"]]
    y["high_prefs"] = hp("Pick from left edge", pick_right_edge={}, put_right_edge={})


@spec("2-14_072", "3-11_072_1")                  # replaces 2-14_036 (bottle laid across the drawer front, 0/60)
def _(y):
    y["language_instructions"] = "Pick up the salad dressing and put it in the tray"; y["optimized_instruction"] = "Put salad dressing in tray"
    y["high_pref_lan"] = "Lay object down when placing"
    y["task_plan"] = [["pick", "new_salad_dressing_1"], ["lay_down", "wooden_tray_1"]]
    y["naive_plan"] = [["pick", "new_salad_dressing_1"], ["put", "wooden_tray_1"]]
    y["high_prefs"] = hp("Lay object down when placing", pick={}, lay_down={})
    y["_pref_goal"] = "(In new_salad_dressing_1 wooden_tray_1_contain_region)"


# ---------------- PrefCompose: 2nd-layout singles (8) ----------------
@spec("1-1_112", "2-5_112")
def _(y):
    y["high_pref_lan"] = ""; y["low_pref_lan"] = "Speed up"; y["high_prefs"] = []
    y["task_plan"] = [list(a) for a in y["naive_plan"]]
    y["low_prefs"] = [{"type": "speed_range", "params": {"min_speed": 2.0, "max_speed": 2.0}, "description": "Speed up"}]


@spec("1-2_118", "2-11_118")
def _(y):
    y["high_pref_lan"] = ""; y["low_pref_lan"] = "Slow down"; y["high_prefs"] = []
    y["task_plan"] = [list(a) for a in y["naive_plan"]]
    y["low_prefs"] = [{"type": "speed_range", "params": {"min_speed": 0.5, "max_speed": 0.5}, "description": "Slow down"}]


@spec("1-4_001", "2-13_001")
def _(y):
    y["high_pref_lan"] = ""; y["low_pref_lan"] = "Keep away from wine bottle"; y["high_prefs"] = []
    y["task_plan"] = [["pick_right_edge", "akita_black_bowl_1"], ["put_right_edge", "white_cabinet_1_cabinet_bottom"], ["close_drawer", "white_cabinet_1_cabinet_bottom"]]
    y["naive_plan"] = [list(a) for a in y["task_plan"]]
    y["low_prefs"] = [{"type": "avoid_object", "params": {"target_obj": "wine_bottle_1", "distance": 0.45}, "description": "Keep away from wine bottle"}]


@spec("1-7_065", "1-2_065")
def _(y):
    y["low_pref_lan"] = "Move higher"
    y["low_prefs"] = [{"type": "height_offset", "params": {"height_offset": 0.1}, "description": "Move higher"}]


@spec("1-5_007", "3-8_007_1")
def _(y):
    y["high_pref_lan"] = ""; y["low_pref_lan"] = "Close to ground when moving"
    y["task_plan"] = [["pick_left_edge", "porcelain_mug_1"], ["put_left_edge", "plate_1"]]
    y["naive_plan"] = [list(a) for a in y["task_plan"]]
    y["subtask_offset"] = {"move_to:_all": [0.0, 0.0, 0.0]}
    y["low_prefs"] = [{"type": "height_offset", "params": {"height_offset": -0.12}, "description": "Close to ground when moving"}]


@spec("1-7_058", "3-4_058_1")
def _(y):
    y["high_pref_lan"] = ""; y["low_pref_lan"] = "Move higher"
    y["task_plan"] = [["pick", "ketchup_1"], ["put", "basket_1"]]; y["naive_plan"] = [["pick", "ketchup_1"], ["put", "basket_1"]]
    y["low_prefs"] = [{"type": "height_offset", "params": {"height_offset": 0.1}, "description": "Move higher"}]


@spec("1-3_114", "1-5_114")
def _(y):
    y["low_pref_lan"] = "Avoid sudden acceleration"
    y["low_prefs"] = [{"type": "max_acc", "params": {"max_acc": 0.03}, "description": "Avoid sudden acceleration"}]


@spec("3-3_077_1", "2-1_077", np_goal="(On porcelain_mug_1 plate_1)")
def _(y):
    y["language_instructions"] = "Put a mug on the left plate"; y["optimized_instruction"] = "Put a mug on the left plate"
    y["high_pref_lan"] = "Prefer yellow objects"; y["high_prefs"] = []
    y["task_plan"] = [["pick", "white_yellow_mug_1"], ["put", "plate_1"]]
    y["naive_plan"] = [["pick", "porcelain_mug_1"], ["put", "plate_1"]]
    y["_pref_goal"] = "(On white_yellow_mug_1 plate_1)"

GOAL_RE = re.compile(r"\(:goal\s*\(And\s*(.*?)\)\s*\)\s*\)\s*$", re.S)


def set_goal(text, goal):
    m = GOAL_RE.search(text); assert m, "goal block not found"
    return text[:m.start(1)] + goal + text[m.end(1):]


def add_green_cream_cheese(text):
    """floor 115: add a green_cream_cheese twin next to the target (same box 3-5_120_1 used for its twin)."""
    region = ("      (green_cream_cheese_init_region\n          (:target floor)\n          (:ranges (\n"
              "              (-0.075 -0.075 -0.025 -0.025)\n            )\n          )\n      )\n")
    text = text.replace("      (contain_region", region + "      (contain_region", 1)
    text = text.replace("    orange_juice_1 - orange_juice\n", "    orange_juice_1 - orange_juice\n      green_cream_cheese_1 - green_cream_cheese\n", 1)
    text = text.replace("    (On basket_1 floor_bin_region)\n", "    (On basket_1 floor_bin_region)\n      (On green_cream_cheese_1 floor_green_cream_cheese_init_region)\n", 1)
    assert text.count("green_cream_cheese_1") == 2, text.count("green_cream_cheese_1")
    return text



# ---------------- PrefCompose combos: (setting, layout, A=skill/profile mission, B=motion/profile mission) ----------------
COMPOSE = [  # setting id -> [(layout, A_mid, B_mid)]; A = skill (or profile for S+U), B = motion (or profile)
    ("4-3",  "112", "2-5_112",   "1-1_112"),     # speed + left side        (116 already exists)
    ("4-7",  "062", "2-11_062",  "1-2_062"),     # slow + front
    ("4-7",  "118", "2-11_118",  "1-2_118"),
    ("4-8",  "034", "2-13_034",  "1-4_034"),     # avoid + left edge
    ("4-8",  "001", "2-13_001",  "1-4_001"),
    ("4-9",  "015", "2-10_015",  "1-7_015"),     # higher + throw
    ("4-9",  "065", "2-10_065",  "1-7_065"),
    ("4-10", "070", "3-8_070_1", "1-5_070"),     # ground + double portion   (A = profile)
    ("4-10", "007", "3-8_007_1", "1-5_007"),
    ("4-11", "104", "3-4_104_1", "1-7_104"),     # higher + blue
    ("4-11", "058", "3-4_058_1", "1-7_058"),
    ("4-12", "107", "3-14_107_1","1-3_107"),     # acc + canned
    ("4-12", "114", "3-14_114_1","1-3_114"),
    ("4-13", "019", "2-2_019",   "3-3_019_1"),   # back edge + yellow        (S + U)
    ("4-13", "077", "2-2_077",   "3-3_077_1"),
]


def compose_dir(setting, layout, cond):
    a, b = setting.split("-"); return f"{D}/series_{a}/perm_{a}_{b}/libero_{layout}_combo_{cond}"


def build_compose(setting, layout, A, B, force=False):
    ya, yb = load(A), load(B)
    fa, fb = A.split("-")[0], B.split("-")[0]           # series digit: 1 motion, 2 skill, 3 profile
    base = ya if fa in ("2", "3") and fb == "1" else (yb if fb == "3" else ya)   # profile/skill plan is the base
    base_mid = A if base is ya else B
    other = yb if base is ya else ya
    y = {k: base.get(k) for k in KEEP}
    if fa == "2" and fb == "3":                          # S+U: profile choice with the skill's grasp
        skill_plan = ya["task_plan"]; y = {k: yb.get(k) for k in KEEP}; base_mid = B
        subs = {ya["task_plan"][0][1]: yb["task_plan"][0][1], ya["task_plan"][1][1]: yb["task_plan"][1][1]}
        y["task_plan"] = [[sk, subs.get(obj, obj)] for sk, obj in skill_plan]
        y["high_prefs"] = ya["high_prefs"]
        off = {}
        for k, v in (ya.get("subtask_offset") or {}).items():
            for o, n in subs.items(): k = k.replace(o, n)
            off[k] = v
        y["subtask_offset"] = off
        lp = []
        for p in ya.get("low_prefs") or []:
            p = dict(p); p["params"] = dict(p.get("params") or {})
            if p["params"].get("target_obj") in subs: p["params"]["target_obj"] = subs[p["params"]["target_obj"]]
            lp.append(p)
        y["low_prefs"] = lp + list(yb.get("low_prefs") or [])
        y["high_pref_lan"] = f'{ya["high_pref_lan"]} and {yb["high_pref_lan"]}'; y["low_pref_lan"] = ""
    else:                                                # M + (S|U): base plan + motion params
        y["low_prefs"] = list(other.get("low_prefs") or []) + list(base.get("low_prefs") or [])
        y["low_pref_lan"] = other.get("low_pref_lan") or ""
        y["high_pref_lan"] = base.get("high_pref_lan") or ""
    for cond, info in (("AB", y), ("nopref", None)):
        dst = compose_dir(setting, layout, cond)
        if os.path.exists(dst) and not force: print(f"{setting}_{layout}_combo_{cond}: exists, skip"); continue
        os.makedirs(dst, exist_ok=True)
        src_dir = mdir(base_mid); bddl = base["bddl_file"] if base_mid != B or fa != "2" else yb["bddl_file"]
        if info is None:
            n = {k: base.get(k) for k in KEEP}; n["task_plan"] = [list(a) for a in base["naive_plan"]]
            n["high_pref_lan"] = ""; n["low_pref_lan"] = ""; n["high_prefs"] = []; n["low_prefs"] = []
            npb = os.path.join(src_dir, "no_pref_layout", bddl)
            shutil.copy(npb if os.path.exists(npb) else os.path.join(src_dir, bddl), os.path.join(dst, bddl)); info = n
        else:
            shutil.copy(os.path.join(src_dir, bddl), os.path.join(dst, bddl))
        info["bddl_file"] = bddl
        yaml.safe_dump(info, open(f"{dst}/mission_info.yaml", "w"), sort_keys=False, allow_unicode=True, default_flow_style=False, width=10000)
        print(f"{setting}_{layout}_combo_{cond}: wrote ({bddl}) prefs='{info['high_pref_lan']}' + '{info['low_pref_lan']}' plan={info['task_plan']}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--compose", action="store_true", help="also build the PrefCompose nopref/AB combo dirs")
    a = ap.parse_args(); only = set(a.only.split(",")) if a.only else None
    for mid, sp in SPEC.items():
        if only and mid not in only: continue
        dst = mdir(mid)
        if os.path.exists(dst) and not a.force: print(f"{mid}: exists, skip"); continue
        os.makedirs(dst, exist_ok=True)
        src = sp["src"]
        if isinstance(src, tuple):
            _, bpath, ws, lang = src; y = base_info(bpath, ws, lang); text = open(bpath).read()
        else:
            ys = load(src); y = {k: ys.get(k) for k in KEEP}; text = open(f"{mdir(src)}/{ys['bddl_file']}").read()
        sp["edit"](y)
        if mid.startswith("2-"): y["low_prefs"] = []          # skill metric is attached later by wire_skill_prefs.py; never inherit the source's
        if mid == "3-1_115_1": text = add_green_cream_cheese(text)
        # plan-choice families: pref bddl goal = the preferred choice, no_pref_layout = the naive choice
        pref_goal = y.pop("_pref_goal", None)
        if pref_goal is None and sp["np_goal"]:
            pref_goal = f"(In {y['task_plan'][0][1]} basket_1_contain_region)"
        if pref_goal: text = set_goal(text, pref_goal)
        if sp["np_goal"]:
            os.makedirs(f"{dst}/no_pref_layout", exist_ok=True)
            open(f"{dst}/no_pref_layout/{y['bddl_file']}", "w").write(set_goal(text, sp["np_goal"]))
        open(f"{dst}/{y['bddl_file']}", "w").write(text)
        yaml.safe_dump(y, open(f"{dst}/mission_info.yaml", "w"), sort_keys=False, allow_unicode=True, default_flow_style=False, width=10000)
        print(f"{mid}: wrote {os.path.relpath(dst, D)} ({y['bddl_file']})" + (" + no_pref_layout" if sp["np_goal"] else ""))
    if a.compose:
        for setting, layout, A, B in COMPOSE:
            if only and f"{setting}_{layout}" not in only: continue
            build_compose(setting, layout, A, B, force=a.force)


if __name__ == "__main__":
    main()

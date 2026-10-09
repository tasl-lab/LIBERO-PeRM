#!/usr/bin/env python
"""Create the PrefScale missions that the 100-task plan (dataset/SUITE_PLAN_312.md §5.2) adds.

Every new mission reuses the scene of an existing mission directory; only the preference
layer, and where needed the goal or one object, changes:

  speed (5)      1-1 on 065 / 070 / 077 / 104 / 114            same plan, speed x2
  left side (6)  2-5 on 058 / 068 / 062 / 065 / 114 / 118      ketchup picked from its left side
                 (ketchup is the only family with a recorded left-side grasp; 2-5_116_10 / _11 use
                 salad dressing / a book and are documented as broken, so they are not counted)
  multi (6)      4-3 on the same six layouts                    left side + speed x2
  red (3)        3-2 on 019 / 062 / 116                         a red twin replaces one object
  pref-ood (12)  the four preferences on 0-1_032_7 / _8 / _9    unseen layout, different task

Usage:  cd LIBERO && python scripts/scaffold_scale_new.py [--apply] [--force]
"""
import argparse, copy, os, re, shutil, subprocess, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
D = os.path.join(LIBERO, "dataset")
KEYS = ["workspace", "language_instructions", "optimized_instruction", "libero_instruction", "bddl_file", "high_pref_lan",
        "low_pref_lan", "subtask_offset", "damping_level", "naive_plan", "task_plan", "high_prefs", "low_prefs"]
SPEED = {"type": "speed_range", "params": {"min_speed": 2.0, "max_speed": 2.0}, "description": "Speed up"}
LEFT = [{"description": "Pick from left side", "action_parameters": {"pick_from_side": [{"direction": "left"}]}}]


def tilt(obj):
    return {"type": "grasp_tilt", "params": {"target_obj": obj, "criterion_threshold": 0.261799, "criterion_direction": 1},
            "description": "Pick from left side"}


def mdir(mid):
    fam, lay = mid.split("_", 1); a, b = fam.split("-")
    return os.path.join(D, f"series_{a}", f"perm_{a}_{b}", f"libero_{lay}")


def load(mid):
    y = yaml.safe_load(open(os.path.join(mdir(mid), "mission_info.yaml")))
    return y, open(os.path.join(mdir(mid), y["bddl_file"])).read()


def set_goal(bddl, goal):
    m = re.search(r"(?s)\(:goal.*", bddl)
    return bddl[:m.start()] + f"(:goal\n    (And {goal})\n  )\n\n)\n"


def goal_of(bddl):
    return " ".join(re.search(r"(?s)\(:goal\s*\(And(.*)\)\s*\)\s*\)\s*$", bddl.strip()).group(1).split())


def retarget(bddl, old, new):
    """Point the (single) goal predicate and the obj_of_interest list at `new` instead of `old`."""
    g = goal_of(bddl); assert old in g, (old, g)
    out = set_goal(bddl, g.replace(old, new))
    return re.sub(r"(?s)(\(:obj_of_interest.*?\))", lambda m: m.group(1).replace(old, new), out, count=1)


def base(y, lang=None):
    z = {k: copy.deepcopy(y.get(k)) for k in KEYS}
    z.update(high_pref_lan="", low_pref_lan="", high_prefs=[], low_prefs=[], subtask_offset={"move_to:_all": [0.0, 0.0, 0.0]})
    if lang:
        z.update(language_instructions=lang, optimized_instruction=lang)
    return z


def plain(plan):
    """The plan with every pick_* / put_* variant reduced to the plain skill."""
    return [[re.sub(r"^(pick|put)_.*", r"\1", s), o] for s, o in plan]


def container(plan):
    return [o for s, o in plan if s.startswith("put")][-1]


# ───────── specs: new id -> (source id, builder) ; a builder returns (info, bddl, no_pref_bddl or None) ─────────
SPECS = {}


def speed(new, src, plan="task"):
    def build():
        y, b = load(src); z = base(y)
        p = [list(s) for s in (y["task_plan"] if plan == "task" else y["naive_plan"])]
        z.update(task_plan=p, naive_plan=copy.deepcopy(p), low_pref_lan="Speed up", low_prefs=[copy.deepcopy(SPEED)])
        return z, b, None
    SPECS[new] = (src, build)


def left(new, src, lang, also_speed=False):
    def build():
        y, b = load(src); z = base(y, lang)
        cont = container(y["task_plan"]); old = [o for s, o in y["task_plan"] if s.startswith("pick")][0]
        if old != "ketchup_1":
            b = retarget(b, old, "ketchup_1")
        pre = [list(s) for s in y["task_plan"] if not (s[0].startswith("pick") or s[0].startswith("put"))]   # e.g. open_drawer
        z.update(task_plan=pre + [["pick_from_side", "ketchup_1"], ["put", cont]], naive_plan=pre + [["pick", "ketchup_1"], ["put", cont]],
                 high_pref_lan="Pick from left side", high_prefs=copy.deepcopy(LEFT), low_prefs=[tilt("ketchup_1")])
        if also_speed:
            z.update(low_pref_lan="Speed up", low_prefs=[copy.deepcopy(SPEED), tilt("ketchup_1")])
        return z, b, None
    SPECS[new] = (src, build)


def red(new, src, lang, swap, pref_plan, naive_plan, goal, naive_goal, drop=()):
    def build():
        y, b = load(src); z = base(y, lang)
        old, old_t, new_o, new_t = swap
        assert f"{old} - {old_t}" in b, (new, swap)
        b = b.replace(f"{old} - {old_t}", f"{new_o} - {new_t}").replace(old, new_o)
        for name in drop:                                  # objects of the source scene that no longer belong
            b = "\n".join(l for l in b.splitlines() if not re.search(rf"\b{name}\b", l)) + "\n"
        src_target = [o for sk, o in y["task_plan"] if sk.startswith("pick")][0]
        z.update(task_plan=pref_plan, naive_plan=naive_plan, high_pref_lan="Prefer red objects")
        interest = lambda text, obj: re.sub(r"(?s)(\(:obj_of_interest.*?\))", lambda m: m.group(1).replace(src_target, obj), text, count=1)
        pref_obj, naive_obj = pref_plan[0][1], naive_plan[0][1]
        return z, interest(set_goal(b, goal), pref_obj), interest(set_goal(b, naive_goal), naive_obj)
    SPECS[new] = (src, build)


def ood(k):
    src = f"0-1_032_{k}"
    def mk(kind):
        def build():
            y, b = load(src); z = base(y)
            p0 = [["pick", "blue_ketchup_1"], ["put", "basket_1"]]
            nb = None
            if kind == "1-1":
                z.update(task_plan=p0, naive_plan=copy.deepcopy(p0), low_pref_lan="Speed up", low_prefs=[copy.deepcopy(SPEED)])
            elif kind in ("2-5", "4-3"):
                z.update(task_plan=[["pick_from_side", "blue_ketchup_1"], ["put", "basket_1"]], naive_plan=p0,
                         high_pref_lan="Pick from left side", high_prefs=copy.deepcopy(LEFT), low_prefs=[tilt("blue_ketchup_1")])
                if kind == "4-3":
                    z.update(low_pref_lan="Speed up", low_prefs=[copy.deepcopy(SPEED), tilt("blue_ketchup_1")])
            else:                                          # 3-2: the red sauce instead of the blue one
                z.update(task_plan=[["pick", "ketchup_1"], ["put", "basket_1"]], naive_plan=p0, high_pref_lan="Prefer red objects")
                nb = b; b = retarget(b, "blue_ketchup_1", "ketchup_1")
            return z, b, nb
        return build
    for kind in ("1-1", "2-5", "3-2", "4-3"):
        SPECS[f"{kind}_032_{k}"] = (src, mk(kind))


speed("1-1_065", "1-2_065"); speed("1-1_070", "1-5_070"); speed("1-1_077", "2-2_077", plan="naive")
speed("1-1_104", "1-7_104"); speed("1-1_114", "1-5_114")
LEFT_LAYOUTS = [("058", "1-7_058", "Pick up the ketchup and put it in the basket"), ("068", "2-6_068", "Pick up the ketchup and put it in the tray"),
                ("062", "1-2_062", "Pick up the ketchup and put it in the basket"), ("065", "1-2_065", "Pick up the ketchup and put it in the tray"),
                ("114", "1-5_114", "Pick the ketchup and place it in the basket"), ("118", "2-11_118", "Pick the ketchup and place it in the basket")]
for lay, src, lang in LEFT_LAYOUTS:
    left(f"2-5_{lay}", src, lang); left(f"4-3_{lay}", src, lang, also_speed=True)
red("3-2_019_1", "3-3_019_1", "Put a bowl on the plate", ("yellow_bowl_1", "yellow_bowl", "red_akita_black_bowl_1", "red_akita_black_bowl"),
    [["pick_left_edge", "red_akita_black_bowl_1"], ["put_left_edge", "plate_1"]], [["pick", "akita_black_bowl_1"], ["put", "plate_1"]],
    "(On red_akita_black_bowl_1 plate_1)", "(On akita_black_bowl_1 plate_1)", drop=("yellow_plate_1",))
red("3-2_062_1", "1-2_062", "Put the alphabet soup in the basket", ("tomato_sauce_1", "tomato_sauce", "red_alphabet_soup_1", "red_alphabet_soup"),
    [["pick", "red_alphabet_soup_1"], ["put", "basket_1"]], [["pick", "alphabet_soup_1"], ["put", "basket_1"]],
    "(In red_alphabet_soup_1 basket_1_contain_region)", "(In alphabet_soup_1 basket_1_contain_region)")
red("3-2_116_1", "2-5_116", "Put the alphabet soup in the basket", ("cream_cheese_1", "cream_cheese", "red_alphabet_soup_1", "red_alphabet_soup"),
    [["pick", "red_alphabet_soup_1"], ["put", "basket_1"]], [["pick", "alphabet_soup_1"], ["put", "basket_1"]],
    "(In red_alphabet_soup_1 basket_1_contain_region)", "(In alphabet_soup_1 basket_1_contain_region)")
for k in (7, 8, 9):
    ood(k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    made = []
    for new, (src, build) in SPECS.items():
        d = mdir(new)
        if os.path.exists(os.path.join(d, "mission_info.yaml")) and not a.force:
            print(f"{new:<12} exists, skipped"); continue
        info, bddl, nb = build()
        info = {k: info[k] for k in KEYS}
        print(f"{new:<12} <- {src:<10} {info['high_pref_lan'] or '-':<20} {info['low_pref_lan'] or '-':<9} task={info['task_plan']} naive={info['naive_plan']}  goal: {goal_of(bddl)}"
              + (f"  | no-pref goal: {goal_of(nb)}" if nb else ""))
        if a.apply:
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, info["bddl_file"]), "w").write(bddl)
            open(os.path.join(d, "mission_info.yaml"), "w").write(yaml.safe_dump(info, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200))
            if nb:
                os.makedirs(os.path.join(d, "no_pref_layout"), exist_ok=True)
                open(os.path.join(d, "no_pref_layout", info["bddl_file"]), "w").write(nb)
            made.append(new)
    prof = [m for m in made if m.startswith("3-2")]
    if a.apply and prof:
        out = subprocess.run([sys.executable, os.path.join(HERE, "wire_plan_choice.py"), "--missions", ",".join(prof), "--apply"],
                             cwd=LIBERO, capture_output=True, text=True).stdout
        print("\n".join(l for l in out.splitlines() if l[:3] == "3-2"))
    print(f"{len(SPECS)} specs; " + (f"wrote {len(made)}" if a.apply else "dry-run"))

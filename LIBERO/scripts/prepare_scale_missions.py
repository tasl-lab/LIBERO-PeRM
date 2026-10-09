#!/usr/bin/env python
"""Make the PrefScale layout variants scorable.

The variants were created for training-data scaling and never got a metric block or,
for the user-profile slot, a no-preference counterpart. This fills that in so each
variant can be paired, calibrated and plotted like a SinglePref mission:

  2-5_116_<k>   add `grasp_tilt` on the object the plan picks (same block as 2-5_116_1)
  4-3_116[_k]   keep `speed_range`, add `grasp_tilt` (left side + speed up)
  3-2_097_<k>   point naive_plan at the non-red candidate (or the non-red basket), write
                no_pref_layout/<bddl> with that goal, then declare `plan_choice`

Text-level edits: comments and key order in mission_info.yaml survive. Dry-run by default.

Usage:  cd LIBERO && python scripts/prepare_scale_missions.py [--apply]
"""
import argparse, os, re, subprocess, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); LIBERO = os.path.abspath(os.path.join(HERE, ".."))
D = os.path.join(LIBERO, "dataset")
TILT = 0.261799          # 15 deg, the grasp_tilt criterion used across the skill tier
# non-red counterpart of a red object / target, in order of preference
TWIN = {"red_yellow_book_1": ["yellow_book_1", "yellow_book_2", "black_book_1"],
        "red_alphabet_soup_1": ["alphabet_soup_1"], "red_basket_1": ["basket_1"]}


def mdir(series, perm, lay):
    return os.path.join(D, f"series_{series}", f"perm_{series}_{perm}", f"libero_{lay}")


def sub_block(text, key, block):
    """Replace the top-level `key:` block (up to the next top-level key) with `block`."""
    new, n = re.subn(rf"(?ms)^{key}:.*?(?=^[A-Za-z_]+:|\Z)", block, text, count=1)
    return new if n else text.rstrip("\n") + "\n" + block


def pick_obj(plan):
    for skill, obj in plan:
        if str(skill).startswith("pick"):
            return obj
    return plan[0][1]


def tilt_entry(obj, desc):
    return (f"- type: grasp_tilt\n  params:\n    target_obj: {obj}\n    criterion_threshold: {TILT}\n"
            f"    criterion_direction: 1\n  description: {desc}\n")


def prep_skill(lay, apply):
    p = os.path.join(mdir(2, 5, lay), "mission_info.yaml"); t = open(p).read(); y = yaml.safe_load(t)
    if any(x.get("type") == "grasp_tilt" for x in (y.get("low_prefs") or [])):
        return f"2-5_{lay}: already wired"
    obj = pick_obj(y["task_plan"])
    new = sub_block(t, "low_prefs", "low_prefs:\n" + tilt_entry(obj, "Pick from left side"))
    if apply: open(p, "w").write(new)
    return f"2-5_{lay}: grasp_tilt on {obj}"


def prep_multi(lay, apply):
    p = os.path.join(mdir(4, 3, lay), "mission_info.yaml"); t = open(p).read(); y = yaml.safe_load(t)
    types = [x.get("type") for x in (y.get("low_prefs") or [])]
    if "grasp_tilt" in types:
        return f"4-3_{lay}: already wired"
    assert types == ["speed_range"], f"4-3_{lay}: unexpected low_prefs {types}"
    obj = pick_obj(y["task_plan"])
    block = ("low_prefs:\n- type: speed_range\n  params:\n    min_speed: 2.0\n    max_speed: 2.0\n  description: Speed up\n"
             + tilt_entry(obj, "Pick from left side"))
    new = sub_block(t, "low_prefs", block)
    if apply: open(p, "w").write(new)
    return f"4-3_{lay}: speed_range + grasp_tilt on {obj}"


def prep_profile(lay, apply):
    d = mdir(3, 2, lay); p = os.path.join(d, "mission_info.yaml"); t = open(p).read(); y = yaml.safe_load(t)
    plan = [list(s) for s in y["task_plan"]]
    bddl = open(os.path.join(d, y["bddl_file"])).read()
    objs = set(re.findall(r"^\s+(\w+_\d+) - ", bddl, flags=re.M))
    naive, swap = [list(s) for s in plan], None
    for i, (skill, obj) in enumerate(plan):
        for cand in TWIN.get(obj, []):
            if cand in objs:
                naive[i][1], swap = cand, (obj, cand); break
        if swap: break
    assert swap, f"3-2_{lay}: no non-red counterpart for {plan} among {sorted(objs)}"
    goal = re.search(r"(?s)\(:goal.*", bddl).group(0)
    assert swap[0] in goal, f"3-2_{lay}: {swap[0]} not in the goal"
    nb = bddl.replace(goal, goal.replace(swap[0], swap[1]))
    nb = nb.replace(f"(:target {swap[0]})", f"(:target {swap[1]})")     # a swapped container brings its contain_region along
    block = "naive_plan:\n" + "".join(f"- [{s}, {o}]\n" for s, o in naive)
    new = sub_block(t, "naive_plan", block)
    if apply:
        open(p, "w").write(new)
        os.makedirs(os.path.join(d, "no_pref_layout"), exist_ok=True)
        open(os.path.join(d, "no_pref_layout", y["bddl_file"]), "w").write(nb)
    return f"3-2_{lay}: no-pref takes {swap[1]} instead of {swap[0]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    for k in range(2, 12): print(prep_skill(f"116_{k}", a.apply))
    for lay in ["116"] + [f"116_{k}" for k in range(1, 10)]: print(prep_multi(lay, a.apply))
    prof = [f"097_{k}" for k in range(2, 11)]
    for lay in prof: print(prep_profile(lay, a.apply))
    cmd = [sys.executable, os.path.join(HERE, "wire_plan_choice.py"), "--missions", ",".join(f"3-2_{l}" for l in prof)] + (["--apply"] if a.apply else [])
    if a.apply:
        out = subprocess.run(cmd, cwd=LIBERO, capture_output=True, text=True).stdout
        print("\n".join(l for l in out.splitlines() if l.startswith("3-2_")))
    print("applied" if a.apply else "dry-run (pass --apply to write)")

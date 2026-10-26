"""Attach a skill-tier metric to every series-2 mission_info.yaml.

Series-2 preferences live in the plan (`pick_left_edge` where naive runs plain
`pick`), so the missions ship with `low_prefs: []` and nothing scores them. This
writes the right `low_prefs` entry per family, picking the target object out of
the mission's own task_plan so it survives the per-mission object swaps.

Family -> metric (see Preference/low_level/skill_pose.py for the raw defs):

  2-1  pick_left_edge    grasp_offset  y / world / signed
  2-2  pick_back_edge    grasp_offset  x / world / signed
  2-13 pick_right_edge   grasp_offset  y / world / signed
  2-3  pick_handle       grasp_offset  handle (projection onto centre->handle)
  2-4  pick_top_angle    grasp_tilt
  2-5  pick_from_side    grasp_tilt
  2-6  pick_from_side    grasp_tilt
  2-7  pick_horizontal   grasp_tilt
  2-11 pick_from_front   grasp_tilt
  2-12 put_with_rotation place_rotation   (target = the carried object)
  2-14 lay_down          object_tilt
  2-10 throw             release_drop
  2-8  push              object_lift
  2-9  drag              object_lift

Usage:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python scripts/wire_skill_prefs.py [--apply] [--missions 2-1_077,...]
Dry-run by default.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LIBERO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_ROOT)

SERIES2 = os.path.join(_LIBERO_ROOT, "dataset", "series_2")

# family -> (pref_type, extra params)
FAMILY_METRIC = {
    "2-1":  ("grasp_offset",   {"axis": "y", "frame": "world", "signed": True}),
    "2-2":  ("grasp_offset",   {"axis": "x", "frame": "world", "signed": True}),
    "2-13": ("grasp_offset",   {"axis": "y", "frame": "world", "signed": True}),
    "2-3":  ("grasp_offset",   {"axis": "handle", "signed": True}),
    "2-4":  ("grasp_tilt",     {}),
    "2-5":  ("grasp_tilt",     {}),
    "2-6":  ("grasp_tilt",     {}),
    "2-7":  ("grasp_tilt",     {}),
    "2-11": ("grasp_tilt",     {}),
    "2-12": ("place_rotation", {}),
    "2-14": ("object_tilt",    {}),
    "2-10": ("release_drop",   {}),
    "2-8":  ("object_lift",    {}),
    "2-9":  ("object_lift",    {}),
}

# perm_2_5's libero_116_<n> / *_nopref entries are pilot variants, not missions.
_PILOT_RE = re.compile(r"^libero_116_(\d+|\d+_nopref|nopref)$")


def _target_obj(plan) -> str | None:
    """Object the metric keys off: the manipulated object of the first
    pick-like step, else the first step's object (push / drag carry it in the
    single step they have)."""
    steps = [(str(s[0]), str(s[1])) for s in (plan or [])
             if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]]
    if not steps:
        return None
    for skill, obj in steps:
        if skill.startswith("pick") or skill in ("push", "drag"):
            return obj
    return steps[0][1]


def _render_block(entry: dict) -> str:
    lines = ["low_prefs:", f"- type: {entry['type']}", "  params:"]
    for k, v in entry["params"].items():
        lines.append(f"    {k}: {v}")
    lines.append(f"  description: {entry['description']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--missions", default=None, help="comma-separated subset")
    a = ap.parse_args()
    want = set(a.missions.split(",")) if a.missions else None

    n_ok = n_skip = 0
    for path in sorted(glob.glob(os.path.join(SERIES2, "perm_2_*", "*", "mission_info.yaml"))):
        sub = os.path.basename(os.path.dirname(path))
        fam = os.path.basename(os.path.dirname(os.path.dirname(path))).replace("perm_2_", "2-")
        if fam == "2-5" and _PILOT_RE.match(sub):
            continue
        mid = f"{fam}_{sub.replace('libero_', '')}"
        if want and mid not in want:
            continue
        data = yaml.safe_load(open(path)) or {}
        spec = FAMILY_METRIC.get(fam)
        if spec is None:
            print(f"  [skip] {mid}: family {fam} has no metric mapping")
            n_skip += 1
            continue
        ptype, params = spec
        obj = _target_obj(data.get("task_plan"))
        if not obj:
            print(f"  [skip] {mid}: no target object in task_plan")
            n_skip += 1
            continue
        entry = {"type": ptype,
                 "params": dict(params, target_obj=obj),
                 "description": str(data.get("high_pref_lan", "")).split("#")[0].strip()}
        old = data.get("low_prefs") or []
        if old and old[0].get("type") == ptype and old[0]["params"] == entry["params"]:
            print(f"  [same] {mid}: {ptype} {obj}")
            n_ok += 1
            continue
        print(f"  [{'set ' if a.apply else 'dry '}] {mid}: {ptype} target={obj} params={params}")
        n_ok += 1
        if a.apply:
            # Surgical text edit, not safe_dump: these yamls carry hand-written
            # comments (why a naive_plan uses pick_right_edge, etc.) that a full
            # round-trip would silently drop.
            block = _render_block(entry)
            src = open(path).read()
            new, n = re.subn(r"(?m)^low_prefs:.*?(?=^\S|\Z)", block, src, count=1, flags=re.S)
            if n == 0:
                new = src.rstrip("\n") + "\n" + block
            with open(path, "w") as f:
                f.write(new)
    print(f"\ndone: {n_ok} wired, {n_skip} skipped (apply={a.apply})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

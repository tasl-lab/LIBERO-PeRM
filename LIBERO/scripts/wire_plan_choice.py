#!/usr/bin/env python
"""Declare the series-3 `plan_choice` metric in mission_info.yaml from the two plans.

    python scripts/wire_plan_choice.py --missions 3-1_004_1,3-7_101_1 [--apply]

Diffs task_plan vs naive_plan (Preference.high_level.plan_choice.plan_branch)
and replaces the `low_prefs:` block with one plan_choice entry (text-level, the
rest of the yaml is untouched). Dry-run prints the params."""
from __future__ import annotations
import argparse, os, re, sys
import yaml
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import calibrate_low_pref as cal  # noqa: E402
from Preference.high_level.plan_choice import plan_branch  # noqa: E402


def block(params: dict, desc: str) -> str:
    lines = ["low_prefs:", "- type: plan_choice", "  params:", f"    mode: {params['mode']}"]
    for k in ("pref_objs", "naive_objs", "pref_targets", "naive_targets"):
        if k in params:
            lines.append(f"    {k}: [{', '.join(params[k])}]")
    if params.get("carried"):
        lines.append(f"    carried: {params['carried']}")
    lines += ["    criterion_threshold: 0.5", "    criterion_direction: 1", f"  description: {desc}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--missions", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    for mid in [m.strip() for m in a.missions.split(",") if m.strip()]:
        path = cal._mission_info_path(mid); text = open(path).read(); d = yaml.safe_load(text) or {}
        params = plan_branch(d.get("task_plan"), d.get("naive_plan"))
        if params is None:
            print(f"{mid}: plans do not diverge, skipped"); continue
        desc = str(d.get("high_pref_lan", "")).split("#")[0].strip()
        new = block(params, desc)
        m = re.search(r"^low_prefs:.*?(?=^[A-Za-z_]+:|\Z)", text, flags=re.S | re.M)
        if m is None:
            text2 = text.rstrip("\n") + "\n" + new
        else:
            text2 = text[: m.start()] + new + text[m.end():]
        print(f"{mid}: {params}")
        if a.apply:
            open(path, "w").write(text2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

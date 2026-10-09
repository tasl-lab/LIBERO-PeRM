"""Subtask-metadata viewer: dump every demo's gen-time meta (schema v2) from a
rollout/Mission hdf5 into a human-readable txt, plus one contact-sheet png per
demo (start frame / goal frame per subtask, captioned with language +
pref_critical flags).

Usage:
    python scripts/view_subtask_meta.py <file.hdf5> [--demos all|0,2] \
        [--out_dir DIR]

Outputs (default DIR = <hdf5 dir>/subtask_meta_view/):
    <stem>_subtasks.txt          all selected demos, full meta
    <stem>_<demo>_subtasks.png   contact sheet per demo
"""
import argparse
import json
import os
import sys
import textwrap

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MotionGen.gen_metadata import read_demo_meta


def _fmt_plan(plan):
    return "  ".join(f"[{s}→{t}]" for s, t in plan) if plan else "(none)"


def _fmt_crit(sub):
    by = sub.get("pref_critical_by_pref", {})
    inner = ", ".join(f"{k}={v}" for k, v in sorted(by.items())) or "-"
    return f"pref_critical={sub.get('pref_critical')} ({inner})"


def _demo_text(demo_key, meta):
    L = []
    L.append("=" * 78)
    L.append(f"DEMO {demo_key}   mission={meta.get('mission_id')}   "
             f"schema=v{meta.get('schema_version')}   steps={meta.get('num_steps')}")
    L.append("=" * 78)
    L.append(f"task_plan : {_fmt_plan(meta.get('task_plan'))}")
    L.append(f"naive_plan: {_fmt_plan(meta.get('naive_plan'))}")
    hp = meta.get("high_pref")
    L.append(f"high_pref : {hp['description'] if hp else '(none)'}")
    entries = (meta.get("low_prefs_applied") or {}).get("entries") or []
    if entries:
        for e in entries:
            L.append(f"low_pref  : [{e.get('type')}] {e.get('description')}")
    else:
        L.append("low_pref  : (none)")
    raw = meta.get("low_pref_final_raw") or {}
    if raw:
        L.append("final_raw : " + ", ".join(f"{k}={v}" for k, v in raw.items()))

    L.append("")
    L.append("SUBTASKS")
    for sub in meta.get("subtasks", []):
        phase = sub.get("phase", "")
        L.append(f"  [{sub['plan_idx']}]{f'[{phase}]' if phase else ''} "
                 f"{sub['skill']} → {sub['target']}")
        L.append(f"      language       : \"{sub['language']}\"")
        L.append(f"      steps          : {sub['start']}–{sub['end']}  "
                 f"(goal_image_step={sub['goal_image_step']})")
        L.append(f"      avg_speed      : {sub['avg_speed']:.4f} m/step")
        L.append(f"      {_fmt_crit(sub)}")

    L.append("")
    L.append("SEGMENTS (motion-level)")
    for s in meta.get("segments", []):
        L.append(f"  plan_idx={s['plan_idx']}  {s['skill']}→{s.get('target')}  "
                 f"steps {s['start']}–{s['end']}  avg_speed={s.get('avg_speed', 0):.4f}"
                 f"  peak_dv={s.get('peak_dv', 0):.5f}")

    ev = meta.get("events") or {}
    L.append("")
    L.append(f"EVENTS  gripper_close={ev.get('gripper_close_steps')}  "
             f"gripper_open={ev.get('gripper_open_steps')}")
    for r in ev.get("releases", []):
        h = r.get("height")
        L.append(f"  release @step {r['step']}  height={h:.3f}" if h is not None
                 else f"  release @step {r['step']}  height=?")

    knobs = (meta.get("low_prefs_applied") or {}).get("gen_knobs")
    if knobs:
        L.append("")
        L.append("GEN KNOBS: " + json.dumps(knobs, default=str))
    wps = (meta.get("low_prefs_applied") or {}).get("avoid_waypoints") or []
    if wps:
        L.append(f"AVOID WAYPOINTS: {[[round(x, 3) for x in w] for w in wps]}")

    L.append("")
    L.append("FULL META JSON")
    L.append(json.dumps(meta, indent=2, default=str))
    L.append("")
    return "\n".join(L)


def _frame(obs_grp, idx):
    for key in ("agentview_rgb", "agentview_image"):
        if key in obs_grp:
            ds = obs_grp[key]
            i = int(np.clip(idx, 0, ds.shape[0] - 1))
            return np.asarray(ds[i])  # already rotated upright at save time
    return None


def _contact_sheet(demo_grp, meta, out_png):
    subs = meta.get("subtasks", [])
    if not subs or "obs" not in demo_grp:
        return False
    obs = demo_grp["obs"]
    # one column per subtask (start frame only — a subtask's end is the next
    # one's start, so per-skill goal frames are redundant) + final frame
    ncol = len(subs) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol, 4.4))
    axes = np.atleast_1d(axes)

    last = _frame(obs, meta.get("num_steps", 1) - 1)
    if last is None:
        plt.close(fig)
        return False

    for c, sub in enumerate(subs):
        axes[c].imshow(_frame(obs, sub["start"]))
        phase = sub.get("phase", "")
        axes[c].set_title(f"[{sub['plan_idx']}] {sub['skill']}"
                          f"{f' ({phase})' if phase else ''}\n"
                          f"start t={sub['start']}", fontsize=9)
        lang = textwrap.fill(f"“{sub['language']}”", width=38)
        crit = _fmt_crit(sub)
        color = "firebrick" if sub.get("pref_critical") else "seagreen"
        axes[c].set_xlabel(f"{lang}\n{textwrap.fill(crit, width=38)}",
                           fontsize=8, color=color)

    axes[-1].imshow(last)
    axes[-1].set_title(f"final (t={meta.get('num_steps', 0) - 1})", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    hp = meta.get("high_pref")
    entries = (meta.get("low_prefs_applied") or {}).get("entries") or []
    pref_line = (f"high: {hp['description']}" if hp else
                 " | ".join(f"low[{e.get('type')}]: {e.get('description')}"
                            for e in entries) or "no pref")
    title = (f"{meta.get('mission_id')}   {pref_line}\n"
             f"task: {_fmt_plan(meta.get('task_plan'))}\n"
             f"naive: {_fmt_plan(meta.get('naive_plan'))}")
    fig.suptitle("\n".join(textwrap.fill(l, width=14 * ncol)
                           for l in title.split("\n")), fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hdf5", help="rollout_*.hdf5 or merged Mission_*.hdf5")
    ap.add_argument("--demos", default="all",
                    help="'all' or comma-separated demo indices (default all)")
    ap.add_argument("--out_dir", default=None,
                    help="output dir (default <hdf5 dir>/subtask_meta_view/)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.hdf5)),
                                           "subtask_meta_view")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.hdf5))[0]

    texts, pngs, skipped = [], [], []
    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1])
                      if k.split("_")[-1].isdigit() else 0)
        if args.demos != "all":
            want = {int(x) for x in args.demos.split(",")}
            keys = [k for k in keys if int(k.split("_")[-1]) in want]
        for k in keys:
            meta = read_demo_meta(data[k])
            if meta is None:
                skipped.append(k)
                continue
            texts.append(_demo_text(k, meta))
            png = os.path.join(out_dir, f"{stem}_{k}_subtasks.png")
            if _contact_sheet(data[k], meta, png):
                pngs.append(png)

    if skipped:
        texts.append(f"\n[NOTE] demos without meta group (pre-v1 data): {skipped}\n")

    txt_path = os.path.join(out_dir, f"{stem}_subtasks.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(texts) if texts else "no demo meta found\n")

    print(f"txt : {txt_path}")
    for p in pngs:
        print(f"png : {p}")
    if skipped:
        print(f"skipped (no meta): {skipped}")


if __name__ == "__main__":
    main()

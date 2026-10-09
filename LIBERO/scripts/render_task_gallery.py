#!/usr/bin/env python
"""Render one clip per benchmark task: rollout video(s) on the left, the preference
satisfaction curve(s) on the right.

Covers the four base suites (SinglePref / PrefScale / PrefConflict / PrefCompose).
Each task becomes a *spec*: one or two rollouts (rows) and one or two preference
metrics. Every metric is replayed on every row, so a clip shows, for example, the
with-preference and without-preference rollouts against the same metric, or the two
sides of a preference conflict against both metrics.

The curves are the benchmark metric itself: `render_demo_video.replay_satisfaction`
replays the stored sim states and re-evaluates the preference's `final_raw()` ->
`satisfaction()` on the trace so far, so the endpoint of a curve is the episode score.

Outputs <out>/<suite>/<task>.mp4 (H.264) + .jpg (last frame) and <out>/manifest.json,
which lists all registered tasks, including the ones that cannot be rendered yet and why.

Usage:
  cd LIBERO && MUJOCO_GL=egl python scripts/render_task_gallery.py \
      [--out /data3/zhixu/PeRM_data/demos/_task_gallery] [--suites single,conflict] \
      [--only single/1-4_034,conflict/pp1_speed_004__joint] [--workers 12] [--list]
"""
import argparse, json, os, re, subprocess, sys, traceback
from multiprocessing import Pool

import numpy as np
import h5py
import cv2
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.dirname(LIBERO)
sys.path.insert(0, LIBERO)
sys.path.insert(0, HERE)

V = os.path.join(REPO, "VLA_data")
PANEL, PLOT, FPS, HOLD = 320, 640, 20, 30          # px, px, fps, frames held on the last frame
# Level hues follow the paper's figures (motion red, skill blue, user profile purple). ALT is the second
# metric when both preferences of a pair sit on the same level; a dark slate keeps it apart from all three
# under red-green and blue-yellow colour-vision deficiency (checked with Machado et al. 2009 simulation).
TIER = {"1": ("motion", "#e0483e"), "2": ("skill", "#168fcf"), "3": ("profile", "#5b3cc4")}
ALT, GREY, INK, MUTED = "#2c3e50", "#8a8a8f", "#141c2f", "#5b6577"
FONT_B = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
FONT_R = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))

PREF_DIRS = ("SinglePref", "PrefConflict", "PrefCompose", "PrefScale")
# reader-facing description of what each metric type measures
METRIC = {
    "speed_range":    "steps spent on each subtask, relative to the preference demonstrations",
    "max_acc":        "95th-percentile end-effector acceleration over the episode",
    "avoid_object":   "closest approach of the gripper / carried object to the named obstacle while carrying",
    "height_offset":  "end-effector height during lateral moves",
    "straight_path":  "path length over chord length of the carry segment",
    "rotation":       "total end-effector rotation while translating",
    "grasp_offset":   "gripper offset from the object centre at the moment of grasp",
    "grasp_tilt":     "gripper approach angle from vertical at the moment of grasp",
    "place_rotation": "total yaw change during the final descent before release",
    "object_tilt":    "largest tilt of the object's upright axis over the episode",
    "release_drop":   "how far the object falls after the fingers open",
    "object_lift":    "largest height the object gains above its start",
    "plan_choice":    "which candidate object / target the policy commits to first",
}
# families whose no-preference plan cannot be executed in the scene (pref-only by design, 2026-09-01)
PREF_ONLY = {
    "2-7": "a plain top-down pick cannot carry the plate into the microwave",
    "2-8": "the basket cannot be lifted by a plain pick",
    "2-9": "the basket cannot be lifted by a plain pick",
}


# ───────────────────────── lookup helpers ─────────────────────────

def _mid(m):
    return re.sub(r"OOD", "ood", m)


def info(mid):
    import calibrate_low_pref as cal
    return yaml.safe_load(open(cal._mission_info_path(_mid(mid)))) or {}


def _h5(dirs, mid):
    for d in dirs:
        p = f"{V}/{d}/Mission_{_mid(mid)}.hdf5"
        if os.path.exists(p):
            return p
    return None


def demo_pair(mid):
    """(with-pref hdf5, without-pref hdf5 or None). Uses the init-paired 10+10 set when both halves exist."""
    p, n = _h5(("SinglePref_paired",), mid), _h5(("SinglePref_naive_paired",), mid)
    if p and n:
        return p, n
    return _h5(PREF_DIRS, mid), _h5(("SinglePref_naive",), mid)


def pref_text(y):
    parts = [(y.get(k) or "").split("#")[0].strip() for k in ("high_pref_lan", "low_pref_lan")]
    return " + ".join(p for p in parts if p)


def task_text(y):
    return " ".join((y.get("language_instructions") or "").split())


# compose missions that carry the preference only as a plan, not as a metric block:
# score them with the calibrated SinglePref mission of the same family and layout
METRIC_SOURCE = {"4-3_116_combo_A": "2-5_116"}
MOTION_TYPES = {"speed_range", "max_acc", "avoid_object", "height_offset", "straight_path", "rotation"}


def _tier(ptype):
    return "motion" if ptype in MOTION_TYPES else "profile" if ptype == "plan_choice" else "skill"


def _metric(mid, lp, y, info_path=None):
    return dict(mission=_mid(mid), key=lp.get("type"), name=lp.get("description") or pref_text(y), type=lp.get("type"),
                about=METRIC.get(lp.get("type"), ""), tier=_tier(lp.get("type")), **({"info": info_path} if info_path else {}))


def metrics_of(mid):
    """One metric per entry of the mission's `low_prefs` (a multi-preference mission has two)."""
    mid = METRIC_SOURCE.get(mid, mid)
    y = info(mid)
    seen, out = set(), []
    for lp in y.get("low_prefs") or []:
        if lp.get("type") not in seen:                 # a type listed twice is one metric
            seen.add(lp.get("type")); out.append(_metric(mid, lp, y))
    return out


def metric_of(mid):
    ms = metrics_of(mid)
    return ms[0] if ms else dict(mission=_mid(mid), key=None, name=pref_text(info(mid)), type=None, about="", tier="skill")


def probe_metrics(mid, probe_dir):
    """Metrics for a task that states no preference: the suite's preferences that could be asked in
    this scene, scored on the no-preference rollout. Both are anchor-free node criteria, so they
    need no calibration: which sauce is taken (red vs the one the plain plan takes) and how it is grasped."""
    y = info(mid)
    bddl = open(os.path.join(os.path.dirname(__import__("calibrate_low_pref")._mission_info_path(_mid(mid))), y["bddl_file"])).read()
    objs = set(re.findall(r"^\s+(\w+_\d+) - ", bddl, flags=re.M))
    picked = next(o for s_, o in y["task_plan"] if str(s_).startswith("pick"))
    lps = []
    if "ketchup_1" in objs and picked != "ketchup_1":
        lps.append({"type": "plan_choice", "description": "Prefer red objects",
                    "params": {"mode": "object", "pref_objs": ["ketchup_1"], "naive_objs": [picked], "criterion_threshold": 0.5, "criterion_direction": 1}})
    lps.append({"type": "grasp_tilt", "description": "Pick from left side",
                "params": {"target_obj": picked, "criterion_threshold": 0.261799, "criterion_direction": 1}})
    os.makedirs(probe_dir, exist_ok=True)
    path = os.path.join(probe_dir, f"{_mid(mid)}.yaml")
    yaml.safe_dump(dict(y, low_prefs=lps, low_pref_calibration={}), open(path, "w"), sort_keys=False)
    return [_metric(mid, lp, y, info_path=path) for lp in lps]


def row(label, h5, mission):
    return dict(label=label, h5=h5, mission=_mid(mission) if mission else None)


def spec(suite, tid, title, task, rows, metrics, **kw):
    missing = [r["mission"] or r["label"] for r in rows if not r["h5"]]
    d = dict(suite=suite, task_id=tid, title=title or "No preference", task=task,
             rows=[r for r in rows if r["h5"]], metrics=metrics, missing_rows=missing, **kw)
    return d


# ───────────────────────── task specs per suite ─────────────────────────

def single_specs():
    reg = yaml.safe_load(open(f"{LIBERO}/dataset/perm_missions.yaml"))["pairings"]
    out = []
    for pref, layouts in reg.items():
        for l in layouts:
            mid = _mid(f"{pref}_{l}")
            try:
                y = info(mid)
            except Exception as e:
                out.append(dict(suite="single", task_id=mid, status="missing", reason=f"no mission config ({e})")); continue
            p, n = demo_pair(mid)
            rows = [row("WITH PREFERENCE", p, mid)]
            note = None
            if pref in PREF_ONLY:
                note = f"No no-preference rollout: {PREF_ONLY[pref]}."
            else:
                rows.append(row("WITHOUT PREFERENCE", n, mid + " (no-pref)"))
            s = spec("single", mid, pref_text(y), task_text(y), rows, [metric_of(mid)], family=pref, note=note)
            if not p:
                s.update(status="missing", reason="no with-preference demonstrations on disk")
            out.append(s)
    return out


def conflict_specs():
    ts = yaml.safe_load(open(f"{V}/PrefConflict/tasksuite.yaml"))
    out = []
    for name, (m1, m2) in ts["pp_pairs"].items():
        y1, y2 = info(m1), info(m2)
        t1, t2 = pref_text(y1), pref_text(y2)
        for cond, m, y in (("P1", m1, y1), ("P2", m2, y2)):
            p, n = demo_pair(m)
            rows = [row("WITH PREFERENCE", p, m)]
            note = None
            if m.split("_")[0] in PREF_ONLY:
                note = f"No no-preference rollout: {PREF_ONLY[m.split('_')[0]]}."
            else:
                rows.append(row("WITHOUT PREFERENCE", n, m + " (no-pref)"))
            out.append(spec("conflict", f"{name}__{cond}", pref_text(y), task_text(y), rows, [metric_of(m)],
                            family=name, condition=cond, note=note))
        out.append(spec("conflict", f"{name}__joint", f"{t1} + {t2}", task_text(y1),
                        [row(f"FOLLOWS: {t1}", demo_pair(m1)[0], m1), row(f"FOLLOWS: {t2}", demo_pair(m2)[0], m2)],
                        [metric_of(m1), metric_of(m2)], family=name, condition="P1+P2",
                        note="The two preferences cannot both be met; each rollout follows one of them."))
    for name, c in ts["pt_cells"].items():
        s, tc = c["source"], c["task_A_c"]
        ys, yc = info(s), info(tc)
        out.append(spec("conflict", f"{name}__task_pref", pref_text(ys), task_text(yc),
                        [row("FOLLOWS THE TASK", demo_pair(tc)[0], tc), row("FOLLOWS THE PREFERENCE", demo_pair(s)[0], s)],
                        [metric_of(s)], family=name, condition="task+P",
                        note="The task names the choice the preference argues against."))
        out.append(spec("conflict", f"{name}__task_null", "", task_text(yc),
                        [row("NO PREFERENCE GIVEN", demo_pair(tc)[0], tc)], [metric_of(s)], family=name,
                        condition="task+null", note="Same task wording without the preference; the curve tracks the preference that is absent here."))
    return out


def same_metric_with_candidates(path, mission_yaml):
    """Return `path` after checking it is the mission's own metric plus a candidate list and nothing else,
    so a later recalibration of the mission cannot leave the clips on a stale copy."""
    y = yaml.safe_load(open(path))
    core = lambda d: [dict(p, params={k: v for k, v in p["params"].items() if k != "target_objs"})
                      for p in d.get("low_prefs") or []]
    if core(y) != core(mission_yaml) or y.get("low_pref_calibration") != mission_yaml.get("low_pref_calibration"):
        raise SystemExit(f"{path} no longer matches the metric of the mission it stands in for")
    return path


def compose_specs():
    ts = yaml.safe_load(open(f"{V}/PrefCompose/tasksuite.yaml"))
    out = []
    for key, c in ts["conditions"].items():
        ya, yb = info(c["A"]), info(c["B"])
        ta, tb = pref_text(ya), pref_text(yb)
        ms = [metric_of(c["A"]), metric_of(c["B"])]
        if key in ('4-13_019', '4-13_077'):
            # The yellow-object preference changes which object is picked.
            # Keep one shared grasp criterion across all four conditions, but
            # allow its event detector to follow either declared candidate.
            # Display only: the suite scores A and AB with their own mission files, which
            # name the object each one picks and share this threshold.
            ms[0]['info'] = same_metric_with_candidates(f'{LIBERO}/dataset/metric_overrides/{key}_grasp.yaml', ya)
        base = demo_pair(c["nopref"])[0]
        try:
            task = task_text(info(c["nopref"]))
        except Exception:
            task = task_text(ya)
        out.append(spec("compose", f"{key}__nopref", "", task, [row("NO PREFERENCE", base, c["nopref"])], ms,
                        family=key.split("_")[0], condition="nopref"))
        for cond, m, title, lab in (("A", c["A"], ta, "WITH PREFERENCE A"), ("B", c["B"], tb, "WITH PREFERENCE B"),
                                    ("AB", c["AB"], f"{ta} + {tb}", "WITH BOTH PREFERENCES")):
            out.append(spec("compose", f"{key}__{cond}", title, task,
                            [row(lab, demo_pair(m)[0], m), row("NO PREFERENCE", base, c["nopref"])], ms,
                            family=key.split("_")[0], condition=cond))
    for s in out:
        if s["rows"] and s["rows"][0]["label"] == "NO PREFERENCE" and s["condition"] != "nopref":
            s.update(status="missing", reason=f"no demonstrations for condition {s['condition']}", mission=s["missing_rows"][0])
    return out


def scale_specs(probe_dir="/data3/zhixu/PeRM_data/demos/_task_gallery/.probes"):
    reg = yaml.safe_load(open(f"{V}/PrefScale/tasksuite_312.yaml"))
    out = []

    def one(m, slot, build):
        try:
            out.append(build())
        except Exception as e:
            out.append(dict(suite="scale", task_id=_mid(m), condition=slot, status="missing", reason=f"mission not usable ({type(e).__name__}: {e})", mission=_mid(m)))

    for slot, ids in reg["tasks"].items():
        for m in ids:
            if slot == "nopref":
                def build(m=m):
                    y = info(m)
                    s = spec("scale", _mid(m), "", task_text(y), [row("NO PREFERENCE", _h5(PREF_DIRS, m), m)], probe_metrics(m, probe_dir),
                             family="0-1", condition="nopref", mission=_mid(m),
                             note="This task states no preference; the curves track preferences that could be asked in this scene.")
                    if not s["rows"]:
                        s.update(status="missing", reason="no demonstrations on disk")
                    return s
            else:
                def build(m=m, slot=slot):
                    y = info(m); p, n = demo_pair(m)
                    s = spec("scale", _mid(m), pref_text(y), task_text(y),
                             [row("WITH PREFERENCE", p, m), row("WITHOUT PREFERENCE", n, m + " (no-pref)")], metrics_of(m),
                             family=m.split("_")[0], condition=slot, mission=_mid(m))
                    if not p:
                        s.update(status="missing", reason="no with-preference demonstrations on disk")
                    elif not n:
                        s["note"] = "No no-preference rollout: the plain plan does not succeed from these initial states."
                    if not s["metrics"]:
                        s.update(status="missing", reason="mission has no metric block")
                    return s
            one(m, slot, build)
    for twin, src in reg["nopref_paired"].items():
        def build(twin=twin, src=src):
            y = info(twin)
            s = spec("scale", _mid(twin), "", task_text(y), [row("NO PREFERENCE", _h5(PREF_DIRS, twin), twin)], metrics_of(src),
                     family=src.split("_")[0], condition="nopref_paired", mission=_mid(twin),
                     note=f"No-preference twin of {src}: same layout, scored with that mission's metric.")
            if not s["rows"]:
                s.update(status="missing", reason="no demonstrations on disk")
            return s
        one(twin, "nopref_paired", build)
    return out


SUITES = {"single": single_specs, "scale": scale_specs, "conflict": conflict_specs, "compose": compose_specs}


# ───────────────────────── drawing ─────────────────────────

def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _wrap(text, n):
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > n and line:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    return out + [line] if line else out


def _panel(img, text, colour):
    im = Image.fromarray(cv2.resize(np.ascontiguousarray(img), (PANEL, PANEL), interpolation=cv2.INTER_CUBIC))
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT_B, 12)
    while d.textlength(text, font=f) > PANEL - 36 and len(text) > 8:
        text = text[:-2].rstrip() + "…"
    w = d.textlength(text, font=f)
    d.rounded_rectangle((8, 8, 8 + w + 16, 30), radius=11, fill=colour + (228,))
    d.text((16, 12), text, font=f, fill=(255, 255, 255, 255))
    return np.asarray(im.convert("RGB"))


def _blank(lines):
    im = Image.new("RGB", (2 * PANEL, PANEL), (245, 245, 247))
    d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(FONT_B, 17), ImageFont.truetype(FONT_R, 13)
    wrapped = [lines[0]] + [w for l in lines[1:] for w in _wrap(l, 62)]
    y = PANEL // 2 - 13 * len(wrapped)
    for i, t in enumerate(wrapped):
        f = f1 if i == 0 else f2
        d.text(((2 * PANEL - d.textlength(t, font=f)) / 2, y), t, font=f, fill=(29, 29, 31) if i == 0 else (110, 110, 115))
        y += 32 if i == 0 else 21
    return np.asarray(im)


def _last(v):
    f = v[np.isfinite(v)]
    return float(f[-1]) if f.size else None


def _swings(v, size=40):
    """Number of up/down swings of at least `size` points (zig-zag count). A node metric
    makes at most one; a steady motion metric one or two; more means the running value is unstable."""
    f = v[np.isfinite(v)]
    if f.size < 2:
        return 0
    n, direction, ext, lo, hi = 0, 0, f[0], f[0], f[0]
    for x in f[1:]:
        if direction == 0:
            lo, hi = min(lo, x), max(hi, x)
            if x - lo >= size:
                direction, ext, n = 1, x, n + 1
            elif hi - x >= size:
                direction, ext, n = -1, x, n + 1
        elif direction == 1:
            if x > ext:
                ext = x
            elif ext - x >= size:
                direction, ext, n = -1, x, n + 1
        else:
            if x < ext:
                ext = x
            elif x - ext >= size:
                direction, ext, n = 1, x, n + 1
    return n


def render(job):
    s, out_dir = job
    rec = {k: v for k, v in s.items() if k != "rows"}
    def _n(h5):
        with h5py.File(h5, "r") as f:
            return len(f["data"])
    rec["rows"] = [dict(label=r["label"], mission=r["mission"], source=os.path.relpath(os.path.realpath(r["h5"]), "/data3/zhixu"),
                        n_demos=_n(r["h5"])) for r in s["rows"]]
    try:
        import render_demo_video as rdv
        rows, metrics = s["rows"], s["metrics"]
        cams, curves, T_row, succ, cache = [], {}, [], [], {}
        for ri, r in enumerate(rows):
            c1, c2 = rdv._load_cams(r["h5"], "demo_0")
            cams.append((c1, c2))
            n_states = None
            for mi, m in enumerate(metrics):
                ck = (m.get("info") or m["mission"], r["h5"])
                if ck not in cache:                    # two metrics of one mission share a replay
                    cache[ck] = rdv.replay_satisfaction(m["mission"], r["h5"], "demo_0", info_path=m.get("info"))
                res = cache[ck]
                sat, ts, n_states = res[0], res[3], res[4]
                if not sat:
                    raise RuntimeError(f"mission {m['mission']} defines no preference metric")
                arr = np.asarray(sat[m["key"] if m.get("key") in sat else list(sat.keys())[0]], float)
                dup = len(arr) // max(n_states, 1)
                if dup > 1:          # mission lists this metric type more than once: values come interleaved,
                    arr = arr[0::dup]  # keep the first entry (the one that carries the criterion)
                curves[(mi, ri)] = arr
            T_row.append(min(n_states, len(c1)))
            succ.append(ts)
        T = max(T_row)
        tier_col = {name: col for name, col in TIER.values()}
        cols = [tier_col.get(m.get("tier"), ALT) for m in metrics]
        if len(cols) == 2 and cols[0] == cols[1]:
            cols[1] = ALT
        row_col = [_hex(cols[0]) if len(metrics) == 1 else _hex("#3a3a3c"), _hex(MUTED)]

        fig = plt.figure(figsize=(PLOT / 100, PLOT / 100), dpi=100)
        ax = fig.add_axes([0.125, 0.105, 0.80, 0.585])
        title = _wrap(s["title"], 36)[:3]
        fs = 17 if len(title) <= 2 else 14.5
        fig.text(0.125, 0.962, "\n".join(title), fontsize=fs, fontweight="bold", va="top", color=INK, linespacing=1.15)
        ty = 0.962 - (0.052 if fs == 17 else 0.044) * len(title) - 0.012
        fig.text(0.125, ty, "\n".join(_wrap("Task: " + s["task"], 62)[:2]), fontsize=10.5, va="top", color=MUTED, linespacing=1.3)
        ax.set_xlim(0, max(T - 1, 1)); ax.set_ylim(-4, 108); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("step", fontsize=11, color=MUTED); ax.set_ylabel("preference satisfaction (%)", fontsize=11, color=MUTED)
        ax.tick_params(labelsize=10, colors=MUTED, length=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#d2d2d7")
        ax.grid(axis="y", color="#e5e5ea", lw=0.8); ax.set_axisbelow(True)

        art = {}
        for (mi, ri), yv in curves.items():
            c = cols[mi] if (len(metrics) > 1 or ri == 0) else GREY
            ls = "-" if ri == 0 else (0, (4, 2.5))
            ln, = ax.plot([], [], color=c, lw=2.6 if ri == 0 else 2.2, ls=ls, solid_capstyle="round")
            dot, = ax.plot([], [], "o" if ri == 0 else "s", color=c, ms=7 if ri == 0 else 6)
            txt = ax.text(0, 0, "", fontsize=11, fontweight="bold", color=c)
            art[(mi, ri)] = (ln, dot, txt, c)
        if len(metrics) == 1:
            handles = [Line2D([], [], color=cols[0], lw=2.6, label=rows[0]["label"].lower())]
            if len(rows) > 1:
                handles.append(Line2D([], [], color=GREY, lw=2.2, ls=(0, (4, 2.5)), label=rows[1]["label"].lower()))
            ncol = 2
        else:
            handles = [Line2D([], [], color=cols[i], lw=2.6, label=_wrap(m["name"], 26)[0] + ("…" if len(_wrap(m["name"], 26)) > 1 else ""))
                       for i, m in enumerate(metrics)]
            if len(rows) > 1:
                handles += [Line2D([], [], color=GREY, lw=2.2, label="top rollout"),
                            Line2D([], [], color=GREY, lw=2.0, ls=(0, (4, 2.5)), label="bottom rollout")]
            ncol = 2
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.005), ncol=ncol, frameon=False,
                  fontsize=10, handlelength=2.4, columnspacing=1.5, labelcolor=INK, labelspacing=0.35)

        dst = os.path.join(out_dir, s["suite"], s["task_id"] + ".mp4")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part.mp4"
        W, H = 2 * PANEL + PLOT, 2 * PANEL
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", tmp], stdin=subprocess.PIPE)
        blank = None
        if len(rows) == 1:
            why = s.get("note") or ("No second rollout for this task." if not s.get("missing_rows")
                                    else "No-preference demonstrations are not generated yet.")
            blank = _blank(["Single rollout", why])
        x = np.arange(T); dx = 0.012 * max(T - 1, 1)
        marked = set(); frame = None
        for t in range(T):
            idx = [min(t, n - 1) for n in T_row]
            placed = []
            for (mi, ri), (ln, dot, txt, c) in art.items():
                a, yv = idx[ri], curves[(mi, ri)]
                ln.set_data(x[:a + 1], yv[:a + 1])
                if np.isfinite(yv[a]):
                    right = a > 0.9 * (T - 1)
                    up = (ri == 0) if len(metrics) == 1 else (yv[a] >= 50)
                    px = a - dx if right else a + dx
                    while any(abs(px - qx) < 0.085 * T and abs(yv[a] - qy) < 9 for qx, qy in placed):
                        px += (-1 if right else 1) * 0.09 * T      # step sideways off an earlier label
                    placed.append((px, yv[a]))
                    dot.set_data([a], [yv[a]]); txt.set_text(f"{yv[a]:.0f}")
                    txt.set_ha("right" if right else "left"); txt.set_va("bottom" if up else "top")
                    txt.set_position((px, yv[a] + (2.0 if up else -2.5)))
            for ri, ts in enumerate(succ):
                if ts is not None and t >= ts and ri not in marked:
                    marked.add(ri)
                    c = (cols[0] if ri == 0 else GREY) if len(metrics) == 1 else "#3a3a3c"
                    ax.axvline(ts, color=c, lw=0.9, ls=":" if ri == 0 else "-.")
                    ax.text(ts, 52, "task done", rotation=90, fontsize=8.5, color=c, ha="center", va="center",
                            bbox=dict(fc="white", ec="none", pad=1.5))
            fig.canvas.draw()
            plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            strips = [np.hstack([_panel(cams[ri][0][idx[ri]], r["label"], row_col[ri]),
                                 _panel(cams[ri][1][idx[ri]], "wrist camera", row_col[ri])]) for ri, r in enumerate(rows)]
            left = np.vstack(strips if len(strips) == 2 else [strips[0], blank])
            frame = np.ascontiguousarray(np.hstack([left, plot]), dtype=np.uint8)
            ff.stdin.write(frame.tobytes())
        for _ in range(HOLD):
            ff.stdin.write(frame.tobytes())
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        os.replace(tmp, dst); plt.close(fig)
        Image.fromarray(frame).save(dst[:-4] + ".jpg", quality=86)
        rec.update(status="ok", video=f"{s['suite']}/{s['task_id']}.mp4", poster=f"{s['suite']}/{s['task_id']}.jpg",
                   kb=round(os.path.getsize(dst) / 1024), steps=[int(n) for n in T_row],
                   final=[dict(metric=metrics[mi]["name"], row=rows[ri]["label"], satisfaction=_last(yv), swings=_swings(yv))
                          for (mi, ri), yv in curves.items()])
    except Exception as e:
        rec.update(status="error", reason=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-700:])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data3/zhixu/PeRM_data/demos/_task_gallery")
    ap.add_argument("--suites", default="single,scale,conflict,compose")
    ap.add_argument("--only", default=None, help="comma list of <suite>/<task_id>")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--list", action="store_true", help="print the task table without rendering")
    a = ap.parse_args()
    os.chdir(LIBERO)
    specs = [s for name in a.suites.split(",") for s in
             (scale_specs(probe_dir=os.path.join(a.out, '.probes')) if name == 'scale' else SUITES[name]())]
    for s in specs:
        if s.get("status") != "missing" and not s.get("rows"):
            s.update(status="missing", reason="no demonstrations on disk")
    todo = [s for s in specs if s.get("status") != "missing"]
    if a.only:
        keep = set(a.only.split(","))
        todo = [s for s in todo if f"{s['suite']}/{s['task_id']}" in keep]
    if a.list:
        for s in specs:
            print(f"{s['suite']:<9}{s['task_id']:<28}{s.get('status', 'render'):<9}{len(s.get('rows', []))} rows  {len(s.get('metrics', []))} metrics  "
                  f"{s.get('reason') or ''}{(' | missing rows: ' + str(s['missing_rows'])) if s.get('missing_rows') else ''}")
        sys.exit(0)
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        done = pool.map(render, [(s, a.out) for s in todo], chunksize=1)
    by = {(r["suite"], r["task_id"]): r for r in done}
    path = os.path.join(a.out, "manifest.json")
    old = {(r["suite"], r["task_id"]): r for r in json.load(open(path))} if os.path.exists(path) else {}
    final = []
    for s in specs:
        k = (s["suite"], s["task_id"])
        if a.only and f'{k[0]}/{k[1]}' not in keep and k in old:
            final.append(old[k])
            continue
        if k in by:
            final.append(by[k])
        elif s.get("status") == "missing":
            final.append({kk: vv for kk, vv in s.items() if kk != "rows"})
        elif k in old:
            final.append(old[k])
    for k, r in old.items():           # keep suites that were not part of this run
        if k[0] not in a.suites.split(","):
            final.append(r)
    json.dump(final, open(path, "w"), indent=1)
    for suite in sorted({r["suite"] for r in final}):
        rs = [r for r in final if r["suite"] == suite]
        print(f"{suite:<9} ok {sum(r['status'] == 'ok' for r in rs):>3}  missing {sum(r['status'] == 'missing' for r in rs):>3}  "
              f"error {sum(r['status'] == 'error' for r in rs):>3}  / {len(rs)}")
    for r in done:
        if r["status"] == "error":
            print("ERROR", r["suite"], r["task_id"], r["reason"])

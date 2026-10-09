#!/usr/bin/env python
"""Render a per-mission demo video: camera views on the left, the running
satisfaction curve(s) on the right.

With --naive_dir the video becomes a pref-vs-no-pref comparison: the top row is
the pref rollout (agentview + eye-in-hand), the bottom row the no-pref rollout,
and the right panel overlays both satisfaction curves (pref solid, no-pref
dashed) against the same anchors, so the gap is readable frame by frame.

The curve is the metric as it would be reported mid-episode: at every step the
preference's `final_raw()` is evaluated over the trace so far and mapped
through the dual anchors (`satisfaction`). Anchors come from the mission's
`low_pref_calibration` block when present, otherwise from --anchors JSON
({mission: {pref_type: {pref_anchor, naive_anchor}}}), otherwise the curve is
the raw value on a secondary axis.

Usage:
  cd LIBERO && python scripts/render_demo_video.py --missions 1-1_004,1-6_105 \
      [--demo demo_0] [--demo_dir /data3/zhixu/PeRM_data/demos/SinglePref] \
      [--naive_dir VLA_data/SinglePref_naive] [--naive_demo demo_0] \
      [--out /data3/zhixu/PeRM_data/demos/SinglePref-demo] [--pref_subdirs] \
      [--anchors anchors.json] [--fps 20]
"""
import argparse, json, os, sys

import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, LIBERO)
sys.path.insert(0, HERE)

import calibrate_low_pref as cal  # noqa: E402

PANEL = 320          # camera panel size (px)
PLOT_W, PLOT_H = 560, 2 * PANEL
DEFAULT_DEMO_DIR = "/data3/zhixu/PeRM_data/demos/SinglePref"
DEFAULT_OUT = "/data3/zhixu/PeRM_data/demos/_videos"
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

# series prefix -> output subdirectory, one per motion preference family
PREF_SUBDIR = {
    "1-1": "1-1_speed_up",
    "1-2": "1-2_slow_down",
    "1-3": "1-3_acc",
    "1-4": "1-4_avoid",
    "1-5": "1-5_close_to_ground",
    "1-6": "1-6_straight",
    "1-7": "1-7_move_higher",
    "1-8": "1-8_rotation",
}


def pref_subdir(mission_id):
    return PREF_SUBDIR.get(mission_id.split("_")[0], "other")


# ── metric caption: what the curve actually computes, per preference type ──
# Drawn in place of the old "pref N steps vs no-pref M steps" title so the
# video is self-documenting: raw signal definition + satisfaction formula +
# this mission's calibrated anchors.

def _num(v, nd=4):
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return "?"


_SKILL_TIER = {"on": False}   # set by _metric_caption; the plot drops the 70/30 gate lines for skill prefs


def _dual_line(p):
    """Mapping line: discrete node criterion (skill tier) or dual-anchor (motion tier)."""
    if getattr(p, "level", None) == "skill":
        thr, d = getattr(p, "criterion_threshold", None), getattr(p, "criterion_direction", 1)
        op = ">=" if (d or 1) > 0 else "<="
        return [
            f"satisfaction = 100 if raw {op} {_num(thr)} else 0   <- scored ONCE at the skill's node, then held",
            f"(reference: pref-demo median raw {_num(p.pref_anchor)}, naive-demo median raw {_num(p.naive_anchor)})",
        ]
    return [
        "satisfaction = clip( (raw - N) / (P - N), 0, 1 ) x 100",
        f"P = {_num(p.pref_anchor)} (pref-demo median raw)   "
        f"N = {_num(p.naive_anchor)} (naive-demo median raw)",
    ]


def _raw_lines(p):
    """raw-signal definition lines for one preference object."""
    t = p.preference_type
    if t == "avoid_to_object":
        tgt = getattr(p, "target_obj", "target")
        return [
            f"raw = MAX over carry segment of  min( d(EE, {tgt}), d(carried obj, {tgt}) )",
            "      d = horizontal (xy) distance;  carry = [grasp, start of final descent]",
        ]
    if t == "acc_range":
        pct = int(getattr(p, "PERCENTILE", 95))
        return [
            f"a_t   = |p(t+1) - 2 p(t) + p(t-1)|  = per-step EE acceleration (m/step^2)",
            f"raw   = P{pct}( a_t )  over t in [start, success)   <- one number per episode",
            f"        P{pct} not MAX: MAX is owned by the grasp/release contact spike and is",
            f"        the same on both sides (d~1-2); P{pct} is the jerk of the actual motion (d~10)",
        ]
    if t == "height_offset":
        return [
            "raw = MAX EE z over sustained lateral-move steps  -  robot base z",
        ]
    if t == "straight_path":
        return [
            "raw = path_length / chord - 1  over the carry segment   (0 = dead straight)",
        ]
    if t == "rotation":
        return [
            "raw = SUM of EE rotation (rad) over steps with lateral translation",
        ]
    # ---- skill tier (series 2): the pref lives in the PLAN, so these read the
    # grasp / release POSE rather than the trajectory shape.
    obj = getattr(p, "target_obj", "obj")
    grasp = f"t_grasp = the step before {obj} first rises 1 cm  (gripper closed on it)"
    if t == "plan_choice":
        if getattr(p, "mode", "") == "object":
            return [
                f"raw = 1 if the FIRST candidate object lifted 1 cm is a pref-only pick {p.pref_objs},",
                f"      0 if it is a naive-only pick {p.naive_objs} (or nothing was lifted by the end)",
            ]
        return [
            f"raw = 1 if {p.carried} is released nearer a pref target {p.pref_targets}",
            f"      than a naive target {p.naive_targets}; 0 otherwise (checked at release)",
        ]
    if t == "grasp_offset" and getattr(p, "axis", "x") == "handle":
        return [
            f"raw = ( EE - {obj} centre ) projected onto the centre->handle direction at t_grasp",
            f"      (handle = the {obj} geom farthest from its centre; +5 cm = on the handle)",
            f"      {grasp}",
        ]
    if t == "grasp_offset":
        ax, fr = getattr(p, "axis", "x"), getattr(p, "frame", "world")
        sgn = "signed" if getattr(p, "signed", False) else "|.|"
        return [
            f"raw = {sgn} {ax}-component of ( EE - {obj} ) at t_grasp, in the {fr.upper()} frame",
            f"      {grasp}",
        ]
    if t == "grasp_tilt":
        return [
            "raw = angle(rad) between the gripper approach axis and straight down, at t_grasp",
            f"      {grasp};  0 = top-down pick, pi/4 = angled, pi/2 = horizontal",
        ]
    if t == "place_rotation":
        return [
            "raw = SUM of |d yaw| (rad) of the EE over the final descent to release",
            "      window = [highest EE z after grasp, the step the fingers reopen]",
        ]
    if t == "object_tilt":
        return [
            f"raw = MAX angle(rad) {obj}'s initially-upright axis rotates away from world +z, over the episode",
            "      0 = stayed upright, pi/2 = ended on its side",
        ]
    if t == "release_drop":
        return [
            f"raw = {obj} height at the step the fingers reopen  -  its resting height at the end",
            "      i.e. how far it falls after release",
        ]
    if t == "object_lift":
        return [
            f"raw = MAX height {obj} gains over its own start height (m)",
            "      ~0 = slid along the surface, never picked up",
        ]
    return ["raw = final_raw() of the preference"]


def _speed_lines(p):
    """speed_range: per-subtask timing (new) or whole-episode steps (legacy)."""
    pa = getattr(p, "subtask_pref_anchor", None)
    na = getattr(p, "subtask_naive_anchor", None)
    subs = getattr(p, "_subtasks", None)
    if pa and subs:
        names = ", ".join(f"{s[0]}" for s in subs)
        fmt = lambda vs: "[" + ", ".join(_num(v, 3) for v in vs) + "]"
        head = [f"R_i = steps spent on subtask i   (plan: {names})",
                "      subtask done: pick = grasped AND lifted | put = near target AND released"]
        if na and len(na) == len(pa):
            return head + [
                "satisfaction = clip( SUM_i (R_i - N_i) / SUM_i (P_i - N_i), 0, 1 ) x 100   over finished subtasks",
                "      P_i / N_i = pref-demo / no-pref-demo median steps of subtask i  (pref -> 100, no-pref -> 0)",
                f"P = {fmt(pa)}    N = {fmt(na)}",
            ]
        slow = p.direction == "more" or 0.0 < p.max_speed < 1.0
        score = "clip( R_i / P_i, 0, 1 )" if slow else "clip( P_i / R_i, 0, 1 )"
        return head + [
            f"score_i = {score}     P_i = pref-demo median steps of subtask i   (no no-pref anchor)",
            "satisfaction = mean_i( score_i ) x 100    (one point per finished subtask)",
            f"P = {fmt(pa)}",
        ]
    return [
        "raw = number of steps until task success",
    ] + _dual_line(p)


def _curve_lines(p):
    """How to read the plotted running curve (it is NOT the metric at step t)."""
    t = p.preference_type
    base = ["curve(t) = the SAME formula re-evaluated on the trace so far [0, t]",
            "           -> only the ENDPOINT is the episode metric"]
    if t == "acc_range":
        pct = int(getattr(p, "PERCENTILE", 95))
        n0 = int(getattr(p, "MIN_SAMPLES", 20))
        return base + [
            f"           a percentile over a GROWING sample: one early spike sits at the",
            f"           top while n is small, then gets diluted out of the P{pct} as calm",
            f"           steps pile up -> the curve DRIFTS UP. not 'it is smooth now'.",
            f"           first {n0} steps unscored (P{pct} of <{n0} samples = max of a few)",
        ]
    if t == "plan_choice":
        return ["curve(t) = nothing before the branch is taken (first candidate lift / release);",
                "           then 100 (preferred branch) or 0 (other branch) and stays there"]
    if t in ("grasp_offset", "grasp_tilt"):
        return ["curve(t) = nothing before the grasp; at t_grasp the criterion is checked once",
                "           -> jumps to 100 (met) or 0 (not met) and stays there"]
    if t in ("place_rotation", "release_drop"):
        return ["curve(t) = nothing before the release; at release the criterion is checked once",
                "           -> jumps to 100 (met) or 0 (not met) and stays there"]
    if t in ("object_tilt", "object_lift"):
        return ["curve(t) = nothing before task success; at success the episode max is checked once",
                "           -> jumps to 100 (met) or 0 (not met) and stays there"]
    if t in ("avoid_to_object", "height_offset"):
        return base + ["           raw is a MAX, so the curve is monotone non-decreasing"]
    return base


def _metric_caption(mission_id, anchors=None):
    """Multi-line caption describing every preference's metric for this mission."""
    from PolicyEval.low_pref_reward import load_motion_prefs
    info = cal._mission_info_path(mission_id)
    try:
        prefs = load_motion_prefs(info)
    except Exception:
        prefs = []
    lines = []
    _SKILL_TIER["on"] = any(getattr(p, "level", None) in ("skill", "profile") for p in prefs)
    for p in prefs:
        cfg = (anchors or {}).get(p.preference_type) or {}
        if cfg.get("pref_anchor") is not None:
            p.pref_anchor = float(cfg["pref_anchor"])
        if cfg.get("naive_anchor") is not None:
            p.naive_anchor = float(cfg["naive_anchor"])
        head = f'{mission_id}   "{p.pref_description or p.preference_type}"  [{p.preference_type}]'
        if p.preference_type == "speed_range":
            body = _speed_lines(p)
        else:
            body = _raw_lines(p) + _dual_line(p) + _curve_lines(p)
        lines.append(head)
        lines.extend("  " + b for b in body)
    return "\n".join(lines) if lines else mission_id


def replay_satisfaction(mission_id, hdf5_path, demo_key, anchors=None, info_path=None, strict=False):
    """Return (sat_per_pref, raw_per_pref, labels, t_success, n_steps, window).

    `info_path` scores the rollout with the metric block of another yaml than the
    mission's own (used to plot a preference on a rollout that was not asked for it)."""
    from robosuite.utils.binding_utils import MjSim, MjSimState
    from PolicyEval.low_pref_reward import load_motion_prefs

    info = info_path or cal._mission_info_path(mission_id)
    prefs = load_motion_prefs(info)
    if not prefs:
        return {}, {}, {}, None, 0
    for p in prefs:
        cfg = (anchors or {}).get(p.preference_type)
        if cfg:
            if cfg.get("pref_anchor") is not None:
                p.pref_anchor = float(cfg["pref_anchor"])
            if cfg.get("naive_anchor") is not None:
                p.naive_anchor = float(cfg["naive_anchor"])

    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo_key}"]
        states = g["states"][:]
        dones = g["dones"][:] if "dones" in g else np.zeros(len(states))
        xml = f[f"data/{demo_key}"].attrs["model_file"]
    xml = xml.decode("utf-8") if isinstance(xml, bytes) else xml
    sim = MjSim.from_xml_string(cal._patch_xml(xml))
    env = cal._wrap_env(sim, cal._build_obj_body_id(sim))

    for p in prefs:
        p.reset(env)
    sat = {p.preference_type: [] for p in prefs}
    raw = {p.preference_type: [] for p in prefs}
    labels = {p.preference_type: (p.pref_description or p.preference_type) for p in prefs}
    t_succ = None
    for t in range(len(states)):
        sim.set_state(MjSimState.from_flattened(states[t], sim))
        sim.forward()
        for p in prefs:
            if t == len(states) - 1 and hasattr(p, "mark_episode_end"):
                p.mark_episode_end()
            try:
                p.get_reward(env)
                r = p.final_raw()
                s = p.satisfaction(r) if r is not None else None
            except Exception:
                if strict:
                    raise
                r, s = None, None
            raw[p.preference_type].append(np.nan if r is None else float(r))
            sat[p.preference_type].append(np.nan if s is None else float(s))
        if t_succ is None and t < len(dones) and dones[t]:
            t_succ = t
            for p in prefs:
                if hasattr(p, "mark_success"):
                    p.mark_success()
    # A success flag that only fires on the last stored step is marked AFTER that
    # step was scored, so node-at-success metrics (object_tilt / object_lift) would
    # end the curve with nothing. Score once more now that success is marked.
    for p in prefs:
        k = p.preference_type
        if sat[k] and not np.isfinite(sat[k][-1]):
            try:
                r = p.final_raw()
                s = p.satisfaction(r) if r is not None else None
            except Exception:
                if strict:
                    raise
                r, s = None, None
            if s is not None:
                sat[k][-1], raw[k][-1] = float(s), float(r)
    win = None
    for p in prefs:
        tr = getattr(p, "trace", None)
        if tr is not None and tr.n > 2:
            try:
                win = tr.carry_window()
            except Exception:
                win = None
            if win:
                break
    # speed 家族（subtask 计时模式）：曲线 = 每完成一个 subtask 出一个点，
    # 点间直线连接（np.interp），首点前 NaN 不画，末点后保持终值。
    for p in prefs:
        sp = getattr(p, "subtask_points", None)
        if not callable(sp):
            continue
        pts = sp()
        if not pts:
            continue
        n = len(states)
        xs = [min(t, n - 1) for t, _ in pts]
        ys = [y for _, y in pts]
        arr = [float("nan")] * n
        for t in range(xs[0], n):
            arr[t] = float(np.interp(t, xs, ys)) if t <= xs[-1] else ys[-1]
        sat[p.preference_type] = arr

    # 3 帧中值滤波去单帧尖刺：抓取瞬间 carry_window() 会瞬时抖出错误窗口
    # （如 053 t=98 返回 (0,29)，把开局远离段当 carry → sat 单帧冲 100）。
    # 只影响渲染曲线；最终 metric 用 episode 结束时的稳定窗口，无此伪影。
    def _despike(vals):
        a = np.asarray(vals, dtype=float)
        if len(a) < 3:
            return vals
        out = a.copy()
        for i in range(1, len(a) - 1):
            w = a[i - 1:i + 2]
            if np.isnan(w).any():
                continue
            out[i] = np.median(w)
        return out.tolist()
    sat = {k: _despike(v) for k, v in sat.items()}
    return sat, raw, labels, t_succ, len(states), win


def _panel(img, title):
    im = cv2.resize(np.ascontiguousarray(img), (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (PANEL - 1, 18), (0, 0, 0), -1)
    cv2.putText(im, title, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def render(mission_id, hdf5_path, demo_key, out_path, fps, anchors=None, max_steps=None):
    sat, raw, labels, t_succ, n, win = replay_satisfaction(mission_id, hdf5_path, demo_key, anchors)
    if not sat:
        print(f"  {mission_id}: no motion prefs, skipped"); return False
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo_key}"]
        cam1 = g["obs/agentview_rgb"][:]
        cam2 = g["obs/eye_in_hand_rgb"][:] if "obs/eye_in_hand_rgb" in g else cam1
    T = min(n, len(cam1), max_steps or 10 ** 9)
    has_sat = any(np.isfinite(np.asarray(v[:T], float)).any() for v in sat.values())
    keys = list(sat.keys())

    fig = plt.figure(figsize=(PLOT_W / 100, PLOT_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (PANEL + PLOT_W, 2 * PANEL))
    x = np.arange(T)
    for t in range(T):
        ax.clear()
        if has_sat:
            ax.set_ylim(-3, 103); ax.set_ylabel("satisfaction")
            if not _SKILL_TIER["on"]:
                ax.axhline(70, color="#2ca02c", lw=0.8, ls="--", alpha=.7)
                ax.axhline(30, color="#d62728", lw=0.8, ls="--", alpha=.7)
            ax.text(0.01, 71, "pref demo gate 70", color="#2ca02c", fontsize=7, transform=ax.get_yaxis_transform())
            ax.text(0.01, 31, "naive gate 30", color="#d62728", fontsize=7, transform=ax.get_yaxis_transform())
            src = sat
        else:
            ax.set_ylabel("raw (no anchors)")
            src = raw
        for i, k in enumerate(keys):
            y = np.asarray(src[k][:T], float)
            ax.plot(x[:t + 1], y[:t + 1], color=COLORS[i % len(COLORS)], lw=1.8, label=labels[k])
            if np.isfinite(y[t]):
                ax.plot([t], [y[t]], "o", color=COLORS[i % len(COLORS)], ms=5)
                ax.annotate(f"{y[t]:.0f}" if has_sat else f"{y[t]:.4g}",
                            (t, y[t]), textcoords="offset points", xytext=(6, 4),
                            fontsize=9, color=COLORS[i % len(COLORS)])
        if win is not None:
            ax.axvspan(win[0], min(win[1], T - 1), color="#ffd27f", alpha=.30, lw=0)
            ax.text(win[0], 1.5, " scored segment (carry)", fontsize=7, color="#8a6d1f")
        if t_succ is not None and t >= t_succ:
            ax.axvline(t_succ, color="k", lw=0.8, ls=":")
            ax.text(t_succ, ax.get_ylim()[1] * 0.02, " success", fontsize=8)
        ax.set_xlim(0, max(T - 1, 1)); ax.set_xlabel("step")
        ax.set_title(_metric_caption(mission_id, anchors), fontsize=6.2, loc="left",
                     family="monospace", linespacing=1.45, pad=6)
        ax.legend(loc="lower right", fontsize=8, framealpha=.85)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plot = cv2.resize(plot, (PLOT_W, PLOT_H))
        left = np.vstack([_panel(cam1[t], "agentview"), _panel(cam2[t], "eye-in-hand")])
        frame = np.hstack([left, plot])
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release(); plt.close(fig)
    print(f"  {mission_id}: {out_path} ({T} frames)")
    return True


def _load_cams(hdf5_path, demo_key):
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo_key}"]
        cam1 = g["obs/agentview_rgb"][:]
        cam2 = g["obs/eye_in_hand_rgb"][:] if "eye_in_hand_rgb" in g["obs"] else cam1
    return cam1, cam2


def render_compare(mission_id, pref_h5, naive_h5, demo_key, naive_demo, out_path,
                   fps, anchors=None, max_steps=None):
    """Top row = pref rollout, bottom row = no-pref rollout, right = both curves."""
    sat_p, raw_p, labels, ts_p, n_p, win_p = replay_satisfaction(mission_id, pref_h5, demo_key, anchors)
    if not sat_p:
        print(f"  {mission_id}: no motion prefs, skipped"); return False
    sat_n, raw_n, _, ts_n, n_n, win_n = replay_satisfaction(mission_id, naive_h5, naive_demo, anchors)

    cp1, cp2 = _load_cams(pref_h5, demo_key)
    cn1, cn2 = _load_cams(naive_h5, naive_demo)
    cap = max_steps or 10 ** 9
    Tp, Tn = min(n_p, len(cp1), cap), min(n_n, len(cn1), cap)
    T = max(Tp, Tn)

    has_sat = any(np.isfinite(np.asarray(v[:Tp], float)).any() for v in sat_p.values())
    src_p, src_n = (sat_p, sat_n) if has_sat else (raw_p, raw_n)
    keys = list(sat_p.keys())

    fig = plt.figure(figsize=(PLOT_W / 100, PLOT_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    caption = _metric_caption(mission_id, anchors)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (2 * PANEL + PLOT_W, 2 * PANEL))
    x = np.arange(T)
    for t in range(T):
        tp, tn = min(t, Tp - 1), min(t, Tn - 1)
        ax.clear()
        if has_sat:
            ax.set_ylim(-3, 103); ax.set_ylabel("satisfaction")
            if not _SKILL_TIER["on"]:
                ax.axhline(70, color="#2ca02c", lw=0.8, ls="--", alpha=.7)
                ax.axhline(30, color="#d62728", lw=0.8, ls="--", alpha=.7)
                ax.text(0.01, 71, "pref gate 70", color="#2ca02c", fontsize=7,
                        transform=ax.get_yaxis_transform())
                ax.text(0.01, 31, "no-pref gate 30", color="#d62728", fontsize=7,
                        transform=ax.get_yaxis_transform())
        else:
            ax.set_ylabel("raw (no anchors)")
        if win_p is not None:
            ax.axvspan(win_p[0], min(win_p[1], Tp - 1), color="#ffd27f", alpha=.28, lw=0)
        if win_n is not None:
            ax.axvspan(win_n[0], min(win_n[1], Tn - 1), color="#b0b0b0", alpha=.20, lw=0)
        for i, k in enumerate(keys):
            c = COLORS[i % len(COLORS)]
            fmt = (lambda v: f"{v:.0f}") if has_sat else (lambda v: f"{v:.4g}")
            yp = np.asarray(src_p.get(k, [])[:Tp], float)
            if yp.size:
                ax.plot(x[:tp + 1], yp[:tp + 1], color=c, lw=2.0, label=f"{labels[k]} - pref")
                if np.isfinite(yp[tp]):
                    ax.plot([tp], [yp[tp]], "o", color=c, ms=5)
                    ax.annotate(fmt(yp[tp]), (tp, yp[tp]), textcoords="offset points",
                                xytext=(6, 4), fontsize=9, color=c)
            yn = np.asarray(src_n.get(k, [])[:Tn], float)
            if yn.size:
                ax.plot(x[:tn + 1], yn[:tn + 1], color=c, lw=1.6, ls="--", alpha=.75,
                        label=f"{labels[k]} - no-pref")
                if np.isfinite(yn[tn]):
                    ax.plot([tn], [yn[tn]], "s", color=c, ms=4, alpha=.75)
                    ax.annotate(fmt(yn[tn]), (tn, yn[tn]), textcoords="offset points",
                                xytext=(6, -13), fontsize=9, color=c, alpha=.85)
        for ts, ls in ((ts_p, ":"), (ts_n, "-.")):
            if ts is not None and t >= ts:
                ax.axvline(ts, color="k", lw=0.8, ls=ls, alpha=.6)
        ax.set_xlim(0, max(T - 1, 1)); ax.set_xlabel("step")
        ax.set_title(caption, fontsize=6.2, loc="left", family="monospace",
                     linespacing=1.45, pad=6)
        # pref curves ride high and no-pref curves ride low, so the mid band is
        # the only spot that stays clear across missions; a fixed loc keeps the
        # legend from jittering frame to frame the way loc="best" would.
        ax.legend(loc="center left", fontsize=7, framealpha=.75)
        ax.grid(alpha=.25)
        fig.tight_layout(); fig.canvas.draw()
        plot = cv2.resize(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], (PLOT_W, PLOT_H))
        top = np.hstack([_panel(cp1[tp], "PREF  agentview"),
                         _panel(cp2[tp], "PREF  eye-in-hand")])
        bot = np.hstack([_panel(cn1[tn], "NO-PREF  agentview"),
                         _panel(cn2[tn], "NO-PREF  eye-in-hand")])
        frame = np.hstack([np.vstack([top, bot]), plot])
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release(); plt.close(fig)
    print(f"  {mission_id}: {out_path}  (pref {Tp} / no-pref {Tn} frames)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--demo_dir", default=DEFAULT_DEMO_DIR)
    ap.add_argument("--naive_dir", default=None,
                    help="no-pref demo dir; enables the pref-vs-no-pref comparison layout")
    ap.add_argument("--naive_demo", default=None, help="defaults to --demo")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--pref_subdirs", action="store_true",
                    help="write into <out>/<series>_<pref name>/ instead of <out>/")
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=None)
    a = ap.parse_args()
    anchors = json.load(open(a.anchors)) if a.anchors else {}
    naive_dir = a.naive_dir
    if naive_dir and not os.path.isabs(naive_dir):
        naive_dir = os.path.join(os.path.dirname(LIBERO), naive_dir) \
            if not os.path.isdir(os.path.join(LIBERO, naive_dir)) \
            else os.path.join(LIBERO, naive_dir)
    for mid in a.missions.split(","):
        h5 = cal._hdf5_in_dir(a.demo_dir, mid)
        if h5 is None:
            print(f"  {mid}: no hdf5 in {a.demo_dir}"); continue
        out_dir = os.path.join(a.out, pref_subdir(mid)) if a.pref_subdirs else a.out
        os.makedirs(out_dir, exist_ok=True)
        if naive_dir:
            nh5 = cal._hdf5_in_dir(naive_dir, mid)
            if nh5 is None:
                print(f"  {mid}: no no-pref hdf5 in {naive_dir}"); continue
            render_compare(mid, h5, nh5, a.demo, a.naive_demo or a.demo,
                           os.path.join(out_dir, f"{mid}_pref_vs_nopref.mp4"),
                           a.fps, anchors.get(mid, {}), a.max_steps)
        else:
            render(mid, h5, a.demo, os.path.join(out_dir, f"{mid}_{a.demo}.mp4"), a.fps,
                   anchors.get(mid, {}), a.max_steps)


if __name__ == "__main__":
    main()

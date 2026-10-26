"""Compose per-mission 3-panel review videos: [pref video | naive video | growing metric curve].

v6: height curves scoped by the SHARED runtime lateral-move detector
    (Preference.low_level.transport_window.lateral_move_mask) — ALL sustained
    lateral-translation segments, approach included, from EE positions only.
    Height prefs are worded "when moving" ("Move higher" / "Close to ground
    when moving") and gen applies path_height_offset to every move_to, so the
    approach move carries signal too. Same function the eval side uses for
    policy rollouts, so what the video shows is what gets scored.
    (v5 scoped to the carry transport window only — missed the approach move;
    v4 used the full gripper-closed span — swept pick lift + put descent in.)

Other curves unchanged from v4:
  rot: cumulative wrist turn accumulated on lateral-motion steps only (dxy > 1mm)
  tpp (series 3): real task_plan_progress score via sim replay (tpp_curves.npz)
  speed (1-2): global by design (SpeedRange raw quantity)
  series-2: proxies (no per-step metric exists; VLM/boolean judged)

Run from LIBERO/:  python scripts/make_compare_videos.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Preference.low_level.transport_window import lateral_move_mask  # noqa: E402

ROOT = "/data3/zhixu/LIBERO-PeRM/VLA_data/terminal-bench-science"
REL = f"{ROOT}/AgentEvalBench_release"
OUT = f"{ROOT}/videos_compare"
FPS = 20
CURVE_W, PANEL_H = 512, 256
LATERAL_THRESH = 0.001  # m, same as RotationPreference.lateral_thresh

METRIC_BY_MISSION = {
    "1-2_113": "time",
    "1-5_114": "height",
    "1-7_104": "height",
    "1-8_114": "rot",
    "2-1_019": "hoff",
    "2-3_077": "hoff",
    "2-5_116": "rot",
    "3-2_001_1": "tpp",
    "3-3_117_2": "tpp",
    "3-4_104_1": "tpp",
    "3-5_064_1": "tpp",
}
TPP = np.load(f"{ROOT}/tpp_curves.npz")
HOFF = np.load(f"{ROOT}/grasp_offset.npz")

# All curves oriented "higher = more preferred" (positive), matching the
# dual-anchor satisfaction polarity (naive->0, pref->1).
LABELS = {"time": "SpeedRange raw: elapsed time (s), more = slower",
          "rot": "Rotation raw: cum. wrist turn (deg, lateral steps)",
          "y": "EE y (m, proxy)",
          "hoff": "EE-to-object xy offset (m, carry)",
          "tpp": "task_plan_progress score (%)"}
# lift-high: running max EE z (peak semantics); hug-ground: lowness below the
# pair's highest transport altitude (positive, bigger = closer to ground).
HEIGHT_MODE = {"1-7_104": "runmax", "1-5_114": "lowness"}
HEIGHT_LABELS = {"runmax": "HeightOffset raw: running max EE z (m, move segs)",
                 "lowness": "hug-ground: lowness vs pair max (m, move segs)"}
SKILL_PROXY_LABELS = {"2-1_019": "edge-grasp: EE y-offset vs bowl center (m, + = left edge)",
                      "2-3_077": "handle-grasp: EE-to-mug-center xy offset (m, carry)",
                      "2-5_116": "side-pick proxy: cum. turn (deg, lateral)"}


def load_demo(h5path):
    with h5py.File(h5path, "r") as f:
        demo = f["data"][list(f["data"].keys())[0]]
        return (demo["obs/ee_pos"][:], demo["obs/ee_ori"][:], demo["obs/gripper_states"][:])


def to_curve(pos, ori, kind):
    if kind == "time":
        return np.arange(len(pos), dtype=float) / FPS
    if kind == "height":
        return pos[:, 2]
    if kind == "y":
        return pos[:, 1]
    if kind == "rot":
        dxy = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
        drot = np.linalg.norm(np.diff(ori, axis=0), axis=1)
        contrib = np.where(dxy > LATERAL_THRESH, drot, 0.0)
        return np.degrees(np.concatenate([[0.0], np.cumsum(contrib)]))
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS
    d = np.concatenate([[0.0], d])
    k = 15
    return np.convolve(d, np.ones(k) / k, mode="same")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)  # videos_gt already upright (flipped in place 2026-07-19)
    cap.release()
    return frames


def curve_renderer(mid, tier, label, yp, yn):
    """Curves may contain NaN outside the metric's scoped segment — those
    steps are simply not drawn."""
    fig = plt.figure(figsize=(CURVE_W / 100, PANEL_H / 100), dpi=100)
    ax = fig.add_subplot(111)
    nmax = max(len(yp), len(yn))
    lo = min(np.nanmin(yp), np.nanmin(yn)); hi = max(np.nanmax(yp), np.nanmax(yn))
    pad = 0.08 * (hi - lo + 1e-9)
    ax.set_xlim(0, nmax); ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{mid} [{tier}] {label}", fontsize=8)
    ax.tick_params(labelsize=6)
    lp, = ax.plot([], [], color="#1a7f37", lw=1.6, label="pref (1)")
    ln, = ax.plot([], [], color="#c62828", lw=1.6, label="naive (0)")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.4)
    def render(kp, kn):
        lp.set_data(np.arange(kp), yp[:kp])
        ln.set_data(np.arange(kn), yn[:kn])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR), (CURVE_W, PANEL_H))
    return fig, render


def tag(frame, text, color):
    fr = frame.copy()
    cv2.rectangle(fr, (0, 0), (86, 22), (0, 0, 0), -1)
    cv2.putText(fr, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return fr


def main():
    manifest = json.load(open(f"{REL}/manifest.json"))
    by_mission = {}
    for e in manifest["entries"]:
        by_mission.setdefault(e["mission_id"], {})[e["label"]] = e

    os.makedirs(OUT, exist_ok=True)
    for mid, pair in sorted(by_mission.items()):
        tier = pair[1]["tier"]
        kind = METRIC_BY_MISSION[mid]
        if kind == "tpp":
            yp, yn = TPP[f"{mid}_1"], TPP[f"{mid}_0"]
        elif kind == "hoff":
            yp, yn = HOFF[f"{mid}_1"], HOFF[f"{mid}_0"]
        else:
            pp, po, pg = load_demo(f"{REL}/{pair[1]['file']}")
            np_, no, ng = load_demo(f"{REL}/{pair[0]['file']}")
            yp = to_curve(pp, po, kind)
            yn = to_curve(np_, no, kind)
            if kind == "height":
                mode = HEIGHT_MODE[mid]
                for arr, pos, grip in ((yp, pp, pg), (yn, np_, ng)):
                    mask = lateral_move_mask(pos)
                    if mode == "runmax" and mask.any():
                        arr[mask] = np.maximum.accumulate(arr[mask])
                    arr[~mask] = np.nan
                if mode == "lowness":
                    ref = max(np.nanmax(yp), np.nanmax(yn))
                    yp, yn = ref - yp, ref - yn
        if kind == "height":
            label = HEIGHT_LABELS[HEIGHT_MODE[mid]]
        else:
            label = SKILL_PROXY_LABELS.get(mid, LABELS[kind])
        fp = read_frames(f"{REL}/videos_gt/{mid}_pref.mp4")
        fn = read_frames(f"{REL}/videos_gt/{mid}_naive.mp4")
        total = max(len(fp), len(fn))
        fig, render = curve_renderer(mid, tier, label, yp, yn)

        tmp = f"{OUT}/.tmp_{mid}.mp4"
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (256 * 2 + CURVE_W, PANEL_H))
        for i in range(total):
            pf = tag(fp[min(i, len(fp) - 1)], "PREF", (80, 220, 80))
            nf = tag(fn[min(i, len(fn) - 1)], "NAIVE", (80, 80, 230))
            kp = min(int((min(i, len(fp) - 1) + 1) / len(fp) * len(yp)) + 1, len(yp))
            kn = min(int((min(i, len(fn) - 1) + 1) / len(fn) * len(yn)) + 1, len(yn))
            vw.write(np.hstack([pf, nf, render(kp, kn)]))
        vw.release()
        plt.close(fig)

        out = f"{OUT}/{mid}_compare.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
        os.remove(tmp)
        print(f"{mid}: metric={kind} -> {out}", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()

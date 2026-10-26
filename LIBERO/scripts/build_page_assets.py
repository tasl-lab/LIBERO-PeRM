#!/usr/bin/env python
"""Copy the rendered task clips into the project page and write its data file.

Reads <gallery>/manifest.json (render_task_gallery.py + review_task_gallery.py) and produces,
inside the page repo:
  assets/tasks/<suite>/<task>.mp4|.jpg   every rendered clip that is not held for review
  assets/tasks/<suite>/<task>.curves.json  the per-step values behind that clip's curves (with --curves)
  assets/tasks/tasks.js                  window.PERM_TASKS = [...]  (loaded with a <script> tag,
                                         so the page also works from file://)
  assets/reel/*.mp4                      square third-person clips for the three hero lanes (one per level)

The "preference tour" marks two SinglePref tasks per preference family (37 x 2). A family whose
clips are all held for review still gets its two entries, flagged, so the tour stays complete.

Usage:  python scripts/build_page_assets.py [--gallery DIR] [--curves DIR] [--page /data3/zhixu/LIBERO-PeRM-page] [--no-reel]
        --curves is the output folder of export_gallery_curves.py (<suite>/<task>.json).
"""
import argparse, collections, json, math, os, shutil, subprocess
import h5py, numpy as np

GALLERY = "/data3/zhixu/PeRM_data/demos/_task_gallery"
V = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "VLA_data")
# hero: one lane of clips per preference level, eight preferences each (written to PERM_META.reel)
REEL = ["1-1_004", "1-2_065", "1-3_107", "1-4_034", "1-5_114", "1-6_105", "1-7_104", "1-8_061",
        "2-1_019", "2-3_077", "2-5_116", "2-7_ood1", "2-8_ood3", "2-10_015", "2-12_085", "2-14_068",
        "3-2_001_1", "3-4_104_1", "3-5_023_1", "3-6_012_1", "3-8_081_2", "3-9_118_1", "3-12_060_1", "3-14_114_1"]
TIER = {"1": "motion", "2": "skill", "3": "profile"}


def reel_clip(mid, dst, n=200):
    src = next((p for d in ("SinglePref_paired", "SinglePref", "PrefConflict")
                for p in [f"{V}/{d}/Mission_{mid}.hdf5"] if os.path.exists(p)), None)
    if not src:
        return False
    with h5py.File(src, "r") as f:
        frames = f["data/demo_0/obs/agentview_rgb"][:n]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256",
                           "-r", "24", "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "30",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    ff.stdin.write(np.ascontiguousarray(frames, dtype=np.uint8).tobytes()); ff.stdin.close()
    return ff.wait() == 0


def public_curves(src, r):
    """The numbers behind one clip's curves, for download from the page. Keeps the per-step values and
    drops what the export records about this machine (source files, config paths, hashes). Returns None
    unless the file covers exactly the clip's rows x metrics and ends where the clip ends."""
    if not os.path.exists(src):
        return None
    d = json.load(open(src))
    if d.get("status") != "exported" or d.get("errors"):
        return None
    num = lambda v, nd: None if v is None or not math.isfinite(v) else round(float(v), nd)
    shown = {(m["name"], x["label"]): f["satisfaction"]
             for (m, x), f in zip([(m, x) for x in r["rows"] for m in r["metrics"]], r["final"])}
    out = []
    for c in d["curves"]:
        k = (c["metric"], c["row"])
        end, clip_end = num(c["final"], 2), num(shown.get(k), 2)
        if k not in shown or (end is None) != (clip_end is None) or (end is not None and abs(end - clip_end) > 0.5):
            return None
        out.append(dict(row=c["row"], metric=c["metric"], steps=c["steps"], task_done_step=c.get("stored_done_step"),
                        final=end, satisfaction=[num(v, 2) for v in c["satisfaction"]],
                        raw=[num(v, 6) for v in c["raw"]]))
    if len(out) != len(shown):
        return None
    return dict(task=f"{r['suite']}/{r['task_id']}", title=r["title"], instruction=r["task"], rollout=d.get("demo"),
                about="Values behind the curves in the clip, one per simulator step of the rollout shown. "
                      "satisfaction is the preference-satisfaction score (0-100) and is null until the metric "
                      "produces a score; raw is the measured quantity the score is computed from. One "
                      "demonstration per row, not an average over demonstrations or policy rollouts.",
                curves=out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=GALLERY)
    ap.add_argument("--curves", default="", help="folder written by export_gallery_curves.py; adds a download per clip")
    ap.add_argument("--page", default="/data3/zhixu/LIBERO-PeRM-page")
    ap.add_argument("--no-reel", action="store_true")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.gallery, "manifest.json")))
    ok = [r for r in man if r["status"] == "ok"]

    tour = set()
    fam = collections.OrderedDict()
    for r in ok:
        if r["suite"] == "single":
            fam.setdefault(r["family"], []).append(r)
    for f, rs in fam.items():
        clean = [r for r in rs if not r.get("review")]
        pick = (clean + [r for r in rs if r.get("review")])[:2]
        tour.update(r["task_id"] for r in pick)

    out = os.path.join(a.page, "assets", "tasks")
    shutil.rmtree(out, ignore_errors=True)
    tasks, size, n_curves = [], 0, 0
    for r in ok:
        in_tour = r["suite"] == "single" and r["task_id"] in tour
        if r.get("review") and not in_tour:
            continue                                   # held back until the curve is signed off
        d = os.path.join(out, r["suite"]); os.makedirs(d, exist_ok=True)
        for ext in (".mp4", ".jpg"):
            src = os.path.join(a.gallery, r["suite"], r["task_id"] + ext)
            shutil.copy2(src, os.path.join(d, r["task_id"] + ext)); size += os.path.getsize(src)
        cv = public_curves(os.path.join(a.curves, r["suite"], r["task_id"] + ".json"), r) if a.curves else None
        if cv:
            with open(os.path.join(d, r["task_id"] + ".curves.json"), "w") as f:
                json.dump(cv, f, separators=(",", ":"), allow_nan=False)
            n_curves += 1
        nm, nr = len(r["metrics"]), len(r["rows"])
        order = [(mi, ri) for ri in range(nr) for mi in range(nm)]
        tasks.append(dict(
            id=r["task_id"], suite=r["suite"], family=r.get("family"), cond=r.get("condition"),
            tier=TIER.get(r["task_id"][0], "multi") if r["suite"] in ("single", "scale") else "multi",
            title=r["title"], task=r["task"], note=r.get("note"),
            metrics=[dict(name=m["name"], about=m["about"], tier=m["tier"]) for m in r["metrics"]],
            rows=[x["label"] for x in r["rows"]],
            final=[dict(m=mi, r=ri, v=None if x["satisfaction"] is None else round(x["satisfaction"]))
                   for (mi, ri), x in zip(order, r["final"])],
            video=f"assets/tasks/{r['suite']}/{r['task_id']}.mp4", poster=f"assets/tasks/{r['suite']}/{r['task_id']}.jpg",
            curves=f"assets/tasks/{r['suite']}/{r['task_id']}.curves.json" if cv else None,
            tour=in_tour, held=bool(r.get("review"))))
    counts = collections.Counter(t["suite"] for t in tasks)
    title = {r["task_id"]: r["title"] for r in ok if r["suite"] == "single"}
    reel = [dict(id=m, family=m.split("_")[0], tier=TIER[m[0]], title=title.get(m, "")) for m in REEL]
    meta = dict(rendered=len(ok), shown=len(tasks), by_suite=counts, tour=sum(t["tour"] for t in tasks), reel=reel)
    with open(os.path.join(out, "tasks.js"), "w") as f:
        f.write("window.PERM_TASKS = " + json.dumps(tasks, separators=(",", ":")) + ";\n")
        f.write("window.PERM_META = " + json.dumps(meta, separators=(",", ":")) + ";\n")
    print(f"tasks: {len(tasks)} clips ({size / 1e6:.0f} MB) -> {out}; by suite {dict(counts)}; tour {meta['tour']}")
    if a.curves:
        print(f"curve data: {n_curves} of {len(tasks)} clips; without: {[t['suite'] + '/' + t['id'] for t in tasks if not t['curves']]}")

    if not a.no_reel:
        rd = os.path.join(a.page, "assets", "reel"); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        made = [m for m in REEL if reel_clip(m, os.path.join(rd, m + ".mp4"))]
        if len(made) != len(REEL):
            raise SystemExit(f"reel: no demonstrations for {sorted(set(REEL) - set(made))}")
        print(f"reel: {len(made)} clips, {sum(os.path.getsize(os.path.join(rd, m + '.mp4')) for m in made) / 1e6:.1f} MB")

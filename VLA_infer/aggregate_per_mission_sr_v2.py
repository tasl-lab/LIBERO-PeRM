"""v2 aggregator for the overnight re-eval.

Walks new outputs at /data3/zhixu/LIBERO-PeRM/Policy_output/0shot/single-pref/
- univla_{spatial,object,goal,10}_full_v2/ → per-mission max across 4 ckpts (best-of-4)
- smolvla_full_v2/                         → as-is
- groot16_full_v2/                         → as-is

Plus historical baselines for context (pi05_official, openvla-oft 4-in-1 if available).

Output: VLA_infer/per_mission_sr_v2.md  (40-mission × N-VLA matrix + summary)
Also: VLA_infer/univla_4pass_per_mission.md (raw per-suite SR for UniVLA)
"""

from __future__ import annotations
import os, re, sys
from collections import defaultdict
from typing import Dict, List, Tuple
import yaml

OUT_ROOT = "/data3/zhixu/LIBERO-PeRM/Policy_output/0shot"
TASKSUITE = "/data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/tasksuite.yaml"
REPORT = "/data3/zhixu/LIBERO-PeRM/VLA_infer/per_mission_sr_v2.md"
UNIVLA_DETAIL = "/data3/zhixu/LIBERO-PeRM/VLA_infer/univla_4pass_per_mission.md"


def collect_cells(root: str) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Walk <root>/<model>/<cond>/<mission>/rollout_summary.yaml → {(mid,cond): (s,n)}.
    If multiple model subdirs exist, prefer the one with most episodes."""
    out: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if not os.path.isdir(root):
        return out
    for model_name in os.listdir(root):
        mdir = os.path.join(root, model_name)
        if not os.path.isdir(mdir): continue
        for cond in ("base", "pref"):
            cd = os.path.join(mdir, cond)
            if not os.path.isdir(cd): continue
            for mid in os.listdir(cd):
                rs = os.path.join(cd, mid, "rollout_summary.yaml")
                if not os.path.isfile(rs): continue
                try:
                    y = yaml.safe_load(open(rs)) or {}
                except Exception:
                    continue
                eps = y.get("episodes") or []
                s = sum(1 for e in eps if e.get("success"))
                n = len(eps)
                if n == 0: continue
                key = (mid, cond)
                prev = out.get(key)
                if prev is None or n > prev[1]:
                    out[key] = (s, n)
    return out


def best_of(suite_dicts: List[Dict[Tuple[str, str], Tuple[int, int]]]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Per-(mid,cond) take the SR-max across all suite dicts. Tie-break on n."""
    out: Dict[Tuple[str, str], Tuple[int, int]] = {}
    keys = set()
    for d in suite_dicts: keys.update(d.keys())
    for k in keys:
        best_s = -1; best_n = 0
        for d in suite_dicts:
            v = d.get(k)
            if v is None: continue
            s, n = v
            if n == 0: continue
            sr = s / n
            if sr > (best_s / best_n if best_n > 0 else -1):
                best_s, best_n = s, n
        if best_n > 0:
            out[k] = (best_s, best_n)
    return out


def main():
    with open(TASKSUITE) as f:
        ts = yaml.safe_load(f)
    libero_inst: Dict[str, str] = ts.get("libero_instructions") or {}
    missions = list(libero_inst.keys())

    def series(mid):
        m = re.match(r"^(\d+-\d+)_", mid)
        return m.group(1) if m else "?"

    # ── UniVLA: 4 per-suite ckpt sweeps + best-of-4 ──
    UV_DIRS = {
        "spatial": f"{OUT_ROOT}/single-pref/univla_spatial_full_v2",
        "object":  f"{OUT_ROOT}/single-pref/univla_object_full_v2",
        "goal":    f"{OUT_ROOT}/single-pref/univla_goal_full_v2",
        "10":      f"{OUT_ROOT}/single-pref/univla_10_full_v2",
    }
    uv_data = {k: collect_cells(v) for k, v in UV_DIRS.items()}
    uv_best = best_of(list(uv_data.values()))

    # ── X-VLA: 4 per-suite LoRA sweeps + best-of-4 ──
    XV_DIRS = {
        "spatial": f"{OUT_ROOT}/single-pref/xvla_spatial_full_v2",
        "object":  f"{OUT_ROOT}/single-pref/xvla_object_full_v2",
        "goal":    f"{OUT_ROOT}/single-pref/xvla_goal_full_v2",
        "long":    f"{OUT_ROOT}/single-pref/xvla_long_full_v2",
    }
    xv_data = {k: collect_cells(v) for k, v in XV_DIRS.items()}
    xv_best = best_of(list(xv_data.values()))

    # ── SmolVLA + GROOT v2 + historical ──
    sv_data = collect_cells(f"{OUT_ROOT}/single-pref/smolvla_full_v2")
    gr_data = collect_cells(f"{OUT_ROOT}/single-pref/groot16_full_v2")

    # historical for context
    hist = {
        "π0.5 official": collect_cells(f"{OUT_ROOT}/single-pref/pi05_official_singpref_0shot"),
        "OpenVLA-OFT 4-in-1 (best-of-4 HPCC)":  collect_cells(f"{OUT_ROOT}/single-pref/openvla_oft_4in1_singpref_0shot"),
    }

    # OFT 4-in-1: HPCC base (best-of-4 unnorm) + local v6 pref (best-of-4 unnorm)
    OFT_PREF_DIRS = {
        "spatial": f"{OUT_ROOT}/single-pref/oft_4in1_pref_spatial_v2",
        "object":  f"{OUT_ROOT}/single-pref/oft_4in1_pref_object_v2",
        "goal":    f"{OUT_ROOT}/single-pref/oft_4in1_pref_goal_v2",
        "10":      f"{OUT_ROOT}/single-pref/oft_4in1_pref_10_v2",
    }
    oft_pref_data = {k: collect_cells(v) for k, v in OFT_PREF_DIRS.items()}
    oft_pref_best = best_of(list(oft_pref_data.values()))
    oft_combined = dict(hist["OpenVLA-OFT 4-in-1 (best-of-4 HPCC)"] or {})
    for k, v in oft_pref_best.items():
        oft_combined[k] = v

    vlas = [
        ("UniVLA libero-spatial", uv_data["spatial"]),
        ("UniVLA libero-object",  uv_data["object"]),
        ("UniVLA libero-goal",    uv_data["goal"]),
        ("UniVLA libero-10",      uv_data["10"]),
        ("UniVLA best-of-4 ⭐",   uv_best),
        ("X-VLA libero-spatial",  xv_data["spatial"]),
        ("X-VLA libero-object",   xv_data["object"]),
        ("X-VLA libero-goal",     xv_data["goal"]),
        ("X-VLA libero-long",     xv_data["long"]),
        ("X-VLA best-of-4 ⭐",    xv_best),
        ("SmolVLA v2",            sv_data),
        ("GROOT N1.6 v2",         gr_data),
        ("π0.5 official",         hist["π0.5 official"]),
        ("OFT 4-in-1 (best-of-4)", oft_combined),
    ]

    print(f"Walking {len(vlas)} VLAs across {len(missions)} missions...\n")
    for name, d in vlas:
        cb = sum(1 for m in missions if (m, "base") in d)
        cp = sum(1 for m in missions if (m, "pref") in d)
        print(f"  {name:<35s}  base cov={cb}/40  pref cov={cp}/40")
    print()

    lines: List[str] = []
    lines.append("# P-Bench SinglePref 0-shot per-mission SR (v2 — overnight rerun)\n")
    lines.append("Generated by `aggregate_per_mission_sr_v2.py`. Format: `s/n (SR)`.\n")
    lines.append("")
    lines.append("## Aggregate (across all evaluated cells)\n")
    lines.append("| VLA | base | pref | overall | mission cov |")
    lines.append("|---|---|---|---|---|")
    for name, d in vlas:
        bs = bn = ps = pn = 0
        for (m, c), (s, n) in d.items():
            if c == "base": bs += s; bn += n
            else: ps += s; pn += n
        cb = sum(1 for m in missions if (m, "base") in d)
        cp = sum(1 for m in missions if (m, "pref") in d)
        bsr = f"{bs/bn:.3f} ({bs}/{bn})" if bn else "—"
        psr = f"{ps/pn:.3f} ({ps}/{pn})" if pn else "—"
        osr = f"{(bs+ps)/(bn+pn):.3f}" if (bn+pn) else "—"
        lines.append(f"| {name} | {bsr} | {psr} | {osr} | base={cb}, pref={cp} |")

    # Per-series summary
    lines.append("\n## Per-series base+pref SR\n")
    series_list = sorted(set(series(m) for m in missions),
                         key=lambda s: tuple(int(x) for x in s.split("-")))
    cols = ["Series", "n_miss"] + [name for name, _ in vlas]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for s in series_list:
        miss_in_s = [m for m in missions if series(m) == s]
        row = [s, str(len(miss_in_s))]
        for name, d in vlas:
            ss = nn = 0
            for m in miss_in_s:
                for cond in ("base", "pref"):
                    v = d.get((m, cond))
                    if v: ss += v[0]; nn += v[1]
            row.append(f"{ss/nn:.2f}" if nn else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Per-mission table
    lines.append("\n## Per-mission detail\n")
    cols = ["Mission", "Series", "Task"]
    for name, _ in vlas:
        short = name.replace(" libero-", "-").replace(" v2", "").replace(" official", "")
        cols.append(f"{short} base"); cols.append(f"{short} pref")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for mid in missions:
        row = [mid, series(mid), libero_inst[mid][:55]]
        for name, d in vlas:
            for cond in ("base", "pref"):
                v = d.get((mid, cond))
                row.append(f"{v[0]}/{v[1]} ({v[0]/v[1]:.2f})" if v else "")
        lines.append("| " + " | ".join(row) + " |")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {REPORT}")

    # UniVLA per-suite detail
    detail = ["# UniVLA 4-pass per-mission detail (v2)\n"]
    detail.append("| mission | series | spatial | object | goal | 10 | best |")
    detail.append("|" + "|".join(["---"] * 7) + "|")
    for mid in missions:
        row = [mid, series(mid)]
        for s in ("spatial", "object", "goal", "10"):
            d = uv_data[s]
            ss = nn = 0
            for c in ("base", "pref"):
                v = d.get((mid, c))
                if v: ss += v[0]; nn += v[1]
            row.append(f"{ss}/{nn}={ss/nn:.2f}" if nn else "—")
        ss = nn = 0
        for c in ("base", "pref"):
            v = uv_best.get((mid, c))
            if v: ss += v[0]; nn += v[1]
        row.append(f"**{ss}/{nn}={ss/nn:.2f}**" if nn else "—")
        detail.append("| " + " | ".join(row) + " |")
    with open(UNIVLA_DETAIL, "w") as f:
        f.write("\n".join(detail) + "\n")
    print(f"Written: {UNIVLA_DETAIL}")


if __name__ == "__main__":
    main()

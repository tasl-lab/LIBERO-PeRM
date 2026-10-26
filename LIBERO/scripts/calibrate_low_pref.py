"""Offline dual-anchor calibration: scan GT-pref + naive demos to compute
per-mission (pref_anchor, naive_anchor) for every low-level preference.

Dual-anchor (see Doc/design/t2_low_dual_anchor_plan.md):
  - pref_anchor  = median raw signal on GT-pref demos   → satisfaction 100
  - naive_anchor = median raw signal on naive demos     → satisfaction 0
  satisfaction = clip((R - naive)/(pref - naive), 0, 1)*100 ; sign(pref-naive)
  auto-encodes direction (speed_up vs slow_down). |pref-naive|<eps → invalid.

The raw signal per pref is `pref.final_raw(mean_reward)` (identity for most
metrics; cumulative metrics like SpeedRange override → total step count).

GT-pref demos live in a FLAT dir of Mission_<id>.hdf5 (authoritative source =
/data4/zhixu/SinglePref, per the 2026-06-16 calibration decision). Naive demos
come from `step1_data_gen.py --gen_naive_anchor` (written to
VLA_data/<tasksuite>_naive/). If --naive_dir is omitted (or a mission has no
naive demo), only pref_anchor is written → satisfaction falls back to legacy
single-anchor clip(R/pref).

Writes mission_info.yaml `low_pref_calibration:` as
  {pref_type: {pref_anchor, n_pref[, naive_anchor, n_naive]}}.

Usage:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python scripts/calibrate_low_pref.py --suite single-pref \
        --pref_dir /data4/zhixu/SinglePref \
        --naive_dir VLA_data/SinglePref_naive [--apply] [--n_demos 50]

Dry-run by default (prints diff). Pass --apply to write yaml.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LIBERO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_ROOT)

# Heavy deps (robosuite / configs / pref_gen) are imported lazily inside the
# functions that need them so pure helpers (_build_calibration) and downstream
# diagnostics stay unit-testable without a sim install.

# Authoritative GT-pref demo source (flat dir of Mission_<id>.hdf5).
DEFAULT_PREF_DIR = "/data4/zhixu/SinglePref"
DATASET_ROOT = os.path.join(_LIBERO_ROOT, "dataset")
# Demos recorded before the T1 rename embed old absolute asset paths in their
# model_file XML; patch so MjSim can resolve meshes/textures. No-op if absent.
_XML_PATH_FIXES = (("/data3/zhixu/P_Bench/", "/data3/zhixu/LIBERO-PeRM/"),)


def _mission_info_path(mission_id: str) -> Optional[str]:
    """3-1_004_1 → series_3/perm_3_1/libero_004_1/mission_info.yaml"""
    m = re.match(r"^(\d+)-(\d+)_(.+)$", mission_id)
    if not m:
        return None
    series, sub, lib = m.group(1), m.group(2), m.group(3)
    p = os.path.join(
        DATASET_ROOT, f"series_{series}", f"perm_{series}_{sub}",
        f"libero_{lib}", "mission_info.yaml",
    )
    return p if os.path.exists(p) else None


def _resolve_dir(d: Optional[str]) -> Optional[str]:
    """Allow naive_dir to be given relative to the LIBERO root."""
    if not d:
        return None
    if os.path.isabs(d):
        return d
    return os.path.join(_LIBERO_ROOT, d)


def _hdf5_in_dir(base_dir: Optional[str], mission_id: str) -> Optional[str]:
    if not base_dir:
        return None
    p = os.path.join(base_dir, f"Mission_{mission_id}.hdf5")
    return p if os.path.exists(p) else None


def _build_obj_body_id(sim) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for bid in range(sim.model.nbody):
        bn = sim.model.body_id2name(bid)
        if not bn:
            continue
        if bn.endswith("_main"):
            out[bn[: -len("_main")]] = bid
        out[bn] = bid
    return out


def _wrap_env(sim, obj_body_id):
    from types import SimpleNamespace
    return SimpleNamespace(
        sim=sim, obj_body_id=obj_body_id,
        robots=[SimpleNamespace(eef_name="gripper0_eef")],
    )


def _patch_xml(xml: str) -> str:
    for old, new in _XML_PATH_FIXES:
        xml = xml.replace(old, new)
    return xml


def _build_prefs(entries: List[dict]) -> List:
    from Preference.pref_gen import create_motion_preference
    prefs = []
    for item in entries:
        p = create_motion_preference(
            preference_type=item["type"],
            preference_parameters=item["params"],
            pref_description=item.get("description", ""),
            pos_description=item.get("pos_description"),
        )
        if p is not None:
            prefs.append(p)
    return prefs


def _replay_one_demo(sim, states: np.ndarray, env, prefs: List, max_steps: Optional[int] = None,
                     dones: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Return {pref_type: raw} for this demo via pref.final_raw().

    `dones` (HDF5 per-step success flags) marks the first success step on
    every pref so the raw ignores the post-success tail, exactly as the
    online LowPrefAccumulator does."""
    from robosuite.utils.binding_utils import MjSimState
    for p in prefs:
        try:
            p.reset(env)
        except Exception:
            pass
    n = states.shape[0] if max_steps is None else min(states.shape[0], max_steps)
    sums: Dict[str, float] = {p.preference_type: 0.0 for p in prefs}
    counts: Dict[str, int] = {p.preference_type: 0 for p in prefs}
    marked = False
    for t in range(n):
        sim.set_state(MjSimState.from_flattened(states[t], sim))
        sim.forward()
        for p in prefs:
            try:
                r = float(p.get_reward(env))
            except Exception:
                continue
            if r != r:  # NaN
                continue
            sums[p.preference_type] += r
            counts[p.preference_type] += 1
        if not marked and dones is not None and t < len(dones) and dones[t]:
            marked = True
            for p in prefs:
                if hasattr(p, "mark_success"):
                    p.mark_success()
    for p in prefs:
        if hasattr(p, "mark_episode_end"):
            p.mark_episode_end()
    out: Dict[str, float] = {}
    for p in prefs:
        t = p.preference_type
        if counts.get(t, 0) <= 0:
            continue
        mean_r = sums[t] / counts[t]
        try:
            raw = p.final_raw(mean_r)
        except Exception:
            raw = mean_r
        if raw is None:
            raw = mean_r
        out[t] = float(raw)
        # skill tier: the discrete node verdict (0/100) next to the raw, so the
        # gate / pairing can require "criterion met" instead of a raw threshold
        if getattr(p, "level", None) in ("skill", "profile"):
            try:
                sv = p.satisfaction(float(raw))
            except Exception:
                sv = None
            out[t + "__sat"] = float("nan") if sv is None else float(sv)
        # per-subtask 耗时向量（speed 家族）：全部完成才计入锚
        sub_fn = getattr(p, "subtask_raw", None)
        if callable(sub_fn):
            v = sub_fn()
            if v and all(x is not None for x in v):
                out[t + "__subtasks"] = tuple(float(x) for x in v)
    return out


def _raw_lists_for_dir(prefs: List, hdf5_path: str, n_demos: int) -> Dict[str, List[float]]:
    """Replay up to n_demos in hdf5_path through `prefs`; return per-pref-type
    list of per-demo raw signals {pref_type: [raw, ...]}."""
    from robosuite.utils.binding_utils import MjSim
    with h5py.File(hdf5_path, "r") as f:
        # numeric order (h5py lists demo_10 before demo_2): callers index the
        # returned lists by demo number to gate / pair demos
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:n_demos]
        if not demos:
            return {}
        per_demo: List[Dict[str, float]] = []
        for k in demos:
            # Per-demo sim: fixtures (e.g. an avoid-target stove) are static —
            # their sampled init pose lives only in this demo's model_file, not
            # in `states`, so a shared sim would freeze them at demo_0's spot.
            xml = f[f"data/{k}"].attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode("utf-8")
            sim = MjSim.from_xml_string(_patch_xml(xml))
            env = _wrap_env(sim, _build_obj_body_id(sim))
            states = np.asarray(f[f"data/{k}/states"][:])
            dones = np.asarray(f[f"data/{k}/dones"][:]) if f"data/{k}/dones" in f else None
            try:
                per_demo.append(_replay_one_demo(sim, states, env, prefs, dones=dones))
            except Exception:
                continue
    out: Dict[str, List[float]] = {}
    for d in per_demo:
        for t, v in d.items():
            out.setdefault(t, []).append(v)
    return out


def _medians_for_dir(prefs: List, hdf5_path: str, n_demos: int) -> Dict[str, Tuple[float, int]]:
    """Replay up to n_demos in hdf5_path through `prefs`; return
    {pref_type: (median_raw, n_demos_used)}."""
    raw = _raw_lists_for_dir(prefs, hdf5_path, n_demos)
    return {t: (round(float(np.median(vs)), 6), len(vs))
            for t, vs in raw.items() if vs and not t.endswith("__subtasks") and not t.endswith("__sat")}


def _subtask_medians(raw: Dict[str, List]) -> Dict[str, List[float]]:
    """{pref_type: [per-subtask median duration, ...]} from __subtasks vectors."""
    out: Dict[str, List[float]] = {}
    for t, vecs in raw.items():
        if not t.endswith("__subtasks") or not vecs:
            continue
        L = min(len(v) for v in vecs)
        if L == 0:
            continue
        base = t[: -len("__subtasks")]
        out[base] = [round(float(np.median([v[i] for v in vecs])), 1) for i in range(L)]
    return out


def _build_calibration(
    pref_medians: Dict[str, Tuple[float, int]],
    naive_medians: Dict[str, Tuple[float, int]],
) -> Dict[str, Dict[str, float]]:
    """Combine per-pref-type pref + naive medians into the calibration block.

    Pure (no I/O / sim) → unit-testable. Keys present in pref_medians drive the
    output; naive_anchor is added only when the same pref_type was measured on
    naive demos.
    """
    cal: Dict[str, Dict[str, float]] = {}
    for ptype, (pref_anchor, n_pref) in pref_medians.items():
        entry: Dict[str, float] = {"pref_anchor": pref_anchor, "n_pref": n_pref}
        if ptype in naive_medians:
            naive_anchor, n_naive = naive_medians[ptype]
            entry["naive_anchor"] = naive_anchor
            entry["n_naive"] = n_naive
        cal[ptype] = entry
    return cal


def _calibrate_mission(
    mid: str, pref_dir: str, naive_dir: Optional[str], n_demos: int
) -> Tuple[Optional[Dict], Optional[str]]:
    info_path = _mission_info_path(mid)
    if info_path is None:
        return None, "no_yaml"
    pref_hdf5 = _hdf5_in_dir(pref_dir, mid)
    if pref_hdf5 is None:
        return None, "no_pref_hdf5"
    with open(info_path, "r") as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("low_prefs") or []
    if not entries:
        return {}, None  # no prefs → empty calibration block
    prefs = _build_prefs(entries)
    if not prefs:
        return None, "no_valid_prefs"
    # Match runtime (load_motion_prefs): inject task_plan objects so prefs with
    # transport gating (avoid: min(EE, carried-obj) distance) replay the same
    # metric here as at eval time — otherwise anchors are EE-only biased.
    task_objects = [
        str(s[1]) for s in (data.get("task_plan") or [])
        if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]
    ]
    for p in prefs:
        if hasattr(p, "set_task_context"):
            p.set_task_context(task_objects)
        if hasattr(p, "set_task_plan"):
            p.set_task_plan(data.get("task_plan"))

    pref_raw = _raw_lists_for_dir(prefs, pref_hdf5, n_demos)
    pref_medians = {t: (round(float(np.median(vs)), 6), len(vs))
                    for t, vs in pref_raw.items()
                    if vs and not t.endswith("__subtasks") and not t.endswith("__sat")}
    if not pref_medians:
        return None, "no_pref_replay"

    naive_medians: Dict[str, Tuple[float, int]] = {}
    naive_raw: Dict[str, List] = {}
    naive_hdf5 = _hdf5_in_dir(naive_dir, mid)
    if naive_hdf5 is not None:
        naive_raw = _raw_lists_for_dir(prefs, naive_hdf5, n_demos)
        naive_medians = {t: (round(float(np.median(vs)), 6), len(vs))
                         for t, vs in naive_raw.items()
                         if vs and not t.endswith("__subtasks") and not t.endswith("__sat")}

    cal = _build_calibration(pref_medians, naive_medians)
    # per-subtask 锚（speed 家族）：每个 subtask 耗时的 median 列表
    for base, med in _subtask_medians(pref_raw).items():
        if base in cal:
            cal[base]["subtask_pref_anchor"] = med
    for base, med in _subtask_medians(naive_raw).items():
        if base in cal:
            cal[base]["subtask_naive_anchor"] = med
    return cal, None


def _write_calibration(info_path: str, calibration: Dict, dry_run: bool):
    """Replace / append only the top-level `low_pref_calibration:` block, as
    text. A yaml.dump round-trip would drop every hand-written comment in
    mission_info.yaml (naive_plan rationale, left/right conventions)."""
    if dry_run:
        return
    with open(info_path, "r") as f:
        text = f.read()
    block = yaml.dump({"low_pref_calibration": calibration}, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
    pat = re.compile(r"^low_pref_calibration:.*?(?=^\S|\Z)", re.M | re.S)
    if pat.search(text):
        text = pat.sub(lambda _m: block, text, count=1)
    else:
        text = text.rstrip("\n") + "\n" + block
    with open(info_path, "w") as f:
        f.write(text)


def _fmt_anchors(cal: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    out = {}
    for t, c in cal.items():
        if "naive_anchor" in c:
            out[t] = f"pref={c['pref_anchor']} naive={c['naive_anchor']}"
        else:
            out[t] = f"pref={c['pref_anchor']} (no naive)"
    return out


# ─── discrimination table (--table; merged from low_pref_discrimination.py) ──
# NON-gating diagnostics: per (mission × pref_type) replay pref vs naive demos,
# report mean raw / gap / cohen_d / satisfaction. cohen_d >= GATE_COHEN_D is
# flagged "OK", below → "low-d" (a demo-gen improvement candidate, NOT a
# failure) — the table always exits 0.

GATE_COHEN_D = 0.5  # tunable: "OK" threshold for the diagnostic flag (not a gate)


def cohen_d(a: List[float], b: List[float]) -> Optional[float]:
    """Standardized mean difference (a - b) / pooled_sd. None if undefined
    (n<2 on either side, or zero pooled variance)."""
    import math
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled <= 0:
        return None
    return (float(np.mean(a)) - float(np.mean(b))) / pooled


def _mean(x: List[float]) -> Optional[float]:
    return float(np.mean(x)) if x else None


def _mean_sat(pref_obj, raws: List[float]) -> Optional[float]:
    """Mean satisfaction() over raw values; None if pref has no anchors / all None."""
    sats = []
    for r in raws:
        try:
            s = pref_obj.satisfaction(r)
        except Exception:
            s = None
        if s is not None:
            sats.append(float(s))
    return float(np.mean(sats)) if sats else None


def _rows_for_mission(mid: str, pref_dir: str, naive_dir: Optional[str], n_demos: int) -> List[dict]:
    from PolicyEval.low_pref_reward import load_motion_prefs
    info_path = _mission_info_path(mid)
    if info_path is None:
        return []
    pref_hdf5 = _hdf5_in_dir(pref_dir, mid)
    naive_hdf5 = _hdf5_in_dir(naive_dir, mid)
    if pref_hdf5 is None or naive_hdf5 is None:
        return []
    prefs = load_motion_prefs(info_path)  # anchors injected if calibration present
    if not prefs:
        return []
    pref_by_type = {p.preference_type: p for p in prefs}
    pref_raw = _raw_lists_for_dir(prefs, pref_hdf5, n_demos)
    naive_raw = _raw_lists_for_dir(prefs, naive_hdf5, n_demos)
    rows = []
    for ptype in sorted(set(pref_raw) | set(naive_raw)):
        if ptype.endswith("__sat") or ptype.endswith("__subtasks"):
            continue
        a, b = pref_raw.get(ptype, []), naive_raw.get(ptype, [])
        mp, mn = _mean(a), _mean(b)
        d = cohen_d(a, b)
        p_obj = pref_by_type.get(ptype)
        # skill tier: sat_* = share of demos whose discrete node verdict is 100
        # (per-demo verdicts from the replay), not the dual-anchor interpolation
        def _crit_mean(lst):
            v = [x for x in (lst or []) if x == x]
            return float(np.mean(v)) if v else None
        sp_, sn_ = pref_raw.get(ptype + "__sat"), naive_raw.get(ptype + "__sat")
        rows.append({
            "mission": mid,
            "pref_type": ptype,
            "mean_pref": mp,
            "mean_naive": mn,
            "gap": (mp - mn) if (mp is not None and mn is not None) else None,
            "cohen_d": d,
            "sat_pref": _crit_mean(sp_) if sp_ is not None else (_mean_sat(p_obj, a) if p_obj else None),
            "sat_naive": _crit_mean(sn_) if sn_ is not None else (_mean_sat(p_obj, b) if p_obj else None),
            "n_pref": len(a),
            "n_naive": len(b),
        })
    return rows


def _fmt(v, nd=4):
    return "  n/a" if v is None else f"{v:.{nd}f}"


def _run_table(ids: List[str], pref_dir: str, naive_dir: Optional[str],
               n_demos: int, csv_path: Optional[str]) -> None:
    all_rows: List[dict] = []
    for mid in ids:
        all_rows.extend(_rows_for_mission(mid, pref_dir, naive_dir, n_demos))

    hdr = (f"{'mission':<14}{'pref_type':<22}{'mean_pref':>11}{'mean_naive':>12}"
           f"{'gap':>10}{'cohen_d':>10}{'sat_pref':>10}{'sat_naive':>11}  flag")
    print(hdr)
    print("-" * len(hdr))
    per_metric: Dict[str, List[float]] = {}
    for r in all_rows:
        flag = ""
        if r["cohen_d"] is not None:
            per_metric.setdefault(r["pref_type"], []).append(abs(r["cohen_d"]))
            flag = "OK" if abs(r["cohen_d"]) >= GATE_COHEN_D else "low-d"
        print(f"{r['mission']:<14}{r['pref_type']:<22}{_fmt(r['mean_pref']):>11}"
              f"{_fmt(r['mean_naive']):>12}{_fmt(r['gap']):>10}{_fmt(r['cohen_d']):>10}"
              f"{_fmt(r['sat_pref'],1):>10}{_fmt(r['sat_naive'],1):>11}  {flag}")

    print("\n=== per-metric |cohen_d| summary (>= %.1f = OK) ===" % GATE_COHEN_D)
    for ptype in sorted(per_metric):
        ds = per_metric[ptype]
        n_ok = sum(1 for d in ds if d >= GATE_COHEN_D)
        print(f"  {ptype:<24} median|d|={np.median(ds):.3f}  OK {n_ok}/{len(ds)} missions")

    if csv_path:
        import csv
        cols = ["mission", "pref_type", "mean_pref", "mean_naive", "gap", "cohen_d",
                "sat_pref", "sat_naive", "n_pref", "n_naive"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[csv] wrote {len(all_rows)} rows → {csv_path}")

    print("\n(diagnostic only — exit 0 regardless; low-d metrics are demo-gen "
          "improvement candidates, not failures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="task_suite name (e.g. single-pref)")
    ap.add_argument("--pref_dir", default=DEFAULT_PREF_DIR,
                    help=f"flat dir of GT-pref Mission_<id>.hdf5 (default: {DEFAULT_PREF_DIR})")
    ap.add_argument("--naive_dir", default=None,
                    help="flat dir of naive Mission_<id>.hdf5 (from --gen_naive_anchor). "
                         "Omit → single-anchor (pref only). Relative paths resolve under LIBERO/.")
    ap.add_argument("--n_demos", type=int, default=50)
    ap.add_argument("--apply", action="store_true", help="write yaml (otherwise dry-run)")
    ap.add_argument("--missions", default=None, help="comma-separated mission_id subset")
    ap.add_argument("--table", action="store_true",
                    help="print the pref-vs-naive discrimination table instead of calibrating")
    ap.add_argument("--csv", default=None, help="CSV output path for --table")
    args = ap.parse_args()

    from configs.task_suites import load_suite
    pref_dir = _resolve_dir(args.pref_dir)
    naive_dir = _resolve_dir(args.naive_dir)
    if args.missions:
        ids = [s.strip() for s in args.missions.split(",") if s.strip()]
    else:
        ids = list(load_suite(args.suite))

    if args.table:
        if args.apply:
            ap.error("--table and --apply are mutually exclusive")
        if naive_dir is None:
            ap.error("--table requires --naive_dir")
        _run_table(ids, pref_dir, naive_dir, args.n_demos, args.csv)
        return
    print(f"Calibrating {len(ids)} mission(s) from suite={args.suite}")
    print(f"  pref_dir ={pref_dir}")
    print(f"  naive_dir={naive_dir if naive_dir else '(none → single-anchor)'}")
    print(f"  n_demos={args.n_demos}, apply={args.apply}")
    summary = {"ok": 0, "skip": 0, "err": 0, "with_naive": 0}
    for mid in ids:
        cal, err = _calibrate_mission(mid, pref_dir, naive_dir, args.n_demos)
        if cal is None:
            print(f"  [skip] {mid}  ({err})")
            summary["skip" if err in {"no_yaml", "no_pref_hdf5", "no_valid_prefs"} else "err"] += 1
            continue
        info_path = _mission_info_path(mid)
        if cal:
            print(f"  [ok]   {mid}  → {_fmt_anchors(cal)}")
            if any("naive_anchor" in c for c in cal.values()):
                summary["with_naive"] += 1
            _write_calibration(info_path, cal, dry_run=not args.apply)
        else:
            print(f"  [---]  {mid}  (no low_prefs)")
        summary["ok"] += 1
    print(f"\nDone: {summary}")
    if not args.apply:
        print("(dry-run; pass --apply to write yaml)")


if __name__ == "__main__":
    main()

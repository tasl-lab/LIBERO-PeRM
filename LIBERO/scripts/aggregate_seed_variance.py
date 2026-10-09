#!/usr/bin/env python3
"""Aggregate 5-seed eval reruns into mean ± std for the main-table pi05 rows.

Experiment Z (error bars): each of the 4 pi05-finetune settings was evaluated
with 5 eval seeds — the original run (seed 7) plus 4 seed-variant reruns
(seed 107/207/307/407, runfiles in VLA_infer/run_profile/LoRA/Pi05/SeedVar/).

For each (setting, condition, seed) this script pools success over all missions
(same OVERALL definition as aggregate_eval.py: sum n_success / sum n_trials),
then reports across-seed mean ± sample std (ddof=1) in percent.

Usage:
    python scripts/aggregate_seed_variance.py
    python scripts/aggregate_seed_variance.py --root /path/to/Policy_output/lora_ft/single-pref
    python scripts/aggregate_seed_variance.py --out /path/to/seed_variance.csv
"""
import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if LIBERO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_ROOT)

from configs.task_suites import load_suite  # noqa: E402

DEFAULT_ROOT = "/data3/zhixu/LIBERO-PeRM/Policy_output/lora_ft/single-pref"
SEEDS = [7, 107, 207, 307, 407]  # 7 = original run (no suffix unless seed7_name says otherwise)

# label -> (orig experiment_name, server name, conditions, seed7_name)
# seed7_name: dir name of the seed-7 run when it differs from the canonical
# experiment_name (U's unified 1250-step rerun lives under an _s7 suffix; the
# canonical U dir holds the deprecated 600-step + merge-max results).
SETTINGS = [
    ("pi05-LoRA (libero official data)", "W_pi05_libero_lora_v1_step29999",
     "pi05_libero_lora_v1_step29999", ["base", "pref"], None),
    ("pi05-Mix-LoRA", "Y_pi05_singpref_mix_step45k",
     "pi05_singpref_mix_run1_step44999", ["base", "pref"], None),
    ("pi05-Seq-LoRA", "U_pi05_singpref_step45k",
     "pi05_singpref_run1_step44999", ["pref"], "U_pi05_singpref_step45k_s7"),
    ("pi05-LoRA (libero+singpref)", "V_pi05_libero_singpref_step45k",
     "pi05_libero_singpref_run1_step44999", ["base", "pref"], None),
]


def _exp_dir(root: str, expname: str, server: str, seed: int, seed7_name: Optional[str]) -> str:
    if seed == 7:
        name = seed7_name or expname
    else:
        name = f"{expname}_s{seed}"
    return os.path.join(root, name, server)


def _pooled_sr(exp_dir: str, condition: str, whitelist: set) -> Optional[Tuple[float, int, int, set]]:
    """Return (sr_percent, n_success, n_trials, mission_set) or None if no data.

    Missions outside `whitelist` (current single-pref v2 = 36 missions) are
    skipped, so legacy 40-mission seed-7 runs aggregate on the same subset as
    the new seed-variant runs.
    """
    cdir = os.path.join(exp_dir, condition)
    if not os.path.isdir(cdir):
        return None
    n_succ = n_tri = 0
    missions = set()
    for mid in sorted(os.listdir(cdir)):
        if mid not in whitelist:
            continue
        sfile = os.path.join(cdir, mid, "rollout_summary.yaml")
        if not os.path.isfile(sfile):
            continue
        with open(sfile) as f:
            s = yaml.safe_load(f) or {}
        eps = s.get("episodes") or []
        if not eps:
            continue
        missions.add(mid)
        n_succ += sum(1 for e in eps if e.get("success"))
        n_tri += len(eps)
    if n_tri == 0:
        return None
    return 100.0 * n_succ / n_tri, n_succ, n_tri, missions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="single-pref suite root under Policy_output/lora_ft")
    ap.add_argument("--out", default=None,
                    help="optional CSV output path (default: <root>/seed_variance.csv)")
    args = ap.parse_args()

    rows: List[Dict] = []
    warnings: List[str] = []
    whitelist = set(load_suite("single-pref"))
    print(f"mission whitelist: current single-pref suite, {len(whitelist)} missions\n")

    for label, expname, server, conditions, seed7_name in SETTINGS:
        for cond in conditions:
            per_seed: Dict[int, float] = {}
            mission_sets: Dict[int, set] = {}
            for seed in SEEDS:
                d = _exp_dir(args.root, expname, server, seed, seed7_name)
                got = _pooled_sr(d, cond, whitelist)
                if got is None:
                    warnings.append(f"MISSING  {label} [{cond}] seed {seed}: {d}")
                    continue
                sr, _, n_tri, missions = got
                per_seed[seed] = sr
                mission_sets[seed] = missions
            # mission-set consistency across seeds (partial runs show up here)
            if len({frozenset(m) for m in mission_sets.values()}) > 1:
                counts = {s: len(m) for s, m in sorted(mission_sets.items())}
                warnings.append(f"MISMATCH {label} [{cond}] mission counts differ per seed: {counts}")
            srs = [per_seed[s] for s in SEEDS if s in per_seed]
            mean = statistics.mean(srs) if srs else float("nan")
            std = statistics.stdev(srs) if len(srs) >= 2 else float("nan")
            rows.append({
                "setting": label, "condition": cond, "n_seeds": len(srs),
                **{f"seed{s}": (f"{per_seed[s]:.1f}" if s in per_seed else "") for s in SEEDS},
                "mean": f"{mean:.1f}", "std": f"{std:.1f}",
                "table_cell": f"{mean:.1f} ± {std:.1f}",
            })

    cols = ["setting", "condition", "n_seeds"] + [f"seed{s}" for s in SEEDS] + ["mean", "std", "table_cell"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

    out = args.out or os.path.join(args.root, "seed_variance.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv -> {out}")

    if warnings:
        print("\nWARNINGS:", file=sys.stderr)
        for w in warnings:
            print(f"  {w}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

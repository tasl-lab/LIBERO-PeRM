#!/usr/bin/env python3
"""Best-of-N aggregation used for Table 2 (OpenVLA-OFT, UniVLA).

Those two policies ship one checkpoint / unnorm-key per LIBERO suite, so the
paper runs each SinglePref mission once per suite key and keeps, per mission
and condition, the pass with the highest SR ("best-of-4", same rule as
VLA_infer/aggregate_per_mission_sr_v2.py).

Usage:
  python aggregate_best_of_n.py <exp_dir_1> <exp_dir_2> ... [--out best_of.md]

Each <exp_dir> is Policy_output/<exp_type>/<suite>/<experiment_name>/ and is
walked as <exp_dir>/<model>/<base|pref>/<mission>/rollout_summary.yaml.
"""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict
import yaml

Cells = dict  # (mission, cond) -> (successes, episodes)


def collect_cells(root: str) -> Cells:
    out: Cells = {}
    if not os.path.isdir(root):
        print(f"[warn] not a dir: {root}", file=sys.stderr)
        return out
    for model in os.listdir(root):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir):
            continue
        for cond in ("base", "pref"):
            cdir = os.path.join(mdir, cond)
            if not os.path.isdir(cdir):
                continue
            for mid in os.listdir(cdir):
                rs = os.path.join(cdir, mid, "rollout_summary.yaml")
                if not os.path.isfile(rs):
                    continue
                eps = (yaml.safe_load(open(rs)) or {}).get("episodes") or []
                if not eps:
                    continue
                s = sum(1 for e in eps if e.get("success"))
                prev = out.get((mid, cond))
                if prev is None or len(eps) > prev[1]:
                    out[(mid, cond)] = (s, len(eps))
    return out


def best_of(runs: list[Cells]) -> Cells:
    out: Cells = {}
    for key in set().union(*(r.keys() for r in runs)):
        best = None
        for r in runs:
            v = r.get(key)
            if v and v[1] > 0 and (best is None or v[0] / v[1] > best[0] / best[1]):
                best = v
        if best:
            out[key] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--out", help="write markdown table here (default: stdout only)")
    a = ap.parse_args()

    runs = [collect_cells(d) for d in a.exp_dirs]
    merged = best_of(runs)
    lines = ["| condition | SR | successes/episodes | missions |", "|---|---|---|---|"]
    for cond in ("base", "pref"):
        cells = [v for (m, c), v in merged.items() if c == cond]
        s, n = sum(v[0] for v in cells), sum(v[1] for v in cells)
        lines.append(f"| {cond} | {100 * s / n if n else 0:.2f}% | {s}/{n} | {len(cells)} |")
    lines += ["", "| mission | " + " | ".join(("base", "pref")) + " |", "|---|---|---|"]
    for mid in sorted({m for m, _ in merged}):
        row = []
        for cond in ("base", "pref"):
            v = merged.get((mid, cond))
            row.append(f"{v[0]}/{v[1]}" if v else "—")
        lines.append(f"| {mid} | " + " | ".join(row) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

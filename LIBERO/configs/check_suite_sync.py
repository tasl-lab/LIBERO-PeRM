"""Verify task_suites.yaml is in sync with VLA_data training files.

Source of truth: VLA_data/<TaskSuite>/Mission_*.hdf5
Checks:
  1. Every suite mission_id has a corresponding training hdf5
  2. Every training hdf5 is registered in the suite
  3. Every mission resolves to an actual dataset/series_X/perm_X_Y/libero_ZZZ[_N]/

Run:
  python LIBERO/configs/check_suite_sync.py           # all suites
  python LIBERO/configs/check_suite_sync.py single-pref
Exits non-zero on any mismatch so it can gate CI / eval startup.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Set, Tuple

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
P_BENCH_ROOT = os.path.abspath(os.path.join(LIBERO_ROOT, ".."))
DATASET_ROOT = os.path.join(LIBERO_ROOT, "dataset")
VLA_DATA_ROOT = os.path.join(P_BENCH_ROOT, "VLA_data")
SUITES_YAML = os.path.join(_HERE, "task_suites.yaml")

# suite name → VLA_data subdirectory
SUITE_TO_VLA_DIR = {
    "single-pref": "SinglePref",
    "multi-pref": "MultiPref",
    "pref-learn-eff": "PrefLearnEff",
    "pref-compose": "PrefCompose",
    "liberoscale": "PrefScale",
}


def _mid_from_hdf5(fname: str) -> str | None:
    m = re.match(r"Mission_(.+)\.hdf5$", fname)
    return m.group(1) if m else None


def _resolve_dataset_dir(mid: str) -> str:
    m = re.match(r"^(\d+-\d+)_(.+)$", mid)
    if not m:
        raise ValueError(f"Invalid mission_id: {mid!r}")
    perm = f"perm_{m.group(1).replace('-','_')}"
    libero = f"libero_{m.group(2)}"
    series = f"series_{m.group(1).split('-')[0]}"
    return os.path.join(DATASET_ROOT, series, perm, libero)


def _vla_mids(suite: str) -> Set[str]:
    subdir = SUITE_TO_VLA_DIR.get(suite)
    if not subdir:
        return set()
    path = os.path.join(VLA_DATA_ROOT, subdir)
    if not os.path.isdir(path):
        return set()
    return {m for m in (_mid_from_hdf5(f) for f in os.listdir(path)) if m}


def _load_yaml() -> dict:
    with open(SUITES_YAML) as f:
        return yaml.safe_load(f) or {}


def verify_suite(name: str) -> None:
    """Raise RuntimeError if suite is out of sync with VLA_data + dataset."""
    cfg = _load_yaml()
    if name not in cfg:
        raise RuntimeError(f"Unknown task_suite {name!r}. Available: {list(cfg)}")
    errs = check_suite(name, cfg[name])
    if errs:
        raise RuntimeError("task_suite out of sync:\n" + "\n".join(errs))


def check_suite(suite: str, body: dict) -> List[str]:
    errors: List[str] = []
    suite_mids = set(body.get("missions") or [])
    vla_mids = _vla_mids(suite)

    if vla_mids:
        only_suite = suite_mids - vla_mids
        only_vla = vla_mids - suite_mids
        if only_suite:
            errors.append(f"[{suite}] in yaml but no training hdf5 in VLA_data/{SUITE_TO_VLA_DIR[suite]}:")
            for m in sorted(only_suite):
                errors.append(f"    - {m}")
        if only_vla:
            errors.append(f"[{suite}] training hdf5 exists but not in yaml:")
            for m in sorted(only_vla):
                errors.append(f"    + {m}")

    for mid in sorted(suite_mids):
        p = _resolve_dataset_dir(mid)
        if not os.path.isdir(p):
            dep = p + "_deprecated"
            hint = f"  (found {os.path.basename(dep)})" if os.path.isdir(dep) else ""
            errors.append(f"[{suite}] dataset missing for {mid}: {p}{hint}")
    return errors


def main() -> int:
    cfg = _load_yaml()
    targets = sys.argv[1:] or list(cfg.keys())
    all_errs: List[str] = []
    for suite in targets:
        if suite not in cfg:
            print(f"[{suite}] unknown suite; skipping")
            continue
        errs = check_suite(suite, cfg[suite])
        if errs:
            all_errs.extend(errs)
        else:
            n = len(cfg[suite].get("missions") or [])
            src = SUITE_TO_VLA_DIR.get(suite, "—")
            print(f"[{suite}] OK ({n} missions, source={src})")
    if all_errs:
        print("\n".join(all_errs))
        print(f"\n{len(all_errs)} issue(s) — fix yaml or VLA_data.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

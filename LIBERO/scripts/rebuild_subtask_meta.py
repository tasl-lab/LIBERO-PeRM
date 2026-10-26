"""Rebuild stored per-demo subtask metadata under the current schema.

Deterministic recomputation from stored segments + ee_pos — equivalent to
regenerating the demos with today's `gen_metadata` (used for v2→v3 split and
v3→v4 pref-critical scope fix). Only `meta/subtasks` and the schema_version
attr change; everything else in the file is untouched.

Usage (from LIBERO/):
    python scripts/rebuild_subtask_meta.py "VLA_data/MetaSmoke/Mission_*.hdf5"
"""
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MotionGen.gen_metadata import (GenMetadataCollector, read_demo_meta,  # noqa: E402
                                    SCHEMA_VERSION)


def rebuild(pattern: str) -> None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    for path in paths:
        with h5py.File(path, "r+") as f:
            for k in f["data"]:
                grp = f["data"][k]
                meta = read_demo_meta(grp)
                if meta is None or "segments" not in meta:
                    print(f"{os.path.basename(path)} {k}: no rebuildable meta, skipped")
                    continue
                hp = meta.get("high_pref")
                c = GenMetadataCollector(
                    task_plan=meta["task_plan"],
                    naive_plan=meta["naive_plan"],
                    low_pref_entries=(meta.get("low_prefs_applied") or {}).get("entries"),
                    high_pref_desc=hp["description"] if hp else None,
                    mission_id=meta.get("mission_id"),
                )
                ee_pos = np.asarray(grp["obs/ee_pos"][()], dtype=float)
                subs = c._build_subtasks(meta["segments"], ee_pos,
                                         int(meta["num_steps"]))
                g = grp["meta"]
                del g["subtasks"]
                g.create_dataset("subtasks", data=json.dumps(subs))
                g.attrs["schema_version"] = SCHEMA_VERSION
                print(f"{os.path.basename(path)} {k}: -> v{SCHEMA_VERSION}  "
                      f"{[(s['plan_idx'], s['phase'], s['pref_critical']) for s in subs]}")
    print("done")


if __name__ == "__main__":
    rebuild(sys.argv[1] if len(sys.argv) > 1 else
            "/data3/zhixu/LIBERO-PeRM/VLA_data/MetaSmoke/Mission_*.hdf5")

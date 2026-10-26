"""Replay GT-pref demos through TaskPlanProgressPreference and report scores.

Loads each demo's MuJoCo states and walks them through the metric without
running the policy. Mission's task_plan comes from mission_info.yaml.

Usage:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python -m Preference.eval_task_plan_progress_on_demos \
        --hdf5 /data3/zhixu/LIBERO-PeRM/VLA_data/SinglePref/Mission_3-1_004_1.hdf5 \
        --mission_info dataset/series_3/perm_3_1/libero_004_1/mission_info.yaml \
        --n_demos 10
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import h5py
import numpy as np
import yaml
from robosuite.utils.binding_utils import MjSim, MjSimState

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LIBERO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_ROOT)

from Preference.high_level.task_plan_progress import (  # noqa: E402
    TaskPlanProgressPreference,
    TaskPlanCompliancePreference,
    load_task_plan_progress,
)


_CABINET_SUFFIXES = (
    "_cabinet_top", "_cabinet_middle", "_cabinet_bottom",
    "_top_side", "_top_region", "_middle_region", "_bottom_region",
    "_right_contain_region", "_left_contain_region",
    "_front_contain_region", "_back_contain_region", "_contain_region",
    "_button", "_cook_region",
)


def _strip_region_suffix(name: str) -> str:
    for s in _CABINET_SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


_BODY_BLACKLIST_PREFIXES = ("robot0_", "gripper0_", "mount0_")
_BODY_BLACKLIST_NAMES = {"world", "floor", "table", "living_room_table", "living_room_table_col"}


def _is_scene_body(name: str) -> bool:
    if not name or name in _BODY_BLACKLIST_NAMES:
        return False
    return not any(name.startswith(p) for p in _BODY_BLACKLIST_PREFIXES)


def _build_obj_body_id(sim, expected_objs: List[str]) -> Dict[str, int]:
    """Map scene-object bodies → body_id (excludes robot/mount/world/table)."""
    body_names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
    out: Dict[str, int] = {}
    for bn in body_names:
        if not _is_scene_body(bn):
            continue
        if bn.endswith("_main"):
            out[bn[: -len("_main")]] = sim.model.body_name2id(bn)
        else:
            out[bn] = sim.model.body_name2id(bn)
    # Region-suffix fallback for expected objs (e.g. wooden_cabinet_1_top_side).
    for obj in expected_objs:
        if obj in out:
            continue
        base = _strip_region_suffix(obj)
        if base in out:
            out[obj] = out[base]
    return out


def _wrap_env(sim, obj_body_id: Dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        sim=sim,
        obj_body_id=obj_body_id,
        robots=[SimpleNamespace(eef_name="gripper0_eef")],
    )


def _load_metric(mission_info_path: str, plan_key: str = "task_plan",
                 metric_name: str = "task_plan_progress") -> TaskPlanProgressPreference:
    """Load metric from mission_info.yaml, using `plan_key` (task_plan / naive_plan).

    `metric_name` picks the class: task_plan_progress (coarse per-obj coverage)
    or task_plan_compliance (per-subtask 2-stage engaged/completed).
    """
    with open(mission_info_path, "r") as f:
        data = yaml.safe_load(f) or {}
    plan = data.get(plan_key) or []
    plan_tuples = [
        (str(s[0]), str(s[1])) for s in plan
        if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]
    ]
    if not plan_tuples:
        raise RuntimeError(f"No {plan_key} in {mission_info_path}")
    cls = (TaskPlanCompliancePreference if metric_name == "task_plan_compliance"
           else TaskPlanProgressPreference)
    return cls(task_plan=plan_tuples)


def _replay_demo(sim, states: np.ndarray, env, metric, max_steps: Optional[int] = None) -> dict:
    metric.reset(env)
    n = states.shape[0] if max_steps is None else min(states.shape[0], max_steps)
    for t in range(n):
        sim.set_state(MjSimState.from_flattened(states[t], sim))
        sim.forward()
        metric.get_reward(env)
    return {"score": metric.compute_score(), "raw": metric.raw()}


def _summarise(results: List[dict], label: str) -> None:
    scores = np.asarray([r["score"] for r in results], dtype=np.float64)
    print(f"\n=== {label} (n={len(results)}) ===")
    print(f"  score   mean={scores.mean():.3f}  median={np.median(scores):.3f}  std={scores.std():.3f}  min={scores.min():.3f}  max={scores.max():.3f}")
    # Spurious / order breakdown.
    spurious = sum(1 for r in results if r["raw"]["spurious_first_obj"])
    full_order = sum(1 for r in results if r["raw"]["order_match"] >= 0.999)
    print(f"  spurious_first non-null: {spurious}/{len(results)}")
    print(f"  order_match==1.0       : {full_order}/{len(results)}")
    if results:
        print(f"  example raw (demo[0])  : {results[0]['raw']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="Path to merged Mission_*.hdf5")
    ap.add_argument("--mission_info", required=True, help="Path to mission_info.yaml")
    ap.add_argument("--n_demos", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--plan_key", choices=["task_plan", "naive_plan"], default="task_plan",
                    help="Which plan to evaluate against (naive_plan = naive baseline check).")
    ap.add_argument("--metric", choices=["task_plan_progress", "task_plan_compliance"],
                    default="task_plan_progress", help="Metric class to replay.")
    args = ap.parse_args()

    print(f"hdf5         : {args.hdf5}")
    print(f"mission_info : {args.mission_info}")
    print(f"plan_key     : {args.plan_key}")

    print(f"metric       : {args.metric}")
    metric = _load_metric(args.mission_info, plan_key=args.plan_key, metric_name=args.metric)
    print(f"expected_seq        : {metric.expected_seq}")
    print(f"expected_seq_unique : {metric.expected_seq_unique}")

    with h5py.File(args.hdf5, "r") as f:
        demos = list(f["data"].keys())[: args.n_demos]
        print(f"replaying {len(demos)} demos ...")
        # Build sim once from first demo's xml — all demos in a mission share xml.
        xml = f[f"data/{demos[0]}"].attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8")
        # Pre-rename demos embed the old absolute asset root; redirect to current repo.
        xml = xml.replace("/data3/zhixu/P_Bench/", "/data3/zhixu/LIBERO-PeRM/")
        sim = MjSim.from_xml_string(xml)

        obj_body_id = _build_obj_body_id(sim, list(metric.expected_set))
        missing = metric.expected_set - set(obj_body_id.keys())
        if missing:
            print(f"WARN: missing body ids for {sorted(missing)}")
        env = _wrap_env(sim, obj_body_id)

        results = []
        for demo_key in demos:
            states = np.asarray(f[f"data/{demo_key}/states"][:])
            try:
                results.append(_replay_demo(sim, states, env, metric, args.max_steps))
            except Exception as e:
                print(f"[{demo_key}] replay error: {e}")

    _summarise(results, label=f"GT-pref demos ({os.path.basename(args.hdf5)})")


if __name__ == "__main__":
    main()

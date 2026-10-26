"""Stock-LIBERO 0-shot evaluator for WebSocket VLA policy servers.

Diagnostic utility: runs a VLA (hosted on ws://host:port) on the UNMODIFIED
LIBERO benchmark suites (libero_spatial / libero_object / libero_goal /
libero_10 / libero_90), to compare against published per-suite SR (e.g. the
SmolVLA paper numbers: Spatial 90 / Object 96 / Goal 92 / Long 71).

Deliberately standalone — no coupling to P_Bench's `task_suites.yaml` or
mission-id routing. Helpers are copied verbatim from
`LIBERO/PolicyEval/eval.py:163-188` so the obs dict we send to the server
matches the one the P-Bench eval sends.

Usage:
    # Smoke (5 min):
    python eval_stock_libero.py \\
        --suite libero_spatial --num_tasks 3 --trials_per_task 2 \\
        --output_dir /data3/zhixu/LIBERO-PeRM/Policy_output/0shot/stock_libero_smoke/smolvla

    # Full per-suite (~2h each):
    for s in libero_spatial libero_object libero_goal libero_10; do
        python eval_stock_libero.py \\
            --suite $s --trials_per_task 20 \\
            --output_dir /data3/zhixu/LIBERO-PeRM/Policy_output/0shot/stock_libero_smolvla/$s
    done
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import yaml

# Stock LIBERO API (unmodified upstream)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# `libero.libero.benchmark.Benchmark.get_task_init_states` does
# `torch.load(init_states_path)` without `weights_only=False`; on
# torch>=2.6 the default flipped to weights_only=True and those .init
# files pickle numpy arrays. Monkey-patch to load with weights_only=False.
import torch as _torch
_orig_torch_load = _torch.load
def _torch_load_unsafe(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
_torch.load = _torch_load_unsafe

# Reuse P_Bench's WebsocketBackend (same wire protocol our server speaks)
sys.path.insert(0, "/home/vla-reasoning/proj/P_Bench/LIBERO")
from PolicyEval.backends.websocket_backend import WebsocketBackend  # noqa: E402

logger = logging.getLogger("eval_stock_libero")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirror UniVLA's run_libero_eval.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

RESIZE_SIZE = 360  # match lerobot LiberoEnv default (observation_height/width=360); SmolVLA-libero ckpt was trained on 360→model-internal-resize_with_pad to 512
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # 7-dim, gripper open

# Aligned with lerobot/src/lerobot/envs/libero.py:TASK_SUITE_MAX_STEPS
SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ─────────────────────────────────────────────────────────────────────────────
# Obs helpers — VERBATIM copies from LIBERO/PolicyEval/eval.py:163-188
# so the obs dict matches what our existing servers know how to parse.
# ─────────────────────────────────────────────────────────────────────────────


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[:2] == (size, size):
        return img.astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _quat2axisangle(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = math.sqrt(1.0 - q[3] * q[3])
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def _obs_element(raw_obs: Dict, prompt: str) -> Dict:
    img = _resize(np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]), RESIZE_SIZE)
    wrist = _resize(np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]), RESIZE_SIZE)
    state = np.concatenate([
        np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        _quat2axisangle(raw_obs["robot0_eef_quat"]),
        np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
    ])
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": prompt}


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────


def run_episode(env, backend: WebsocketBackend, init_state, prompt: str,
                max_steps: int, replan_steps: int) -> tuple[bool, int]:
    env.reset()
    obs = env.set_init_state(init_state)

    # Sim stabilization — match UniVLA's convention.
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    action_plan: deque = deque()
    success = False
    steps = 0

    for t in range(max_steps):
        if not action_plan:
            reply = backend.infer(_obs_element(obs, prompt))
            chunk = np.asarray(reply["actions"])
            action_plan.extend(chunk[:replan_steps])

        action = action_plan.popleft()
        action = np.asarray(action).tolist()
        obs, reward, done, info = env.step(action)
        steps = t + 1
        if done:
            success = True
            break

    return success, steps


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True,
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--trials_per_task", type=int, default=20)
    p.add_argument("--num_tasks", type=int, default=None,
                   help="Limit number of tasks in the suite. Default: all.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--replan_steps", type=int, default=1,
                   help="Open-loop steps before replanning. lerobot SmolVLA ckpt config has n_action_steps=1 (per-step replan).")
    p.add_argument("--max_steps_override", type=int, default=None)
    p.add_argument("--camera_resolution", type=int, default=360,
                   help="OffScreenRenderEnv camera_heights=camera_widths. Default 360 matches lerobot LiberoEnv + SmolVLA-libero training.")
    args = p.parse_args()

    max_steps = args.max_steps_override or SUITE_MAX_STEPS[args.suite]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading suite: {args.suite}")
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    total_tasks_in_suite = task_suite.n_tasks
    n_tasks = args.num_tasks if args.num_tasks is not None else total_tasks_in_suite
    logger.info(f"  total tasks in suite: {total_tasks_in_suite}  running: {n_tasks}")

    logger.info(f"Connecting to ws://{args.host}:{args.port}")
    backend = WebsocketBackend(args.host, args.port)

    per_task = []
    total_episodes = 0
    total_successes = 0
    t_start = time.monotonic()

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        logger.info(f"── task {task_id}/{n_tasks - 1}: {task.language[:60]!r}")
        logger.info(f"   BDDL: {bddl}")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=args.camera_resolution,
            camera_widths=args.camera_resolution,
        )
        env.seed(0)
        init_states = task_suite.get_task_init_states(task_id)

        t_task = 0
        s_task = 0
        for ep_idx in range(args.trials_per_task):
            state = init_states[ep_idx % len(init_states)]
            t0 = time.monotonic()
            success, steps = run_episode(
                env, backend, state, task.language, max_steps, args.replan_steps,
            )
            t_task += 1
            s_task += int(success)
            logger.info(f"   ep {ep_idx}: {'✅' if success else '❌'}  steps={steps}  wall={time.monotonic() - t0:.1f}s")

        env.close()

        sr = s_task / t_task if t_task else 0.0
        logger.info(f"   task SR = {sr:.2f}  ({s_task}/{t_task})")
        per_task.append({
            "task_id": task_id,
            "language": task.language,
            "bddl_file": task.bddl_file,
            "problem_folder": task.problem_folder,
            "successes": s_task,
            "trials": t_task,
            "sr": round(sr, 4),
        })
        total_episodes += t_task
        total_successes += s_task

    wall = time.monotonic() - t_start
    summary = {
        "suite": args.suite,
        "overall_sr": round(total_successes / total_episodes, 4) if total_episodes else 0.0,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "n_tasks": n_tasks,
        "trials_per_task": args.trials_per_task,
        "replan_steps": args.replan_steps,
        "max_steps": max_steps,
        "camera_resolution": args.camera_resolution,
        "wall_time_s": round(wall, 1),
        "server": f"ws://{args.host}:{args.port}",
        "per_task": per_task,
    }

    out_file = out_dir / "summary.yaml"
    with open(out_file, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    logger.info("=" * 60)
    logger.info(f"Suite {args.suite}: SR = {summary['overall_sr']:.3f}  "
                f"({total_successes}/{total_episodes})  wall={wall:.0f}s")
    logger.info(f"Summary: {out_file}")
    logger.info("=" * 60)

    try:
        backend.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

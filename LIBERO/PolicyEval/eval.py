"""Policy eval — runfile-driven sweep over (condition × mission × trial).

One entry point. One file. Reads a runfile yaml describing what to run and
which inference server to talk to, then walks every (condition, mission)
cell in-process (LIBERO import is expensive — don't pay it per cell).

Output tree:
    /data3/zhixu/LIBERO-PeRM/Policy_output/{exp_type}/{task_suite}/{exp_name}/
        {model}/{condition}/{mission_id}/
            Videos/Ep<ep>_success.mp4   (+ fail_case/)  [if save_video, default on]
            GIFs/Ep<ep>_success.gif     (+ fail_case/)  [if save_gif]
            prompt.txt
            rollout_summary.yaml

Runfile output toggles (all optional):
    save_hdf5:  false   # default off  (not yet implemented — True raises)
    save_video: true    # default on (mp4)
    save_gif:   false   # default off

Usage:
    python PolicyEval/eval.py /data3/zhixu/LIBERO-PeRM/VLA_infer/run_profile/ZeroShot/smoke_pi05.yaml
    python PolicyEval/eval.py <runfile.yaml> --dry_run

Preferred entry: bash /data3/zhixu/LIBERO-PeRM/VLA_infer/run_client.sh <runfile.yaml>
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBERO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LIBERO_ROOT not in sys.path:
    sys.path.insert(0, _LIBERO_ROOT)

from libero.libero.envs import OffScreenRenderEnv

from configs.task_suites import load_suite
from configs.check_suite_sync import verify_suite, _resolve_dataset_dir
from PolicyEval.backends import WebsocketBackend  # HttpBackend lazy-imported below
from PolicyEval.low_pref_reward import LowPrefAccumulator, load_motion_prefs
from PolicyEval.hdf5_writer import RolloutRecorder
from Preference.high_level.task_plan_progress import load_task_plan_progress, load_task_plan_compliance
from Utility.frame import get_img_from_obs

logger = logging.getLogger("policy_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OUT_ROOT = os.environ.get(
    "POLICY_OUTPUT_ROOT",
    os.path.join(os.path.dirname(_LIBERO_ROOT), "Policy_output"),
)
DATASET_ROOT = os.path.join(_LIBERO_ROOT, "dataset")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224
NUM_STEPS_WAIT = 10
MAX_STEPS_DEFAULT = 600
# After done=True we keep the policy looping to record the post-success state
# (useful for skill-compliance analysis: lay_down release, throw landing, etc.).
EXTRA_STEPS_AFTER_SUCCESS = 21
NUM_INIT_STATES_CACHED = 15


class ServerMismatchError(RuntimeError):
    """Server's reported model/checkpoint doesn't match runfile expectation."""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt composition
# ─────────────────────────────────────────────────────────────────────────────


def compose_prompt(mission_info: Dict, condition: str, *,
                   mission_id: str = None, prompt_source_yaml: str = None,
                   exp_type: str = None,
                   comp_pref_yaml: str = None, comp_pref_setting: str = None) -> str:
    """Modes (priority):

    0. **base condition (always)**: use `libero_instruction` (verbatim LIBERO task
       lan from bddl filename) regardless of exp_type. Both 0-shot LIBERO ckpts
       and LoRA ckpts (trained on anchor `Task: ... Preference: ...` SingPref data)
       see this distribution at base; the LoRA's underlying pi05_libero / pi0_libero
       was trained on plain LIBERO natural language, so libero_instruction matches
       its base-task prior. Anchor `Task: <task>.` (no Preference) would be double-OOD.
    1. **comp_pref_yaml given** (pref condition, ComprePref mode): read the mission's
       `S1` (fused) or `S2` (reasoning) string from series_7/language.yaml and return
       it verbatim as the prompt — no Task:/Preference: anchor wrapping.
    2. **prompt_source_yaml given** (pref condition): read tasks + preferences,
       format `Task: ... Preference: ...` — must byte-for-byte match the train-side
       convert script's compose_prompt.
    3. **legacy** (no source yaml, pref condition): read language_instructions
       / high_pref_lan / low_pref_lan from mission_info.yaml, plain " "-join.
    """
    if condition == "base":
        lib = (mission_info.get("libero_instruction") or "").strip().rstrip(".")
        if not lib:
            raise ValueError(
                f"base condition requires `libero_instruction` in mission_info "
                f"(mission {mission_id!r}); run dataset/_propagate_libero_instruction.py --apply"
            )
        # NOTE: OFT 4-in-1 was trained with LIBERO `task.language` strings which
        # have NO trailing period. Adding a "." is enough to OOD the prompt and
        # tank SR from 100% → 0% (verified via mode D vs E ablation 2026-05-01).
        return lib

    if comp_pref_yaml:
        if mission_id is None:
            raise ValueError("comp_pref_yaml requires mission_id")
        if comp_pref_setting not in ("S1", "S2"):
            raise ValueError(
                f"comp_pref_setting must be 'S1' or 'S2', got {comp_pref_setting!r}"
            )
        with open(comp_pref_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
        entry = next(
            (m for m in (data.get("missions") or []) if m.get("mission_id") == mission_id),
            None,
        )
        if not entry:
            raise ValueError(f"mission_id {mission_id!r} not in {comp_pref_yaml}")
        text = (entry.get(comp_pref_setting) or "").strip()
        if not text:
            raise ValueError(
                f"empty {comp_pref_setting} for mission {mission_id!r} in {comp_pref_yaml}"
            )
        return text

    if prompt_source_yaml:
        if mission_id is None:
            raise ValueError("prompt_source_yaml requires mission_id")
        with open(prompt_source_yaml, "r") as f:
            ts = yaml.safe_load(f) or {}
        task = (ts.get("tasks", {}).get(mission_id) or "").strip().rstrip(".")
        if condition == "base":
            return f"Task: {task}." if task else ""
        if condition != "pref":
            raise ValueError(f"Unknown condition: {condition!r}")
        pref = (ts.get("preferences", {}).get(mission_id) or "").strip().rstrip(".")
        if not task and not pref:
            return ""
        if not pref:
            return f"Task: {task}."
        if not task:
            return f"Preference: {pref}."
        return f"Task: {task}. Preference: {pref}."

    # Legacy: read from mission_info.yaml fields
    inst = (mission_info.get("language_instructions") or "").strip()
    if condition == "base":
        return inst
    if condition != "pref":
        raise ValueError(f"Unknown condition: {condition!r}")
    high = (mission_info.get("high_pref_lan") or "").strip()
    low = (mission_info.get("low_pref_lan") or "").strip()
    parts = [p for p in (inst, high, low) if p]
    return " ".join(p if p.endswith(".") else p + "." for p in parts)


# ─────────────────────────────────────────────────────────────────────────────
# Mission resolution
# ─────────────────────────────────────────────────────────────────────────────


def parse_mission_id(mission_id: str) -> Tuple[str, str]:
    """'1-1_004' → ('perm_1_1', 'libero_004'); '3-1_004_1' → (..., 'libero_004_1')."""
    m = re.match(r"^(\d+-\d+)_(.+)$", mission_id)
    if not m:
        raise ValueError(f"Invalid mission_id: {mission_id!r}")
    return f"perm_{m.group(1).replace('-', '_')}", f"libero_{m.group(2)}"


def resolve_mission_dir(mission_id: str) -> str:
    perm, libero = parse_mission_id(mission_id)
    from configs.path_utils import mission_dir as _mdir
    mdir = _mdir(perm, libero, dataset_root=DATASET_ROOT)
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f"Mission dir not found: {mdir}")
    return mdir


def pick_bddl(mission_dir: str, condition: str) -> str:
    """Series 3 + base → no_pref_layout/*.bddl; otherwise the mission's default bddl."""
    np_dir = os.path.join(mission_dir, "no_pref_layout")
    use_np = condition == "base" and os.path.isdir(np_dir)
    search_dir = np_dir if use_np else mission_dir
    bddls = [p for p in os.listdir(search_dir) if p.endswith(".bddl")]
    if not bddls:
        raise FileNotFoundError(f"No .bddl in {search_dir}")
    if len(bddls) > 1:
        logger.warning(f"Multiple bddls in {search_dir}, picking first: {bddls[0]}")
    return os.path.join(search_dir, bddls[0])


# ─────────────────────────────────────────────────────────────────────────────
# Server validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_server_metadata(metadata: Dict, expected_config: str, expected_checkpoint: str) -> None:
    """Strict equality check: the server was launched from the same runfile as the
    client (both read `server.config` / `server.checkpoint`), so the server's reported
    metadata must exactly match. Any drift = launched against wrong yaml → abort."""
    if not metadata:
        raise ServerMismatchError("Server returned empty metadata; cannot validate.")

    got_config = metadata.get("config_name")
    got_ckpt = metadata.get("checkpoint_dir")
    if got_config != expected_config:
        raise ServerMismatchError(
            f"Server config mismatch:\n  expected: {expected_config!r}\n  reported: {got_config!r}\n  metadata: {metadata}"
        )
    if got_ckpt != expected_checkpoint:
        raise ServerMismatchError(
            f"Server checkpoint mismatch:\n  expected: {expected_checkpoint!r}\n  reported: {got_ckpt!r}"
        )
    logger.info(f"Server metadata OK  (config={expected_config!r}  ckpt={expected_checkpoint!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Observation pre-processing
# ─────────────────────────────────────────────────────────────────────────────


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _quat2axisangle(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = math.sqrt(1.0 - q[3] * q[3])
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def _obs_element(raw_obs: Dict, prompt: str, env=None) -> Dict:
    img = _resize(np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]), RESIZE_SIZE)
    wrist = _resize(np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]), RESIZE_SIZE)
    state = np.concatenate([
        np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        _quat2axisangle(raw_obs["robot0_eef_quat"]),
        np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
    ])
    out = {"observation/image": img, "observation/wrist_image": wrist,
           "observation/state": state, "prompt": prompt}
    # Optional: expose controller's site-frame ee_ori_mat (3x3) for backends that
    # need the orientation in the same frame X-VLA's training proprio used.
    # `robot0_eef_quat` is the body-frame quat; site_xmat (= controller.ee_ori_mat)
    # differs by a 90° rot on Panda. HttpBackend reads this when present.
    if env is not None:
        try:
            mat = np.asarray(env.env.robots[0].controller.ee_ori_mat, dtype=np.float32).reshape(3, 3)
            out["observation/eef_ori_mat"] = mat
        except Exception:
            pass
    return out


def _summarize_gripper_debug(raw_values: List[float], processed_values: List[float]) -> Dict:
    if not raw_values:
        return {}
    raw = np.asarray(raw_values, dtype=np.float32)
    processed = np.asarray(processed_values, dtype=np.float32)
    return {
        "num_actions": int(len(raw_values)),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "processed_close_count": int(np.sum(processed > 0)),
        "processed_open_count": int(np.sum(processed < 0)),
        "raw_first_40": [float(x) for x in raw_values[:40]],
        "processed_first_40": [float(x) for x in processed_values[:40]],
    }


def _init_state_path(init_dir: str, ep: int, base_name: str) -> str:
    """Cache key for env initial state — wraps over NUM_INIT_STATES_CACHED so that
    trial 0..N-1 reuse a fixed pool of saved states (deterministic replay)."""
    return os.path.join(init_dir, f"{base_name}_{ep % NUM_INIT_STATES_CACHED}.init")


# ─────────────────────────────────────────────────────────────────────────────
# Episode loop
# ─────────────────────────────────────────────────────────────────────────────


def run_episode(
    env: OffScreenRenderEnv,
    backend: WebsocketBackend,
    prompt: str,
    motion_prefs,
    max_steps: int,
    replan_steps: int,
    episode_idx: int,
    collect_frames: bool,
    init_state_path: Optional[str] = None,
    save_init: bool = False,
    abs_action_mode: bool = False,
    debug_gripper: bool = False,
    recorder=None,
) -> Tuple[bool, int, LowPrefAccumulator, List[np.ndarray], Dict]:
    env.reset()
    # Restore controller to delta mode BEFORE warmup, in case a previous
    # episode in the same env left use_delta=False (the abs-mode switch is
    # applied AFTER warmup; we must undo it before the next episode resets).
    try:
        for robot in env.env.robots:
            robot.controller.use_delta = True
    except Exception:
        pass
    action_plan = collections.deque()
    replay_frames: List[np.ndarray] = []

    # Init state: load if cached, else step warmup and (optionally) save.
    # Cached init state guarantees trial-to-trial determinism on the same seed.
    if init_state_path and os.path.exists(init_state_path):
        import pickle
        with open(init_state_path, "rb") as f:
            obs = env.set_init_state(pickle.load(f))
        wait_for_settle = False
    else:
        obs = env._get_observations() if hasattr(env, "_get_observations") else None
        wait_for_settle = True

    acc = LowPrefAccumulator(motion_prefs)
    raw_gripper_values: List[float] = []
    processed_gripper_values: List[float] = []

    if wait_for_settle:
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        if save_init and init_state_path:
            import pickle
            os.makedirs(os.path.dirname(init_state_path), exist_ok=True)
            with open(init_state_path, "wb") as f:
                pickle.dump(env.get_sim_state(), f)
            logger.info(f"Saved init state: {init_state_path}")

    # X-VLA libero: switch OSC controller to absolute targeting AFTER warmup.
    if abs_action_mode:
        for robot in env.env.robots:
            robot.controller.use_delta = False

    success, n_executed, t = False, 0, 0
    while t < max_steps:
        try:
            if not action_plan:
                response = backend.infer(_obs_element(obs, prompt, env))
                chunk = response["actions"]
                raw_chunk = response.get("raw_actions")
                take = min(len(chunk), replan_steps)
                if take < 1:
                    raise RuntimeError(f"Server returned empty chunk: {chunk.shape}")
                for idx, a in enumerate(chunk[:take]):
                    if debug_gripper and raw_chunk is not None and idx < len(raw_chunk):
                        raw_gripper_values.append(float(raw_chunk[idx][-1]))
                        processed_gripper_values.append(float(a[-1]))
                    action_plan.append(np.asarray(a, dtype=np.float32))
                # Inform backends that maintain a closed-loop chunk pointer
                # (e.g., HttpBackend) how many frames will actually be executed,
                # so they can snap proprio to action[take-1] not action[-1].
                if hasattr(backend, "notify_consumed"):
                    backend.notify_consumed(take)

            action = action_plan.popleft()
            obs, reward, done, _ = env.step(action.tolist())
            n_executed += 1

            step_rewards = acc.step(env, action.tolist(), success=bool(done))
            if recorder is not None:
                try:
                    sim_state = env.get_sim_state() if hasattr(env, "get_sim_state") else env.sim.get_state().flatten()
                    recorder.append(obs, action.tolist(), reward, done, sim_state, step_rewards)
                except Exception as _e:
                    logger.warning(f"recorder.append failed: {_e}")

            if collect_frames:
                fv = np.ascontiguousarray(get_img_from_obs(obs, "agentview_image"))
                eih = np.ascontiguousarray(get_img_from_obs(obs, "robot0_eye_in_hand_image"))
                replay_frames.append(np.concatenate([fv, eih], axis=1))

            if done:
                success = True
                # Continue policy loop for EXTRA_STEPS_AFTER_SUCCESS frames so
                # the rollout captures post-success behaviour (e.g. lay_down
                # release, throw landing) — useful for skill-compliance VLM.
                for _ in range(EXTRA_STEPS_AFTER_SUCCESS):
                    try:
                        if not action_plan:
                            response = backend.infer(_obs_element(obs, prompt, env))
                            chunk = response["actions"]
                            take = min(len(chunk), replan_steps)
                            if take < 1:
                                break
                            for a in chunk[:take]:
                                action_plan.append(np.asarray(a, dtype=np.float32))
                            if hasattr(backend, "notify_consumed"):
                                backend.notify_consumed(take)
                        action_extra = action_plan.popleft()
                        obs, reward, _done, _ = env.step(action_extra.tolist())
                        n_executed += 1
                        step_rewards = acc.step(env, action_extra.tolist())
                        if recorder is not None:
                            try:
                                sim_state = env.get_sim_state() if hasattr(env, "get_sim_state") else env.sim.get_state().flatten()
                                recorder.append(obs, action_extra.tolist(), reward, _done, sim_state, step_rewards)
                            except Exception:
                                pass
                        if collect_frames:
                            fv = np.ascontiguousarray(get_img_from_obs(obs, "agentview_image"))
                            eih = np.ascontiguousarray(get_img_from_obs(obs, "robot0_eye_in_hand_image"))
                            replay_frames.append(np.concatenate([fv, eih], axis=1))
                    except Exception:
                        break
                break
            t += 1
        except Exception as e:
            logger.exception(f"Episode {episode_idx} step {t}: {e}")
            break

    return success, n_executed, acc, replay_frames, _summarize_gripper_debug(
        raw_gripper_values, processed_gripper_values
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-cell rollout (one model × one condition × one mission, num_trials eps)
# ─────────────────────────────────────────────────────────────────────────────


def run_cell(cfg: Dict, condition: str, mission_id: str, backend: WebsocketBackend,
             server_metadata: Dict) -> Dict:
    srv = cfg["server"]
    mission_dir = resolve_mission_dir(mission_id)
    with open(os.path.join(mission_dir, "mission_info.yaml"), "r") as f:
        mission_info = yaml.safe_load(f) or {}
    bddl_path = pick_bddl(mission_dir, condition)
    prompt = compose_prompt(
        mission_info, condition,
        mission_id=mission_id,
        prompt_source_yaml=cfg.get("prompt_source_yaml"),
        exp_type=cfg.get("exp_type"),
        comp_pref_yaml=cfg.get("comp_pref_yaml"),
        comp_pref_setting=cfg.get("comp_pref_setting"),
    )

    save_hdf5 = bool(cfg.get("save_hdf5", False))
    save_video = bool(cfg.get("save_video", True))
    save_gif = bool(cfg.get("save_gif", False))

    # Output dirs
    parts = [OUT_ROOT, cfg["exp_type"], cfg["task_suite"]]
    if cfg.get("experiment_name"):
        parts.append(cfg["experiment_name"])
    parts += [srv["name"], condition, mission_id]
    cell_root = os.path.join(*parts)
    videos_dir = os.path.join(cell_root, "Videos")
    videos_fail = os.path.join(videos_dir, "fail_case")
    gifs_dir = os.path.join(cell_root, "GIFs")
    gifs_fail = os.path.join(gifs_dir, "fail_case")
    init_dir = os.path.join(cell_root, "init_states")
    os.makedirs(init_dir, exist_ok=True)
    if save_video:
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(videos_fail, exist_ok=True)
    if save_gif:
        os.makedirs(gifs_dir, exist_ok=True)
        os.makedirs(gifs_fail, exist_ok=True)
    hdf5_dir = os.path.join(cell_root, "HDF5")
    hdf5_fail = os.path.join(hdf5_dir, "fail_case")
    if save_hdf5:
        os.makedirs(hdf5_dir, exist_ok=True)
        os.makedirs(hdf5_fail, exist_ok=True)
    with open(os.path.join(cell_root, "prompt.txt"), "w") as f:
        f.write(prompt + "\n")

    logger.info(f"BDDL   : {bddl_path}")
    logger.info(f"Prompt : {prompt}")
    logger.info(f"Output : {cell_root}")

    max_steps = cfg.get("max_steps") or MAX_STEPS_DEFAULT
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        # robosuite default horizon=1000 forces done=True at step 1000 → next env.step
        # raises "executing action in terminated episode". Push horizon above our cap
        # plus EXTRA_STEPS_AFTER_SUCCESS so post-success continuation has headroom.
        horizon=max_steps + NUM_STEPS_WAIT + EXTRA_STEPS_AFTER_SUCCESS + 10,
    )
    env.seed(cfg["seed"])
    # Absolute control mode: opt-in via runfile `server.controller_mode: absolute`.
    # Default is "relative" (delta). The official X-VLA libero_client uses absolute,
    # so HTTP-protocol X-VLA runfiles set this; the legacy lerobot xvla server (WS)
    # outputs deltas and must NOT switch to absolute. Switch happens AFTER warmup
    # inside run_episode (warmup needs delta mode for the dummy [0,...,-1] no-op).
    abs_action_mode = ((srv.get("controller_mode") or "relative").lower() == "absolute")
    motion_prefs = load_motion_prefs(os.path.join(mission_dir, "mission_info.yaml")) if condition == "pref" else []
    # Always attach task_plan_compliance (user-profile metric) — not gated on
    # condition since it monitors the planned obj sequence regardless of pref.
    tpp = load_task_plan_compliance(os.path.join(mission_dir, "mission_info.yaml"))
    if tpp is not None:
        motion_prefs.append(tpp)
    logger.info(f"Loaded {len(motion_prefs)} preferences (low + profile).")

    base_name = os.path.splitext(os.path.basename(bddl_path))[0]
    summary = {
        "mission_id": mission_id, "task_suite": cfg["task_suite"], "exp_type": cfg["exp_type"],
        "experiment_name": cfg.get("experiment_name"), "model": srv["name"],
        "condition": condition, "bddl_file": bddl_path, "prompt": prompt,
        "seed": cfg["seed"], "max_steps": max_steps, "num_trials": cfg["num_trials"],
        "server_metadata": server_metadata, "episodes": [],
    }

    collect_frames = save_video or save_gif
    debug_gripper = bool(cfg["server"].get("debug_gripper", False))

    n_success = 0
    for ep in range(cfg["num_trials"]):
        env.seed(cfg["seed"] + ep)
        init_path = _init_state_path(init_dir, ep, base_name)

        recorder = None
        if save_hdf5:
            recorder = RolloutRecorder()
            try:
                with open(bddl_path, "r") as _bf:
                    pass  # bddl_path is text; xml comes from sim
            except Exception:
                pass
        t0 = time.time()
        success, n_steps, acc, frames, gripper_debug = run_episode(
            env=env, backend=backend, prompt=prompt, motion_prefs=motion_prefs,
            max_steps=max_steps, replan_steps=cfg["replan_steps"],
            init_state_path=init_path,
            save_init=True, episode_idx=ep, collect_frames=collect_frames,
            abs_action_mode=abs_action_mode, debug_gripper=debug_gripper,
            recorder=recorder,
        )
        wall = time.time() - t0

        tag = "success" if success else "failure"
        video_rel = gif_rel = hdf5_rel = None

        if save_video and frames:
            vdir = videos_dir if success else videos_fail
            vpath = os.path.join(vdir, f"Ep{ep}_{tag}.mp4")
            imageio.mimwrite(vpath, frames, fps=20)
            video_rel = os.path.relpath(vpath, cell_root)

        if save_gif and frames:
            gdir = gifs_dir if success else gifs_fail
            gpath = os.path.join(gdir, f"Ep{ep}_{tag}.gif")
            imageio.mimwrite(gpath, frames, fps=20)
            gif_rel = os.path.relpath(gpath, cell_root)

        if save_hdf5 and recorder is not None:
            try:
                recorder.set_model_xml(env.sim.model.get_xml())
            except Exception:
                pass
            hdir = hdf5_dir if success else hdf5_fail
            hpath = os.path.join(hdir, f"Ep{ep}_{tag}.hdf5")
            try:
                recorder.write(hpath)
                hdf5_rel = os.path.relpath(hpath, cell_root)
            except Exception as _e:
                logger.warning(f"hdf5 write failed: {_e}")

        if success:
            n_success += 1

        # Split bucketed summary into 3 flat keys for downstream consumers
        # (aggregate_eval.py reads `low_pref_reward` directly).
        bucket = acc.summary() or {}
        ep_entry = {
            "episode": ep, "success": success, "n_steps": n_steps, "wall_seconds": wall,
            "hdf5": hdf5_rel,
            "video": video_rel,
            "gif": gif_rel,
            "low_pref_reward": bucket.get("low", {}),
            "skill_pref_reward": bucket.get("skill", {}),
            "profile_pref_reward": bucket.get("profile", {}),
            "gripper_debug": gripper_debug,
        }
        summary["episodes"].append(ep_entry)
        logger.info(f"Ep{ep}: {'✅' if success else '❌'} | steps={n_steps} | wall={wall:.1f}s")

    summary["sr"] = n_success / max(1, cfg["num_trials"])
    summary_path = os.path.join(cell_root, "rollout_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"SR = {summary['sr']:.2f} ({n_success}/{cfg['num_trials']}) → {summary_path}")
    env.close()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Runfile + sweep
# ─────────────────────────────────────────────────────────────────────────────


def parse_runfile(path: str) -> Dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    for key in ("experiment_name", "task_suite", "server"):
        if key not in raw:
            raise ValueError(f"runfile {path!r}: missing required key {key!r}")

    exp_type = raw.get("exp_type", "0shot")
    if exp_type not in ("0shot", "lora_ft"):
        raise ValueError(f"exp_type must be 0shot or lora_ft, got {exp_type!r}")

    explicit_missions = raw.get("missions") or []
    if explicit_missions:
        missions = explicit_missions
        missing = [m for m in missions if not os.path.isdir(_resolve_dataset_dir(m))]
        if missing:
            raise RuntimeError(
                "runfile contains mission(s) with missing dataset dirs:\n"
                + "\n".join(f"  - {m}: {_resolve_dataset_dir(m)}" for m in missing)
            )
    else:
        # Prefer warning over hard failure so full-suite eval can still run
        # against the currently registered suite even if VLA_data has drifted.
        try:
            verify_suite(raw["task_suite"])
        except RuntimeError as exc:
            logger.warning(f"task_suite sync warning for {raw['task_suite']!r}:\n{exc}")
        missions = load_suite(raw["task_suite"])

    suite_set = set(load_suite(raw["task_suite"]))
    extras = [m for m in missions if m not in suite_set]
    if extras:
        logger.warning(f"{len(extras)} mission(s) not in suite {raw['task_suite']!r}: {extras[:5]}")

    conditions = raw.get("conditions") or ["base", "pref"]
    for c in conditions:
        if c not in ("base", "pref"):
            raise ValueError(f"unknown condition {c!r}")

    srv = raw["server"]
    if not isinstance(srv, dict):
        raise ValueError("`server` must be a mapping")
    for k in ("vla", "name", "port", "config", "checkpoint"):
        if k not in srv or srv[k] in (None, ""):
            raise ValueError(f"`server.{k}` is required")

    return {
        "experiment_name": raw["experiment_name"],
        "exp_type": exp_type,
        "task_suite": raw["task_suite"],
        "missions": missions,
        "conditions": conditions,
        "num_trials": int(raw.get("num_trials", 5)),
        "replan_steps": int(raw.get("replan_steps", 5)),
        "max_steps": raw.get("max_steps"),
        "seed": int(raw.get("seed", 7)),
        "save_hdf5": bool(raw.get("save_hdf5", False)),
        "save_video": bool(raw.get("save_video", True)),
        "save_gif": bool(raw.get("save_gif", False)),
        "prompt_source_yaml": raw.get("prompt_source_yaml"),
        "comp_pref_yaml": raw.get("comp_pref_yaml"),
        "comp_pref_setting": raw.get("comp_pref_setting"),
        "server": {
            "vla": srv["vla"],
            "name": srv["name"],
            "host": srv.get("host", "0.0.0.0"),
            "port": int(srv["port"]),
            "config": srv["config"],
            "checkpoint": srv["checkpoint"],
            "protocol": srv.get("protocol", "ws"),
            "steps": int(srv.get("steps", 10)),
            "controller_mode": srv.get("controller_mode", "relative"),
        },
    }


def run_sweep(cfg: Dict, dry_run: bool = False) -> None:
    srv = cfg["server"]
    cells = [(c, m) for c in cfg["conditions"] for m in cfg["missions"]]

    logger.info(f"Run    : {cfg['experiment_name']}  ({cfg['exp_type']} / {cfg['task_suite']})")
    logger.info(f"Server : ws://{srv['host']}:{srv['port']}  [vla={srv['vla']} name={srv['name']}]")
    logger.info(f"Expect : config={srv['config']!r}  ckpt={srv['checkpoint']!r}")
    logger.info(f"Sweep  : {len(cfg['conditions'])} × {len(cfg['missions'])} = {len(cells)} cells")

    if dry_run:
        for i, (c, m) in enumerate(cells, 1):
            logger.info(f"  [dry] {i:>4}/{len(cells)}: {srv['name']} | {c} | {m}")
        return

    # One backend, one validation — reused across cells.
    # `server.protocol` ∈ {"ws" (default), "http"} → swap backend type.
    proto = (srv.get("protocol") or "ws").lower()
    if proto == "http":
        from PolicyEval.backends.http_backend import HttpBackend  # lazy: only OFT path needs json_numpy
        backend = HttpBackend(
            host=srv["host"], port=srv["port"],
            config=srv.get("config", ""), checkpoint=srv.get("checkpoint", ""),
            steps=int(srv.get("steps", 10)),
        )
    else:
        backend = WebsocketBackend(host=srv["host"], port=srv["port"])
    server_metadata = backend._client.get_server_metadata()
    try:
        validate_server_metadata(server_metadata, srv["config"], srv["checkpoint"])
    except ServerMismatchError:
        backend.shutdown_server()
        backend.close()
        raise

    n_ok, n_fail = 0, 0
    try:
        for i, (condition, mission_id) in enumerate(cells, 1):
            logger.info(f"─── cell {i}/{len(cells)}: {srv['name']} / {condition} / {mission_id} ───")
            try:
                run_cell(cfg, condition, mission_id, backend, server_metadata)
                n_ok += 1
            except Exception as e:
                logger.exception(f"cell failed: {e}")
                n_fail += 1
    finally:
        backend.shutdown_server()
        backend.close()
    logger.info(f"done: {n_ok} ok, {n_fail} failed")


def main():
    ap = argparse.ArgumentParser(description="Policy eval — runfile-driven rollout sweep.")
    ap.add_argument("runfile", help="Path to runfile yaml.")
    ap.add_argument("--dry_run", action="store_true", help="List cells without running.")
    ap.add_argument("--missions", default=None,
                    help="Comma-separated mission_ids; overrides runfile.missions (re-eval subset).")
    ap.add_argument("--conditions", default=None,
                    help="Comma-separated; overrides runfile.conditions (e.g. 'base').")
    opts = ap.parse_args()

    cfg = parse_runfile(opts.runfile)
    if opts.missions:
        cfg["missions"] = [m.strip() for m in opts.missions.split(",") if m.strip()]
    if opts.conditions:
        conds = [c.strip() for c in opts.conditions.split(",") if c.strip()]
        for c in conds:
            if c not in ("base", "pref"):
                raise ValueError(f"unknown condition {c!r}")
        cfg["conditions"] = conds
    run_sweep(cfg, dry_run=opts.dry_run)


if __name__ == "__main__":
    main()

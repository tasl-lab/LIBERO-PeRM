"""
Generate heuristic trajectories as videos and hdf5 files.
"""
import sys
import os
import random
import subprocess
import logging
import numpy as np

logger = logging.getLogger("MotionGen")


def _reencode_mp4(path):
    """Re-encode mp4v to H.264 for browser/player compatibility."""
    tmp = path + ".tmp.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", tmp],
            check=True, capture_output=True,
        )
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)


from scipy.spatial.transform import Rotation as R
import Preference

# from TaskGen.layout_gen.utils import get_object_size

from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from TaskGen.task import TaskPlan

import os
import matplotlib.pyplot as plt
os.environ["MUJOCO_GL"] = "egl"   # 启用 GPU(EGL) 渲染
os.environ.setdefault("EGL_DEVICE_ID", "0")  # 多卡时可选：指定第0号GPU
# from Preference.test_utils import get_min_body_distance
# from MotionGen.heuristic_skill.heuristic import turn_off_stove,put_with_rotation_lift, put_with_rotation_horizontal, put_with_rotation_place, \
#     pick_top_angle

from MotionGen.heuristic_skill.heuristic import move_to, pick, pick_left_edge, pick_right_edge, pick_back_edge, pick_z_90, pick_with_rotation, pick_right_edge, put, put_left_edge, put_right_edge, put_right_edge, put_on, push, pick_bottle, \
    get_push_param, turn_on_stove, move_to_waypoint, lift_up, pick_from_side, open_drawer, put_handle, pick_handle, \
    pick_horizontal, put_horizontal, get_pick_from_side_direction, close_drawer, load_markers, clear_markers, \
    pick_from_front, pick_from_back, put_with_rotation, put_back_edge, drag, throw, DEFAULT_MOVE_TO_HEIGHT, \
    PICK_SCOOP_Z_PREDROP, lay_down, turn_off_stove, fine_tune_pos, reset_pose, _load_grasp_params_for_pick, pick_top_angle

import MotionGen.heuristic_skill.heuristic as heuristic_module
from Preference.low_level.apply_low_pref import (
    enforce_max_acceleration,
    apply_low_prefs_to_plan,
    compute_avoid_waypoint,
    plan_avoid_segment,
    rotation_premove_target_override,
    rotation_premove_orn,
)
from MotionGen.gen_metadata import GenMetadataCollector
from typing import List, Tuple
skill_names = [fn.__name__ for fn in [move_to, pick, pick_left_edge, pick_right_edge, pick_back_edge, pick_z_90, pick_with_rotation, put, put_left_edge, put_right_edge, put_on, pick_bottle]] #skill for naive task planning
from MotionGen.motion_planner.ompl_planner import ompl_plan_osc_pose,OmplOSCPosePlanner,axis_angle_to_quat_wxyz,quat_mul_wxyz
from Utility.tool_func import to_sim_env_name, is_noop, get_mug_direction, env_name2obj_name, get_target_xpos
from Utility.frame import get_img_from_obs
import cv2
from MotionGen.motion_planner.example import get_body_size_3d_v1
import csv
import robosuite.utils.transform_utils as T

import h5py
import json

def _resolve_target_position(env, target):
    target = to_sim_env_name(target)
    pos = get_target_xpos(env, target)
    # Determine kind: site if body lookup would fail
    try:
        env.sim.data.get_body_xpos(target)
        return target, pos, "body"
    except ValueError:
        return target, pos, "site"


def _compute_move_to_target_xpos(env, target, picked_target=None):
    """Compute the same move-to goal used by standard move_to for all variants."""
    target, target_xpos, _ = _resolve_target_position(env, target)
    target_xpos = np.array(target_xpos, dtype=float).copy()

    target_xpos[2] += DEFAULT_MOVE_TO_HEIGHT
    if target == "basket_1_main":
        target_xpos[2] += 0.15  # adjust for better placing into basket
    if "cabinet" in target and "cabinet_top" in target:
        target_xpos[2] += 0.12  # 2026-08-31: 0.20→0.12, 去顶层抽屉的巡航压低 8cm (naive+pref 同降)
    elif "cabinet" in target and "cabinet_middle" in target:
        target_xpos[2] += 0.10
    elif "cabinet" in target and "cabinet_bottom" in target:
        target_xpos[2] += 0.00
    elif "cabinet" in target and "top_side" not in target:
        target_xpos[2] += 0.15
    if "shelf" in target:
        target_xpos[2] += 0.15
    if "wooden_tray" in target:
        target_xpos[2] += 0.05  # align with put()'s above_pos to avoid 5cm extra lift

    # When carrying an object, preserve the current grasp transform so the object,
    # rather than the gripper center, aligns with the destination.
    if picked_target is not None:
        try:
            eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
            obj_pos = env.sim.data.get_body_xpos(picked_target).copy()
            grasp_offset = eef_pos - obj_pos
            target_xpos += grasp_offset
        except Exception:
            pass

    return target, target_xpos

def _get_robot_state_vector(obs):
    return np.concatenate(
        [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
    )

def get_video_writer_and_path(path_dict,batch_id,mission_id,frame_size=(256*3,256), fps=20):
    layout_folder = path_dict["layout_folder"]

    # In perm mode, layout_folder == output_data_folder == mission dir (no nesting).
    # In legacy mode, output_data_folder is the dataset root, so we add Mission_{id}.
    if path_dict["output_data_folder"] == layout_folder:
        # perm per-mission structure: no extra nesting
        output_data_folder = layout_folder
        high_pref_folder = layout_folder
    else:
        # legacy structure
        output_data_folder = os.path.join(path_dict["output_data_folder"], f"Mission_{mission_id}")
        high_pref_folder = os.path.join(layout_folder, f"Mission_{mission_id}")

    video_folder= os.path.join(output_data_folder,"Videos")
    os.makedirs (video_folder, exist_ok=True)
    plot_folder= os.path.join(output_data_folder,"Plots")
    os.makedirs (plot_folder, exist_ok=True)

    video_path_s = os.path.join (video_folder, f"rollout_{batch_id}.mp4")
    video_path_f=os.path.join (video_folder,'fail_case', f"rollout_{batch_id}.mp4")


    os.makedirs (os.path.dirname (video_path_s), exist_ok=True)
    video_writer_s = cv2.VideoWriter (video_path_s, cv2.VideoWriter_fourcc (*"mp4v"), fps, frame_size)

    os.makedirs (os.path.dirname (video_path_f), exist_ok=True)
    video_writer_f = cv2.VideoWriter (video_path_f, cv2.VideoWriter_fourcc (*"mp4v"), fps, frame_size)

    snap_path = os.path.join(high_pref_folder, "screenshot")
    os.makedirs(snap_path, exist_ok=True)
    snap_paths={"agentview_image":snap_path,"frontview_closer_image":snap_path}
    plot_path = os.path.join (plot_folder, f"rollout_{batch_id}.png")

    hdf5_folder= os.path.join (output_data_folder,"HDF5")


    hdf5_path_s = os.path.join (hdf5_folder, f"rollout_{batch_id}.hdf5")
    hdf5_path_f = os.path.join (hdf5_folder,'fail_case', f"rollout_{batch_id}.hdf5")
    os.makedirs (os.path.dirname (hdf5_path_s), exist_ok=True)
    os.makedirs (os.path.dirname (hdf5_path_f), exist_ok=True)

    gif_folder= os.path.join (output_data_folder, "GIFs")
    os.makedirs (gif_folder, exist_ok=True)


    return video_writer_s,video_writer_f, video_path_s,video_path_f,snap_paths, plot_path,hdf5_path_s,hdf5_path_f,hdf5_folder,gif_folder

def to_positions(init_pos, delta_list):
    """
    Convert delta_list into absolute position trajectory.
    init_pos: initial absolute position (3,)
    delta_list: list of Δpos (first 3 dims used)
    """
    deltas = np.array([d[:3] for d in delta_list], dtype=float) * 0.0125

    # 累加后的相对轨迹
    rel_positions = np.cumsum(deltas, axis=0)  # (N,3)

    # 绝对轨迹 = 初始位置 + 相对轨迹
    positions = np.vstack([
        init_pos,                    # (1,3)
        init_pos + rel_positions     # (N,3)
    ])

    return positions


def get_random_ball_pos(center:np.ndarray, radius:float, z_scale=0.5):
    # 随机方向（均匀）
    v = np.random.normal(size=3)
    v /= np.linalg.norm(v)

    # 随机半径（均匀体积）
    u = np.random.uniform(0, 1)
    r = radius * (u ** (1/3))

    # 缩小 z 扰动
    v[2] *= z_scale

    return center + v * r

def save_as_hdf5(raw_obs, action, done, env, reward, hdf5_dict,reward_dict):
    """
    收集 raw_obs / action / state / rgb / depth / robot_state / low_pref_reward
    并存入 hdf5_dict，结构与 LIBERO 数据集完全一致。
    """

    obs_dict = hdf5_dict["obs"]

    # ===== 1. Proprio =====
    if "robot0_gripper_qpos" in raw_obs:
        obs_dict["gripper_states"].append(raw_obs["robot0_gripper_qpos"])

    if "robot0_joint_pos" in raw_obs:
        obs_dict["joint_states"].append(raw_obs["robot0_joint_pos"])

    if "robot0_eef_pos" in raw_obs and "robot0_eef_quat" in raw_obs:
        eef_pos = raw_obs["robot0_eef_pos"]
        eef_ori = T.quat2axisangle(raw_obs["robot0_eef_quat"])
        ee_state = np.hstack([eef_pos, eef_ori])

        obs_dict["ee_states"].append(ee_state)
        obs_dict["ee_pos"].append(eef_pos)
        obs_dict["ee_ori"].append(eef_ori)


    # ===== 2. robot_state (9D) =====
    robot_state_vec = _get_robot_state_vector(raw_obs)
    hdf5_dict["robot_states"].append(robot_state_vec)


    # ===== 3. 图像处理：RGB + flip =====
    # agentview camera
    if "agentview_image" in raw_obs:
        img = get_img_from_obs(raw_obs, key="agentview_image")
        obs_dict["agentview_rgb"].append(img)

    # eye-in-hand camera
    if "robot0_eye_in_hand_image" in raw_obs:
        eih = get_img_from_obs (raw_obs, key="robot0_eye_in_hand_image")
        obs_dict["eye_in_hand_rgb"].append(eih)
    if "frontview_image" in raw_obs:
        far_img = get_img_from_obs(raw_obs, key="frontview_image")
        obs_dict["frontview_rgb"].append(far_img)



    # ===== 4. 动作 / 状态 / reward / done =====
    hdf5_dict["actions"].append(action)
    hdf5_dict["states"].append(env.sim.get_state().flatten())

    hdf5_dict["rewards"].append(reward)
    hdf5_dict["dones"].append(1 if done else 0)


    # ===== 5. Low-level preference reward logging ===]==
    if reward_dict is not None:
        for key in hdf5_dict["low_pref_rewards"]:
            if key in reward_dict:
                hdf5_dict["low_pref_rewards"][key].append(reward_dict[key])
            else:
                hdf5_dict["low_pref_rewards"][key].append(None)




import imageio

def _compressed(grp, name, data):
    """无损压缩：shuffle + gzip-1 (fast)"""
    arr = np.stack(data, axis=0) if isinstance(data, list) else np.array(data)
    chunks = (1,) + arr.shape[1:] if arr.ndim > 1 else True
    grp.create_dataset(name, data=arr, chunks=chunks,
                       shuffle=True, compression="gzip", compression_opts=1)


ACC_FEEDBACK_MAX_ROUNDS = 3
ACC_FEEDBACK_TOLERANCE = 1.10  # accept peaks within 10% of the band edge


def execute_plan_with_acc_feedback(env, _plan, task_plan, dataset_id, path_dict,
                                   high_pref, low_pref2apply, subtask_offset, **kwargs):
    """Rejection-sampling feedback closing the gen↔eval acceleration gap.

    The plan-time clamp converts action deltas via ACTION_DELTA_TO_EE_ACC — a
    calibrated but skill/load-dependent factor, so the sim-measured peak can
    overshoot the pref's max_acc. That peak is already tracked for free during
    execution (AccRangePreference.get_reward each step); if it exceeds the band
    edge, tighten the clamp proportionally and regenerate the same batch_id
    (deterministic seed → same plan, only the clamp differs; hdf5/videos are
    overwritten). Non-acc missions pass through with a single call.
    """
    acc_prefs = [p for p in (low_pref2apply or [])
                 if getattr(p, "type", "") == "AccRangePreference"
                 and np.isfinite(getattr(p, "max_acc", np.inf))]
    common = dict(env=env, _plan=_plan, task_plan=task_plan, dataset_id=dataset_id,
                  path_dict=path_dict, high_pref=high_pref,
                  low_pref2apply=low_pref2apply, subtask_offset=subtask_offset,
                  **kwargs)
    if not acc_prefs:
        return execute_plan(**common)

    target = min(p.max_acc for p in acc_prefs)
    clamp = None  # round 1: pref's own acc_max
    prev_peak = None
    for round_idx in range(ACC_FEEDBACK_MAX_ROUNDS):
        result = execute_plan(acc_max_override=clamp, **common)
        peak = max(p.final_raw() for p in acc_prefs)
        if peak <= target * ACC_FEEDBACK_TOLERANCE:
            if round_idx:
                logger.info(f"[acc-feedback] converged round {round_idx + 1}: "
                            f"peak {peak:.4f} <= {target:.4f}x{ACC_FEEDBACK_TOLERANCE}")
            return result
        if prev_peak is not None and abs(peak - prev_peak) <= 0.02 * prev_peak:
            # A tighter clamp did not move the peak → it comes from sim
            # dynamics the xyz-delta rescale cannot touch (contact/gripper
            # transients). Further rounds only slow the demo down.
            logger.info(f"[acc-feedback] peak stuck at {peak:.4f} despite clamp "
                        f"{clamp:.4f} — residual sim-dynamics floor, stopping")
            return result
        prev_peak = peak
        prev = clamp if clamp is not None else target
        clamp = prev * (target / peak)
        logger.info(f"[acc-feedback] round {round_idx + 1}: peak {peak:.4f} > "
                    f"target {target:.4f} — regenerating with clamp {clamp:.4f}")
    logger.warning(f"[acc-feedback] still over after {ACC_FEEDBACK_MAX_ROUNDS} rounds: "
                   f"peak {peak:.4f} > {target:.4f} (last demo kept)")
    return result


def execute_plan(env,_plan:List[Tuple[str,str]],task_plan:TaskPlan, dataset_id,path_dict,high_pref,low_pref2apply,subtask_offset,generate_evaluation_data=False,batch_id=None,damping_level=1,skill_debug=False,n_save_video=None,save_eval_hdf5=False,save_gif=False,acc_max_override=None):
    """Given a taskplan and environment, execute the plan and save video and reward plots."""
    if batch_id is not None:
        random.seed(batch_id)
        np.random.seed(batch_id)
        env.seed(batch_id)
    heuristic_module.DAMPING_LEVEL = damping_level
    actions = []

    _self_adjustable_skills=["pick_bottle"]
    _draw_plot=False
    _handle_direction=None
    _OBJ_TO_APPROACH = None
    _APPROACH_DISTANCE = None
    _OBJ_TO_AVOID = None
    last_step_actions=None
    _AVOID_MIN_THRESHOLD = None
    _AVOID_MAX_THRESHOLD = None
    _action_cnt={}
    #Zhixu: Use motion planner or move_to_waypoint
    use_motion_planner = False
    if n_save_video is None:
        _save_all_video = True
        n_save_video = 10
    else:
        _save_all_video = False
    gif_frames = []
    GIF_STRIDE = 3  # 每 5 帧取一帧，防止 gif 太大（可调）
    speed_range = None
    _hand_occupied = False
    _picked_target = None
    _acc_max = None
    _move_to_height_offset=None
    _path_height_offset = None
    _pick_horizontal_rot_offset=None
    _pick_with_rotation_angle=None
    _pending_move_to_straight_goal = None
    _prefer_straight_path = False
    _prefer_rotation_while_moving = False
    # orig_step_num=0
    # acc_step_num=0

    NO_DUMMY_ACTION=True
    _Done=0
    _speed_scale=1.0

    # ==========================================
    # Skill debug: capture screenshots during non-move_to skills
    # ==========================================
    _skill_debug_dir = None
    _SKILL_DEBUG_STRIDE = 5  # capture every N frames
    if skill_debug:
        _skill_debug_dir = os.path.join(path_dict["layout_folder"], "Screenshots")
        os.makedirs(_skill_debug_dir, exist_ok=True)
        print(f"  [Skill Debug] Screenshots will be saved to {_skill_debug_dir}")

    # 记录初始 eef pose，用于 reset_pose
    _init_eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
    _init_eef_quat = env.sim.data.body_xquat[env.sim.model.body_name2id("gripper0_eef")]
    _init_eef_quat_xyzw = np.array([_init_eef_quat[1], _init_eef_quat[2], _init_eef_quat[3], _init_eef_quat[0]])
    _init_eef_orn = R.from_quat(_init_eef_quat_xyzw).as_euler("XYZ")

    # ==========================================
    # ↓↓↓↓↓ HDF5 data storage ↓↓↓↓↓
    # ==========================================

    hdf5_dict = {
        "frames": [],
        "depths": [],
        "obs":{
            "gripper_states": [],
            "agentview_rgb": [],
            "eye_in_hand_rgb": [],
            "frontview_rgb": [],
            "ee_ori": [],
            "ee_pos": [],
            "ee_states": [],
            "joint_states": [],
            },
        "robot_states": [],
        "actions": [],
        "states": [],
        "rewards": [],
        "dones": [],
        "low_pref_rewards": {
            pref.preference_type: [] for pref in low_pref2apply
        }
    }
    model_xml = env.sim.model.get_xml ()

    # ==========================================
    # ↑↑↑↑↑ HDF5 data storage ↑↑↑↑↑
    # ==========================================


    if not _plan:
        print("[Motion Gen] Warning: Task plan is empty.")
        raise ValueError


    # max_mp_tries=2
    video_writer_s,video_writer_f, video_path_s,video_path_f,snap_paths, plot_path,hdf5_path_s,hdf5_path_f,hdf5_folder,gif_folder = get_video_writer_and_path(path_dict,
    mission_id=dataset_id,
    batch_id=batch_id,
    )
    if _save_all_video or batch_id<=n_save_video:
        pass
    else:
        video_writer_s=None
        video_writer_f=None


    # Setup for evaluation data generation
    eval_image_dir = None
    log_path = None
    if generate_evaluation_data:
        eval_image_dir = path_dict["evaluation_image_folder"]
        eval_image_dir=os.path.join(eval_image_dir)
        os.makedirs(eval_image_dir, exist_ok=True)
        eval_image_dir= os.path.join(eval_image_dir, f"rollout_{batch_id}")
        os.makedirs(eval_image_dir, exist_ok=True)
        log_path = os.path.join(eval_image_dir,"evaluation_log.csv")
        
        # Write header to log file if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["main_goal", "sub_task", "image_path", "vlm_prediction", "human_label"])

    # Plan-level subtasks for gen metadata: the non-move_to entries BEFORE
    # avoid/approach rewriting (those keep the original task_plan skills).
    _meta_plan_entries = [s for s in _plan if s[0] != "move_to"]
    # Same task context the eval side injects (avoid transport gating) — keeps
    # gen-time final_raw consistent with calibrate/PolicyEval replays.
    _task_objects = [t for _, t in _meta_plan_entries if t]
    for _p in low_pref2apply:
        if hasattr(_p, "set_task_context"):
            _p.set_task_context(_task_objects)
    # Apply all low-level preferences (priority-sorted) — plan rewrites +
    # gen knobs live in Preference/low_level/apply_low_pref.py.
    _plan, _low_cfg = apply_low_prefs_to_plan(low_pref2apply, _plan, env)
    if acc_max_override is not None and _low_cfg.acc_max is not None:
        # Tightened clamp from execute_plan_with_acc_feedback; mutate the cfg so
        # gen metadata records the knob actually enforced this round.
        _low_cfg.acc_max = acc_max_override
    _meta_collector = GenMetadataCollector(
        task_plan=_meta_plan_entries,
        naive_plan=list(getattr(task_plan, "naive_plans", None) or []),
        high_pref_desc=(getattr(high_pref, "pref_description", None)
                        if high_pref is not None else None),
        low_pref_entries=[{"type": p.preference_type,
                           "description": p.pref_description}
                          for p in low_pref2apply],
        gen_cfg=_low_cfg,
        env=env,
        mission_id=str(dataset_id) if dataset_id is not None else None,
    )
    _OBJ_TO_APPROACH = _low_cfg.obj_to_approach
    _APPROACH_DISTANCE = _low_cfg.approach_distance
    _OBJ_TO_AVOID = _low_cfg.obj_to_avoid
    _AVOID_MIN_THRESHOLD = _low_cfg.avoid_min_threshold
    _AVOID_MAX_THRESHOLD = _low_cfg.avoid_max_threshold
    _speed_scale = _low_cfg.speed_scale
    _acc_max = _low_cfg.acc_max
    _path_height_offset = _low_cfg.path_height_offset
    _prefer_straight_path = _low_cfg.prefer_straight_path
    _prefer_rotation_while_moving = _low_cfg.prefer_rotation_while_moving

    logger.debug(f"Starting motion generation for {high_pref}")
    pref_rewards = {pref.preference_type: [] for pref in low_pref2apply}  # per-pref reward record
    gripper_state = -1.0  # open at the beginning


    # 初始化执行队列（先进先出）
    step_queue = deque (_plan)
    step_idx = 0
    offset_key_cnt = {}
    _prev_skill = None  # 上一个 skill，用于上下文感知的 height offset (e.g. lay_down → move_to 降低 arch)


    env.reset ()
    # Paired init layout (2026-09-01): naive demos normally land on the same
    # sampled layout as the pref demos because both reseed with batch_id, but a
    # plan change (1-8 --force_naive_plan: pick vs pick_with_rotation) consumes
    # RNG differently and shifts an object by a few cm. With INIT_STATE_POOL set
    # (step1_data_gen reads it from the pref hdf5), replay attempt `batch_id`'s
    # pref init state so the two rows of the comparison video — and the anchors —
    # differ only in the preference, never in the layout.
    _pool = globals().get("INIT_STATE_POOL")
    # Attempt b replays pref init #b (rollout_<b>.hdf5 <-> pref demo_b). A failed
    # attempt is NOT retried on the same init: the sim is deterministic, so it
    # would fail again; the pairing step downstream keeps only the inits where
    # the naive rollout succeeded. PAIR_INDEX is a debug override.
    _pidx = globals().get("PAIR_INDEX")
    if _pidx is None:
        _pidx = batch_id
    if _pool is not None and _pidx is not None and len(_pool) > 0:
        _st = np.asarray(_pool[_pidx % len(_pool)], dtype=float)
        try:
            # env.set_init_state (wrapper) = set_state + forward + re-derive
            # observables; a bare sim.set_state_from_flattened leaves the wrapper's
            # cached obs/success state stale and the layout snaps back.
            env.set_init_state(_st)
            print(f"[paired-init] batch {batch_id} <- pref init_state "
                  f"#{_pidx % len(_pool)}", flush=True)
        except Exception as e:
            print(f"[paired-init] FAILED on batch {batch_id}: {e}", flush=True)
    # Stateful low-pref metrics (AccRange running peak, Avoid min-dist) live on
    # instances reused across batch attempts — clear them or episode N>1 rewards
    # and final_raw inherit the previous episode's extremes.
    for _pref in (low_pref2apply or []):
        if hasattr(_pref, "reset"):
            _pref.reset(env)
    load_markers(env=env)
    num_steps_wait=5

    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    for _ in range(num_steps_wait):
        raw_obs, reward, done, info =env.step (LIBERO_DUMMY_ACTION)  # wait for a few steps to let the environment stabilize after reset

        if _save_all_video or batch_id<=n_save_video:
            frame = get_img_from_obs (raw_obs, key="agentview_image")
            eih_frame = get_img_from_obs (raw_obs, key="robot0_eye_in_hand_image")
            far_frame = get_img_from_obs (raw_obs, key="frontview_image")

            # Concatenate agentview and eye-in-hand images side by side
            TARGET_H = 256
            TARGET_W = 256

            def resize_fixed (img):
                return cv2.resize (
                    img,
                    (TARGET_W, TARGET_H),
                    interpolation=cv2.INTER_LINEAR
                )

            frame_resized = resize_fixed (frame)
            eih_resized = resize_fixed (eih_frame)
            far_resized = resize_fixed (far_frame)

            combined = np.concatenate (
                [frame_resized, eih_resized, far_resized],
                axis=1
            )

            video_writer_s.write (
                cv2.cvtColor (combined, cv2.COLOR_RGB2BGR)
            )
            video_writer_f.write (
                cv2.cvtColor (combined, cv2.COLOR_RGB2BGR)
            )


    while step_queue:
        step_idx += 1

        step= step_queue.popleft ()  # 从队首取出当前要执行的任务

        skill, target = step

        # 记录当前 skill 执行前的 eef pose，用于 recovery
        _pre_skill_eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
        _pre_skill_eef_quat = env.sim.data.body_xquat[env.sim.model.body_name2id("gripper0_eef")]
        _pre_skill_eef_quat_xyzw = np.array([_pre_skill_eef_quat[1], _pre_skill_eef_quat[2], _pre_skill_eef_quat[3], _pre_skill_eef_quat[0]])
        _pre_skill_eef_orn = R.from_quat(_pre_skill_eef_quat_xyzw).as_euler("XYZ")

        if skill not in _action_cnt.keys():
            _action_cnt[skill]=0
        _action_cnt[skill]+=1
        step_actions = []
        _n_pre_executed = 0  # number of actions already executed inside skill dispatch

        offset_key = f"{skill}:{env_name2obj_name(target)}"
        offset_key_all = f"{skill}:_all"

        if offset_key in subtask_offset:
            if offset_key not in offset_key_cnt.keys ():
                offset_key_cnt [offset_key] = 0
            else:
                offset_key_cnt [offset_key] += 1

            _OFFSET = subtask_offset [offset_key] [
                offset_key_cnt [offset_key] % len (subtask_offset [offset_key])
            ]
        elif offset_key_all in subtask_offset:
            _OFFSET = subtask_offset [offset_key_all] if isinstance(subtask_offset [offset_key_all], list) else subtask_offset [offset_key_all]
        else:
            _OFFSET = None
        if skill=="move_to":
            skill_para_key=f"{skill}_height:{env_name2obj_name(target)}"
            if skill_para_key in subtask_offset:
                _move_to_height_offset=subtask_offset [skill_para_key]
            # lay_down 之后的 move_to: 降低 safe_height arch 让 gripper 不出 camera FOV
            if _prev_skill == "lay_down" and _move_to_height_offset is None:
                _move_to_height_offset = -0.15
        if skill=="pick_horizontal":
            skill_para_key = f"{skill}_rot:{env_name2obj_name (target)}"
            if skill_para_key in subtask_offset:
                 _pick_horizontal_rot_offset= subtask_offset [skill_para_key]
        if skill=="pick_with_rotation":
            skill_para_key_deg = f"{skill}_deg:{env_name2obj_name(target)}"
            skill_para_key_deg_all = f"{skill}_deg:_all"
            skill_para_key_rad = f"{skill}_rot:{env_name2obj_name(target)}"
            skill_para_key_rad_all = f"{skill}_rot:_all"
            _pick_with_rotation_angle = None
            if skill_para_key_deg in subtask_offset:
                _pick_with_rotation_angle = np.deg2rad(subtask_offset[skill_para_key_deg])
            elif skill_para_key_deg_all in subtask_offset:
                _pick_with_rotation_angle = np.deg2rad(subtask_offset[skill_para_key_deg_all])
            elif skill_para_key_rad in subtask_offset:
                _pick_with_rotation_angle = subtask_offset[skill_para_key_rad]
            elif skill_para_key_rad_all in subtask_offset:
                _pick_with_rotation_angle = subtask_offset[skill_para_key_rad_all]


        print(f"  \033[1;36m▶ Step {step_idx}\033[0m  \033[1;33m{skill}\033[0m → \033[1m{target}\033[0m")
        logger.debug(f"Step {step_idx}: {step}, offset={_OFFSET}")

        if skill == "move_to":
            target, target_xpos, target_kind = _resolve_target_position(env, target)
            # 检查 subtask_offset 中是否有 skip 标记
            next_skill = None
            next_target = None
            if step_queue:
                next_skill, next_target = step_queue[0]
                skip_key = f"skip_move_to_before:{next_skill}:{env_name2obj_name(next_target)}"
                if skip_key in subtask_offset:
                    logger.debug(f"Skipping move_to before {next_skill} (yaml skip flag)")
                    continue
                if next_skill == "turn_on_stove":
                    logger.info(f"Skipping move_to before {next_skill}; stove skill handles its own approach")
                    continue
                # open_drawer 有 grasp_params 时自动跳过前置 move_to
                if next_skill == "open_drawer":
                    drawer_target = to_sim_env_name(next_target)
                    if _load_grasp_params_for_pick(drawer_target, skill_type="open_drawer") is not None:
                        logger.info(f"Skipping move_to before open_drawer (grasp_params found for {drawer_target})")
                        continue
                # pick_from_side 有 grasp_params 时自动跳过前置 move_to
                if next_skill == "pick_from_side":
                    side_target = to_sim_env_name(next_target)
                    for _st in ["pick_from_side_left", "pick_from_side_right", "pick_from_side"]:
                        if _load_grasp_params_for_pick(side_target, skill_type=_st) is not None:
                            logger.info(f"Skipping move_to before pick_from_side (grasp_params found for {side_target})")
                            break
                    else:
                        _st = None
                    if _st is not None:
                        continue
                # pick_horizontal 有 grasp_params 时自动跳过前置 move_to
                if next_skill == "pick_horizontal":
                    h_target = to_sim_env_name(next_target)
                    if _load_grasp_params_for_pick(h_target, skill_type="pick_horizontal") is not None:
                        logger.info(f"Skipping move_to before pick_horizontal (grasp_params found for {h_target})")
                        continue
                # pick_from_front 有 grasp_params 时自动跳过前置 move_to
                if next_skill == "pick_from_front":
                    f_target = to_sim_env_name(next_target)
                    if _load_grasp_params_for_pick(f_target, skill_type="pick_from_front") is not None:
                        logger.info(f"Skipping move_to before pick_from_front (grasp_params found for {f_target})")
                        continue
            actual_move_to_target = target
            if _prefer_rotation_while_moving:
                _rot_override = rotation_premove_target_override(next_skill, next_target)
                if _rot_override is not None:
                    actual_move_to_target = _rot_override
                    logger.info(
                        f"[rotation_pref] move_to display target={target}, actual target={actual_move_to_target}"
                    )
            target, target_xpos = _compute_move_to_target_xpos(
                env, actual_move_to_target, picked_target=_picked_target
            )

            gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
            logger.debug(f"move_to from {gripper_pos} to {target_xpos} ({target_kind})")
            eef_id = env.sim.model.body_name2id ("gripper0_eef")
            eef_quat = env.sim.data.body_xquat [eef_id]
            eef_quat_xyzw = np.array ([eef_quat [1], eef_quat [2], eef_quat [3], eef_quat [0]])
            target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")
            if _prefer_rotation_while_moving:
                _premove_orn = rotation_premove_orn(env, next_skill, next_target, subtask_offset)
                if _premove_orn is not None:
                    target_orn = _premove_orn
            if (
                _prefer_straight_path
                and _picked_target is not None
                and next_skill in {"put", "put_left_edge", "put_right_edge", "put_back_edge", "put_horizontal", "lay_down"}
                and target == to_sim_env_name(next_target)
                and any(k in target for k in ("tray", "stove", "plate"))
            ):
                logger.info(
                    f"[straight_path] Using direct move_to_waypoint above tray for {target}"
                )
                step_actions = move_to_waypoint(
                    env,
                    target_pos=target_xpos,
                    target_orn=target_orn,
                    speed=0.85e-2 * _speed_scale,
                    gripper_state=gripper_state,
                    offset=_OFFSET,
                    path_height_offset=_path_height_offset,
                )
                _move_to_height_offset = None
            else:
                logger.info(f"[move_to] gripper_pos={np.round(gripper_pos, 4)} target_xpos={np.round(target_xpos, 4)}")
                logger.info(f"[move_to] gripper_orn={np.round(np.degrees(target_orn), 1)} picked={_picked_target}")
                effective_move_to_height_offset = _path_height_offset
                if _move_to_height_offset is not None:
                    effective_move_to_height_offset = (
                        (_path_height_offset or 0.0) + _move_to_height_offset
                    )
                logger.debug(f"Executing step: {step}, safe height offset={effective_move_to_height_offset}")
                logger.debug(f"move_to safe height offset={effective_move_to_height_offset}")
                step_actions = move_to (env, target_pos=target_xpos, target_orn=target_orn, speed=0.85e-2 * _speed_scale,
                                        gripper_state=gripper_state, offset=_OFFSET, safe_height_offset=effective_move_to_height_offset)
                _move_to_height_offset=None

        elif skill == "lift_up":
            step_actions = lift_up (env, gripper_state=gripper_state)

        elif skill == "move_to_approach":

            target= to_sim_env_name (target)
            logger.debug(f"[Approach] move_to {target}, approaching {_OBJ_TO_APPROACH}")

            if _OBJ_TO_APPROACH=="ground_1":
                gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()

                # 目标物体位置
                target_pos = get_target_xpos (env, target)
                # ================================
                # Ground approach: 贴地平移
                # ================================
                z_offset = _OFFSET[2] if _OFFSET is not None else 0.03

                approach_pos = gripper_pos.copy ()
                approach_pos [0] = 0.5 * (gripper_pos [0] + target_pos [0])
                approach_pos [1] = 0.5 * (gripper_pos [1] + target_pos [1])
                approach_pos [2] = z_offset
                pos2approach=approach_pos
                print(
                    f"  [Ground Approach] "
                    f"EEF XY {gripper_pos [:2]} → Target XY {target_pos [:2]}, "
                    f"z={z_offset:.3f}"
                )
            else:
                approach_pos = env.sim.data.get_body_xpos (_OBJ_TO_APPROACH).copy () 
                _size = get_body_size_3d_v1 (env, _OBJ_TO_APPROACH)
                pos2approach= get_random_ball_pos(approach_pos, radius=max(0.03,max(_size['size'])/2))
                pos2approach[2]+=0.1  # lift a bit

            eef_id = env.sim.model.body_name2id ("gripper0_eef")
            eef_quat = env.sim.data.body_xquat [eef_id]
            eef_quat_xyzw = np.array ([eef_quat [1], eef_quat [2], eef_quat [3], eef_quat [0]])
            target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")

            step_actions = move_to_waypoint(
                env,
                pos2approach,
                target_orn,
                gripper_state=gripper_state,
                path_height_offset=_path_height_offset,
            )
            _pending_move_to_straight_goal = None
            step_queue.appendleft (("move_to_straight", target))
            if _OBJ_TO_APPROACH !="ground_1":
                step_queue.appendleft(("lift_up", None))  # First lift up to avoid collision

        elif skill == "move_to_away":
            target, target_xpos = _compute_move_to_target_xpos(
                env, target, picked_target=_picked_target
            )
            logger.debug(f"[Away] move_to {target}, avoiding {_OBJ_TO_AVOID}")
            # OMPL keep-away transport (clearance = _AVOID_MIN_THRESHOLD);
            # falls back to the mirrored single waypoint when no path is found.
            _ompl_res = plan_avoid_segment(
                env, target_xpos, _OBJ_TO_AVOID, _AVOID_MIN_THRESHOLD,
                speed=0.85e-2 * _speed_scale, gripper_state=gripper_state,
                offset=_OFFSET, picked_target=_picked_target,
            )
            if _ompl_res is not None:
                step_actions, waypoint = _ompl_res
                _meta_collector.record_avoid_waypoint(waypoint)
            else:
                waypoint = compute_avoid_waypoint(
                    env, target_xpos, _OBJ_TO_AVOID,
                    avoid_range=(_AVOID_MIN_THRESHOLD, _AVOID_MAX_THRESHOLD),
                    offset=_OFFSET, picked_target=_picked_target,
                )
                _meta_collector.record_avoid_waypoint(waypoint)
                eef_id = env.sim.model.body_name2id ("gripper0_eef")
                eef_quat = env.sim.data.body_xquat [eef_id]
                eef_quat_xyzw = np.array ([eef_quat [1], eef_quat [2], eef_quat [3], eef_quat [0]])
                target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")
                step_actions = move_to_waypoint(
                    env,
                    target_pos=waypoint,
                    target_orn=target_orn,
                    speed=0.85e-2 * _speed_scale,
                    gripper_state=gripper_state,
                    path_height_offset=_path_height_offset,
                )

            _pending_move_to_straight_goal = (target, target_xpos.copy())
            step_queue.appendleft (("move_to_straight", target))
            # step_queue.appendleft(("lift_up", None))  # First lift up to avoid collision

        elif skill == "move_to_straight":
            target, _target_pos_raw, target_kind = _resolve_target_position(env, target)
            if (
                _pending_move_to_straight_goal is not None
                and _pending_move_to_straight_goal[0] == target
            ):
                target_xpos = _pending_move_to_straight_goal[1].copy()
            else:
                _, target_xpos = _compute_move_to_target_xpos(
                    env, target, picked_target=_picked_target
                )
            gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
            logger.debug(f"move_to straight from {gripper_pos} to {target_xpos} ({target_kind}) + {_OFFSET if _OFFSET is not None else 0}")
            eef_id = env.sim.model.body_name2id ("gripper0_eef")
            eef_quat = env.sim.data.body_xquat [eef_id]
            eef_quat_xyzw = np.array ([eef_quat [1], eef_quat [2], eef_quat [3], eef_quat [0]])
            target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")

            step_actions = move_to_waypoint(
                env,
                target_pos=target_xpos,
                target_orn=target_orn,
                speed=0.85e-2 * _speed_scale,
                gripper_state=gripper_state,
                offset=_OFFSET,
                path_height_offset=_path_height_offset,
            )
            _pending_move_to_straight_goal = None

        elif skill == "pick":
            target= to_sim_env_name (target)
            step_actions = pick (env, target_name=target,offset=_OFFSET,speed_scale=_speed_scale,yaml_offset=True)
            gripper_state = 1.0  # close after pick
            _hand_occupied=True
            _picked_target = target
        elif skill == "pick_left_edge":
            target = to_sim_env_name (target)
            step_actions = pick_left_edge (env, target_name=target, offset=_OFFSET,speed_scale=_speed_scale)
            gripper_state = 1.0  # close after pick
            _hand_occupied = True
            _picked_target = target
        elif skill == "pick_right_edge":
            target = to_sim_env_name (target)
            step_actions = pick_right_edge (env, target_name=target, offset=_OFFSET,speed_scale=_speed_scale)
            gripper_state = 1.0
            _hand_occupied = True
            _picked_target = target

        elif skill == "pick_bottle":
            target= to_sim_env_name (target)
            step_actions = pick_bottle(env, target,offset=_OFFSET,scale=_speed_scale)
            actions.extend(step_actions)
            gripper_state = 1.0
            _hand_occupied = True
            _picked_target = target

        elif skill == "pick_top_angle":
            target = to_sim_env_name(target)
            step_actions = pick_top_angle(env, target_name=target, offset=_OFFSET, speed_scale=_speed_scale)
            gripper_state = 1.0
            _hand_occupied = True
            _picked_target = target

        elif skill == "pick_from_side":
            _side_param = high_pref.action_parameters[skill][0]
            direction = get_pick_from_side_direction(_side_param.get('direction', _side_param.get('side')))
            target= to_sim_env_name (target)
            step_actions = pick_from_side (env, target_name=target,direction=direction,offset=_OFFSET, speed_scale=_speed_scale)
            gripper_state = 1.0  # close after pick
            _hand_occupied = True
            _picked_target = target

        elif skill == "pick_handle":
            target= to_sim_env_name (target)
            _handle_direction=get_mug_direction(target)
            step_actions = pick_handle (env, target_name=target,handle_direction=_handle_direction,offset=_OFFSET)
            gripper_state = 1.0  # close after pick
            _hand_occupied = True
            _picked_target = target

        elif skill == "pick_horizontal":
            target = to_sim_env_name(target)
            step_actions = pick_horizontal(env, target_name=target,offset=_OFFSET,rotate_angle=_pick_horizontal_rot_offset)
            gripper_state = 1.0  # close after pick
            _hand_occupied = True
            _picked_target = target
            _pick_horizontal_rot_offset=None
            _move_to_height_offset = 0.10  # 抬高后续 move_to 的 safe_height，避免搬运时撞微波炉

        elif skill == "pick_back_edge":
            target = to_sim_env_name(target)
            step_actions = pick_back_edge(env, target_name=target, offset=_OFFSET, speed_scale=_speed_scale)
            gripper_state = 1.0
            _hand_occupied = True
            _picked_object_name = target
            _picked_bowl_offset = None  # 将在 actions 执行后读取

        elif skill == "put_back_edge":
            target = to_sim_env_name(target)
            step_actions = put_back_edge(env, target_name=target, offset=_OFFSET, speed_scale=_speed_scale)
            gripper_state = -1.0
            _hand_occupied = False

        # elif skill == "pick_scoop":  # DEPRECATED
        # elif skill == "put_scoop":   # DEPRECATED
        # elif skill == "pick_rim":    # DEPRECATED
        # elif skill == "put_rim":     # DEPRECATED
        # elif skill == "pick_both_edge":  # DEPRECATED
        # elif skill == "put_both_edge":   # DEPRECATED

        elif skill == "lay_down":
            target = to_sim_env_name(target)
            step_actions = lay_down(env, target_name=target, offset=_OFFSET, speed_scale=_speed_scale)
            gripper_state = -1.0
            _hand_occupied = False

        elif skill == "put":
            target = to_sim_env_name (target)
            obj_offset = np.zeros(3)
            if "bowl" in target:
                obj_offset[2] += 0.07
            if _OFFSET is None:
                skill_offset = obj_offset
            else:
                skill_offset = [_OFFSET[i] + obj_offset[i] for i in range(3)]

            step_actions = put(env, target_name=target,offset=skill_offset, speed_scale=_speed_scale, held_obj=_picked_target)

            gripper_state = -1.0  # open after put
            _hand_occupied = False
            _picked_target = None
        elif skill == "put_left_edge":
            target= to_sim_env_name (target)
            step_actions = put_left_edge (env, target_name=target,offset=_OFFSET,speed_scale=_speed_scale, held_obj=_picked_target)

            gripper_state = -1.0  # open after put
            _hand_occupied = False
            _picked_target = None
        elif skill == "put_right_edge":
            target= to_sim_env_name (target)
            step_actions = put_right_edge (env, target_name=target,offset=_OFFSET,speed_scale=_speed_scale, held_obj=_picked_target)

            gripper_state = -1.0  # open after put
            _hand_occupied = False
            _picked_target = None
        elif skill == "put_handle":
            target= to_sim_env_name (target)
            step_actions = put_handle (env, target_name=target,handle_direction=_handle_direction,offset=_OFFSET)
            gripper_state = -1.0  # open after put
            _hand_occupied = False
        elif skill == "put_horizontal":
            target = to_sim_env_name(target)
            step_actions = put_horizontal(env, target_name=target,offset=_OFFSET)
            gripper_state = -1.0
            _hand_occupied = False

        elif skill == "pick_from_front":
            target = to_sim_env_name(target)
            step_actions = pick_from_front(env, target_name=target, offset=_OFFSET, speed_scale=_speed_scale)
            gripper_state = 1.0
            _hand_occupied = True

        elif skill == "pick_z_90":
            target = to_sim_env_name(target)
            step_actions = pick_z_90(env, target_name=target, offset=_OFFSET)
            gripper_state = 1.0
            _hand_occupied = True

        elif skill == "pick_with_rotation":
            target = to_sim_env_name(target)
            step_actions = pick_with_rotation(
                env,
                target_name=target,
                offset=_OFFSET,
                speed_scale=_speed_scale,
                rotate_angle=_pick_with_rotation_angle,
            )
            _pick_with_rotation_angle = None
            gripper_state = 1.0
            _hand_occupied = True
            _picked_object_name = target  # 记住抓的是哪个物体
            _picked_bowl_offset = None  # 将在 actions 执行后读取

        elif skill == "put_with_rotation":
            target = to_sim_env_name(target)
            step_actions = put_with_rotation(
                env,
                target_name=target,
                offset=_OFFSET,
                speed_scale=_speed_scale,
            )
            gripper_state = -1.0
            _hand_occupied = False

        elif skill == "drag":
            direction, distance = get_push_param(high_pref.action_parameters['drag'][_action_cnt.get('drag', 1)-1], env=env)
            target = to_sim_env_name(target)
            step_actions = drag(env, target_name=target, direction=direction, distance=distance, offset=_OFFSET)
            gripper_state = -1.0
            _hand_occupied = False

        elif skill == "throw":
            target = to_sim_env_name(target)
            obj_offset = np.zeros(3)
            if "bowl" in target:
                obj_offset[2] += 0.07
            if _OFFSET is None:
                skill_offset = obj_offset
            else:
                skill_offset = [_OFFSET[i] + obj_offset[i] for i in range(3)]
            step_actions = throw(env, target_name=target, offset=skill_offset)
            gripper_state = -1.0
            _hand_occupied = False

        elif skill == "push":
            direction,distance=get_push_param(high_pref.action_parameters['push'][_action_cnt['push']-1], env=env)
            print(f"  push direction: {direction}, distance: {distance}")
            target= to_sim_env_name (target)
            step_actions = push(env, target_name=target, direction=direction, distance=distance,offset=_OFFSET)

        elif skill == "turn_on_stove":
            target = to_sim_env_name(target)
            if not target.endswith("_button"):
                if target.endswith("_main"):
                    target = target[:-5] + "_button"
                else:
                    target = target + "_button"
            idx = env.sim.model.body_name2id (target)
            target_xpos = env.sim.data.body_xpos [idx]
            logger.debug(f"turn on stove target pos: {target_xpos}")
            step_actions = turn_on_stove (env, target_name=target)
            gripper_state = -1.0  # open after put
            _hand_occupied = False

        elif skill == "turn_off_stove":
            target = target + "_button"
            idx = env.sim.model.body_name2id (target)
            target_xpos = env.sim.data.body_xpos [idx]
            print ("turn off stove target pos: {}".format (target_xpos))
            step_actions = turn_off_stove (env, target_name=target, offset=_OFFSET)
            gripper_state = -1.0
            _hand_occupied = False

        elif skill=="open_drawer":
            target= to_sim_env_name (target)
            step_actions=open_drawer (env, target_name=target, offset=_OFFSET)


        elif skill=="close_drawer":
            target= to_sim_env_name (target)
            step_actions=close_drawer (env, target_name=target, offset=_OFFSET)

        else:
            raise ValueError (f"Unknown skill: {skill}")

        if _acc_max is not None:
            step_actions = enforce_max_acceleration (
                step_actions, _acc_max=_acc_max,
                last_step_actions=last_step_actions,skill=skill,
                _self_adjustable_skills=_self_adjustable_skills,
            )

        if skill == "pick_handle":
            rot_z = np.asarray(step_actions)[:, 5] * 0.125
            sum_rot_z = np.sum (rot_z)
            print(f"  total rotation: {sum_rot_z}")
        #Store the step actions
        actions.extend (step_actions)
        last_skill=skill
        last_step_actions=step_actions

        """
        Execute step action
        """

        frame = None # Ensure frame is available outside the loop
        
        # Calculate indices to save to achieve approximately 'target_frame_count' frames per subtask
        # This ensures we get key frames (start/end) and evenly spaced intermediates regardless of duration
        target_frame_count = 5
        indices_to_save = set()
        if len(step_actions) > 0:
            indices_to_save = set(np.linspace(0, len(step_actions) - 1, num=min(len(step_actions), target_frame_count), dtype=int))

        pre_action=None
        _executed_count = 0
        _skipped_count = 0
        for action_idx, action in enumerate (step_actions):
            if action_idx < _n_pre_executed:
                # These actions were already executed inside the skill dispatch
                # (e.g., pick_back_edge retreat). Skip execution but keep in HDF5.
                continue
            if NO_DUMMY_ACTION:
                if is_noop(action,pre_action):
                    pre_action=action
                    logger.debug(f"Skipping Dummy action at index {action_idx}:{action} of skill {skill}")
                    continue
                else:
                    pre_action=action
            _executed_count += 1
            raw_obs, reward, done, info  = env.step (action)


            if done and not _Done:
                _Done=1
                print(f"[Debug] done=True at action_idx={action_idx}/{len(step_actions)} of skill {skill}")
            elif not done:
                pass

            # Skill debug: save screenshots for non-move_to skills
            if _skill_debug_dir and skill not in ("move_to", "move_to_straight", "move_to_away", "move_to_approach", "lift_up") and _executed_count % _SKILL_DEBUG_STRIDE == 0:
                dbg_frame = get_img_from_obs(raw_obs, key="agentview_image")
                dbg_eih = get_img_from_obs(raw_obs, key="robot0_eye_in_hand_image")
                dbg_combined = np.concatenate([
                    cv2.resize(dbg_frame, (256, 256), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(dbg_eih, (256, 256), interpolation=cv2.INTER_LINEAR),
                ], axis=1)
                dbg_path = os.path.join(_skill_debug_dir, f"step{step_idx}_{skill}_{action_idx:04d}.png")
                cv2.imwrite(dbg_path, cv2.cvtColor(dbg_combined, cv2.COLOR_RGB2BGR))

            if _save_all_video or batch_id<=n_save_video:
                frame = get_img_from_obs(raw_obs, key="agentview_image")
                eih_frame= get_img_from_obs(raw_obs, key="robot0_eye_in_hand_image")
                far_frame= get_img_from_obs(raw_obs, key="frontview_image")


                #Concatenate agentview and eye-in-hand images side by side
                TARGET_H = 256
                TARGET_W = 256

                def resize_fixed (img):
                    return cv2.resize (
                        img,
                        (TARGET_W, TARGET_H),
                        interpolation=cv2.INTER_LINEAR
                    )

                frame_resized = resize_fixed (frame)
                eih_resized = resize_fixed (eih_frame)
                far_resized = resize_fixed (far_frame)

                combined = np.concatenate (
                    [frame_resized, eih_resized, far_resized],
                    axis=1
                )

                video_writer_s.write (
                    cv2.cvtColor (combined, cv2.COLOR_RGB2BGR)
                )
                video_writer_f.write (
                    cv2.cvtColor (combined, cv2.COLOR_RGB2BGR)
                )

                if step_idx <= 1:
                    for key, path in snap_paths.items():
                        path = os.path.join(path, key + f"_{step_idx}.png")  # key : agentview_image
                        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        # plt.figure (figsize=(4, 4))
                        # plt.imshow (frame)  # frame = RGB
                        # plt.axis ("off")
                        # plt.show ()

                if save_gif:
                    if action_idx % GIF_STRIDE == 0:
                        gif_frames.append (combined.copy ())  # RGB, uint8



            pref_reward={}
            if low_pref2apply is not None:
                for _pref in low_pref2apply:
                    pref_reward[_pref.preference_type] = _pref.get_reward (env)
                    pref_rewards [_pref.preference_type].append (reward)

            save_as_hdf5 (raw_obs=raw_obs, action=action, done=done, reward=reward, env=env, hdf5_dict=hdf5_dict,
                          reward_dict=pref_reward)

            # Early stop for approach skill
            if skill == "move_to_approach":
                gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
                target_pos = get_target_xpos (env, to_sim_env_name (target))
                # print(f"Distance to target during approach: {np.linalg.norm (gripper_pos - target_pos):.4f}")
                if np.linalg.norm (gripper_pos - target_pos) <=_APPROACH_DISTANCE:
                    logger.debug(f"Reached approach distance: {np.linalg.norm(gripper_pos - target_pos):.4f} <= {_APPROACH_DISTANCE}. Early stopping.")
                    break  # Stop the approach when close enough

            # Evaluation image saving inside the loop
            if generate_evaluation_data and frame is not None and skill not in ["lift_up"]:
                if action_idx in indices_to_save:
                    eval_frame = get_img_from_obs(raw_obs, key="agentview_image")
                    # Sanitize target name for filename
                    target_str = str(target)
                    sanitized_target_str = target_str.replace("_main", "").replace(" ", "_")

                    # 1. Create unique filename with frame index
                    image_filename = f"{dataset_id}_subtask{step_idx}_{skill}_{sanitized_target_str}_frame{action_idx}.png"
                    image_path = os.path.join(eval_image_dir, image_filename)

                    # 2. Save the frame
                    cv2.imwrite(image_path, cv2.cvtColor(eval_frame, cv2.COLOR_RGB2BGR))

                    # 3. Log to CSV
                    main_goal = task_plan.language_instruction
                    sub_task_str = f"{skill}({target_str})"
                    with open(log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([main_goal, sub_task_str, os.path.abspath(image_path), "", ""])

        print(f"[Debug] Skill {skill}: total={len(step_actions)}, executed={_executed_count}, skipped={_skipped_count}")

        # pick_top_angle / lay_down 执行完后: reset_pose 回到 skill 前位姿
        if skill in ("pick_top_angle", "lay_down"):
            recover_actions = reset_pose(env, _pre_skill_eef_pos, _pre_skill_eef_orn, gripper_state=gripper_state, speed_scale=_speed_scale)
            for action in recover_actions:
                raw_obs, reward, done, info = env.step(action)
                if done and not _Done:
                    _Done = 1
                save_as_hdf5(raw_obs=raw_obs, action=action, done=done, reward=reward, env=env, hdf5_dict=hdf5_dict,
                             reward_dict={})
                if _save_all_video or batch_id <= n_save_video:
                    frame = get_img_from_obs(raw_obs, key="agentview_image")
                    eih = get_img_from_obs(raw_obs, key="robot0_eye_in_hand_image")
                    far = get_img_from_obs(raw_obs, key="frontview_closer_image")
                    frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                    eih_resized = cv2.resize(eih, (256, 256), interpolation=cv2.INTER_LINEAR)
                    far_resized = cv2.resize(far, (256, 256), interpolation=cv2.INTER_LINEAR)
                    combined = np.concatenate([frame_resized, eih_resized, far_resized], axis=1)
                    video_writer_s.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    video_writer_f.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        # pick_back_edge 执行完后: 碗已稳定, 读取稳定的 gripper-bowl 偏移
        if skill == "pick_back_edge" and '_picked_object_name' in dir():
            _eef_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
            _bowl_pos = env.sim.data.get_body_xpos(_picked_object_name).copy()
            _picked_bowl_offset = _eef_pos[:2] - _bowl_pos[:2]
            print(f"[Debug] Stable bowl offset after settle: gripper={_eef_pos[:2]}, bowl={_bowl_pos[:2]}, offset=[{_picked_bowl_offset[0]:.4f}, {_picked_bowl_offset[1]:.4f}]")

        _prev_skill = skill  # 给下个 step 的 height offset 用
        _meta_collector.on_segment(skill, target, len(hdf5_dict["actions"]))

    # ====== Debug: 打印 pick 后各物体位置 ======
    gripper_final = env.sim.data.get_body_xpos("gripper0_eef").copy()
    print(f"[Debug] After all steps: gripper={gripper_final}")
    for bname in env.sim.model.body_names:
        if 'akita_black_bowl' in bname and 'main' in bname:
            bpos = env.sim.data.body_xpos[env.sim.model.body_name2id(bname)].copy()
            dist = np.linalg.norm(bpos - gripper_final)
            print(f"[Debug]   {bname}: pos={bpos}, dist_to_gripper={dist:.3f}")
        if 'plate' in bname and 'main' in bname:
            bpos = env.sim.data.body_xpos[env.sim.model.body_name2id(bname)].copy()
            print(f"[Debug]   {bname}: pos={bpos}")

    # ====== 末尾等待: 最多 10 步 dummy action 帮助 BDDL goal state 判定 ======
    # 这些步同样写进 HDF5: 轨迹够快时 done 只在这段尾巴里发生, 不写就会得到 dones 全 0 的
    # "成功但没标记" 数据 (2026-08-30 acc ramp 把 1-3 从 210 步缩到 139 步后暴露)。
    # done 之后再记 2 步就停, 保持尾巴短。
    if not _Done:
        dummy_action = np.zeros(7)
        dummy_action[6] = step_actions[-1][6] if len(step_actions) > 0 else 1.0
        _post_done_steps = 0
        for _ in range(10):
            raw_obs, reward, done, info = env.step(dummy_action)
            if done and not _Done:
                _Done = 1
            _tail_pref_reward = {}
            if low_pref2apply is not None:
                for _pref in low_pref2apply:
                    _tail_pref_reward[_pref.preference_type] = _pref.get_reward(env)
            save_as_hdf5(raw_obs=raw_obs, action=dummy_action.tolist(), done=done, reward=reward,
                         env=env, hdf5_dict=hdf5_dict, reward_dict=_tail_pref_reward)
            if _Done:
                _post_done_steps += 1
            if _save_all_video or batch_id <= n_save_video:
                frame = get_img_from_obs(raw_obs, key="agentview_image")
                eih = get_img_from_obs(raw_obs, key="robot0_eye_in_hand_image")
                far = get_img_from_obs(raw_obs, key="frontview_closer_image")
                frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                eih_resized = cv2.resize(eih, (256, 256), interpolation=cv2.INTER_LINEAR)
                far_resized = cv2.resize(far, (256, 256), interpolation=cv2.INTER_LINEAR)
                combined = np.concatenate([frame_resized, eih_resized, far_resized], axis=1)
                video_writer_s.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                video_writer_f.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            if _post_done_steps >= 2:
                break

    # ====== 保存 HDF5 数据 ======
    if _Done:
        logger.debug("Episode ended with done=True during execution.")
        hdf5_path= hdf5_path_s
    else:
        hdf5_path= hdf5_path_f

    h5py_f = h5py.File (hdf5_path, "w")

    grp = h5py_f.create_group ("data")
    ep_data_grp = grp.create_group (f"demo_{batch_id}")
    obs_grp = ep_data_grp.create_group ("obs")

    obs_dict = hdf5_dict ["obs"]

    # ====== 写 obs 子 group ======
    for key, data_list in obs_dict.items ():
        if len (data_list) == 0:
            continue
        _compressed (obs_grp, key, data_list)

    # ====== 写动作、状态、机器人状态等 ======
    _compressed (ep_data_grp, "actions", hdf5_dict ["actions"])
    _compressed (ep_data_grp, "states", hdf5_dict ["states"])
    _compressed (ep_data_grp, "robot_states", hdf5_dict ["robot_states"])
    _compressed (ep_data_grp, "rewards", hdf5_dict ["rewards"])
    _compressed (ep_data_grp, "dones", hdf5_dict ["dones"])
    # ====== 写 low preference reward ======
    low_pref_grp = ep_data_grp.create_group ("low_pref_rewards")
    for key, lst in hdf5_dict ["low_pref_rewards"].items ():
        _compressed (low_pref_grp, key, lst)

    # ====== 属性 ======
    ep_data_grp.attrs ["model_file"] = model_xml
    ep_data_grp.attrs ["num_samples"] = len (hdf5_dict ["actions"])
    ep_data_grp.attrs ["init_state"] = hdf5_dict ["states"] [0]

    # ====== gen-time metadata (T3): per-subtask 边界/language/speed/事件/pref raw ======
    try:
        _meta_collector.write_to(ep_data_grp, hdf5_dict=hdf5_dict, env=env,
                                 low_pref2apply=low_pref2apply)
    except Exception as _meta_err:
        logger.warning(f"[meta] gen metadata write failed (demo saved without meta): {_meta_err}")

    h5py_f.close ()
    if _Done:
        print (f"  ✅ HDF5 saved: {hdf5_path}")
        logger.info(f"✅ HDF5 saved: {hdf5_path}")
        if save_eval_hdf5:
            # 复制成功的 rollout HDF5 到 Eval/ 目录，供后续 VLM evaluation 使用
            import shutil
            eval_folder = os.path.join(os.path.dirname(hdf5_folder), "Eval")
            os.makedirs(eval_folder, exist_ok=True)
            eval_hdf5_path = os.path.join(eval_folder, f"rollout_{batch_id}.hdf5")
            shutil.copy2(hdf5_path, eval_hdf5_path)
            logger.info(f"📋 Eval copy: {eval_hdf5_path}")
    else:
        print (f"  ❌ HDF5 saved: {hdf5_path}")
        logger.info(f"❌ HDF5 saved: {hdf5_path}")
    if save_gif:
        if len (gif_frames) > 0:
            if _Done:
                succ="Succ"
            else:
                succ="Fail"
            gif_path = os.path.join (
                gif_folder,
                f"{succ}_mission_{dataset_id}_batch_{batch_id}.gif"
            )
            imageio.mimsave (
                gif_path,
                gif_frames,
                duration=1000/10,  # 建议 1000/X
                loop=0  # 无限循环
            )
            print (f"  🎞️ GIF saved: {gif_path}")
            logger.info(f"🎞️ GIF saved: {gif_path}")

    if _draw_plot:
        plt.figure (figsize=(8, 4))
        for pref_type, rs in pref_rewards.items ():
            plt.plot (rs, label=pref_type)
        plt.xlabel ("Timestep")
        plt.ylabel ("Reward")
        plt.title ("Low-level preference rewards over time")
        plt.legend ()
        plt.grid (True)
        plt.savefig (plot_path, dpi=300, bbox_inches="tight")
        plt.close ()
        print (f"  ✅ Plot saved: {plot_path}")
        logger.info(f"✅ Plot saved: {plot_path}")

    if _save_all_video or batch_id<=n_save_video:
        video_writer_f.release ()
        video_writer_s.release ()
        if _Done:
            os.remove (video_path_f)  # ❗删掉失败视频
            _reencode_mp4(video_path_s)
            print (f"✅ Success Video saved at: {video_path_s}")
            logger.info(f"✅ Video saved: {video_path_s}")
        else:
            os.remove (video_path_s)  # ❗删掉成功视频
            _reencode_mp4(video_path_f)
            print (f"❌ Fail Video saved at: {video_path_f}")
            logger.info(f"❌ Video saved: {video_path_f}")
    else:
        os.remove (video_path_f)  # ❗删掉失败视频
        os.remove (video_path_s)  # ❗删掉成功视频
    clear_markers()
    return _Done, hdf5_folder, len(hdf5_dict["actions"])

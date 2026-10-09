import logging
import math
import random
from dataclasses import dataclass
from  typing import List, Tuple
from Utility.tool_func import to_sim_env_name
import numpy as np

logger = logging.getLogger(__name__)


# Converts action-delta differences (‖Δ action[:3]‖ per control step) to the
# scale of AccRangePreference's raw signal (per-step change of the EE linear
# velocity read from sim, m/s). Empirically calibrated on Mission 1-3_107
# (2026-07-01): a command jump 0→0.0085/step produces a measured peak
# Δ‖ee_vel‖ ≈ 0.052 → factor ≈ 6.1. The previous default (0.0125) under-
# estimated ~500×, so `enforce_max_acceleration` never triggered and max_acc
# generation was a silent no-op (pref trajectory ≡ naive, bitwise).
ACTION_DELTA_TO_EE_ACC: float = 6.0


def max_acceleration(
    delta_list: List[np.ndarray],
    dt: float = 1.0,
    scale: float = ACTION_DELTA_TO_EE_ACC
) -> float:
    """
    delta_list: motion list
    dt: duration between two motions(in seconds)
    scale: action-delta → metric-scale conversion (see ACTION_DELTA_TO_EE_ACC)
    """
    if not delta_list:
        return 0.0

    arr = np.vstack(delta_list)
    v = arr[:, :3]

    if v.shape[0] < 2:
        return 0.0

    dv = np.diff(v, axis=0)
    a = np.linalg.norm(dv, axis=1)
    a = a / dt
    a=a*scale
    return float(a.max())

ACC_RESCALE_FLOOR = 0.67  # legacy uniform-slowdown path: max slowdown 1.5x (0.20 = 5x made acc demos slow-motion, 137 -> 720 steps)

# --- ramp path (default) -------------------------------------------------
# The legacy path rescales a whole segment by one factor whenever its PEAK
# action-delta exceeds max_acc. Because that peak is the start-of-segment jump
# (0 -> cruise in one step, ~5 in metric units vs max_acc 0.03), every segment
# hit the floor and the demo became uniformly slow — i.e. "avoid sudden
# acceleration" turned into "slow down" (measured: 1-3 demos 1.5x slower than
# naive, while 1-2 slow-down demos scored BETTER on the acc metric).
#
# The ramp path instead keeps the geometric path and the cruise speed and only
# re-times the accelerating parts: a trapezoidal speed profile bounded by
# ACC_RAMP_LIMIT_M. A segment grows by ~v_cruise/limit frames (~7 at the
# default), not by 1/floor x.
ACC_USE_RAMP = True
ACC_RAMP_LIMIT_M = 0.0012   # m/step^2 on the commanded EE path (naive P95 ~ 0.0041)
ACTION_POS_UNIT_M = 0.012   # action[:3] == dpos / 0.012


def ramp_limit_actions(step_actions, last_action=None, accel_m: float = ACC_RAMP_LIMIT_M):
    """Re-time one skill segment so |Δ speed| <= `accel_m` per step.

    Same geometric path and same peak speed as the input; only the ramp-up /
    ramp-down parts are stretched, so segments that already satisfy the limit
    come back unchanged (up to resampling noise). Returns a list of 7-dim
    actions."""
    arr = np.asarray(step_actions, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 7:
        return step_actions
    pos = arr[:, :3] * ACTION_POS_UNIT_M                 # per-step displacement (m)
    seg = np.linalg.norm(pos, axis=1)
    S = float(seg.sum())
    v_max = float(seg.max())
    a = float(accel_m)
    if S < 1e-6 or v_max < 1e-9 or a <= 0:
        return step_actions
    # already smooth? (peak |Δv| within the limit and starts/ends at rest-ish)
    dv = np.abs(np.diff(seg))
    v0 = 0.0
    if last_action is not None and len(np.ravel(last_action)) >= 3:
        v0 = min(float(np.linalg.norm(np.asarray(last_action, float).ravel()[:3] * ACTION_POS_UNIT_M)), v_max)
    if dv.size and dv.max() <= a and abs(seg[0] - v0) <= a and seg[-1] <= a:
        return step_actions

    s_orig = np.concatenate([[0.0], np.cumsum(seg)])     # arc length of input samples
    P = np.vstack([np.zeros(3), np.cumsum(pos, axis=0)])
    Rc = np.vstack([np.zeros(3), np.cumsum(arr[:, 3:6], axis=0)])

    # trapezoidal profile: accelerate by `a`, cap at v_max, brake in time to stop
    s, v, s_new = 0.0, v0, [0.0]
    guard = int(10 * len(seg)) + 2000
    while s < S - 1e-9 and len(s_new) < guard:
        v_brake = math.sqrt(max(2.0 * a * (S - s), 0.0))
        v = max(min(v_max, v + a, v_brake), min(a, v_max))
        s = min(S, s + v)
        s_new.append(s)
    s_new = np.asarray(s_new)

    P_new = np.column_stack([np.interp(s_new, s_orig, P[:, k]) for k in range(3)])
    R_new = np.column_stack([np.interp(s_new, s_orig, Rc[:, k]) for k in range(3)])
    idx = np.clip(np.searchsorted(s_orig, s_new[1:], side="left") - 1, 0, len(arr) - 1)
    out = np.zeros((len(s_new) - 1, arr.shape[1]))
    out[:, :3] = np.diff(P_new, axis=0) / ACTION_POS_UNIT_M
    out[:, 3:6] = np.diff(R_new, axis=0)
    out[:, 6:] = arr[idx, 6:]
    return [row for row in out]


def _rescale_factor(current_acc: float, acc_max: float,
                    margin: float = 0.9, floor: float = ACC_RESCALE_FLOOR) -> float:
    """Speed-scale needed to bring `current_acc` under `acc_max`.

    Slowing a trajectory by factor s scales consecutive deltas — and hence
    their differences — by ~s, so s = acc_max/current_acc lands exactly on the
    limit; `margin` leaves headroom. (The old tiered mapping computed the ratio
    upside-down: acc_max/current < 1 whenever the limit was exceeded, so every
    violation fell through to the same 0.50 — a tightened clamp changed
    nothing, which broke rejection-sampling feedback.) `floor` caps the
    slowdown at 1/floor× to keep episode length bounded.
    """
    return max(floor, margin * acc_max / max(current_acc, 1e-9))


def enforce_max_acceleration (
        step_actions: List [np.ndarray],
        _acc_max: float,
        last_step_actions: List [np.ndarray],
        skill: str,
        _self_adjustable_skills: List [str] = None,
        sensitive_skills: List [str] = None
) -> List [np.ndarray]:
    """
    Enforce max acceleration limit on step_actions in two phases:
    1. Transition phase: Handle acceleration between last 1/5 of previous step
       and first 1/5 of current step by inserting interpolation frames
    2. Current phase: Handle acceleration within current step_actions
    """
    if _self_adjustable_skills is None:
        _self_adjustable_skills = []
    if sensitive_skills is None:
        sensitive_skills = []

    if ACC_USE_RAMP:
        # Ramp path: re-time this segment (cruise speed untouched); segments
        # already within the limit come back unchanged.
        if skill in _self_adjustable_skills:
            logger.debug(f"[Motion Gen][Acc][ramp] skill '{skill}' self-adjustable, skipping")
            return step_actions
        last = last_step_actions[-1] if (last_step_actions is not None and len(last_step_actions) > 0) else None
        before = max_acceleration(step_actions)
        out = ramp_limit_actions(step_actions, last_action=last)
        logger.debug(f"[Motion Gen][Acc][ramp] {skill}: {len(step_actions)} -> {len(out)} steps, "
                     f"action-peak {before:.4f} -> {max_acceleration(out):.4f}")
        return out

    # ==================== Phase 1: Transition Phase ====================
    if last_step_actions is not None and len (last_step_actions) > 0 and len (step_actions) > 0:
        # Get last 1/5 of previous step_actions
        last_fifth_idx = len (last_step_actions) * 4 // 5
        last_fifth = last_step_actions [last_fifth_idx:]

        # Get first 1/5 of current step_actions
        first_fifth_idx = max (1, len (step_actions) // 5)
        first_fifth = step_actions [:first_fifth_idx]

        # Combine transition zone
        transition_zone = last_fifth + first_fifth
        transition_acc = max_acceleration (transition_zone)

        logger.debug (f"[Motion Gen][Acc][Phase 1 - Transition] Before: max acceleration = {transition_acc}")

        if transition_acc > _acc_max:
            scale = _rescale_factor(transition_acc, _acc_max)

            last_action = last_step_actions [-1]
            first_action = step_actions [0]

            transition_frames = rescale_traj_speed_transition (
                last_action=last_action,
                first_action=first_action,
                scale=scale
            )

            if len (transition_frames) > 0:
                step_actions = transition_frames + step_actions

            # Recalculate transition acceleration
            first_fifth_idx_new = max (1, len (step_actions) // 5)
            first_fifth_new = step_actions [:first_fifth_idx_new]
            transition_zone_new = last_fifth + first_fifth_new
            transition_acc_new = max_acceleration (transition_zone_new)

            logger.debug (f"[Motion Gen][Acc][Phase 1 - Transition] After: max acceleration = {transition_acc_new}")
        else:
            logger.debug (f"[Motion Gen][Acc][Phase 1 - Transition] Within limit, no adjustment needed")
    else:
        logger.debug (f"[Motion Gen][Acc][Phase 1 - Transition] No previous step actions, skipping")

    # ==================== Phase 2: Current Step Phase ====================
    current_acc = max_acceleration (step_actions)
    logger.debug (f"[Motion Gen][Acc][Phase 2 - Current Step] Before: max acceleration = {current_acc}")

    if current_acc > _acc_max:
        scale = _rescale_factor(current_acc, _acc_max)

        if skill in _self_adjustable_skills:
            logger.debug (f"[Motion Gen][Acc][Phase 2 - Current Step] Skill '{skill}' is self-adjustable, skipping")
        elif skill in sensitive_skills:
            # For sensitive skills, use the special acc-safe function
            step_actions = rescale_traj_speed_safe_for_acc (step_actions, scale=scale)
            current_acc_new = max_acceleration (step_actions)
            logger.debug (
                f"[Motion Gen][Acc][Phase 2 - Current Step] After (safe_for_acc): max acceleration = {current_acc_new}")
        else:
            step_actions = rescale_traj_speed (delta_list=step_actions, scale=scale)
            current_acc_new = max_acceleration (step_actions)
            logger.debug (f"[Motion Gen][Acc][Phase 2 - Current Step] After (rescale): max acceleration = {current_acc_new}")
    else:
        logger.debug (f"[Motion Gen][Acc][Phase 2 - Current Step] Within limit, no adjustment needed")

    # ==================== Final Summary ====================
    final_acc = max_acceleration (step_actions)
    logger.debug (f"[Motion Gen][Acc][Summary] Final max acceleration: {final_acc}, limit: {_acc_max}")

    return step_actions


def rescale_traj_speed_transition (
        last_action: np.ndarray,
        first_action: np.ndarray,
        scale: float
) -> List [np.ndarray]:
    """
    Insert transition frames to smooth the acceleration between last_action and first_action.
    """
    last_action = np.array (last_action)
    first_action = np.array (first_action)

    num_frames = max (1, int (1.0 / scale) - 1)

    if num_frames == 0:
        return []

    transition_frames = []
    for i in range (1, num_frames + 1):
        t = i / (num_frames + 1)
        transition_delta = last_action * (1 - t) + first_action * t
        transition_frames.append (transition_delta)

    return transition_frames


def rescale_traj_speed_safe_for_acc (
        delta_list: List [List [float]],
        scale: float,
        _acc_max: float = None,
        max_iterations: int = 5
) -> List [List [float]]:
    """
    专门为 max acceleration enforcement 设计的安全减速函数。
    将每个 delta 拆分成多个更小的 delta。
    如果提供 _acc_max，会迭代直到满足加速度要求。

    Args:
        delta_list: List of delta actions
        scale: Scale factor (smaller = more splits = lower acceleration)
        _acc_max: Target max acceleration (if provided, will iterate until satisfied)
        max_iterations: Maximum number of iterations to avoid infinite loop

    Returns:
        List of split delta actions
    """
    if not delta_list:
        return delta_list

    deltas = np.asarray (delta_list, dtype=np.float64)
    T = deltas.shape [0]

    if scale >= 1.0:
        return delta_list

    # Calculate initial num_splits based on scale
    if scale <= 0.20:
        num_splits = 5
    elif scale <= 0.25:
        num_splits = 4
    elif scale <= 0.33:
        num_splits = 3
    else:
        num_splits = 2

    # Detect gripper change indices
    protected = set ()
    for i in range (1, T):
        if deltas [i, 6] != deltas [i - 1, 6]:
            protected.add (i - 1)
            protected.add (i)
            if i + 1 < T:
                protected.add (i + 1)

    current_deltas = deltas
    current_protected = protected

    for iteration in range (max_iterations):
        new_actions = []
        new_protected = set ()
        action_idx = 0

        for i in range (len (current_deltas)):
            action = current_deltas [i]

            if i in current_protected:
                new_actions.append (action if isinstance (action, np.ndarray) else np.array (action))
                new_protected.add (action_idx)
                action_idx += 1
            else:
                small_delta = np.zeros (7)
                small_delta [:6] = action [:6] / num_splits
                small_delta [6] = action [6]

                for _ in range (num_splits):
                    new_actions.append (small_delta.copy ())
                    action_idx += 1

        # Check if we meet the acceleration requirement
        if _acc_max is not None:
            from . import max_acceleration  # 或者直接调用
            current_acc = max_acceleration (new_actions)
            if current_acc <= _acc_max:
                logger.debug (
                    f"[Motion Gen][Acc][safe_for_acc] Converged after {iteration + 1} iteration(s), acc = {current_acc}")
                return [a.tolist () if isinstance (a, np.ndarray) else a for a in new_actions]
            else:
                # Prepare for next iteration
                current_deltas = new_actions
                # Update protected indices for next iteration (simplified: clear it)
                current_protected = new_protected
                logger.debug (f"[Motion Gen][Acc][safe_for_acc] Iteration {iteration + 1}, acc = {current_acc}, continuing...")
        else:
            # No target, just return after one pass
            return [a.tolist () if isinstance (a, np.ndarray) else a for a in new_actions]

    # Max iterations reached
    logger.debug (f"[Motion Gen][Acc][safe_for_acc] Max iterations reached, returning best effort")
    return [a.tolist () if isinstance (a, np.ndarray) else a for a in new_actions]

def rescale_traj_speed(delta_list, scale: float):
    if scale <= 0:
        raise ValueError("scale must be larger than 0")

    deltas = np.asarray(delta_list, dtype=np.float64)
    T = deltas.shape[0]
    if T == 0:
        return []

    deltas_pos = deltas[:, :6]
    k_vals = deltas[:, 6]

    states_pos = np.cumsum(deltas_pos, axis=0)
    states_pos_full = np.vstack([np.zeros((1, 6)), states_pos])

    k_full = np.empty(T + 1, dtype=np.float64)
    k_full[0] = k_vals[0]
    k_full[1:] = k_vals

    t_orig = np.arange(T + 1, dtype=np.float64)

    T_new = int(np.round(T / scale))
    T_new = max(1, T_new)

    t_sample = np.arange(T_new + 1, dtype=np.float64) * scale
    t_sample = np.clip(t_sample, 0.0, float(T))

    new_states_pos = np.empty((T_new + 1, 6), dtype=np.float64)
    for j in range(6):
        new_states_pos[:, j] = np.interp(t_sample, t_orig, states_pos_full[:, j])

    new_k_full = np.interp(t_sample, t_orig, k_full)

    new_deltas_pos = np.diff(new_states_pos, axis=0)

    new_k_vals = new_k_full[1:]

    new_deltas = np.zeros((T_new, 7), dtype=np.float64)
    new_deltas[:, :6] = new_deltas_pos
    new_deltas[:, 6] = new_k_vals

    return [new_deltas[i] for i in range(T_new)]



def generate_avoid_waypoints(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    obstacle_pos: np.ndarray,
    avoid_range: Tuple[float, float],  # (_AVOID_MIN_THRESHOLD, _AVOID_MAX_THRESHOLD)
    z_offset: float = 0.15
) -> np.ndarray:
    """
    新版避障点生成逻辑：
    1. 用 gripper_pos 与 target_pos 的中点作为局部系原点。
    2. 将 obstacle_pos 转成此系下的 local 坐标。
    3. 对 local.x, local.y 执行镜像反转（对称点），z 不变。
    4. 得到镜像点 mir_global。
    5. obstacle_pos -> mir_global 形成射线。
    6. 根据 avoid_range 限制距离，生成最终 waypoint。
    7. z 可保持不变或加 z_offset。
    """

    _AVOID_MIN_THRESHOLD, _AVOID_MAX_THRESHOLD = avoid_range

    # Step 1: 取中点 M
    mid = (start_pos + end_pos) / 2.0

    # Step 2: obstacle 在局部系下的坐标
    obs_local = obstacle_pos - mid  # (x, y, z)

    # Step 3: 镜像 (x, y)，z 不变
    mir_local = np.array([-obs_local[0], -obs_local[1], obs_local[2]])

    # Step 4: 转回全局系
    mir_global = mid + mir_local

    # Step 5: obstacle → mir_global 的方向向量
    ray_vec = mir_global - obstacle_pos
    ray_len = np.linalg.norm(ray_vec)

    if ray_len < 1e-6:
        ray_len = 1e-5  # 防止除零

    ray_dir = ray_vec / ray_len

    # Step 6: 距离限制
    # 镜像距离一般不一定刚好落在范围内，所以需要 clamp
    dist = np.clip(ray_len, _AVOID_MIN_THRESHOLD, _AVOID_MAX_THRESHOLD)

    # Step 7: waypoint = obstacle_pos + ray_dir * dist
    waypoint = obstacle_pos + ray_dir * dist

    # Step 8: z 处理：保持 mir_global 的 z 或进行轻微上抬
    waypoint[2] = mir_global[2] + z_offset

    return waypoint



def apply_approach(naive_plan: List[Tuple[str, str]],
                   pref_target_obj: str,
                   env,
                   half_approach=False) -> List[Tuple[str, str, str]]:
    """

    """
    if naive_plan is None:
        raise AttributeError("TaskPreference object must have 'plans' before generating real plans.")

    if pref_target_obj is None:
        raise AttributeError("Preferred target object must be specified.")
    real_plans = []

    # 初始 gripper 状态
    gripper_state = "open"   # open/close/None
    for skill, obj in naive_plan:
        # 如果还没做 approach 注入，并且 skill 是 move_to
        obj = to_sim_env_name(obj)
        if skill == "move_to":
            if half_approach:
                if gripper_state=="close" and obj != pref_target_obj:
                    # 插入 approach
                    if pref_target_obj == "ground_1":
                        real_plans.append (("move_to_approach", obj))
                    else:
                        real_plans.append(("lift_up", None))
                        real_plans.append(("move_to_approach", obj))
                else:
                    real_plans.append((skill, obj))

            else:
                if pref_target_obj == "ground_1":
                    real_plans.append (("move_to_approach", obj))
                else:
                    if obj != pref_target_obj:
                        # 插入 approach
                        real_plans.append(("lift_up", None))
                        real_plans.append(("move_to_approach", obj))
                    else:
                        # 普通步骤
                        real_plans.append((skill, obj))
        else:
            # 普通步骤
            real_plans.append((skill, obj))

        # 之后再更新 gripper 状态（作用于下一步）
        if skill.startswith("pick"):
            gripper_state = "close"
        elif skill.startswith("put"):
            gripper_state = "open"
        else:
            # 不改变
            pass

    return real_plans



def apply_away (naive_plan: List [Tuple [str, str]], env) -> List [Tuple [str, str]]:
    """
    Add step to move away from the preferred target before each move_to action.
    """

    # pref_target_obj_main=to_sim_env_name(pref_target_obj)
    # idx = env.sim.model.body_name2id (pref_target_obj_main)
    # target_xpos2away = env.sim.data.body_xpos [idx]
    # gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
    # obst_size = get_object_size_3d (env, pref_target_obj)  # e.g. [dx, dy, dz]
    # 自动根据物体大小确定安全球半径
    # wps = ball_waypoint_gen (
    #     center=target_xpos,
    #     obst_size=obst_size,
    #     n_points=n_points,
    #     theta_range=(0, np.pi / 2),
    #     phi_range=(0, np.pi / 2),
    #     z_offset=0.03
    # )
    real_plans = []
    gripper_state = "open"
    for skill, obj in naive_plan:
        if obj is None:
            real_plans.append ((skill, obj))
            continue
        obj= to_sim_env_name (obj)
        if skill == "move_to" :
            # lift_up only while carrying. The empty-hand (pre-pick) lift used
            # to push the EE to z≈1.34 (start 1.176 + 0.15 + overshoot) — out
            # of agentview, and the horizontal-only avoid metric gains nothing.
            if gripper_state == "close":
                real_plans.append(("lift_up", None))
            real_plans.append (("move_to_away", obj))
        else:
            real_plans.append ((skill, obj))
        if skill.startswith("pick"):
            gripper_state = "close"
        elif skill.startswith("put"):
            gripper_state = "open"

    return real_plans



# def apply_approach_v0 (env, env2, motion_planner, target_name, gripper_val=-1):
#     target_xpos = env.sim.data.get_body_xpos (target_name).copy ()
#     target_size = get_body_size_3d (env, target_name)
#     aabb_max = target_size ["aabb_max"]
#     aabb_min = target_size ["aabb_min"]
#     gripper_xpos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
#     logger.debug (f"Target '{target_name}' AABB min: {aabb_min}, max: {aabb_max}")
#     logger.debug (f"Gripper position: {gripper_xpos}")
#     mid_xpos = (aabb_min + aabb_max) / 2
#     target_xpos = mid_xpos
#     target_xpos [2] = aabb_max [2] + 0.1  # 抬高一点
#     target_xpos [0] = np.clip (gripper_xpos [0], aabb_min [0] - 0.05, aabb_max [0] + 0.05)
#     target_xpos [1] = np.clip (gripper_xpos [1], aabb_min [1] - 0.05, aabb_max [1] + 0.05)
#     logger.debug (f"Approach target position: {target_xpos}")
#     actions = plan_relative_osc_pose (env, env2, motion_planner, target_xpos, gripper_val=gripper_val)
#     return actions

# =====================================================
# Gen-side low_pref application layer (extracted from motion_gen 2026-07-01)
# — motion_gen.execute_plan 只保留 skill 分发；type 分支/优先级/waypoint
#   计算全部集中在这里，与 metric 侧 (Preference/low_level/*.py) 同目录对照。
# =====================================================

# speed_range 施加时允许的离散速度倍率（与 rescale 机制的插帧档位一致）。
ALLOWED_SPEED_SCALES = [0.2, 0.25, 0.33, 0.5, 1.0, 1, 1.5, 2.0]


def get_pref_priority(pref) -> int:
    """low_pref 施加顺序（数值越小越优先）：
    OnTop > Approach > Avoid > SpeedRange > AccRange > 其余。"""
    from Preference.low_level.extra.relationship import OnTopPreference
    from Preference.low_level.distance import (
        ApproachToObjectPreference, AvoidToObjectPreference,
    )
    from Preference.low_level.speed import SpeedRangePreference, AccRangePreference
    if isinstance(pref, OnTopPreference):
        return 0
    if isinstance(pref, ApproachToObjectPreference):
        return 1
    if isinstance(pref, AvoidToObjectPreference):
        return 2
    if isinstance(pref, SpeedRangePreference):
        return 3
    if isinstance(pref, AccRangePreference):
        return 4
    return 99


@dataclass
class LowPrefGenPlan:
    """execute_plan 消费的全部 gen 侧旋钮（由 apply_low_prefs_to_plan 解析）。"""
    speed_scale: float = 1.0
    acc_max: float | None = None
    path_height_offset: float | None = None
    prefer_straight_path: bool = False
    prefer_rotation_while_moving: bool = False
    obj_to_approach: str | None = None
    approach_distance: float | None = None
    obj_to_avoid: str | None = None
    avoid_min_threshold: float | None = None
    avoid_max_threshold: float | None = None


def apply_low_prefs_to_plan(low_pref2apply, plan, env, rng=None):
    """按优先级施加全部 low_pref：改写 plan（approach/away 插桩）并解析旋钮。

    Returns:
        (plan, LowPrefGenPlan)
    """
    cfg = LowPrefGenPlan()
    if not low_pref2apply:
        return plan, cfg
    rng = rng or random
    for low_pref in sorted(low_pref2apply, key=get_pref_priority):
        logger.debug(f"Applying low-level preference: {low_pref.type}")
        if low_pref.type == "ApproachToObjectPreference":
            cfg.obj_to_approach = to_sim_env_name(low_pref.target_obj)
            logger.debug(f"  Approach target: {cfg.obj_to_approach}")
            plan = apply_approach(plan, cfg.obj_to_approach, env=env, half_approach=True)
            cfg.approach_distance = 0.2 if cfg.obj_to_approach == "ground_1" else 0.25
        if low_pref.type == "AvoidToObjectPreference":
            cfg.obj_to_avoid = to_sim_env_name(low_pref.target_obj)
            plan = apply_away(plan, env=env)
            # clearance = mission's `distance`/`threshold` param, floored at 0.3
            # (the old mirrored-waypoint minimum). Read by plan_avoid_segment.
            cfg.avoid_min_threshold = max(0.3, float(getattr(low_pref, "threshold", 0.0) or 0.0))
            cfg.avoid_max_threshold = 0.6
        if low_pref.type == "SpeedRangePreference":
            speed_range = (low_pref.min_speed, low_pref.max_speed)
            valid_scales = [
                s for s in ALLOWED_SPEED_SCALES
                if speed_range[0] <= s <= speed_range[1]
            ]
            if not valid_scales:
                raise ValueError(
                    f"No valid speed_scale in range {speed_range}. "
                    f"Allowed values: {ALLOWED_SPEED_SCALES}"
                )
            cfg.speed_scale = rng.choice(valid_scales)
            logger.debug(f"Set speed scale to {cfg.speed_scale}")
        if low_pref.type == "AccRangePreference":
            cfg.acc_max = low_pref.max_acc
        if low_pref.type == "HeightOffsetPreference":
            cfg.path_height_offset = low_pref.height_offset
        if low_pref.type == "StraightPathPreference":
            cfg.prefer_straight_path = True
        if low_pref.type == "RotationPreference":
            cfg.prefer_rotation_while_moving = True
    return plan, cfg


def compute_avoid_waypoint(env, target_xpos, obj_to_avoid, avoid_range,
                           offset=None, picked_target=None):
    """avoid 绕行 waypoint：镜像绕行点 + z 策略（空手横移 / 持物保守抬高）。"""
    avoid_obj_name = to_sim_env_name(obj_to_avoid)
    gripper_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
    pos2avoid = env.sim.data.get_body_xpos(avoid_obj_name)
    waypoint = generate_avoid_waypoints(
        start_pos=gripper_pos, end_pos=target_xpos,
        obstacle_pos=pos2avoid, avoid_range=avoid_range,
    )
    if offset is not None:
        waypoint[0] += offset[0]
        waypoint[1] += offset[1]
        waypoint[2] += offset[2]
    # Empty-handed avoidance should feel like a lateral translation instead
    # of an unnecessary rise. When carrying an object, stay conservative.
    if picked_target is None:
        waypoint[2] = max(target_xpos[2] + 0.15, gripper_pos[2])
    else:
        waypoint[2] = max(waypoint[2], target_xpos[2], gripper_pos[2])
    # Hard ceiling: keep the EE inside agentview (naive peaks ≈1.30). A tall
    # place target still gets target_z + 0.10 headroom.
    waypoint[2] = min(waypoint[2], max(AVOID_Z_CAP_M, target_xpos[2] + 0.10))
    return waypoint


# OMPL keep-away transport is opt-in. On 1-4_034 (bowl 0.23 m / cabinet 0.17 m
# from the bottle, 25 cm carry) the whole segment lies inside the endpoint
# exemption balls, so OMPL either returns the straight line or a random 0.8 m
# detour — a layout problem, not a planner problem. Use it only for missions
# whose carry endpoints are >= clearance away from the obstacle.
AVOID_USE_OMPL = False
AVOID_Z_CAP_M = 1.25           # avoid 绕行点 z 上限（比全局 EE_Z_CAP_M 更紧；高 place 目标有 target_z+0.10 豁免）
AVOID_HELD_RADIUS_M = 0.05     # carried object radius added to the EE clearance
AVOID_ENDPOINT_EXEMPT_M = 0.12 # start/goal neighbourhood exempt from clearance (matches metric GRASP_EXCLUDE_M)
AVOID_OMPL_TIMEOUT_S = 5.0
AVOID_OMPL_INTERP_N = 60


def plan_avoid_segment(env, target_xpos, obj_to_avoid, clearance, *, speed,
                       gripper_state, offset=None, picked_target=None):
    """OMPL keep-away transport: plan EE start → goal (above target, at the
    current carry height) with `obj_to_avoid` as a spherical obstacle of
    radius `clearance` (+ held-object radius when carrying).

    Returns (step_actions, farthest_path_point) or None when OMPL finds no
    path (caller falls back to the mirrored-waypoint path)."""
    if not AVOID_USE_OMPL:
        return None
    from MotionGen.motion_planner.ompl_planner import ompl_plan_osc_pose
    avoid_body = to_sim_env_name(obj_to_avoid)
    gripper_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
    goal = np.asarray(target_xpos, dtype=float).copy()
    if offset is not None:
        goal[:3] += np.asarray(offset[:3], dtype=float)
    goal[2] = max(goal[2], gripper_pos[2])          # stay at the lifted height
    eef_quat_wxyz = env.sim.data.get_body_xquat("gripper0_eef").copy()
    clear = float(clearance) + (AVOID_HELD_RADIUS_M if picked_target is not None else 0.0)
    z_lo, z_hi = min(gripper_pos[2], goal[2]) - 0.02, max(gripper_pos[2], goal[2]) + 0.25
    try:
        wps, ok, to_osc = ompl_plan_osc_pose(
            env, goal, eef_quat_wxyz, obstacle_names=[avoid_body], avoid_distance=clear,
            timeout=AVOID_OMPL_TIMEOUT_S, interpolate_n=AVOID_OMPL_INTERP_N, z_bounds=(z_lo, z_hi),
            endpoint_exempt_radius=AVOID_ENDPOINT_EXEMPT_M,
        )
    except Exception as e:  # planner import / sim name errors → fallback
        logger.warning(f"[avoid-ompl] planning failed ({e}); falling back to waypoint")
        return None
    if not ok or not wps:
        logger.warning(f"[avoid-ompl] no path within {AVOID_OMPL_TIMEOUT_S}s; falling back to waypoint")
        return None
    actions = []
    for dpos, daxisangle in to_osc(step_pos=speed, step_rot=0.1):
        actions.append(np.concatenate([dpos / 0.012, daxisangle / 0.125, [gripper_state]]))
    obst = env.sim.data.get_body_xpos(avoid_body)
    far = max((w["pos"] for w in wps), key=lambda p: np.linalg.norm(p - obst))
    logger.debug(f"[avoid-ompl] {len(actions)} steps, clearance {clear:.2f}, farthest {np.linalg.norm(far - obst):.2f}")
    return actions, np.asarray(far, dtype=float)


def rotation_premove_target_override(next_skill, next_target):
    """rotate-while-moving：move_to 需要提前对准的实际目标（仅 turn_off_stove）。"""
    if next_skill == "turn_off_stove":
        return f"{next_target}_button"
    return None


def rotation_premove_orn(env, next_skill, next_target, subtask_offset):
    """rotate-while-moving 的 move_to 途中预旋目标姿态；不适用/失败 → None（保持当前姿态）。"""
    if next_skill == "turn_off_stove":
        knob_target = f"{next_target}_button"
        try:
            qpos_addr = env.sim.model.get_joint_qpos_addr(knob_target)
            knob_qpos = env.sim.data.qpos[qpos_addr]
            logger.info(
                f"[rotation_pref] Pre-rotating during move_to for {knob_target}, "
                f"knob_qpos={knob_qpos:.4f}"
            )
            return np.array([np.pi, 0, -np.pi / 2 - knob_qpos])
        except Exception:
            logger.warning(
                f"[rotation_pref] Failed to compute pre-rotation for {knob_target}; "
                "falling back to current orientation."
            )
            return None
    if next_skill == "pick_with_rotation":
        from Utility.tool_func import env_name2obj_name
        next_target_obj = env_name2obj_name(next_target)
        rotate_angle = None
        for key, conv in (
            (f"pick_with_rotation_deg:{next_target_obj}", np.deg2rad),
            ("pick_with_rotation_deg:_all", np.deg2rad),
            (f"pick_with_rotation_rot:{next_target_obj}", lambda x: x),
            ("pick_with_rotation_rot:_all", lambda x: x),
        ):
            if key in subtask_offset:
                rotate_angle = conv(subtask_offset[key])
                break
        if rotate_angle is not None:
            return np.array([np.pi, 0, -np.pi / 2 + rotate_angle])
    return None

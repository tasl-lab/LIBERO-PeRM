"""
Preference-Aware Layout Generation Module

This module generates initial scene layouts for robot manipulation tasks based on 
high-level user preferences. It uses a two-stage LLM pipeline:
  Stage 1: Convert task preferences to semantic spatial descriptions
  Stage 2: Convert semantic descriptions to precise (x, y) coordinates

KNOWN LIMITATIONS AND TODO ITEMS:

1. MULTI-INSTANCE LIMITATION (Critical)
   - Layout.object_init_position uses base names as dict keys (e.g., "milk" not "milk_1")
   - Cannot track multiple instances of same object type (e.g., milk_1 and milk_2)
   - Only last instance's position is stored, earlier instances are lost
   - Repositioning/removal always targets "_1" suffix, can't distinguish instances
   WHY THIS IS BAD: Real scenarios often need multiple similar objects (e.g., 2 cups)
   TODO: Redesign Layout.object_init_position to Dict[str, Dict[int, List[float]]]
         Example: {"milk": {1: [x1, y1, yaw1], 2: [x2, y2, yaw2]}}

2. PERFORMANCE SCALING (Efficiency)
   - Stage 1: ONE LLM call per preference, outputs descriptions for ALL objects in that preference
   - Stage 2: ONE LLM call per object to convert description → coordinates
   - Total: P + N LLM calls where P=preferences, N=objects to place
   - Example: 2 preferences + 5 repositioned objects = 7 LLM calls
   - LLM latency ~1-3s per call → 7-21 seconds for this example
   - Stage 2 calls are sequential (not parallel) to allow later objects to see earlier placements
   WHY THIS IS BAD: Slow iteration during testing, poor user experience for complex layouts
   TODO: Parallelize Stage 2 calls for independent objects (no "near X" constraints)
         Consider caching or pre-computing common spatial patterns

Additional notes:
- Two-pass approach (implemented) fixes race conditions but doesn't solve ordering
- Objects with bidirectional "near" constraints may not be optimally placed
- No global optimization - positions are determined greedily in iteration order
- Coordination between objects: Stage 1 sees relationships, Stage 2 sees incremental layout
  Example: "sauces grouped" → Stage 1 outputs both with "near each other" descriptions
           Stage 2 places tomato_sauce first, then salad_dressing sees its position and places nearby
"""


import logging
import numpy as np
import os
import random

logger = logging.getLogger("MotionGen")
import math
from math import sqrt
import re
from numba.typed.dictimpl import impl_dict
from typing import Set,Tuple, Dict, List, Optional
from TaskGen.task import TaskPlan
from TaskGen.layout_gen.layout import Layout
from TaskGen.layout_gen.utils import get_object_size
from libero.libero.envs import OffScreenRenderEnv
DUMMY_ACTION=[0,0,0,0,0,0,-1]
from Utility.tool_func import to_sim_env_name, ask_choice



def collision_detection_2d(new_min, new_max, existing_objects, margin=0.02):
    """
    In dense layout setting, we need to check whether two objects overlap in 2D top-down view.
    I.e. project the 3D bounding box to 2D plane, and check whether two 2D bounding boxes overlap.
    """
    x_min, y_min = new_min[0] - margin, new_min[1] - margin
    x_max, y_max = new_max[0] + margin, new_max[1] + margin

    for obj in existing_objects:
        ox_min, oy_min = obj["min"]
        ox_max, oy_max = obj["max"]

        if not (x_max < ox_min or x_min > ox_max or y_max < oy_min or y_min > oy_max):
            return True
    return False


def insert_obst_2d(
    num_obstacles=3, table_bounds=(-0.3, -0.3, 0.3, 0.3), existing_objects=None
):
    """    
    在2D平面随机放置障碍物，避免碰撞
    Args:
            num_obstacles: 要放置的障碍物数量
            table_bounds: 桌子边界 (x_min, y_min, x_max, y_max)
            existing_objects: 已有物体 [{'name':.., 'min':[x,y], 'max':[x,y]}]
    """
    if existing_objects is None:
        existing_objects = []

    placed = []
    for i in range(num_obstacles):
        width, height = 0.05, 0.05  # 障碍物大小，可改
        max_attempts, success = 500, False

        for _ in range(max_attempts):
            # 随机生成左下角
            x_min = random.uniform(table_bounds[0], table_bounds[2] - width)
            y_min = random.uniform(table_bounds[1], table_bounds[3] - height)
            x_max, y_max = x_min + width, y_min + height

            if not collision_detection_2d(
                [x_min, y_min], [x_max, y_max], existing_objects
            ):
                obj_id = f"obstacle_{len(existing_objects)+1}"
                new_obj = {"name": obj_id, "min": [x_min, y_min], "max": [x_max, y_max]}
                existing_objects.append(new_obj)  # ✅ 更新 existing_objects
                placed.append(new_obj)
                success = True
                print(f"✅ 放置 {obj_id} 在 [{x_min:.2f}, {y_min:.2f}]")
                break

        if not success:
            print(f"⚠️ 无法放置 obstacle_{i+1}")

    return placed, existing_objects


def collision_detection_3d(env, objects_of_interest: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Checks for collisions in the current simulation frame, with the ability to focus on specific objects.

    Args:
        env: The simulation environment instance.
        objects_of_interest (Optional[List[str]]): A list of object body names to check.
            If None, this function will check for any collision between any two non-static,
            non-robot objects in the scene. If provided, it will only report collisions
            where at least one of the bodies involved is in this list.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the names
        of two bodies that are in collision. Returns an empty list if no relevant
        collisions are detected.
    """
    sim = env.sim
    colliding_pairs = []

    # A set for faster lookups
    interest_set = set(objects_of_interest) if objects_of_interest else None

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]

        geom1_id = contact.geom1
        geom2_id = contact.geom2

        # In MuJoCo, geoms are the collision primitives. We need to find the bodies they belong to.
        body1_id = sim.model.geom_bodyid[geom1_id]
        body2_id = sim.model.geom_bodyid[geom2_id]

        # We get the names of the bodies, as these are more meaningful (e.g., "alphabet_soup_1").
        body1_name = sim.model.body_id2name(body1_id)
        body2_name = sim.model.body_id2name(body2_id)

        # Skip if names are invalid or if it's a self-collision
        if not body1_name or not body2_name or body1_id == body2_id:
            continue

        # Ignore collisions with static world elements like the table and floor, or robot parts.
        ignored_keywords = ['floor', 'table', 'robot', 'gripper']
        if any(keyword in body1_name for keyword in ignored_keywords) or \
           any(keyword in body2_name for keyword in ignored_keywords):
            continue

        # print (f"⚠️🔍 Detected collision between '{body1_name}' and '{body2_name}'")

        # If a specific list of objects is provided, filter collisions.
        # We only care if at least one of the colliding bodies is in our interest list.

        if interest_set:
            if body1_name not in interest_set and body2_name not in interest_set:
                continue

        # To avoid duplicate pairs like (A, B) and (B, A), we sort the names.
        pair = tuple(sorted((body1_name, body2_name)))
        if pair not in colliding_pairs:
            colliding_pairs.append(pair)

    return colliding_pairs


def is_scene_stable_v0(env, velocity_threshold=0.01, angular_threshold=0.08, objects_of_interest: Optional[List[str]] = None):
    """
    Checks if movable objects are stable by checking for near-zero velocity. (Legacy)

    This function determines stability by checking if the linear and angular velocities of
    each object are below a given threshold. It is a stateless, instantaneous check that
    is robust for most post-placement stability evaluations.

    It is generally more reliable than the position-based check (`is_scene_stable`) for
    verifying stability immediately after a scene modification because it is less sensitive
    to the minor physics jitter present in all simulations.
    
    Args:
        env: The simulation environment instance.
        velocity_threshold (float): The maximum linear speed (m/s) for an
                                     object to be considered stable.
        angular_threshold (float): The maximum angular speed (rad/s) for an
                                    object to be considered stable.
        objects_of_interest (Optional[List[str]]): A list of object body names to check.
            If None, all non-static objects are checked.

    Returns:
        bool: True if the scene is stable, False otherwise.
    """
    sim = env.sim
    interest_set = set(objects_of_interest) if objects_of_interest else None

    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        
        # If we have a specific list, only check objects in that list.
        if interest_set and body_name not in interest_set:
            continue

        # We only care about movable objects, not the robot or static parts
        # of the world like the floor or the table itself. This logic
        # assumes movable objects don't have these keywords in their names.
        if (body_name is None or 
            'robot' in body_name or 
            'gripper' in body_name or 
            'floor' in body_name or 
            'table' in body_name):
            continue

        # MuJoCo's `cvel` provides the velocity of the body's center of mass.
        # It's a 6D vector: [vx, vy, vz, wx, wy, wz].
        body_vel = sim.data.cvel[i]
        linear_velocity = body_vel[0:3]
        angular_velocity = body_vel[3:6]

        linear_speed = np.linalg.norm(linear_velocity)
        angular_speed = np.linalg.norm(angular_velocity)

        if linear_speed > velocity_threshold or angular_speed > angular_threshold:
            print(f"⚠️  Instability detected: Object '{body_name}' is moving.")
            print(f"   - Linear speed: {linear_speed:.4f} m/s (Threshold: {velocity_threshold})")
            print(f"   - Angular speed: {angular_speed:.4f} rad/s (Threshold: {angular_threshold})")
            return False

    if interest_set:
        print(f"✅ Objects {list(interest_set)} are stable.")
    else:
        print("✅ Scene is stable. All objects are at rest.")
    return True


def is_scene_stable(env, prev_layout_state=None, threshold=0.08):
    """
    Checks for stability by comparing object positions between two time steps.

    This is a stateful function that determines stability by measuring how far
    objects have moved since the last check.

    WARNING: This method is highly sensitive to the `threshold` value. A threshold
    that is too strict (e.g., < 0.005) may incorrectly report instability on a
    settled scene due to microscopic physics simulation jitter. A threshold that
    is too lenient may fail to detect genuinely unstable objects. For immediate,
    one-off checks, `is_scene_stable_v0` is often more reliable.

    Args:
        env: simulation environment (with .sim)
        prev_layout_state (dict): previous step's {body_name: np.array([x, y, z])}
        threshold (float): position change threshold (m)

    Returns:
        (bool, dict): (is_stable, layout_state)
            - is_stable: True if all movable bodies moved less than threshold
            - layout_state: current {body_name: np.array([x, y, z])}
    """
    sim = env.sim
    layout_state = {}
    is_stable = True

    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        if (
            body_name is None
            or "robot" in body_name
            or "gripper" in body_name
            or "floor" in body_name
            or "table" in body_name
        ):
            continue

        pos = sim.data.body_xpos[i].copy()
        layout_state[body_name] = pos

        # 若存在上一帧状态，则比较位置差
        if prev_layout_state and body_name in prev_layout_state:
            delta = np.linalg.norm(pos - prev_layout_state[body_name])
            if delta > threshold:
                print(f"[⚠️Not Stable⚠️] '{body_name}' moved {delta:.6f} m (> {threshold})")
                is_stable = False

    return is_stable, layout_state

def settle_and_check_stability(env, steps=20, **kwargs):
    """
    Steps the simulation for a given number of steps to allow objects to settle,
    then checks if the scene is stable.

    Args:
        env: The simulation environment instance.
        steps (int): The number of simulation steps to wait. With a default
                     control_freq of 20Hz, 20 steps is 1.0 second.
        **kwargs: Arguments to pass to `is_scene_stable` (e.g., velocity_threshold).

    Returns:
        bool: True if the scene is stable after settling, False otherwise.
    """
    print(f"Simulating for {steps} steps to let objects settle...")
    for _ in range(steps):
        env.sim.step()
    
    return is_scene_stable_v0(env, **kwargs)


def are_all_objects_stable(env, **kwargs):
    """
    Checks if ALL movable objects in the scene are stable.

    This is a convenience wrapper around is_scene_stable that explicitly checks
    every non-static, non-robot object in the entire scene. It is useful
    when you need to confirm the stability of the whole environment, not just
    a single object you may have placed.

    Args:
        env: The simulation environment instance.
        **kwargs: Arguments to pass to `is_scene_stable` (e.g., velocity_threshold).

    Returns:
        bool: True if all objects are stable, False otherwise.
    """
    # By passing objects_of_interest=None, is_scene_stable defaults to checking all movable bodies.
    return is_scene_stable_v0(env, objects_of_interest=None, **kwargs)


def check_stability_with_position(env, settle_steps=10, check_steps=10, threshold=0.01):
    """
    Checks for stability by running the simulation and ensuring object positions do not change
    significantly between consecutive steps. This uses the position-based stability check.

    Args:
        env: The simulation environment.
        settle_steps (int): Initial steps to allow the scene to settle.
        check_steps (int): Number of consecutive steps to monitor for stability.
        threshold (float): The maximum allowed position change (in meters) between two steps for the scene
                           to be considered stable. This should be lenient enough to ignore minor physics jitter.

    Returns:
        bool: True if the scene is stable for the entire check duration, False otherwise.
    """
    # 1. Let the scene settle for an initial period.
    for _ in range(settle_steps):
        env.sim.step()

    # 2. Get the initial state to start the iterative check.
    is_stable, state = is_scene_stable(env)

    # On its first run, is_scene_stable just returns the state, so is_stable is always True.
    # The real check happens inside the loop.
    for _ in range(check_steps):
        env.sim.step()
        is_stable_this_step, new_state = is_scene_stable(env, prev_layout_state=state, threshold=threshold)
        if not is_stable_this_step:
            return False  # Instability detected
        state = new_state # Update the state for the next comparison
    
    return True # If the loop completes, the scene was stable for all check_steps.


def get_3d_mesh_distance(env, dist_max, robo_part_name, valid_obj_pair, verbose=False):
    """
    使用增大margin进行距离检测，并直接启用nativeccd。

    Args:
            verbose: 是否打印详细信息
            detect_margin: 用于远距离检测的margin值
            collision_threshold: 碰撞阈值，小于此值视为碰撞

    Returns:
            collision_flag: 是否发生碰撞
            collision_info: 碰撞详细信息
            distance_info: 距离信息
    """
    # sim = self.env.env.sim
    sim = env.sim
    model = sim.model
    data = sim.data

    # 获取夹持器的geom ID
    robo_part_id = model.geom_name2id(robo_part_name)

    sim.forward()

    # 获取所有需要检测的物体
    object_ids = [geom_id for geom_id, geom_name in valid_obj_pair]

    distance_info = []
    for object_id in object_ids:
        object_name = model.geom_id2name(object_id)
        object_pos = data.geom_xpos[object_id]
        gripper_pos = data.geom_xpos[robo_part_id]
        central_distance = np.linalg.norm(object_pos - gripper_pos)

        # 使用 mj_geomDistance 计算 mesh 距离
        # fromto = np.zeros(6, dtype=np.float64)  # 6D 数组存储接触点信息
        # # model_native = self.env.sim.model._model  # 原生 MjModel
        # # data_native =self.env.sim.data._data  # 原生 MjData
        #
        # model_native = sim.model._model # 原生 MjModel
        # data_native =sim.data._data   # 原生 MjData
        # mesh_distance = mujoco.mj_geomDistance(model_native, data_native, robo_part_id, object_id, dist_max,
        #                                                 fromto)
        # from mujoco_py import functions as mjfunc
        # mesh_distance = mjfunc.mj_geomDistance(model_native, data_native, robo_part_id, object_id, dist_max,
        #                                        fromto)
        mesh_distance = get_obj2obj_mesh_distance(
            sim, robo_part_id, object_id, dist_max
        )

        distance_info.append(
            {
                "object_name": object_name,
                "robo_part_name": robo_part_name,
                "central_distance": central_distance,
                "mesh_distance": mesh_distance,
            }
        )
    sim.forward()
    return distance_info


def get_obj2obj_mesh_distance(sim, robo_part_id, object_id, max_margin=0.1):
    model = sim.model
    data = sim.data

    # 记录原始 margin
    original_margin = model.geom_margin[robo_part_id].copy()

    # 临时增大 margin，以扩大检测范围
    model.geom_margin[robo_part_id] = max_margin
    sim.forward()

    mesh_distance = None

    for i in range(data.ncon):
        contact = data.contact[i]

        if (contact.geom1 == robo_part_id and contact.geom2 == object_id) or (
            contact.geom2 == robo_part_id and contact.geom1 == object_id
        ):
            mesh_distance = contact.dist
            break

    # 恢复 margin
    model.geom_margin[robo_part_id] = original_margin
    sim.forward()
    return mesh_distance


def get_object_layout_2d(env):
    """
    获取环境中物体的布局信息（不包括地板）

    Args:
            env: OffScreenRenderEnv 实例
    Returns:
            object_layout: 包含物体位置信息的字典
    """
    object_layout = {}
    # 处理所有物体
    # Zhixu: for fixure, it may contain multiple bodies, we can use the biggest one to represent the fixure, or handle it case by case heuristically
    for body_id in range(env.sim.model.nbody):
        body_name = env.sim.model.body_id2name(body_id)

        # 如果不是有效的物体名称，跳过
        if body_name is None:
            continue
        if "table" in body_name.lower():
            continue
        # 检查是否是盒子，适应新的命名格式
        is_box = "_main" in body_name and any(
            body_name.startswith(f"box_{chr(97 + i)}") for i in range(10)
        )

        # Zhixu: We also consider fixture and draw them in the layout
        # # 检查是否是可移动物体或者桌子
        # is_movable = body_name.endswith('_main')
        # is_table = 'table' in body_name.lower()
        #
        # # 我们只关心可移动物体和桌子
        # if not is_movable:
        #     continue

        # 获取body位置
        center_pos = env.sim.data.body_xpos[body_id]

        # 初始化边界
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        # 找到属于这个body的所有geom并计算边界
        found_geoms = False
        for geom_id in range(env.sim.model.ngeom):
            if env.sim.model.geom_bodyid[geom_id] == body_id:
                found_geoms = True
                geom_pos = env.sim.model.geom_pos[geom_id]
                geom_size = env.sim.model.geom_size[geom_id]
                geom_name = env.sim.model.geom_id2name(geom_id)

                # geom在世界坐标系中的位置
                geom_world_pos = center_pos + geom_pos

                # 根据geom类型获取尺寸
                if len(geom_size) >= 3:
                    # 球体或长方体
                    geom_half_x = geom_size[0]
                    geom_half_y = geom_size[1]
                elif len(geom_size) >= 2:
                    # 圆柱体
                    geom_half_x = geom_size[0]
                    geom_half_y = geom_size[0]
                else:
                    print(f"Warning: Unknown geom size for {geom_name}: {geom_size}")
                    continue

                # 计算这个geom的2D边界（交换X和Y坐标）
                geom_min_x = geom_world_pos[1] - geom_half_x  # 使用Y作为X
                geom_max_x = geom_world_pos[1] + geom_half_x
                geom_min_y = geom_world_pos[0] - geom_half_y  # 使用X作为Y
                geom_max_y = geom_world_pos[0] + geom_half_y

                # 更新物体总边界
                min_x = min(min_x, geom_min_x)
                max_x = max(max_x, geom_max_x)
                min_y = min(min_y, geom_min_y)
                max_y = max(max_y, geom_max_y)

        # 如果找到了有效的边界，存储信息
        if found_geoms and min_x < max_x and min_y < max_y:
            # 存储物体布局信息
            object_layout[body_name] = {
                "center": [center_pos[1], center_pos[0]],  # 交换X和Y
                "min": [min_x, min_y],
                "max": [max_x, max_y],
                "size": [max_x - min_x, max_y - min_y],
            }
    return object_layout


def draw_object_layout_2d(object_layout, save_path):
    """
    可视化物体布局（2D矩形图）
    Args:
            object_layout: dict, {obj_name: {"min": [x_min,y_min], "max":[x_max,y_max], "size":[w,h]}}
            save_path: 图片保存路径 (必须是 .png)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not object_layout:
        print("No objects to plot")
        return

    # 计算范围
    all_mins = [v["min"] for v in object_layout.values()]
    all_maxs = [v["max"] for v in object_layout.values()]
    min_x = min(m[0] for m in all_mins)
    min_y = min(m[1] for m in all_mins)
    max_x = max(m[0] for m in all_maxs)
    max_y = max(m[1] for m in all_maxs)

    # 画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.2)

    # 画每个物体
    for obj_name, bounds in object_layout.items():
        rect = patches.Rectangle(
            bounds["min"],
            bounds["size"][0],
            bounds["size"][1],
            linewidth=2,
            edgecolor="red",
            facecolor="lightblue",
            alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            bounds["min"][0] + bounds["size"][0] / 2,
            bounds["min"][1] + bounds["size"][1] / 2,
            obj_name,
            fontsize=9,
            ha="center",
            va="center",
            weight="bold",
        )

    # 设置范围和比例
    margin = 0.1
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("2D Object Layout")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Layout saved to {save_path}")


def find_free_position(
    layout: Layout,
    table_bounds=(-0.25, -0.25, 0.25, 0.25),
    max_attempts=100,
    margin=0.05,
) -> Tuple[float, float]:
    """
    Find a random free position that doesn't collide with existing objects.

    Args:
            layout: Current layout with existing objects
            table_bounds: (x_min, y_min, x_max, y_max) bounds for valid positions
            max_attempts: Maximum number of random position attempts
            margin: Minimum distance to keep from existing objects

    Returns:
            (x, y) position tuple
    """
    existing_objects = []
    for obj_name, pos in layout.object_init_position.items():
        obj_size = get_object_size(obj_name)
        if obj_size and len(obj_size) >= 2:
            size_x, size_y = float(obj_size[0]), float(obj_size[1])
        else:
            size_x, size_y = 0.05, 0.05  # Default size

        existing_objects.append(
            {
                "name": obj_name,
                "min": (pos[0] - size_x / 2, pos[1] - size_y / 2),
                "max": (pos[0] + size_x / 2, pos[1] + size_y / 2),
            }
        )

    for attempt in range(max_attempts):
        # Random position within bounds
        x = random.uniform(table_bounds[0] + 0.05, table_bounds[2] - 0.05)
        y = random.uniform(table_bounds[1] + 0.05, table_bounds[3] - 0.05)

        # Check if this position is collision-free
        test_obj_size = (0.05, 0.05)  # Assume default object size
        new_min = (x - test_obj_size[0] / 2, y - test_obj_size[1] / 2)
        new_max = (x + test_obj_size[0] / 2, y + test_obj_size[1] / 2)

        if not collision_detection_2d(
            new_min, new_max, existing_objects, margin=margin
        ):
            return (x, y)

    # If all attempts fail, return a position anyway (will be caught by collision detection)
    return (
        random.uniform(table_bounds[0], table_bounds[2]),
        random.uniform(table_bounds[1], table_bounds[3]),
    )


def object_locating(
    lan_description: str, layout: Layout, obj_name: str, use_llm: bool = True
) -> Tuple[float, float]:
    """
    Convert natural language spatial description to actual (x, y) coordinates using LLM.

    This is Stage 2 of the two-stage layout generation pipeline. It takes semantic spatial
    descriptions (from Stage 1: obj_pos_description) and converts them to actual coordinates
    while considering existing object positions and collision avoidance.

    Args:
        lan_description: Natural language position constraint 
                        (e.g., "accessible center, place first for bowl-grip")
        layout: Current layout with existing objects
        obj_name: Name of object to position
        use_llm: If True, use GPT-5. If False, fall back to random positioning.

    Returns:
        (x, y) position tuple in table coordinate space

    Examples:
        "accessible center" → (0.0, 0.0)
        "near the basket" → position 0.1m from basket's current location
        "far from workspace" → (-0.20, -0.20) away from main area
    """
    if not use_llm or not lan_description:
        # Fallback: random free position
        print(f"    [object_locating] No LLM or description, using random position")
        return find_free_position(layout)

    try:
        from TaskGen.FMs import ask_gpt5_text
    except ImportError:
        print(f"    [object_locating] Failed to import GPT-5, using random position")
        return find_free_position(layout)

    # Prepare context about existing objects and their positions
    existing_objects_info = {}
    for obj, pos in layout.object_init_position.items():
        try:
            obj_size = get_object_size(obj)
            size = [float(obj_size[0]), float(obj_size[1])]
        except FileNotFoundError as e:
            print(f"    [object_locating] Warning: Unknown object size for '{obj}', using generic default")
            size = [0.05, 0.05]
        
        existing_objects_info[obj] = {
            "position": [float(pos[0]), float(pos[1])],
            "size": size
        }

    # Get size of object being placed
    try:
        obj_size = get_object_size(obj_name)
        obj_dimensions = [float(obj_size[0]), float(obj_size[1])]
    except FileNotFoundError as e:
        print(f"    [object_locating] Warning: Unknown object size for '{obj_name}', using generic default")
        obj_dimensions = [0.05, 0.05]

    # Table bounds (standard LIBERO workspace)
    table_bounds = {"x_min": -0.25, "x_max": 0.25, "y_min": -0.25, "y_max": 0.25}

    # Build LLM prompt
    prompt = f"""You are generating the INITIAL POSITION for an object in a robot manipulation task testing scenario.

CONTEXT:
- This is creating the starting scene layout for a robot task
- The robot will manipulate objects from these initial positions
- We are generating diverse test layouts (same constraints → different valid positions)
- Objects must be reachable and graspable by the robot

Current Scene State:
Object to place: {obj_name}
Object size: {obj_dimensions[0]:.3f}m x {obj_dimensions[1]:.3f}m
Spatial constraint: "{lan_description}"

Existing objects already placed:
{chr(10).join(f"  - {name}: position=({info['position'][0]:.3f}, {info['position'][1]:.3f}), size=({info['size'][0]:.3f}m x {info['size'][1]:.3f}m)" for name, info in existing_objects_info.items()) if existing_objects_info else "  (no objects yet)"}

Workspace bounds: x ∈ [{table_bounds['x_min']}, {table_bounds['x_max']}], y ∈ [{table_bounds['y_min']}, {table_bounds['y_max']}]

YOUR TASK:
Interpret "{lan_description}" and choose collision-free (x, y) coordinates that:
1. **Satisfy the spatial constraint** (accessible/near/far/clustered as specified)
2. **Enable robot manipulation** (reachable, graspable, not too close to edges)
3. **CRITICAL: Avoid collisions.** You MUST maintain at least 0.03m clearance from ALL existing objects. Check your chosen coordinates against the positions and sizes of the objects listed below.
4. **Add testing variety** (choose different valid positions on each generation)

Constraint Interpretation Guidelines:
- "accessible"/"center"/"reachable" → near (0.0, 0.0), but vary exact position within ±0.10m
- "near X" → within 0.05-0.08m of X (close proximity), vary direction (left/right/front/back)
- "far from X" → at least 0.15-0.20m from X (across workspace), choose different corners/edges
- "clustered with X" → very close to X (0.02-0.04m, just beyond collision), vary relative angle
- "left/right side" → x < -0.05 or x > 0.05 respectively, vary y position
- "front/back" → y > 0.05 or y < -0.05 respectively, vary x position

Note: Objects are typically 0.06-0.12m in size, workspace is 0.5m × 0.5m. The minimum clearance is 0.03m. Double-check your final coordinates to ensure they are not too close to any existing objects.

Physical Constraints (MUST satisfy):
- Stay within table bounds with margin for object size
- Maintain at least 0.03m clearance from ALL existing objects. This is the most important rule.
- Position should be reachable by robot (not at extreme edges)

Return ONLY valid JSON (no markdown, no explanation):
{{"x": float, "y": float, "reasoning": str}}

Example: {{"x": 0.05, "y": -0.10, "reasoning": "Accessible center-left, clear of salad_dressing, enables easy grasp"}}"""

    try:
        # Note: GPT-5 doesn't support temperature parameter, but we explicitly request variety in the prompt.
        # The LLM is instructed to choose different valid positions on each run for testing diversity.
        response = ask_gpt5_text(
            instruction=prompt,
            role_instruction="You are a robotics spatial planning assistant generating diverse test layouts for robot manipulation tasks.",
        )

        # Parse JSON response
        import json
        import re

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)

        x, y = float(result['x']), float(result['y'])

        # Validate coordinates are within bounds
        if not (table_bounds['x_min'] <= x <= table_bounds['x_max'] and 
                table_bounds['y_min'] <= y <= table_bounds['y_max']):
            print(f"    [object_locating] LLM coordinates out of bounds, adjusting...")
            x = max(table_bounds['x_min'], min(table_bounds['x_max'], x))
            y = max(table_bounds['y_min'], min(table_bounds['y_max'], y))

        print(f"    [object_locating] LLM placed {obj_name} at ({x:.3f}, {y:.3f}): {result.get('reasoning', '')}")
        return (x, y)

    except Exception as e:
        print(f"    [object_locating] LLM error: {e}, falling back to random position")
        return find_free_position(layout)


def get_exact_pos(description: str, layout: Layout, objects_of_interest: Set[str]) -> Dict[str, Tuple[float, float]]:
    """
    Given a natural language description, and layout, object of interest, determine the exact (x,y) position for an object.
    This function now uses object_locating to convert the description to coordinates.
    """
    positions = {}
    if not description:
        return positions

    for obj_name in objects_of_interest:
        # We assume the description applies to all objects of interest for this preference.
        # A more sophisticated approach might generate descriptions per object.
        pos_xy = object_locating(description, layout, obj_name)
        if pos_xy:
            positions[obj_name] = pos_xy
    return positions


def get_low_level_pos_description(pref) -> str:
    """
    Generates a spatial layout description heuristically from a low-level motion preference.
    This is a faster, more deterministic alternative to using an LLM.
    """
    pref_type = pref.preference_type

    try:
        if pref_type == "distance_to_object":
            metric = getattr(pref, 'distance_metric', 'less_than')
            # Heuristic: position objects to make the distance constraint meaningful.
            if metric == 'greater_than':
                return "place at the edge of the workspace"
            else: # less_than or other metrics
                return "place in a central, easily accessible location"

        elif pref_type == "speed_range":
            # Heuristic: To test speed constraints, place the object in a slightly
            # cluttered area to encourage more deliberate movements.
            return "place near other objects to create a moderately crowded area"

        elif pref_type in ["on_top", "in", "in_contact","left_to","right_to"]:
            # Heuristic: For relational placements, put the two objects near each other.
            # The 2D layout generator can't handle verticality, so "near" is the best we can do.
            obj1 = getattr(pref, 'obj1', '')
            obj2 = getattr(pref, 'obj2', '')
            if obj1 and obj2:
                # The description is for positioning one of the objects, so we make it relative to the other.
                return f"place near {obj2}"
            return "place in an accessible central location"

        elif pref_type in ["opened", "closed"]:
            # Heuristic: Place the object with the joint in a way that it's easy to access.
            return "place in a central, unobstructed location with its front facing the robot"

        elif pref_type == "distance_to_area":
            # Heuristic: Place relevant objects inside the specified area.
            return "place inside the designated target area"

        elif pref_type == "distance_to_point":
            # Heuristic: Place relevant objects near the specified point.
            return "place near the designated target point"

    except AttributeError as e:
        print(f"[get_low_level_pos_description] Warning: Could not apply heuristic for {pref_type}: {e}")

    return "" # Default for unhandled preferences










def gen_preference_layouts(layout: Layout, task: TaskPlan,layout_id, bddl_output_folder,gen_pos4high_pref:bool=False,enable_collision_check=None) -> Layout:
    """
    Generate personalized layouts based on high-level task preferences.

    This function takes a naive/basic layout and modifies it to accommodate high-level
    preferences specified in the task. It determines which objects are needed and where
    they should be positioned to enable preference satisfaction.

    1. NAIVE LAYOUT (Input):
    - Simple scene from existing LIBERO BDDL files
    - Example: soup, basket, and a few distractor objects on a table
    - Task: "Pick the soup and place it in the basket"

    2. HIGH-LEVEL PREFERENCES (Input):
    - Natural language or structured constraints about WHAT to do
    - Controls: object choices, skill sequences, task structure
    - Examples:
            * "Use red_bowl instead of blue_bowl"
            * "Must pick objects in a specific order"
            * "Avoid interacting with milk"
    - Source: TaskPreference class with object/skill/pair preferences

    3. PERSONALIZED LAYOUT (Output):
    - Modified scene with objects and positions that enable preferences
    - Example: Adds red_bowl, positions it accessibly, places milk far away
    - Saved as new BDDL file for simulation


    IMPORTANT LIMITATIONS:
    - Currently only supports ONE instance per object type (e.g., one milk, one bowl).
      Multiple instances of same type (e.g., milk_1, milk_2) are NOT properly handled
      because Layout.object_init_position can only store one position per base type.
    
    TODO: Add support for multi-instance objects in preference-based layout generation.


    Args:
            layout: The base Layout object (typically loaded from a naive BDDL file)
            task: TaskPlan containing high-level preferences that guide layout generation

    Returns:
            Layout: Modified layout with objects and positions that support the preferences

    Flow:
            1. Collect object requirements from high-level preferences
            2. Collect spatial positioning constraints from high-level preferences
            3. Determine actual (x,y) positions for each object (considering constraints & collisions)
            4. Insert new objects and update existing positions in the layout
            5. Save modified layout to BDDL file for simulation

    Example:
            naive_bddl = "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
            layout = Layout(naive_bddl)

            task = TaskPlan("pick the soup")
            high_pref = TaskPreference(task.plans, task.language_instruction)
            high_pref.object_prefs["required"] = ["red_soup", "basket"]
            task.add_high_level_pref(high_pref)

            new_layout = gen_preference_layouts(layout, task)
            # new_layout now contains red_soup at an appropriate position
    """
    # =========================================================================
    # STEP 1: Collect Required Objects from Preferences (and mark new ones)
    # =========================================================================
    object_set: Set[str] = layout.object_set.copy()
    object2add: Set[str] = set()
    specialized_skills_dict= {
    # "pick_from_side": [
    #     {
    #         "pos": "unchanged",
    #         "direction": np.array([0, 1]),#right
    #     },
    #     {
    #         "pos": "unchanged",
    #         "direction": np.array ([0, -1]),#pick from left
    #     },
    #     # {
    #     #     "pos": [0, 0],
    #     #     "direction": np.array([-1, -1]),
    #     # }
    # ],
    # "open_drawer": [
    #     {
    #         "pos": [0, 0], #drawer position
    #     },
    #     {
    #         "pos": [0, 0],
    #     }
    # ],

    }
    if enable_collision_check is None:
        enable_collision_check=ask_choice("Enable collision check for layout generation?", 1)

    specialized_skills=set(specialized_skills_dict.keys())


    # 🔧 辅助函数：解析编号后缀
    def parse_name_and_index (name: str):
        """
        解析对象名和编号:
          'alphabet_soup_3' -> ('alphabet_soup', 3)
          'basket' -> ('basket', None)
        """
        m = re.match (r"^(.*?)(?:_(\d+))?$", name)
        if not m:
            return name, None
        base, idx = m.group (1), m.group (2)
        return base, int (idx) if idx is not None else None


    def merge_objects_from_pref (pref_obj_set: Set [str]):
        """
        Merge objects required by a preference (as a set of names, possibly with indices)
        into the layout object set, and record newly added ones.
        """
        nonlocal object_set, object2add
        for raw_name in pref_obj_set:
            if raw_name in object_set:
                # Already exists in layout
                continue
            else:
                if raw_name not in layout.object_set:

                    object2add.add(raw_name)
                    object_set.add(raw_name)




    for high_pref in task.high_level_prefs:
        if high_pref is None:
            continue
        required_objs = high_pref.obj_of_interest ()  # ✅ Set[str]
        print (f"[Layout Gen] object required by high-level preferences: {required_objs}")
        merge_objects_from_pref (required_objs)


    for low_prefs in task.low_level_prefs_list:
        if not low_prefs:
            continue
        for low_pref in low_prefs:
            required_objs = low_pref.objects_of_interest  # ✅ Set[str]
            print (f"[Layout Gen] object required by low-level preferences: {required_objs}")
            merge_objects_from_pref (required_objs)



    logger.debug(f"[Layout Gen] Final object set: {object_set}")
    logger.debug(f"[Layout Gen] Objects to add or expand: {object2add}")


    # =========================================================================
    # STEP 2: Collect Spatial Positioning Constraints from High-Level Preferences
    # =========================================================================
    """
    determine position with low-level preference requirement first, then add and assign position for all object
    """
    #Zhixu: here we assume no position requirement for high-level preference, so only object of low-level preference is considered
    pos_constraint_map: Dict[str, str] = {} # Holds descriptions, not coordinates

    # if gen_pos4high_pref:
    #     # Process high-level preferences to get spatial descriptions
    #     for high_pref in task.high_level_prefs:
    #         # This method can use an LLM to generate descriptions like:
    #         # {'milk': 'place far away', 'red_bowl': 'place in center'}
    #
    #         descriptions = high_pref.obj_pos_description()
    #         pos_constraint_map.update(descriptions)


    #Zhixu: fix object position for specialized skills, if collision, remove collided object and retry.
    Done=[]
    for high_pref in task.high_level_prefs:
        if high_pref is None:
            continue
        for (skill, obj) in high_pref.single_pref_plan:
            if (skill, obj) in Done:
                continue
            if skill in specialized_skills:
                # For specialized skills, we may need to add extra objects
                add_done = False

                idx = random.randrange (len (specialized_skills_dict [skill]))
                if "pos" in specialized_skills_dict [skill] [idx].keys () and specialized_skills_dict [skill] [idx] ["pos"]=="unchanged":
                    add_done =True
                else:
                    layout.object_remove(obj) # Remove existing instance to re-add at new position later
                while not add_done:
                    if "pos" in specialized_skills_dict[skill][idx].keys():
                        pos_xy=specialized_skills_dict[skill][idx]["pos"]
                        add_done, collision_with =layout.object_add(obj, position_xy=pos_xy, margin=0.03)
                        if not add_done:
                            layout.object_remove (collision_with)
                            object2add.add (collision_with)
                    else:
                        pos_xy= find_free_position (layout)
                        add_done, collision_with =layout.object_add(obj, position_xy=pos_xy, margin=0.03)
                    if add_done:
                        print (f"  ✓ Placed {obj} for skill {skill} at {pos_xy}")

                task.add_skill_param(skill, to_sim_env_name(obj), specialized_skills_dict[skill][idx]) #Save the specialized skill, target object and parameters in task for later
                specialized_skills_dict [skill].pop (idx)
                Done.append((skill,obj))
    task.save_skill_params(filepath=bddl_output_folder)


    #ZHIXU: for fixtures , we do not reposition them
    fixture={"wooden_cabinet_1,flat_stove_1"}

    for low_prefs in task.low_level_prefs_list:
        if not low_prefs:
            continue
        for low_pref in low_prefs:
            # First, try to get exact positions from the preference's own definition
            exact_pos_from_pref = low_pref.obj_position()
            if exact_pos_from_pref:
                pos_constraint_map.update(exact_pos_from_pref)
                print(f"[Layout Gen] Got exact position from low_pref: {exact_pos_from_pref}")
                continue
            if low_pref.pos_description:
                for obj_name in low_pref.objects_of_interest:
                    if obj_name not in fixture:
                        pos_constraint_map[obj_name] = low_pref.pos_description

        # If no exact position, generate a description and find a position
        # if not low_pref.pos_description:
        #     # Generate a new description if one doesn't exist
        #     generated_description = get_low_level_pos_description(low_pref)
        #     if generated_description:
        #         # For low-level, we apply the same description to all its objects of interest
        #         for obj_name in low_pref.objects_of_interest:
        #             pos_constraint_map[obj_name] = generated_description


    print(f"[Layout Gen] Position constraints collected: {pos_constraint_map}")


    # Pass 1: Identify objects that need to be repositioned based on our collected constraints
    objects_to_reposition = []
    for obj in layout.object_set:
        if obj in pos_constraint_map.keys() :
            objects_to_reposition.append(obj)

    no_need2_place=["wooden_cabinet_1_cabinet_middle", "wooden_cabinet_1_cabinet_top","ground_1","white_cabinet_1_cabinet_top","white_cabinet_1_cabinet_middle"]

    for obj_name in objects_to_reposition:
        print(f"  - Will reposition {obj_name} with constraint: '{pos_constraint_map[obj_name]}'")
        layout.object_remove(obj_name) # Remove the existing instance, then we will re-add it at new position

    # Pass 2: Add all necessary objects (new and repositioned)
    # This logic now handles both newly required objects and those we just removed.
    all_objects_to_place = list(object2add) + objects_to_reposition

    rst_layout=None

    collision_free = False
    is_stable=False

    while (not collision_free or not is_stable):
        new_layout = layout.copy ()
        for obj_name in all_objects_to_place:
            if to_sim_env_name(obj_name) in no_need2_place:
                continue
            if obj_name in new_layout.object_set:
                continue
            description = pos_constraint_map.get(obj_name) #return None if not specified
            # pos_xy = object_locating(description, new_layout, obj_name)
            pos_xy = find_free_position (new_layout)

            max_tries = 50
            try_count = 0
            while try_count < max_tries:
                # Use a slightly larger margin to be safe
                add_done,_= new_layout.object_add (obj_name, pos_xy, margin=0.03)
                if add_done:
                    print(f"  ✓ Placed {obj_name} at {pos_xy} using constraint: '{description}'")
                    break
                else:
                    print(f"  ⚠️ 2D Collision detected when placing {obj_name} at {pos_xy}, retrying...")
                    pos_xy = find_free_position(new_layout)
                    try_count += 1
                    if try_count>=max_tries:
                        print(f"  ✗ Failed to place {obj_name} after {max_tries} attempts, placing anyway at last position {pos_xy}")
                        new_layout.object_add(obj_name, pos_xy, margin=0.0,force_insert=True) # Force add without margin

        #last step: ensure layout is static and no-collision when running dummy action
        new_layout.gen_bddl (bddl_folder=bddl_output_folder,layout_id=layout_id)
        env_args = {
        "bddl_file_name": new_layout.bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
        # Use OSC_position control in default
        }
        env = OffScreenRenderEnv(**env_args, ignore_done=True)
        env.reset()
        state=None
        max_test_step=30
        for idx in range(max_test_step):
            obs,_,_,_=env.step(DUMMY_ACTION)
            if idx<=max_test_step/2: #wait to let the object to drop and settle down
                continue
            collision_pair=collision_detection_3d(env)
            is_stable_t,state= is_scene_stable(env,state)
            if not enable_collision_check:
                collision_pair=[]
                is_stable=True
                is_stable_t=True
            if not is_stable_t:
                is_stable=False
                print(f"[Layout Gen]✗✗✗ The layout is not stable, re-generate bddl file.")
                break
            if len(collision_pair)!=0 :
                collision_free = False
                print(f"[Layout Gen]✗✗✗ Collision detected in final check, re-generate bddl file. Collisions: {collision_pair}")
                break
            if  idx==max_test_step-1 and len(collision_pair)==0 and is_stable_t==True:
                print("[Layout Gen]✓✓✓ No collisions and stable detected in final check.")
                collision_free = True
                is_stable=True
                rst_layout=new_layout

        env.close()

    return rst_layout


def create_naive_test_preference(task: TaskPlan):
    """
    NAIVE TEST IMPLEMENTATION: Create a simple test preference that adds red_bowl.

    This is a placeholder that demonstrates the workflow. Real preferences would
    come from user input or LLM generation.

    Args:
            task: TaskPlan to add preferences to
    """
    try:
        from Preference.high_level.task_preference import TaskPreference
    except ModuleNotFoundError:
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        from Preference.high_level.task_preference import TaskPreference

    # Create a simple test preference
    high_pref = TaskPreference(
        required=[],
        prohibited=[],
        order=[],
        pref_description="A naive test preference to add a red bowl.",
        single_pref_plan=[]
    )

    # NAIVE TEST: Always require red_bowl (this simulates preference generation)
    # In real implementation, this would come from analyzing user preferences
    high_pref._test_required_objects = {"red_bowl": 1}
    high_pref._test_position_constraints = {}

    # Monkey-patch the methods to return test data
    def obj_of_interest_test():
        return high_pref._test_required_objects

    def obj_pos_description_test():
        return high_pref._test_position_constraints

    high_pref.obj_of_interest = obj_of_interest_test
    high_pref.obj_pos_description = obj_pos_description_test

    task.add_high_level_pref(high_pref)
    print("[Test] Added naive test preference: requires red_bowl")


if __name__ == "__main__":
    """
    Quick demo of layout generation.
    For full testing, use: python test_pref_layout_gen.py
    """
    print("Running quick demo...")
    print()

    # Quick demonstration
    naive_bddl_file = "../../libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl"
    layout = Layout(naive_bddl_file)
    task = TaskPlan(layout.lan_inst)
    create_naive_test_preference(task)

    personalized_layout = gen_preference_layouts(layout, task)

    if "red_bowl" in personalized_layout.object_init_position:
        print(
            f"✓ Demo successful: red_bowl added at {personalized_layout.object_init_position['red_bowl']}"
        )
    else:
        print("✗ Demo failed: red_bowl not added")

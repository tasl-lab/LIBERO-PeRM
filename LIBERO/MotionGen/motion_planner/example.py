import os
import cv2
import torch
import numpy as np
# from curobo.types.math import Pose
# from curobo.types.robot import JointState

# import transforms3d as t3d
# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
# from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from typing import List
from libero.libero.envs import OffScreenRenderEnv
# from curobo.types.base import TensorDeviceType
# BASE_OFFSET = np.array([0.6, 0.0, 0.0], dtype=np.float32)   # cuRobo vs LIBERO 世界系
# BASE_OFFSET=[ 6.6547894e-01, -4.2187573e-12, -8.1565565e-01]# Zhixu: On table
# BASE_OFFSET=[0.6054789232420629, -4.218757287954844e-12, 0.09634436960918108] #Zhixu: On the floor
# from curobo.wrap.reacher.ik_solver import IKSolver
# import curobo.util_file
from TaskGen.layout_gen.utils import get_object_size_3d


# ========== 关节名字映射表 ==========
JOINT_MAPPING = {
    "panda_joint1": "robot0_joint1",
    "panda_joint2": "robot0_joint2",
    "panda_joint3": "robot0_joint3",
    "panda_joint4": "robot0_joint4",
    "panda_joint5": "robot0_joint5",
    "panda_joint6": "robot0_joint6",
    "panda_joint7": "robot0_joint7",
    "panda_finger_joint1": "gripper0_finger_joint1",
    "panda_finger_joint2": "gripper0_finger_joint2",
}

# ================== 自动标定 offset ==================
def calibrate_offset(libero_env, ref_body="gripper0_eef"):
    """
    自动估计 cuRobo 空间和 LIBERO 空间的 offset。
    (通过 forward kinematics vs simulator 实际 pose)

    参数:
        motion_gen: MotionGen
        libero_env: LIBERO Env
        ref_body: str, body 名称

    返回:
        offset: np.array(3,)
    """
    motion_gen=init_planner()


    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

    # Step 1: 取当前 qpos
    qpos = libero_env.sim.data.qpos.copy()
    libero_joint_names = list(libero_env.sim.model.joint_names)
    q_vals = []
    for i in range(1, 8):
        j_name = JOINT_MAPPING[f"panda_joint{i}"]
        j_id = libero_joint_names.index(j_name)
        q_vals.append(qpos[j_id])

    q_state = JointState.from_position(
        tensor_args.to_device([q_vals]),
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
    )

    # Step 2: cuRobo 前向运动学
    cu_state = motion_gen.compute_kinematics(q_state)
    cu_pos = cu_state.ee_pos_seq[0].cpu().numpy()   # (3,)
    cu_quat = cu_state.ee_quat_seq[0].cpu().numpy() # (4,)

    # Step 3: 模拟器中的实际位置
    li_pos = libero_env.sim.data.get_body_xpos(ref_body).copy()
    li_quat = libero_env.sim.data.get_body_xquat(ref_body).copy()  # [x,y,z,w]

    # Step 4: 计算 offset
    offset = cu_pos-li_pos

    # 打印对比
    print("\n=== Gripper Pose 对齐检查 ===")
    print(f"[cuRobo] pos: {cu_pos.round(4)}, quat (qw,qx,qy,qz): {cu_quat.round(4)}")
    print(f"[LIBERO] pos: {li_pos.round(4)}, quat (x,y,z,w):    {li_quat.round(4)}")
    print(f"估计 offset: {offset.round(4)}\n")
    return offset

def generate_obstacles(env, obst_name_list):
    obstacles = []

    for obst_name in obst_name_list:
        obst_pos = env.sim.data.get_body_xpos(obst_name).copy()
        body_id = env.sim.model.body_name2id(obst_name)
        #ZHixu: here is a bug, body id is not geom id
        # obst_size = env.sim.model.geom_size[body_id]
        obst_size = get_object_size_3d(env, obst_name)

        scale_const=1.5
        obstacles.append({
            "pos": obst_pos,
            "size": np.array([obst_size[0]/2*scale_const, obst_size[1]/2*scale_const, obst_size[2]/2*scale_const]),  # half size
        })
    print(f"Generated {len(obstacles)} obstacles from env. They are:")
    for idx, obst in enumerate(obstacles):
        print(f"  Obstacle {idx+1}: pos {obst['pos']}, size {obst['size']*2}")
    return obstacles





def get_body_size_3d_v0(env, body_name: str, verbose=False):
    """
    计算 MuJoCo body 的 3D AABB（支持 box 和 mesh）。
    返回:
        {
            "aabb_min": np.ndarray(3,),
            "aabb_max": np.ndarray(3,),
            "center":   np.ndarray(3,),
            "size":     np.ndarray(3,),   # full size
            "geom_count": int
        }
    """
    model, data = env.sim.model, env.sim.data
    bid = model.body_name2id(body_name)
    start = model.body_geomadr[bid]
    count = model.body_geomnum[bid]

    if count == 0:
        raise ValueError(f"Body '{body_name}' has no geoms.")

    MJ_GEOM_BOX  = 6
    MJ_GEOM_MESH = 7

    aabb_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    aabb_max = -aabb_min
    used = 0

    for gid in range(start, start + count):
        gtype = int(model.geom_type[gid])
        gname = model.geom_id2name(gid)

        # === BOX ===
        if gtype == MJ_GEOM_BOX:
            gpos = data.geom_xpos[gid]
            gmat = data.geom_xmat[gid].reshape(3, 3)
            gsize = model.geom_size[gid]  # half-lengths
            half_world = np.abs(gmat) @ gsize
            gmin = gpos - half_world
            gmax = gpos + half_world

        # === MESH ===
        elif gtype == MJ_GEOM_MESH:
            mesh_id = model.geom_dataid[gid]
            if mesh_id < 0:
                raise RuntimeError(f"Geom '{gname}' has invalid mesh id.")

            vadr = model.mesh_vertadr[mesh_id]
            vnum = model.mesh_vertnum[mesh_id]
            verts = model.mesh_vert[vadr:vadr + vnum, :]

            scale = model.geom_size[gid].copy()
            if hasattr(model, "mesh_scale"):
                try:
                    extra_scale = model.mesh_scale[mesh_id]
                    if not np.allclose(extra_scale, 1.0):
                        scale *= extra_scale
                except Exception:
                    pass

            gpos = data.geom_xpos[gid]
            gmat = data.geom_xmat[gid].reshape(3, 3)
            verts_scaled = verts * scale[None, :]
            verts_world = (verts_scaled @ gmat.T) + gpos[None, :]
            gmin = np.min(verts_world, axis=0)
            gmax = np.max(verts_world, axis=0)

            if verbose:
                print(f"AABB for mesh '{gname}': min={np.round(gmin, 4)}, max={np.round(gmax, 4)}")

        else:
            raise NotImplementedError(
                f"Only box and mesh geoms are supported; found type {gtype} ({gname})"
            )

        aabb_min = np.minimum(aabb_min, gmin)
        aabb_max = np.maximum(aabb_max, gmax)
        used += 1

    if used == 0:
        raise RuntimeError(f"Body '{body_name}' has no usable geoms.")

    center = 0.5 * (aabb_min + aabb_max)
    size = aabb_max - aabb_min  # full size

    if not np.isfinite(size).all():
        raise RuntimeError(f"Invalid AABB computed for '{body_name}'.")

    return {
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "center": center,
        "size": size,
        "geom_count": used,
    }


def get_body_size_3d_v1(env, body_name: str, verbose=False):
    model, data = env.sim.model, env.sim.data
    MJ_GEOM_BOX, MJ_GEOM_MESH = 6, 7

    def collect_body_ids(bid: int):
        """递归收集当前body及其子body的id"""
        ids = [bid]
        for cid in range(model.nbody):
            if model.body_parentid[cid] == bid:
                ids.extend(collect_body_ids(cid))
        return ids

    bid_root = model.body_name2id(body_name)
    body_ids = collect_body_ids(bid_root)

    aabb_min = np.array([np.inf, np.inf, np.inf])
    aabb_max = -aabb_min
    used = 0

    for bid in body_ids:
        start = model.body_geomadr[bid]
        count = model.body_geomnum[bid]
        for gid in range(start, start + count):
            gtype = int(model.geom_type[gid])
            gname = model.geom_id2name(gid)

            if gtype == MJ_GEOM_BOX:
                # print("box: ", gname)
                gpos = data.geom_xpos[gid]
                gmat = data.geom_xmat[gid].reshape(3, 3)
                gsize = model.geom_size[gid]
                half_world = np.abs(gmat) @ gsize
                gmin = gpos - half_world
                gmax = gpos + half_world

            elif gtype == MJ_GEOM_MESH:
                # print("mesh: ", gname)
                mesh_id = model.geom_dataid[gid]
                vadr = model.mesh_vertadr[mesh_id]
                vnum = model.mesh_vertnum[mesh_id]
                verts = model.mesh_vert[vadr:vadr + vnum, :]
                scale = model.geom_size[gid].copy()
                gpos = data.geom_xpos[gid]
                gmat = data.geom_xmat[gid].reshape(3, 3)
                verts_scaled = verts * scale[None, :]
                verts_world = (verts_scaled @ gmat.T) + gpos[None, :]
                gmin = np.min(verts_world, axis=0)
                gmax = np.max(verts_world, axis=0)

            else:
                continue

            aabb_min = np.minimum(aabb_min, gmin)
            aabb_max = np.maximum(aabb_max, gmax)
            used += 1

            if verbose:
                print(f"[{gname}] min={np.round(gmin, 4)}, max={np.round(gmax, 4)}")

    if used == 0:
        raise RuntimeError(f"No geoms found under '{body_name}' or its children.")

    center = 0.5 * (aabb_min + aabb_max)
    size = aabb_max - aabb_min

    return {
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "center": center,
        "size": size,
        "geom_count": used,
    }


from typing import Any, Dict
import numpy.typing as npt
def get_gripper_pose_in_curobo(ik_solver,obs: Dict[str, Any]) -> npt.NDArray[np.float64]:
    """
    Get the current end-effector (gripper) pose in the cuRobo robot frame.

    We use cuRobo's forward kinematics (FK) to compute the EEF pose in the robot base frame.
    This avoids depending on LIBERO's world frame proprioception, whose world-to-robot
    transform varies across scenes.

    Args:
        obs (dict): Observation dictionary from LIBERO simulator,
                    expected to include 'robot0_joint_pos' (7-DOF joint angles)

    Returns:
        gripper_pose: np.ndarray of shape (7,)
                      [x, y, z, rx, ry, rz, gripper_state]
    """
    # === 1️⃣ 取当前关节角 ===
    current_joint_pos = obs['robot0_joint_pos']

    # === 2️⃣ 转换为 cuRobo Tensor 输入 ===
    current_joint_states = (
        torch.tensor(current_joint_pos)
        .reshape(1, 7)
        .to(ik_solver.tensor_args.device)
        .float()
    )

    # === 3️⃣ 使用 cuRobo 的 FK 求 EEF 位姿 ===
    current_eef_states = ik_solver.fk(current_joint_states)

    # === 4️⃣ 提取位置 + 姿态（四元数→欧拉角） ===
    ee_pos = current_eef_states.ee_position[0].cpu().numpy()
    ee_quat = current_eef_states.ee_quaternion[0].cpu().numpy()
    ee_euler = t3d.euler.quat2euler(ee_quat, axes='sxyz')

    # === 5️⃣ 拼接 gripper 状态 ===
    gripper_state = np.array([-1])
    gripper_pose = np.concatenate([ee_pos, ee_euler, gripper_state])

    # === 6️⃣ 调试打印（可选） ===
    # print(f"[cuRobo FK] EEF position: {np.round(ee_pos, 4)}")
    # print(f"[cuRobo FK] EEF euler: {np.round(ee_euler, 4)}")
    # print(f"[cuRobo FK] Gripper state: {gripper_state[0]}")

    return gripper_pose



# ================== 新增: 构建 world (含障碍物) ==================
def build_world_config(BASE_OFFSET,obstacles):
    # 在 LIBERO 系中定义障碍物

    # alphabet_soup_pos = np.array([-0.12 , -0.24 ,  0.015])
    # alphabet_soup_size=[0.08795146*2, 0.0967818*2 , 0.13696476*2]
    #
    # obstacle_2_pos_libero = np.array([ 0.05 , 0  ,  0.035])   # ✅ 你可以改这里
    # obstacle_2_dims = [0.05502723*2, 0.07550338*2, 0.14811387*2]  # x,y,z (长宽高)
    # obstacle_pos_libero = alphabet_soup_pos
    # obstacle_dims = alphabet_soup_size

    cuboid_obsts={}
    for idx, obst in enumerate(obstacles):
        pos=obst["pos"]+ BASE_OFFSET
        cuboid_obsts[f"obstacle_{idx+1}"]={
            "dims": obst["size"]*2,   # x,y,z (长宽高)
            "pose": [*pos, 1, 0, 0, 0],  # [x,y,z,qw,qx,qy,qz], in default, no rotation
        }




    # 转换到 cuRobo 系
    # obstacle_pos_curobo = obstacle_pos_libero + BASE_OFFSET
    # obstacle_pose = [*obstacle_pos_curobo, 1, 0, 0, 0]  # [x,y,z,qw,qx,qy,qz]
    # world_config = {
    #     "cuboid": {
    #         "obstacle_1": {
    #             "dims": cuboid_obsts["obstacle_1"]["dims"],
    #             "pose": cuboid_obsts["obstacle_1"]["pose"],
    #         },
    #         "obstacle_2": {
    #             "dims": cuboid_obsts["obstacle_2"]["dims"],
    #             "pose": cuboid_obsts["obstacle_2"]["pose"]
    #         },
    #     }
    # }
    for name, obst in cuboid_obsts.items():
        print(f"{name}: dims {obst['dims']}, pose {obst['pose']}")
    world_config = {
        "cuboid": {
            name: {
                "dims": obst["dims"],
                "pose": obst["pose"]
            }
            for name, obst in cuboid_obsts.items()
        }
    }

    return world_config

# ================== 初始化 cuRobo ==================
def init_planner(base_offset:List[int]=None,obstacles=None):
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    if obstacles is None or len(obstacles) == 0:
        world_config="collision_table.yml"
        cfg = MotionGenConfig.load_from_robot_config(
            "franka.yml",
            # world_config,  # ✅ 这里直接传 dict
            "collision_table.yml",
            tensor_args,
            interpolation_dt=0.05,
            num_ik_seeds=20,
            num_trajopt_seeds=4,
            use_cuda_graph=True
        )
    else:
        world_config = build_world_config(base_offset,obstacles)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(current_dir, "../../franka_with_extended_finger/franka.yml")
        print("world_config:", world_config)
        cfg = MotionGenConfig.load_from_robot_config(
            curobo.util_file.load_yaml (cfg_file) ["robot_cfg"],#"franka.yml",
            world_config,    # ✅ 这里直接传 dict
            # "collision_table.yml",
            tensor_args,
            interpolation_dt=0.05,
            num_ik_seeds=20,
            num_trajopt_seeds=4,
            use_cuda_graph=True,
        )

    mg = MotionGen(cfg)
    mg.warmup()
    return mg


def plan_relative_osc_pose(env, env2, motion_gen, target_xpos, gripper_val=-1):

    if motion_gen is None:
        raise ValueError("motion_gen cannot be None, when use_motion_gen flag is set to True.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(current_dir, "../../franka_with_extended_finger/franka.yml")
    robot_cfg = curobo.util_file.load_yaml (cfg_file) [
        "robot_cfg"]
    ik_solver = IKSolver (IKSolver.load_from_robot_config (
        robot_cfg,
        None,
        TensorDeviceType (),
    ))

    from MotionGen.heuristic_skill.heuristic import move_to
    from scipy.spatial.transform import Rotation as R

    gripper_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
    print("gripper_pos:", gripper_pos)

    obs1,_,_,_=env.step([0,0,0,0,0,0,-1]) # move the robot to see the target object clearly

    eef_id = env.sim.model.body_name2id ("gripper0_eef")
    eef_quat = env.sim.data.body_xquat [eef_id]
    eef_quat_xyzw = np.array ([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
    target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")

    env2.reset()
    actions=move_to (env2, target_pos=target_xpos, target_orn=target_orn, speed=1e-3,
             gripper_state=-1)
    obs2=None
    for act in actions:
        obs2, _, _, _ = env2.step (act)
    current_joint_pos = obs2 ['robot0_joint_pos']

    # === 2️⃣ 转换为 cuRobo Tensor 输入 ===
    current_joint_states = (
        torch.tensor (current_joint_pos)
        .reshape (1, 7)
        .to (ik_solver.tensor_args.device)
        .float ()
    )
    # === 3️⃣ 使用 cuRobo 的 FK 求 EEF 位姿 ===
    current_eef_states = ik_solver.fk(current_joint_states)

    # === 4️⃣ 提取位置 + 姿态（四元数→欧拉角） ===
    ee_pos = current_eef_states.ee_position[0].cpu().numpy()
    ee_quat = current_eef_states.ee_quaternion[0].cpu().numpy()
    ee_euler = t3d.euler.quat2euler(ee_quat, axes='sxyz')

    tensor_args = TensorDeviceType (device=torch.device ("cuda:0"))
    goal_pose = Pose (
        position=tensor_args.to_device ([ee_pos]),
        quaternion=tensor_args.to_device ([ee_quat]),
    )
    qpos = env.sim.data.qpos.copy ()
    libero_joint_names = list (env.sim.model.joint_names)
    q_start_vals = []
    for i in range (1, 8):
        j_name = JOINT_MAPPING [f"panda_joint{i}"]
        j_id = libero_joint_names.index (j_name)
        q_start_vals.append (qpos [j_id])
    q_start = JointState.from_position (
        tensor_args.to_device([q_start_vals]),
        joint_names=[f"panda_joint{i}" for i in range (1, 8)],
    )

    print(f"Planing: In curobo space,from:{ get_gripper_pose_in_curobo(ik_solver,obs1)} to {get_gripper_pose_in_curobo(ik_solver,obs2)}")

    rst=motion_gen.plan_single (goal_pose=goal_pose, start_state=q_start)
    if not rst.success:
        print("❌ 规划失败，无法执行相对运动。")
        return
    q_traj=rst.get_interpolated_plan ().position.detach ().cpu ().numpy ()

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

    print("\n=== cuRobo Relative 执行开始 ===")
    cu_pos_prev = None

    actions = []
    for t, q in enumerate(q_traj):
        q_js = JointState.from_position(
            tensor_args.to_device([q.tolist()]),
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        )
        state = motion_gen.compute_kinematics(q_js)
        cu_pos = state.ee_pos_seq[0].cpu().numpy()

        if cu_pos_prev is not None:
            diff = cu_pos - cu_pos_prev
            action = np.concatenate([diff * 80.0, np.zeros(3), [gripper_val]])

            max_val = np.max(np.abs(action[:-1]))
            if max_val > 1.0:
                half_action = np.concatenate([action[:-1] / 2.0, [gripper_val]])
                actions.append(half_action)
                actions.append(half_action)
                # cu_gripper_pose=get_gripper_pose_in_curobo (ik_solver=ik_solver, obs=obs)
            else:
                actions.append(action)
                # cu_gripper_pose=get_gripper_pose_in_curobo (ik_solver=ik_solver, obs=obs)

        cu_pos_prev = cu_pos
        
    return actions

def get_object_world_bbox (env, body_name: str):
    model = env.sim.model
    data = env.sim.data

    # 1️⃣ 获取 body id
    bid = model.body_name2id (body_name)

    # 2️⃣ 递归查找子 body
    def collect_all_child_bodies (bid):
        children = []
        for i, parent in enumerate (model.body_parentid):
            if parent == bid:
                children.append (i)
                children.extend (collect_all_child_bodies (i))
        return children

    # 当前 body + 所有子 body
    all_bodies = [bid] + collect_all_child_bodies (bid)

    # 3️⃣ 收集所有 geom
    geom_ids = []
    for b in all_bodies:
        start = model.body_geomadr [b]
        count = model.body_geomnum [b]
        geom_ids.extend (range (start, start + count))

    if not geom_ids:
        raise ValueError (f"Body '{body_name}' has no geom in itself or children")

    # 4️⃣ 获取每个 geom 的世界坐标位置和尺寸
    geom_pos = data.geom_xpos [geom_ids]
    geom_size = model.geom_size [geom_ids]

    # 5️⃣ 计算整体包围盒
    bbox_min = (geom_pos - geom_size).min (axis=0)
    bbox_max = (geom_pos + geom_size).max (axis=0)
    bbox_size = bbox_max - bbox_min

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size": bbox_size,
        "geom_count": len (geom_ids),
    }


# ================== 执行：相对控制 ==================
def run_relative_osc_pose(env, env2,motion_gen,target_obj_name, gripper_val=0, save_path="motion_planner_output.mp4"):
    robot_cfg = curobo.util_file.load_yaml ("../../franka_with_extended_finger/franka.yml") [
        "robot_cfg"]
    ik_solver = IKSolver (IKSolver.load_from_robot_config (
        robot_cfg,
        None,
        TensorDeviceType (),
    ))
    gripper_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
    print("gripper_pos:", gripper_pos)
    target_pos=env.sim.data.get_body_xpos(target_obj_name)
    target_pos = target_pos.copy()



    from MotionGen.heuristic_skill.heuristic import move_to
    from scipy.spatial.transform import Rotation as R

    obs1,_,_,_=env.step([0,0,0,0,0,0,-1]) # move the robot to see the target object clearly

    idx = env.sim.model.body_name2id (target_obj_name)
    target_xpos = env.sim.data.body_xpos [idx]
    if target_obj_name=="basket_1_main":
        target_xpos[2] += 0.3  # 抬高一点
    else:
        target_xpos[2] += 0.05  # 抬高一点
    gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
    eef_id = env.sim.model.body_name2id ("gripper0_eef")
    eef_quat = env.sim.data.body_xquat [eef_id]
    eef_quat_xyzw = np.array ([eef_quat [1], eef_quat [2], eef_quat [3], eef_quat [0]])
    target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")
    env2.reset()
    actions=move_to (env2, target_pos=target_xpos, target_orn=target_orn, speed=1e-3,
             gripper_state=-1)
    obs2=None
    for act in actions:
        obs2, _, _, _ = env2.step (act)

    current_joint_pos = obs2 ['robot0_joint_pos']

    # === 2️⃣ 转换为 cuRobo Tensor 输入 ===
    current_joint_states = (
        torch.tensor (current_joint_pos)
        .reshape (1, 7)
        .to (ik_solver.tensor_args.device)
        .float ()
    )
    # === 3️⃣ 使用 cuRobo 的 FK 求 EEF 位姿 ===
    current_eef_states = ik_solver.fk(current_joint_states)

    # === 4️⃣ 提取位置 + 姿态（四元数→欧拉角） ===
    ee_pos = current_eef_states.ee_position[0].cpu().numpy()
    ee_quat = current_eef_states.ee_quaternion[0].cpu().numpy()
    ee_euler = t3d.euler.quat2euler(ee_quat, axes='sxyz')



    # gripper_state = np.array ([-1])
    # gripper_pose = np.concatenate ([ee_pos, ee_euler, gripper_state])
    tensor_args = TensorDeviceType (device=torch.device ("cuda:0"))
    goal_pose = Pose (
        position=tensor_args.to_device ([ee_pos]),
        quaternion=tensor_args.to_device ([ee_quat]),
    )
    qpos = env.sim.data.qpos.copy ()
    libero_joint_names = list (env.sim.model.joint_names)
    q_start_vals = []
    for i in range (1, 8):
        j_name = JOINT_MAPPING [f"panda_joint{i}"]
        j_id = libero_joint_names.index (j_name)
        q_start_vals.append (qpos [j_id])
    q_start = JointState.from_position (
        tensor_args.to_device ([q_start_vals]),
        joint_names=[f"panda_joint{i}" for i in range (1, 8)],
    )

    print(f"Planing: In curobo space,from:{ get_gripper_pose_in_curobo(ik_solver,obs1)} to {get_gripper_pose_in_curobo(ik_solver,obs2)}")

    rst=motion_gen.plan_single (goal_pose=goal_pose, start_state=q_start)
    #Doing planing
    q_traj=rst.get_interpolated_plan ().position.detach ().cpu ().numpy ()
    # dxyz = target_pos - gripper_pos
    # # dxyz=[0,-0.1,0]
    # print("需要移动的 Δpos:", dxyz)
    # q_traj = plan_delta_pose(motion_gen, env, BASE_OFFSET=base_offset, dx=dxyz[0], dy=dxyz[1], dz=dxyz[2],goal_pose=gripper_pose)
    if q_traj is None:
        print("❌ 规划失败，无法执行相对运动。")
        return
    h, w = 256, 256
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

    print("\n=== cuRobo Relative 执行开始 ===")
    cu_pos_prev = None


    for t, q in enumerate(q_traj):
        q_js = JointState.from_position(
            tensor_args.to_device([q.tolist()]),
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        )
        #Forward dynamics1
        state = motion_gen.compute_kinematics(q_js)
        cu_pos = state.ee_pos_seq[0].cpu().numpy()

        if cu_pos_prev is not None:
            diff = cu_pos - cu_pos_prev
            action = np.concatenate([diff * 80.0, np.zeros(3), [gripper_val]])

            max_val = np.max(np.abs(action[:-1]))
            if max_val > 1.0:
                half_action = np.concatenate([action[:-1] / 2.0, [gripper_val]])
                obs, _, _, _ = env.step(half_action)
                obs, _, _, _ = env.step(half_action)

                #more precise forward kinematics
                cu_gripper_pose=get_gripper_pose_in_curobo (ik_solver=ik_solver, obs=obs)
                #这样更新cu_pos，替换到line 568的motion_gen.compute_kinematics(q_js)
                #cu_pos=cu_gripper_pose[:3]
            else:
                obs, _, _, _ = env.step(action)
                cu_gripper_pose=get_gripper_pose_in_curobo (ik_solver=ik_solver, obs=obs)
                # cu_pos=cu_gripper_pose[:3]

            frame = obs["agentview_image"][::-1, ::-1]
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            li_pos = env.sim.data.get_body_xpos("gripper0_eef").copy()
            print(f"[cu_pos: {cu_pos}| cu_gripper_pose: {cu_gripper_pose} | libero_pos: {li_pos}]")


        cu_pos_prev = cu_pos


    video_writer.release()
    print(f"\n🎉 执行结束！视频已保存到 {save_path}")


def approach(env, env2, motion_planner, target_name, gripper_val=-1):
    target_xpos = env.sim.data.get_body_xpos(target_name).copy()
    target_size = get_object_size_3d(env, target_name)
    aabb_max = target_size["aabb_max"]
    aabb_min = target_size["aabb_min"]
    gripper_xpos = env.sim.data.get_body_xpos("gripper0_eef").copy()
    print(f"Target '{target_name}' AABB min: {aabb_min}, max: {aabb_max}")
    print(f"Gripper position: {gripper_xpos}")
    mid_xpos = (aabb_min + aabb_max) / 2
    target_xpos = mid_xpos
    target_xpos[2] = aabb_max[2] + 0.1  # 抬高一点
    target_xpos[0] = np.clip(gripper_xpos[0], aabb_min[0]-0.05, aabb_max[0]+0.05)
    target_xpos[1] = np.clip(gripper_xpos[1], aabb_min[1]-0.05, aabb_max[1]+0.05)
    print(f"Approach target position: {target_xpos}")
    actions = plan_relative_osc_pose(env, env2, motion_planner, target_xpos, gripper_val=gripper_val)
    return actions
 


if __name__ == "__main__":
    env_args = {
        # "bddl_file_name": "/home/jiachenl/Safe_DP/Dita/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
        # "bddl_file_name":"../../libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
        "bddl_file_name":"/home/jiachenl/P_Bench/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate.bddl",
        "camera_heights": 256, "camera_widths": 256,
        "controller":"OSC_POSE"
    }
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset ()
    env2 = OffScreenRenderEnv (**env_args)
    obs2 = env2.reset ()
    # env.sim.model.body_name2id("-1")


    # obst_names = ["basket_1_main","milk_1_main"]
    obst_names = ["wooden_cabinet_1_main"]
    model = env.sim.model
    bid = model.body_name2id("wooden_cabinet_1_base")
    # bid2= model.body_name2id("alphabet_soup_1_main")

    start = model.body_geomadr[bid]
    # start2= model.body_geomadr[bid2]
    count = model.body_geomnum[bid]
    # count2= model.body_geomnum[bid2]
    geom_ids = range(start, start + count)
    # geom_ids2= range(start2, start2 + count2)
    for gid in geom_ids:
        print(f"geom id={gid}, name={model.geom_id2name(gid)}, type={model.geom_type[gid]}, size={model.geom_size[gid]}")
    # for _gid in geom_ids2:
    #     print(f"geom id={_gid}, name={model.geom_id2name(_gid)}, type={model.geom_type[_gid]}, size={model.geom_size[_gid]}")


    print(f"{get_body_size_3d_v0(env, 'wooden_cabinet_1_base')['size']}\n")
    print (f"{get_body_size_3d_v0 (env, 'wooden_cabinet_1_cabinet_top') ['size']}\n")
    print (f"{get_body_size_3d_v0 (env, 'wooden_cabinet_1_cabinet_middle') ['size']}\n")
    print (f"{get_body_size_3d_v0 (env, 'wooden_cabinet_1_cabinet_bottom') ['size']}\n")
    print(f"{get_object_world_bbox(env,'wooden_cabinet_1_main')}\n ")

    print(f"{get_body_size_3d_v0(env,'alphabet_soup_1_main')['size']}\n ")
    print(f"{get_body_size_3d_v0(env,'cream_cheese_1_main')['size']}\n ")
    offset = calibrate_offset(env)
    obstacles = generate_obstacles (env, obst_names)

    motion_planner = init_planner (offset ,obstacles)

    # motion_planner = init_planner (offset )
    # base_xpos = env.sim.data.get_body_xpos ("robot0_base")
    # run_relative_osc_pose (env, env2,motion_planner, target_obj_name="alphabet_soup_1_main",base_offset=offset, save_path="motion_planner_output_1.mp4")
    # run_relative_osc_pose (env,env2,motion_planner,target_obj_name="cream_cheese_1_main",base_offset=offset, save_path="motion_planner_output_2.mp4")
    # run_relative_osc_pose (env,env2,motion_planner,target_obj_name="basket_1_main",base_offset=offset, save_path="motion_planner_output_3.mp4")
    video_writer = cv2.VideoWriter("motion_planner_output_approach.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (256, 256))    

    actions = approach(env, env2, motion_planner, target_name="alphabet_soup_1_main", gripper_val=-1)
    for action in actions:
        obs, _, _, _ = env.step(action)
        frame = obs["agentview_image"][::-1, ::-1]
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    actions = approach(env, env2, motion_planner, target_name="cream_cheese_1_main", gripper_val=-1)
    for action in actions:
        obs, _, _, _ = env.step(action)
        frame = obs["agentview_image"][::-1, ::-1]
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    actions = approach(env, env2, motion_planner, target_name="basket_1_main", gripper_val=0)
    for action in actions:
        obs, _, _, _ = env.step(action)
        frame = obs["agentview_image"][::-1, ::-1]
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    env.close()


    # obst1_name="glazed_rim_porcelain_ramekin_1_main"
    # obst1_pos= env.sim.data.get_body_xpos(obst1_name).copy()
    # body_id = env.sim.model.body_name2id(obst1_name)
    # obst1_size = env.sim.model.geom_size[body_id]#Zhixu: here we use assume this body only has this geom in mujoco, but for some body, it may have multiple geoms
    # print("obst1_size: ", obst1_size)
    # obstacles.append({
    #     "pos": [-0.2,0,0.0],
    #     # "size":  np.array([obst1_size[0], obst1_size[1], 0.1]),#Make the obst higher
    #     "size":  [0.2,0.2,0.2],
    # })

    # obst2_name = "box_h_1_main"
    # obst2_pos = env.sim.data.get_body_xpos(obst2_name).copy()
    # obst2_size = env.sim.model.geom_size[env.sim.model.body_name2id(obst2_name)]
    # print("obst2_size: ", obst2_size)
    # body_id = env.sim.model.body_name2id(obst2_name)
    # obst2_size = env.sim.model.geom_size[body_id]
    # obst2_size[0]=0.05
    # obst2_size[1]=0.05
    # obst2_size[2]=obst1_pos[2]#Make the obst no collision in z axis
    # obstacles.append({
    #     "pos": obst2_pos,
    #     "size": np.array([0.025, 0.025, 0.25]),
    #     # "size":obst2_size,
    # })
    
    # mg = init_planner(obstacles)
    # print("obj_list: ", env.sim.model.body_names)


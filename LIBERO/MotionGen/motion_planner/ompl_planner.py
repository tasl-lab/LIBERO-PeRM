"""
An OMPL Motion Planner
"""
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from TaskGen.task import TaskPlan
from TaskGen.layout_gen.layout import Layout
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import bddl_utils as BDDLUtils
import os
import robosuite
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("EGL_DEVICE_ID", "0")
import cv2
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import ompl.base as ob
import ompl.geometric as og
import ompl.util as ou
ou.setLogLevel(ou.LOG_WARN)

def quat_xyzw_to_wxyz(qxyzw):
    x, y, z, w= qxyzw
    return np.array([w, x, y, z], dtype=float) / np.linalg.norm(qxyzw)

def quat_wxyz_to_xyzw(qwxyz):
    w, x, y, z = qwxyz
    return np.array([x, y, z, w], dtype=float) / np.linalg.norm(qwxyz)

def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2 
    ], dtype=float) / np.linalg.norm(q1) / np.linalg.norm(q2)

def quat_inv_wxyz(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / np.linalg.norm(q)

def quat_to_axis_angle_wxyz(q, eps=5e-4):
    w, x, y, z = q
    if w > 1: q = q / np.linalg.norm(q)
    angle = 2*np.arccos(w)
    s = np.sqrt(max(1-w*w, 0.0))
    if s < eps:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return np.array([x/s, y/s, z/s]), angle

def axis_angle_to_quat_wxyz(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([0,0,0,1], dtype=float)
    axis = axis / n
    s = np.sin(angle/2.0)
    return np.array([np.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)

class PathLengthObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si):
        super(PathLengthObjective, self).__init__(si)
        self.setCostThreshold(ob.Cost(6))


class CustomObjective(ob.OptimizationObjective):
    def __init__(self, si, env, approach_names: Optional[List[str]] = []):
        super().__init__(si)
        self.env = env
        self.approach_names = approach_names if approach_names is not None else []
        self.setCostThreshold(float("inf"))
    
    def combineCosts(self, c1, c2):
        return ob.Cost(c1.value() + c2.value())
    
    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        pos1 = np.array([s1[0][i] for i in range(3)], dtype=float)
        pos2 = np.array([s2[0][i] for i in range(3)], dtype=float)
        cost = ob.Cost(0.0)
        for target in approach_names:
            target_pos = self.env.sim.data.get_body_xpos(target)
            cost = self.combineCosts(cost, self.distCost(pos2, target_pos))
        return cost
    
    def distCost(self, eef_pos: np.ndarray, target_pos: np.ndarray) -> ob.Cost:
        dist = np.linalg.norm(target_pos - eef_pos)
        if dist < 0.05:
            # cost = ob.Cost(0.05 / dist - 1)
            cost = ob.Cost(0.0)
        if dist >= 0.05:
            cost = ob.Cost(dist / 0.05 - 1)
        return cost

class RoundGoalRegion(ob.GoalSampleableRegion):
    def __init__(self, si, goal_pose, radius: float):
        super(RoundGoalRegion, self).__init__(si)
        self.goal_pose = goal_pose
        self.radius = radius

    def distanceGoal(self, state: ob.State) -> float:
        goal_pos = np.array(self.goal_pose["pos"], dtype=float)
        cur_pos = np.array([state[0][i] for i in range(3)], dtype=float)
        dist = np.linalg.norm(cur_pos - goal_pos) - self.radius
        return dist

    def maxSampleCount(self) -> int:
        return int(2**31 - 1)

    def sampleGoal(self, state: ob.State) -> None:
        goal_pos = np.array(self.goal_pose["pos"], dtype=float)

        u = np.random.normal(size=3)
        u /= (np.linalg.norm(u) + 1e-9)

        r = self.radius * (random.random() ** (1.0 / 3.0))

        sample_pos = goal_pos + r * u

        for i in range(3):
            state[0][i] = float(sample_pos[i])

        qw, qx, qy, qz = self.goal_pose["quat_wxyz"]
        state[1].w = float(qw)
        state[1].x = float(qx)
        state[1].y = float(qy)
        state[1].z = float(qz)

class OmplOSCPosePlanner:
    def __init__(
        self,
        env,
        robot_index: int = 0,
        obstacle_names: Optional[List[str]] = [],
        approach_names: Optional[List[str]] = [],
        avoid_distance=0.05,
        z_bounds: Optional[Tuple[float, float]] = None,
        endpoint_exempt_radius: float = 0.0,
    ):
        self.env = env
        self.avoid_distance = avoid_distance
        self.z_bounds = z_bounds
        # States within this radius of the segment's start/goal are exempt from
        # the obstacle clearance (the grasp / place point may legitimately sit
        # inside the keep-away zone; the constraint applies to the path between).
        self.endpoint_exempt_radius = float(endpoint_exempt_radius)
        self._endpoints: List[np.ndarray] = []
        self.robot = env.robots[robot_index]
        self.model = env.sim.model
        self.joint_names = ["robot0_joint1", "robot0_joint2", "robot0_joint3", 
                            "robot0_joint4", "robot0_joint5", "robot0_joint6", "robot0_joint7"]
        self.eef_name = "gripper0_eef"
        self.dof = len(self.joint_names)
        self.data = env.sim.data
        self.jnt_idx = [self.model.get_joint_qpos_addr(joint_name) for joint_name in self.joint_names]
        self.joint_low = np.array([self.model.jnt_range[self.model.joint_name2id(name)][0] for name in self.joint_names], dtype=float)
        self.joint_high = np.array([self.model.jnt_range[self.model.joint_name2id(name)][1] for name in self.joint_names], dtype=float)
        
        self.obstacle_names = obstacle_names if obstacle_names is not None else []
        self.approach_names = approach_names if approach_names is not None else []
 
        self.space = ob.CompoundStateSpace()
        R3 = ob.RealVectorStateSpace(3)
        SO3 = ob.SO3StateSpace()
        self.space.addSubspace(R3, 1.0)
        self.space.addSubspace(SO3, 10)
        robot_base_xpos = self.data.get_body_xpos("robot0_base")
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, robot_base_xpos[0])
        bounds.setHigh(0, robot_base_xpos[0] + 1.0)
        bounds.setLow(1, robot_base_xpos[1] - 1.0)
        bounds.setHigh(1, robot_base_xpos[1] + 1.0)
        if z_bounds is not None:
            bounds.setLow(2, float(z_bounds[0]))
            bounds.setHigh(2, float(z_bounds[1]))
        else:
            bounds.setLow(2, robot_base_xpos[2] - 0.05)
            bounds.setHigh(2, robot_base_xpos[2] + 1.0)
        self.space.getSubspace(0).setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.si = self.ss.getSpaceInformation()
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._state_valid))
        self.ss.getProblemDefinition().clearSolutionPaths()
        # self.ss.setOptimizationObjective(CustomObjective(self.si, env, self.approach_names))

    def cleanup(self):
        self.ss.clear()
        self.ss.getProblemDefinition().clearSolutionPaths()
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(lambda s: True)
        )

    def _get_qpos(self):
        return np.array([self.data.qpos[idx] for idx in self.jnt_idx], dtype=float) 
    
    def _state_valid(self, state: ob.State) -> bool:
        w = state[1].w
        x = state[1].x
        y = state[1].y
        z = state[1].z
        cur = np.array([state[0][i] for i in range(3)])
        if self.endpoint_exempt_radius > 0 and any(
            np.linalg.norm(cur - p) < self.endpoint_exempt_radius for p in self._endpoints
        ):
            return True
        for obst in self.obstacle_names:
            obst_pos = self.data.get_body_xpos(obst).copy()
            dist = np.linalg.norm(cur - obst_pos)
            if dist < self.avoid_distance: #Zhixu: adjust the avoidance distance here
                return False
        for target in self.approach_names:
            target_xpos = self.data.get_body_xpos(target).copy()
            dist = np.linalg.norm(cur - target_xpos)
            # print(dist)
            if dist < self.avoid_distance:
                return False
        return True
    
    def get_eef_pose(self) -> Dict[str, np.ndarray]:
        pos = self.data.get_body_xpos(self.eef_name).copy()
        quat = self.data.get_body_xquat(self.eef_name).copy()
        return {"pos": pos.copy(), "quat_wxyz": quat}

    def plan(
        self,
        start_pose: Optional[Dict[str, np.ndarray]],
        goal_pose: Optional[Dict[str, np.ndarray]],
        planner_name: str = "rrtc",
        radius: float = 0.01,
        timeout: float = 5.0,
        interpolate_n: int = 40,
    ) -> Tuple[List[Dict[str, np.ndarray]], bool]:


        start_pose = start_pose if start_pose is not None else self.get_eef_pose()
        self._endpoints = [np.asarray(start_pose["pos"], dtype=float), np.asarray(goal_pose["pos"], dtype=float)]
        self.ss.getProblemDefinition().clearSolutionPaths()
        s = ob.State(self.space); g = ob.State(self.space)
        for i in range(3):
            s[i] = float(start_pose["pos"][i])
        s[3] = start_pose["quat_wxyz"][1]
        s[4] = start_pose["quat_wxyz"][2]
        s[5] = start_pose["quat_wxyz"][3]
        s[6] = start_pose["quat_wxyz"][0]

        for i in range(3):
            g[i] = float(goal_pose["pos"][i])
        g[3] = goal_pose["quat_wxyz"][1]
        g[4] = goal_pose["quat_wxyz"][2]
        g[5] = goal_pose["quat_wxyz"][3]
        g[6] = goal_pose["quat_wxyz"][0]
 
        self.ss.setStartAndGoalStates(s, g, radius)

        planner = self._make_planner(planner_name)
        self.ss.setPlanner(planner)

        ok = self.ss.solve(timeout)
        if self.ss.haveExactSolutionPath():
            print("  Found exact solution.")
        if not ok or self.ss.getSolutionPath() is None:
            return [], False
        self.ss.simplifySolution()
        path_geometric = self.ss.getSolutionPath()
        path_geometric.interpolate(interpolate_n)
        waypoints = []
        states = path_geometric.getStates()
        for st in states:
            pos = np.array([st[0][j] for j in range(3)], dtype=float)
            quat_wxyz = np.array([st[1].w, st[1].x, st[1].y, st[1].z], dtype=float)
            waypoints.append({"pos": pos, "quat_wxyz": quat_wxyz})
        print(f"  Path found. Length: {path_geometric.getStateCount()}")

        return waypoints, True

    def osc_command_generator(
        self,
        waypoints: List[Dict[str, np.ndarray]],
        start_pose = None,
        step_pos: float = 0.01,
        step_rot: float = 0.03
    ):
        if not waypoints:
            return
        cur = start_pose
        for wp in waypoints[1:]:
            delta_p = wp["pos"] - cur["pos"]
            dist = np.linalg.norm(delta_p)
            q_err = quat_mul_wxyz(wp["quat_wxyz"], quat_inv_wxyz(cur["quat_wxyz"]))
            axis, ang = quat_to_axis_angle_wxyz(q_err)

            n_pos = int(max(1, np.ceil(dist / step_pos)))
            n_rot = int(max(1, np.ceil(ang / step_rot)))
            # debug
            # print(n_pos, n_rot)
            # print(dist, q_err)
            N = int(max(n_pos, n_rot))

            for k in range(N):
                tgt_p = cur["pos"] + delta_p / N
                tgt_q = quat_mul_wxyz(axis_angle_to_quat_wxyz(axis, ang / N), cur["quat_wxyz"])

                dpos = tgt_p - cur["pos"]
                q_step_err = quat_mul_wxyz(tgt_q, quat_inv_wxyz(cur["quat_wxyz"]))
                a_step, ang_step = quat_to_axis_angle_wxyz(q_step_err)
                daxisangle = a_step * ang_step

                yield dpos, daxisangle

                cur = {"pos": tgt_p.copy(), "quat_wxyz": tgt_q.copy()}

    def _make_planner(self, name: str):
        name = (name or "").lower()

        if name in ("rrtconnect", "rrtc"):
            return og.RRTConnect(self.si)
        if name in ("rrtstar", "rrt*"):
            return og.RRTstar(self.si)
        if name in ("bitstar", "bit*"):
            return og.BITstar(self.si)
        if name in ("prm", "prmstar", "prm*"):
            return og.PRMstar(self.si)

        return og.RRTConnect(self.si)

def ompl_plan_osc_pose(
    env,
    goal_pos: np.ndarray,
    goal_quat_wxyz: np.ndarray,
    start_pos: Optional[np.ndarray] = None,
    start_quat_wxyz: Optional[np.ndarray] = None,
    planner_name: str = "rrtc",
    obstacle_names: List[str] = [],
    approach_names: List[str] = [],
    approach_distance: float = 0.1,
    avoid_distance: float = 0.05,
    timeout: float = 5.0,
    interpolate_n: int = 40,
    z_bounds: Optional[Tuple[float, float]] = None,
    endpoint_exempt_radius: float = 0.0,
):
    planner = OmplOSCPosePlanner(env = env,avoid_distance= avoid_distance,obstacle_names=obstacle_names, approach_names=approach_names,
                                 z_bounds=z_bounds, endpoint_exempt_radius=endpoint_exempt_radius)
    goal_pose = {"pos": np.asarray(goal_pos, dtype=float),
                 "quat_wxyz": np.asarray(goal_quat_wxyz, dtype=float)}
    waypoints = []
    ok = True
    start_pose = {}
    if start_pos is None:
        start_pose["pos"] = planner.get_eef_pose()["pos"]
    else:
        start_pose["pos"] = start_pos

    if start_quat_wxyz is None:
        start_pose["quat_wxyz"] = planner.get_eef_pose()["quat_wxyz"]
    else:
        start_pose["quat_wxyz"] = start_quat_wxyz
    start_pose = start_pos is not None and start_quat_wxyz is not None and \
        {"pos": np.asarray(start_pos, dtype=float), 
         "quat_wxyz": np.asarray(start_quat_wxyz, dtype=float)} or planner.get_eef_pose()
    eef_quat = start_pose["quat_wxyz"]
    for target in approach_names:
        target_xpos = env.sim.data.get_body_xpos(target).copy()
        waypoints_segment, ok_segment = planner.plan(
            start_pose=start_pose,
            goal_pose={"pos": target_xpos, "quat_wxyz": eef_quat},
            planner_name=planner_name,
            radius = approach_distance,#Zhixu: adjust approach distance here
            timeout=timeout,
            interpolate_n=interpolate_n,
        )
        ok = ok and ok_segment
        if not ok:
            break
        waypoints.extend(waypoints_segment)
        start_pose = waypoints_segment[-1]

    waypoints_segment, ok_segment = planner.plan (
        start_pose=start_pose,
        goal_pose=goal_pose,
        planner_name=planner_name,
        timeout=timeout,
        interpolate_n=interpolate_n,
    )
    ok = ok and ok_segment
    waypoints.extend (waypoints_segment)
    planner.cleanup()

    return waypoints, ok, (lambda **kw: planner.osc_command_generator (waypoints, start_pose, **kw))


if __name__ == "__main__":
    layout_dir = "../../libero/libero/bddl_files/libero_10/"
    layouts = [f for f in os.listdir(layout_dir) if f.endswith(".bddl")]
    print(layouts)
    naive_bddl_file = "../../libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl"
    layout = Layout(naive_bddl_file)
    naive_task = TaskPlan(layout.lan_inst)
 
    #Initialize video recorder and Env
    output_dir =  "../../dataset/"
    video_path = os.path.join(output_dir, "MotionGen_Heuristic.mp4")
    frame_size = (256, 256)
    fps = 20
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    os.makedirs(output_dir, exist_ok=True)

    env_args = {
        "bddl_file_name": naive_bddl_file,
        # define your own bddl file here
        "camera_heights": 256,
        "camera_widths": 256,
        # Use OSC_position control in default
    }
    env = OffScreenRenderEnv(**env_args, ignore_done=True)

    problem_info = BDDLUtils.get_problem_info(naive_bddl_file)
    problem_name = problem_info["problem_name"]

    robots = ["Panda"]
    controller_config = robosuite.controllers.load_controller_config(default_controller="OSC_POSE")
    config = {
        "robots": robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(naive_bddl_file)

    env.seed(0)
    obs = env.reset() 

    pose_now = OmplOSCPosePlanner(env).get_eef_pose()
    goal_pos = env.sim.data.get_body_xpos("flat_stove_1_main").copy() + np.array([0.0, 0.0, 0.1])
    z_rot_wxyz = axis_angle_to_quat_wxyz(np.array([0.0, 0.0, 1.0]), np.pi/6.0)
    goal_quat = pose_now["quat_wxyz"]
    goal_quat = quat_mul_wxyz(z_rot_wxyz, pose_now["quat_wxyz"])
    print(pose_now)
    print(goal_pos, goal_quat)

    approach_names = ["plate_1_main"]

    wps, ok, to_osc = ompl_plan_osc_pose(env, goal_pos, goal_quat, planner_name="rrtstar", approach_names = approach_names, timeout=5.0, interpolate_n=40)
    for wp in wps:
        print(wp["pos"], wp["quat_wxyz"])

    if ok:
        for dpos, daxisangle in to_osc(step_pos=0.01, step_rot=0.1):
            action = np.concatenate([dpos / 0.012, daxisangle / 0.125, [-1.0]])
            raw_obs, reward, done, info = env.step(action)
            frame = raw_obs["agentview_image"]
            frame = frame[::-1, ::-1]
            if video_writer:
                video_writer.write(frame)
 
    video_writer.release()
    print("video saved to:", video_path)
    env.close()
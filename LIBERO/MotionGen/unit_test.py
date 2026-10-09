"""
Test the functionality of the module here
define testing input and expected output

Headless usage (render offscreen and save MP4):
  - This script is configured for offscreen rendering (no GUI) and writes
    frames to an MP4 at: LIBERO/dataset/MotionGen_UnitTest.mp4
  - Run on a remote Linux host:
      python LIBERO/MotionGen/unit_test.py
    If your OpenGL / EGL stack requires a virtual display, use:
      xvfb-run -a -s "-screen 0 1920x1080x24" python LIBERO/MotionGen/unit_test.py

Viewing / copying the MP4 to your local machine:
  - You can play the MP4 directly on the remote host or copy it locally.
  - Generic copy with scp from your local machine (replace placeholders):
      scp <user>@<host>:<remote_project_root>/LIBERO/dataset/MotionGen_UnitTest.mp4 <local_download_dir>/
    Examples of placeholders:
      <user>                your remote username
      <host>                remote hostname or IP
      <remote_project_root> absolute path to the project on the remote host
      <local_download_dir>  local directory to save the file
"""
import sys
import os

from Utility.tool_func import to_sim_env_name

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from TaskGen.task import TaskPlan
from TaskGen.layout_gen.layout import Layout
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
import os
import matplotlib.pyplot as plt
import robosuite
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("EGL_DEVICE_ID", "0")
import cv2
import random
import math
from MotionGen.heuristic_skill.heuristic import *
import numpy as np
from typing import List, Tuple, Optional, Dict

#The distance between gripper and target in x axis is 0.0130
#The distance between gripper and target in y axis is 0.0502
#The distance between gripper and target in z axis is 0.2343

if __name__ == "__main__":
    naive_bddl_file = "MotionGen/layout_test/layout_1.bddl"
    
    #Initialize video recorder and Env
    output_dir =  "../dataset/unit_test/"
    os.makedirs (output_dir, exist_ok=True)
    

    for i in range(100):
        layout = Layout(naive_bddl_file)
        naive_task = TaskPlan(layout.lan_inst)
        video_path = os.path.join(output_dir, f"MotionGen_Heuristic_{i}.mp4")
        frame_size = (512, 512)
        fps = 20
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
        env_args = {
            "bddl_file_name": naive_bddl_file,
            # define your own bddl file here
            "camera_heights": 512,
            "camera_widths": 512,
            # Use OSC_position control in default
        }
        env = OffScreenRenderEnv(**env_args, ignore_done=True)

        # print all objects in the environment
        # print("Objects in the environment:")
        # for obj in env.sim.model.body_names:
        #     print(obj)

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

        obj_info = get_body_size_3d_v1(env, "bbq_sauce_1_main")

        actions = move_to(env, target_pos = env.sim.data.get_body_xpos("basket_1_main") + np.array([0, 0, 0.2]),
                            target_orn = [-np.pi, 0, -np.pi/2])
        for action in actions:
            raw_obs, reward, done, info = env.step(action)
            frame = raw_obs["agentview_image"]
            frame = frame[::-1, ::-1]
            if video_writer:
                video_writer.write(frame)

        actions = drag(env, "basket_1_main", [1.0, 0], 0.2)
        for action in actions:
            raw_obs, reward, done, info = env.step(action)
            frame = raw_obs["agentview_image"]
            frame = frame[::-1, ::-1]
            if video_writer:
                video_writer.write(frame)
        print(env.sim.get_state().qpos.shape)

        actions = move_to(env, target_pos = env.sim.data.get_body_xpos("basket_1_main") + np.array([0, 0, 0.2]),
                            target_orn = [-np.pi, 0, -np.pi/2], gripper_state=1.0)
        for action in actions:
            raw_obs, reward, done, info = env.step(action)
            frame = raw_obs["agentview_image"]
            frame = frame[::-1, ::-1]
            if video_writer:
                video_writer.write(frame)
        actions = throw(env, "basket_1_main")
        for action in actions:
            raw_obs, reward, done, info = env.step(action)
            frame = raw_obs["agentview_image"]
            frame = frame[::-1, ::-1]
            if video_writer:
                video_writer.write(frame)

        video_writer.release()
        print("video saved to:", video_path)
        env.close()

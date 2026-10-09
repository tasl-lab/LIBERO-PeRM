import argparse
import json
import os
import random

import h5py
import numpy as np

from pynput import keyboard

import robosuite

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--bddl-file",
        type=str,
        required=True,
        help="Path to the BDDL file to use.",
    )
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="frontview",
        help="Camera name to use for rendering.",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]

    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    print(env_info) 

    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    print("Found {} demonstration episodes".format(len(demos)))
    print("demos: {}".format(demos))

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        ep = random.choice(demos)
        
        # print attributes
        # print("Episode attributes:")
        # for k, v in f["data/{}".format(ep)].attrs.items():
        #     print("  {}: {}".format(k, v))
        # print("\n")
        
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        # print("Using model: {}".format(model_xml))

        env.reset()
        #xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        recording = False
        stop_signal = False
        record_state = []
        record_action = []
        recording_printed = False  # 新增标志变量

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        print("Using actions for playback")
        # load the initial state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # load the actions and play them back open-loop
        actions = np.array(f["data/{}/actions".format(ep)][()])
        num_actions = actions.shape[0]
        print(env.viewer.__class__)

        for name in env.sim.model.body_names:
            idx = env.sim.model.body_name2id(name)
            pos = env.sim.data.body_xpos[idx]
            print(f"  {name}: {pos}")
            env.viewer.add_marker(
                pos=pos,  # 世界坐标
                label="Anchor Point",
                size=[0.01, 0.01, 0.01],
                rgba=[1, 0, 0, 1],
                type=env.viewer.MARKER_SPHERE
            )
 
        for j, action in enumerate(actions):
            env.step(action)
            env.render()
            

        f.close()
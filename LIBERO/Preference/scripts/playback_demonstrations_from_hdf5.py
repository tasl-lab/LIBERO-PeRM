"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np
from pynput import keyboard

# scripts/.. -> LIBERO/
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from libero.libero.envs import *
from libero.libero.envs import OffScreenRenderEnv,DemoRenderEnv

import libero.libero.envs.bddl_utils as BDDLUtils
from Utility.tool_func import ask_choice, is_noop


def on_press(key):
    global recording, stop_signal,critical_recoding,critical_stop_signal
    try:
        if key.char == 'r':
            print("Start recording")
            recording = True
        if key.char == 's':
            print("Stop recording")
            stop_signal = True
        if key.char == 'a':
            print("Critical demo start")
            critical_recoding = True
        if key.char == 'd':
            print("Critical demo stop")
            critical_stop_signal = True
    except AttributeError:
        pass


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

    parser.add_argument (
        "--hdf5-name",
        type=str,
        default="demo.hdf5",
        help="The name of the hdf5 file containing the demonstrations.",
    )
    parser.add_argument(
        "--hdf5-action",
        type=int,
        default=1,
        help="If set, will use the actions stored in the hdf5 file to playback the demo.",
    )
    args = parser.parse_args()
    if args.hdf5_action:

        demo_path = args.folder
        hdf5_name=args.hdf5_name
        hdf5_path = os.path.join(demo_path, hdf5_name)
        hdf5_output_path = os.path.join(demo_path, f"recorded_{hdf5_name}")
        hdf5_noop_path= os.path.join(demo_path, f"recorded_noop_{hdf5_name}")
        f = h5py.File (hdf5_path, "r")
        demos = list (f ["data"].keys ())
        print ("Found {} demonstration episodes".format (len (demos)))
        print ("demos: {}".format (demos))
        # problem_info = BDDLUtils.get_problem_info(args.bddl_file)
        # problem_name = problem_info["problem_name"]


    env_args = {
        # "bddl_file_name": layout.bddl_file,
        "bddl_file_name": args.bddl_file,
        # define your own bddl file here
        "camera_heights": 256,  # 128, follow the setting in libero dataset
        "camera_widths": 256,  # 1024
        "render_camera": args.camera,
        "camera_names": ["agentview"],
        "use_camera_obs": False,
        # "control_freq" : 20,
        # Use OSC_position control in default
    }
    # 初始化环境

    env = DemoRenderEnv (**env_args, ignore_done=True)
    # env = OffScreenRenderEnv (**env_args, ignore_done=True)

    # print(env_info)
    #
    # env = TASK_MAPPING[problem_name](
    #     bddl_file_name=args.bddl_file,
    #     **env_info,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera=args.camera,
    #     ignore_done=True,
    #     use_camera_obs=False,
    #     reward_shaping=True,
    #     control_freq=20,
    # )


    # list of all demonstrations episodes


    while True:
        print("Playing back random episode... (press ESC to quit)")




        # print attributes
        # print("Episode attributes:")
        # for k, v in f["data/{}".format(ep)].attrs.items():
        #     print("  {}: {}".format(k, v))
        # print("\n")



        env.reset()
        # env.sim.reset()
        # env.viewer.set_camera(0)

        recording = False
        stop_signal = False
        record_state = []
        record_action = []
        record_noop_action = []

        critical_action_index=[]
        critical_action_pair=None
        critical_recoding = False
        critical_stop_signal = False

        recording_printed = False  # 新增标志变量

        listener = keyboard.Listener(on_press=on_press)
        listener.start()



        if args.hdf5_action:
            # load the actions and play them back open-loop
            # select an episode randomly
            ep = random.choice (demos)
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape [0]
        else:
            from MotionGen.heuristic_skill.heuristic import pick_from_side,pick_bottle,move_to
            from Utility.tool_func import to_sim_env_name
            from scipy.spatial.transform import Rotation as R
            print(f"Use non-hdf5 action playback method")
            target="bbq_sauce_1"
            target = to_sim_env_name (target)
            idx = env.sim.model.body_name2id (target)
            target_xpos = env.sim.data.body_xpos [idx]
            gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
            print (f"[Motion Gen] move_to from {gripper_pos} to {target_xpos}")
            eef_id = env.sim.model.body_name2id ("gripper0_eef")
            eef_quat = env.sim.data.body_xquat [eef_id]
            eef_quat_xyzw = np.array ([eef_quat [1], eef_quat [2], eef_quat [3], eef_quat [0]])
            target_orn = R.from_quat (eef_quat_xyzw).as_euler ("XYZ")
            target_xpos [2] += 0.2
            actions = move_to (env, target_pos=target_xpos, target_orn=target_orn, speed=2e-2,
                                    gripper_state=1)
            actions.extend(pick_from_side (env, target_name=target,
                                           direction=[-1, -1]))#[-1, 1]
            num_actions = len(actions)




        step_cnt=0
        op_action_cnt=0
        # speed=1/2
        speed = 1 / 3
        print(f"Original demo length: {num_actions} steps, playing at speed {speed}x")
        for j, action in enumerate(actions):

            env.step(action)
            env.env.render()

            # if recording and not recording_printed:
                # print("Object positions at recording start:")
                # for name in env.sim.model.body_names:
                #     idx = env.sim.model.body_name2id(name)
                #     pos = env.sim.data.body_xpos[idx]
                #     print(f"  {name}: {pos}")
                # recording_printed = True
            if critical_recoding:
                critical_action_pair=(j,None)
                critical_recoding=False

            if critical_action_pair is None:
                if not is_noop (action, None if j == 0 else actions [j - 1]):
                    record_noop_action.append (action)
                    print(f"Recording no-op action {action} at step {j}")
                else:
                    op_action_cnt+=1
                    if op_action_cnt%(1/speed)==0:
                        record_noop_action.append (action)
                        print (f"Recording dummy action {action} at step {j}")
                    else:
                        print(f"Filtering out no-op {action} at step {j}")


            else:
                record_noop_action.append (action)
                print (f"Recording critical action {action} at step {j}")



            if recording:
                record_state.append(env.sim.get_state().flatten())
                record_action.append(action)

                if step_cnt==0:
                    gripper_pos = env.sim.data.get_body_xpos ("gripper0_eef").copy ()
                    target_name=None    #"plate_1_main"
                    if target_name is not None:
                        target_pos= env.sim.data.get_body_xpos ().copy ()
                        print(f"The distance between gripper and target in x axis is {gripper_pos[0]-target_pos[0]:.4f}")
                        print(f"The distance between gripper and target in y axis is {gripper_pos[1]-target_pos[1]:.4f}")
                        print(f"The distance between gripper and target in z axis is {gripper_pos[2]-target_pos[2]:.4f}")

                step_cnt += 1


            if critical_stop_signal or (j == num_actions - 1 and critical_action_pair is not None):
                critical_action_pair=(critical_action_pair[0],j)
                critical_stop_signal=False
                print(f"Critical action segment: {critical_action_pair}")
                critical_action_index.append(critical_action_pair)
                critical_action_pair=None

            if stop_signal or (j == num_actions - 1 and recording):
                print("Stop recording")
                recording = False
                stop_signal = False
                record_state = np.array(record_state)
                record_action = np.array(record_action)

                print(f"recorded {record_state.shape[0]} steps")
                save_path = hdf5_output_path
                print(record_action.shape)
                with h5py.File(save_path, "w") as rf:
                    grp = rf.create_group("data")
                    # grp.attrs['env'] = env_name
                    # grp.attrs['env_info'] = json.dumps(env_info)
                    ep_grp = grp.create_group("ep")
                    # ep_grp.create_dataset("states", data=record_state)
                    ep_grp.create_dataset("actions", data=record_action)
                print(f"Saved to {save_path}")
                record_state = []
                record_action = []
                recording_printed = False  # 录制结束后重置标志
            # if j < num_actions - 1:
            #     # ensure that the actions deterministically lead to the same recorded states
            #     state_playback = env.sim.get_state().flatten()
            #     if not np.all(np.equal(states[j + 1], state_playback)):
            #         err = np.linalg.norm(states[j + 1] - state_playback)
            #         print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")


        play_noop_actions=ask_choice(f"Play the no-op actions demo? The length of no-op actions is {len(record_noop_action)}, "
                                     f"filter out {len(actions)-len(record_noop_action)} steps", default=1)
        if play_noop_actions:
            print(f"Playing no-op actions...")
            # 初始化环境
            env_noop = DemoRenderEnv (**env_args, ignore_done=True)
            env_noop.reset ()

            # load the actions and play them back open-loop
            for action_idx,_action in enumerate(record_noop_action):
                env_noop.step(_action)
                env_noop.env.render ()
            save_noop=ask_choice("Save the no-op actions demo?", default=1)
            if save_noop:
                with h5py.File(hdf5_noop_path, "w") as rf:
                    grp = rf.create_group("data")
                    ep_grp = grp.create_group("ep")
                    ep_grp.create_dataset("actions", data=record_noop_action)
                print(f"Saved to {hdf5_noop_path}")


        next_eps=ask_choice("Play another episode?", default=0)
        if not next_eps:
            break
    f.close()
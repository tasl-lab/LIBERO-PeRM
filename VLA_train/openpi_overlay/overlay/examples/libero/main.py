import collections
import dataclasses
import logging
import math
import pathlib
import sys
libero_path = "/data3/zhixu/P_Bench/VLAs/openpi/third_party/libero"

# 2. 将该路径插入到 sys.path 的最前面（优先级最高）
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

from libero.libero import benchmark
from libero.libero import get_libero_path
# 4. 检查 libero 识别的资源根目录
check_path=get_libero_path("bddl_files")
print(f"[*] 确认 libero 资源路径: {check_path}")
import imageio

from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import re
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

sys.path.append('/data3/zhixu/P_Bench/LIBERO/Preference')
sys.path.append('/data3/zhixu/P_Bench/LIBERO')
from pref_eval import gen_low_pref_eval,max_acceleration


def get_img_from_obs(obs,key="agentview_image"):
    if key not in obs.keys():
        raise ValueError(f"The key {key} is not in the observation dict")
    else:
        img = obs["agentview_image"]
        # img = img [:, :, ::-1] # The image is already in RGB format, so comement this line
        img = img [::-1, ::]
        img = img [::, ::-1]
    return img

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    use_init:int = 0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        # "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
        "libero_10"
        # "libero_object_p"
        # "libero_pbench"
        # "libero_pbench_eval"
        # "libero_pbench_eval_m"
        # "libero_pbench_eval_ood"
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5  # Number of rollouts per task，50 in default
    save_init_file=True
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "../video_demo"  # Path to save videos

    Step_50k_running_mission_ids=[
        "8", "15", "19", "20", "28",
        "29", "30", "32", "36", "42",
        "46", "50"
    ]
    Step_45k_running_mission_ids=[
        "2", "3", "4", "5", "6", "7",
        "9", "10","11", "12", "13", "14",
        "17", "23", "24","27", "31","33",
        "34", "39", "40", "41","44", "45",
        "51", "52"
    ]
    Step_common_running_mission_ids=[
    "1", "16", "18", "21", "22", "25",
    "26", "35", "37", "38", "43", "47",
    "48"
    ]
    running_mission_ids_0 = [
    "26","27","28","29","30",
    "31","32","33","34","35","36",
    "37","38","39","40","41","42",
    "43","44","45","46","47","48",
    "49","50","51","52"
    ]
    running_mission_ids = [
    "3", "7", "13", "15", "21", "27", "47"
    ]
    running_mission_ids_m = [
    "m1","m2","m3","m4","m5","m6",
    "m7","m8","m9","m10","m11","m12",
    "m13","m14","m15","m16","m17","m18",
    "m19","m20","m21","m22","m23","m24",
    "m25","m26","m27","m28","m29","m30",
    "m31","m32","m33","m34","m35","m36",
    "m37","m38","m39","m40","m41","m42",
    "m43","m44","m45","m46","m47","m48",
    "m49","m50","m51","m52"
    ]
    running_mission_ids_all = [
        "1", "2", "3", "4", "5", "6"
        "7", "8", "9", "10", "11", "12",
        "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30",
        "31", "32", "33", "34", "35", "36",
        "37", "38", "39", "40", "41", "42",
        "43", "44", "45", "46", "47", "48",
        "49", "50", "51", "52"


    ]
    highSR_mission_ids = [
    "1", "2","3","4","5","6"
    "7","8","9","10","11","12",
    "13","14", "18", "25","26",
    "28","31","32","33","34","36",
    "39","43","44","45","46","47",
    "49","50","51"
    ]

    low_SR_running_mission_ids=[
        "8", "14", "15", "16", "19",
        "20", "21", "22", "23", "24",
        "28", "29", "30", "35", "36",
        "37", "38", "39", "42", "44",
        "46", "48", "50"
    ]

    seed: int = 7  # Random Seed (for reproducibility)

def save_img(img, path):
    import imageio
    imageio.imwrite(path, img)
    print("✅ 图片保存路径：", path)

import pickle
def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    load_init_file=args.use_init

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()

    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_object_p":
        max_steps = 500
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 505  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    elif  args.task_suite_name == "libero_pbench":
        max_steps = 600
    elif args.task_suite_name == "libero_pbench_eval":
        max_steps = 1200
    elif args.task_suite_name == "libero_pbench_eval_m":
        max_steps = 1200
    elif args.task_suite_name == "libero_pbench_eval_ood":
        max_steps = 1200
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


    # Start evaluation
    total_episodes, total_successes = 0, 0
    #iterate through each task in task set
    import os
    txt_log_path = os.path.join(args.video_out_path, f"Mission_SR_log.txt")
    with open(txt_log_path, "a") as f:
        f.write(
            f"\n//=====================================\n")


    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        low_pref=False
        # Get task
        task = task_suite.get_task(task_id) #task id: 0,1,2,3,4,5...

        mission_id = task.bddl_file.replace('.bddl', '')
        mission_id = mission_id.replace('Mission_', '')
        # if mission_id.startswith('m'):
        #     mid=mission_id.replace('m','')
        # else:
        #     mid=mission_id
        mid = re.sub(r'_L.*', '', mission_id)
        # if int(_mid)<=10:
        low_pref=True
        if mid not in args.running_mission_ids:
            logging.info(f"Skipping Mission {mission_id} as it's not in the running list.")
            continue




        # Task(name='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket',
        # language='put both the alphabet soup and the tomato sauce in the basket', problem='Libero', problem_folder='libero_10',
        # bddl_file='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl',
        # init_states_file='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.pruned_init')


        # Get default LIBERO initial states
        # 在循环开始前，统计当前 mission 已有的 init_state 数量
        init_files_dir = "/data3/zhixu/P_Bench/VLAs/openpi/third_party/libero/libero/libero/init_files/libero_pbench"  # 或 libero_pbench
        init_files_dir=os.path.join(init_files_dir,f"Mission_{mission_id}")
        pathlib.Path(init_files_dir).mkdir(parents=True, exist_ok=True)

        # 获取基础文件名
        base_init_name = task.bddl_file.replace('.bddl', '')

        # initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        print(f"\n=== Loaded bDDL file: {task.bddl_file} ===")
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        if low_pref:
            low_pref2apply=gen_low_pref_eval(mission_id=mission_id)
            pref_step_reward = {pref.preference_type: [] for pref in low_pref2apply}  # per-pref reward record
            pref_total_reward = {pref.preference_type: 0 for pref in low_pref2apply}  # per-pref reward record
            pref_ave_reward = {pref.preference_type: None for pref in low_pref2apply}  # per-pref reward record

        # task_description = "pick_up_the_cream_cheese_and_place_it_in_the_basket"
        # task_description = "now_place_the_cheese_in_the_basket"
        print(f"Language instruction: {task_description}, Mission id: {mission_id}")


        video_output_folder= os.path.join(args.video_out_path, f"Mission_{mission_id}")
        pathlib.Path(video_output_folder).mkdir(parents=True, exist_ok=True)

        # 统计已有的 init 文件数量
        existing_inits = list(pathlib.Path(init_files_dir).glob(f"{base_init_name}_*.init"))
        init_counter = len(existing_inits)
        # Start episodes
        task_episodes, task_successes = 0, 0
        task_low_pref_reward = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            save_init = args.save_init_file

            # Reset environment
            env.reset()
            task_actions = []
            action_plan = collections.deque()
            init_state_path = os.path.join(
                init_files_dir,
                f"{base_init_name}_{episode_idx % 15}.init"  # 循环使用 0-14
            )

            if load_init_file:
                if os.path.exists(init_state_path):
                    with open(init_state_path, 'rb') as f:
                        init_state = pickle.load(f)
                    obs = env.set_init_state(init_state)
                    print(f"✅ Loaded init state: {init_state_path}")
                    save_init=False
                else:
                    print(f"⚠️⚠️⚠️⚠️⚠️ No saved init state, using default")

            # Set initial states
            # obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_img_eih=[]
            init_state=None

            _Done=False
            task_action_idx=0
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait and save_init :
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    if t == args.num_steps_wait: #save a framep png
                        img=get_img_from_obs(obs, key="agentview_image")

                        import os
                        img_path= os.path.join(video_output_folder,"agentview_image.png")
                        imageio.imwrite(img_path, img)
                        print("✅ 图片保存路径：", img_path)

                        # 等待物体稳定后获取 init_state
                        init_state = env.get_sim_state()



                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    replay_img_eih.append(wrist_img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }


                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    # if t%7==0:
                    #     print(f"The action (1/7 sampled) is :>>>>>\n {action}")
                   # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    if low_pref:
                        task_action_idx+=1
                        for _pref in low_pref2apply:
                            if _pref.preference_type == "acc_range":
                                task_actions.append(action.tolist())
                            elif _pref.preference_type == "speed_range":
                                pref_step_reward[_pref.preference_type].append(_pref.get_reward(env))
                                pref_total_reward[_pref.preference_type] += pref_step_reward[_pref.preference_type][-1]
                                pref_ave_reward[_pref.preference_type] = pref_total_reward[
                                                                             _pref.preference_type] / task_action_idx
                            elif _pref.preference_type == "approach_to_object" or _pref.preference_type=="avoid_to_object":
                                pref_step_reward[_pref.preference_type].append(_pref.get_reward(env))
                                pref_total_reward[_pref.preference_type] += pref_step_reward[_pref.preference_type][-1]
                                pref_ave_reward[_pref.preference_type] = pref_total_reward[_pref.preference_type] / task_action_idx
                            else:
                                raise NotImplementedError
                    if done:
                        _Done=True
                        task_successes += 1
                        total_successes += 1
                        # 检查是否已达到 15 个上限
                        if init_counter < 15 and save_init:
                            # 保存为 .init 格式（带序号）
                            init_state_path = os.path.join(
                                init_files_dir,
                                f"{base_init_name}_{init_counter}.init"
                            )

                            with open(init_state_path, 'wb') as f:
                                pickle.dump(init_state, f)

                            print(f"✅ Init state saved: {init_state_path}")
                            init_counter += 1
                        else:
                            print(f"⚠️ Already have 15 init states for this mission")
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if _Done else "failure"
            task_segment = task_description.replace(" ", "_")
            assert len(replay_images) == len(replay_img_eih)

            concat_frames = []

            for fv, eih in zip(replay_images, replay_img_eih):
                fv = np.asarray(fv)
                eih = np.asarray(eih)

                # 横向拼接
                combined = np.concatenate([fv, eih], axis=1)  # (256, 512, 3)
                concat_frames.append(combined)

            imageio.mimwrite(
                pathlib.Path(video_output_folder)
                / f"Ep{episode_idx}_{suffix}_{task_segment}.mp4",
                concat_frames,
                fps=10,
            )

            # imageio.mimwrite(
            #     pathlib.Path(args.video_out_path) / f"Fv_Ep{episode_idx}_{task_segment}_{suffix}.mp4",
            #     [np.asarray(x) for x in replay_images],
            #     fps=10,
            # )
            # imageio.mimwrite(
            #     pathlib.Path(args.video_out_path) / f"Eih_Ep{episode_idx}_{task_segment}_{suffix}.mp4",
            #     [np.asarray(x) for x in replay_img_eih],
            #     fps=10,
            # )

            # Log current results
            logging.info(f"Success: {_Done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            # 打开文件进行追加写入
            if low_pref:
                with open('/data3/zhixu/P_Bench/video_demo/low_pref_data.txt', 'a') as f:
                    for _pref in low_pref2apply:
                        if _pref.preference_type == "acc_range":
                            ave_rwd=max_acceleration(task_actions)
                        elif _pref.preference_type == "speed_range":
                            ave_rwd= pref_ave_reward[_pref.preference_type]
                            satisfaction = min(ave_rwd/0.0450,100)
                        elif _pref.preference_type == "approach_to_object" or _pref.preference_type == "avoid_to_object":
                            ave_rwd= pref_ave_reward[_pref.preference_type]
                            satisfaction = min(ave_rwd/0.2325,100)
                        else:
                            raise NotImplementedError
                        message = f"[Motion Gen] Average reward for {_pref.preference_type} for mission {mid}, rollout{episode_idx}: {ave_rwd:.4f}\n"
                        message1=f"[Motion Gen] Satisfaction level for { _pref.preference_type} for mission {mid}, rollout{episode_idx}: {satisfaction:.2f}\n"
                        print(message, end='')  # 打印到控制台
                        print(message1, end='')  # 打印到控制台
                        f.write(message)  # 写入文件
                        f.write(message1)  # 写入文件


        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        current_sr = float(task_successes) / float(task_episodes)
        if current_sr<0.8 and current_sr>0.1:
            suggestion="Try set init state"
        elif current_sr<=0.1:
            suggestion="Optimize the Mission"
        else:
            suggestion=""
        with open(txt_log_path, "a") as f:
            f.write(f"Mission {mission_id} - Success Rate: {current_sr:.4f} ({task_successes}/{task_episodes}) || {suggestion}\n")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)

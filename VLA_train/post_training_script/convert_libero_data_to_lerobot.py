"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os
import h5py
import numpy as np
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

ROBOT_TYPE = "panda"


def parse_language_from_filename(hdf5_path: str) -> str:
    """从 pick_white_mug_put_on_right_plate.hdf5 解析指令"""
    name = os.path.basename(hdf5_path).replace(".hdf5", "")
    words = name.split("_")
    command = []
    for w in words:
        if "SCENE" in w:
            command = []
            continue
        command.append(w)
    return " ".join(command)


def parse_language_from_filename(hdf5_path: str) -> str:
    """从 hdf5 文件名解析语言指令"""
    name = os.path.basename(hdf5_path).replace(".hdf5", "")
    words = name.split("_")
    command = []
    for w in words:
        if "SCENE" in w:
            command = []
            continue
        command.append(w)
    return " ".join(command)

def get_finetune_data():
    pass

from typing import List
import json
def main(
        *,
        json_config: str = "/data3/zhixu/LIBERO-PeRM/LIBERO/batch_run_m/batch_run_m.json",
        task_lan_json: str = "/data3/zhixu/LIBERO-PeRM/LIBERO/batch_run_m/task_lan_m.json",  # 包含 mission_id -> task_lan 映射
        hdf5_base_dir: str = "/data4/zhixu/PBenchData",
        output_dir: str,
        repo_id:str,
):
    # 从 JSON 读取并构建路径
    with open(json_config, 'r') as f:
        config = json.load(f)

    mission_ids = config["convertable_hdf5_mission_id"]

    with open(task_lan_json, "r") as f:
        task_lan_list = json.load(f)

    # mission_id_to_task[7]-> "pick up the cheese and put it on the plate. preference: keep away from the stove"
    mission_id_to_task = {
        str(item["mission_id"]): item["task_lan"].strip().lower()
        for item in task_lan_list
    }

    hdf5_paths = [
        os.path.join(hdf5_base_dir, f"Mission_{str(mid)}.hdf5")
        for mid in mission_ids
    ]

    # 创建映射字典
    mission_mapping = {}
    global_episode_idx = 0

    output_dir = os.path.join(output_dir,repo_id)
    output_dir = Path(output_dir)


    # 创建 LeRobot dataset（纯本地）
    dataset = LeRobotDataset.create(
        repo_id=repo_id,  # 只是名字，不会 push
        root=output_dir,  # 你指定的输出路径
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    for hdf5_path in map(Path, hdf5_paths):
        assert hdf5_path.exists(), f"HDF5 not found: {hdf5_path}"
        mission_id = hdf5_path.stem.replace("Mission_", "")
        assert mission_id in mission_id_to_task, \
            f"Mission {mission_id} not found in task_lan.json"

        command = mission_id_to_task[mission_id]

        print(f"Processing {hdf5_path.name}, command = {command}")
        mission_demos = []

        with h5py.File(hdf5_path, "r") as F:
            for demo_idx, demo_key in enumerate(F["data"].keys()):  # 改这行，添加 enumerate
                if demo_idx >= 100: #100: #50 # 已经处理了 100 个 (0-99)
                    break
                demo = F["data"][demo_key]
                actions = demo["actions"][()]
                ee_states = demo["obs"]["ee_states"][()]
                gripper_states = demo["obs"]["gripper_states"][()]
                images = demo["obs"]["agentview_rgb"][()]
                wrist_images = demo["obs"]["eye_in_hand_rgb"][()]

                T = actions.shape[0]

                for t in range(T):
                    state = np.concatenate(
                        [ee_states[t], gripper_states[t]], axis=-1
                    ).astype(np.float32)

                    dataset.add_frame(
                        {
                            "image": images[t],
                            "wrist_image": wrist_images[t],
                            "state": state,
                            "actions": actions[t].astype(np.float32),
                            "task": command,
                        }
                    )

                # 一个 demo = 一个 episode
                dataset.save_episode()
                # 记录映射
                chunk_id = global_episode_idx // 1000
                mission_demos.append({
                    "demo_key": demo_key,
                    "chunk_id": chunk_id,
                    "global_episode_idx": global_episode_idx,
                })
                global_episode_idx += 1

        # 保存 mission 映射
        mission_mapping[f"mission_{mission_id}"] = {
            "demos": mission_demos,
            "total_demos": len(mission_demos)
        }

        # 在这里删除 HDF5 文件
        # try:
        #     hdf5_path.unlink()  # 删除文件
        #     print(f"Deleted {hdf5_path.name}")
        # except Exception as e:
        #     print(f"Failed to delete {hdf5_path.name}: {e}")

    # 保存映射文件
    mapping_file = output_dir / "mission_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mission_mapping, f, indent=2)
    print(f"Mapping saved to: {mapping_file}")

    print(f"✅ All HDF5 converted. Output at: {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)

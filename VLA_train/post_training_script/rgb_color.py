import h5py
import numpy as np
import cv2
import os

def check_img(hdf5_file, out_dir="check_images", idx=0):
    """
    Check and save agentview_rgb images from an HDF5 file to verify RGB vs BGR.

    Args:
        hdf5_file (str): path to hdf5 file
        out_dir (str): directory to save images
        idx (int): which frame to inspect (default: 0)
    """

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(hdf5_file, "r") as f:
        imgs = f["data/demo_0/obs/agentview_rgb"][()]
        eih_imgs = f["data/demo_0/obs/eye_in_hand_rgb"][()]

    img = imgs[idx]
    eih_img= eih_imgs[idx]


    # ---- save images ----
    rgb_path = os.path.join(out_dir, f"agentview_rgb_idx{idx}.png")
    eih_path = os.path.join(out_dir, f"eye_in_hand_rgb_idx{idx}.png")

    # 1️⃣ assume RGB, convert to BGR for OpenCV
    cv2.imwrite(rgb_path,img)
    cv2.imwrite(eih_path,eih_img)



    print(f"✅ Saved images:")






def to_RGB(hdf5_file):
    """
    将 hdf5 中所有 demo 的 agentview_rgb / eye_in_hand_rgb
    从 BGR 转换为 RGB（通道反转）
    """
    with h5py.File(hdf5_file, "r+") as f:
        for demo_key in f["data"].keys():          # e.g. demo_0, demo_1, ...
            obs_grp = f[f"data/{demo_key}/obs"]

            for key in ["agentview_rgb", "eye_in_hand_rgb"]:
                if key not in obs_grp:
                    continue

                imgs = obs_grp[key]                # shape: (T, H, W, 3)
                T = imgs.shape[0]

                for step_i in range(T):
                    img = imgs[step_i]
                    imgs[step_i] = img[:, :, ::-1] # BGR -> RGB





if __name__ == "__main__":
    # Example usage
    hdf5_file = "/data3/zhixu/LIBERO-PeRM/hdf5/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate.hdf5"
    hdf5_file = "/data3/zhixu/LIBERO-PeRM/hdf5/replayed_demo.hdf5"
    check_img(hdf5_file, out_dir="check_images", idx=0)
    # to_RGB(hdf5_file)
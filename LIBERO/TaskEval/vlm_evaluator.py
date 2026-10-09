"""
VLM Evaluator — 用 Vision-Language Model 评估机器人 subtask 是否完成。

工作原理:
  1. 从 HDF5 rollout 文件中提取关键帧（key frames）
     - HDF5 里存了每一步的 agentview RGB 图像、action 向量等
     - 关键帧 = Z轴加速度突变（抓取/放下动作）或 gripper 开合状态变化
  2. 把所有关键帧纵向拼接成一张长图（stitched image）
  3. 构造 prompt 描述要评估的 subtask，连同拼接图一起发给 VLM
  4. 解析 VLM 回答中的 Yes/No verdict，判断 subtask 是否完成
"""

import os
import sys
from typing import List, Tuple, Optional
from itertools import groupby
import textwrap

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from TaskEval.hdf5_player import HDF5_Player
from TaskGen.FMs import ask_FM_image

# ────────────────────────────────────────────────────────────────
# subtask 技能描述（供 VLM 理解每种技能的语义）
# ────────────────────────────────────────────────────────────────
subtask_description="""
- Pick
Generally pick everything

- Pick_bowl
Pick bowl/plate/open-cup shaped objects that need to pick the edges.

- Pick_bottle
Pick bottle-shaped objects that need to pick from the bottleneck.

- Pick_from_side
Pick any object from the side.

- Put
Generally put everything on sth/somewhere.

- Put_bowl
Put bowl/plate/open-cup shaped objects.

- Put_on
Put objects on top of sth.

- Push
Push objects toward sth or push certain direction.

"""

# 更完整的技能→自然语言描述映射，用于 prompt 中给 VLM 参考
subtask_description_dict = {
    "pick": "Generally pick everything",
    "pick_left_edge": "Pick objects by the edge.",
    "pick_back_edge": "Scoop the object up from underneath.",
    # "pick_back_edge_v0": "Pick objects from the back edge.",  # deprecated
    "pick_bottle": "Pick bottle-shaped objects that need to pick from the bottleneck.",
    "pick_from_side": "Pick any object from the side.",
    "pick_from_front": "Pick any object from the front.",
    "pick_horizontal": "Pick objects with the gripper fully horizontal.",
    "pick_handle": "Pick objects by the handle.",
    "put": "Generally put everything on sth/somewhere.",
    "put_left_edge": "Put objects by the edge.",
    "put_back_edge": "Put objects after scooping up.",
    # "put_back_edge_v0": "Put objects from the back edge.",  # deprecated
    "put_horizontal": "Put objects with horizontal gripper.",
    "put_handle": "Put objects by the handle.",
    "put_with_rotation": "Put objects while rotating.",
    "put_on": "Put objects on top of sth.",
    "push": "Push objects toward sth or push certain direction.",
    "drag": "Drag objects in a direction.",
    "throw": "Throw the object instead of putting it.",
    "pick_scoop": "Scoop the object up from underneath using a tilted gripper.",
    "put_scoop": "Put objects after scooping up.",
}

class VLM_Evaluator:
    """
    VLM-based subtask evaluator。

    使用流程:
        evaluator = VLM_Evaluator("path/to/rollout.hdf5", "gemini")
        evaluator.extract_key_frames()          # 从 HDF5 提取关键帧
        raw_text, success, verdict = evaluator.evaluate("(pick akita_black_bowl_1)")

    内部原理:
        - HDF5Player 加载 rollout 的 obs (agentview_rgb) 和 actions
        - extract_key_frames() 通过检测 action 向量的突变来筛选关键帧:
            * Z轴加速度突变 → 机械臂正在做竖直方向运动（抓/放）
            * gripper 状态变化 → 夹爪开/合（抓取或释放物体）
        - 关键帧纵向拼接成一张图，发给 VLM 做视觉判断
        - VLM 回答 Yes/No + 理由，程序解析 verdict
    """
    def __init__(self, hdf5_path, model_name: str):
        self.model_name = model_name
        self.frames = []
        # HDF5_Player 会加载 rollout 的 model_xml, states, obs, actions
        # 其中 obs["agentview_rgb"] 是每步的 agentview 相机图像
        self._player = HDF5_Player(hdf5_path)

    def extract_key_frames(self, interval = 10, acc_z_thres = 0.05):
        """
        从 rollout 中提取关键帧。

        关键帧的判据:
        1. Z轴加速度突变 (|action_t[2] - action_{t-1}[2]| > acc_z_thres)
           → 机械臂在 Z 方向有明显加速/减速，通常对应抬起或放下动作
           → 取该时刻的 agentview 图像，然后跳过 5 步避免重复
        2. Gripper 状态变化 (|action_t[6] - action_{t-1}[6]| > 0.5)
           → action[6] 是 gripper 的开合指令（0=开, 1=闭）
           → 变化 > 0.5 说明夹爪正在开或合（抓取/释放物体）
           → 同时取当前帧和 9 步后的帧（看到动作完成后的状态）

        action 向量格式: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                         [0]  [1]  [2]  [3]     [4]     [5]     [6]
        """
        t = 0
        cnt = 0
        while t < self._player.length:
            action_t = self._player.get_action(t)
            # 条件1: Z轴加速度突变 → 竖直方向运动
            if abs(action_t[2] - self._player.get_action(max(0, t-1))[2]) > acc_z_thres:
                cnt += 1
                print(f"no.{cnt}: significant acceleration in z axis at step {t}")
                self.frames.append(self._player.get_agentview(t))
                t += 5
            # 条件2: gripper 开合状态变化 → 抓取或释放
            elif abs(action_t[6] - self._player.get_action(max(0, t-1))[6]) > 0.5:
                cnt += 1
                print(f"no.{cnt}: gripper state changed at step {t}")
                # 取变化时刻 + 变化完成后 9 步的帧（两张图看前后对比）
                self.frames.append(self._player.get_agentview(t))
                self.frames.append(self._player.get_agentview(min(self._player.length, t + 9)))
                t += 10
            else:
                t += 1

    def stitch_frames(self):
        """
        将所有关键帧纵向拼接成一张长图。
        VLM 通过这张图一次性看到整个 rollout 的关键时刻。
        """
        if not self.frames:
            return None
        h, w, c = self.frames[0].shape
        stitched_image = np.zeros((h * len(self.frames), w, c), dtype=np.uint8)
        for i, frame in enumerate(self.frames):
            stitched_image[i*h:(i+1)*h] = frame
        return stitched_image

    def evaluate(self, subtask):
        """
        评估单个 subtask 是否完成。

        参数:
            subtask: 格式如 "(pick_left_edge akita_black_bowl_1)"

        返回:
            (raw_text, success, verdict)
            - raw_text: VLM 原始回答
            - success: bool, 是否判定为成功
            - verdict: "Yes" / "No" / "Unknown"

        流程:
            1. stitch_frames() → 拼接关键帧为一张图
            2. RGB→BGR 转换后保存为 PNG（cv2 用 BGR）
            3. 构造 prompt，包含 subtask 描述和技能说明
            4. 调用 ask_FM_image() 发送图片+prompt 给 VLM
            5. 正则解析 VLM 回答中的 Yes/No
        """
        if len(self.frames) == 0:
            print("No key frames extracted. Please run extract_key_frames() first.")
            return None, False
        stitched_image = self.stitch_frames()
        # agentview_rgb 是 RGB 格式，cv2.imwrite 需要 BGR
        stitched_image = stitched_image[:, :, ::-1]
        cv2.imwrite("stitched_image.png", stitched_image)
        vlm_prompt = textwrap.dedent(
            f""" \
            As an expert robot task evaluator, your job is to determine if a specific sub-task has been successfully completed by looking at an image.

            Context:
            - Main Goal: ""
            - Sub-task to evaluate: "{subtask}"
            And FYI: here are descriptions of subtasks that help you decide if they are completed:
            {textwrap.indent(subtask_description, " " * 12)}


            Question:
            Based on the image provided, has the sub-task "" been fully and correctly carried out?

            Answer with a "Yes" or "No", followed by a brief justification for your decision.
            """)
        print("\033[36m" + vlm_prompt + "\033[0m")
        response = ask_FM_image(image_path="stitched_image.png", instruction=vlm_prompt, model=self.model_name)
        print(response)

        # ── 解析 VLM 回答，提取 Yes/No verdict ──
        import re
        raw_text = str(response).strip()
        text_lower = raw_text.lower()

        # 优先匹配开头的 "Yes" 或 "No"
        match = re.match(r'^\s*(yes|no)\b[\s\:\-\,]*', text_lower)
        if match:
            verdict = match.group(1).capitalize()
            success = verdict == "Yes"
            justification = raw_text[len(match.group(0)):].strip()
        # 回答中只出现 yes（没有 no）→ 判定为 Yes
        elif re.search(r'\byes\b', text_lower) and not re.search(r'\bno\b', text_lower):
            verdict = "Yes"
            success = True
            justification = raw_text
        # 回答中只出现 no（没有 yes）→ 判定为 No
        elif re.search(r'\bno\b', text_lower) and not re.search(r'\byes\b', text_lower):
            verdict = "No"
            success = False
            justification = raw_text
        # yes 和 no 都出现或都没出现 → Unknown
        else:
            verdict = "Unknown"
            success = False
            justification = raw_text
        return raw_text, success, verdict

if __name__ == "__main__":
    eval = VLM_Evaluator("./rollout_0.hdf5", "gemini")
    eval.extract_key_frames()
    eval.evaluate("(pick_bowl akita_black_bowl)")

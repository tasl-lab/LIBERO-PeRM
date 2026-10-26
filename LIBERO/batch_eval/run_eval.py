"""
单个 mission 的 VLM 评估入口。

工作流程:
  1. 解析 mission_id → 定位 dataset/series_X/perm_X_Y/libero_NNN/ 目录
  2. 读取 mission_info.yaml 获取 task_plan（subtask 列表）
  3. 在 Eval/ 目录下查找 per-rollout HDF5 文件（由 batch_run 生成时自动复制过来的）
  4. 对每个 HDF5 文件，逐 subtask 调用 VLM_Evaluator 进行评估
  5. 结果（verdict + VLM 原文）写入 Eval/eval_results.yaml

VLM Evaluator 原理（详见 TaskEval/vlm_evaluator.py）:
  - 从 HDF5 的 obs["agentview_rgb"] 中提取关键帧（Z轴加速度突变 or gripper 开合变化）
  - 关键帧纵向拼接成一张长图
  - 构造 prompt + 图片发给 VLM（Gemini/GPT），VLM 回答 Yes/No
  - 解析 verdict，判断 subtask 是否完成

使用:
  python batch_eval/run_eval.py --mission_id "2-1_019" --model gemini
  python batch_eval/run_eval.py --mission_id "1-1_004" --model gemini --eval_fail_case
"""

import os
import sys
import re
import glob
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from TaskEval.vlm_evaluator import VLM_Evaluator


def parse_mission_id(mission_id: str):
    """
    解析 mission_id 为目录路径组件。
    '1-1_004'   → ('perm_1_1', 'libero_004')
    '3-1_004_1' → ('perm_3_1', 'libero_004_1')
    '2-7_ood1'  → ('perm_2_7', 'libero_ood1')
    """
    m = re.match(r'^(\d+-\d+)_(.+)$', mission_id)
    if not m:
        raise ValueError(f"Invalid mission_id format: {mission_id}")
    pairing_key, libero_id = m.group(1), m.group(2)
    perm_name = f"perm_{pairing_key.replace('-', '_')}"
    libero_name = f"libero_{libero_id}"
    return perm_name, libero_name


def find_hdf5_files(mission_dir: str, mission_id: str, eval_fail_case: bool):
    """
    在 Eval/ 目录下查找要评估的 HDF5 文件。

    搜索逻辑:
      - 主要: Eval/rollout_*.hdf5（成功 rollout，batch_run 时自动复制过来的）
      - 可选: HDF5/fail_case/rollout_*.hdf5（失败 rollout，需 --eval_fail_case）

    返回: [(hdf5_path, label), ...]
    """
    files = []

    # Eval/ 目录下的成功 rollout
    eval_dir = os.path.join(mission_dir, "Eval")
    if os.path.isdir(eval_dir):
        for p in sorted(glob.glob(os.path.join(eval_dir, "rollout_*.hdf5"))):
            label = os.path.splitext(os.path.basename(p))[0]  # e.g. "rollout_0"
            files.append((p, label))

    # fail_case rollouts（可选）
    if eval_fail_case:
        fail_dir = os.path.join(mission_dir, "HDF5", "fail_case")
        if os.path.isdir(fail_dir):
            for p in sorted(glob.glob(os.path.join(fail_dir, "rollout_*.hdf5"))):
                label = "fail_" + os.path.splitext(os.path.basename(p))[0]
                files.append((p, label))

    return files


def build_subtask_list(mission_info: dict):
    """
    从 mission_info.yaml 的 task_plan 构建 subtask 字符串列表。

    task_plan 格式:
      - [pick_left_edge, akita_black_bowl_1]
      - [put_left_edge, plate_1]

    输出:
      ["(pick_left_edge akita_black_bowl_1)", "(put_left_edge plate_1)"]

    这个格式和 VLM_Evaluator.evaluate() 的 subtask 参数一致。
    """
    task_plan = mission_info.get("task_plan", [])
    subtasks = []
    for step in task_plan:
        if isinstance(step, list) and len(step) >= 2:
            subtasks.append(f"({step[0]} {step[1]})")
        elif isinstance(step, list) and len(step) == 1:
            subtasks.append(f"({step[0]})")
    return subtasks


def evaluate_single_hdf5(hdf5_path: str, label: str, subtasks: list,
                         model: str, eval_dir: str):
    """
    对单个 HDF5 文件，逐 subtask 运行 VLM 评估。

    流程:
      1. 创建 VLM_Evaluator，加载 HDF5（读取 obs/actions/states）
      2. extract_key_frames(): 遍历 action 序列，检测 Z轴加速度突变和 gripper 变化
         → 提取对应时刻的 agentview 图像作为关键帧
      3. 对每个 subtask 调用 evaluate():
         → 关键帧拼接成长图 → 构造 prompt → 发给 VLM → 解析 Yes/No
      4. 保存 stitched 关键帧图片到 Eval/ 目录
    """
    print(f"\n  ▶ Evaluating: {label}")
    print(f"    HDF5: {hdf5_path}")

    results = {
        "hdf5": os.path.basename(hdf5_path),
        "subtasks": [],
        "summary": {"total": 0, "success": 0, "fail": 0, "unknown": 0},
    }

    # Step 1: 加载 HDF5 — HDF5_Player 会读取 model_xml, states, obs, actions
    evaluator = VLM_Evaluator(hdf5_path, model)

    # Step 2: 提取关键帧（基于 action 向量的突变检测）
    evaluator.extract_key_frames()

    if not evaluator.frames:
        print("    ⚠ No key frames extracted, skipping.")
        results["error"] = "no_key_frames"
        return results

    # Step 3: 对每个 subtask 调用 VLM 评估
    for subtask in subtasks:
        print(f"    Subtask: {subtask}")
        raw_text, success, verdict = evaluator.evaluate(subtask)
        results["subtasks"].append({
            "subtask": subtask,
            "verdict": verdict,
            "success": success,
            "response": raw_text,
        })
        results["summary"]["total"] += 1
        if verdict == "Yes":
            results["summary"]["success"] += 1
        elif verdict == "No":
            results["summary"]["fail"] += 1
        else:
            results["summary"]["unknown"] += 1

    # Step 4: 保存关键帧拼接图到 Eval/ 目录
    import cv2
    stitched = evaluator.stitch_frames()
    if stitched is not None:
        stitched_path = os.path.join(eval_dir, f"{label}_stitched.png")
        cv2.imwrite(stitched_path, stitched[:, :, ::-1])  # RGB→BGR

    total = results["summary"]["total"]
    ok = results["summary"]["success"]
    print(f"    ✅ {ok}/{total} subtasks passed")
    return results


def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation for a single mission")
    parser.add_argument("--mission_id", type=str, required=True,
                        help="Mission ID like '1-1_004' or '3-1_004_1'")
    parser.add_argument("--model", type=str, default="gemini",
                        help="VLM model name (default: gemini)")
    parser.add_argument("--eval_fail_case", action="store_true", default=False,
                        help="Also evaluate fail_case rollouts from HDF5/fail_case/")
    args = parser.parse_args()

    # ── 定位 mission 目录 ──
    perm_name, libero_name = parse_mission_id(args.mission_id)
    from configs.path_utils import mission_dir as _mdir
    mission_dir = _mdir(perm_name, libero_name)

    if not os.path.isdir(mission_dir):
        print(f"❌ Mission directory not found: {mission_dir}")
        sys.exit(1)

    # ── 读取 mission_info.yaml ──
    info_path = os.path.join(mission_dir, "mission_info.yaml")
    with open(info_path, "r") as f:
        mission_info = yaml.safe_load(f) or {}

    language_inst = mission_info.get("language_instructions", "")
    high_pref = mission_info.get("high_pref_lan", "")
    low_pref = mission_info.get("low_pref_lan", "")

    print(f"╔{'═'*60}╗")
    print(f"║  Mission: {args.mission_id:<48} ║")
    print(f"║  Task: {language_inst[:50]:<51} ║")
    if high_pref:
        print(f"║  High pref: {high_pref[:46]:<47} ║")
    if low_pref:
        print(f"║  Low pref: {low_pref[:47]:<48} ║")
    print(f"╚{'═'*60}╝")

    # ── 从 task_plan 构建 subtask 列表 ──
    # task_plan 里每个 step 是 [skill_name, object_name]
    # 转成 "(skill_name object_name)" 格式给 VLM evaluator
    subtasks = build_subtask_list(mission_info)
    if not subtasks:
        print("⚠ No subtasks found in task_plan, skipping.")
        sys.exit(0)
    print(f"  Subtasks: {subtasks}")

    # ── 查找 Eval/ 目录下的 HDF5 文件 ──
    hdf5_files = find_hdf5_files(mission_dir, args.mission_id, args.eval_fail_case)
    if not hdf5_files:
        print(f"⚠ No HDF5 files found in {mission_dir}/Eval/")
        sys.exit(0)
    print(f"  Found {len(hdf5_files)} HDF5 file(s)")

    # ── Eval 目录（存结果 + stitched 图片） ──
    eval_dir = os.path.join(mission_dir, "Eval")
    os.makedirs(eval_dir, exist_ok=True)

    # ── 逐 HDF5 评估 ──
    all_results = {
        "mission_id": args.mission_id,
        "model": args.model,
        "language_instructions": language_inst,
        "high_pref_lan": high_pref,
        "low_pref_lan": low_pref,
        "task_plan": mission_info.get("task_plan", []),
        "rollouts": [],
    }

    for hdf5_path, label in hdf5_files:
        result = evaluate_single_hdf5(
            hdf5_path, label, subtasks, args.model, eval_dir
        )
        all_results["rollouts"].append(result)

    # ── 保存结果到 Eval/eval_results.yaml ──
    results_path = os.path.join(eval_dir, "eval_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  📄 Results saved to: {results_path}")


if __name__ == "__main__":
    main()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import pandas as pd



"""
Example Usage:

1. Run with GPT-5.1 (OpenAI):
   python run_task_vlm_evaluation.py --model gpt

2. Run with Qwen3-VL (Local Server):
   # Ensure serve_qwen.py is running first!
   python run_task_vlm_evaluation.py --model qwen

3. Run with Gemini (Google):
   # Ensure GEMINI_API_KEY is set
   python run_task_vlm_evaluation.py --model gemini

4. Run on a specific trajectory (e.g. pref0) for testing:
   python run_task_vlm_evaluation.py --model qwen --test-pref-id 0
"""

def list_rollouts(task_folder):
    """
    返回 task_folder 下所有 rollout_xxx 子目录（按数字排序）。
    例如：
        ['rollout_0', 'rollout_1', ..., 'rollout_50']
    """
    if not os.path.isdir(task_folder):
        return []

    rollout_pattern = re.compile(r"^rollout_(\d+)$")
    rollouts = []

    for name in os.listdir(task_folder):
        full_path = os.path.join(task_folder, name)
        if os.path.isdir(full_path):
            m = rollout_pattern.match(name)
            if m:
                rollouts.append((int(m.group(1)), name))  # (数字, 文件夹名)

    # 按 rollout number 排序
    rollouts_sorted = [name for _, name in sorted(rollouts, key=lambda x: x[0])]
    return rollouts_sorted

def list_task_folders(eval_folder):
    """
    返回 eval_folder 下所有形如 Task_xxx 的子目录名字。
    """
    if not os.path.isdir(eval_folder):
        raise ValueError(f"Path not found or not a directory: {eval_folder}")

    task_dirs = []
    pattern = re.compile(r"^Task_\w+$")   # 匹配 Task_0 / Task_naive / Task_10 等等

    for name in os.listdir(eval_folder):
        full_path = os.path.join(eval_folder, name)
        if os.path.isdir(full_path) and pattern.match(name):
            task_dirs.append(name)

    return sorted(task_dirs)

def _extract_pref_and_subtask(image_path: str):
    """
    Parse filenames like 'pref0_step3_pick_orange_juice_1.png' and return (pref_id, step_idx).

    Zhixu: now it's like:
    'task1_step2_move_to_approach_alphabet_soup_1.png'
    """
    name = os.path.basename(image_path)
    m = re.search(r"task(\d+)_subtask(\d+)_", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))





def run_task_level_evaluation(
    log_path: str,
    out_path: str,
    test_task_id=None,
    model_provider: str = "gpt",
):
    """
    Group rows in the step-level evaluation log into trajectories (by pref id),
    feed the full image sequence + sub-task descriptions to the VLM once per
    trajectory, and write a compact task-level log.
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    df = pd.read_csv(log_path)

    # Group by (main_goal, pref_id)
    trajectories = {}
    for idx, row in df.iterrows():
        image_path = row["image_path"]
        task_id, subtask_idx = _extract_pref_and_subtask(image_path)
        if task_id is None or subtask_idx is None:
            continue
        key = (row["main_goal"], task_id)
        trajectories.setdefault(key, []).append((subtask_idx, idx))

    print(f"[Task VLM Eval] Found {len(trajectories)} trajectories.")

    records = []
    for (main_goal, task_id), step_list in trajectories.items():
        if test_task_id is not None and task_id != test_task_id:
            continue

        # Sort by step index to preserve temporal order
        step_list_sorted = sorted(step_list, key=lambda x: x[0])
        indices = [idx for _, idx in step_list_sorted]
        traj_rows = df.loc[indices]

        sub_tasks = traj_rows["sub_task"].tolist()
        image_paths = traj_rows["image_path"].tolist()

        print(f"\n[Task VLM Eval] Evaluating trajectory task{task_id} with {len(image_paths)} frames using {model_provider}...")
        start_time = time.time()
        raw_response, standardized_answer = evaluate_task_sequence(
            main_goal,
            sub_tasks,
            image_paths,
            collage_folder=os.path.dirname(log_path),
            trajectory_id=f"task{task_id}",
            model_provider=model_provider,
        )
        duration = time.time() - start_time

        records.append(
            {
                "main_goal": main_goal,
                "task_id": task_id,
                "num_steps": len(image_paths),
                "vlm_prediction": standardized_answer,
                "vlm_reason": raw_response,
                "model": model_provider,
                "duration_sec": round(duration, 2)
            }
        )

        # Save incrementally in case of long runs
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"[Task VLM Eval] Updated task-level log for task{task_id} (took {duration:.2f}s).")
        print(f"[Task VLM Eval] VLM decision for task{task_id}: {standardized_answer}")
        print(f"[Task VLM Eval] VLM reasoning for task{task_id}: {raw_response}")

    print("\nTask-level evaluation complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run task-level VLM evaluation over trajectories.")

    parser.add_argument(
        "--data_folder",
        type=str,
        default="dataset/high_plus_low/Layout_0/HighPref_1",
        help="Path to the step-level evaluation_log.csv",
    )


    parser.add_argument(
        "--test_task_id",
        type=int,
        default=None,
        help="If set, only evaluate this single pref trajectory (e.g., 0, 1, ...).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt", "qwen", "gemini"],
        default="gemini",
        help="Model provider to use: 'gpt' (OpenAI), 'qwen' (Local), or 'gemini'.",
    )

    args = parser.parse_args()
    task_folder = args.data_folder
    eval_folder= os.path.join(task_folder,"Evaluation_Images")

    for _task in list_task_folders (eval_folder):

        task_dir = os.path.join (eval_folder, _task)
        out_path = os.path.join (task_dir, "task_evaluation_log.csv")

        # 遍历 rollouts
        for _rollout in list_rollouts (task_dir):
            log_path = os.path.join (task_dir, _rollout, "evaluation_log.csv")

            run_task_level_evaluation (
                log_path=log_path,
                out_path=out_path,
                test_task_id=args.test_task_id,
                model_provider=args.model,
            )


if __name__ == "__main__":
    main()
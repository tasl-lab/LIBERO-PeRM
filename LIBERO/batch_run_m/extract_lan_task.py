import json
import os
import re
from typing import Dict, List, Optional


def extract_objects_from_bddl (bddl_path: str) -> List [str]:
    with open (bddl_path, 'r', encoding='utf-8') as f:
        content = f.read ()

    init_match = re.search (r'\(:init\s*(.*?)\s*\)\s*\(:goal', content, re.DOTALL)
    if not init_match:
        print (f"Warning: Cannot find :init section in {bddl_path}")
        return []

    init_content = init_match.group (1)
    objects = set ()
    pattern = r'\((on|in)\s+([a-zA-Z_]+_\d+)\s+'
    matches = re.findall (pattern, init_content, re.IGNORECASE)

    for _, obj_name in matches:
        objects.add (obj_name)

    return sorted (list (objects))


def find_bddl_file (layout_id: str, dataset_dir: str = "dataset") -> Optional [str]:
    possible_folders = [
        os.path.join (dataset_dir, f"Layout_{layout_id}"),
        os.path.join (dataset_dir, f"layout_{layout_id}"),
    ]

    layout_folder = None
    for folder in possible_folders:
        if os.path.exists (folder):
            layout_folder = folder
            break

    if layout_folder is None:
        print (f"Warning: Layout folder not found for layout_id {layout_id}")
        return None

    bddl_files = [f for f in os.listdir (layout_folder) if f.startswith ("P_BENCH") and f.endswith (".bddl")]

    if len (bddl_files) == 0:
        print (f"Warning: No P_BENCH*.bddl file found in {layout_folder}")
        return None
    elif len (bddl_files) > 1:
        print (f"ERROR: Multiple P_BENCH*.bddl files found in {layout_folder}: {bddl_files}")
        print ("Please resolve this manually before continuing.")
        return None

    return os.path.join (layout_folder, bddl_files [0])


def extract_task_plan (mission_id: str, layout_id: str, dataset_dir: str = "dataset") -> Optional [str]:
    possible_folders = [
        os.path.join (dataset_dir, f"Layout_{layout_id}"),
        os.path.join (dataset_dir, f"layout_{layout_id}"),
    ]

    layout_folder = None
    for folder in possible_folders:
        if os.path.exists (folder):
            layout_folder = folder
            break

    if layout_folder is None:
        print (f"Warning: Layout folder not found for layout_id {layout_id}")
        return None

    # 尝试多种可能的路径
    possible_paths = [
        (os.path.join (layout_folder, f"Mission_{mission_id}", f"High_pref_for_mission_{mission_id}.txt"), False),
        (os.path.join (layout_folder, f"High_pref_for_mission_{mission_id}.txt"), False),
        (os.path.join (layout_folder, "naive_task_plan_response.txt"), True),
    ]

    txt_path = None
    is_naive = False
    for path, naive_flag in possible_paths:
        if os.path.exists (path):
            txt_path = path
            is_naive = naive_flag
            break

    if txt_path is None:
        print (f"Warning: Task file not found for mission {mission_id} in {layout_folder}")
        return None

    if is_naive:
        print (f"  Using naive_task_plan_response.txt for mission {mission_id}")

    with open (txt_path, 'r', encoding='utf-8') as f:
        content = f.read ().strip ()

    if is_naive:
        # naive格式: (pick cream_cheese_1),(put plate_1)
        # 直接返回内容，转换为统一格式
        # 将 (action obj) 转换为 ('action', 'obj')
        task_plan = re.sub (r'\((\w+)\s+(\w+)\)', r"('\1', '\2')", content)
    else:
        # High_pref格式: 1.|| ... ||
        match = re.search (r'1\.\|\|\s*(.*?)\s*\|\|', content, re.DOTALL)
        if not match:
            print (f"Warning: Cannot find task plan (1.||...||) in {txt_path}")
            return None
        task_plan = match.group (1).strip ()

    # 去掉参数 |...|
    task_plan_cleaned = re.sub (r'\|[^|]+\|', '', task_plan)
    task_plan_cleaned = re.sub (r'_+(\)|,|\')', r'\1', task_plan_cleaned)

    return task_plan_cleaned


def sort_by_mission_id (missions: List [Dict]) -> List [Dict]:
    def sort_key (item):
        mid = item.get ("mission_id", "")
        if "naive" in mid.lower ():
            return (2, 0, mid)
        if mid.startswith ("m"):
            num_match = re.search (r'\d+', mid)
            num = int (num_match.group ()) if num_match else 0
            return (1, num, mid)
        try:
            return (0, int (mid), mid)
        except ValueError:
            return (2, 0, mid)

    return sorted (missions, key=sort_key)


def process_all_missions (json_path: str = "batch_run/refined_pref_s1.json",
                          dataset_dir: str = "dataset",
                          output_path: str = "batch_run/extracted_missions_task_lan.json") -> List [Dict]:
    with open (json_path, 'r', encoding='utf-8') as f:
        missions = json.load (f)

    results = []

    for mission in missions:
        mission_id = mission.get ("mission_id", "")
        layout_id = mission.get ("layout_id", "")

        print (f"Processing mission {mission_id} (layout {layout_id})...")

        result = {
            "mission_id": mission_id,
            "layout_id": layout_id,
            "language_instructions": mission.get ("language_instructions", ""),
            "high_pref_lan": mission.get ("high_pref_lan", ""),
            "low_pref_lan": mission.get ("low_pref_lan", ""),
            "source_file": mission.get ("source_file", ""),
            "layout_objs": [],
            "task_plan_raw": ""
        }

        bddl_path = find_bddl_file (layout_id, dataset_dir)
        if bddl_path:
            result ["layout_objs"] = extract_objects_from_bddl (bddl_path)

        task_plan = extract_task_plan (mission_id, layout_id, dataset_dir)
        if task_plan:
            result ["task_plan_raw"] = task_plan

        results.append (result)

    results = sort_by_mission_id (results)

    with open (output_path, 'w', encoding='utf-8') as f:
        json.dump (results, f, indent=2, ensure_ascii=False)

    print (f"\nResults saved to {output_path}")
    print (f"Processed {len (results)} missions")

    return results


if __name__ == "__main__":
    process_all_missions ()
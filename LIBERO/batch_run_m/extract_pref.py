import json
import os
import re
from pathlib import Path


def extract_mission_data (cfgs_dir: str = "cfgs"):
    """
    从cfgs目录下读取所有layout*.json文件，提取mission信息
    """
    results = []

    # 获取所有以layout开头的json文件
    cfgs_path = Path (cfgs_dir)
    layout_files = list (cfgs_path.glob ("layout*.json"))

    for file_path in layout_files:
        try:
            with open (file_path, 'r', encoding='utf-8') as f:
                data = json.load (f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print (f"Warning: Failed to read {file_path}: {e}")
            continue

        # 提取所需字段
        mission_ids = data.get ("mission_ids", [])
        language_instructions = data.get ("language_instructions", "")
        high_pref_lan = data.get ("high_pref_lan", [])
        low_pref_lan = data.get ("low_pref_lan", [])
        layout_id = data.get ("layout_id", file_path.stem)

        # 遍历每个mission
        for i, mission_id in enumerate (mission_ids):
            # 跳过以naive开头的mission
            if str (mission_id).startswith ("naive"):
                continue

            # 提取对应索引的preference（如果存在）
            high_pref = high_pref_lan [i] if i < len (high_pref_lan) else ""
            low_pref = low_pref_lan [i] if i < len (low_pref_lan) else ""

            results.append ({
                "mission_id": str (mission_id),
                "layout_id": layout_id,
                "language_instructions": language_instructions,
                "high_pref_lan": high_pref,
                "low_pref_lan": low_pref,
                "source_file": file_path.name
            })

    # 排序：先按类型（纯数字 vs m+数字），再按数字大小
    def sort_key (item):
        mid = item ["mission_id"]
        # 检查是否是 m+数字 格式
        if mid.startswith ("m"):
            num = int (re.search (r'\d+', mid).group ()) if re.search (r'\d+', mid) else 0
            return (1, num)  # 1表示m类型，排后面
        else:
            try:
                num = int (mid)
                return (0, num)  # 0表示纯数字类型，排前面
            except ValueError:
                return (2, 0)  # 其他类型排最后

    results.sort (key=sort_key)

    return results


if __name__ == "__main__":
    cfgs_directory = "cfgs"

    missions = extract_mission_data (cfgs_directory)

    # 导出为JSON
    output_file = "batch_run/extracted_missions.json"
    with open (output_file, "w", encoding="utf-8") as f:
        json.dump (missions, f, indent=2, ensure_ascii=False)

    print (f"已导出 {len (missions)} 个mission到 {output_file}")
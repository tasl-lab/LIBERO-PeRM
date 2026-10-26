

import json
import os
import re
from typing import List
from datetime import datetime
def _clean_layout_ids(layout_ids: List[str]) -> List[str]:
    """
    Strip spaces and drop empty layout ids.
    """
    cleaned = []
    for lid in layout_ids:
        lid = str(lid).strip()
        if lid:
            cleaned.append(lid)
    return cleaned
def _load_mission_ids(cfg_path: str) -> List[str]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mids = cfg.get("mission_ids", [])
    return [str(x).strip() for x in mids if x is not None]
def _filter_valid_mission_ids(mission_ids: List[str], valid_mission_ids: List[str]) -> List[str]:
    """
    Rules:
      - drop anything starting with 'naive' (case-insensitive)
      - keep:
          * digits: "87"
          * m + digits: "m41"
    """
    valid = []
    valid_set = set(str(x).strip().lower() for x in valid_mission_ids)
    for mid in mission_ids:
        mid = mid.strip()
        if not mid:
            continue

        if mid.lower().startswith("naive"):
            continue

        if re.fullmatch(r"\d+", mid):
            if valid_mission_ids and mid.lower() not in valid_set:
                continue
            valid.append(mid)
            continue

        if re.fullmatch(r"[mM]\d+", mid):
            if valid_mission_ids and mid.lower() not in valid_set:
                continue
            valid.append(mid.lower())

    def sort_key(x):
        if x.startswith('m'):
            return (1, int(x[1:]))
        return (0, int(x))

    valid.sort(key=sort_key)

    return valid

def _generate_command_block(mission_id: str, batch_size: int,overwrite_old:int,use_forcibly:int) -> str:
    return f"""mid="{mission_id}"
python main.py \\
  --naive_llm_plan 0 \\
  --run_layout_gen 0 \\
  --high_pref_gpt_plan 0 \\
  --low_pref_gpt_plan 0 \\
  --generate_evaluation_data 0 \\
  --enable_collision_check 1 \\
  --batch_size {batch_size} \\
  --batch_start 0 \\
  --mis_id "$mid" \\
  --cfg "$CFG" \\
  --manually_set_offset 1 \\
  --check_init_layout 0 \\
  --gen_in_batch 1 \\
  --overwrite_old {overwrite_old} \\
  --use_forcibly {use_forcibly} || true
  
python main.py \\
  --naive_llm_plan 0 \\
  --run_layout_gen 0 \\
  --high_pref_gpt_plan 0 \\
  --low_pref_gpt_plan 0 \\
  --generate_evaluation_data 0 \\
  --enable_collision_check 1 \\
  --batch_size {batch_size} \\
  --batch_start {batch_size} \\
  --mis_id "$mid" \\
  --cfg "$CFG" \\
  --manually_set_offset 1 \\
  --check_init_layout 0 \\
  --gen_in_batch 1 \\
  --overwrite_old {overwrite_old} \\
  --use_forcibly {use_forcibly} || true
  
python main.py \\
  --naive_llm_plan 0 \\
  --run_layout_gen 0 \\
  --high_pref_gpt_plan 0 \\
  --low_pref_gpt_plan 0 \\
  --generate_evaluation_data 0 \\
  --enable_collision_check 1 \\
  --batch_size {batch_size} \\
  --batch_start {batch_size*2} \\
  --mis_id "$mid" \\
  --cfg "$CFG" \\
  --manually_set_offset 1 \\
  --check_init_layout 0 \\
  --gen_in_batch 1 \\
  --overwrite_old {overwrite_old} \\
  --use_forcibly {use_forcibly} || true

"""


def generate_script_from_batch_run(
    batch_run_json: str,
    *,
    cfg_dir: str = "cfgs",
    output_dir: str = ".",
    output_name: str = "batch_run_m.sh",
):

    """
    Take in a batch_run.json and generate a runnable .sh script.
    """
    with open(batch_run_json, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)


    layout_ids = _clean_layout_ids(run_cfg.get("layout_ids", []))
    batch_size = int(run_cfg.get("data_batch_size", 1))
    overwrite_old=int(run_cfg.get("overwrite_old", None))
    use_forcibly=int(run_cfg.get("use_forcibly", None))

    if not layout_ids:
        raise ValueError("No valid layout_ids found in batch_run.json")

    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, output_name)



    total_SR_log = os.path.join ("dataset", f"total_SR_log_m.txt")
    # total_SR_log = os.path.join("/data4/zhixu/PBenchData", f"total_SR_log_m.txt")

    # 在 total_SR_log 文件开头写入分割符
    with open (total_SR_log, "a", encoding="utf-8") as log_f:
        log_f.write ("\n" + "=" * 80 + "\n")
        log_f.write (f"// New Run - {datetime.now ().strftime ('%m-%d %H:%M')}\n")
        log_f.write ("=" * 80 + "\n\n")



    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        f.write("# Auto-generated from batch_run.json\n\n")



        total_mission_list=[]
        for layout_id in layout_ids:
            cfg_path = os.path.join(cfg_dir, f"layout{layout_id}.json")

            if not os.path.exists(cfg_path):
                f.write(f'echo "[WARN] cfg not found: {cfg_path}, skip"\n\n')
                continue

            mission_ids = _load_mission_ids(cfg_path)
            valid_mids_cfg = run_cfg.get("valid_mission_ids", None)

            valid_mids = _filter_valid_mission_ids(mission_ids,valid_mission_ids=valid_mids_cfg)

            f.write(f'echo "===== Layout {layout_id} ====="\n')
            f.write(f'CFG="{cfg_path}"\n\n')

            if not valid_mids:
                f.write('echo "[INFO] no valid mission ids, skip"\n\n')
                continue

            for mid in valid_mids:

                total_mission_list.append(mid)
                f.write(_generate_command_block(mid, batch_size,overwrite_old=overwrite_old,use_forcibly=use_forcibly))
                f.write("\n")

            f.write("\n")
    total_mission_list.sort(key=lambda x: (1, int(x[1:])) if x.startswith('m') else (0, int(x)))
    print(f"Total valid mission number: {len(total_mission_list)}")
    print(f"Total mission list: {total_mission_list}")
    os.chmod(script_path, 0o755)
    return script_path

if __name__ == '__main__':
    script = generate_script_from_batch_run ("batch_run_m/batch_run_m.json")
    print (f"Generated: {script}")


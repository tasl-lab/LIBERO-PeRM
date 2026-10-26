import yaml
import os
from typing import List


def _generate_eval_block(mission_id: str, model: str, eval_fail_case: bool) -> str:
    label = f"Eval {mission_id}"
    padded = label.center(48)
    bar = "══" * 24
    cmd = f"""echo ""
echo "📊📊📊 {bar} 📊📊📊"
echo "📊📊📊 {padded} 📊📊📊"
echo "📊📊📊 {bar} 📊📊📊"
echo ""
python batch_eval/run_eval.py \\
  --mission_id "{mission_id}" \\
  --model "{model}" """
    if eval_fail_case:
        cmd += "\\\n  --eval_fail_case "
    cmd += "|| true\n\n"
    return cmd


def generate_eval_script(
    batch_eval_yaml: str,
    *,
    output_dir: str = ".",
    output_name: str = "batch_eval.sh",
):
    with open(batch_eval_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mission_ids: List[str] = cfg.get("mission_ids", [])
    model = cfg.get("model", "gemini")
    eval_fail_case = cfg.get("eval_fail_case", False)

    if not mission_ids:
        raise ValueError("No mission_ids found in batch_eval.yaml")

    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, output_name)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        f.write("# Auto-generated from batch_eval.yaml\n\n")
        for mid in mission_ids:
            f.write(_generate_eval_block(str(mid), model, eval_fail_case))

    os.chmod(script_path, 0o755)
    print(f"Total missions to evaluate: {len(mission_ids)}")
    print(f"Model: {model}")
    print(f"Generated: {script_path}")
    return script_path


if __name__ == "__main__":
    generate_eval_script("batch_eval/batch_eval.yaml")
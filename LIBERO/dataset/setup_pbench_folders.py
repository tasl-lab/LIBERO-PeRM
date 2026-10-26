import re
import yaml
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).parent

with open(DATASET_DIR / "perm_missions.yaml") as f:
    data = yaml.safe_load(f)

tasks = data.get("tasks", {})
preferences = data.get("preferences", {})


def extract_bddl_info(bddl_path: Path) -> dict:
    """Extract workspace and language_instructions from a BDDL file."""
    text = bddl_path.read_text()
    workspace = ""
    language = ""

    # Extract language from (:language ...)
    lang_match = re.search(r"\(:language\s+(.+?)\)", text)
    if lang_match:
        language = lang_match.group(1).strip()

    # Workspace from filename prefix (e.g., LIVING_ROOM_SCENE1_put_... -> LIVING_ROOM_SCENE1)
    # Convention: everything before the first lowercase segment
    fname = bddl_path.stem  # without .bddl
    # Split on the task description part (first lowercase word)
    ws_match = re.match(r"^([A-Z0-9_]+?)_(?=[a-z])", fname)
    if ws_match:
        workspace = ws_match.group(1).lower()

    return {"workspace": workspace, "language_instructions": language}


for key, mission_ids in data["pairings"].items():
    folder_name = f"perm_{key.replace('-', '_')}"
    series_name = f"series_{key.split('-')[0]}"
    perm_dir = DATASET_DIR / series_name / folder_name

    for mid in mission_ids:
        mid_str = str(mid)
        # For IDs like "004_1", strip suffix to find source mission_004
        base_mid = re.sub(r'_\d+$', '', mid_str) if re.search(r'_\d+$', mid_str) else mid_str
        # OOD IDs like "OOD1" -> source dir "mission_ood1"
        if base_mid.startswith("OOD"):
            mission_dir = DATASET_DIR / "LIBERO_mission" / f"mission_{base_mid.lower()}"
        else:
            mission_dir = DATASET_DIR / "LIBERO_mission" / f"mission_{base_mid}"
        bddl_files = list(mission_dir.glob("*.bddl"))
        if not bddl_files:
            print(f"WARNING: No bddl file found in {mission_dir}")
            continue

        libero_dir = perm_dir / f"libero_{mid_str}"
        libero_dir.mkdir(parents=True, exist_ok=True)

        for bddl in bddl_files:
            dest = libero_dir / bddl.name
            shutil.copy2(bddl, dest)
            print(f"Copied {bddl.name} -> {libero_dir.relative_to(DATASET_DIR)}/")

        # Generate mission_info.yaml
        bddl_file = bddl_files[0]
        info = extract_bddl_info(bddl_file)

        compound_id = f"{key}_{mid_str}"
        task_description = tasks.get(compound_id, info["language_instructions"])

        # Determine pref_lan based on series
        pref_text = preferences.get(compound_id, "")
        high_pref_lan = ""
        low_pref_lan = ""
        if key.startswith(("2-", "3-")):
            high_pref_lan = pref_text
        elif key.startswith("1-"):
            low_pref_lan = pref_text

        mission_info = {
            "workspace": info["workspace"],
            "language_instructions": task_description,
            "bddl_file": bddl_file.name,
            "high_pref_lan": high_pref_lan,
            "low_pref_lan": low_pref_lan,
            "subtask_offset": {},
            "naive_plan": [],
            "task_plan": [],
            "high_prefs": [],
            "low_prefs": [],
        }

        mission_info_path = libero_dir / "mission_info.yaml"
        with open(mission_info_path, "w") as f:
            yaml.dump(mission_info, f, default_flow_style=False, sort_keys=False)
        print(f"  Generated {mission_info_path.relative_to(DATASET_DIR)}")

print("Done!")

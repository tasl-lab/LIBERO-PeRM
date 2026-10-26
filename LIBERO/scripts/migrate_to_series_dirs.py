"""Dry-run migration plan: dataset/perm_X_Y/ → dataset/series_X/perm_X_Y/

Prints (a) directory moves and (b) code/yaml/doc edits needed.
Does NOT apply changes by default. Use --apply to execute (NOT recommended;
review plan first, then run edits via tooling).

Usage:
    python LIBERO/scripts/migrate_to_series_dirs.py            # dry-run
    python LIBERO/scripts/migrate_to_series_dirs.py --output plan.md
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "LIBERO" / "dataset"


def collect_moves() -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []
    for entry in sorted(DATASET_DIR.iterdir()):
        if not entry.is_dir():
            continue
        m = re.fullmatch(r"perm_(\d+)_\d+", entry.name)
        if not m:
            continue
        series = m.group(1)
        target = DATASET_DIR / f"series_{series}" / entry.name
        moves.append((entry, target))
    return moves


# (file, line range, kind, snippet, suggested patch)
CODE_SITES = [
    ("LIBERO/main.py", 129,
     "path-build",
     'mission_dir = os.path.join("dataset", perm_name, libero_name)',
     'mission_dir = os.path.join("dataset", _series_for(perm_name), perm_name, libero_name)'),
    ("LIBERO/PolicyEval/eval.py", 65,
     "path-build",
     'DATASET_ROOT = os.path.join(_LIBERO_ROOT, "dataset")',
     'DATASET_ROOT = os.path.join(_LIBERO_ROOT, "dataset")  # +helper _series_for() applied at use site'),
    ("LIBERO/PolicyEval/eval.py", 154,
     "path-build",
     'perm, libero = parse_mission_id(mission_id)',
     'perm, libero = parse_mission_id(mission_id); series = _series_for(perm)'),
    ("LIBERO/configs/check_suite_sync.py", 48,
     "path-build",
     'os.path.join(DATASET_ROOT, f"perm_{m.group(1).replace(\'-\',\'_\')}", f"libero_{m.group(2)}")',
     'os.path.join(DATASET_ROOT, f"series_{m.group(1).split(\'-\')[0]}", f"perm_...", f"libero_{m.group(2)}")'),
    ("LIBERO/batch_eval/run_eval.py", 178,
     "path-build",
     'mission_dir = os.path.join("dataset", perm_name, libero_name)',
     'mission_dir = os.path.join("dataset", _series_for(perm_name), perm_name, libero_name)'),
    ("LIBERO/Utility/replay_hdf5_to_video.py", 51,
     "path-build",
     'LIBERO_ROOT / "dataset" / perm / libero / "HDF5" / ...',
     'LIBERO_ROOT / "dataset" / _series_for(perm) / perm / libero / "HDF5" / ...'),
    ("LIBERO/Utility/replay_hdf5_to_video.py", 62,
     "path-build",
     'mission_dir = LIBERO_ROOT / "dataset" / perm / libero',
     'mission_dir = LIBERO_ROOT / "dataset" / _series_for(perm) / perm / libero'),
    ("LIBERO/build_visualizer_v2.py", 27,
     "path-build (constant)",
     'DATASET_DIR = LIBERO_DIR / "dataset"',
     'unchanged; usage sites use _series_for()'),
    ("LIBERO/build_visualizer_v2.py", 222,
     "path-build",
     'perm_name = f"perm_{parsed[\'series\']}_{parsed[\'task\']}"',
     'mission_dir = DATASET_DIR / f"series_{parsed[\'series\']}" / perm_name / libero_name'),
    ("LIBERO/dataset/fill_plans.py", 439,
     "path-build",
     'folder = f"perm_{pairing.replace(\'-\', \'_\')}"',
     'folder = os.path.join(f"series_{pairing.split(\'-\')[0]}", f"perm_{pairing.replace(\'-\', \'_\')}")'),
    ("LIBERO/dataset/setup_perm_folders.py", 38,
     "path-build",
     'folder_name = f"perm_{key.replace(\'-\', \'_\')}"',
     'folder_name = os.path.join(f"series_{key.split(\'-\')[0]}", f"perm_{key.replace(\'-\', \'_\')}")'),
    ("LIBERO/dataset/_propagate_libero_instruction.py", 0,
     "glob",
     'glob "dataset/perm_*/libero_*/mission_info.yaml"',
     'glob "dataset/series_*/perm_*/libero_*/mission_info.yaml"'),
    ("LIBERO/dataset/_propagate_pref_rename.py", 0,
     "glob",
     'similar glob over dataset/perm_*',
     'glob via dataset/series_*/perm_*/...'),
    ("LIBERO/dataset/_propagate_task_optimized.py", 0,
     "glob",
     'similar',
     'similar'),
    ("LIBERO/dataset/export_single_pref_csv.py", 0,
     "glob",
     'walks dataset/perm_X_Y/libero_*',
     'walks dataset/series_*/perm_X_Y/libero_*'),
    ("LIBERO/layout_adjuster.py", 107,
     "path-build",
     'mission_dir = os.path.join("dataset", suite_task_dir, f"libero_{libero_idx}")',
     'add _series_for() prefix'),
    ("LIBERO/skill_tuner.py", 304,
     "path-build", "(same shape)", 'add _series_for() prefix'),
    ("LIBERO/teleop_analyzer.py", 238,
     "path-build", "(same shape)", 'add _series_for() prefix'),
    ("LIBERO/teleop_collector.py", 163,
     "path-build", "(same shape)", 'add _series_for() prefix'),
    ("LIBERO/batch_run/extract_lan_task.py", 27,
     "path-build", "find_bddl_file uses dataset_dir param", 'callers pass series-aware path'),
    ("LIBERO/batch_run_m/extract_lan_task.py", 27,
     "path-build", "(same as batch_run/extract_lan_task.py)", 'same fix'),
    ("VLA_train/post_training_script/convert_perm_hdf5_to_lerobot_singpref.py", 0,
     "glob/path-build", "reads dataset/perm_*/libero_*/mission_info.yaml", 'glob dataset/series_*/perm_*/...'),
    ("LIBERO/configs/task_suites.yaml", 9,
     "comment", "Each mission_id resolves to dataset/perm_X_Y/libero_ZZZ[_N]/", 'update to dataset/series_X/perm_X_Y/...'),
    ("LIBERO/configs/check_suite_sync.py", 7,
     "comment", "docstring", 'update'),
    ("LIBERO/batch_eval/run_eval.py", 5,
     "comment", "docstring", 'update'),
    ("LIBERO/visualizer_v2/README.md", 42,
     "doc", "scan paths", 'update glob'),
    ("INSTALL.md", 121,
     "doc", "example output path", 'update'),
]


HELPER_SNIPPET = '''
# Add to LIBERO/configs/path_utils.py (NEW FILE) — single source of truth:

import os, re

def series_of(perm_name: str) -> str:
    """e.g. 'perm_3_15' -> 'series_3'."""
    m = re.fullmatch(r"perm_(\\d+)_\\d+", perm_name)
    if not m:
        raise ValueError(f"bad perm_name: {perm_name!r}")
    return f"series_{m.group(1)}"

def mission_dir(perm_name: str, libero_name: str, dataset_root: str = "dataset") -> str:
    return os.path.join(dataset_root, series_of(perm_name), perm_name, libero_name)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None, help="write plan to this file (default stdout)")
    ap.add_argument("--apply-moves", action="store_true",
                    help="execute git mv directory moves (does NOT touch code)")
    args = ap.parse_args()

    moves = collect_moves()
    out = []
    out.append("# Migration Plan: dataset/perm_X_Y → dataset/series_X/perm_X_Y\n")
    out.append(f"## Phase 1: Directory moves ({len(moves)} dirs)\n")
    out.append("```")
    for src, dst in moves:
        rel_src = src.relative_to(REPO_ROOT)
        rel_dst = dst.relative_to(REPO_ROOT)
        out.append(f"git mv {rel_src} {rel_dst}")
    out.append("```\n")

    out.append("## Phase 2: Helper module\n")
    out.append(HELPER_SNIPPET)

    out.append("## Phase 3: Code/config/doc edits\n")
    out.append("| File | Line | Kind | Note |")
    out.append("|------|-----:|------|------|")
    for fp, ln, kind, snippet, _ in CODE_SITES:
        out.append(f"| `{fp}` | {ln} | {kind} | {snippet[:60]} |")
    out.append("")

    out.append("## Phase 4: Validation\n")
    out.append("- run `python LIBERO/configs/check_suite_sync.py` (must pass for SinglePref/Compose)")
    out.append("- run `python -c \"from LIBERO.configs import path_utils; print(path_utils.mission_dir('perm_3_1','libero_004_1'))\"`")
    out.append("- spot-check: `ls dataset/series_3/perm_3_1/libero_004_1/mission_info.yaml`")

    text = "\n".join(out)
    if args.output:
        Path(args.output).write_text(text)
        print(f"plan written to {args.output}")
    else:
        print(text)

    if args.apply_moves:
        print("\n--- applying git mv ---")
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "mv", str(src.relative_to(REPO_ROOT)),
                            str(dst.relative_to(REPO_ROOT))],
                           cwd=REPO_ROOT, check=True)
        print("done.")


if __name__ == "__main__":
    main()

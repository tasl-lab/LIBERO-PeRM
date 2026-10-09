import os.path
import logging
from step1_data_gen import gen_data_in_scale, ask_choice
import json
import yaml
from libero.libero.envs import OffScreenRenderEnv
from Utility.frame import get_img_from_obs



def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_obs_image(bddl_file,path):
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 512,
        "camera_widths": 512,
    }
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()

    img = get_img_from_obs(obs, key="agentview_image")


    if path is not None:
        import cv2
        cv2.imwrite (path, cv2.cvtColor (img, cv2.COLOR_RGB2BGR))
        print(f"The initial layout image is saved at {path}, please check it to decide the high preference image")

    env.close()
    return img

import argparse
def int2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    v = str(v).lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        "Boolean value expected (0/1 or true/false)"
    )


def parse_mission_id(mission_id: str):
    """
    Parse compound mission_id like '1-1_004' or '3-1_004_1' into directory components.
    Returns (perm_dir_name, libero_dir_name).
    E.g. '1-1_004'   -> ('perm_1_1', 'libero_004')
         '3-1_004_1' -> ('perm_3_1', 'libero_004_1')
         '2-7_ood1'  -> ('perm_2_7', 'libero_ood1')
    """
    import re
    m = re.match(r'^(\d+-\d+)_(.+)$', mission_id)
    if not m:
        raise ValueError(f"Invalid mission_id format: {mission_id}. Expected 'X-Y_NNN'.")
    pairing_key, libero_id = m.group(1), m.group(2)
    perm_name = f"perm_{pairing_key.replace('-', '_')}"
    libero_name = f"libero_{libero_id}"
    return perm_name, libero_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser (
        description="P_Bench main pipeline"
    )
    parser.add_argument ("--check_init_layout", type=int2bool, default=None)
    parser.add_argument ("--run_layout_gen", type=int2bool, default=None)
    # Legacy GPT flags (kept for backward compatibility with batch scripts, not used)
    parser.add_argument ("--naive_llm_plan", type=int2bool, default=None)
    parser.add_argument ("--run_naive_plan", type=int2bool, default=None)
    parser.add_argument ("--high_pref_gpt_plan", type=int2bool, default=None)
    parser.add_argument ("--low_pref_gpt_plan", type=int2bool, default=None)
    parser.add_argument ("--generate_evaluation_data", type=int2bool, default=None)
    parser.add_argument ("--gen_in_batch", type=int2bool, default=None)
    parser.add_argument ("--enable_collision_check", type=int2bool, default=None)
    parser.add_argument("--batch_size", type=int, default=50,
                        help="In data_gen with --max_attempts set, interpreted as target "
                             "number of successful demos; otherwise fixed number of attempts.")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="data_gen target-success cap. If set, keep running until "
                             "batch_size successes OR max_attempts attempts (whichever first). "
                             "Default None = preserve legacy fixed-count behavior.")
    parser.add_argument("--manually_set_offset",type=int,default=0)
    parser.add_argument ("--improve_fluency_flag", type=int, default=0)
    parser.add_argument ("--batch_start", type=int)
    parser.add_argument("--overwrite_old",type=int,default=None)
    parser.add_argument("--use_forcibly",type=int,default=None)
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose debug output for motion generation")

    # New: per-mission compound ID (e.g. "1-1_004")
    parser.add_argument("--mission_id", type=str, default=None,
                        help="Compound mission ID like '1-1_004'. Uses perm folder structure.")
    # Legacy: layout-based config
    parser.add_argument("--cfg", type=str, default=None,
                        help="Path to legacy layout config JSON (mutually exclusive with --mission_id)")
    parser.add_argument("--mis_id", type=str, default="Test",
                        help="Legacy mission ID filter (used with --cfg)")
    parser.add_argument("--tasksuite", type=str, default=None,
                        help="Task suite name (e.g. SinglePref). Merged HDF5 goes to "
                             "VLA_data/{tasksuite}/Mission_{mission_id}.hdf5 when set.")
    parser.add_argument("--n_save_video", type=int, default=None,
                        help="Save video only for the first N rollouts per mission. "
                             "Unset = save all (legacy behavior).")
    parser.add_argument("--save_eval_hdf5", action="store_true",
                        help="Also copy each successful rollout HDF5 to <mission>/Eval/ "
                             "for downstream VLM evaluation. Off by default (doubles disk usage).")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save rollout GIFs under <mission>/GIFs/. Off by default.")
    parser.add_argument("--debug_markers", action="store_true",
                        help="Render the heuristic-skill debug spheres into the sim "
                             "views. Off by default — never use when generating data "
                             "(the markers bake into agentview_rgb).")
    parser.add_argument("--gen_naive_anchor", action="store_true",
                        help="Naive-anchor mode: force low_pref2apply=[] (no low-level "
                             "motion pref) so demos serve as the score-0 dual-anchor "
                             "baseline. Merged demos go to VLA_data/<tasksuite>_naive/ "
                             "so the GT-pref Mission_<id>.hdf5 is left untouched.")
    parser.add_argument("--use_pref_layout", type=int2bool, default=None,
                        help="With --gen_naive_anchor: ignore the mission's "
                             "no_pref_layout/ bddl and run the naive plan in the pref "
                             "scene, so pref and naive differ only in the preference "
                             "(1-8's no_pref_layout swaps in a non-rotated object asset).")
    parser.add_argument("--pair_init_from", type=str, default=None,
                        help="Path to a pref Mission_<id>.hdf5; each rollout replays "
                             "that file's demo-k init_state so naive demos start from "
                             "exactly the pref layouts (paired comparison). Use when the "
                             "naive plan differs from task_plan and RNG-seeded layouts "
                             "would otherwise drift a few cm.")
    parser.add_argument("--force_naive_plan", type=int2bool, default=None,
                        help="With --gen_naive_anchor: execute mission_info's naive_plan "
                             "even for missions whose pref lives in the plan skills "
                             "(e.g. 1-8 pick_with_rotation). Default off — series-1 "
                             "anchors normally keep their task_plan grasp variants.")

    flags = parser.parse_args()

    # Debug markers are OFF by default (they render into saved agentview_rgb
    # and cause visual OOD at eval) — opt in with --debug_markers.
    from MotionGen.heuristic_skill.heuristic import set_data_gen_mode
    set_data_gen_mode(not flags.debug_markers)

    if flags.mission_id:
        # ===== New per-mission path =====
        perm_name, libero_name = parse_mission_id(flags.mission_id)
        from configs.path_utils import mission_dir as _mission_dir_path
        mission_dir = _mission_dir_path(perm_name, libero_name)

        if not os.path.isdir(mission_dir):
            raise FileNotFoundError(f"Mission directory not found: {mission_dir}")

        # Configure logging: MotionGen debug → file (+ console if --verbose)
        # Root logger stays at WARNING to suppress third-party noise (PIL, h5py, etc.)
        log_folder = os.path.join(mission_dir, "Logs")
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, f"run_{flags.mission_id}.log")
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
        mg_logger = logging.getLogger("MotionGen")
        mg_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(fmt)
        mg_logger.addHandler(fh)
        if flags.verbose:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            mg_logger.addHandler(sh)
        print(f"  📝 Log file: {log_file}")

        # Load mission_info.yaml
        info_path = os.path.join(mission_dir, "mission_info.yaml")
        with open(info_path, "r") as f:
            mission_info = yaml.safe_load(f) or {}

        # Find the .bddl file
        bddl_filename = mission_info.get("bddl_file", "")
        if not bddl_filename:
            bddl_files = [f for f in os.listdir(mission_dir) if f.endswith(".bddl")]
            if not bddl_files:
                raise FileNotFoundError(f"No .bddl file found in {mission_dir}")
            bddl_filename = bddl_files[0]

        workspace = mission_info.get("workspace", "")
        language_instructions = mission_info.get("language_instructions", "")
        high_pref_lan = mission_info.get("high_pref_lan", "")
        low_pref_lan = mission_info.get("low_pref_lan", "")
        subtask_offset = mission_info.get("subtask_offset", {})
        damping_level = mission_info.get("damping_level", 0)  # forced to 0 (LIBERO-aligned, no tail decel stutter)
        damping_level = 0  # global override 2026-04-23
        skill_debug = mission_info.get("skill_debug", False)

        check_init_layout = flags.check_init_layout
        if check_init_layout is None:
            check_init_layout = ask_choice(
                "Check initial layout image?",
                default=0
            )
        if check_init_layout:
            bddl_file = os.path.join(mission_dir, bddl_filename)
            init_image_folder = "dataset/init_layout_img/"
            os.makedirs(init_image_folder, exist_ok=True)
            get_obs_image(bddl_file, path=os.path.join(init_image_folder, f"init_img_{flags.mission_id}.png"))
            ask_choice("Press any key to continue", no_default=True)

        if flags.manually_set_offset:
            subtask_offset_dict_list = [subtask_offset]
        else:
            subtask_offset_dict_list = [{}]

        gen_data_in_scale(
            workspaces=workspace,
            bddl_file_names=bddl_filename,
            lan_insts=language_instructions,
            layout_id=f"{perm_name}/{libero_name}",
            high_pref_lans=[high_pref_lan] if high_pref_lan else [""],
            low_pref_lans=[low_pref_lan] if low_pref_lan else [""],
            flags=flags,
            dataset_ids=[flags.mission_id],
            subtask_offset_list=subtask_offset_dict_list,
            mission_mask=[1],
            mission_dir=mission_dir,
            damping_level=damping_level,
            skill_debug=skill_debug,
            tasksuite=flags.tasksuite,
            n_save_video=flags.n_save_video,
            save_eval_hdf5=flags.save_eval_hdf5,
            save_gif=flags.save_gif,
        )

    elif flags.cfg:
        # Fallback logging for legacy path — only MotionGen, no third-party noise
        if flags.verbose:
            mg_logger = logging.getLogger("MotionGen")
            mg_logger.setLevel(logging.DEBUG)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("  %(message)s"))
            mg_logger.addHandler(sh)
        # ===== Legacy layout-based path =====
        cfg = load_config(flags.cfg)

        check_init_layout = flags.check_init_layout
        if check_init_layout is None:
            check_init_layout = ask_choice(
                "Check initial layout image?",
                default=0
            )

        if check_init_layout:
            bddl_file = cfg["bddl_file_names"]
            bddl_file = os.path.join("libero/libero/bddl_files", bddl_file)
            init_image_folder = "dataset/init_layout_img/"
            os.makedirs(init_image_folder, exist_ok=True)
            layout_id = cfg["layout_id"]
            get_obs_image(bddl_file, path=os.path.join(init_image_folder, f"init_img_L{layout_id}.png"))
            ask_choice("Press any key to continue", no_default=True)

        if flags.manually_set_offset:
            subtask_offset_dict_list = cfg.get("subtask_offset", [])
        else:
            subtask_offset_dict_list = []
        if cfg["fluency_improvement"] == 1:
            flags.improve_fluency_flag = 1

        gen_data_in_scale(
            workspaces=cfg["workspaces"],
            bddl_file_names=cfg["bddl_file_names"],
            lan_insts=cfg["language_instructions"],
            layout_id=cfg["layout_id"],
            high_pref_lans=cfg["high_pref_lan"],
            low_pref_lans=cfg["low_pref_lan"],
            flags=flags,
            dataset_ids=cfg["mission_ids"],
            subtask_offset_list=subtask_offset_dict_list,
            mission_mask=cfg["mission_mask"],
            tasksuite=flags.tasksuite,
            n_save_video=flags.n_save_video,
            save_eval_hdf5=flags.save_eval_hdf5,
            save_gif=flags.save_gif,
        )
    else:
        parser.error("Either --mission_id or --cfg must be provided.")

    # Force-exit to skip C-level destructors (MuJoCo/mujoco-py)
    # that cause "free(): invalid pointer" on process teardown.
    os._exit(0)

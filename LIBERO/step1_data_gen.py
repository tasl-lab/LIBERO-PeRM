"""
Generate heuristic trajectories as videos and hdf5 files.
"""
import sys
import os
import logging
from datetime import datetime
from sre_parse import FLAGS

logger = logging.getLogger("MotionGen")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from TaskGen.task import TaskPlan
from libero.libero.envs import OffScreenRenderEnv
import os
os.environ["MUJOCO_GL"] = "egl"   # 启用 GPU(EGL) 渲染
os.environ.setdefault("EGL_DEVICE_ID", "0")  # 多卡时可选：指定第0号GPU
from MotionGen.heuristic_skill.heuristic import move_to, pick, pick_left_edge, pick_right_edge, pick_back_edge, pick_z_90, pick_with_rotation, put, put_left_edge, put_right_edge, open_drawer,close_drawer, turn_on_stove
from TaskGen.task_gen.naive_task_gen import gen_naive_task_plan
from TaskGen.layout_gen.layout import Layout
from TaskGen.layout_gen.pref_layout_gen import gen_preference_layouts
from Preference.pref_gen import gen_low_pref,gen_and_apply_high_pref
from Utility.concatenate_hdf5 import concatenate_hdf5_demos
import glob
from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from MotionGen.motion_gen import execute_plan_with_acc_feedback
from Utility.tool_func import to_sim_env_name
skill_names = [fn.__name__ for fn in [pick, pick_left_edge, pick_right_edge, pick_back_edge, pick_z_90, pick_with_rotation, put, put_left_edge, put_right_edge, open_drawer, close_drawer, turn_on_stove]]



def get_path_dict(output_dir: str, layout_cnt: str, dataset_id, clear_old_file=False, mission_dir=None) -> dict:
    """
    Build path dictionary for outputs.

    If mission_dir is provided (new per-mission perm structure), the mission
    directory IS the output folder -- no extra Mission_{id} nesting.
    Otherwise falls back to the legacy Layout_{layout_cnt}/Mission_{dataset_id} structure.
    """
    path_dict = {}

    if mission_dir is not None:
        # ===== New perm per-mission structure =====
        layout_folder = mission_dir
        os.makedirs(layout_folder, exist_ok=True)
        path_dict["naive_plan"] = layout_folder
        path_dict["layout_folder"] = layout_folder
        path_dict["output_data_folder"] = layout_folder

        if dataset_id is not None:
            # In perm mode, mission folder IS the output -- no nesting
            high_pref_folder = layout_folder
            path_dict["high_level_plan"] = high_pref_folder
            path_dict["low_pref"] = high_pref_folder

            for sub in ["Plots", "Videos", "HDF5", "Evaluation_Images"]:
                sub_path = os.path.join(high_pref_folder, sub)
                os.makedirs(sub_path, exist_ok=True)

            path_dict["hdf5_folder"] = os.path.join(high_pref_folder, "HDF5")
            path_dict["video_folder"] = os.path.join(high_pref_folder, "Videos")
            path_dict["plot_folder"] = os.path.join(high_pref_folder, "Plots")
            path_dict["evaluation_image_folder"] = os.path.join(high_pref_folder, "Evaluation_Images")
            path_dict["evaluation_data_folder"] = path_dict["evaluation_image_folder"]

            if clear_old_file:
                video_folder = os.path.join(high_pref_folder, "Videos")
                for f in glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True):
                    os.remove(f)

        return path_dict

    # ===== Legacy layout-based structure =====
    layout_folder=os.path.join (output_dir, f"Layout_{layout_cnt}")
    os.makedirs (layout_folder, exist_ok=True)
    path_dict ["naive_plan"] = layout_folder
    path_dict ['layout_folder'] = layout_folder
    path_dict ["output_data_folder"]=output_dir

    if dataset_id is not None:
        high_pref_folder=os.path.join (layout_folder, f"Mission_{dataset_id}")
        os.makedirs (high_pref_folder, exist_ok=True)
        path_dict ["high_level_plan"] = high_pref_folder
        path_dict ['low_pref'] = high_pref_folder
        plot_folder=os.path.join (high_pref_folder, f"Plots")
        os.makedirs (plot_folder, exist_ok=True)
        video_folder=os.path.join (high_pref_folder, f"Videos")
        os.makedirs (video_folder, exist_ok=True)
        hdf5_folder=os.path.join (high_pref_folder, f"HDF5")
        os.makedirs (hdf5_folder, exist_ok=True)

        path_dict ["hdf5_folder"] =hdf5_folder
        path_dict ['video_folder'] = video_folder
        path_dict ['plot_folder'] = plot_folder
        evaluation_image_folder= os.path.join (high_pref_folder, f"Evaluation_Images")
        path_dict ["evaluation_image_folder"] = evaluation_image_folder
        os.makedirs (evaluation_image_folder, exist_ok=True)
        path_dict ["evaluation_data_folder"]=evaluation_image_folder

        if clear_old_file:
            output_data_folder = os.path.join (path_dict ["output_data_folder"], f"Mission_{dataset_id}")
            video_folder = os.path.join (output_data_folder, "Videos")
            for f in glob.glob (os.path.join (video_folder, "**/*.mp4"), recursive=True):
                os.remove (f)

    return path_dict






WIDTH = 64

def stage_start(number, title):
    label = f"  [{number}] {title}  "
    inner = label.ljust(WIDTH - 2, " ")
    print(f"\n{'╔' + '═' * (WIDTH - 2) + '╗'}")
    print(f"{'║'}{inner}{'║'}")
    print(f"{'╚' + '═' * (WIDTH - 2) + '╝'}")
    logger.info(f"=== START [{number}] {title} ===")

def stage_done(number, title):
    label = f" [{number}] {title} ✔ "
    pad = WIDTH - len(label) - 2
    print(f"──{label}{'─' * max(pad, 0)}\n")
    logger.info(f"=== DONE  [{number}] {title} ===")


def ask_choice(prompt: str, default: int = 0, no_default=False) -> bool:
    """
    prompt: 提示语
    default: 默认值 (0 or 1)
    return: True if user chooses 1, else False
    """
    if not no_default:
        msg = f"{prompt} (1 = Yes, 0 = No; default = {default}): "
    else:
        msg = f"{prompt}"
    choice = input(msg).strip()

    if choice == "":
        choice = str(default)

    return choice == "1"

def gen_naive_task(bddl_file_name,lan_inst,workspace,path_dict,mission_dir=None)->(Layout,TaskPlan):
    stage_start("1/5", "Naive Task Plan Generation")

    if mission_dir is not None:
        naive_bddl_file = os.path.join(mission_dir, bddl_file_name)
        # naive-anchor mode: series-3 missions keep a no_pref_layout/ bddl with
        # the SAME scene (pref objects included) but the naive :goal — required
        # so the naive rollout's BDDL done check targets the naive object.
        # --use_pref_layout keeps the pref bddl for the naive run too. 1-8's
        # no_pref_layout swaps the object asset (rotated_butter -> butter), so
        # the two rows of the comparison video would differ in scene, not just
        # in the preference. Opt in when a paired comparison matters more than
        # the naive-friendly object pose.
        if (getattr(FLAGS, "gen_naive_anchor", False)
                and not getattr(FLAGS, "use_pref_layout", False)):
            np_dir = os.path.join(mission_dir, "no_pref_layout")
            if os.path.isdir(np_dir):
                np_bddl = os.path.join(np_dir, bddl_file_name)
                if not os.path.exists(np_bddl):
                    cands = [f for f in os.listdir(np_dir) if f.endswith(".bddl")]
                    if len(cands) != 1:
                        raise FileNotFoundError(
                            f"no_pref_layout has {len(cands)} bddl files, expected 1: {np_dir}")
                    np_bddl = os.path.join(np_dir, cands[0])
                print(f"  [NaiveAnchor] Using no_pref_layout bddl: {np_bddl}")
                naive_bddl_file = np_bddl
    else:
        problem_folder = os.path.join(os.path.dirname(__file__), "libero/libero/bddl_files")
        naive_bddl_file = os.path.join(problem_folder, bddl_file_name)

    if not os.path.exists(naive_bddl_file):
        raise FileNotFoundError(f"Could not find BDDL file: {naive_bddl_file}")

    layout = Layout(bddl_file=naive_bddl_file,lan_inst=lan_inst,bddl_workspace=workspace)

    task_plan=gen_naive_task_plan(layout,skill_names,path_dict=path_dict)

    return layout,task_plan



def gen_high_task(task_plan,layout,high_pref,dataset_id,path_dict,force_naive=False)->(Layout,TaskPlan):
    stage_start("2/5", "High Level Task Generation")

    gen_and_apply_high_pref(task_plan, layout, high_level_pref=high_pref,dataset_id=dataset_id,path_dict=path_dict,force_naive=force_naive)

    stage_done("2/5", "High Level Task Generation")
    return layout,task_plan



def gen_low_task(task_plan:TaskPlan,layout:Layout,path_dict ,dataset_id,low_pref_lan)->(Layout,TaskPlan):
    stage_start("3/5", "Low Level Preference Generation")

    gen_low_pref (task_plan, layout, path_dict=path_dict,dataset_id=dataset_id,low_pref_lan=low_pref_lan)

    stage_done("3/5", "Low Level Preference Generation")
    return layout,task_plan


def gen_demo(layout:Layout,task_plan:TaskPlan,path_dict,dataset_ids,layout_id,mission_mask,subtask_offset_list=None,damping_level=1,skill_debug=False,tasksuite=None,n_save_video=None,save_eval_hdf5=False,save_gif=False):
    stage_start("4/5", "Layout Generation")

    gen_pos4high_pref=False

    """Preference-aware Layout Generation"""
    bddl_path_txt = os.path.join (path_dict ["layout_folder"], "bddl_path.txt")
    bddl_output_folder = path_dict ['layout_folder']
    if FLAGS.run_layout_gen is None:
        run_layout_gen = ask_choice (
            "Run layout generation?",
            default=0  # in this case, default is no
        )
    else:
        run_layout_gen = FLAGS.run_layout_gen
    if run_layout_gen:
        layout = gen_preference_layouts (layout=layout, task=task_plan, gen_pos4high_pref=gen_pos4high_pref,
                                         bddl_output_folder=bddl_output_folder,
                                         enable_collision_check=FLAGS.enable_collision_check,
                                         layout_id=layout_id,
                                         )
        bddl_location = layout.bddl_file

        # 📝 写入 txt 文件
        with open (bddl_path_txt, "w") as f:
            f.write (bddl_location)
    else:
        # 📖 从 txt 读取 BDDL 路径，找不到则用 layout 自带的路径
        # naive-anchor 模式必须用 layout 自带 bddl（可能是 no_pref_layout 的），
        # bddl_path.txt 是 pref 跑留下的、指向 pref bddl，不能沿用。
        if getattr(FLAGS, "gen_naive_anchor", False):
            bddl_location = layout.bddl_file
            print(f"[NaiveAnchor] Using layout.bddl_file (skip bddl_path.txt): {bddl_location}")
        elif os.path.exists(bddl_path_txt):
            with open (bddl_path_txt, "r") as f:
                bddl_location = f.read ().strip ()
        else:
            bddl_location = layout.bddl_file
            print(f"bddl_path.txt not found, using layout.bddl_file: {bddl_location}")
    skill_params_path = os.path.join(bddl_output_folder, "skill_params.txt")
    if os.path.exists(skill_params_path):
        task_plan.load_skill_params(bddl_output_folder)

    stage_done("4/5", "Layout Generation")



    stage_start("5/5", "Motion Generation")

    generate_evaluation_data=FLAGS.generate_evaluation_data
    if generate_evaluation_data is None:
        generate_evaluation_data=ask_choice(
            "Generate data log for later evaluation? ",
            default=1  # in this case, default is yes
        )


    env_args = {
        # "bddl_file_name": layout.bddl_file,
        "bddl_file_name": bddl_location,
        # define your own bddl file here
        "camera_heights": 256,  # 128, follow the setting in libero dataset
        "camera_widths": 256,  # 1024
        "camera_names": ["agentview", "frontview_closer","frontview","robot0_eye_in_hand"],
        # Use OSC_position control in default
    }
    # 初始化环境
    env = OffScreenRenderEnv (**env_args, ignore_done=True)
    try:
        env.reset ()
        print(f"  🕹️ Env initialized: {bddl_location}")
        logger.info(f"🕹️ Env initialized: {bddl_location}")
        for obj_name in getattr(layout, "obj_of_interest", []):
            sim_name = to_sim_env_name(obj_name)
            try:
                obj_pos = env.sim.data.get_body_xpos(sim_name).copy()
                obj_quat = env.sim.data.get_body_xquat(sim_name).copy()
                obj_quat_xyzw = np.array([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
                obj_euler = R.from_quat(obj_quat_xyzw).as_euler("XYZ")
                logger.info(
                    f"[init_pose] {sim_name} pos={np.round(obj_pos, 4)} "
                    f"euler_deg={np.round(np.degrees(obj_euler), 2)} yaw_deg={np.degrees(obj_euler[2]):.2f}"
                )
            except Exception as exc:
                logger.warning(f"[init_pose] Failed to read pose for {sim_name}: {exc}")
        ## debug
        # for body_name in env.sim.model.body_names:
        #     print (f"Body name in the env: {body_name}")
        ## debug


        num_of_done_lan_pref ={}
        total_lan_pref={}
        step_counts_per_mission = {}
        hdf5_folders=[]
        hdf5_save_path=[]
        gen_in_batch = FLAGS.gen_in_batch
        if FLAGS.batch_size is not None:
            batch_size = FLAGS.batch_size
            batch_start = FLAGS.batch_start if FLAGS.batch_start is not None else 0
        else:
            batch_size = 170
            batch_start = 0

        _offset = 0
        for _cnt, dataset_id in enumerate(dataset_ids):
            subtask_offset = subtask_offset_list[_cnt]
            if mission_mask[_cnt] == 0 or (dataset_id != getattr(FLAGS, 'mis_id', 'Test') and getattr(FLAGS, 'mis_id', 'Test') != "Test"):
                print(f"Skipping mission id:{dataset_id} as per mission mask.")
                _offset += 1
                continue

            dataset_id_cnt = _cnt - _offset
            print(len(task_plan.high_level_prefs))

            high_task_pref = task_plan.high_level_prefs[dataset_id_cnt]
            if high_task_pref is None:
                _plan = task_plan.get_real_plans()
            else:
                _plan = task_plan.get_real_plans(high_task_pref.single_pref_plan)

            is_done, hdf5_folder, num_steps = execute_plan_with_acc_feedback(
                env,
                _plan=_plan,
                dataset_id=dataset_id,
                task_plan=task_plan,
                path_dict=path_dict,
                high_pref=high_task_pref,
                low_pref2apply=([] if getattr(FLAGS, "gen_naive_anchor", False)
                                else task_plan.low_level_prefs_list[dataset_id_cnt]),
                generate_evaluation_data=generate_evaluation_data,
                batch_id=batch_start,
                subtask_offset=subtask_offset,
                damping_level=damping_level,
                skill_debug=skill_debug,
                n_save_video=n_save_video,
                save_eval_hdf5=save_eval_hdf5,
                save_gif=save_gif,
            )
            hdf5_folders.append(hdf5_folder)
            hdf5_save_path.append(dataset_id)
            num_of_done_lan_pref[f"{dataset_id}"] = is_done
            total_lan_pref[f"{dataset_id}"] = 1
            step_counts_per_mission.setdefault(f"{dataset_id}", []).append(num_steps)

        stage_start("SR", "BDDL Goal State Success Rate (Preview)")
        for key in num_of_done_lan_pref.keys():
            msg = f"Mission {key} success rate: {num_of_done_lan_pref[key]}/{total_lan_pref[key]}"
            print(f"  {msg}")
            logger.info(msg)

        if gen_in_batch is None:
            ask_choice(
                f">>>>>>>>>Please check the generate demo videos and plots and adjust the offset if needed.\n FYI: the objects we have in layout are:{layout.object_set}\n>>>>>>>>>Press and key to continue.",
                default=0
            )

            gen_in_batch = ask_choice(
                f"Generate more demos in batch? ({batch_size} demo per task)",
                default=0
            )

        if gen_in_batch:
            _offset = 0
            for _cnt, dataset_id in enumerate(dataset_ids):
                subtask_offset = subtask_offset_list[_cnt]
                if mission_mask[_cnt] == 0 or (dataset_id != getattr(FLAGS, 'mis_id', 'Test') and getattr(FLAGS, 'mis_id', 'Test') != "Test"):
                    print(f"Skipping mission id:{dataset_id} as per mission mask.")
                    _offset += 1
                    continue

                dataset_id_cnt = _cnt - _offset
                high_task_pref = task_plan.high_level_prefs[dataset_id_cnt]
                if high_task_pref is None:
                    _plan = task_plan.get_real_plans()
                else:
                    _plan = task_plan.get_real_plans(high_task_pref.single_pref_plan)

                # Loop semantics:
                #   --max_attempts set (data_gen): target = batch_size successes, cap = max_attempts attempts
                #   --max_attempts unset (dev/test): legacy fixed-count; cap = batch_size attempts
                # 首次 execute_plan 已在上方跑过，贡献 1 attempt (+ is_done success)。
                _max_attempts = getattr(FLAGS, "max_attempts", None)
                cap_attempts = _max_attempts if _max_attempts is not None else batch_size
                target_successes = batch_size

                batch_id = batch_start + 1
                while (num_of_done_lan_pref[f"{dataset_id}"] < target_successes
                       and total_lan_pref[f"{dataset_id}"] < cap_attempts):
                    is_done, _, num_steps = execute_plan_with_acc_feedback(
                        env,
                        _plan=_plan,
                        dataset_id=dataset_id,
                        task_plan=task_plan,
                        path_dict=path_dict,
                        high_pref=high_task_pref,
                        low_pref2apply=([] if getattr(FLAGS, "gen_naive_anchor", False)
                                        else task_plan.low_level_prefs_list[dataset_id_cnt]),
                        generate_evaluation_data=generate_evaluation_data,
                        batch_id=batch_id,
                        subtask_offset=subtask_offset,
                        damping_level=damping_level,
                        skill_debug=skill_debug,
                        n_save_video=n_save_video,
                        save_eval_hdf5=save_eval_hdf5,
                        save_gif=save_gif,
                    )
                    num_of_done_lan_pref[f"{dataset_id}"] += is_done
                    total_lan_pref[f"{dataset_id}"] += 1
                    step_counts_per_mission.setdefault(f"{dataset_id}", []).append(num_steps)
                    succ = num_of_done_lan_pref[f"{dataset_id}"]
                    tot = total_lan_pref[f"{dataset_id}"]
                    print(f"Mission {dataset_id} success rate: {succ}/{tot} (target {target_successes}, cap {cap_attempts})")
                    batch_id += 1

                succ = num_of_done_lan_pref[f"{dataset_id}"]
                tot = total_lan_pref[f"{dataset_id}"]
                if succ >= target_successes:
                    print(f"[OK] Mission {dataset_id}: reached target {target_successes} successes in {tot} attempts.")
                else:
                    print(f"[WARN] Mission {dataset_id}: stopped at {succ}/{tot} — cap {cap_attempts} reached before hitting target {target_successes}.")

            stage_start("SR", "BDDL Goal State Success Rate (Batch)")
            save_data = []
            sr_list = []

            total_SR_log = os.path.join("dataset", "total_SR_log.txt")
            sr_summary_log = os.path.join("batch_run", "sr_summary.txt")
            for key in num_of_done_lan_pref.keys():
                sr = num_of_done_lan_pref[key] / total_lan_pref[key]
                steps = step_counts_per_mission.get(key, [])
                avg_steps = sum(steps) / len(steps) if steps else 0
                msg = f"Mission {key} success rate: {num_of_done_lan_pref[key]}/{total_lan_pref[key]}  |  Avg Steps: {avg_steps:.1f} ({len(steps)} eps)"
                print(f"  {msg}")
                logger.info(msg)
                sr_list.append(sr)

                save_current = True
                if sr <= 0.6:
                    warn = f"Warning: Mission {key} has low success rate: {sr:.2f}."
                    print(f"  {warn}")
                    logger.info(warn)
                    if FLAGS.use_forcibly is None:
                        use_forcibly = ask_choice("Do you want to use this low SR data forcibly ?.", default=0)
                    else:
                        use_forcibly = FLAGS.use_forcibly
                    save_current = bool(use_forcibly)
                save_data.append(save_current)

                with open(total_SR_log, "a") as f:
                    f.write(f"{key}\t{num_of_done_lan_pref[key]}/{total_lan_pref[key]}\tSR={sr:.2f}\tAvgSteps={avg_steps:.1f}\n")

                timestamp = datetime.now().strftime("%d-%H-%M")
                with open(sr_summary_log, "a") as f:
                    f.write(f"{key}\t{num_of_done_lan_pref[key]}/{total_lan_pref[key]}\tSR={sr:.2f}\n")
                    f.write(f"  Avg Steps: {avg_steps:.1f} ({len(steps)} episodes)\n")
                    f.write(f"────────────────────────────────── {timestamp}\n")

            for __cnt, _folder in enumerate(hdf5_folders):
                if save_data[__cnt]:
                    concatenate_hdf5_demos(
                        mission_id=hdf5_save_path[__cnt],
                        hdf5_folder_path=_folder,
                        overwrite_old=FLAGS.overwrite_old,
                        tasksuite=(f"{tasksuite}_naive"
                                   if getattr(FLAGS, "gen_naive_anchor", False) and tasksuite
                                   else tasksuite),
                    )
                else:
                    print(f"Skipping saving data for mission id:{dataset_ids[__cnt]} due to low success rate: {sr_list[__cnt]:.2f}")
    finally:
        env.close()



def gen_data_in_scale(workspaces,bddl_file_names,lan_insts,high_pref_lans,low_pref_lans,mission_mask,dataset_ids,layout_id,flags,subtask_offset_list,mission_dir=None,damping_level=1,skill_debug=False,tasksuite=None,n_save_video=None,save_eval_hdf5=False,save_gif=False):
    """
    Main function to run the entire data generation pipeline.

    If mission_dir is provided, uses the new perm per-mission folder structure.
    """

    global FLAGS
    FLAGS = flags

    # naive-anchor mode must redirect output to <tasksuite>_naive; without a
    # tasksuite the merged demo lands in the in-mission HDF5 folder and can
    # clobber the pref demo. Fail fast instead of silently overwriting.
    if getattr(FLAGS, "gen_naive_anchor", False) and not tasksuite:
        raise ValueError(
            "--gen_naive_anchor requires --tasksuite (else naive demos would "
            "overwrite the in-mission pref HDF5 instead of going to <tasksuite>_naive/)"
        )

    # --pair_init_from <pref hdf5>: replay that file's per-demo init_state so the
    # naive rollouts start from exactly the pref layouts (see motion_gen's
    # INIT_STATE_POOL). Needed when the naive plan differs from task_plan
    # (1-8 --force_naive_plan), which otherwise shifts an object a few cm.
    _pair_src = getattr(FLAGS, "pair_init_from", None)
    if _pair_src:
        import h5py as _h5
        import MotionGen.motion_gen as _mg
        with _h5.File(_pair_src, "r") as _f:
            _keys = sorted(_f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            _mg.INIT_STATE_POOL = [_f[f"data/{k}"]["states"][0].copy() for k in _keys]
        print(f"  [paired-init] {len(_mg.INIT_STATE_POOL)} init states from {_pair_src}")

    layout_cnt=layout_id

    bddl_file_name = bddl_file_names
    lan_inst = lan_insts
    workspace = workspaces
    path_dict=get_path_dict(output_dir=f"./dataset/", layout_cnt=layout_cnt, dataset_id=None, mission_dir=mission_dir)
    _layout, _task_plan=gen_naive_task(bddl_file_name,lan_inst,workspace,path_dict=path_dict,mission_dir=mission_dir)
    _layout.set_layout_id(layout_cnt)

    for cnt, dataset_id in enumerate(dataset_ids):
        if mission_mask[cnt]==0 or (dataset_id!=getattr(FLAGS, 'mis_id', 'Test') and getattr(FLAGS, 'mis_id', 'Test')!="Test"):
            print (f"Skipping mission id:{dataset_id} as per mission mask.")
            continue

        print(f"\n{'─' * WIDTH}")
        print(f"  High pref: {high_pref_lans[cnt]}")
        print(f"  Low pref:  {low_pref_lans[cnt]}")
        print(f"  Layout:    {layout_cnt}")
        logger.info(f"High pref: {high_pref_lans[cnt]} | Low pref: {low_pref_lans[cnt]} | Layout: {layout_cnt}")
        print(f"{'─' * WIDTH}")

        path_dict=get_path_dict(output_dir=f"./dataset/",layout_cnt=layout_cnt,dataset_id=dataset_id,mission_dir=mission_dir)
        # force_naive only for missions whose pref lives in the plan (series 2/3,
        # high_pref_lan set). Series-1 naive anchors keep executing task_plan
        # (== naive_plan modulo grasp-skill variants) with low prefs stripped.
        _layout, _task_plan=gen_high_task(high_pref=high_pref_lans[cnt],layout=_layout, task_plan=_task_plan,dataset_id=dataset_id,path_dict=path_dict,
                                          force_naive=(getattr(FLAGS, "gen_naive_anchor", False)
                                                       and (bool(high_pref_lans[cnt])
                                                            or bool(getattr(FLAGS, "force_naive_plan", False)))))
        _layout, _task_plan=gen_low_task(layout=_layout,task_plan=_task_plan,path_dict=path_dict,dataset_id=dataset_id,low_pref_lan=low_pref_lans[cnt])
    gen_demo(layout=_layout,task_plan=_task_plan,
             path_dict=path_dict,subtask_offset_list=subtask_offset_list,
             dataset_ids=dataset_ids,
             layout_id=layout_id,
             mission_mask=mission_mask,
             damping_level=damping_level,
             skill_debug=skill_debug,
             tasksuite=tasksuite,
             n_save_video=n_save_video,
             save_eval_hdf5=save_eval_hdf5,
             save_gif=save_gif,
             )
    _task_plan.clean_preferences()

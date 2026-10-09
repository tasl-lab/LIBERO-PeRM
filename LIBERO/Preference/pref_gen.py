
import logging
import os
import re
from TaskGen.task import TaskPlan
from TaskGen.layout_gen.layout import Layout
from typing import Dict, List, Any, Optional

logger = logging.getLogger("MotionGen")


"""
Given a task (i.e. List[(skill:str, object:str)] ) sequence, generate high level preferences
"""


"""
Zhixu: I catagorized objects based on their shapes, according to the object_snapshots in the folder "object_snapshots".
Some high-level preference is "pick red bowl" vs "pick green bowl",
so this py file is to define the shape-based object sets to provide alternatives.
"""

"""
Objects that in shape of a box.
"""
object_shape_box = {
    "black_book",
    "butter",
    "chocolate_pudding",
    "cookies",
    "cream_cheese",
    "macaroni_and_cheese",
    "milk",
    "orange_juice",
    "popcorn",
    "yellow_book",
}

"""
Objects that in shape of a cylinder.
"""
object_shape_cylinder = {
    "alphabet_soup",
    "bbq_sauce",
    "tomato_sauce",
}

"""
Objects that in shape of a bowl.
"""
object_shape_bowl = {
    "akita_black_bowl",
    "glazed_rim_porcelain_ramekin",
    "porcelain_mug",
    "red_coffee_mug",
    "white_yellow_mug",
    "red_bowl",
}
"""
Objects that in shape of a plate.
"""
object_shape_plate = {
    "plate",
    "blue_plate",
    "red_plate",

}

"""
Objects that in shape of a bottle.
"""
object_shape_bottle = {
    "ketchup",
    "salad_dressing",
    "wine_bottle",

}

"""
Objects that can serve as containers to put sth. in.
"""
object_shape_container = {
    "basket",

    "bowl_drainer",
    "chefmate_8_frypan",
    "desk_caddy",
    "wooden_tray",

}

"""
Objects that can serve as surfaces to put sth. on.
Usually very big, not that movable.
"""
object_shape_surface = {
    "white_storage_box",
    "wooden_shelf",
    "white_cabinet",
    "wooden_cabinet",
    "short_cabinet",
    "short_fridge",
    "microwave",
    "slide_cabinet",
}

object_shape_unique = {
    "dining_set_group",
    "moka_pot",
    "simple_rack",
    "window",
}

libero_pro_set={
"YellowCabinet",
"YellowStove",
"BrownRack",
}

known_object_bases = (
    object_shape_box
    | object_shape_cylinder
    | object_shape_bowl
    | object_shape_plate
    | object_shape_bottle
    | object_shape_container
    | object_shape_surface
    | object_shape_unique
)
available_obj_set={
    "black_book | (in shape of box)",
    "yellow_book | (in shape of box)",
    "red_yellow_book | (in shape of box)",
    "black_book_side | (in shape of box)",
    "yellow_book_side | (in shape of box)",
    "red_yellow_book_side | (in shape of box)",
    "butter | (in shape of box)",
    "green_butter | (in shape of box)",
    "chocolate_pudding | (in shape of box)",
    "green_chocolate_pudding | (in shape of box)",
    "cookies | (in shape of box)",
    "yellow_cookies | (in shape of box)",
    "cream_cheese | (in shape of box)",
    "bigger_cheese | (in shape of box)",
    "red_cream_cheese | (in shape of box)",
    "macaroni_and_cheese | (in shape of box)",
    "milk | (in shape of box)",
    "bigger_milk | (in shape of box)",
    "yellow_milk | (in shape of box)",
    "orange_juice | (in shape of box)",
    "red_orange_juice | (in shape of box)",
    "popcorn | (in shape of box)",
    "moka_pot | (in shape of box)",
    "yellow_moka_pot | (in shape of box)",
    "alphabet_soup | (in shape of box/can)",
    "bigger_alphabet_soup | (in shape of box/can)",
    "red_alphabet_soup | (in shape of box/can)",
    "tomato_sauce | (in shape of box/can)",
    "bigger_tomato_sauce | (in shape of box/can)",
    "yellow_tomato_sauce | (in shape of box/can)",
    "glazed_rim_porcelain_ramekin | (in shape of box)",
    "red_ramekin | (in shape of box)",
    "ketchup | (in shape of bottle)",
    "blue_ketchup | (in shape of bottle)",
    "green_ketchup | (in shape of bottle)",
    "salad_dressing | (in shape of bottle)",
    "red_salad_dressing | (in shape of bottle)",
    "wine_bottle | (in shape of bottle)",
    "bbq_sauce | (in shape of bottle)",
    "green_bbq_sauce | (in shape of bottle)",
    "plate | (in shape of bowl)",
    "blue_plate | (in shape of bowl)",
    "red_plate | (in shape of bowl)",
    "bigger_plate | (in shape of bowl)",
    "yellow_plate | (in shape of bowl)",
    "akita_black_bowl | (in shape of bowl)",
    "bigger_akita_black_bowl | (in shape of bowl)",
    "red_akita_black_bowl | (in shape of bowl)",
    "red_bowl | (in shape of bowl)",
    "black_bowl | (in shape of bowl)",
    "yellow_bowl | (in shape of bowl)",
    "porcelain_mug | (in shape of bowl)",
    "white_porcelain_mug | (in shape of bowl)",
    "red_coffee_mug | (in shape of bowl)",
    "white_yellow_mug | (in shape of bowl)",
    "basket | (in shape of container)",
    "white_storage_box | (in shape of container)",
    "red_basket | (in shape of container)",
    "desk_caddy | (in shape of container)",
    "yellow_desk_caddy | (in shape of container)",
    "wooden_cabinet_1_cabinet_top | top drawer of wooden cabinet",
    "wooden_cabinet_1_cabinet_middle | middle drawer of wooden cabinet",
    "wooden_cabinet_1_cabinet_bottom | bottom drawer of wooden cabinet",
    "white_cabinet_1_cabinet_top | top drawer of white cabinet",
    "white_cabinet_1_cabinet_middle | middle drawer of white cabinet",
    "white_cabinet_1_cabinet_bottom | bottom drawer of white cabinet",
    "wooden_two_layer_shelf_1",
    "ground_1"

}



from Preference.high_level.task_preference import TaskPreference
from Preference.low_level.motion_preference import BaseMotionPreference
# Import subclasses so __init_subclass__ registers them into BaseMotionPreference._registry.
from Preference.low_level.distance import AvoidToObjectPreference, ApproachToObjectPreference  # noqa: F401
from Preference.low_level.speed import SpeedRangePreference, AccRangePreference  # noqa: F401
from Preference.low_level.trajectory import StraightPathPreference, HeightOffsetPreference, RotationPreference  # noqa: F401
# NOTE: relationship preferences (OnTop / In / InContact / Opened / Closed) are
# binary task-state predicates rather than motion preferences and are no longer
# auto-registered into BaseMotionPreference._registry. Their class definitions
# remain available for `motion_gen` (used in skill execution via isinstance
# checks) — import them on demand from Preference.low_level.extra.relationship.
from Preference.low_level.extra.release_height import ReleaseHeightPreference  # noqa: F401
# Skill-tier (series 2): the pref lives in the PLAN (pick_left_edge vs pick), so
# task-plan compliance scores both sides the same — these read the grasp/release
# POSE instead. See Preference/low_level/skill_pose.py.
from Preference.low_level.skill_pose import (  # noqa: F401
    GraspOffsetPreference,
    GraspTiltPreference,
    PlaceRotationPreference,
    ObjectTiltPreference,
    ReleaseDropPreference,
    ObjectLiftPreference,
)
from Preference.high_level.plan_choice import PlanChoicePreference  # noqa: F401
from Preference.high_level.task_plan_progress import (  # noqa: F401
    TaskPlanProgressPreference,
    load_task_plan_progress,
)


# =====================================================
# Skill-to-put mapping: special pick -> corresponding put
# =====================================================
# NOTE:
#   pick_left_edge is the renamed pick_bowl (same implementation)
#   pick_horizontal is the renamed pick_with_load (same implementation)
#   put_left_edge is the renamed put_bowl (same implementation)
#   put_horizontal is the renamed put_plate_with_load (same implementation)
SKILL_PICK_TO_PUT = {
    "pick":             "put",
    "pick_left_edge":   "put_left_edge",
    "pick_back_edge":   "put_back_edge",
    # "pick_back_edge_v0": "put_back_edge_v0",  # deprecated
    "pick_scoop":       "put_scoop",
    "pick_bottle":      "put",
    "pick_from_side":   "put",
    "pick_from_front":  "put",
    "pick_horizontal":  "put_horizontal",
    "pick_handle":      "put_handle",
}




# Backward-compat aliases for legacy yaml keys → canonical registry keys.
_TYPE_ALIASES = {
    "avoid_object": "avoid_to_object",
    "approach_object": "approach_to_object",
    "max_acc": "acc_range",
}


def _get_pref_class(preference_type: str):
    canonical = _TYPE_ALIASES.get(preference_type, preference_type)
    return BaseMotionPreference._registry.get(canonical)


def create_motion_preference(
    preference_type: str,
    preference_parameters: Dict[str, Any],
    pref_description: str,
    pos_description: Optional[str],
):
    PrefClass = _get_pref_class(preference_type)
    if PrefClass is None:
        raise ValueError(f"Unknown preference type: {preference_type}")

    normalized_parameters = dict(preference_parameters)

    # Backward compatibility for older dataset/preference schemas.
    if preference_type in {"avoid_object", "approach_object", "avoid_to_object", "approach_to_object"}:
        if "target_obj" not in normalized_parameters and "obj" in normalized_parameters:
            normalized_parameters["target_obj"] = normalized_parameters.pop("obj")
        if "threshold" not in normalized_parameters and "distance" in normalized_parameters:
            normalized_parameters["threshold"] = normalized_parameters.pop("distance")
    if preference_type == "height_offset":
        if "height_offset" not in normalized_parameters and "offset" in normalized_parameters:
            normalized_parameters["height_offset"] = normalized_parameters.pop("offset")
    # Relationship: opened/closed use `object_name`; legacy yaml may use `object`.
    if preference_type in {"opened", "closed"}:
        if "object_name" not in normalized_parameters and "object" in normalized_parameters:
            normalized_parameters["object_name"] = normalized_parameters.pop("object")

    try:
        return PrefClass(
            **normalized_parameters,
            pref_description=pref_description,
            pos_description=pos_description,
        )
    except (TypeError, ValueError) as e:
        print(
            f"Error instantiating {PrefClass.__name__} with params {normalized_parameters}: {e}"
        )
        return None
    except NotImplementedError:
        print(f"Preference type '{preference_type}' is not yet implemented.")
        return None


def gen_and_apply_high_pref (task_plan: TaskPlan, layout: Layout, high_level_pref:str,dataset_id,path_dict,force_naive=False):
    """
    Load and apply high level preferences from mission_info.yaml.

    force_naive: naive-anchor data gen for missions whose pref lives in the
    plan itself (series 2 skill / series 3 object-choice). Executes the
    mission's naive_plan instead of task_plan. For missions with a
    no_pref_layout/ bddl (series 3) the executed plan keeps the task_plan's
    SKILLS but the naive_plan's OBJECTS — object choice is the pref dimension
    there, and plain `pick` physically fails on objects the pref plan grasps
    via edge skills (e.g. akita bowl). Without force_naive, an empty
    high_level_pref still falls back to the PREF task_plan from mission_info
    (and trips the family validation on series-3 object swaps).
    """

    yaml_data = task_plan.load_task_plan_yaml(filepath=path_dict['high_level_plan'])

    def _skill_family(skill_name: str) -> str:
        if skill_name.startswith("pick"):
            return "pick"
        if skill_name.startswith("put") or skill_name == "lay_down":
            return "put"
        if skill_name in {"throw"}:
            return "put"
        if skill_name.startswith("move_to"):
            return "move_to"
        return skill_name

    def _object_family(obj_name: str) -> str:
        cabinet_suffixes = (
            "_right_contain_region",
            "_left_contain_region",
            "_front_contain_region",
            "_back_contain_region",
            "_cabinet_top",
            "_cabinet_middle",
            "_cabinet_bottom",
            "_top_region",
            "_middle_region",
            "_bottom_region",
            "_top_side",
            "_contain_region",
            "_button",
            "_cook_region",
            "_main",
        )
        for suffix in cabinet_suffixes:
            if obj_name.endswith(suffix):
                obj_name = obj_name[: -len(suffix)]
                break

        match = re.match(r"^(.*?)(?:_(\d+))?$", obj_name)
        stem = match.group(1) if match else obj_name
        instance_id = match.group(2) if match else None

        canonical_base = stem
        for base in sorted(known_object_bases, key=len, reverse=True):
            if stem == base or stem.endswith(f"_{base}"):
                canonical_base = base
                break

        if instance_id is not None:
            return f"{canonical_base}_{instance_id}"
        return canonical_base

    def _resolve_task_plan_aliases(base_plan, evolved_plan):
        relation_suffixes = ("_right", "_left", "_front", "_back")
        resolved_plan = []

        for step_idx, evolved_step in enumerate(evolved_plan):
            if len(evolved_step) < 2:
                resolved_plan.append(evolved_step)
                continue

            evolved_skill, evolved_obj = evolved_step[0], evolved_step[1]

            resolved_obj = evolved_obj
            for suffix in relation_suffixes:
                if evolved_obj.endswith(suffix):
                    resolved_obj = f"{evolved_obj}_contain_region"
                    break

            if (
                resolved_obj in {"right", "left", "front", "back"}
                and step_idx < len(base_plan)
                and len(base_plan[step_idx]) >= 2
            ):
                base_obj = base_plan[step_idx][1]
                resolved_obj = f"{base_obj}_{resolved_obj}_contain_region"

            resolved_plan.append((evolved_skill, resolved_obj, *evolved_step[2:]))

        return resolved_plan

    def _validate_task_plan_family(base_plan, evolved_plan):
        if len(base_plan) != len(evolved_plan):
            raise ValueError(
                f"task_plan length {len(evolved_plan)} does not match naive_plan length {len(base_plan)}"
            )

        for step_idx, (base_step, evolved_step) in enumerate(zip(base_plan, evolved_plan)):
            if len(base_step) < 2 or len(evolved_step) < 2:
                raise ValueError(f"Invalid plan step at index {step_idx}: {base_step} vs {evolved_step}")

            base_skill, base_obj = base_step[0], base_step[1]
            evolved_skill, evolved_obj = evolved_step[0], evolved_step[1]

            if _object_family(base_obj) != _object_family(evolved_obj):
                raise ValueError(
                    f"task_plan object mismatch at step {step_idx}: {base_obj} vs {evolved_obj}"
                )
            if _skill_family(base_skill) != _skill_family(evolved_skill):
                raise ValueError(
                    f"task_plan skill family mismatch at step {step_idx}: {base_skill} vs {evolved_skill}"
                )

    if yaml_data is None:
        raise FileNotFoundError(
            f"No mission_info.yaml found at {path_dict['high_level_plan']}"
        )

    naive_plan_tuples = [tuple(p) for p in yaml_data.get("naive_plan", [])]
    task_plan_tuples = [tuple(p) for p in yaml_data.get("task_plan", [])]
    if task_plan_tuples:
        task_plan_tuples = _resolve_task_plan_aliases(naive_plan_tuples or task_plan_tuples, task_plan_tuples)

    if force_naive:
        if not naive_plan_tuples:
            print("  [PrefGen] force_naive: no naive_plan in mission_info, "
                  "falling back to auto-generated naive plan.")
            task_plan.add_high_pref(None)
            return
        # Skill-borrowing applies to series-3 only: there the pref dimension is
        # WHICH object/target, so task_plan grasp variants are incidental
        # (feasibility) and safe to reuse. In series-1/2 a differing skill IS
        # the pref (pick_with_rotation, pick_handle, ...) — the naive demo must
        # not inherit it. (Keying on no_pref_layout existence was wrong: some
        # series-1 missions have that dir too.)
        is_series3 = str(dataset_id).startswith("3")
        exec_plan = naive_plan_tuples
        if (is_series3 and task_plan_tuples
                and len(task_plan_tuples) == len(naive_plan_tuples)
                and all(_skill_family(t[0]) == _skill_family(n[0])
                        for t, n in zip(task_plan_tuples, naive_plan_tuples))):
            # pick steps borrow the pref plan's grasp skill (plain `pick`
            # physically fails on e.g. the akita bowl); every other step keeps
            # the naive skill — edge-put variants deliberately place off-center
            # and can drop the object outside the goal region.
            # Only UPGRADE a plain naive `pick` to the pref's grasp variant; a
            # naive step that already names a variant (pick_left_edge on the
            # akita bowl while the pref plain-picks a red bowl, 3-15_106_1) is
            # deliberate and must not be downgraded to plain `pick`.
            exec_plan = [
                (t[0], n[1], *t[2:])
                if (_skill_family(t[0]) == "pick" and n[0] == "pick" and t[0] != "pick")
                else (n[0], n[1], *n[2:])
                for t, n in zip(task_plan_tuples, naive_plan_tuples)
            ]
        print(f"  [PrefGen] force_naive: executing naive plan {exec_plan}")
        logger.info(f"[PrefGen] force_naive: executing naive plan {exec_plan}")
        task_plan.add_high_pref(TaskPreference(
            pref_description="",
            single_pref_plan=exec_plan,
            action_parameters={},
            layout_id=layout.get_layout_id(),
            lan_pref_id=dataset_id,
            task_id=0,
        ))
        return

    if task_plan_tuples and high_level_pref == "":
        # 仅对无 high-level preference 的任务做严格校验
        # 3-x profile preference 允许不同物体和不同步数
        _validate_task_plan_family(naive_plan_tuples or task_plan_tuples, task_plan_tuples)


    if high_level_pref == "":
        if task_plan_tuples:
            default_task_plan = TaskPreference(
                pref_description="",
                single_pref_plan=task_plan_tuples,
                action_parameters={},
                layout_id=layout.get_layout_id(),
                lan_pref_id=dataset_id,
                task_id=0,
            )
            print("  [PrefGen] No high-level language preference provided, using task_plan from mission_info.yaml.")
            logger.info("[PrefGen] No high-level language preference provided, using task_plan from mission_info.yaml.")
            task_plan.add_high_pref(default_task_plan)
        else:
            print(f"  [PrefGen] No high-level preference provided, using naive_plan only.")
            logger.info("[PrefGen] No high-level preference provided, using naive_plan only.")
            task_plan.add_high_pref (None)
        return

    if task_plan_tuples:
        rst = []
        for hp in yaml_data.get("high_prefs", [{}]):
            rst.append({
                "task_plan": task_plan_tuples,
                # series-3 yamls keep high_prefs entries without description;
                # the mission-level high_pref_lan is the pref language then
                "description": hp.get("description", "") or high_level_pref,
                "action_parameters": hp.get("action_parameters", {}),
            })
        if not rst:
            # series-3 profile missions: high_prefs is an empty list, the pref
            # language only lives in high_pref_lan (== high_level_pref here)
            rst = [{"task_plan": task_plan_tuples,
                    "description": high_level_pref, "action_parameters": {}}]
    else:
        raise FileNotFoundError(
            f"No task_plan/high_prefs found in mission_info.yaml at {path_dict['high_level_plan']}"
        )

    for id, plan_w_pref in enumerate (rst):
        plan = plan_w_pref ["task_plan"]
        high_pref = TaskPreference (pref_description=plan_w_pref ["description"],
                                    single_pref_plan=plan,
                                    action_parameters=plan_w_pref["action_parameters"],
                                    layout_id=layout.get_layout_id(),
                                    lan_pref_id=dataset_id,
                                    task_id=id,
                                    )
        print(f"  🌙 [PrefGen] Loaded high-level preference: {high_pref}")
        logger.info(f"🌙 [PrefGen] Loaded high-level preference: {high_pref}")
        task_plan.add_high_pref (high_pref)

    print(f"  👌 [PrefGen] Loaded {len(rst)} high-level preferences.")
    logger.info(f"👌 [PrefGen] Loaded {len(rst)} high-level preferences.")



def gen_low_pref (task_plan: TaskPlan, layout: Layout, path_dict,low_pref_lan,dataset_id):
    """
    Load low level preferences from mission_info.yaml.
    """
    if low_pref_lan=="":
        print(f"  [PrefGen] No low-level preference provided, using empty.")
        logger.info("[PrefGen] No low-level preference provided, using empty.")
        task_plan.add_low_level_pref([])
        return

    motion_pref_list=[]

    # Load from mission_info.yaml
    yaml_data = task_plan.load_task_plan_yaml(filepath=path_dict['low_pref'])
    if yaml_data is not None and "low_prefs" in yaml_data:
        rst = [{"preference": {"type": item["type"], "params": item["params"]},
                "description": item.get("description", ""),
                "pos_description": item.get("pos_description")}
               for item in yaml_data["low_prefs"]]
    else:
        raise FileNotFoundError(
            f"No low_prefs found in mission_info.yaml at {path_dict['low_pref']}"
        )

    for pref_dict in rst:
        motion_pref = create_motion_preference(
            preference_type=pref_dict["preference"]["type"],
            preference_parameters=pref_dict["preference"]["params"],
            pref_description=pref_dict["description"],
            pos_description=pref_dict["pos_description"],
        )

        if motion_pref:
            motion_pref_list.append(motion_pref)
            print(f"  ⭐ [PrefGen] Generated low-level preference: {motion_pref}")
            logger.info(f"⭐ [PrefGen] Generated low-level preference: {motion_pref}")
        else:
            print (f"⚠️ Generated low-level preference is not doable, please check the parameters.")

    task_plan.add_low_level_pref (motion_pref_list)
    print(f"  👌 [PrefGen] Generated {len(motion_pref_list)} low-level preferences.")
    logger.info(f"👌 [PrefGen] Generated {len(motion_pref_list)} low-level preferences.")

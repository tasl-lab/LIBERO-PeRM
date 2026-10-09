"""
Given a layout, generate a naive task for it.
Then we will apply pref_gen module to steer the naive task to one with preferences.
"""

from TaskGen.task import TaskPlan
from TaskGen.layout_gen.layout import Layout
from typing import List


def gen_naive_task_plan(layout: Layout, skill_names: list, path_dict) -> TaskPlan:
    """
    Load naive task plan from mission_info.yaml.
    """
    lan_inst = layout.lan_inst
    task_plan = TaskPlan(lan_inst)
    img_path = "Naive_layout.png"
    layout.get_obs_image(img_path)
    import logging
    logging.getLogger("MotionGen").debug(f"Object set: {layout.object_set}")

    # Load from mission_info.yaml
    yaml_data = task_plan.load_task_plan_yaml(filepath=path_dict['naive_plan'])
    if yaml_data is not None and "naive_plan" in yaml_data:
        skill_obj_list = [tuple(pair) for pair in yaml_data["naive_plan"]]
    else:
        raise FileNotFoundError(
            f"No naive_plan found in mission_info.yaml at {path_dict['naive_plan']}"
        )

    for skill, obj in skill_obj_list:
        if skill in skill_names:
            task_plan.add_plan(skill, obj)
        else:
            print(f"Warning: skill {skill} not in the given skill list.")
            print("skill_obj_list:", skill_obj_list)
    task_plan.naive_plans = task_plan.get_plans()

    return task_plan
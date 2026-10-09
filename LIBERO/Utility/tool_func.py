def get_target_xpos(env, name: str):
    """Get target position by body name, falling back to site if not found."""
    try:
        pos = env.sim.data.get_body_xpos(name).copy()
    except ValueError:
        sid = env.sim.model.site_name2id(name)
        pos = env.sim.data.site_xpos[sid].copy()
    if "white_cabinet_1_cabinet_bottom" in name:
        pos[1] -= 0.01
    if "desk_caddy" in name and "left_contain_region" in name:
        pos[0] += 0.02
    return pos

def to_sim_env_name(name: str) -> str:
    """
    Convert environment name to simulation environment name.
    """
    cabinet_passthrough_suffixes = (
        "_cabinet_top",
        "_cabinet_middle",
        "_cabinet_bottom",
        "_top_region",
        "_middle_region",
        "_bottom_region",
        "_top_side",
        "_contain_region",
        "_right_contain_region",
        "_left_contain_region",
        "_front_contain_region",
        "_back_contain_region",
        "_button",
        "_cook_region",
    )
    if name == "ground_1" or any(name.endswith(suffix) for suffix in cabinet_passthrough_suffixes):
        return name
    else:
        if not name.endswith("_main"):
            sim_name = name + "_main"
        else:
            sim_name = name
    return sim_name

def env_name2obj_name(name: str) -> str:
    """
    Convert environment name to simulation environment name.
    """
    if name is None:
        return name
    cabinet_passthrough_suffixes = (
        "_cabinet_top",
        "_cabinet_middle",
        "_cabinet_bottom",
        "_top_region",
        "_middle_region",
        "_bottom_region",
        "_top_side",
        "_contain_region",
        "_right_contain_region",
        "_left_contain_region",
        "_front_contain_region",
        "_back_contain_region",
        "_button",
        "_cook_region",
    )
    if any(name.endswith(suffix) for suffix in cabinet_passthrough_suffixes):
        return name
    else:
        if name.endswith("_main"):
            obj_name = name[:-5]  # 去掉 "_main"
        else:
            obj_name = name
    return obj_name

def ask_choice(prompt: str, default: int = 0, no_default=False) -> bool:
    """
    prompt: 提示语
    default: 默认值 (0 or 1)
    return: True if user chooses 1, else False
    """
    if not no_default:
        msg = f"{prompt} (1 = Yes, 0 = No; default = {default}): "
    else:
        msg = f"{prompt} "
    choice = input(msg).strip()

    if choice == "":
        choice = str(default)

    return choice == "1"

import  numpy as np
def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def get_mug_direction(obj:str)->str:
    """
    Convert direction to yaw handle description.
    """
    if obj.startswith("white_porcelain_mug") or obj.startswith("porcelain_mug"):
        direction="left"
    elif obj.startswith("red_coffee_mug") or obj.startswith("white_yellow_mug"):
        direction="right"
    else:
        raise NotImplementedError
    return direction

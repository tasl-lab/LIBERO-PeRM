from bddl.parsing import *

import itertools
import os
import numpy as np

pi = np.pi

# ── init-pose randomization knobs (2026-09-01) ─────────────────────────────
# The bddl `:ranges` boxes are tiny (typically 5x5 cm, some 2x2 cm) and
# `:yaw_rotation` is usually pinned to (0, 0), so every rollout starts from
# nearly the same layout. Rather than editing hundreds of bddl files, scale
# the parsed ranges about their centre and widen the yaw band here — one knob
# for the whole suite, set per run via env vars:
#
#   PERM_INIT_RANGE_SCALE   float, default 1.0  (2.0 = double each side)
#   PERM_INIT_YAW_JITTER    float rad, default 0.0 (adds +-jitter to yaw band)
#
# The samplers run with ensure_valid_placement=True, so a widened box that
# lands on a collision is re-drawn; only genuinely impossible layouts fail.
def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


INIT_RANGE_SCALE = _env_float("PERM_INIT_RANGE_SCALE", 1.0)
INIT_YAW_JITTER = _env_float("PERM_INIT_YAW_JITTER", 0.0)


def _scale_range(rect, scale):
    """Grow a (x1, y1, x2, y2) box about its centre by `scale`."""
    if scale == 1.0:
        return rect
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hx, hy = abs(x2 - x1) / 2.0 * scale, abs(y2 - y1) / 2.0 * scale
    return [cx - hx, cy - hy, cx + hx, cy + hy]


def get_regions(t, regions, group):
    group.pop(0)
    while group:
        region = group.pop(0)
        region_name = region[0]
        target_name = None
        region_dict = {
            "target": None,
            "ranges": [],
            "extra": [],
            "yaw_rotation": [0, 0],
            "rgba": [0, 0, 1, 0],
        }
        for attribute in region[1:]:
            if attribute[0] == ":target":
                assert len(attribute) == 2
                region_dict["target"] = attribute[1]
                target_name = attribute[1]
            elif attribute[0] == ":ranges":
                for rect_range in attribute[1]:
                    assert (
                        len(rect_range) == 4
                    ), f"Dimension of rectangular range mismatched!!, supposed to be 4, only found {len(rect_range)}"
                    region_dict["ranges"].append(
                        _scale_range([float(x) for x in rect_range], INIT_RANGE_SCALE)
                    )
            elif attribute[0] == ":yaw_rotation":
                # print(attribute[1])
                for value in attribute[1]:
                    yaw = [eval(x) for x in value]
                    if INIT_YAW_JITTER:
                        yaw = [yaw[0] - INIT_YAW_JITTER, yaw[1] + INIT_YAW_JITTER]
                    region_dict["yaw_rotation"] = yaw
            elif attribute[0] == ":rgba":
                assert (
                    len(attribute[1]) == 4
                ), f"Missing specification for rgba color, supposed to be 4 dimension, but only got  {attribute[1]}"
                region_dict["rgba"] = [float(x) for x in attribute[1]]
            else:
                raise NotImplementedError
        regions[target_name + "_" + region_name] = region_dict


def get_scenes(t, scene_properties, group):
    group.pop(0)
    while group:
        scene_property = group.pop(0)
        scene_properties_dict = {}
        for attribute in region[1:]:
            if attribute[0] == ":floor":
                assert len(attribute) == 2
                scene_properties_dict["floor_style"] = attribute[1]
            elif attribute[0] == ":wall":
                assert len(attribute) == 2
                scene_properties_dict["wall_style"] = attribute[1]
            else:
                raise NotImplementedError


def get_problem_info(problem_filename):
    domain_name = "unknown"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                domain_name = "robosuite"
            elif t == ":language":
                group.pop(0)
                language_instruction = group
    return {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "language_instruction": " ".join(language_instruction),
    }


def robosuite_parse_problem(problem_filename):
    domain_name = "robosuite"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        fixtures = {}
        regions = {}
        scene_properties = {}
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group

            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            else:
                print("%s is not recognized in problem" % t)
        return {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
        }
    else:
        raise Exception(
            f"Problem {behavior_activity} {activity_definition} does not match problem pattern"
        )

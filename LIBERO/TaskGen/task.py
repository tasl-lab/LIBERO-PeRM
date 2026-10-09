
from typing import Callable, Any, Dict, Tuple, List

import ast
import logging
import yaml

logger = logging.getLogger("MotionGen")
from Preference.low_level.motion_preference import BaseMotionPreference
from Preference.high_level.task_preference import TaskPreference
import os
import numpy as np
class TaskPlan:
    """"
    After having the BDDL file, we can have a task plan, which is a sequence of (skill, object) pairs. (BTW, the motion plan is trajectory.)
    The init parameter is just the very abstract language instruction.
    A task plan has:
    - a detailed description that explain preference (str)
    - a language description (str)
    - high-level preferences (dict or custom object)
    - low-level preferences (dict or custom object)
    - plans: List[(skill:str, object:str)]
    FYI: for the whole project, when we say task, we mean the instance of a TaskPlan.
    """
    def __init__(self,language_inst: str):
        self.description = None
        self.language_instruction =language_inst
        self.low_level_prefs_list = []  # list of MotionPreference objects
        self.high_level_prefs: List[TaskPreference]=[] # dictionary of high-level preferences
        self.plans = []  # list of (skill, object) pairs
        self.naive_plans=[] # list of (skill, object) pairs before applying high-level preferences
        self.skill_params={}
    def add_skill_param(self, skill: str,obj:str , params: Any):
        """
        Add a skill parameter to the task plan.
        Each skill can have multiple associated objects.
        """
        if skill not in self.skill_params:
            self.skill_params[skill] = {}
        if obj not in self.skill_params[skill]:
            self.skill_params[skill][obj] = params #Zhixu, skill-obj will only have one param set
        print(f"Added parameters for skill '{skill}' and object '{obj}'")


    def get_skill_params(self,skill,obj:str,key:str )-> Any:
        """
        Get a dictionary of skill parameters from the task plan.
        Each skill maps to a list of objects it is associated with in the plan.
        """
        if skill in self.skill_params and obj in self.skill_params[skill]:
            return self.skill_params[skill][obj].get (key,None)
        else:
            raise ValueError(f"No parameters found for skill '{skill}' and object '{obj}'.")

    def save_skill_params (self, filepath: str):
        os.makedirs (filepath, exist_ok=True)
        full_path = os.path.join (filepath, "skill_params.txt")

        with open (full_path, "w", encoding="utf-8") as f:
            for skill, obj_dict in self.skill_params.items ():
                for obj, params in obj_dict.items ():
                    clean_params = {}
                    for k, v in params.items ():
                        if isinstance (v, np.ndarray):
                            clean_params [k] = v.tolist ()
                        else:
                            clean_params [k] = v

                    f.write (f"Skill: {skill}, Object: {obj}, Params: {clean_params}\n")

    def load_skill_params (self, filepath: str):
        full_path = os.path.join (filepath, "skill_params.txt")
        if not os.path.exists (full_path):
            raise FileNotFoundError (f"No such file: {full_path}")

        with open (full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip ()

                skill = line.split ("Skill: ") [1].split (",") [0]
                obj = line.split ("Object: ") [1].split (",") [0]

                params_prefix = "Params: "
                params_str = line [line.index (params_prefix) + len (params_prefix):]

                params = ast.literal_eval (params_str)
                self.add_skill_param (skill, obj, params)

    def add_plan(self, skill: str, obj: str):
        """Add a (skill, object) pair to the task plan."""
        self.plans.append((skill, obj))

    def get_plans(self):
        """Return the list of all (skill, object) pairs."""
        return self.plans

    def get_real_plans (self,naive_plan:List[Tuple[str, str]] =None):
        """
        Get the real plans after applying high-level preferences.
        Automatically insert ('move_to', object) before each non-move_to skill.
        """

        if naive_plan is None:
            naive_plan = self.plans

        if not hasattr (self, "plans"):
            raise AttributeError ("TaskPreference object must have 'plans' before generating real plans.")

        real_plans = []
        for skill, obj in naive_plan:
            # 忽略本身就是 move_to 的
            if skill != "move_to":
                # 在每个有效 skill 前插入 move_to
                real_plans.append (("move_to", obj))
                # 然后再加入当前 skill
                real_plans.append ((skill, obj))

        return real_plans

    def check_success_wo_pref(self, current_state) -> bool:
        """
        Check if the current state satisfies success condition without considering preferences.
        (这里可以接BDDL的goal_state判定逻辑，你先留个接口)
        """
        # placeholder
        return False

    def check_success_w_pref(self, current_state) -> bool:
        """
        Check if the current state satisfies success condition considering preferences.
        """
        # placeholder
        return False

    def add_high_pref(self, high_pref):
        """
        Add or update a high-level preference.
        """
        self.high_level_prefs.append(high_pref)



    def save(self, key, value, filepath: str="./dataset/GPT_calls"):
        """
        Save the task plan to a file.
        """
        # 1️⃣ 确保目录存在
        os.makedirs(filepath, exist_ok=True)

        # 2️⃣ 拼接完整路径
        full_path = os.path.join(filepath, f"{key}.txt")

        # 3️⃣ 写入文件
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(f"{value}\n")
        print(f">>>>>>>>>>>Save to {full_path}")
        return full_path

    def load(self, key, filepath:str):
        """
        Load the task plan from a file.
        """
        full_path = os.path.join(filepath, f"{key}.txt")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No such file: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            value = f.read().strip()
        print(f"<<<<<<<<<<<<<Load from {full_path}")
        return value

    #Zhixu: Here we apply the preference in task class to avoid circular import
    def apply_high_pref(self, high_pref: TaskPreference=None):
        """
        Apply the preference to the task, modify the task plan accordingly.
        """
        pass


    def add_low_level_pref(self, low_pref_list):
        """
        Add a low-level preference with a name and a measure function.
        The measure function should take relevant parameters and return a float score.
        """
        if not low_pref_list:
            self.low_level_prefs_list.append([])
            return
        self.low_level_prefs_list.append(low_pref_list)
    def add_high_level_pref(self, high_pref:TaskPreference):
        """
        Add or update a high-level preference.
        """
        self.high_level_prefs.append(high_pref)
    def save_task_plan_yaml(self, filepath, data):
        """Save plan data to mission_info.yaml. Merges with existing file if present."""
        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, "mission_info.yaml")
        # Merge with existing data if file exists
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
            existing.update(data)
            data = existing
        with open(full_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=None, allow_unicode=True)
        print(f">>>>>>>>>>>Save YAML to {full_path}")
        return full_path

    def load_task_plan_yaml(self, filepath):
        """Load plan data from mission_info.yaml."""
        full_path = os.path.join(filepath, "mission_info.yaml")
        if not os.path.exists(full_path):
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Load YAML from {full_path}")
        return data

    def clean_preferences(self):
        """
        Clean all preferences.
        """
        self.low_level_prefs_list = []
        self.high_level_prefs = []

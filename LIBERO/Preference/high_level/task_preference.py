"""
Definition of a high-level preference for a task.
It can

"""

from typing import Set, List, Tuple



class TaskPreference:
    """Task level preference class
    A task preference is a list of high level preferences
    Zhixu: Here we define high-level task as a sequence of (skill, object) pairs, and do mutations on such pairs without disobeying the language instruction,
    so we may need ChatGPT API to verify the mutated task plan still obeys the instructio or directly use it to generate the mutations.
    """

    def __init__(
        self,
        pref_description: str,
        single_pref_plan: List[Tuple[str, str]],
        task_id: int ,
        layout_id=None,
        lan_pref_id=None,
        action_parameters=None,
        # Legacy params (ignored, kept for backward compat)
        required=None,
        prohibited=None,
        order=None,
    ):
        self.task_id = task_id
        self.pref_description = pref_description
        self.single_pref_plan = single_pref_plan
        self.action_parameters=action_parameters
        self.lan_pref_id = lan_pref_id
        self.layout_id = layout_id

    def __str__(self):
        str2print=f"🌹🌹🌹Task Preference in language : {self.pref_description}\n Single pref plan: {self.single_pref_plan}\n"
        if self.action_parameters is not None:
            str2print+=f"Action params: {self.action_parameters}\n"

        return str2print

    def obj_of_interest (self) -> Set [str]:
        """
        Collect all objects mentioned in single_pref_plan.
        """
        object_names: Set [str] = set ()
        for _, obj in self.single_pref_plan:
            if obj and obj.strip():
                object_names.add(obj.strip())
        return object_names



"""
Given a env, return the relationship of object at that time step
"""

from typing import Callable, Dict, Any, Set
from ..motion_preference import BaseMotionPreference


def check_on_relationship(env, subject_name: str, target_name: str) -> bool:
	"""
	Return True if `subject_name` is on top of `target_name` according to env predicates.

	This reuses the env's object state interfaces and the built-in "on" predicate.

	Args:
		env: A LIBERO env that exposes `object_states_dict`.
		subject_name: Name of the object that is supposed to be on top.
		target_name: Name of the support object / site to check against.

	Returns:
		bool indicating whether the relationship holds at the current timestep.
	"""
	# Fetch object state handles from the environment
	subject_state = env.object_states_dict[subject_name]
	target_state = env.object_states_dict[target_name]

	# Directly use the same semantics as the built-in "on" predicate:
	# On(arg1, arg2) -> arg2.check_ontop(arg1)
	return bool(target_state.check_ontop(subject_state))


def on_top() -> Callable[[Any, Dict[str, Any]], float]:
	"""
	Factory that returns a callable: reward(env, params) -> float.

	Required params:
	- subject: name of the object expected to be on top
	- target: name of the support object or site

	Reward:
	- 1.0 if subject is on top of target, else 0.0
	"""

	def reward(env, params):
		subject_name = params.get("subject")
		target_name = params.get("target")
		if not subject_name or not target_name:
			raise ValueError("params must include 'subject' and 'target' names")
		return 1.0 if check_on_relationship(env, subject_name, target_name) else 0.0

	return reward


def in_relation() -> Callable[[Any, Dict[str, Any]], float]:
	"""
	Factory: reward(env, params) -> float

	Required params:
	- subject: name of the object expected to be inside
	- target: name of the container object or site

	Semantics match the built-in In predicate: target.check_contact(subject) and target.check_contain(subject)
	"""

	def reward(env, params):
		subject_name = params.get("subject")
		target_name = params.get("target")
		if not subject_name or not target_name:
			raise ValueError("params must include 'subject' and 'target' names")
		subject_state = env.object_states_dict[subject_name]
		target_state = env.object_states_dict[target_name]
		ok = bool(target_state.check_contact(subject_state) and target_state.check_contain(subject_state))
		return 1.0 if ok else 0.0

	return reward


def in_contact() -> Callable[[Any, Dict[str, Any]], float]:
	"""
	Factory: reward(env, params) -> float

	Required params:
	- subject: name of the first object
	- target: name of the second object or site

	Reward:
	- 1.0 if subject and target are in contact (per env state), else 0.0
	"""

	def reward(env, params):
		subject_name = params.get("subject")
		target_name = params.get("target")
		if not subject_name or not target_name:
			raise ValueError("params must include 'subject' and 'target' names")
		subject_state = env.object_states_dict[subject_name]
		target_state = env.object_states_dict[target_name]
		ok = bool(subject_state.check_contact(target_state))
		return 1.0 if ok else 0.0

	return reward


def opened() -> Callable[[Any, Dict[str, Any]], float]:
	"""
	Factory: reward(env, params) -> float

	Required params:
	- object: name of the object with openable joint(s)

	Reward: 1.0 if object is open, else 0.0
	"""

	def reward(env, params):
		name = params.get("object")
		if not name:
			raise ValueError("params must include 'object' name")
		obj_state = env.object_states_dict[name]
		ok = bool(getattr(obj_state, "is_open")() )
		return 1.0 if ok else 0.0

	return reward


def closed() -> Callable[[Any, Dict[str, Any]], float]:
	"""
	Factory: reward(env, params) -> float

	Required params:
	- object: name of the object with openable joint(s)

	Reward: 1.0 if object is closed, else 0.0
	"""

	def reward(env, params):
		name = params.get("object")
		if not name:
			raise ValueError("params must include 'object' name")
		obj_state = env.object_states_dict[name]
		ok = bool(getattr(obj_state, "is_close")() )
		return 1.0 if ok else 0.0

	return reward


class OnTopPreference(BaseMotionPreference):
    def __init__(
        self,
        subject: str,
        target: str,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.subject = subject
        self.target = target
        self._reward_func = on_top()

    def get_reward(self, env: Any) -> float:
        return self._reward_func(env, {"subject": self.subject, "target": self.target})

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.subject, self.target}


class InPreference(BaseMotionPreference):
    def __init__(
        self,
        subject: str,
        target: str,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.subject = subject
        self.target = target
        self._reward_func = in_relation()

    def get_reward(self, env: Any) -> float:
        return self._reward_func(env, {"subject": self.subject, "target": self.target})

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.subject, self.target}


class InContactPreference(BaseMotionPreference):
    def __init__(
        self,
        subject: str,
        target: str,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.subject = subject
        self.target = target
        self._reward_func = in_contact()

    def get_reward(self, env: Any) -> float:
        return self._reward_func(env, {"subject": self.subject, "target": self.target})

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.subject, self.target}


class OpenedPreference(BaseMotionPreference):
    def __init__(
        self,
        object_name: str,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.object_name = object_name
        self._reward_func = opened()

    def get_reward(self, env: Any) -> float:
        return self._reward_func(env, {"object": self.object_name})

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.object_name}


class ClosedPreference(BaseMotionPreference):
    def __init__(
        self,
        object_name: str,
        pref_description: str = "",
        pos_description: str = "",
    ):
        super().__init__(pref_description, pos_description)
        self.object_name = object_name
        self._reward_func = closed()

    def get_reward(self, env: Any) -> float:
        return self._reward_func(env, {"object": self.object_name})

    @property
    def objects_of_interest(self) -> Set[str]:
        return {self.object_name}


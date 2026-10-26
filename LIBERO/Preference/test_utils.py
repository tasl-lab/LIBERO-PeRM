"""
Lightweight test utilities for faking a minimal LIBERO-like environment.

Use make_fake_env(...) to create an object with the fields used by low-level
preferences, specifically gripper position via env.robots[0].eef_name and
env.sim.model.body_name2id / env.sim.data.body_xpos. Optionally supports
site positions (env.sim.data.get_site_xpos) and an obj_body_id mapping.

Example
-------
from .test_utils import make_fake_env
from .low_level import distance

env = make_fake_env(gripper_pos=[1.0, 2.0, 3.0])
reward_fn = distance.distance_to_point()
assert reward_fn(env, {"point": [1.0, 2.0, 3.0]}) == 1.0
"""
import pdb

from typing import Dict, List, Optional
import numpy as np

__all__ = ["make_fake_env"]

def get_min_body_distance(env, body_name, dist_max=0.5,
                          robo_part_name="gripper0_hand_collision",
                          verbose=False):

	"""
	获取指定 body 与 gripper 的最小 mesh distance。

	Args:
		env: 环境
		body_name: str, 目标 body 名称
		dist_max: float, 最大检测距离
		robo_part_name: str, 机器人部件 geom 名称
		verbose: 是否打印详细信息

	Returns:
		min_distance: float, 该 body 下所有 geom 的最小 mesh distance
		min_info: dict, 对应的 geom 及详细信息
	"""
	sim = env.sim
	model = sim.model
	data = sim.data

	# 获取 gripper geom id
	robo_part_id = model.geom_name2id(robo_part_name)

	# 获取 body id
	# body_name="cream_cheese_1_main"
	try:
		body_id = model.body_name2id(body_name)
	except Exception:
		raise ValueError(f"Body name {body_name} not found in model.")

	# 找到该 body 的所有 geom
	geom_ids = [gid for gid in range(model.ngeom)
				if model.geom_bodyid[gid] == body_id]
	# for gid in geom_ids:
	# 	print(f"Body {body_name} has geom {model.geom_id2name(gid)}")
	# import pdb; pdb.set_trace()

	if len(geom_ids) == 0:
		raise ValueError(f"Body {body_name} has no associated geoms!")

	if verbose:
		print(f"[{body_name}] has {len(geom_ids)} geoms: "
			  f"{[model.geom_id2name(gid) for gid in geom_ids]}")

	min_distance = float("inf")
	min_info = None
	# pdb.set_trace()

	# 遍历 geom，找最小 mesh distance
	for object_id in geom_ids:
		object_name = model.geom_id2name(object_id)
		object_pos = data.geom_xpos[object_id]
		gripper_pos = data.geom_xpos[robo_part_id]
		central_distance = np.linalg.norm(object_pos - gripper_pos)

		# ⚡ mesh distance
		mesh_distance = get_obj2obj_mesh_distance(sim,
												  robo_part_id,
												  object_id,
												  dist_max)
		if mesh_distance is None:
			continue



		if mesh_distance < min_distance:
			# print(f"Update min distance: {min_distance:.4f} -> {mesh_distance:.4f}, ")
			min_distance = mesh_distance
			min_info = {
				"body_name": body_name,
				"geom_name": object_name,
				"robo_part_name": robo_part_name,
				"central_distance": central_distance,
				"mesh_distance": mesh_distance,
			}


	return min_distance, min_info


def get_obj2obj_mesh_distance(sim, robo_part_id, object_id, max_margin):
	model = sim.model
	data = sim.data

	# 记录原始 margin
	original_margin = model.geom_margin[robo_part_id].copy()

	# 临时增大 margin，以扩大检测范围
	model.geom_margin[robo_part_id] = max_margin
	sim.forward()

	mesh_distance = None

	for i in range(data.ncon):
		contact = data.contact[i]
		g1, g2 = contact.geom1, contact.geom2
		g1_name = model.geom_id2name(g1)
		g2_name = model.geom_id2name(g2)
		# if "wine" in g1_name or "wine" in g2_name:
		# 		print(f"Contact {i}: geom1={g1}({g1_name}), geom2={g2}({g2_name}), dist={contact.dist:.4f}")
		if ((contact.geom1 == robo_part_id and contact.geom2 == object_id) or
			(contact.geom2 == robo_part_id and contact.geom1 == object_id)):
			mesh_distance = contact.dist
			break

	# import pdb; pdb.set_trace()
	# 恢复 margin
	model.geom_margin[robo_part_id] = original_margin
	sim.forward()
	return mesh_distance




class _Model:
	def __init__(self, name_to_id: Dict[str, int]):
		self._name_to_id = dict(name_to_id)

	def body_name2id(self, name: str) -> int:
		return self._name_to_id[name]



class _Data:
	def __init__(self, id_to_pos: List[np.ndarray], site_name_to_pos: Dict[str, np.ndarray], body_linvel_by_name: Dict[str, np.ndarray], id_to_quat: Optional[List[np.ndarray]] = None):
		# World-frame body positions (meters), shaped (n_bodies, 3), indexed by body id
		# to mimic MuJoCo's sim.data.body_xpos behavior used in tests.
		self.body_xpos = np.asarray(id_to_pos, dtype=float)
		# World-frame body quaternions (wxyz), indexed by body id. Defaults to identity.
		if id_to_quat is None:
			id_to_quat = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in id_to_pos]
		self.body_xquat = np.asarray(id_to_quat, dtype=float)
		self._site_name_to_pos = {k: np.asarray(v, dtype=float) for k, v in site_name_to_pos.items()}
		# Optional world-frame linear velocities for bodies, keyed by body name
		self._body_linvel_by_name = {k: np.asarray(v, dtype=float) for k, v in body_linvel_by_name.items()}

	def get_site_xpos(self, site_name: str) -> np.ndarray:
		# World-frame site position (3,), mimicking sim.data.get_site_xpos(site_name)
		return self._site_name_to_pos[site_name]

	def get_body_xvelp(self, body_name: str) -> np.ndarray:
		# World-frame linear velocity (3,) for a named body; default zeros if unspecified
		return self._body_linvel_by_name.get(body_name, np.zeros(3, dtype=float))


class _Sim:
	def __init__(self, model: _Model, data: _Data):
		self.model = model
		self.data = data


class _Robot:
	def __init__(self, eef_name: str):
		self.eef_name = eef_name


class _Env:
	def __init__(self, robots, sim, obj_body_id: Optional[Dict[str, int]] = None):
		self.robots = robots
		self.sim = sim
		self.obj_body_id = obj_body_id or {}


def make_fake_env(
	gripper_pos: List[float],
	bodies: Optional[Dict[str, List[float]]] = None,
	sites: Optional[Dict[str, List[float]]] = None,
	eef_name: str = "gripper0_eef",
	include_obj_body_id: bool = True,
	eef_vel: Optional[List[float]] = None,
	body_vels: Optional[Dict[str, List[float]]] = None,
	body_quats: Optional[Dict[str, List[float]]] = None,
	base_z: Optional[float] = None,
):
	"""Create a minimal fake env for unit tests.

	Args:
		gripper_pos: 3-list for the end-effector world position.
		bodies: optional mapping name -> [x,y,z]. The eef body (eef_name) will be
			added automatically using gripper_pos if not provided.
		sites: optional mapping site_name -> [x,y,z] for get_site_xpos.
		eef_name: name of the end-effector body used by tests (default: gripper0_eef).
		include_obj_body_id: if True, expose env.obj_body_id mapping from body name to id.

	Returns:
		An object mimicking the subset of env attributes used by low-level distance tests.
	"""
	# Prepare bodies dict including eef
	bodies = dict(bodies or {})
	if eef_name not in bodies:
		bodies[eef_name] = gripper_pos
	# Ensure robot0_base exists for prefs that read base z (HeightOffset).
	if "robot0_base" not in bodies:
		base_z_val = 0.0 if base_z is None else float(base_z)
		bodies["robot0_base"] = [0.0, 0.0, base_z_val]

	# Assign stable ids
	name_to_id: Dict[str, int] = {}
	id_to_pos: List[np.ndarray] = []
	id_to_quat: List[np.ndarray] = []
	body_quats = dict(body_quats or {})
	for idx, (name, pos) in enumerate(bodies.items()):
		name_to_id[name] = idx
		id_to_pos.append(np.asarray(pos, dtype=float))
		q = body_quats.get(name, [1.0, 0.0, 0.0, 0.0])
		id_to_quat.append(np.asarray(q, dtype=float))

	model = _Model(name_to_id)
	# Prepare body linear velocities
	body_linvel_by_name: Dict[str, np.ndarray] = {}
	for name, vel in (body_vels or {}).items():
		body_linvel_by_name[name] = np.asarray(vel, dtype=float)
	if eef_vel is not None:
		body_linvel_by_name[eef_name] = np.asarray(eef_vel, dtype=float)

	data = _Data(id_to_pos=id_to_pos, site_name_to_pos=sites or {}, body_linvel_by_name=body_linvel_by_name, id_to_quat=id_to_quat)
	sim = _Sim(model=model, data=data)
	robots = [_Robot(eef_name=eef_name)]
	obj_body_id = dict(name_to_id) if include_obj_body_id else {}

	return _Env(robots=robots, sim=sim, obj_body_id=obj_body_id)





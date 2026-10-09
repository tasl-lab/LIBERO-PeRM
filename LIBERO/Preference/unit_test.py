"""
Test the functionality of the module here
define testing input and expected output
"""

import math
from .low_level import distance as distance
from .test_utils import make_fake_env
import numpy as np
from .low_level.motion_preference import BaseMotionPreference
import unittest
from .low_level import speed as speed


# Legacy imports: these classes no longer exist in distance.py; wrap so the
# module still imports and the refactored tests can run.
try:
    from .low_level.distance import (
        DistanceToPointPreference,
        DistanceToAreaPreference,
        DistanceToObjectPreference,
    )
    _LEGACY_DISTANCE_AVAILABLE = True
except ImportError:
    DistanceToPointPreference = None
    DistanceToAreaPreference = None
    DistanceToObjectPreference = None
    _LEGACY_DISTANCE_AVAILABLE = False
from .low_level.extra.relationship import (
    OnTopPreference,
    InPreference,
    InContactPreference,
    OpenedPreference,
    ClosedPreference,
)
from .low_level.speed import SpeedRangePreference





class TestMotionPreference(unittest.TestCase):
	"""Dedicated tests for MotionPreference behavior."""

	def test_distance_to_point_minimal(self):
		pref = DistanceToPointPreference(target_point=[0.0, 0.0, 0.0])
		env = make_fake_env(gripper_pos=[0.0, 0.0, 0.0])
		self.assertTrue(math.isclose(pref.get_reward(env), 1.0, rel_tol=1e-6))

	def test_speed_band_minimal(self):
		# Desired 0.1-0.2 m/s; gripper at 0.15 -> reward 1
		pref = SpeedRangePreference(min_speed=0.1, max_speed=0.2)
		env = make_fake_env(gripper_pos=[0.0, 0.0, 0.0], eef_vel=[0.0, 0.0, 0.15])
		self.assertTrue(math.isclose(pref.get_reward(env), 1.0, rel_tol=1e-6))


class TestDistanceToPoint(unittest.TestCase):
	def test_basic(self):
		reward_fn = distance.distance_to_point()
		# Case 1: Zero distance -> reward 1.0
		reward = reward_fn(env=None, params={"from": [1.0, 2.0, 3.0], "point": [1.0, 2.0, 3.0]})
		self.assertTrue(math.isclose(reward, 1.0, rel_tol=1e-6))
		# Case 2: Distance 5 -> reward 1/(1+5) = 1/6
		reward = reward_fn(env=None, params={"from": [0.0, 0.0, 0.0], "point": [3.0, 4.0, 0.0]})
		self.assertTrue(math.isclose(reward, 1.0 / 6.0, rel_tol=1e-6))
		# Case 3: Distance 3 -> reward 1/(1+3) = 0.25
		reward = reward_fn(env=None, params={"from": [0.0, 0.0, 0.0], "point": [1.0, 2.0, 2.0]})
		self.assertTrue(math.isclose(reward, 0.25, rel_tol=1e-6))

	def test_gripper_fake_env(self):
		reward_fn = distance.distance_to_point()
		env = make_fake_env(gripper_pos=[1.0, 2.0, 3.0])
		reward = reward_fn(env, {"point": [1.0, 2.0, 3.0]})
		self.assertTrue(math.isclose(reward, 1.0, rel_tol=1e-6))


class TestDistanceToArea(unittest.TestCase):
	def test_aabb_inside_outside(self):
		reward_fn = distance.distance_to_area()
		area = {"type": "aabb", "min": [0.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
		# Inside -> reward = 1.0
		r = reward_fn(env=None, params={"from": [0.5, 0.0, 0.0], "area": area})
		self.assertTrue(math.isclose(r, 1.0, rel_tol=1e-6))
		# Outside by 1.0 along +x -> reward 0.5
		r = reward_fn(env=None, params={"from": [2.0, 0.0, 0.0], "area": area})
		self.assertTrue(math.isclose(r, 0.5, rel_tol=1e-6))

	def test_box_center_size(self):
		reward_fn = distance.distance_to_area()
		area = {"type": "box", "center": [1.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]}
		# On boundary -> inside -> 1.0
		r = reward_fn(env=None, params={"from": [0.0, 0.0, 0.0], "area": area})
		self.assertTrue(math.isclose(r, 1.0, rel_tol=1e-6))
		# Outside by 1.0 along +x -> 0.5
		r = reward_fn(env=None, params={"from": [3.0, 0.0, 0.0], "area": area})
		self.assertTrue(math.isclose(r, 0.5, rel_tol=1e-6))

	def test_site_box_with_fake_env(self):
		reward_fn = distance.distance_to_area()
		env = make_fake_env(gripper_pos=[1.25, 0.0, 0.0], sites={"target_zone": [1.0, 0.0, 0.0]})
		area = {"type": "site_box", "site": "target_zone", "size": [0.2, 0.2, 0.2]}
		# Half-size = 0.1, gripper outside by 0.15 -> 1/(1+0.15)
		r = reward_fn(env, params={"area": area})
		self.assertTrue(math.isclose(r, 1.0 / 1.15, rel_tol=1e-6))


class TestDistanceToObject(unittest.TestCase):
	def test_body_and_site(self):
		# This test is tricky because the original reward function doesn't use the
		# parameters that the new class requires (mode, min/max dist).
		# We test the class init and that the reward works as it did before.
		pref = DistanceToObjectPreference(
            target_obj="target_body",
            distance_metric="greater_than",
            threshold=0.1
        )
		reward_fn = pref.get_reward
		# Body target at [1,0,0] from origin -> 0.5
		env = make_fake_env(gripper_pos=[0.0, 0.0, 0.0], bodies={"target_body_main": [1.0, 0.0, 0.0]})
		r = reward_fn(env)
		self.assertTrue(math.isclose(r, 1.0, rel_tol=1e-6))


class TestRelationshipPreference(unittest.TestCase):
	class _StubState:
		def __init__(self, name, pos):
			self.object_name = name
			self._pos = np.asarray(pos, dtype=float)

		def get_geom_state(self):
			return {"pos": self._pos}

		def check_contact(self, other):
			# Assume contact whenever XY distance is small
			p = self._pos
			op = other.get_geom_state()["pos"]
			dxy = math.hypot(float(p[0] - op[0]), float(p[1] - op[1]))
			return dxy < 0.03

		def check_ontop(self, other):
			# Mimic target.check_ontop(subject): subject above target and close in XY
			p = self._pos  # target
			op = other.get_geom_state()["pos"]  # subject
			dxy = math.hypot(float(p[0] - op[0]), float(p[1] - op[1]))
			return (p[2] <= op[2]) and (dxy < 0.03) and self.check_contact(other)

	def _make_env(self, subject_pos, target_pos):
		# Minimal env exposing object_states_dict expected by relationship.on_top
		env = type("_RelEnv", (), {})()
		env.object_states_dict = {
			"bowl": self._StubState("bowl", subject_pos),
			"plate": self._StubState("plate", target_pos),
		}
		return env

	def test_on_top_true(self):
		pref = OnTopPreference(subject="bowl", target="plate")
		env = self._make_env(subject_pos=[0.0, 0.0, 0.55], target_pos=[0.0, 0.0, 0.50])
		self.assertEqual(pref.get_reward(env), 1.0)

	def test_on_top_false_xy_far(self):
		pref = OnTopPreference(subject="bowl", target="plate")
		env = self._make_env(subject_pos=[0.10, 0.00, 0.55], target_pos=[0.0, 0.0, 0.50])
		self.assertEqual(pref.get_reward(env), 0.0)

	def test_on_top_false_below(self):
		pref = OnTopPreference(subject="bowl", target="plate")
		env = self._make_env(subject_pos=[0.0, 0.0, 0.45], target_pos=[0.0, 0.0, 0.50])
		self.assertEqual(pref.get_reward(env), 0.0)


class TestRelationshipPreferenceMore(unittest.TestCase):
	class _StubState:
		def __init__(self, name, pos, contain_ok=False, open_ok=False, close_ok=False):
			self.object_name = name
			self._pos = np.asarray(pos, dtype=float)
			self._contain_ok = contain_ok
			self._open_ok = open_ok
			self._close_ok = close_ok

		def get_geom_state(self):
			return {"pos": self._pos}

		def check_contact(self, other):
			# Contact iff XY distance < 0.03
			p = self._pos
			op = other.get_geom_state()["pos"]
			dxy = math.hypot(float(p[0] - op[0]), float(p[1] - op[1]))
			return dxy < 0.03

		def check_contain(self, other):
			return bool(self._contain_ok)

		def is_open(self):
			return bool(self._open_ok)

		def is_close(self):
			return bool(self._close_ok)

	def _make_env(self, subject_state, target_state=None):
		env = type("_RelEnv2", (), {})()
		d = {subject_state.object_name: subject_state}
		if target_state is not None:
			d[target_state.object_name] = target_state
		env.object_states_dict = d
		return env

	def test_in_true(self):
		# subject inside target and in contact
		subj = self._StubState("obj", [0, 0, 0])
		tgt = self._StubState("container", [0, 0, 0], contain_ok=True)
		env = self._make_env(subj, tgt)
		pref = InPreference(subject="obj", target="container")
		self.assertEqual(pref.get_reward(env), 1.0)

	def test_in_false_no_contain(self):
		subj = self._StubState("obj", [0, 0, 0])
		tgt = self._StubState("container", [0.1, 0.1, 0.0], contain_ok=False)
		env = self._make_env(subj, tgt)
		pref = InPreference(subject="obj", target="container")
		self.assertEqual(pref.get_reward(env), 0.0)

	def test_in_contact_true(self):
		subj = self._StubState("a", [0.0, 0.0, 0.0])
		tgt = self._StubState("b", [0.01, 0.01, 0.0])
		env = self._make_env(subj, tgt)
		pref = InContactPreference(subject="a", target="b")
		self.assertEqual(pref.get_reward(env), 1.0)

	def test_in_contact_false(self):
		subj = self._StubState("a", [0.0, 0.0, 0.0])
		tgt = self._StubState("b", [0.1, 0.1, 0.0])
		env = self._make_env(subj, tgt)
		pref = InContactPreference(subject="a", target="b")
		self.assertEqual(pref.get_reward(env), 0.0)

	def test_opened_and_closed(self):
		obj_open = self._StubState("door", [0, 0, 0], open_ok=True, close_ok=False)
		env_open = self._make_env(obj_open)
		self.assertEqual(OpenedPreference(object_name="door").get_reward(env_open), 1.0)
		self.assertEqual(ClosedPreference(object_name="door").get_reward(env_open), 0.0)

		obj_closed = self._StubState("drawer", [0, 0, 0], open_ok=False, close_ok=True)
		env_closed = self._make_env(obj_closed)
		self.assertEqual(OpenedPreference(object_name="drawer").get_reward(env_closed), 0.0)
		self.assertEqual(ClosedPreference(object_name="drawer").get_reward(env_closed), 1.0)

class TestRegistryAndRefactoredRewards(unittest.TestCase):
	"""Tests for the refactored BaseMotionPreference + registry + fixed rewards."""

	def test_registry_populated(self):
		# Touch all subclass modules to trigger __init_subclass__ registration.
		from .low_level import speed, distance, trajectory, relationship  # noqa: F401
		reg = BaseMotionPreference._registry
		for key in (
			"speed_range", "acc_range",
			"approach_to_object", "avoid_to_object",
			"straight_path", "height_offset", "rotation",
			"on_top", "in", "in_contact", "opened", "closed",
		):
			self.assertIn(key, reg, f"registry missing {key}")

	def test_satisfaction_normalization(self):
		from .low_level.speed import SpeedRangePreference
		p = SpeedRangePreference(min_speed=0.1, max_speed=0.2)
		# saturation = 0.0450; mean_reward=0.0450 → 100%
		self.assertAlmostEqual(p.satisfaction(0.0450), 100.0, places=4)
		self.assertAlmostEqual(p.satisfaction(0.0225), 50.0, places=4)
		self.assertAlmostEqual(p.satisfaction(1.0), 100.0, places=4)  # clamped to 100%

	def test_acc_range_band(self):
		from .low_level.speed import AccRangePreference
		# "accelerate sharply": min_acc high
		p_aggressive = AccRangePreference(min_acc=1.0, max_acc=float("inf"))
		# Build env with changing EE velocity to simulate acceleration
		env1 = make_fake_env(gripper_pos=[0, 0, 0], eef_vel=[0, 0, 0])
		env2 = make_fake_env(gripper_pos=[0, 0, 0], eef_vel=[0, 0, 2.0])
		p_aggressive.reset(env1)
		r1 = p_aggressive.get_reward(env1)
		r2 = p_aggressive.get_reward(env2)
		# First step: prev_vel None → a_mag=0; peak=0 < min_acc → sigmoid(-k) small, < 0.5
		self.assertLess(r1, 0.5)
		# Second step: Δvel=[0,0,2] → a_mag=2; peak=2 in [1, inf] → reward=1.0
		self.assertAlmostEqual(r2, 1.0, places=6)

		# "ease in": max_acc low
		p_gentle = AccRangePreference(min_acc=0.0, max_acc=0.5)
		env3 = make_fake_env(gripper_pos=[0, 0, 0], eef_vel=[0, 0, 5.0])
		p_gentle.reset(env1)
		p_gentle.get_reward(env1)  # a_mag=0
		r_gentle_violated = p_gentle.get_reward(env3)  # a_mag=5 > max_acc=0.5
		self.assertLess(r_gentle_violated, 0.5)

	def test_straight_path(self):
		from .low_level.trajectory import StraightPathPreference
		p = StraightPathPreference(max_deviation=0.05)
		# Simulate straight motion: ee moves [0,0,0] → [1,0,0] → [2,0,0]
		envs = [make_fake_env(gripper_pos=[x, 0, 0]) for x in (0.0, 1.0, 2.0)]
		p.reset(envs[0])
		rewards = [p.get_reward(e) for e in envs]
		# All should be ≈ 1.0 (perfectly straight)
		for r in rewards:
			self.assertGreater(r, 0.95)
		# Now simulate L-M-D zig-zag: [0,0,0] → [0,0,0.2] → [1,0,0.2] → [1,0,0]
		p2 = StraightPathPreference(max_deviation=0.05)
		zigzag_envs = [
			make_fake_env(gripper_pos=[0, 0, 0]),
			make_fake_env(gripper_pos=[0, 0, 0.2]),
			make_fake_env(gripper_pos=[1, 0, 0.2]),
			make_fake_env(gripper_pos=[1, 0, 0]),
		]
		p2.reset(zigzag_envs[0])
		final_r = [p2.get_reward(e) for e in zigzag_envs][-1]
		# path=1.4, disp=1.0, inefficiency=0.4; tolerance=0.5 → reward ≈ 0.2
		self.assertLess(final_r, 0.5)

	def test_height_offset(self):
		from .low_level.trajectory import HeightOffsetPreference
		# Target +0.1 above default safe_height (base_z + 0.2 = 0 + 0.2 = 0.2)
		p = HeightOffsetPreference(height_offset=0.1)
		# Episode where ee max z reaches 0.3 → observed_offset = 0.3 - 0.2 = 0.1 = target
		envs = [make_fake_env(gripper_pos=[0, 0, z], base_z=0.0) for z in (0.0, 0.2, 0.3, 0.2)]
		p.reset(envs[0])
		rewards = [p.get_reward(e) for e in envs]
		self.assertAlmostEqual(rewards[-1], 1.0, places=4)

		# Mismatch: target +0.1, but ee only rises to 0.25 → observed=0.05, err=0.05, reward=0.5
		p2 = HeightOffsetPreference(height_offset=0.1)
		envs2 = [make_fake_env(gripper_pos=[0, 0, z], base_z=0.0) for z in (0.0, 0.25)]
		p2.reset(envs2[0])
		[p2.get_reward(e) for e in envs2]
		final_r = p2.get_reward(envs2[-1])
		self.assertTrue(0.3 < final_r < 0.7)

	def test_rotation(self):
		from .low_level.trajectory import RotationPreference
		p = RotationPreference(angle_per_step=0.5)
		# Stationary (no pos change) → reward 1 regardless of rotation
		env_static = make_fake_env(gripper_pos=[0, 0, 0])
		p.reset(env_static)
		r_init = p.get_reward(env_static)
		self.assertAlmostEqual(r_init, 1.0, places=6)
		r_static = p.get_reward(env_static)
		self.assertAlmostEqual(r_static, 1.0, places=6)

		# Translating + rotating 0.5 rad/step → reward near 1.0
		# quat (wxyz): identity [1,0,0,0]; rotated by 0.5 rad about z: [cos(0.25), 0, 0, sin(0.25)]
		import math as _math
		q1 = [1.0, 0.0, 0.0, 0.0]
		q2 = [_math.cos(0.25), 0.0, 0.0, _math.sin(0.25)]  # 0.5 rad about z
		p2 = RotationPreference(angle_per_step=0.5)
		env_a = make_fake_env(gripper_pos=[0, 0, 0], body_quats={"gripper0_eef": q1})
		env_b = make_fake_env(gripper_pos=[0.1, 0, 0], body_quats={"gripper0_eef": q2})
		p2.reset(env_a)
		p2.get_reward(env_a)
		r_rotating = p2.get_reward(env_b)
		self.assertGreater(r_rotating, 0.9)

		# Translating but NOT rotating → reward 0 (dang=0, dpos>0)
		p3 = RotationPreference(angle_per_step=0.5)
		env_c = make_fake_env(gripper_pos=[0, 0, 0], body_quats={"gripper0_eef": q1})
		env_d = make_fake_env(gripper_pos=[0.1, 0, 0], body_quats={"gripper0_eef": q1})
		p3.reset(env_c)
		p3.get_reward(env_c)
		r_trans_only = p3.get_reward(env_d)
		self.assertAlmostEqual(r_trans_only, 0.0, places=4)

	def test_factory_relationship_registered(self):
		"""Previously relationship classes were NOT in PREFERENCE_CLASSES;
		after refactor they should be reachable via create_motion_preference."""
		from .pref_gen import create_motion_preference
		p = create_motion_preference(
			preference_type="on_top",
			preference_parameters={"subject": "a", "target": "b"},
			pref_description="test",
			pos_description=None,
		)
		self.assertIsNotNone(p)
		p2 = create_motion_preference(
			preference_type="opened",
			preference_parameters={"object": "door"},  # legacy key 'object' → 'object_name'
			pref_description="test",
			pos_description=None,
		)
		self.assertIsNotNone(p2)
		# Legacy alias 'max_acc' → 'acc_range'
		p3 = create_motion_preference(
			preference_type="max_acc",
			preference_parameters={"max_acc": 0.5},
			pref_description="test",
			pos_description=None,
		)
		self.assertIsNotNone(p3)


if __name__ == "__main__":
	unittest.main()
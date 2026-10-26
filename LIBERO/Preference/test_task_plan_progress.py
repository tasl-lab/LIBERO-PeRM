"""Pure-Python smoke test for TaskPlanProgressPreference (no MuJoCo needed).

Run:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python -m unittest Preference.test_task_plan_progress -v
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from Preference.high_level.task_plan_progress import (
    TaskPlanProgressPreference,
    _is_subsequence,
    _lcs_len,
)


class _MockContact:
    __slots__ = ("geom1", "geom2")

    def __init__(self, g1: int, g2: int):
        self.geom1 = g1
        self.geom2 = g2


class _MockEnv:
    """Minimal env supporting body_xpos/body_xquat lookup + injected contacts."""

    def __init__(self, obj_states: Dict[str, Dict[str, np.ndarray]], eef_pos):
        self._states = obj_states
        self._name_to_id: Dict[str, int] = {n: i for i, n in enumerate(obj_states.keys())}
        self.obj_body_id: Dict[str, int] = dict(self._name_to_id)
        self._eef_pos = np.asarray(eef_pos, dtype=np.float32)
        self._eef_id = len(self._name_to_id)

        body_names = list(self._name_to_id.keys()) + ["gripper0_eef"]
        body_xpos = [self._states[n]["pos"] for n in body_names[:-1]] + [self._eef_pos]
        body_xquat = [self._states[n]["quat"] for n in body_names[:-1]] + [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ]

        # Geoms: 1 per obj using name "<obj>_g0"; 1 gripper geom "gripper0_finger_g0".
        self._geom_names: List[str] = []
        self._geom_name_to_id: Dict[str, int] = {}
        for n in obj_states.keys():
            gid = len(self._geom_names)
            gname = f"{n}_g0"
            self._geom_names.append(gname)
            self._geom_name_to_id[gname] = gid
        gripper_gid = len(self._geom_names)
        self._geom_names.append("gripper0_finger_g0")
        self._geom_name_to_id["gripper0_finger_g0"] = gripper_gid
        self._gripper_gid = gripper_gid

        self._contacts: List[_MockContact] = []

        sim_model = SimpleNamespace(
            body_name2id=lambda name: self._eef_id if name == "gripper0_eef" else self._name_to_id[name],
            ngeom=len(self._geom_names),
            geom_id2name=lambda gid: self._geom_names[gid] if 0 <= gid < len(self._geom_names) else None,
        )
        sim_data_ns = SimpleNamespace(
            body_xpos=np.asarray(body_xpos, dtype=np.float32),
            body_xquat=np.asarray(body_xquat, dtype=np.float32),
        )
        # Bind ncon as property-like via attribute that reads list length each time.
        outer = self
        sim_data_ns.contact = outer._contacts

        class _SimData:
            body_xpos = sim_data_ns.body_xpos
            body_xquat = sim_data_ns.body_xquat
            contact = outer._contacts

            @property
            def ncon(self_inner):  # noqa: N805
                return len(outer._contacts)

        self.sim = SimpleNamespace(model=sim_model, data=_SimData())
        self.robots = [SimpleNamespace(eef_name="gripper0_eef")]

    def move_obj(self, name: str, new_pos):
        idx = self._name_to_id[name]
        self.sim.data.body_xpos[idx] = np.asarray(new_pos, dtype=np.float32)
        self._states[name]["pos"] = self.sim.data.body_xpos[idx]

    def move_eef(self, new_pos):
        self.sim.data.body_xpos[self._eef_id] = np.asarray(new_pos, dtype=np.float32)

    def set_contacts(self, pairs: List[Tuple[str, str]]):
        """Each pair is (geom_name_or_obj, geom_name_or_obj_or_'gripper')."""
        self._contacts.clear()
        for a, b in pairs:
            ga = self._resolve_geom(a)
            gb = self._resolve_geom(b)
            self._contacts.append(_MockContact(ga, gb))

    def _resolve_geom(self, x: str) -> int:
        if x == "gripper":
            return self._gripper_gid
        return self._geom_name_to_id.get(f"{x}_g0", self._gripper_gid)


def _state(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> Dict[str, np.ndarray]:
    return {"pos": np.asarray(pos, dtype=np.float32), "quat": np.asarray(quat, dtype=np.float32)}


class TestPureHelpers(unittest.TestCase):
    def test_subsequence(self):
        self.assertTrue(_is_subsequence(["a", "b", "c"], ["a", "b", "c"]))
        self.assertTrue(_is_subsequence(["x", "a", "y", "b"], ["a", "b"]))
        self.assertFalse(_is_subsequence(["b", "a"], ["a", "b"]))

    def test_lcs(self):
        self.assertEqual(_lcs_len(["a", "b", "c"], ["a", "b", "c"]), 3)
        self.assertEqual(_lcs_len(["a", "x", "b"], ["a", "b"]), 2)
        self.assertEqual(_lcs_len(["b", "a"], ["a", "b"]), 1)


class TestTaskPlanProgress(unittest.TestCase):
    def setUp(self):
        # task_plan = pick alphabet → put basket → pick cream → put basket
        self.task_plan = [
            ("pick", "alphabet_soup_1"),
            ("put", "basket_1"),
            ("pick", "cream_cheese_1"),
            ("put", "basket_1"),
        ]
        self.metric = TaskPlanProgressPreference(task_plan=self.task_plan)
        self.env = _MockEnv(
            obj_states={
                "alphabet_soup_1": _state([0.5, 0.0, 0.9]),
                "basket_1": _state([0.7, 0.2, 0.9]),
                "cream_cheese_1": _state([0.5, 0.3, 0.9]),
                "milk_1": _state([0.3, 0.0, 0.9]),
            },
            eef_pos=[0.0, 0.0, 1.2],
        )
        self.metric.reset(self.env)

    def _move_and_step(self, obj: str, dz=0.05, gripper_contact=True, basket_contact=False):
        new_pos = self.env._states[obj]["pos"] + np.array([0.0, 0.0, dz], dtype=np.float32)
        self.env.move_obj(obj, new_pos)
        pairs = []
        if gripper_contact:
            pairs.append((obj, "gripper"))
        if basket_contact:
            pairs.append((obj, "basket_1"))
        self.env.set_contacts(pairs)
        self.metric.get_reward(self.env)

    def test_perfect_rollout_score_one(self):
        # gripper picks alphabet, drops in basket (alphabet-basket contact),
        # picks cream, drops in basket.
        self._move_and_step("alphabet_soup_1", gripper_contact=True)
        self._move_and_step("alphabet_soup_1", dz=0.0, gripper_contact=False, basket_contact=True)
        self._move_and_step("cream_cheese_1", gripper_contact=True)
        self._move_and_step("cream_cheese_1", dz=0.0, gripper_contact=False, basket_contact=True)
        score = self.metric.compute_score()
        raw = self.metric.raw()
        self.assertAlmostEqual(score, 1.0, places=3)
        self.assertEqual(set(raw["touched_seq"]), {"alphabet_soup_1", "basket_1", "cream_cheese_1"})
        self.assertEqual(raw["progress_per_subtask"], [1.0, 1.0, 1.0])
        self.assertIsNone(raw["spurious_first_obj"])

    def test_spurious_first_object(self):
        # Gripper contacts AND moves an unexpected obj → spurious flagged.
        self.env.set_contacts([("milk_1", "gripper")])
        self.env.move_obj("milk_1", [0.3, 0.0, 1.10])  # > 5cm move
        self.metric.get_reward(self.env)
        raw = self.metric.raw()
        self.assertEqual(raw["spurious_first_obj"], "milk_1")

    def test_brush_past_no_spurious(self):
        # Gripper touches unexpected obj but obj doesn't move → no spurious.
        self.env.set_contacts([("milk_1", "gripper")])
        self.metric.get_reward(self.env)
        raw = self.metric.raw()
        self.assertIsNone(raw["spurious_first_obj"])

    def test_no_spurious_when_expected_touched_first(self):
        # Expected obj contacted first → later unexpected contact does not flag.
        self.env.set_contacts([("alphabet_soup_1", "gripper")])
        self.metric.get_reward(self.env)
        self.env.set_contacts([("milk_1", "gripper")])
        self.metric.get_reward(self.env)
        raw = self.metric.raw()
        self.assertIsNone(raw["spurious_first_obj"])

    def test_passive_movement_no_spurious(self):
        # Distractor jostles (no gripper contact) — should NOT flag spurious.
        self.env.move_obj("milk_1", [0.3, 0.0, 1.0])
        self.metric.get_reward(self.env)
        raw = self.metric.raw()
        self.assertIsNone(raw["spurious_first_obj"])

    def test_touch_alone_completes_subtask(self):
        # Contact alone (no displacement) is enough — gripper-touched fixture
        # case (e.g., basket receiving an object) maps to this.
        self.env.set_contacts([("alphabet_soup_1", "gripper")])
        self.metric.get_reward(self.env)
        raw = self.metric.raw()
        self.assertIn("alphabet_soup_1", raw["touched_seq"])
        self.assertNotIn("alphabet_soup_1", raw["manipulated_set"])
        self.assertEqual(raw["progress_per_subtask"][0], 1.0)

    def test_order_diagnostic_only(self):
        # Order is reported in raw() but does NOT affect compute_score under
        # the satisfaction-based scheme.
        self.env.set_contacts([("cream_cheese_1", "gripper")])
        self.metric.get_reward(self.env)
        self.env.set_contacts([("alphabet_soup_1", "gripper")])
        self.metric.get_reward(self.env)
        raw = self.metric.raw()
        self.assertLess(raw["order_match"], 1.0)
        # cream_cheese_1 is NOT in the test plan's expected_seq (only alphabet/
        # basket/cream); progress for cream/alphabet are both touched → 0.5+0.5+0
        self.assertGreater(self.metric.compute_score(), 0.0)

    def test_zero_when_no_touch_no_move(self):
        for _ in range(5):
            self.metric.get_reward(self.env)
        self.assertEqual(self.metric.compute_score(), 0.0)


if __name__ == "__main__":
    unittest.main()

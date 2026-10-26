"""Pure-Python tests for TaskPlanCompliancePreference (no MuJoCo needed).

Reuses the `_MockEnv` / `_state` harness from test_task_plan_progress (import,
not copy). Covers per-subtask 2-stage (engaged/completed) scoring, mean
aggregation, degrade paths, reset, and the loader.

Run:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python -m unittest Preference.test_task_plan_compliance -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

from Preference.test_task_plan_progress import _MockEnv, _state
from Preference.high_level.task_plan_progress import (
    TaskPlanCompliancePreference,
    load_task_plan_compliance,
)


def _env() -> _MockEnv:
    return _MockEnv(
        obj_states={
            "alphabet_soup_1": _state([0.5, 0.0, 0.9]),
            "basket_1": _state([0.7, 0.2, 0.9]),
            "cream_cheese_1": _state([0.5, 0.3, 0.9]),
            "button_1": _state([0.3, 0.0, 0.9]),
        },
        eef_pos=[0.0, 0.0, 1.2],  # far from objs → no proximity-touch noise
    )


class TestTaskPlanCompliance(unittest.TestCase):
    # ─── helpers driving the mock through a subtask ─────────────────────
    def _pick(self, env, m, obj, lift=True, grip=True):
        if lift:
            p = env._states[obj]["pos"]
            env.move_obj(obj, [float(p[0]), float(p[1]), 0.95])  # init z 0.9 → +0.05 > LIFT
        env.set_contacts([(obj, "gripper")] if grip else [])
        m.get_reward(env)

    def _put(self, env, m, carried, target, release=True):
        tp = env._states[target]["pos"]
        env.move_obj(carried, [float(tp[0]), float(tp[1]), float(tp[2]) + 0.05])  # near target
        env.set_contacts([] if release else [(carried, "gripper")])
        m.get_reward(env)

    def _new(self, task_plan):
        env = _env()
        m = TaskPlanCompliancePreference(task_plan=task_plan)
        m.reset(env)
        return env, m

    # ─── full / partial rollouts (mean aggregation) ─────────────────────
    def test_perfect_k4_score_one(self):
        tp = [("pick", "alphabet_soup_1"), ("put", "basket_1"),
              ("pick", "cream_cheese_1"), ("put", "basket_1")]
        env, m = self._new(tp)
        self._pick(env, m, "alphabet_soup_1")
        self._put(env, m, "alphabet_soup_1", "basket_1")
        self._pick(env, m, "cream_cheese_1")
        self._put(env, m, "cream_cheese_1", "basket_1")
        self.assertAlmostEqual(m.compute_score(), 1.0, places=3)
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(m.satisfaction(0.0), 100.0, places=3)  # cumulative override

    def test_pick_done_put_missing_is_half(self):
        env, m = self._new([("pick", "alphabet_soup_1"), ("put", "basket_1")])
        self._pick(env, m, "alphabet_soup_1")  # only the pick
        self.assertAlmostEqual(m.compute_score(), 0.5, places=3)
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0, 0.0])

    def test_all_zero(self):
        env, m = self._new([("pick", "alphabet_soup_1"), ("put", "basket_1")])
        env.set_contacts([])
        m.get_reward(env)  # touch nothing
        self.assertEqual(m.compute_score(), 0.0)
        self.assertEqual(m.raw()["per_subtask_compliance"], [0.0, 0.0])

    def test_k4_partial_mean(self):
        tp = [("pick", "alphabet_soup_1"), ("put", "basket_1"),
              ("pick", "cream_cheese_1"), ("put", "basket_1")]
        env, m = self._new(tp)
        self._pick(env, m, "alphabet_soup_1")
        self._put(env, m, "alphabet_soup_1", "basket_1")
        self._pick(env, m, "cream_cheese_1")  # skip final put
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(m.compute_score(), 0.75, places=3)

    # ─── engaged (0.5) vs completed (1.0) distinction ───────────────────
    def test_pick_engaged_only(self):
        env, m = self._new([("pick", "alphabet_soup_1")])
        self._pick(env, m, "alphabet_soup_1", lift=False)  # contact, no lift
        self.assertEqual(m.raw()["per_subtask_compliance"], [0.5])

    def test_pick_completed(self):
        env, m = self._new([("pick", "alphabet_soup_1")])
        self._pick(env, m, "alphabet_soup_1")  # contact + lift
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0])

    def test_put_engaged_not_released(self):
        env, m = self._new([("pick", "alphabet_soup_1"), ("put", "basket_1")])
        self._pick(env, m, "alphabet_soup_1")
        self._put(env, m, "alphabet_soup_1", "basket_1", release=False)  # at target, still holding
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0, 0.5])
        self.assertAlmostEqual(m.compute_score(), 0.75, places=3)

    # ─── degrade paths ──────────────────────────────────────────────────
    def test_non_pick_put_degrades_binary(self):
        env, m = self._new([("push", "button_1")])
        env.set_contacts([("button_1", "gripper")])
        m.get_reward(env)
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0])  # contact → 1.0 (no 0.5)

    def test_put_without_preceding_pick_degrades(self):
        env, m = self._new([("put", "basket_1")])
        self.assertEqual(m._carried, [None])
        env.set_contacts([("basket_1", "gripper")])
        m.get_reward(env)
        self.assertEqual(m.raw()["per_subtask_compliance"], [1.0])

    # ─── reset clears latch across episodes ─────────────────────────────
    def test_reset_clears_latch(self):
        env, m = self._new([("pick", "alphabet_soup_1"), ("put", "basket_1")])
        self._pick(env, m, "alphabet_soup_1")
        self.assertEqual(m.raw()["per_subtask_compliance"][0], 1.0)
        m.reset(env)  # next trial reuses the same object
        self.assertEqual(m.raw()["per_subtask_compliance"], [0.0, 0.0])
        self.assertEqual(m.compute_score(), 0.0)

    # ─── loader ─────────────────────────────────────────────────────────
    def test_loader_none_and_instance(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("some_key: 1\n")  # no task_plan
            p_none = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("task_plan:\n  - [pick, a]\n  - [put, b]\n")
            p_plan = f.name
        try:
            self.assertIsNone(load_task_plan_compliance(p_none))
            m = load_task_plan_compliance(p_plan)
            self.assertIsNotNone(m)
            self.assertEqual(m.subtasks, [("pick", "a"), ("put", "b")])
            self.assertEqual(m.preference_type, "task_plan_compliance")
        finally:
            os.unlink(p_none)
            os.unlink(p_plan)


if __name__ == "__main__":
    unittest.main()

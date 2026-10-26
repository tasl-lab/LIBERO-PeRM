"""Regression tests for low-level metric parameterization (Block 1).

Confirms the lifted shaping constants exist with their documented defaults and
are actually wired into the reward formula (changing the instance value changes
the reward). The metric *form* is unchanged.

Run:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python -m unittest Preference.test_low_pref_param_consts -v
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from Preference.low_level.speed import AccRangePreference
from Preference.low_level.trajectory import (
    HeightOffsetPreference,
    StraightPathPreference,
)


class _FakeEnv:
    """Minimal env exposing sim.data.body_xpos for EE position reads."""

    def __init__(self):
        self._pos = np.zeros(3, dtype=float)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        data = types.SimpleNamespace(body_xpos={0: self._pos}, body_xquat={0: self._quat})
        model = types.SimpleNamespace(body_name2id=lambda n: 0)
        self.sim = types.SimpleNamespace(data=data, model=model)
        self.robots = [types.SimpleNamespace(eef_name="gripper0_eef")]

    def set_pos(self, p):
        self._pos[:] = np.asarray(p, dtype=float)
        self.sim.data.body_xpos[0] = self._pos


class TestParamConstants(unittest.TestCase):
    def test_defaults_present(self):
        self.assertEqual(AccRangePreference.SIGMOID_K, 6.0)
        self.assertEqual(StraightPathPreference.INEFFICIENCY_HEADROOM, 10.0)
        self.assertEqual(HeightOffsetPreference.DEFAULT_SAFE_LIFT_M, 0.2)

    def _run_path(self, p, env, points):
        r = None
        for pt in points:
            env.set_pos(pt)
            r = p.get_reward(env)
        return r

    def test_straightpath_headroom_wired(self):
        # Same moderately-inefficient path; larger headroom → higher reward.
        path = [(0, 0, 0), (0.5, 0, 0.5), (1, 0, 0)]  # path/disp ≈ 1.414 → ineff ≈ 0.414
        p10 = StraightPathPreference(max_deviation=0.05)  # headroom 10 (default)
        r10 = self._run_path(p10, _FakeEnv(), path)
        p20 = StraightPathPreference(max_deviation=0.05)
        p20.INEFFICIENCY_HEADROOM = 20.0  # instance shadow
        r20 = self._run_path(p20, _FakeEnv(), path)
        self.assertGreater(r20, r10)
        self.assertGreater(r10, 0.0)
        self.assertLess(r20, 1.0)


if __name__ == "__main__":
    unittest.main()

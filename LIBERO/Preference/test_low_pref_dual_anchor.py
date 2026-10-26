"""Unit tests for the dual-anchor satisfaction formula (LT3).

Covers BaseMotionPreference.satisfaction dual-anchor (naive=0, pref=100, sign
auto-direction, degenerate→None, legacy single-anchor + saturation fallback),
None-safe mapped_satisfaction, and SpeedRange routing through the base formula
via final_raw (step count) after its override was removed.

Run:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python -m unittest Preference.test_low_pref_dual_anchor -v
"""

from __future__ import annotations

import unittest
from typing import Any, Set

from Preference.low_level.motion_preference import BaseMotionPreference
from Preference.low_level.speed import SpeedRangePreference


class _Dummy(BaseMotionPreference):
    """Minimal concrete metric to exercise the base satisfaction formula."""

    saturation = 1.0

    def get_reward(self, env: Any) -> float:
        return 1.0

    @property
    def objects_of_interest(self) -> Set[str]:
        return set()


class TestDualAnchor(unittest.TestCase):
    def test_endpoints_and_mid(self):
        p = _Dummy()
        p.pref_anchor, p.naive_anchor = 0.8, 0.2  # pref=A, naive=B, A!=B
        self.assertAlmostEqual(p.satisfaction(0.2), 0.0, places=3)    # naive → 0
        self.assertAlmostEqual(p.satisfaction(0.8), 100.0, places=3)  # pref  → 100
        self.assertAlmostEqual(p.satisfaction(0.5), 50.0, places=3)   # mid   → 50

    def test_direction_sign_auto(self):
        # pref < naive (speed_up: fewer = better); sign(pref-naive) handles it.
        p = _Dummy()
        p.pref_anchor, p.naive_anchor = 100.0, 400.0
        self.assertAlmostEqual(p.satisfaction(400.0), 0.0, places=3)
        self.assertAlmostEqual(p.satisfaction(100.0), 100.0, places=3)
        self.assertAlmostEqual(p.satisfaction(250.0), 50.0, places=3)

    def test_clip_outside_range(self):
        p = _Dummy()
        p.pref_anchor, p.naive_anchor = 0.8, 0.2
        self.assertEqual(p.satisfaction(1.0), 100.0)  # beyond pref → clip 100
        self.assertEqual(p.satisfaction(0.0), 0.0)    # below naive → clip 0

    def test_degenerate_returns_none(self):
        p = _Dummy()
        p.pref_anchor, p.naive_anchor = 0.5, 0.5  # pref == naive → degenerate
        self.assertIsNone(p.satisfaction(0.5))        # no crash, None
        self.assertIsNone(p.mapped_satisfaction(0.5))

    def test_legacy_single_anchor(self):
        p = _Dummy()
        p.pref_anchor, p.naive_anchor = 2.0, None  # naive absent → legacy clip(R/pref)
        self.assertAlmostEqual(p.satisfaction(1.0), 50.0, places=3)
        self.assertAlmostEqual(p.satisfaction(2.0), 100.0, places=3)

    def test_saturation_fallback(self):
        p = _Dummy()  # both anchors None, saturation=1.0
        self.assertAlmostEqual(p.satisfaction(0.5), 50.0, places=3)

    def test_mapped_buckets(self):
        p = _Dummy()
        p.pref_anchor, p.naive_anchor = 1.0, 0.0
        self.assertEqual(p.mapped_satisfaction(1.0), 1.00)  # sat 100 → 1.0
        self.assertEqual(p.mapped_satisfaction(0.0), 0.00)  # sat 0   → 0.0
        self.assertEqual(p.mapped_satisfaction(0.5), 0.25)  # sat 50  → ≥40 → 0.25

    def test_speedrange_uses_base_via_final_raw(self):
        # Override removed → base dual-anchor formula over final_raw (step count).
        p = SpeedRangePreference(direction="less")
        p.pref_anchor, p.naive_anchor = 100.0, 400.0  # pref = fewer steps
        p._n_steps = 250
        self.assertEqual(p.final_raw(0.0), 250.0)  # final_raw = step count, ignores arg
        self.assertAlmostEqual(p.satisfaction(p.final_raw(0.0)), 50.0, places=3)
        p._n_steps = 100
        self.assertAlmostEqual(p.satisfaction(p.final_raw(0.0)), 100.0, places=3)
        p._n_steps = 400
        self.assertAlmostEqual(p.satisfaction(p.final_raw(0.0)), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()

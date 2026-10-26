"""Unit tests for the dual-anchor calibration helpers (LT2) and the
discrimination stat (LT4) — pure functions, no sim required.

Run:
    cd /data3/zhixu/LIBERO-PeRM/LIBERO
    python scripts/test_low_pref_calibration.py
"""

from __future__ import annotations

import os
import sys
import unittest

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from calibrate_low_pref import _build_calibration, cohen_d  # noqa: E402


class TestBuildCalibration(unittest.TestCase):
    def test_pref_and_naive(self):
        cal = _build_calibration({"speed_range": (120.0, 10)}, {"speed_range": (300.0, 8)})
        self.assertEqual(cal["speed_range"]["pref_anchor"], 120.0)
        self.assertEqual(cal["speed_range"]["naive_anchor"], 300.0)
        self.assertEqual(cal["speed_range"]["n_pref"], 10)
        self.assertEqual(cal["speed_range"]["n_naive"], 8)

    def test_pref_only(self):
        cal = _build_calibration({"approach_to_object": (0.7, 5)}, {})
        self.assertEqual(cal["approach_to_object"]["pref_anchor"], 0.7)
        self.assertNotIn("naive_anchor", cal["approach_to_object"])

    def test_pref_drives_output(self):
        # pref_medians drives output; a naive-only pref_type is ignored.
        cal = _build_calibration({"a": (1.0, 3)}, {"b": (2.0, 3)})
        self.assertIn("a", cal)
        self.assertNotIn("b", cal)
        self.assertNotIn("naive_anchor", cal["a"])


class TestCohenD(unittest.TestCase):
    def test_separated(self):
        d = cohen_d([10, 11, 9, 10, 10], [1, 2, 0, 1, 1])
        self.assertIsNotNone(d)
        self.assertGreater(d, 2.0)

    def test_small_n_none(self):
        self.assertIsNone(cohen_d([1.0], [2.0]))

    def test_zero_variance_none(self):
        self.assertIsNone(cohen_d([1, 1, 1], [1, 1, 1]))


if __name__ == "__main__":
    unittest.main()

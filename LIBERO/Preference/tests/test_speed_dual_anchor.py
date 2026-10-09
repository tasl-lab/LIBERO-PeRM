"""SpeedRangePreference: cumulative per-subtask dual anchor (pref median -> 100, no-pref median -> 0)."""
import pytest

from Preference.low_level.speed import SpeedRangePreference


def make(pref, naive, durations, max_speed=2.0):
    p = SpeedRangePreference(min_speed=max_speed, max_speed=max_speed)
    p._subtasks = [("pick", "a"), ("put", "b")] * (len(pref) // 2)
    p.subtask_pref_anchor, p.subtask_naive_anchor = pref, naive
    t = 0
    for d in durations:                       # subtask i finished after d more steps
        t += d
        p._done_steps.append(t)
    p._cur = len(durations)
    return p


P, N = [40.0, 70.0], [43.0, 96.0]            # the pick barely changes with speed, the carry does


def test_pref_pace_scores_100_and_nopref_pace_scores_0():
    assert make(P, N, [40, 70]).satisfaction(110.0) == pytest.approx(100.0)
    assert make(P, N, [43, 96]).satisfaction(139.0) == pytest.approx(0.0)


def test_halfway_between_the_anchors():
    assert make(P, N, [41.5, 83]).satisfaction(124.5) == pytest.approx(50.0)


def test_clipped_outside_the_anchors():
    assert make(P, N, [30, 50]).satisfaction(80.0) == pytest.approx(100.0)
    assert make(P, N, [60, 120]).satisfaction(180.0) == pytest.approx(0.0)


def test_slow_down_uses_the_sign_of_the_anchor_gap():
    slow_p, slow_n = [45.0, 140.0], [43.0, 96.0]
    assert make(slow_p, slow_n, [45, 140], max_speed=0.5).satisfaction(185.0) == pytest.approx(100.0)
    assert make(slow_p, slow_n, [43, 96], max_speed=0.5).satisfaction(139.0) == pytest.approx(0.0)


def test_no_point_until_the_anchors_separate():
    p = make(P, N, [41])                      # only the pick is done: 3 steps of gap on 40 is noise
    assert p._dual_score(1) is None
    assert p.subtask_points() is None
    pts = make(P, N, [40, 70]).subtask_points()
    assert [t for t, _ in pts] == [110] and pts[0][1] == pytest.approx(100.0)


def test_without_nopref_anchor_falls_back_to_the_pref_ratio():
    p = make(P, None, [40, 70])
    assert p.satisfaction(110.0) == pytest.approx(100.0)
    q = make(P, None, [80, 140])              # twice as slow as the preference demos
    assert q.satisfaction(220.0) == pytest.approx(50.0)

"""Unit tests for runtime transport-window detection (synthetic trajectories)."""
import numpy as np
import pytest

from Preference.low_level.transport_window import (
    carry_span,
    detect_transport_window,
    lateral_move_mask,
    transport_mask,
)


def _traj(n=220, grasp=50, lift_start=60, lift_end=100, descent_start=150,
          release=180, z_low=0.02, z_carry=0.30, z_drop=0.12):
    """Approach → grasp-hold → lift → plateau → descent → hover → release."""
    z = np.full(n, z_low)
    z[lift_start:lift_end] = np.linspace(z_low, z_carry, lift_end - lift_start)
    z[lift_end:descent_start] = z_carry
    z[descent_start:release - 10] = np.linspace(
        z_carry, z_drop, release - 10 - descent_start)
    z[release - 10:] = z_drop
    pos = np.zeros((n, 3))
    pos[:, 2] = z
    # lateral translation: approach move [10, 40), carry transport [100, 150)
    x = np.zeros(n)
    x[10:40] = np.linspace(0.0, 0.30, 30)
    x[40:100] = 0.30
    x[100:150] = np.linspace(0.30, 0.60, 50)
    x[150:] = 0.60
    pos[:, 0] = x
    grip = np.zeros((n, 2))
    grip[:, 0], grip[:, 1] = 0.04, -0.04            # open
    grip[grasp:release + 1, 0] = 0.001              # closed
    grip[grasp:release + 1, 1] = -0.001
    return pos, grip


def test_carry_span():
    pos, grip = _traj()
    assert carry_span(grip) == (50, 180)


def test_transport_excludes_lift_and_put():
    pos, grip = _traj()
    win = detect_transport_window(pos, grip)
    assert win is not None
    start, end = win
    # starts after the lift tops out, ends at the placing-descent onset
    assert 95 <= start <= 110
    assert 140 <= end <= 155
    # strictly inside the gripper-closed span (the old carry proxy)
    assert start > 50 and end < 180


def test_no_gripper_actuation_returns_none():
    pos, grip = _traj()
    grip[:] = 0.04
    grip[:, 1] *= -1
    assert detect_transport_window(pos, grip) is None
    assert not transport_mask(pos, grip).any()


def test_truncated_mid_carry_keeps_carry_end():
    # cut the episode during the plateau: no placing descent to detect
    pos, grip = _traj()
    cut = 140
    pos, grip = pos[:cut], grip[:cut]
    win = detect_transport_window(pos, grip)
    assert win is not None
    assert win[1] == carry_span(grip)[1]


def test_no_lift_starts_at_grasp():
    # hug-ground extreme: carried at grasp height the whole way
    pos, grip = _traj()
    pos[:, 2] = 0.02
    win = detect_transport_window(pos, grip)
    assert win is not None
    assert win[0] == carry_span(grip)[0]


def test_brief_pause_inside_descent_is_bridged():
    pos, grip = _traj()
    # 3-step hover in the middle of the descent (< SUSTAIN=5) must not split it
    pos[158:161, 2] = pos[158, 2]
    win = detect_transport_window(pos, grip)
    assert win is not None
    assert 140 <= win[1] <= 155


def test_mask_matches_window():
    pos, grip = _traj()
    win = detect_transport_window(pos, grip)
    mask = transport_mask(pos, grip)
    idx = np.flatnonzero(mask)
    assert idx[0] == win[0] and idx[-1] == win[1]


def test_lateral_mask_covers_approach_and_carry_moves():
    pos, _ = _traj()
    mask = lateral_move_mask(pos)
    # both translation segments in, everything vertical/static out
    assert mask[15:38].all() and mask[105:148].all()
    assert not mask[:9].any()            # pre-move idle
    assert not mask[45:95].any()         # grasp-hold + vertical lift
    assert not mask[185:].any()          # post-release hover


def test_lateral_mask_drops_noise_blips():
    pos, _ = _traj()
    pos[60:62, 1] = [0.02, 0.04]         # 2-step lateral jitter inside the lift
    mask = lateral_move_mask(pos)
    assert not mask[58:66].any()

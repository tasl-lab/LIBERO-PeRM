"""Runtime transport-window detection from simulator observables.

Two scopes are provided, both from observables only:

- `detect_transport_window` / `transport_mask` — the CARRY segment: after the
  object has been grasped AND lifted to travel height, before the final
  placing descent begins. For preferences that only make sense while holding
  the object (e.g. avoid-with-object gating).
- `lateral_move_mask` — ALL sustained lateral-translation segments, approach
  included. For preferences whose language scopes to "when moving" (e.g.
  height_offset "Move higher" / "Close to ground when moving": gen applies
  `path_height_offset` to every move_to, so the approach move carries signal
  too and must be scored).

The detector uses ONLY quantities observable during a live policy rollout
(EE position + gripper finger states) — no gen-time subtask indices, which do
not exist when evaluating a policy. It is therefore shared by:
  - review/compare video rendering (curve scoping), and
  - eval-side metric scoping for policy rollouts.

Boundary rules (on the sustained gripper-closed span [c0, c1]):
  lift_end      = first step where smoothed EE z stops rising for
                  `sustain` consecutive steps (fallback: argmax z)
  descent_start = first step of the final falling run into the release,
                  found by walking backward from c1 through falling steps
                  (pauses shorter than `sustain` are tolerated; a flat run of
                  `sustain`+ steps is the carry plateau and stops the walk).
                  Ignored if the total drop is < `min_drop` (no placing
                  descent yet, e.g. episode truncated mid-carry)
"""
from __future__ import annotations

import numpy as np

SMOOTH_K = 5        # moving-average window (steps) for EE z
RISE_EPS = 1e-3     # m/step; |dz| below this counts as "not rising/falling"
SUSTAIN = 5         # consecutive flat/falling steps that end the lift
MIN_DROP = 0.03     # m; final descent must drop at least this to count as put
MIN_LIFT = 0.02     # m; rise since grasp required before a flat run can end the lift
LATERAL_THRESH = 0.003  # m/step; |dxy| above this counts as lateral translation
MIN_RUN = 5         # steps; shorter lateral runs are noise and dropped


def _smooth(z: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    if len(z) < 2 or k <= 1:
        return z.astype(float)
    pad = k // 2
    zp = np.pad(z.astype(float), pad, mode="edge")
    return np.convolve(zp, np.ones(k) / k, mode="same")[pad:pad + len(z)]


def carry_span(gripper_states: np.ndarray) -> tuple[int, int] | None:
    """Sustained gripper-closed span [first, last] from finger separation."""
    g = np.asarray(gripper_states)
    w = np.abs(g[:, 0] - g[:, 1])
    if w.max() - w.min() < 1e-6:      # gripper never actuated
        return None
    closed = w < (w.min() + w.max()) / 2
    idx = np.flatnonzero(closed)
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1])


def detect_transport_window(
    ee_pos: np.ndarray,
    gripper_states: np.ndarray,
    *,
    min_drop: float = MIN_DROP,
    rise_eps: float = RISE_EPS,
    sustain: int = SUSTAIN,
) -> tuple[int, int] | None:
    """Return [start, end] (inclusive) transport step range, or None.

    None means "no carry detected". If a carry exists but lift/descent
    boundaries cannot be resolved, the corresponding side falls back to the
    carry-span edge, so the result is always within (and at worst equal to)
    the gripper-closed span.
    """
    span = carry_span(gripper_states)
    if span is None:
        return None
    c0, c1 = span
    if c1 - c0 < 2 * sustain:
        return span
    z = _smooth(np.asarray(ee_pos)[:, 2])
    dz = np.diff(z)

    # lift_end: first index in [c0, c1) where z stops rising for `sustain`
    # steps AFTER an actual lift (>= MIN_LIFT above grasp height). The gate
    # keeps the flat gripper-closing steps right at the grasp from ending the
    # "lift" before it started; if the demo never lifts (carry at grasp
    # height), transport legitimately starts at c0.
    lift_end = c0
    flat = 0
    for t in range(c0, c1):
        if dz[t] < rise_eps:
            flat += 1
            if flat >= sustain and z[t] - z[c0] >= MIN_LIFT:
                lift_end = t - sustain + 1
                break
        else:
            flat = 0
    else:
        if z[c0:c1 + 1].max() - z[c0] >= MIN_LIFT:
            lift_end = c0 + int(np.argmax(z[c0:c1 + 1]))

    # descent_start: walk backward from c1 through the final falling run.
    # Falling steps extend the run; non-falling pauses shorter than `sustain`
    # are tolerated (brief hovers during the put); a flat stretch of
    # `sustain`+ steps is the carry plateau and ends the walk.
    falling = dz < -rise_eps
    t = c1 - 1
    while t >= lift_end and not falling[t]:
        t -= 1          # skip the settle/hover at the release point (any length)
    descent_start = c1
    flat = 0
    while t >= lift_end:
        if falling[t]:
            descent_start = t
            flat = 0
        else:
            flat += 1
            if flat >= sustain:
                break
        t -= 1
    if (z[descent_start] - z[c1]) < min_drop:
        descent_start = c1

    if descent_start <= lift_end:
        return span
    return int(lift_end), int(descent_start)


def transport_mask(ee_pos: np.ndarray, gripper_states: np.ndarray, **kw) -> np.ndarray:
    """Boolean per-step mask of the transport window (all-False if none)."""
    n = len(ee_pos)
    mask = np.zeros(n, dtype=bool)
    win = detect_transport_window(ee_pos, gripper_states, **kw)
    if win is not None:
        mask[win[0]:win[1] + 1] = True
    return mask


def lateral_move_mask(
    ee_pos: np.ndarray,
    *,
    thresh: float = LATERAL_THRESH,
    min_run: int = MIN_RUN,
    bridge: int = SUSTAIN,
) -> np.ndarray:
    """Boolean per-step mask of every sustained lateral-move segment.

    Covers approach AND carry translations; excludes vertical lifts/descents
    and in-place manipulation (their |dxy| stays under `thresh`). Gaps shorter
    than `bridge` steps are bridged; surviving runs shorter than `min_run`
    steps are dropped as noise.
    """
    p = np.asarray(ee_pos)[:, :2]
    dxy = np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0), axis=1)])
    mask = dxy > thresh
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    for a, b in zip(idx[:-1], idx[1:]):        # bridge short gaps
        if 1 < b - a <= bridge:
            mask[a:b] = True
    # drop short runs
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    for s, e in zip(starts, ends):
        if e - s < min_run:
            mask[s:e] = False
    return mask

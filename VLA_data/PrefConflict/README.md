# VLA_data/PrefConflict/

LIBERO-PrefConflict suite training data placeholder.

## Status

**Empty by design** — PrefConflict is 0-shot first; no demos required for O₁ / O₂.

## When to populate

Only if running R₁ (pref-pref LoRA, Q5a training variant) or R₂ (task-pref cross-task LoRA, Q6d). Both train **single-pref** demos, never conflict-joint demos (task-first stance).

## Source mapping

For R₁: each pp pair needs `(layout, P1)` and `(layout, P2)` single-pref demo. Reuse the corresponding `Mission_<id>.hdf5` from `VLA_data/SinglePref/` (or generate same-layout P2 demos in worst case — ~6 new missions).

For R₂: train `(A, ∅)` + `(B, ∅)` + `(B, P_B)` disjoint demos. All resolvable from `VLA_data/SinglePref/` + Series 3 `no_pref_layout/` HDF5.

See: `LIBERO/dataset/series_6/task_suite.yaml`, `LIBERO/dataset/series_6/conflict_pair_reference.md`.

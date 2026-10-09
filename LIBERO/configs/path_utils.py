"""Single source of truth for dataset path resolution.

Layout (post-2026-04-28 migration):
    LIBERO/dataset/series_<X>/perm_<X>_<Y>/libero_<id>/...

Use `mission_dir(perm_name, libero_name)` everywhere instead of building
`os.path.join("dataset", perm_name, libero_name)` by hand.
"""
from __future__ import annotations

import os
import re

_PERM_RE = re.compile(r"^perm_(\d+)_\d+$")


def series_of(perm_name: str) -> str:
    """e.g. 'perm_3_15' -> 'series_3'."""
    m = _PERM_RE.match(perm_name)
    if not m:
        raise ValueError(f"bad perm_name: {perm_name!r}")
    return f"series_{m.group(1)}"


def perm_subdir(perm_name: str) -> str:
    """e.g. 'perm_3_15' -> 'series_3/perm_3_15'."""
    return os.path.join(series_of(perm_name), perm_name)

pbench_subdir = perm_subdir


def mission_dir(perm_name: str, libero_name: str,
                dataset_root: str = "dataset") -> str:
    """Full mission dir under a given dataset_root."""
    return os.path.join(dataset_root, series_of(perm_name), perm_name, libero_name)

"""Loader for configs/task_suites.yaml — the task suite → mission_id registry."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import yaml

SUITES_YAML = os.path.join(os.path.dirname(__file__), "task_suites.yaml")


@lru_cache(maxsize=1)
def _load_all() -> Dict[str, dict]:
    with open(SUITES_YAML, "r") as f:
        return yaml.safe_load(f) or {}


def list_suites() -> List[str]:
    return list(_load_all().keys())


def load_suite(name: str) -> List[str]:
    """Return the list of mission_ids registered under `name`."""
    data = _load_all()
    if name not in data:
        raise KeyError(f"Unknown task_suite '{name}'. Available: {list(data)}")
    missions = data[name].get("missions") or []
    return list(missions)


def mission_to_suite(mission_id: str) -> str:
    """Reverse lookup. Raises if the mission is in 0 or >1 suites."""
    hits = [s for s, body in _load_all().items() if mission_id in (body.get("missions") or [])]
    if not hits:
        raise KeyError(f"mission_id '{mission_id}' not registered in any suite")
    if len(hits) > 1:
        raise ValueError(f"mission_id '{mission_id}' appears in multiple suites: {hits}")
    return hits[0]


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        for s in list_suites():
            ms = load_suite(s)
            print(f"{s}: {len(ms)} missions")
    else:
        for s in sys.argv[1:]:
            print(f"=== {s} ===")
            for m in load_suite(s):
                print(m)

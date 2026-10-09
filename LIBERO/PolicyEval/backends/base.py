"""Inference backend contract.

All backends expose the same schema:

    infer(element: dict) -> dict with key "actions" -> np.ndarray [N, 7]

where element has:
    observation/image        uint8 [H, W, 3]   agentview, 224x224
    observation/wrist_image  uint8 [H, W, 3]   eye-in-hand, 224x224
    observation/state        float32 [8]       eef_pos(3) + axisangle(3) + gripper(2)
    prompt                   str               already condition-composed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class InferenceBackend(ABC):
    @abstractmethod
    def infer(self, element: Dict) -> Dict:
        """Return a dict with key 'actions' -> np.ndarray [N, 7]."""

    def close(self) -> None:
        """Override if the backend holds a persistent connection."""

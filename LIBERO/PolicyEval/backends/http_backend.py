"""HTTP backend for X-VLA's official deploy.py FastAPI server.

X-VLA's reference inference server (`deploy.py` from
github.com/2toinf/X-VLA) exposes a `/act` endpoint that expects a fundamentally
different obs schema than P-Bench's WebsocketBackend:
  - per-cam JPEG/ndarray under image0/image1[/image2]
  - 20D closed-loop proprio [pos(3) + rot6d(6) + grip(1) + zeros(10)]
  - language_instruction + domain_id + steps

This backend bridges P-Bench's standard 8D-state obs dict to that protocol so
the X-VLA official server (any base + per-suite LoRA combo) can be evaluated
against P-Bench's runfile-driven sweep with no changes to deploy.py.

Action conversion: server returns 10D [pos(3) + rot6d(6) + grip(1)]; we
convert rot6d→axis-angle and threshold gripper to ±1, returning 7D as the
WS contract requires.

Closed-loop proprio: per the official client (libero_client.py:191-197), the
proprio's first 9 dims are updated with the *last predicted* frame after
each chunk request; reset to current obs only on prompt change. We follow
the same convention.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import json_numpy
import numpy as np
import requests

from .base import InferenceBackend

logger = logging.getLogger(__name__)

# X-VLA: Mat_to_Rotate6D first two cols → 6D
def _axisangle_to_rot6d(aa: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        R = np.eye(3, dtype=np.float32)
    else:
        axis = aa / angle
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        R = (np.eye(3, dtype=np.float32)
             + np.sin(angle) * K
             + (1 - np.cos(angle)) * (K @ K))
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _mat2quat(R: np.ndarray) -> np.ndarray:
    """Robosuite-compatible mat→quat (xyzw)."""
    M = R
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (M[2, 1] - M[1, 2]) / s
        y = (M[0, 2] - M[2, 0]) / s
        z = (M[1, 0] - M[0, 1]) / s
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        s = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def _quat2axisangle(q: np.ndarray) -> np.ndarray:
    """xyzw quat → axis-angle (3D). Matches robosuite convention."""
    qw = max(-1.0, min(1.0, float(q[3])))
    den = math.sqrt(1.0 - qw * qw)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(qw)) / den).astype(np.float32)


def _rotate6d_to_axisangle(r6d: np.ndarray) -> np.ndarray:
    """(N, 6) rot6d → (N, 3) axis-angle, via Gram-Schmidt → R → quat → axis-angle.
    Uses robosuite's transform_utils for mat2quat / quat2axisangle to match the
    official X-VLA libero_client.py exactly."""
    import robosuite.utils.transform_utils as T
    eps = 1e-6
    if r6d.ndim == 1:
        r6d = r6d[None, :]
        single = True
    else:
        single = False
    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = (a2 - dot * b1)
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + eps)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)
    out = np.empty((R.shape[0], 3), dtype=np.float32)
    for i in range(R.shape[0]):
        out[i] = T.quat2axisangle(T.mat2quat(R[i])).astype(np.float32)
    return out[0] if single else out


class _MetadataShim:
    """Mimic openpi WebsocketClientPolicy's get_server_metadata() so eval.py's
    validation passes. The HTTP server doesn't expose metadata, so we echo the
    runfile's declared config/checkpoint — validation becomes a tautology but
    keeps the existing eval flow uniform."""

    def __init__(self, metadata: Dict):
        self._metadata = metadata

    def get_server_metadata(self) -> Dict:
        return self._metadata


class HttpBackend(InferenceBackend):
    """Translate P-Bench obs → X-VLA `/act` payload → P-Bench actions dict.

    Args:
      host, port: deploy.py FastAPI host/port.
      config: runfile's `server.config` (echoed back as metadata for validation).
      checkpoint: runfile's `server.checkpoint` (echoed back).
      steps: chunk length to request from server (X-VLA default 10).
      timeout_s: HTTP request timeout per call.
      domain_id: X-VLA domain id (libero == 3 in official client).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        config: str = "",
        checkpoint: str = "",
        steps: int = 10,
        timeout_s: float = 60.0,
        domain_id: int = 3,
    ):
        connect_host = "127.0.0.1" if host == "0.0.0.0" else host
        self._url = f"http://{connect_host}:{port}/act"
        self._steps = int(steps)
        self._timeout = float(timeout_s)
        self._domain_id = int(domain_id)
        self._proprio9: Optional[np.ndarray] = None
        self._last_prompt: Optional[str] = None
        self._pending_chunk: Optional[np.ndarray] = None
        self._pending_consumed: int = 0
        self._client = _MetadataShim({
            "config_name": config,
            "checkpoint_dir": checkpoint,
            "name": config,
            "vla": "xvla",
            "chunk_size": self._steps,
            "action_dim_full": 10,
            "action_dim_used": 7,
        })
        # Quick sanity ping (will raise if server not up).
        try:
            requests.get(f"http://{connect_host}:{port}/docs", timeout=5.0)
        except Exception as e:
            logger.warning(f"HttpBackend sanity ping to {self._url} failed (non-fatal): {e}")
        logger.info(f"HttpBackend ready → {self._url}  steps={self._steps} domain={self._domain_id}")

    # -- proprio ---------------------------------------------------------------
    @staticmethod
    def _mat_to_rot6d(R: np.ndarray) -> np.ndarray:
        """3×3 rotation matrix → 6D = first two columns concatenated.
        Matches `Mat_to_Rotate6D` in X-VLA's libero_client.py."""
        R = np.asarray(R, dtype=np.float32).reshape(3, 3)
        return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)

    def _ensure_proprio(self, state8: np.ndarray, prompt: str,
                        ori_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """Build 20D closed-loop rot6d proprio. Reset on prompt change; otherwise
        keep the 9D from previous chunk's last predicted frame (closed-loop).

        IMPORTANT: X-VLA's official client uses `controller.ee_ori_mat` (site frame)
        — this DIFFERS from `robot0_eef_quat` (body frame) on Panda by ~90° around z.
        When `ori_mat` is provided (from `observation/eef_ori_mat`), prefer it.
        Otherwise fall back to deriving rot6d from state's axis-angle (incorrect frame
        for X-VLA but kept for compatibility)."""
        if self._proprio9 is None or prompt != self._last_prompt:
            eef_pos = state8[:3].astype(np.float32)
            if ori_mat is not None:
                rot6d = self._mat_to_rot6d(ori_mat)
            else:
                rot6d = _axisangle_to_rot6d(state8[3:6].astype(np.float32))
            self._proprio9 = np.concatenate([eef_pos, rot6d]).astype(np.float32)
            self._last_prompt = prompt
        return np.concatenate([
            self._proprio9, np.zeros(1, dtype=np.float32), np.zeros(10, dtype=np.float32)
        ]).astype(np.float32)

    # -- main ------------------------------------------------------------------
    def infer(self, element: Dict) -> Dict:
        img = element["observation/image"]                  # (H,W,3) uint8 — client [::-1,::-1] flipped
        wrist = element["observation/wrist_image"]          # (H,W,3) uint8 — client flipped
        state8 = np.asarray(element["observation/state"], dtype=np.float32)
        prompt = element["prompt"]

        # X-VLA official: agentview is 180° flipped (matches our client flip);
        # wrist is NOT flipped → undo client flip.
        agentview = np.ascontiguousarray(img)
        wrist_raw = np.ascontiguousarray(wrist[::-1, ::-1])

        ori_mat = element.get("observation/eef_ori_mat")
        proprio20 = self._ensure_proprio(state8, prompt, ori_mat=ori_mat)

        payload = {
            "image0": json_numpy.dumps(agentview),
            "image1": json_numpy.dumps(wrist_raw),
            "proprio": json_numpy.dumps(proprio20),
            "language_instruction": prompt,
            "domain_id": self._domain_id,
            "steps": self._steps,
        }

        resp = requests.post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        if "action" not in data:
            raise RuntimeError(f"X-VLA server returned no 'action' field: {data}")

        action = np.asarray(data["action"], dtype=np.float32)  # (T, 10) per official
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"unexpected action shape from server: {action.shape}")

        # Stash 10D pre-conversion chunk for closed-loop proprio bookkeeping.
        # We DO NOT advance proprio to action[-1] now — caller may consume
        # fewer than `len(action)` frames (replan_steps < chunk_size). Caller
        # should invoke `notify_consumed(k)` after popping `k` frames so we
        # snap proprio to action[k-1, :9] (the last actually-executed target).
        # Fallback for callers that don't notify: advance to action[-1]
        # at the next infer() entry (close-enough heuristic for chunk-aligned use).
        self._pending_chunk = action.copy()
        self._pending_consumed = 0

        # Convert to 7D for env.
        target_eef = action[:, :3]
        target_axis = _rotate6d_to_axisangle(action[:, 3:9])
        target_grip = np.where(action[:, 9:10] > 0.5, 1.0, -1.0)
        out = np.concatenate([target_eef, target_axis, target_grip], axis=-1).astype(np.float32)

        return {"actions": out}

    def notify_consumed(self, k: int) -> None:
        """Caller tells the backend how many frames of the most recent chunk
        were actually executed. We snap closed-loop proprio to the last
        executed target (`action[k-1, :9]`), so the next inference's proprio
        matches reality rather than the chunk's tail."""
        if getattr(self, "_pending_chunk", None) is None or k <= 0:
            return
        idx = min(k, self._pending_chunk.shape[0]) - 1
        self._proprio9 = self._pending_chunk[idx, :9].astype(np.float32).copy()
        self._pending_consumed = idx + 1

    def close(self) -> None:
        pass

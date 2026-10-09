"""Uniform policy server — GROOT N1.6.

Wraps Gr00tPolicy (from Isaac-GR00T repo) in a WebSocket server that speaks
the same protocol as PolicyEval/backends/websocket_backend.py. This lets
P_Bench's eval.py evaluate GROOT without any client-side changes.

Usage:
    python serve.py <runfile.yaml>

Preferred entry: bash /data3/zhixu/LIBERO-PeRM/VLA_infer/server_groot.sh <runfile.yaml>
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _ws_server import WebsocketPolicyServer

# Add groot_repo to path so we can import gr00t. Use env var so the launcher
# can pick the right version (groot_repo for N1.7, groot_repo_n1d6 for N1.6).
# Falls back to N1.7 repo for backward compat.
_GROOT_REPO = Path(os.environ.get("GROOT_REPO", "/data3/zhixu/LIBERO-PeRM/VLAs/groot_repo"))
sys.path.insert(0, str(_GROOT_REPO))

VLA_NAME = "groot"

logger = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)


# ─────────────────────────────────────────────────────────────────────────────
# Profile loading
# ─────────────────────────────────────────────────────────────────────────────


def load_profile(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    s = cfg.get("server") or {}
    for k in ("vla", "name", "config", "checkpoint", "port"):
        if not s.get(k):
            raise ValueError(f"runfile server: missing {k!r}")
    if s["vla"] != VLA_NAME:
        raise ValueError(f"{VLA_NAME}/serve.py invoked for vla={s['vla']!r}")
    return s


def print_banner(s: dict) -> None:
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "?"
    bar = "─" * 62
    print(bar)
    print(f"  {VLA_NAME} policy server — GR00T N1.6")
    print(bar)
    print(f"  name          : {s['name']}")
    print(f"  config        : {s['config']}")
    print(f"  checkpoint    : {s['checkpoint']}")
    print(f"  host / port   : {hostname} ({ip}) / :{s['port']}")
    print(f"  default_prompt: {s.get('default_prompt')!r}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# Policy loading
# ─────────────────────────────────────────────────────────────────────────────


def load_policy(s: dict):
    """Load Gr00tPolicy and return a wrapper with `.infer(obs) -> {'actions': ...}`.

    Expected obs keys from PolicyEval/eval.py:_obs_element:
        - "observation/image"        : np.ndarray [H, W, 3] uint8, 224x224
        - "observation/wrist_image"  : np.ndarray [H, W, 3] uint8, 224x224
        - "observation/state"        : np.ndarray [8]  (EEF pos 3 + axis-angle 3 + gripper 2)
        - "prompt"                   : str

    Returned: {"actions": np.ndarray (action_horizon, 7)}
    """
    from gr00t.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    # The Gr00tN1.6 ckpt's Eagle vision-language backbone REQUIRES
    # flash_attention_2 (asserts in modeling_eagle3_vl.py). The ckpt's
    # config.json has attn_implementation=null, so AutoModel.from_pretrained()
    # without override hits the assertion. Patch AutoModel.from_pretrained to
    # inject the kwarg before Gr00tPolicy calls it. Idempotent: only wraps once.
    import transformers
    if not getattr(transformers.AutoModel.from_pretrained, "_perm_attn_patch", False):
        _orig_from_pretrained = transformers.AutoModel.from_pretrained

        def _patched_from_pretrained(model_id, *args, **kwargs):
            kwargs.setdefault("attn_implementation", "flash_attention_2")
            return _orig_from_pretrained(model_id, *args, **kwargs)

        _patched_from_pretrained._perm_attn_patch = True
        transformers.AutoModel.from_pretrained = _patched_from_pretrained

    device = s.get("device", "cuda:0")
    embodiment = s.get("embodiment_tag", "LIBERO_PANDA")
    tag = EmbodimentTag(embodiment.lower()) if embodiment.lower() != "libero_panda" else EmbodimentTag.LIBERO_PANDA

    logger.info(f"Loading Gr00tPolicy: model_path={s['checkpoint']}, embodiment={tag}, device={device}")
    policy = Gr00tPolicy(
        model_path=s["checkpoint"],
        embodiment_tag=tag,
        device=device,
    )

    # Query action horizon from modality config
    modality_cfg = policy.get_modality_config()
    action_horizon = len(modality_cfg["action"].delta_indices)
    logger.info(f"Action horizon: {action_horizon}")

    return GrootInferWrapper(policy, action_horizon)


class GrootInferWrapper:
    """Translates P_Bench obs format → GROOT obs format → action array.

    P_Bench eval.py sends:
        observation/image:       (224, 224, 3) uint8
        observation/wrist_image: (224, 224, 3) uint8
        observation/state:       (8,) float32  [pos3 + axisangle3 + gripper2]
        prompt:                  str

    GROOT Gr00tPolicy expects:
        video: {image: (B, T, H, W, 3) uint8, wrist_image: (B, T, H, W, 3) uint8}
        state: {x: (B,T,1), y: ..., z: ..., roll: ..., pitch: ..., yaw: ..., gripper: (B,T,2)}
        language: {task: [[str]]}

    GROOT returns:
        {x: (B,T,1), y: ..., gripper: (B,T,1)}

    We convert back to:
        actions: (action_horizon, 7)
    """

    def __init__(self, policy, action_horizon: int):
        self.policy = policy
        self.action_horizon = action_horizon

    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()

        # ── Unpack P_Bench obs ──
        img = obs["observation/image"]              # (H, W, 3) uint8
        wrist = obs["observation/wrist_image"]      # (H, W, 3) uint8
        state = obs["observation/state"]            # (8,) float32
        prompt = obs["prompt"]                      # str

        # Image orientation: with robosuite IMAGE_CONVENTION="opengl",
        # eval.py's [::-1, ::-1] produces the same orientation as training data.
        # No additional flip needed.

        # ── Build GROOT observation dict ──
        # Add batch (B=1) and time (T=1) dimensions
        groot_obs = {
            "video": {
                "image": img[np.newaxis, np.newaxis, ...].astype(np.uint8),           # (1,1,256,256,3)
                "wrist_image": wrist[np.newaxis, np.newaxis, ...].astype(np.uint8),
            },
            "state": {
                "x": np.array([[[state[0]]]], dtype=np.float32),       # (1,1,1)
                "y": np.array([[[state[1]]]], dtype=np.float32),
                "z": np.array([[[state[2]]]], dtype=np.float32),
                "roll": np.array([[[state[3]]]], dtype=np.float32),
                "pitch": np.array([[[state[4]]]], dtype=np.float32),
                "yaw": np.array([[[state[5]]]], dtype=np.float32),
                "gripper": state[6:8][np.newaxis, np.newaxis, :].astype(np.float32),  # (1,1,2)
            },
            "language": {
                "annotation.human.action.task_description": [[prompt]],
            },
        }

        # ── Run inference ──
        action_dict, info = self.policy.get_action(groot_obs)

        # ── Reassemble action chunk: (action_horizon, 7) ──
        # action_dict keys: x(1,T,1), y(1,T,1), ..., gripper(1,T,1)
        action_chunk = np.concatenate([
            action_dict["x"][0],        # (T, 1)
            action_dict["y"][0],
            action_dict["z"][0],
            action_dict["roll"][0],
            action_dict["pitch"][0],
            action_dict["yaw"][0],
            action_dict["gripper"][0],  # (T, 1)
        ], axis=-1)  # (T, 7)

        # DEBUG: log raw action before gripper conversion
        if not hasattr(self, '_raw_logged'):
            self._raw_logged = True
            logger.info(f"[DEBUG RAW] BEFORE gripper convert: gripper_raw={action_chunk[:3, 6]}, "
                        f"pos_sample={action_chunk[0,:3]}, full_first={action_chunk[0]}")

        # ── Gripper action conversion ──
        # Pretrained (IPEC) data: gripper ∈ [0,1] → need normalize+invert for LIBERO env
        # Finetuned on P_Bench data: gripper ∈ [-1,+1] already matches LIBERO env → skip
        # Controlled by env var GROOT_GRIPPER_CONVERT (default "1" for backward compat)
        import os as _os
        if _os.environ.get("GROOT_GRIPPER_CONVERT", "1") == "1":
            gripper = action_chunk[:, 6:7]
            gripper = 2.0 * gripper - 1.0   # [0,1] → [-1,+1]
            gripper = -gripper               # invert direction
            action_chunk[:, 6:7] = gripper

        # DEBUG: log action stats for first call
        if not hasattr(self, '_logged'):
            self._logged = True
            logger.info(f"[DEBUG] action_chunk shape={action_chunk.shape}, "
                        f"pos_range=[{action_chunk[:,:3].min():.4f},{action_chunk[:,:3].max():.4f}], "
                        f"rot_range=[{action_chunk[:,3:6].min():.4f},{action_chunk[:,3:6].max():.4f}], "
                        f"gripper_range=[{action_chunk[:,6].min():.4f},{action_chunk[:,6].max():.4f}]")

        dt = time.monotonic() - t0
        return {"actions": action_chunk, "policy_timing": {"total_s": dt}}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(runfile: str) -> None:
    s = load_profile(runfile)
    print_banner(s)

    logger.info(f"Loading checkpoint: config={s['config']!r}  dir={s['checkpoint']!r}")
    t0 = time.monotonic()
    policy = load_policy(s)   # ← raises NotImplementedError until filled in
    logger.info(f"Checkpoint loaded in {time.monotonic() - t0:.1f}s")

    metadata = dict(getattr(policy, "metadata", None) or {})
    metadata.setdefault("config_name", s["config"])
    metadata.setdefault("checkpoint_dir", s["checkpoint"])
    metadata.setdefault("name", s["name"])
    metadata.setdefault("vla", VLA_NAME)

    logger.info(f"Listening on ws://0.0.0.0:{s['port']}  metadata={metadata}")
    WebsocketPolicyServer(
        infer_fn=policy.infer,
        host="0.0.0.0",
        port=int(s["port"]),
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runfile", help="Path to runfile yaml (same one used by run_client.sh)")
    args = parser.parse_args()
    main(args.runfile)

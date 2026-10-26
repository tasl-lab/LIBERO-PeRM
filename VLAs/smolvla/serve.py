"""Uniform policy server — SmolVLA (lerobot).

Loads HuggingFaceVLA/smolvla_libero (a 450M SmolVLA fine-tuned on LIBERO
multi-task) and exposes it over the same WebSocket protocol as
server_groot / server_pi05 etc., so PolicyEval/eval.py can evaluate it
without client-side changes.

Usage:
    python serve.py <runfile.yaml>

Preferred entry: bash /home/vla-reasoning/proj/P_Bench/VLA_infer/server_smolvla.sh <runfile.yaml>

Obs protocol (from PolicyEval/eval.py:_obs_element):
    observation/image       : (H, W, 3) uint8 — agentview (224x224 after resize)
    observation/wrist_image : (H, W, 3) uint8 — wrist camera
    observation/state       : (8,) float32 — ee_pos(3) + axisangle(3) + gripper_qpos(2)
    prompt                  : str

Returns:
    actions : (chunk_size, 7) float32 — SmolVLA's default chunk_size=50, first 7 dims
              are LIBERO env-native [pos dx/dy/dz, axis-angle drx/dry/drz, gripper_cmd].
              Gripper convention from LIBERO training = +1 close / -1 open; the base
              LIBERO env expects the same, so no extra flipping.
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
import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _ws_server import WebsocketPolicyServer

VLA_NAME = "smolvla"

# Reduce tokenizer noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Cache VLM downloads in a project-local HF cache
os.environ.setdefault("HF_HOME", "/data1/vla-reasoning/hf_cache")

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
    print(f"  {VLA_NAME} policy server — SmolVLA (lerobot)")
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
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ckpt = s["checkpoint"]
    device = s.get("device", "cuda:0")

    logger.info(f"Loading SmolVLAPolicy: checkpoint={ckpt}, device={device}")
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.to(device).eval()

    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)

    chunk_size = int(getattr(cfg, "chunk_size", 50))
    logger.info(f"Loaded. chunk_size={chunk_size}  n_action_steps={cfg.n_action_steps}")

    return SmolVLAInferWrapper(policy, pre, post, device, chunk_size)


class SmolVLAInferWrapper:
    """Translate P_Bench obs → SmolVLA batch → (T, 7) action chunk.

    Uses `predict_action_chunk` to return the full internal chunk (default 50
    steps) in a single call, so the P_Bench client can honour its own
    `replan_steps` without hammering the server. Calls `policy.reset()` each
    time so the chunk is fresh against the current observation — no stale
    actions carried across episode boundaries.
    """

    def __init__(self, policy, pre, post, device: str, chunk_size: int):
        self.policy = policy
        self.pre = pre
        self.post = post
        self.device = device
        self.chunk_size = chunk_size
        self._first_log = True

    @staticmethod
    def _img_to_chw(x: np.ndarray, size: int | None = None) -> torch.Tensor:
        """(H, W, 3) uint8 → (3, H', W') float32 in [0, 1].

        Pass-through by default: SmolVLA's `prepare_images` does resize_with_pad
        to `resize_imgs_with_padding` (e.g. 512x512) internally. Pre-resizing
        here would only add lossy interpolation. Lerobot's official LIBERO eval
        renders at 360x360 and feeds the raw size straight through.
        """
        if size is not None and x.shape[:2] != (size, size):
            x = cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(
            np.ascontiguousarray(x.astype(np.float32).transpose(2, 0, 1) / 255.0)
        )

    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()

        img = obs["observation/image"]                 # (H, W, 3) uint8
        wrist = obs["observation/wrist_image"]         # (H, W, 3) uint8
        state = np.asarray(obs["observation/state"], dtype=np.float32)  # (8,)
        prompt = obs["prompt"]

        batch = {
            "observation.images.image": self._img_to_chw(img),
            "observation.images.image2": self._img_to_chw(wrist),
            "observation.state": torch.from_numpy(state),
            "task": prompt,
        }

        processed = self.pre(batch)

        # Drop any lingering action queue so we always produce a fresh chunk
        # conditioned on the current observation. (See smolvla.select_action
        # for why the queue exists and why we bypass it.)
        try:
            self.policy.reset()
        except Exception:
            pass

        action_chunk = self.policy.predict_action_chunk(processed)   # (1, chunk_size, 7)
        action_chunk = self.post(action_chunk)                        # same shape, cpu

        if isinstance(action_chunk, torch.Tensor):
            action_np = action_chunk.detach().cpu().float().numpy()
        else:  # defensive: some versions may return ndarray
            action_np = np.asarray(action_chunk, dtype=np.float32)

        # Squeeze batch dim → (chunk_size, 7)
        if action_np.ndim == 3 and action_np.shape[0] == 1:
            action_np = action_np[0]

        if self._first_log:
            self._first_log = False
            logger.info(
                f"[first call] chunk shape={action_np.shape} "
                f"pos[{action_np[:, :3].min():.3f},{action_np[:, :3].max():.3f}] "
                f"rot[{action_np[:, 3:6].min():.3f},{action_np[:, 3:6].max():.3f}] "
                f"grip[{action_np[:, 6].min():.3f},{action_np[:, 6].max():.3f}]"
            )

        dt = time.monotonic() - t0
        return {"actions": action_np.astype(np.float32), "policy_timing": {"total_s": dt}}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(runfile: str) -> None:
    s = load_profile(runfile)
    print_banner(s)

    logger.info(f"Loading checkpoint: config={s['config']!r} dir={s['checkpoint']!r}")
    t0 = time.monotonic()
    policy = load_policy(s)
    logger.info(f"Checkpoint + VLM loaded in {time.monotonic() - t0:.1f}s")

    metadata = {
        "config_name": s["config"],
        "checkpoint_dir": s["checkpoint"],
        "name": s["name"],
        "vla": VLA_NAME,
        "chunk_size": policy.chunk_size,
    }

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

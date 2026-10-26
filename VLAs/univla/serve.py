"""Uniform policy server — UniVLA.

Loads qwbu/univla-7b-224-sft-libero (per-suite: libero-{spatial,object,goal,10})
and exposes it over the same WebSocket protocol as server_groot / server_pi05,
so PolicyEval/eval.py can evaluate it without client-side changes.

Pipeline:
    image + task_label
      → VLA (OpenVLA base) predicts latent-action tokens (+ visual embed)
      → ActionDecoderHead → (window_size=12, 7) normalized action chunk
      → unnormalize with per-suite q01/q99
      → normalize_gripper_action(binarize=True) + invert_gripper_action
      → return (12, 7) chunk to client.

Stateless server: we skip the temporal-ensemble ActionBuffer and the
latent-action-history prompt tokens so there's no per-episode state to reset.
Loss vs the paper recipe is a few SR points but the trade is "correctness on
stream of independent obs calls from a websocket client".

Usage:
    python serve.py <runfile.yaml>

Preferred entry: bash /home/vla-reasoning/proj/P_Bench/VLA_infer/server_univla.sh <runfile.yaml>
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _ws_server import WebsocketPolicyServer

# Make `experiments.robot.*` and `prismatic` importable from the cloned repo.
_UNIVLA_REPO = Path("/data3/zhixu/LIBERO-PeRM/VLAs/univla_repo")
sys.path.insert(0, str(_UNIVLA_REPO))

VLA_NAME = "univla"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/data1/vla-reasoning/hf_cache")

logger = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)


# ─────────────────────────────────────────────────────────────────────────────
# Action decoder (ported from univla_repo/experiments/robot/libero/run_libero_eval.py)
# ─────────────────────────────────────────────────────────────────────────────


class _MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class _ActionDecoderHead(nn.Module):
    """Mirrors ActionDecoderHead in run_libero_eval.py:100-116."""

    def __init__(self, window_size: int = 12):
        super().__init__()
        from prismatic.models.policy.transformer_utils import MAPBlock

        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
        self.proj = nn.Sequential(nn.Linear(512, 7 * window_size), nn.Tanh())

    def forward(self, latent_action_tokens: torch.Tensor, visual_embed: torch.Tensor) -> torch.Tensor:
        # Use last 4 latent tokens, as in the official recipe.
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        return self.proj(self.latent_action_pool(latent_action_tokens, init_embed=visual_embed))


# ─────────────────────────────────────────────────────────────────────────────
# Profile + banner
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
    print(f"  {VLA_NAME} policy server — UniVLA")
    print(bar)
    print(f"  name            : {s['name']}")
    print(f"  config          : {s['config']}")
    print(f"  checkpoint      : {s['checkpoint']}")
    print(f"  action_decoder  : {s.get('action_decoder')}")
    print(f"  unnorm_key      : {s.get('unnorm_key')}")
    print(f"  host / port     : {hostname} ({ip}) / :{s['port']}")
    print(f"  default_prompt  : {s.get('default_prompt')!r}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# cfg shim — UniVLA's robot_utils.get_model expects a draccus cfg. We only
# need the attrs it actually reads.
# ─────────────────────────────────────────────────────────────────────────────


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ─────────────────────────────────────────────────────────────────────────────
# Policy loading + inference wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _patch_openvla_utils_for_sdpa():
    """UniVLA's repo hardcodes attn_implementation='flash_attention_2' in get_vla.
    Our venv has torch SDPA but no flash-attn built for torch 2.10+cu128, so patch
    get_vla to fall back to 'sdpa'. Loss vs flash_attn2: ~10-20% latency + more
    VRAM; numeric output equivalent at BF16."""
    from experiments.robot import openvla_utils as _oul
    if getattr(_oul, "_SERVE_PATCHED", False):
        return

    def _patched_get_vla(cfg):
        """Mirrors univla_repo/experiments/robot/openvla_utils.py:get_vla but:
        (a) uses 'sdpa' instead of 'flash_attention_2' (we don't have flash-attn),
        (b) drops the task_suite_name gating on `.to(DEVICE)` (we always move to cuda).
        """
        import json
        import os as _os
        import torch
        from transformers import (
            AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
        )
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        # Register OpenVLA model to HF Auto classes (guarded idempotent).
        try:
            AutoConfig.register("openvla", OpenVLAConfig)
        except ValueError:
            pass  # already registered
        try:
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        except ValueError:
            pass
        try:
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        except ValueError:
            pass
        try:
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        except ValueError:
            pass

        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if not (cfg.load_in_4bit or cfg.load_in_8bit):
            vla = vla.to("cuda")

        # Attach norm_stats (original helper loads dataset_statistics.json separately).
        stats_path = _os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
        if _os.path.isfile(stats_path):
            with open(stats_path) as f:
                vla.norm_stats = json.load(f)

        vla.eval()
        return vla

    # `experiments.robot.robot_utils` does `from openvla_utils import get_vla` at
    # import time, so it captures the *reference*. Patch both the source module
    # AND the already-imported binding in robot_utils (if loaded).
    _oul.get_vla = _patched_get_vla
    _oul._SERVE_PATCHED = True
    try:
        import experiments.robot.robot_utils as _ru
        _ru.get_vla = _patched_get_vla
    except ImportError:
        pass


def load_policy(s: dict):
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model

    device = s.get("device", "cuda:0")
    window_size = int(s.get("window_size", 12))
    center_crop = bool(s.get("center_crop", True))
    unnorm_key = s.get("unnorm_key", "libero_10")

    _patch_openvla_utils_for_sdpa()

    cfg = _Cfg(
        model_family="openvla",
        pretrained_checkpoint=s["checkpoint"],
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=center_crop,
        unnorm_key=unnorm_key,
    )

    logger.info(f"Loading UniVLA model: ckpt={s['checkpoint']} unnorm_key={unnorm_key}")
    t0 = time.monotonic()
    model = get_model(cfg)
    processor = get_processor(cfg)
    logger.info(f"  model+processor loaded in {time.monotonic() - t0:.1f}s")

    # Fix up unnorm_key for no_noops-suffixed stats (matches run_libero_eval.py:194-196)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{unnorm_key}_no_noops"
        unnorm_key = cfg.unnorm_key
        logger.info(f"  auto-adjusted unnorm_key → {unnorm_key!r}")
    if unnorm_key not in model.norm_stats:
        raise RuntimeError(
            f"unnorm_key {unnorm_key!r} not in model.norm_stats keys: {list(model.norm_stats.keys())}"
        )

    # Load action decoder
    ad_path = s.get("action_decoder") or str(Path(s["checkpoint"]) / "action_decoder.pt")
    logger.info(f"Loading action decoder head: {ad_path}")
    ad = _ActionDecoderHead(window_size=window_size)
    state_dict = torch.load(ad_path, map_location="cpu", weights_only=True)
    missing, unexpected = ad.load_state_dict(state_dict, strict=True)
    ad.eval().to(device).to(torch.float32)
    logger.info(
        f"  decoder loaded  window_size={window_size}  missing={len(missing)}  unexpected={len(unexpected)}"
    )

    return UniVLAInferWrapper(
        model=model,
        processor=processor,
        action_decoder=ad,
        cfg=cfg,
        device=device,
        window_size=window_size,
        unnorm_key=unnorm_key,
    )


class UniVLAInferWrapper:
    """P_Bench obs → UniVLA inference → (window_size, 7) action chunk.

    The VLA predicts latent-action tokens + a visual embedding for a single
    observation; the ActionDecoderHead converts those into a window_size-step
    normalized action chunk. We un-normalize with the loaded suite's q01/q99
    stats, then binarize+flip the gripper to match the LIBERO env convention.
    """

    def __init__(self, model, processor, action_decoder, cfg, device: str, window_size: int, unnorm_key: str):
        self.model = model
        self.processor = processor
        self.action_decoder = action_decoder
        self.cfg = cfg
        self.device = device
        self.window_size = window_size
        self.unnorm_key = unnorm_key

        stats = model.get_action_stats(unnorm_key)
        self._action_low = np.asarray(stats["q01"], dtype=np.float32)
        self._action_high = np.asarray(stats["q99"], dtype=np.float32)
        # Some LIBERO stats include a `mask`: True means normalize (ie unnormalize at test time),
        # False means pass-through (typically gripper dim 6).
        mask = stats.get("mask")
        if mask is None:
            mask = np.ones_like(self._action_low, dtype=bool)
        self._action_mask = np.asarray(mask, dtype=bool)

        self._first_log = True

    def _unnormalize(self, action: np.ndarray) -> np.ndarray:
        """action: (T, 7) in [-1, +1] range → unnormalized action, gripper passed through."""
        low, high, mask = self._action_low, self._action_high, self._action_mask
        unnormed = 0.5 * (action + 1.0) * (high - low) + low
        return np.where(mask, unnormed, action)

    def infer(self, obs: dict) -> dict:
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

        t0 = time.monotonic()

        img = obs["observation/image"]                 # (H, W, 3) uint8, 224x224 after eval.py resize
        # wrist_image is unused by UniVLA (single-cam).
        state = np.asarray(obs["observation/state"], dtype=np.float32)  # (8,)
        prompt = obs["prompt"]

        # UniVLA's get_vla_latent_action expects `full_image` PIL-compatible ndarray.
        univla_obs = {"full_image": img, "state": state}

        # Call the repo's own latent-action helper.  It handles prompt building,
        # center-cropping, processor, and dtype coercion.
        from experiments.robot.robot_utils import get_latent_action

        latent_action, visual_embed, _generated_ids = get_latent_action(
            self.cfg, self.model, univla_obs, prompt, processor=self.processor, hist_action="",
        )

        # Decoder forward → (B, window_size * 7) → reshape to (B, W, 7) → squeeze B=1.
        pred = self.action_decoder(latent_action.to(torch.float32), visual_embed.to(torch.float32))
        action_chunk = pred.reshape(-1, self.window_size, 7).detach().cpu().float().numpy()[0]  # (W, 7)

        # Un-normalize pos/rot; keep gripper as produced by the head (tanh output in [-1,1]).
        action_chunk = self._unnormalize(action_chunk)

        # Gripper post-processing matches run_libero_eval.py:332-338:
        #   1) normalize [0,1] → [-1,+1] + binarize to ±1
        #   2) invert sign (RLDS 0=close,1=open → LIBERO env -1=open,+1=close)
        action_chunk = normalize_gripper_action(action_chunk.copy(), binarize=True)
        action_chunk = invert_gripper_action(action_chunk)

        if self._first_log:
            self._first_log = False
            logger.info(
                f"[first call] chunk shape={action_chunk.shape} "
                f"pos[{action_chunk[:, :3].min():.3f},{action_chunk[:, :3].max():.3f}] "
                f"rot[{action_chunk[:, 3:6].min():.3f},{action_chunk[:, 3:6].max():.3f}] "
                f"grip[{action_chunk[:, 6].min():.3f},{action_chunk[:, 6].max():.3f}]"
            )

        dt = time.monotonic() - t0
        return {"actions": action_chunk.astype(np.float32), "policy_timing": {"total_s": dt}}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(runfile: str) -> None:
    s = load_profile(runfile)
    print_banner(s)

    logger.info(f"Loading checkpoint: config={s['config']!r}  dir={s['checkpoint']!r}")
    t0 = time.monotonic()
    policy = load_policy(s)
    logger.info(f"All components loaded in {time.monotonic() - t0:.1f}s")

    metadata = {
        "config_name": s["config"],
        "checkpoint_dir": s["checkpoint"],
        "action_decoder": s.get("action_decoder"),
        "unnorm_key": policy.unnorm_key,
        "name": s["name"],
        "vla": VLA_NAME,
        "window_size": policy.window_size,
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

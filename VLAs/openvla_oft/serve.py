"""Lightweight WS-based policy server for OpenVLA-OFT.

Loads the model + L1 action head + proprio projector + processor based on a
runfile YAML (see VLA_infer/run_profile/...) and serves over WebSocket using
the openpi protocol so LIBERO/PolicyEval/backends/websocket_backend.py can
talk to it unchanged.

Usage:
  python serve.py /path/to/runfile.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import yaml

# Make experiments.* importable from VLAs/openvla-oft
SCRIPT_DIR = Path(__file__).resolve().parent
OPENVLA_OFT_ROOT = SCRIPT_DIR.parent / "openvla-oft"
if not OPENVLA_OFT_ROOT.is_dir():
    raise FileNotFoundError(
        f"Expected openvla-oft repo at {OPENVLA_OFT_ROOT}; clone it via "
        "`git clone https://github.com/moojink/openvla-oft.git VLAs/openvla-oft`"
    )
sys.path.insert(0, str(OPENVLA_OFT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))                        # for _ws_server, _msgpack_numpy

from experiments.robot.robot_utils import (                            # noqa: E402
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
from experiments.robot.openvla_utils import (                          # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM     # noqa: E402

from _ws_server import WebsocketPolicyServer                            # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("openvla_oft.serve")


def _build_cfg(server_cfg: Dict[str, Any]) -> SimpleNamespace:
    """Translate runfile.server YAML to the cfg object expected by openvla-oft helpers."""
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=server_cfg["checkpoint"],
        load_in_8bit=bool(server_cfg.get("load_in_8bit", False)),
        load_in_4bit=bool(server_cfg.get("load_in_4bit", False)),
        use_l1_regression=bool(server_cfg.get("use_l1_regression", True)),
        use_diffusion=bool(server_cfg.get("use_diffusion", False)),
        use_proprio=bool(server_cfg.get("use_proprio", True)),
        use_film=bool(server_cfg.get("use_film", False)),
        num_images_in_input=int(server_cfg.get("num_images_in_input", 2)),
        unnorm_key=server_cfg.get("unnorm_key", "perm/SingPref"),
        lora_rank=int(server_cfg.get("lora_rank", 32)),
        num_diffusion_steps_train=int(server_cfg.get("num_diffusion_steps_train", 50)),
        num_diffusion_steps_inference=int(server_cfg.get("num_diffusion_steps_inference", 10)),
        center_crop=bool(server_cfg.get("center_crop", True)),
    )


def _resolve_unnorm_key(vla, requested_key: str) -> str:
    """Pick a sensible unnorm key, preferring the requested one but falling back
    to whatever the dataset_statistics.json actually contains."""
    stats = getattr(vla, "norm_stats", None) or {}
    if requested_key in stats:
        return requested_key
    if not stats:
        logger.warning("vla.norm_stats is empty — unnorm will be a no-op")
        return requested_key
    # common synonyms produced by the LeRobot finetune
    synonyms = [
        requested_key.replace("/", "_"),
        requested_key.replace("_", "/"),
        # new perm-named datasets
        "perm/SingPref",
        "perm_singpref",
        "perm_SingPref",
        # legacy pbench-named checkpoints (kept for back-compat with old dataset_statistics.json)
        "pbench/SingPref",
        "pbench_singpref",
        "pbench_SingPref",
    ]
    for k in synonyms:
        if k in stats:
            logger.warning("unnorm_key %r not found; falling back to %r", requested_key, k)
            return k
    only = next(iter(stats))
    logger.warning("unnorm_key %r not found; falling back to first key %r", requested_key, only)
    return only


def _build_policy_fn(cfg, vla, processor, action_head, proprio_projector, image_size: int):
    chunk_size = int(NUM_ACTIONS_CHUNK)

    def _policy_fn(obs_msg: Dict[str, Any]) -> Dict[str, Any]:
        full_image = np.asarray(obs_msg["observation/image"])
        prompt = obs_msg["prompt"]

        # Eval client may send the prompt as bytes (msgpack default)
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8", errors="replace")

        full_image = resize_image_for_policy(full_image, image_size)
        obs: Dict[str, Any] = {"full_image": full_image}

        if cfg.num_images_in_input > 1:
            wrist = obs_msg.get("observation/wrist_image")
            if wrist is None:
                raise RuntimeError(
                    f"num_images_in_input={cfg.num_images_in_input} but client sent no observation/wrist_image"
                )
            obs["wrist_image"] = resize_image_for_policy(np.asarray(wrist), image_size)

        if cfg.use_proprio:
            state = np.asarray(obs_msg["observation/state"], dtype=np.float32)
            obs["state"] = state

        actions = get_action(
            cfg=cfg,
            model=vla,
            obs=obs,
            task_label=prompt,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            use_film=cfg.use_film,
        )

        # `actions` is either a List[np.ndarray] of length chunk_size (OFT path,
        # each (action_dim,)) or a single np.ndarray of shape (action_dim,)
        # (zero-shot / discrete-action path when action_head is None).
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[None, :]
        raw_actions_arr = actions_arr.copy()

        # OFT model outputs gripper in [0,1] with 0=close,1=open (RLDS dataloader
        # convention). LIBERO env expects gripper in [-1,+1] with -1=open,+1=close.
        # Mirror experiments/robot/libero/run_libero_eval.py:process_action so the
        # client can env.step(action) directly without per-VLA post-processing.
        if cfg.model_family == "openvla":
            actions_arr = normalize_gripper_action(actions_arr, binarize=True)
            actions_arr = invert_gripper_action(actions_arr)

        return {
            "actions": actions_arr,
            "raw_actions": raw_actions_arr,
            "chunk_size": int(actions_arr.shape[0]),
        }

    return _policy_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runfile", help="path to runfile YAML (must contain `server:` section)")
    args = ap.parse_args()

    with open(args.runfile, "r") as f:
        full = yaml.safe_load(f) or {}
    s = full.get("server") or {}
    if s.get("vla") not in (None, "openvla_oft"):
        raise SystemExit(f"runfile vla mismatch: expected openvla_oft, got {s.get('vla')!r}")
    for required in ("checkpoint", "port"):
        if not s.get(required):
            raise SystemExit(f"runfile.server.{required} is required")

    if s.get("cuda_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(s["cuda_devices"])

    cfg = _build_cfg(s)
    image_size = int(s.get("image_size", 224))

    # check_model_logic_mismatch() walks `./prismatic/` relative to CWD,
    # so we must run get_model from the openvla-oft repo root for it to find
    # the up-to-date modeling_prismatic.py and copy it into ckpt dirs that
    # were downloaded before the OFT-fork was published (e.g. base openvla-7b).
    saved_cwd = os.getcwd()
    os.chdir(str(OPENVLA_OFT_ROOT))
    try:
        logger.info("loading VLA from %s", cfg.pretrained_checkpoint)
        vla = get_model(cfg)
    finally:
        os.chdir(saved_cwd)
    processor = get_processor(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg, vla.llm_dim, proprio_dim=int(s.get("proprio_dim", PROPRIO_DIM))
        )

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, vla.llm_dim)

    cfg.unnorm_key = _resolve_unnorm_key(vla, cfg.unnorm_key)

    metadata = {
        "name": s.get("name", "openvla_oft"),
        "config": s.get("config", "openvla_oft"),
        "config_name": s.get("config", "openvla_oft"),
        "checkpoint_dir": str(cfg.pretrained_checkpoint),
        "num_images_in_input": cfg.num_images_in_input,
        "use_proprio": cfg.use_proprio,
        "image_size": image_size,
        "action_dim": int(s.get("action_dim", 7)),
        "chunk_size": int(NUM_ACTIONS_CHUNK),
        "unnorm_key": cfg.unnorm_key,
    }

    policy_fn = _build_policy_fn(cfg, vla, processor, action_head, proprio_projector, image_size)

    # ── Warmup ────────────────────────────────────────────────────────────────
    # Pre-compile flash-attn / cuDNN kernels with dummy obs so the first real
    # client call is sub-second. Without this, the first call can take 30+s of
    # JIT compile and trip websocket keepalive timeouts (or worse, look like a
    # hang). 3 calls is enough to reach steady-state on Blackwell sm_120.
    logger.info("warming up policy_fn (3 dummy forwards)")
    _dummy_obs = {
        "observation/image": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        "prompt": "warmup",
    }
    if cfg.num_images_in_input > 1:
        _dummy_obs["observation/wrist_image"] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    if cfg.use_proprio:
        _dummy_obs["observation/state"] = np.zeros((int(s.get("proprio_dim", PROPRIO_DIM)),), dtype=np.float32)
    import time as _time
    for _i in range(3):
        _t = _time.time()
        try:
            _ = policy_fn(_dummy_obs)
            logger.info("warmup %d/3 ok in %.2fs", _i + 1, _time.time() - _t)
        except Exception as _e:
            logger.warning("warmup %d/3 failed: %s", _i + 1, _e)
    logger.info("warmup done")

    server = WebsocketPolicyServer(
        policy_fn=policy_fn,
        host=s.get("host", "0.0.0.0"),
        port=int(s["port"]),
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

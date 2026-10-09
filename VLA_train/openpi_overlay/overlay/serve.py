"""Uniform policy server for openpi (pi0 / pi0.5).

Reads the same runfile yaml the client uses. Extracts `server.*` and launches
a WebsocketPolicyServer. Launched via ../server_pi05.sh, which passes the
runfile path as the sole positional argument.

Usage:
    uv run serve.py <runfile.yaml>

Preferred entry: bash /data3/zhixu/P_Bench/VLA_infer/server_pi05.sh <runfile.yaml>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import time
import traceback

import numpy as np
import yaml

from openpi.policies import policy as _policy  # noqa: F401  (kept for namespace)
from openpi.policies import policy_config as _pc
from openpi.serving import websocket_policy_server as _ws
from openpi.training import config as _config
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _wserver
import websockets.frames

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
    if s["vla"] not in ("pi05", "pi0"):
        raise ValueError(f"openpi/serve.py invoked for vla={s['vla']!r} (expected 'pi05' or 'pi0')")
    return s


def print_banner(s: dict) -> None:
    try:
        import jax
        devices = jax.devices()
    except Exception as e:  # pragma: no cover
        devices = f"(jax.devices() failed: {e})"
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "?"
    bar = "─" * 62
    print(bar)
    print("  pi0.5 (openpi) policy server")
    print(bar)
    print(f"  name          : {s['name']}")
    print(f"  config        : {s['config']}")
    print(f"  checkpoint    : {s['checkpoint']}")
    print(f"  host / port   : {hostname} ({ip}) / :{s['port']}")
    print(f"  default_prompt: {s.get('default_prompt')!r}")
    print(f"  jax devices   : {devices}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server with per-connection summary logging
# ─────────────────────────────────────────────────────────────────────────────


class LoggingWsServer(_ws.WebsocketPolicyServer):
    """WebsocketPolicyServer subclass that logs Connect / Disconnect + per-
    connection summary (reqs, mean/p95 infer_ms). Mirrors the client's
    per-episode log granularity.

    Also supports a client-initiated shutdown: if the client sends a message
    with ``{"__end__": True}``, the server acks and exits cleanly so the GPU
    is released as soon as eval finishes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutdown_event: asyncio.Event | None = None

    async def run(self):
        self._shutdown_event = asyncio.Event()
        async with _wserver.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            async def _wait_and_close():
                await self._shutdown_event.wait()
                logger.info("Shutdown event set — closing websocket server.")
                server.close()

            asyncio.create_task(_wait_and_close())
            await server.serve_forever()

    async def _handler(self, websocket: _wserver.ServerConnection):
        addr = websocket.remote_address
        logger.info(f"Connect {addr}")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        reqs = 0
        lat_ms: list[float] = []
        prev_total_time = None
        try:
            while True:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if isinstance(obs, dict) and obs.get("__end__"):
                    logger.info(f"Received __end__ from {addr}; scheduling server shutdown.")
                    await websocket.send(packer.pack({"__end_ack__": True}))
                    if self._shutdown_event is not None:
                        self._shutdown_event.set()
                    break

                ti = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - ti) * 1000.0
                lat_ms.append(infer_ms)
                reqs += 1

                action["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
        except websockets.ConnectionClosed:
            pass
        except Exception:
            try:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
            finally:
                self._log_summary(addr, reqs, lat_ms)
            raise
        finally:
            self._log_summary(addr, reqs, lat_ms)

    @staticmethod
    def _log_summary(addr, reqs: int, lat_ms: list[float]) -> None:
        if reqs == 0:
            logger.info(f"Disconnect {addr} | reqs=0")
            return
        mean = float(np.mean(lat_ms))
        p95 = float(np.percentile(lat_ms, 95))
        logger.info(
            f"Disconnect {addr} | reqs={reqs} | mean_infer_ms={mean:.1f} | p95_infer_ms={p95:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(runfile: str) -> None:
    s = load_profile(runfile)
    print_banner(s)

    logger.info(f"Loading checkpoint: config={s['config']!r}  dir={s['checkpoint']!r}")
    t0 = time.monotonic()
    policy = _pc.create_trained_policy(
        _config.get_config(s["config"]),
        s["checkpoint"],
        default_prompt=s.get("default_prompt"),
    )
    logger.info(f"Checkpoint loaded in {time.monotonic() - t0:.1f}s")

    metadata = dict(policy.metadata or {})
    metadata.setdefault("config_name", s["config"])
    metadata.setdefault("checkpoint_dir", s["checkpoint"])
    metadata.setdefault("name", s["name"])

    logger.info(f"Listening on ws://0.0.0.0:{s['port']}  metadata={metadata}")
    LoggingWsServer(
        policy=policy,
        host="0.0.0.0",
        port=int(s["port"]),
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runfile", help="Path to runfile yaml (same one used by run_client.sh)")
    args = parser.parse_args()
    main(args.runfile)

"""Vendored, self-contained WebSocket policy server + msgpack-numpy helpers.

Kept in each VLA codebase so the VLA doesn't depend on openpi. Mirrors the
protocol of openpi.serving.websocket_policy_server (msgpack over ws with
numpy arrays). Adds per-connection summary logging to match the client's
per-episode log granularity.

External deps: msgpack, numpy, websockets.
"""

from __future__ import annotations

import asyncio
import functools
import http
import logging
import time
import traceback
from typing import Callable

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger("serve.ws")


# ─────────────────────────────────────────────────────────────────────────────
# msgpack + numpy (vendored from openpi_client.msgpack_numpy)
# ─────────────────────────────────────────────────────────────────────────────


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server
# ─────────────────────────────────────────────────────────────────────────────


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class WebsocketPolicyServer:
    """Generic policy server. `infer_fn` takes an obs dict and returns an action dict.

    On connect: sends msgpack-packed `metadata` dict. Per request: unpacks obs,
    calls `infer_fn(obs)`, injects server_timing, sends back. On disconnect:
    prints summary (reqs, mean/p95 infer_ms).
    """

    def __init__(
        self,
        infer_fn: Callable[[dict], dict],
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._infer = infer_fn
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        addr = websocket.remote_address
        logger.info(f"Connect {addr}")
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        reqs = 0
        lat_ms: list[float] = []
        prev_total_time = None
        try:
            while True:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                ti = time.monotonic()
                action = self._infer(obs)
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

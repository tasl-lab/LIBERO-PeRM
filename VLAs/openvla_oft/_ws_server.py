"""Async WebSocket server that speaks openpi's policy-server protocol.

Wire format:
  - Server sends one msgpack-numpy packed metadata dict on connect.
  - Client sends msgpack-numpy packed obs dict; server replies with msgpack-numpy
    packed action dict.
  - Special obs `{"__end__": True}` triggers graceful shutdown so the GPU is
    released as soon as eval finishes (LIBERO/PolicyEval/backends/websocket_backend.py
    sends this in shutdown_server()).
  - GET /healthz returns 200 OK for liveness probes.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets
import websockets.asyncio.server as _server
import websockets.frames

import _msgpack_numpy as _mp

logger = logging.getLogger(__name__)


PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class WebsocketPolicyServer:
    def __init__(
        self,
        policy_fn: PolicyFn,
        host: str = "0.0.0.0",
        port: int = 8014,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._policy_fn = policy_fn
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._stop_event: Optional[asyncio.Event] = None

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._stop_event = asyncio.Event()
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            # First inference may take 30+s on a cold GPU (flash-attn JIT kernel
            # compile). Default keepalive (20s ping_interval / 20s ping_timeout)
            # would tear the connection mid-call. Disable pings — we don't need
            # liveness pinging for a local-loopback policy server.
            ping_interval=None,
            ping_timeout=None,
        ):
            logger.info("server listening on %s:%d", self._host, self._port)
            print(f"server listening on {self._host}:{self._port}", flush=True)
            await self._stop_event.wait()
        # give in-flight close frames a moment to flush
        await asyncio.sleep(0.1)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        peer = websocket.remote_address
        logger.info("connection from %s opened", peer)
        packer = _mp.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: Optional[float] = None
        while True:
            try:
                start_time = time.monotonic()
                raw = await websocket.recv()
                obs = _mp.unpackb(raw)

                if isinstance(obs, dict) and obs.get("__end__"):
                    logger.info("received __end__; scheduling shutdown")
                    try:
                        await websocket.send(packer.pack({"ack": True}))
                    except Exception:
                        pass
                    if self._stop_event is not None:
                        self._stop_event.set()
                    break

                infer_start = time.monotonic()
                action = self._policy_fn(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0

                if not isinstance(action, dict):
                    raise RuntimeError(f"policy_fn must return dict, got {type(action)}")
                action.setdefault("server_timing", {})
                action["server_timing"]["infer_ms"] = infer_ms
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", peer)
                break
            except Exception:
                tb = traceback.format_exc()
                logger.exception("inference error")
                try:
                    await websocket.send(tb)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="internal server error",
                    )
                except Exception:
                    pass
                # let the loop continue handling reconnects rather than killing the process
                break


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None

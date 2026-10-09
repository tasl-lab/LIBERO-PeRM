"""Generic websocket backend.

Talks to any inference server that follows the openpi websocket contract:
  - connects on ws://host[:port]
  - on connect, server sends msgpack-packed metadata dict
  - client sends msgpack-packed obs dict, server replies with msgpack-packed action dict

Thin wrapper around openpi_client.websocket_client_policy.WebsocketClientPolicy.
Any new backend (openvla_oft, GROOT, ...) that reuses
VLAs/openpi/src/openpi/serving/websocket_policy_server.WebsocketPolicyServer on
the server side works here unchanged.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _wc

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class WebsocketBackend(InferenceBackend):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        connect_host = "127.0.0.1" if host == "0.0.0.0" else host
        self._client = _wc.WebsocketClientPolicy(host=connect_host, port=port, api_key=api_key)
        logger.info(
            f"Connected to ws://{connect_host}:{port}; server metadata keys: "
            f"{list(self._client.get_server_metadata())}"
        )

    def infer(self, element: Dict) -> Dict:
        return self._client.infer(element)

    def shutdown_server(self) -> None:
        """Tell the server the eval run is over; it will close and exit,
        releasing the GPU. Safe to call multiple times / on a broken socket."""
        ws = getattr(self._client, "_ws", None)
        if ws is None:
            return
        try:
            packer = msgpack_numpy.Packer()
            ws.send(packer.pack({"__end__": True}))
            try:
                ws.recv(timeout=5)
            except Exception:
                pass
            logger.info("Sent __end__ to server.")
        except Exception as e:
            logger.warning(f"shutdown_server: {e}")

    def close(self) -> None:
        ws = getattr(self._client, "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

"""Synchronous ZeroMQ client for the GameWorld Breakout browser service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import zmq


PROTOCOL_VERSION = 6
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5561"
EXPECTED_EVALUATION_TIMING = {
    "action_hold_s": 0.2,
    "post_action_idle_s": 0.05,
    "nominal_observation_interval_s": 0.25,
    "evaluation_state_after_action": True,
    "screenshot_before_pause": True,
    "observation_state_after_pause": True,
}


class BreakoutRPCError(RuntimeError):
    """Raised when the browser service times out or violates its protocol."""


@dataclass(frozen=True, slots=True)
class RPCObservation:
    metadata: dict[str, Any]
    png: bytes


class BreakoutRPCClient:
    """One request/reply client; intended for DIAMOND's single real environment."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout_ms: int = 30_000,
        context: zmq.Context | None = None,
    ) -> None:
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._owns_context = context is None
        self._context = context or zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.connect(endpoint)
        self._closed = False

    @staticmethod
    def _decode_metadata(frame: bytes) -> dict[str, Any]:
        try:
            metadata = json.loads(frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BreakoutRPCError("Browser service returned invalid JSON metadata") from exc
        if not isinstance(metadata, dict):
            raise BreakoutRPCError("Browser service metadata must be a JSON object")
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise BreakoutRPCError(
                "Incompatible GameWorld bridge protocol: "
                f"received={metadata.get('protocol_version')!r}, expected={PROTOCOL_VERSION}"
            )
        if metadata.get("ok") is not True:
            raise BreakoutRPCError(
                f"Browser service error ({metadata.get('error_type', 'unknown')}): "
                f"{metadata.get('error', 'no error message')}"
            )
        return metadata

    @classmethod
    def decode_response(
        cls,
        frames: list[bytes],
        *,
        expect_png: bool,
    ) -> tuple[dict[str, Any], bytes | None]:
        expected_frames = 2 if expect_png else 1
        if len(frames) != expected_frames:
            raise BreakoutRPCError(
                f"Browser service returned {len(frames)} frames; expected {expected_frames}"
            )
        metadata = cls._decode_metadata(frames[0])
        png = frames[1] if expect_png else None
        if expect_png and (not png or not png.startswith(b"\x89PNG\r\n\x1a\n")):
            raise BreakoutRPCError("Browser service returned a missing or invalid PNG frame")
        return metadata, png

    def _request(
        self,
        payload: dict[str, Any],
        *,
        expect_png: bool,
    ) -> tuple[dict[str, Any], bytes | None]:
        if self._closed:
            raise BreakoutRPCError("Browser RPC client is closed")
        try:
            self._socket.send_json(payload)
            frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            self.close()
            raise BreakoutRPCError(
                f"Timed out communicating with GameWorld browser service at {self.endpoint}"
            ) from exc
        except zmq.ZMQError as exc:
            self.close()
            raise BreakoutRPCError(
                f"ZeroMQ failure communicating with GameWorld at {self.endpoint}: {exc}"
            ) from exc
        return self.decode_response(frames, expect_png=expect_png)

    def health(self) -> dict[str, Any]:
        metadata, _ = self._request({"cmd": "health"}, expect_png=False)
        if metadata.get("event") != "health":
            raise BreakoutRPCError(f"Unexpected health response: {metadata.get('event')!r}")
        return metadata

    def reset(self, *, level: int, seed: int, initial_lives: int = 3) -> RPCObservation:
        metadata, png = self._request(
            {
                "cmd": "reset",
                "level": level,
                "seed": seed,
                "initial_lives": initial_lives,
            },
            expect_png=True,
        )
        if metadata.get("event") != "reset" or png is None:
            raise BreakoutRPCError("Browser service returned an invalid reset response")
        return RPCObservation(metadata=metadata, png=png)

    def step(self, action: int) -> RPCObservation:
        metadata, png = self._request(
            {"cmd": "step", "action": action},
            expect_png=True,
        )
        if metadata.get("event") != "step" or png is None:
            raise BreakoutRPCError("Browser service returned an invalid step response")
        return RPCObservation(metadata=metadata, png=png)

    def shutdown_server(self) -> None:
        self._request({"cmd": "close"}, expect_png=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def __enter__(self) -> BreakoutRPCClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "BreakoutRPCClient",
    "BreakoutRPCError",
    "DEFAULT_ENDPOINT",
    "EXPECTED_EVALUATION_TIMING",
    "PROTOCOL_VERSION",
    "RPCObservation",
]

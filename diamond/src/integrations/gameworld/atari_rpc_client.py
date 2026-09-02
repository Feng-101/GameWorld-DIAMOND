"""RPC client for the deterministic Atari-style Breakout browser service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import zmq


PROTOCOL_VERSION = 9
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5661"
EXPECTED_ATARI_TIMING = {
    "mode": "deterministic_gameworld_frames",
    "frame_rate_hz": 60,
    "frame_skip": 4,
    "game_time_per_step_s": 4 / 60,
    "action_repeat_frames": 4,
    "screenshots_while_paused": True,
    "observation_frame": "last_executed_frame",
    "max_pool_last_two_frames": False,
    "wall_clock_sleep_s": 0.0,
}
EXPECTED_ACTION_MEANINGS = ["NOOP", "FIRE", "RIGHT", "LEFT"]


class AtariBreakoutRPCError(RuntimeError):
    """Raised when the deterministic browser service violates its contract."""


@dataclass(frozen=True, slots=True)
class AtariRPCObservation:
    metadata: dict[str, Any]
    png: bytes


@dataclass(frozen=True, slots=True)
class AtariRPCRecordingObservation:
    """One agent observation plus every native 60 Hz canvas frame in the step."""

    metadata: dict[str, Any]
    png: bytes
    recorded_pngs: tuple[bytes, ...]


class AtariBreakoutRPCClient:
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
            raise AtariBreakoutRPCError(
                "Atari browser service returned invalid JSON metadata"
            ) from exc
        if not isinstance(metadata, dict):
            raise AtariBreakoutRPCError("RPC metadata must be a JSON object")
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise AtariBreakoutRPCError(
                "Incompatible Atari browser protocol: "
                f"received={metadata.get('protocol_version')!r}, "
                f"expected={PROTOCOL_VERSION}"
            )
        if metadata.get("ok") is not True:
            raise AtariBreakoutRPCError(
                f"Atari browser error ({metadata.get('error_type', 'unknown')}): "
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
            raise AtariBreakoutRPCError(
                f"Atari browser returned {len(frames)} frames; "
                f"expected {expected_frames}"
            )
        metadata = cls._decode_metadata(frames[0])
        png = frames[1] if expect_png else None
        if expect_png and (not png or not png.startswith(b"\x89PNG\r\n\x1a\n")):
            raise AtariBreakoutRPCError("Atari browser returned an invalid PNG")
        return metadata, png

    def _request_raw(self, payload: dict[str, Any]) -> list[bytes]:
        if self._closed:
            raise AtariBreakoutRPCError("Atari browser RPC client is closed")
        try:
            self._socket.send_json(payload)
            frames = self._socket.recv_multipart()
        except zmq.Again as exc:
            self.close()
            raise AtariBreakoutRPCError(
                f"Timed out communicating with Atari browser at {self.endpoint}"
            ) from exc
        except zmq.ZMQError as exc:
            self.close()
            raise AtariBreakoutRPCError(
                f"ZeroMQ failure communicating with Atari browser: {exc}"
            ) from exc
        return frames

    def _request(
        self,
        payload: dict[str, Any],
        *,
        expect_png: bool,
    ) -> tuple[dict[str, Any], bytes | None]:
        frames = self._request_raw(payload)
        return self.decode_response(frames, expect_png=expect_png)

    def health(self) -> dict[str, Any]:
        metadata, _ = self._request({"cmd": "health"}, expect_png=False)
        if metadata.get("event") != "health":
            raise AtariBreakoutRPCError(
                f"Unexpected health event: {metadata.get('event')!r}"
            )
        return metadata

    def reset(
        self,
        *,
        level: int,
        seed: int,
        initial_lives: int = 5,
    ) -> AtariRPCObservation:
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
            raise AtariBreakoutRPCError("Invalid Atari reset response")
        return AtariRPCObservation(metadata=metadata, png=png)

    def step(self, action: int) -> AtariRPCObservation:
        metadata, png = self._request(
            {"cmd": "step", "action": action},
            expect_png=True,
        )
        if metadata.get("event") != "step" or png is None:
            raise AtariBreakoutRPCError("Invalid Atari step response")
        return AtariRPCObservation(metadata=metadata, png=png)

    def step_record(self, action: int) -> AtariRPCRecordingObservation:
        """Step normally and receive every executed native canvas frame.

        This diagnostic command is not used by training. The game stays paused
        during every capture, so video encoding and screenshot latency advance
        zero simulated game time.
        """

        frames = self._request_raw({"cmd": "step_record", "action": action})
        if len(frames) < 3:
            raise AtariBreakoutRPCError(
                f"Atari recording response has {len(frames)} frames; expected >= 3"
            )
        metadata = self._decode_metadata(frames[0])
        if metadata.get("event") != "step":
            raise AtariBreakoutRPCError("Invalid Atari recording step response")
        observation_png = frames[1]
        recorded_pngs = tuple(frames[2:])
        all_pngs = (observation_png, *recorded_pngs)
        if any(
            not png or not png.startswith(b"\x89PNG\r\n\x1a\n")
            for png in all_pngs
        ):
            raise AtariBreakoutRPCError("Atari recording response contains invalid PNG")
        expected_count = metadata.get("recorded_frame_count")
        recording_frames = metadata.get("recording_frames")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != len(recorded_pngs)
            or not isinstance(recording_frames, list)
            or len(recording_frames) != len(recorded_pngs)
        ):
            raise AtariBreakoutRPCError(
                "Atari recording metadata does not match returned canvas frames"
            )
        return AtariRPCRecordingObservation(
            metadata=metadata,
            png=observation_png,
            recorded_pngs=recorded_pngs,
        )

    def shutdown_server(self) -> None:
        self._request({"cmd": "close"}, expect_png=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()


__all__ = [
    "AtariBreakoutRPCClient",
    "AtariBreakoutRPCError",
    "AtariRPCObservation",
    "AtariRPCRecordingObservation",
    "DEFAULT_ENDPOINT",
    "EXPECTED_ACTION_MEANINGS",
    "EXPECTED_ATARI_TIMING",
    "PROTOCOL_VERSION",
]

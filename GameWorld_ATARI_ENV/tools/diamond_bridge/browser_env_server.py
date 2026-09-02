"""Deterministic browser service for GameWorld Breakout.

Unlike GameWorld's evaluation loop, this service never lets screenshot or RPC
latency advance the game.  Breakout remains paused and each action invokes
exactly four 60 Hz engine update/draw frames. The returned observation is the
last executed frame; max pooling is deliberately not used because it erases
GameWorld's black ball against the lighter background.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import json
import logging
import random
import sys
from dataclasses import dataclass
from typing import Any

import zmq
import zmq.asyncio

from env import BrowserConfig, BrowserGameManager, GameLauncher

from .breakout_protocol import (
    ACTION_KEYS,
    ACTION_MEANINGS,
    FRAME_SKIP,
    action_key,
    atari_timing,
    transition_events,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_BIND = "tcp://127.0.0.1:5661"
DEFAULT_GAME_PORT = 8201
PROTOCOL_VERSION = 9
CANVAS_SELECTOR = "#canvas"
EXPECTED_CANVAS_BOUNDS = {"x": 240, "y": 17, "width": 800, "height": 600}
# Date.now is represented near Unix-epoch milliseconds, so adding 1/60 second
# has sub-microsecond IEEE-754 rounding.  One microsecond is still over four
# orders of magnitude tighter than a 16.67 ms game frame.
GAME_TIME_TOLERANCE_MS = 1e-3


@dataclass(slots=True)
class BrowserObservation:
    metadata: dict[str, Any]
    png: bytes
    # Only populated for the diagnostic ``step_record`` command. Normal
    # training/test ``step`` replies are unchanged and pay no extra capture
    # cost.
    recorded_pngs: list[bytes] | None = None


def _game_time_ms(state: dict[str, Any]) -> float:
    value = state.get("gameTimeMs")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Breakout returned invalid gameTimeMs: {value!r}")
    return float(value)


def _is_terminal(state: dict[str, Any]) -> bool:
    terminal = state.get("terminal")
    return bool(isinstance(terminal, dict) and terminal.get("isTerminal") is True)


class AtariBreakoutBrowserEnvironment:
    """One paused Chromium session stepped as a deterministic 60 Hz emulator."""

    def __init__(
        self,
        *,
        game_port: int = DEFAULT_GAME_PORT,
        headless: bool = True,
        width: int = 1280,
        height: int = 720,
        noop_max: int = 30,
    ) -> None:
        if noop_max < 0:
            raise ValueError("noop_max must be non-negative")
        self.game_port = game_port
        self.headless = headless
        self.width = width
        self.height = height
        self.noop_max = noop_max
        self.launcher: GameLauncher | None = None
        self.manager: BrowserGameManager | None = None
        self.previous_state: dict[str, Any] | None = None
        self.step_count = 0
        self._frame_stepper_installed = False

    def _require_manager(self) -> BrowserGameManager:
        if self.manager is None:
            raise RuntimeError("Browser environment has not been started")
        return self.manager

    async def start(self) -> None:
        if self.manager is not None:
            return

        self.launcher = GameLauncher("05_breakout", port=self.game_port)
        game_url = self.launcher.start()
        self.manager = BrowserGameManager(
            BrowserConfig(
                game_url=game_url,
                width=self.width,
                height=self.height,
                headless=self.headless,
                speed_multiplier=1.0,
                random_seed=42,
            )
        )
        try:
            await self.manager.start()
            if not await self.manager.wait_until_actionable(stage="atari-startup"):
                raise RuntimeError("Breakout did not become actionable during startup")
            await self.manager.pause_game()
            await self._install_frame_stepper()
        except Exception:
            await self.close()
            raise

    async def _install_frame_stepper(self) -> None:
        manager = self._require_manager()
        if not manager.page:
            raise RuntimeError("Breakout browser page is unavailable")
        result = await manager.page.evaluate(
            "() => window.__installAtariFrameStepper()"
        )
        if not isinstance(result, dict) or result.get("installed") is not True:
            raise RuntimeError(f"Failed to install Atari frame stepper: {result!r}")
        self._frame_stepper_installed = True

    async def _advance_frames(self, count: int) -> dict[str, Any]:
        manager = self._require_manager()
        if not manager.page or not self._frame_stepper_installed:
            raise RuntimeError("Atari frame stepper is unavailable")
        result = await manager.page.evaluate(
            "(count) => window.__advanceAtariFrames(count)",
            count,
        )
        if (
            not isinstance(result, dict)
            or result.get("framesAdvanced") != count
        ):
            raise RuntimeError(f"Atari frame step failed: {result!r}")
        return result

    async def _render(self) -> None:
        manager = self._require_manager()
        if not manager.page:
            raise RuntimeError("Breakout browser page is unavailable")
        result = await manager.page.evaluate("() => window.__renderAtariFrame()")
        if not isinstance(result, dict) or result.get("rendered") is not True:
            raise RuntimeError(f"Atari render failed: {result!r}")

    async def _state(self) -> dict[str, Any]:
        state = await self._require_manager().get_game_state()
        if not isinstance(state, dict):
            raise RuntimeError("Breakout gameAPI returned no state")
        return state

    async def _frame(self) -> bytes:
        return await self._require_manager().capture_screenshot_bytes()

    async def _canvas_frame(self) -> bytes:
        """Capture the native 800x600 canvas losslessly while paused."""
        manager = self._require_manager()
        if not manager.page:
            raise RuntimeError("Breakout browser page is unavailable")
        data_url = await manager.page.evaluate(
            """(selector) => {
                const canvas = document.querySelector(selector);
                if (!(canvas instanceof HTMLCanvasElement)) return null;
                return canvas.toDataURL('image/png');
            }""",
            CANVAS_SELECTOR,
        )
        prefix = "data:image/png;base64,"
        if not isinstance(data_url, str) or not data_url.startswith(prefix):
            raise RuntimeError("Breakout canvas could not be exported as PNG")
        return base64.b64decode(data_url[len(prefix) :], validate=True)

    async def _paused_frame(self) -> tuple[bytes, float]:
        before = await self._state()
        png = await self._frame()
        after = await self._state()
        advance_ms = _game_time_ms(after) - _game_time_ms(before)
        if abs(advance_ms) > GAME_TIME_TOLERANCE_MS:
            raise RuntimeError(
                "Screenshot advanced the paused Atari environment: "
                f"delta={advance_ms}ms"
            )
        return png, advance_ms

    async def _paused_canvas_frame(self) -> tuple[bytes, float]:
        before = await self._state()
        png = await self._canvas_frame()
        after = await self._state()
        advance_ms = _game_time_ms(after) - _game_time_ms(before)
        if abs(advance_ms) > GAME_TIME_TOLERANCE_MS:
            raise RuntimeError(
                "Canvas screenshot advanced the paused Atari environment: "
                f"delta={advance_ms}ms"
            )
        return png, advance_ms

    @staticmethod
    def _recording_state(
        state: dict[str, Any],
        *,
        frame_index: int,
        screenshot_advance_ms: float,
    ) -> dict[str, Any]:
        game_state = state.get("game_state")
        metrics = state.get("metrics")
        terminal = state.get("terminal")
        return {
            "frame_index": frame_index,
            "game_time_ms": _game_time_ms(state),
            "screenshot_game_time_advance_ms": screenshot_advance_ms,
            "level": game_state.get("level") if isinstance(game_state, dict) else None,
            "score": game_state.get("score") if isinstance(game_state, dict) else None,
            "completion_progress": (
                game_state.get("completion_progress")
                if isinstance(game_state, dict)
                else None
            ),
            "lives": metrics.get("lives") if isinstance(metrics, dict) else None,
            "is_terminal": bool(
                isinstance(terminal, dict) and terminal.get("isTerminal") is True
            ),
            "terminal_outcome": (
                terminal.get("outcome") if isinstance(terminal, dict) else None
            ),
        }

    async def _canvas_bounds(self) -> dict[str, int]:
        manager = self._require_manager()
        if not manager.page:
            raise RuntimeError("Browser page is unavailable")
        raw = await manager.page.evaluate(
            """(selector) => {
                const element = document.querySelector(selector);
                if (!element) return null;
                const rect = element.getBoundingClientRect();
                return {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                };
            }""",
            CANVAS_SELECTOR,
        )
        if not isinstance(raw, dict):
            raise RuntimeError(f"Unable to locate canvas {CANVAS_SELECTOR!r}")
        bounds = {key: int(raw[key]) for key in EXPECTED_CANVAS_BOUNDS}
        mismatches = {
            key: (bounds.get(key), expected)
            for key, expected in EXPECTED_CANVAS_BOUNDS.items()
            if abs(int(bounds.get(key, -10_000)) - expected) > 1
        }
        if mismatches:
            raise RuntimeError(
                "Breakout canvas violates the observation crop contract: "
                f"actual={bounds}, expected={EXPECTED_CANVAS_BOUNDS}"
            )
        return bounds

    async def _release_all_keys(self) -> None:
        manager = self.manager
        if manager is None or not manager.page:
            return
        for key in sorted({key for key in ACTION_KEYS.values() if key is not None}):
            try:
                await manager.page.keyboard.up(key)
            except Exception:  # noqa: BLE001
                LOGGER.debug("Key release skipped for %s", key)

    async def reset(
        self,
        *,
        level: int,
        seed: int,
        initial_lives: int = 5,
    ) -> BrowserObservation:
        if isinstance(level, bool) or level not in range(1, 6):
            raise ValueError(f"level must be an integer from 1 to 5, got {level!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        if initial_lives != 5:
            raise ValueError("Atari-style Breakout always starts with exactly five lives")

        manager = self._require_manager()
        await manager.pause_game()
        await self._release_all_keys()
        did_reset = await manager.reset_game(
            {
                "level": level,
                "seed": seed,
                "initial_lives": initial_lives,
            }
        )
        if not did_reset:
            raise RuntimeError("Breakout gameAPI rejected the Atari reset")
        await self._install_frame_stepper()
        await self._render()

        # DIAMOND's AtariPreprocessing performs 1..noop_max raw NOOP frames.
        reset_noop_frames = (
            random.Random(seed ^ 0xA7A71).randint(1, self.noop_max)
            if self.noop_max > 0
            else 0
        )
        if reset_noop_frames:
            await self._advance_frames(reset_noop_frames)

        state = await self._state()
        png, screenshot_advance_ms = await self._paused_frame()
        bounds = await self._canvas_bounds()

        game_state = state.get("game_state")
        metrics = state.get("metrics")
        observed_level = (
            game_state.get("level") if isinstance(game_state, dict) else None
        )
        observed_lives = metrics.get("lives") if isinstance(metrics, dict) else None
        if (
            state.get("seed") != seed
            or observed_level != level
            or observed_lives != 5
            or _is_terminal(state)
        ):
            raise RuntimeError(
                "Atari reset state is inconsistent: "
                f"seed={state.get('seed')}, level={observed_level}, "
                f"lives={observed_lives}, terminal={_is_terminal(state)}"
            )

        self.previous_state = state
        self.step_count = 0
        return BrowserObservation(
            metadata={
                "ok": True,
                "event": "reset",
                "level": level,
                "seed": seed,
                "initial_lives": 5,
                "step_count": 0,
                "reset_noop_frames": reset_noop_frames,
                "canvas_bounds": bounds,
                "timing": atari_timing(),
                "action_meanings": list(ACTION_MEANINGS),
                "screenshot_game_time_advance_ms": screenshot_advance_ms,
                "state": state,
            },
            png=png,
        )

    async def step(
        self,
        action: int,
        *,
        record_frames: bool = False,
    ) -> BrowserObservation:
        if self.previous_state is None:
            raise RuntimeError("reset must be called before the first step")
        key = action_key(action)
        manager = self._require_manager()
        if not manager.page:
            raise RuntimeError("Breakout browser page is unavailable")

        start_state = self.previous_state
        start_game_time = _game_time_ms(start_state)
        observation_png: bytes | None = None
        screenshot_advance_ms = 0.0
        frames_executed = 0
        recorded_pngs: list[bytes] = []
        recording_frames: list[dict[str, Any]] = []

        # GameWorld's existing launchNow handler is intentionally bound to
        # keyup, whereas paddle movement is bound to keydown/up. Pulse FIRE
        # before advancing the first frame so ALE action 1 takes effect in the
        # same environment step; continue holding directional actions across
        # all four frames.
        pulse_key = key == "Space"
        if pulse_key:
            await manager.page.keyboard.press(key)
        elif key is not None:
            await manager.page.keyboard.down(key)
        try:
            current_state = start_state
            for frame_index in range(1, FRAME_SKIP + 1):
                await self._advance_frames(1)
                frames_executed += 1
                current_state = await self._state()

                if record_frames:
                    recorded_png, recorded_advance_ms = await self._paused_canvas_frame()
                    recorded_pngs.append(recorded_png)
                    recording_frames.append(
                        self._recording_state(
                            current_state,
                            frame_index=frame_index,
                            screenshot_advance_ms=recorded_advance_ms,
                        )
                    )

                if _is_terminal(current_state):
                    observation_png, screenshot_advance_ms = await self._paused_frame()
                    break
                if frame_index == FRAME_SKIP:
                    observation_png, screenshot_advance_ms = await self._paused_frame()
        finally:
            if key is not None and not pulse_key:
                await manager.page.keyboard.up(key)

        if observation_png is None:
            observation_png, screenshot_advance_ms = await self._paused_frame()

        end_game_time = _game_time_ms(current_state)
        actual_advance_ms = end_game_time - start_game_time
        expected_advance_ms = frames_executed * (1000.0 / 60.0)
        if abs(actual_advance_ms - expected_advance_ms) > GAME_TIME_TOLERANCE_MS:
            raise RuntimeError(
                "Atari frame step advanced the wrong game time: "
                f"frames={frames_executed}, actual={actual_advance_ms}, "
                f"expected={expected_advance_ms}"
            )

        self.step_count += 1
        events = transition_events(start_state, current_state)
        self.previous_state = current_state
        return BrowserObservation(
            metadata={
                "ok": True,
                "event": "step",
                "action": action,
                "action_meaning": ACTION_MEANINGS[action],
                "step_count": self.step_count,
                "frames_executed": frames_executed,
                "game_time_advance_ms": actual_advance_ms,
                "timing": atari_timing(),
                "action_meanings": list(ACTION_MEANINGS),
                "screenshot_game_time_advance_ms": screenshot_advance_ms,
                "transition_events": events.to_dict(),
                "state": current_state,
                "recorded_frame_count": len(recorded_pngs),
                "recording_frame_format": (
                    "canvas_png_800x600" if record_frames else None
                ),
                "recording_frames": recording_frames,
            },
            png=observation_png,
            recorded_pngs=recorded_pngs if record_frames else None,
        )

    async def close(self) -> None:
        try:
            await self._release_all_keys()
        finally:
            if self.manager is not None:
                await self.manager.close()
                self.manager = None
            if self.launcher is not None:
                self.launcher.stop()
                self.launcher = None
            self.previous_state = None
            self._frame_stepper_installed = False


class BrowserEnvironmentRPCServer:
    def __init__(
        self,
        environment: AtariBreakoutBrowserEnvironment,
        *,
        bind: str,
    ) -> None:
        self.environment = environment
        self.bind = bind
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)

    @staticmethod
    def _encode(
        metadata: dict[str, Any],
        png: bytes | None = None,
        extra_pngs: list[bytes] | None = None,
    ) -> list[bytes]:
        payload = {"protocol_version": PROTOCOL_VERSION, **metadata}
        frames = [json.dumps(payload, separators=(",", ":")).encode("utf-8")]
        if png is not None:
            frames.append(png)
        if extra_pngs:
            frames.extend(extra_pngs)
        return frames

    async def _handle(
        self,
        request: dict[str, Any],
    ) -> tuple[list[bytes], bool]:
        command = request.get("cmd")
        if command == "health":
            return self._encode(
                {
                    "ok": True,
                    "event": "health",
                    "started": True,
                    "environment": "gameworld_deterministic_breakout",
                    "max_steps": None,
                    "initial_lives": 5,
                    "game_over_lives": 0,
                    "reset_noop_max": self.environment.noop_max,
                    "supports_level_select": True,
                    "viewport": [self.environment.width, self.environment.height],
                    "action_meanings": list(ACTION_MEANINGS),
                    "timing": atari_timing(),
                }
            ), False
        if command == "reset":
            observation = await self.environment.reset(
                level=request.get("level"),
                seed=request.get("seed"),
                initial_lives=request.get("initial_lives", 5),
            )
            return self._encode(observation.metadata, observation.png), False
        if command == "step":
            observation = await self.environment.step(request.get("action"))
            return self._encode(observation.metadata, observation.png), False
        if command == "step_record":
            observation = await self.environment.step(
                request.get("action"),
                record_frames=True,
            )
            return self._encode(
                observation.metadata,
                observation.png,
                observation.recorded_pngs,
            ), False
        if command == "close":
            return self._encode({"ok": True, "event": "close"}), True
        raise ValueError(f"Unknown command: {command!r}")

    async def serve(self) -> None:
        LOGGER.info("Atari-style Breakout service listening on %s", self.bind)
        should_close = False
        while not should_close:
            raw_frames = await self.socket.recv_multipart()
            try:
                if len(raw_frames) != 1:
                    raise ValueError("Requests must contain exactly one JSON frame")
                request = json.loads(raw_frames[0].decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("Request JSON must be an object")
                response, should_close = await self._handle(request)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Atari browser RPC request failed")
                response = self._encode(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            await self.socket.send_multipart(response)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--game-port", type=int, default=DEFAULT_GAME_PORT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    environment = AtariBreakoutBrowserEnvironment(
        game_port=args.game_port,
        headless=not args.headed,
        width=args.width,
        height=args.height,
        noop_max=args.noop_max,
    )
    server: BrowserEnvironmentRPCServer | None = None
    try:
        await environment.start()
        server = BrowserEnvironmentRPCServer(environment, bind=args.bind)
        await server.serve()
    finally:
        try:
            if server is not None:
                server.close()
        finally:
            await environment.close()


def _validate_windows_asyncio_support() -> None:
    if sys.platform == "win32" and importlib.util.find_spec("tornado") is None:
        raise RuntimeError(
            "Windows pyzmq requires tornado>=6.1 with the Proactor event loop"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _validate_windows_asyncio_support()
    try:
        asyncio.run(_run(_parse_args()))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; closing Atari-style Breakout service")


if __name__ == "__main__":
    main()

"""ZeroMQ browser environment service for DIAMOND Breakout training.

Run this module inside the GameWorld Python environment. The DIAMOND process
connects from its separate Python/CUDA environment and receives one full-size
PNG plus game-state metadata for every reset or step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import zmq
import zmq.asyncio

from env import ActionExecutor, BrowserConfig, BrowserGameManager, GameLauncher, RoleControls

from .breakout_protocol import (
    ACTION_DURATION_S,
    action_payload,
    evaluation_timing,
    execute_step_with_evaluation_cadence,
    transition_events,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_BIND = "tcp://127.0.0.1:5561"
DEFAULT_GAME_PORT = 8101
PROTOCOL_VERSION = 6
CANVAS_SELECTOR = "#canvas"
EXPECTED_CANVAS_BOUNDS = {"x": 240, "y": 17, "width": 800, "height": 600}


@dataclass(slots=True)
class BrowserObservation:
    """One browser observation transported as JSON metadata plus a PNG frame."""

    metadata: dict[str, Any]
    png: bytes


class BreakoutBrowserEnvironment:
    """Own one Chromium Breakout session with GameWorld evaluation timing."""

    def __init__(
        self,
        *,
        game_port: int = DEFAULT_GAME_PORT,
        headless: bool = True,
        width: int = 1280,
        height: int = 720,
        max_steps: int = 100,
    ) -> None:
        self.game_port = game_port
        self.headless = headless
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.launcher: GameLauncher | None = None
        self.manager: BrowserGameManager | None = None
        self.executor: ActionExecutor | None = None
        self.previous_observation_state: dict[str, Any] | None = None
        self.previous_evaluation_state: dict[str, Any] | None = None
        self.step_count = 0

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
            if not await self.manager.wait_until_actionable(stage="diamond-startup"):
                raise RuntimeError("Breakout did not become actionable during startup")
            if not self.manager.page:
                raise RuntimeError("Breakout browser page is unavailable after startup")
            self.executor = ActionExecutor(
                self.manager.page,
                controls=RoleControls(
                    allowed_keys={"ArrowLeft", "ArrowRight"},
                    hold_duration=ACTION_DURATION_S,
                    allow_clicks=False,
                ),
            )
        except Exception:
            await self.close()
            raise

    def _require_manager(self) -> BrowserGameManager:
        if self.manager is None:
            raise RuntimeError("Browser environment has not been started")
        return self.manager

    def _require_executor(self) -> ActionExecutor:
        if self.executor is None:
            raise RuntimeError("Browser action executor has not been initialized")
        return self.executor

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
            raise RuntimeError(f"Unable to locate Breakout canvas with selector {CANVAS_SELECTOR!r}")
        return {key: int(raw[key]) for key in EXPECTED_CANVAS_BOUNDS}

    @staticmethod
    def _validate_canvas_bounds(bounds: dict[str, int]) -> None:
        mismatches = {
            key: (bounds.get(key), expected)
            for key, expected in EXPECTED_CANVAS_BOUNDS.items()
            if abs(int(bounds.get(key, -10_000)) - expected) > 1
        }
        if mismatches:
            raise RuntimeError(
                "Breakout canvas geometry differs from the fixed viewport layout contract: "
                f"actual={bounds}, expected={EXPECTED_CANVAS_BOUNDS}, mismatches={mismatches}"
            )

    async def _state(self) -> dict[str, Any]:
        manager = self._require_manager()
        state = await manager.get_game_state()
        if not isinstance(state, dict):
            raise RuntimeError("Breakout gameAPI returned no state")
        return state

    async def _frame(self) -> bytes:
        return await self._require_manager().capture_screenshot_bytes()

    async def reset(
        self,
        *,
        level: int,
        seed: int,
        initial_lives: int = 3,
    ) -> BrowserObservation:
        if isinstance(level, bool) or level not in range(1, 6):
            raise ValueError(f"level must be an integer from 1 to 5, got {level!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        if (
            isinstance(initial_lives, bool)
            or not isinstance(initial_lives, int)
            or initial_lives not in range(1, 6)
        ):
            raise ValueError(
                f"initial_lives must be an integer from 1 to 5, got {initial_lives!r}"
            )

        manager = self._require_manager()
        await manager.resume_game()
        try:
            did_reset = await manager.reset_game(
                {
                    "level": level,
                    "seed": seed,
                    "initial_lives": initial_lives,
                }
            )
            if not did_reset:
                raise RuntimeError("gameAPI reset rejected the requested Breakout episode")
            if not await manager.wait_until_actionable(stage="diamond-reset"):
                raise RuntimeError("Breakout did not become actionable after reset")
            # Match Coordinator._get_raw_action: capture while running, then pause.
            png = await self._frame()
        finally:
            await manager.pause_game()

        bounds = await self._canvas_bounds()
        self._validate_canvas_bounds(bounds)
        state = await self._state()

        state_seed = state.get("seed")
        game_state = state.get("game_state")
        state_level = game_state.get("level") if isinstance(game_state, dict) else None
        metrics = state.get("metrics")
        state_lives = metrics.get("lives") if isinstance(metrics, dict) else None
        debug = state.get("debug")
        managed_boundaries = (
            debug.get("managed_task_boundaries") if isinstance(debug, dict) else None
        )
        if (
            state_seed != seed
            or state_level != level
            or state_lives != initial_lives
            or managed_boundaries is not True
        ):
            raise RuntimeError(
                "Breakout reset did not apply the requested episode: "
                f"requested=(level={level}, seed={seed}, lives={initial_lives}), "
                f"observed=(level={state_level}, seed={state_seed}, "
                f"lives={state_lives}, managed={managed_boundaries})"
            )

        self.previous_observation_state = state
        self.previous_evaluation_state = state
        self.step_count = 0
        return BrowserObservation(
            metadata={
                "ok": True,
                "event": "reset",
                "level": level,
                "seed": seed,
                "initial_lives": initial_lives,
                "step_count": self.step_count,
                "canvas_bounds": bounds,
                "timing": evaluation_timing(),
                "state": state,
            },
            png=png,
        )

    async def step(self, action: int) -> BrowserObservation:
        if self.previous_observation_state is None:
            raise RuntimeError("reset must be called before the first step")
        if self.previous_evaluation_state is None:
            raise RuntimeError("evaluation state is unavailable before the first step")

        manager = self._require_manager()
        executor = self._require_executor()
        payload = action_payload(action)
        evaluation_state, current_state, png = await execute_step_with_evaluation_cadence(
            resume_game=manager.resume_game,
            execute_action=lambda: executor.execute(payload),
            capture_evaluation_state=self._state,
            capture_frame=self._frame,
            pause_game=manager.pause_game,
            capture_observation_state=self._state,
        )

        self.step_count += 1
        events = transition_events(
            self.previous_observation_state,
            current_state,
            step_count=self.step_count,
            max_steps=self.max_steps,
        )
        evaluation_events = transition_events(
            self.previous_evaluation_state,
            evaluation_state,
            step_count=self.step_count,
            max_steps=self.max_steps,
        )
        self.previous_observation_state = current_state
        self.previous_evaluation_state = evaluation_state
        return BrowserObservation(
            metadata={
                "ok": True,
                "event": "step",
                "action": action,
                "action_payload": payload,
                "step_count": self.step_count,
                "timing": evaluation_timing(),
                "transition_events": events.to_dict(),
                "evaluation_transition_events": evaluation_events.to_dict(),
                "evaluation_state": evaluation_state,
                "state": current_state,
            },
            png=png,
        )

    async def close(self) -> None:
        if self.manager is not None:
            await self.manager.close()
            self.manager = None
        if self.launcher is not None:
            self.launcher.stop()
            self.launcher = None
        self.executor = None
        self.previous_observation_state = None
        self.previous_evaluation_state = None


class BrowserEnvironmentRPCServer:
    """Minimal request/reply transport around ``BreakoutBrowserEnvironment``."""

    def __init__(self, environment: BreakoutBrowserEnvironment, *, bind: str) -> None:
        self.environment = environment
        self.bind = bind
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)

    @staticmethod
    def _encode(metadata: dict[str, Any], png: bytes | None = None) -> list[bytes]:
        payload = {"protocol_version": PROTOCOL_VERSION, **metadata}
        frames = [json.dumps(payload, separators=(",", ":")).encode("utf-8")]
        if png is not None:
            frames.append(png)
        return frames

    async def _handle(self, request: dict[str, Any]) -> tuple[list[bytes], bool]:
        command = request.get("cmd")
        if command == "health":
            return self._encode(
                {
                    "ok": True,
                    "event": "health",
                    "started": True,
                    "max_steps": self.environment.max_steps,
                    "supports_initial_lives": True,
                    "managed_task_boundaries": True,
                    "viewport": [self.environment.width, self.environment.height],
                    "timing": evaluation_timing(),
                }
            ), False
        if command == "reset":
            observation = await self.environment.reset(
                level=request.get("level"),
                seed=request.get("seed"),
                initial_lives=request.get("initial_lives", 3),
            )
            return self._encode(observation.metadata, observation.png), False
        if command == "step":
            observation = await self.environment.step(request.get("action"))
            return self._encode(observation.metadata, observation.png), False
        if command == "close":
            return self._encode({"ok": True, "event": "close"}), True
        raise ValueError(f"Unknown command: {command!r}")

    async def serve(self) -> None:
        LOGGER.info("DIAMOND Breakout browser service listening on %s", self.bind)
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
                LOGGER.exception("Browser RPC request failed")
                response = self._encode(
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
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
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--headed", action="store_true", help="Show the Chromium window.")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    environment = BreakoutBrowserEnvironment(
        game_port=args.game_port,
        headless=not args.headed,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    server: BrowserEnvironmentRPCServer | None = None
    try:
        await environment.start()
        server = BrowserEnvironmentRPCServer(environment, bind=args.bind)
        await server.serve()
    finally:
        if server is not None:
            server.close()
        await environment.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(_parse_args()))
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; closing DIAMOND Breakout browser service")


if __name__ == "__main__":
    main()

"""End-to-end timing smoke test for deterministic GameWorld Breakout."""

from __future__ import annotations

import argparse
import asyncio
from io import BytesIO
import json
from statistics import mean
from typing import Any

from PIL import Image

from .browser_env_server import AtariBreakoutBrowserEnvironment


def _game_time_ms(metadata: dict[str, Any]) -> float:
    state = metadata.get("state")
    value = state.get("gameTimeMs") if isinstance(state, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Invalid gameTimeMs: {value!r}")
    return float(value)


def _ball(metadata: dict[str, Any]) -> dict[str, Any]:
    state = metadata.get("state")
    game_state = state.get("game_state") if isinstance(state, dict) else None
    entities = game_state.get("entities") if isinstance(game_state, dict) else None
    if not isinstance(entities, list):
        raise RuntimeError("Breakout state has no entities list")
    ball = next(
        (
            entity
            for entity in entities
            if isinstance(entity, dict) and entity.get("type") == "ball"
        ),
        None,
    )
    if not isinstance(ball, dict):
        raise RuntimeError("Breakout state has no ball entity")
    return ball


def _ball_center_rgb(
    png: bytes,
    *,
    ball: dict[str, Any],
    canvas_bounds: dict[str, int],
) -> tuple[int, int, int]:
    x = round(float(canvas_bounds["x"]) + float(ball["x"]))
    y = round(float(canvas_bounds["y"]) + float(ball["y"]))
    with Image.open(BytesIO(png)) as image:
        rgb = image.convert("RGB").getpixel((x, y))
    return tuple(map(int, rgb))


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    environment = AtariBreakoutBrowserEnvironment(
        game_port=args.game_port,
        headless=not args.headed,
        noop_max=args.noop_max,
    )
    try:
        await environment.start()
        reset = await environment.reset(
            level=args.level,
            seed=args.seed,
            initial_lives=5,
        )
        previous_time = _game_time_ms(reset.metadata)
        advances = []
        screenshot_advances = [reset.metadata["screenshot_game_time_advance_ms"]]
        canvas_bounds = reset.metadata["canvas_bounds"]
        reset_ball = _ball(reset.metadata)
        reset_countdown = (
            reset_ball.get("props", {}).get("countdown")
            if isinstance(reset_ball.get("props"), dict)
            else None
        )
        if reset_ball.get("state") != "attached" or reset_countdown is not None:
            raise RuntimeError(
                "Atari reset must leave an attached ball with no automatic "
                f"countdown: ball={reset_ball!r}"
            )

        # FIRE first, then exercise the remaining three ALE actions.
        actions = [1, 0, 2, 3]
        records = []
        first_moving_step = None
        for index in range(args.steps):
            action = actions[index % len(actions)]
            observation = await environment.step(action)
            current_time = _game_time_ms(observation.metadata)
            measured = current_time - previous_time
            expected = float(observation.metadata["game_time_advance_ms"])
            if abs(measured - expected) > 1e-3:
                raise RuntimeError(
                    f"Step timing mismatch: measured={measured}, expected={expected}"
                )
            advances.append(measured)
            screenshot_advances.append(
                float(observation.metadata["screenshot_game_time_advance_ms"])
            )
            ball = _ball(observation.metadata)
            ball_rgb = _ball_center_rgb(
                observation.png,
                ball=ball,
                canvas_bounds=canvas_bounds,
            )
            if ball.get("state") == "moving" and first_moving_step is None:
                first_moving_step = index + 1
            records.append(
                {
                    "step": index + 1,
                    "action": action,
                    "action_meaning": observation.metadata["action_meaning"],
                    "frames_executed": observation.metadata["frames_executed"],
                    "game_time_advance_ms": measured,
                    "ball_state": ball.get("state"),
                    "ball_center_rgb": list(ball_rgb),
                    "black_ball_visible": max(ball_rgb) <= 25,
                    "events": observation.metadata["transition_events"],
                }
            )
            previous_time = current_time
            if observation.metadata["transition_events"]["game_over"]:
                break

        if any(abs(value) > 1e-3 for value in screenshot_advances):
            raise RuntimeError(
                f"Paused screenshots advanced game time: {screenshot_advances}"
            )
        if not all(record["black_ball_visible"] for record in records):
            raise RuntimeError(
                "The returned final-frame observation did not preserve the black ball"
            )
        if args.steps >= 1 and first_moving_step != 1:
            raise RuntimeError(
                "FIRE did not launch the attached ball on its first four-frame "
                f"step: first_moving_step={first_moving_step}, expected=1"
            )
        return {
            "ok": True,
            "level": args.level,
            "reset_noop_frames": reset.metadata["reset_noop_frames"],
            "action_meanings": reset.metadata["action_meanings"],
            "reset_ball_state": reset_ball.get("state"),
            "reset_launch_countdown": reset_countdown,
            "timing": reset.metadata["timing"],
            "steps_completed": len(records),
            "first_moving_step": first_moving_step,
            "mean_game_time_advance_ms": mean(advances),
            "max_abs_screenshot_game_time_advance_ms": max(
                abs(value) for value in screenshot_advances
            ),
            "records": records,
        }
    finally:
        await environment.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-port", type=int, default=8291)
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(asyncio.run(_run(_parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

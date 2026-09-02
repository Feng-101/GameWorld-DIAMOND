"""Short live-Chromium contract test on the three training layouts.

The test resets Levels 1/3/4 and executes exactly three actions per level. It
does not consume held-out Level 2/5 agent rollouts or run a full task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from .browser_env_server import BreakoutBrowserEnvironment
from .breakout_protocol import evaluation_timing


def _assert_png(png: bytes, expected_size: tuple[int, int]) -> None:
    with Image.open(BytesIO(png)) as image:
        image.verify()
    with Image.open(BytesIO(png)) as image:
        if image.size != expected_size:
            raise AssertionError(
                f"Unexpected screenshot size: actual={image.size}, expected={expected_size}"
            )


async def _run(args: argparse.Namespace) -> None:
    environment = BreakoutBrowserEnvironment(
        game_port=args.game_port,
        headless=not args.headed,
        width=1280,
        height=720,
        max_steps=100,
    )
    summaries: list[dict] = []
    try:
        await environment.start()
        for level in args.levels:
            reset = await environment.reset(
                level=level,
                seed=args.seed,
                initial_lives=args.initial_lives,
            )
            _assert_png(reset.png, (1280, 720))
            if level == 1 and args.export_frame is not None:
                export_path = Path(args.export_frame).expanduser().resolve()
                export_path.parent.mkdir(parents=True, exist_ok=True)
                export_path.write_bytes(reset.png)
            reset_state = reset.metadata["state"]
            if reset.metadata.get("timing") != evaluation_timing():
                raise AssertionError(f"Unexpected reset timing contract: {reset.metadata.get('timing')}")
            reported_level = reset_state.get("game_state", {}).get("level")
            if reported_level != level:
                raise AssertionError(
                    f"Reset selected wrong level: requested={level}, reported={reported_level}"
                )
            reported_lives = reset_state.get("metrics", {}).get("lives")
            if reported_lives != args.initial_lives:
                raise AssertionError(
                    "Reset selected wrong life budget: "
                    f"requested={args.initial_lives}, reported={reported_lives}"
                )

            previous_game_time_ms = reset_state.get("gameTimeMs")
            action_results = []
            for action in (0, 1, 2):
                observation = await environment.step(action)
                _assert_png(observation.png, (1280, 720))
                if observation.metadata["step_count"] != action + 1:
                    raise AssertionError("Step counter did not advance exactly once per action")
                if observation.metadata.get("timing") != evaluation_timing():
                    raise AssertionError(
                        f"Unexpected step timing contract: {observation.metadata.get('timing')}"
                    )
                current_game_time_ms = observation.metadata["state"].get("gameTimeMs")
                evaluation_game_time_ms = observation.metadata["evaluation_state"].get(
                    "gameTimeMs"
                )
                game_time_delta_ms = None
                evaluation_to_observation_ms = None
                if isinstance(previous_game_time_ms, (int, float)) and isinstance(
                    current_game_time_ms, (int, float)
                ):
                    game_time_delta_ms = current_game_time_ms - previous_game_time_ms
                    if game_time_delta_ms < 230:
                        raise AssertionError(
                            "Game advanced less than the expected 0.2-second action plus "
                            f"0.05-second post-action interval: delta_ms={game_time_delta_ms}"
                        )
                if isinstance(evaluation_game_time_ms, (int, float)) and isinstance(
                    current_game_time_ms, (int, float)
                ):
                    evaluation_to_observation_ms = (
                        current_game_time_ms - evaluation_game_time_ms
                    )
                    if evaluation_to_observation_ms < 30:
                        raise AssertionError(
                            "Observation boundary did not follow the evaluator state by the "
                            "expected 0.05-second running interval: "
                            f"delta_ms={evaluation_to_observation_ms}"
                        )
                previous_game_time_ms = current_game_time_ms
                transition_events = observation.metadata["transition_events"]
                action_results.append(
                    {
                        "action": action,
                        "game_time_delta_ms": game_time_delta_ms,
                        "evaluation_to_observation_ms": evaluation_to_observation_ms,
                        "score_delta": transition_events["score_delta"],
                        "positive_score_delta": transition_events[
                            "positive_score_delta"
                        ],
                        "bricks_destroyed": transition_events["bricks_destroyed"],
                        "life_lost": transition_events["life_lost"],
                        "last_life_reset": transition_events["last_life_reset"],
                        "task_success": transition_events["task_success"],
                        "task_time_limit": transition_events["task_time_limit"],
                    }
                )

            summaries.append(
                {
                    "level": level,
                    "seed": args.seed,
                    "initial_lives": args.initial_lives,
                    "canvas_bounds": reset.metadata["canvas_bounds"],
                    "timing": reset.metadata["timing"],
                    "actions": action_results,
                }
            )
    finally:
        await environment.close()

    print(json.dumps({"ok": True, "levels": summaries}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-port", type=int, default=8101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--initial-lives", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--export-frame",
        help="Optionally save the level-1 reset frame as a full 1280x720 PNG.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

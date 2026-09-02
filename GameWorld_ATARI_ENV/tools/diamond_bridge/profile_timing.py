"""Measure deterministic Atari-style browser steps.

Wall-clock latency may vary with screenshot cost, but every non-terminal
transition must advance exactly four 60 Hz game frames and screenshot capture
must advance zero game time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any

from .browser_env_server import AtariBreakoutBrowserEnvironment


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    environment = AtariBreakoutBrowserEnvironment(
        game_port=args.game_port,
        headless=not args.headed,
        width=1280,
        height=720,
        noop_max=args.noop_max,
    )
    records: list[dict[str, Any]] = []
    try:
        await environment.start()
        reset = await environment.reset(
            level=args.level,
            seed=args.seed,
            initial_lives=5,
        )
        for step in range(1, args.steps + 1):
            started = perf_counter()
            observation = await environment.step(args.action)
            wall_ms = (perf_counter() - started) * 1000
            metadata = observation.metadata
            records.append(
                {
                    "step": step,
                    "wall_clock_ms": wall_ms,
                    "frames_executed": metadata["frames_executed"],
                    "game_time_advance_ms": metadata["game_time_advance_ms"],
                    "screenshot_game_time_advance_ms": metadata[
                        "screenshot_game_time_advance_ms"
                    ],
                    "transition_events": metadata["transition_events"],
                }
            )
            if (
                metadata["transition_events"]["game_over"]
                or metadata["transition_events"]["level_cleared"]
            ):
                break

        result = {
            "ok": True,
            "config": {
                "level": args.level,
                "seed": args.seed,
                "action": args.action,
                "steps_requested": args.steps,
                "noop_max": args.noop_max,
            },
            "reset_noop_frames": reset.metadata["reset_noop_frames"],
            "summary": {
                "wall_clock_ms": summarize(
                    [record["wall_clock_ms"] for record in records]
                ),
                "game_time_advance_ms": summarize(
                    [record["game_time_advance_ms"] for record in records]
                ),
                "screenshot_game_time_advance_ms": summarize(
                    [
                        record["screenshot_game_time_advance_ms"]
                        for record in records
                    ]
                ),
            },
            "records": records,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return result
    finally:
        await environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-port", type=int, default=8291)
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--action", type=int, choices=range(4), default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(asyncio.run(run(parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

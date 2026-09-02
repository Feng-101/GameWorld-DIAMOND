"""Verify GameWorld Breakout's two physical terminal boundaries in Chromium.

This diagnostic directly triggers the browser engine's existing lose-ball and
win-level paths. It does not replace the ordinary action-path smoke test; it
isolates the two rare boundaries so they can be checked without playing a full
game.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .breakout_protocol import transition_events
from .browser_env_server import AtariBreakoutBrowserEnvironment


def _terminal(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("terminal")
    if not isinstance(value, dict):
        raise RuntimeError("gameAPI state is missing terminal metadata")
    return value


def _ball(state: dict[str, Any]) -> dict[str, Any]:
    game_state = state.get("game_state")
    entities = game_state.get("entities") if isinstance(game_state, dict) else None
    if not isinstance(entities, list):
        raise RuntimeError("gameAPI state is missing ball entities")
    value = next(
        (
            entity
            for entity in entities
            if isinstance(entity, dict) and entity.get("type") == "ball"
        ),
        None,
    )
    if not isinstance(value, dict):
        raise RuntimeError("gameAPI state is missing the ball")
    return value


async def _force_life_loss(
    environment: AtariBreakoutBrowserEnvironment,
) -> dict[str, Any]:
    manager = environment._require_manager()
    if manager.page is None:
        raise RuntimeError("Chromium page is unavailable")

    previous = await environment._state()
    triggered = await manager.page.evaluate(
        """() => {
            const game = window.__breakoutGame;
            if (!game || typeof game.loseBall !== "function") return false;
            game.loseBall();
            return true;
        }"""
    )
    if triggered is not True:
        raise RuntimeError("Could not trigger Breakout.loseBall()")
    await environment._render()
    current = await environment._state()
    return {
        "state": current,
        "events": transition_events(previous, current).to_dict(),
    }


async def _force_level_clear(
    environment: AtariBreakoutBrowserEnvironment,
) -> dict[str, Any]:
    manager = environment._require_manager()
    if manager.page is None:
        raise RuntimeError("Chromium page is unavailable")

    previous = await environment._state()
    triggered = await manager.page.evaluate(
        """() => {
            const game = window.__breakoutGame;
            if (
                !game ||
                !game.court ||
                !Array.isArray(game.court.bricks) ||
                typeof game.winLevel !== "function"
            ) {
                return false;
            }
            for (const brick of game.court.bricks) brick.hit = true;
            game.court.numhits = game.court.numbricks;
            game.court.rerender = true;
            game.winLevel();
            return true;
        }"""
    )
    if triggered is not True:
        raise RuntimeError("Could not trigger Breakout.winLevel()")
    await environment._render()
    current = await environment._state()
    return {
        "state": current,
        "events": transition_events(previous, current).to_dict(),
    }


async def _run(args: argparse.Namespace) -> None:
    environment = AtariBreakoutBrowserEnvironment(
        game_port=args.game_port,
        headless=not args.headed,
        noop_max=30,
    )
    report: dict[str, Any] = {
        "ok": False,
        "level": args.level,
        "seed": args.seed,
        "life_losses": [],
    }
    try:
        await environment.start()
        await environment.reset(
            level=args.level,
            seed=args.seed,
            initial_lives=5,
        )

        for loss_index in range(1, 6):
            result = await _force_life_loss(environment)
            state = result["state"]
            events = result["events"]
            lives = state["metrics"]["lives"]
            terminal = _terminal(state)
            expected_terminal = loss_index == 5
            if lives != 5 - loss_index:
                raise RuntimeError(
                    f"Life loss {loss_index} produced {lives} lives"
                )
            if events["life_lost"] is not True:
                raise RuntimeError(f"Life loss {loss_index} was not reported")
            if bool(events["game_over"]) != expected_terminal:
                raise RuntimeError(
                    f"Life loss {loss_index} game_over={events['game_over']}"
                )
            if bool(terminal.get("isTerminal")) != expected_terminal:
                raise RuntimeError(
                    f"Life loss {loss_index} terminal={terminal!r}"
                )
            ball = _ball(state)
            record = {
                "loss_index": loss_index,
                "lives": lives,
                "terminal": terminal,
                "events": events,
                "score": state["game_state"]["score"],
                "bricks_remaining": state["metrics"]["bricks_remaining"],
                "ball_state_after_loss": ball.get("state"),
                "launch_countdown_after_loss": ball.get("props", {}).get(
                    "countdown"
                ),
            }
            if (
                ball.get("state") != "attached"
                or record["launch_countdown_after_loss"] is not None
            ):
                raise RuntimeError(
                    "Every lost ball must stop in an attached, no-countdown "
                    f"state: ball={ball!r}"
                )
            if not expected_terminal:
                # The direct engine mutation above is now the browser service's
                # transition origin. One FIRE step must relaunch immediately.
                environment.previous_state = state
                relaunched = await environment.step(1)
                relaunched_ball = _ball(relaunched.metadata["state"])
                if relaunched_ball.get("state") != "moving":
                    raise RuntimeError(
                        f"FIRE did not relaunch after life loss: {relaunched_ball!r}"
                    )
                record["fire_relaunch_state"] = relaunched_ball.get("state")
                record["fire_relaunch_frames"] = relaunched.metadata[
                    "frames_executed"
                ]
            report["life_losses"].append(record)

        fifth_state = report["life_losses"][-1]
        if fifth_state["terminal"].get("outcome") != "fail":
            raise RuntimeError("Fifth life loss is not a failure terminal")
        if fifth_state["terminal"].get("reason") != "no_lives_left":
            raise RuntimeError("Fifth life loss has the wrong terminal reason")

        await environment.reset(
            level=args.level,
            seed=args.seed,
            initial_lives=5,
        )
        clear_result = await _force_level_clear(environment)
        clear_state = clear_result["state"]
        clear_events = clear_result["events"]
        clear_terminal = _terminal(clear_state)
        if clear_state["metrics"]["lives"] != 5:
            raise RuntimeError("Level clear incorrectly changed the life count")
        if clear_state["metrics"]["bricks_remaining"] != 0:
            raise RuntimeError("Level clear did not preserve the cleared board")
        if clear_events["level_cleared"] is not True:
            raise RuntimeError("Level clear event was not reported")
        if clear_events["game_over"] is not False:
            raise RuntimeError("Level clear was incorrectly encoded as game-over")
        if (
            clear_terminal.get("isTerminal") is not True
            or clear_terminal.get("outcome") != "success"
            or clear_terminal.get("reason") != "level_cleared"
        ):
            raise RuntimeError(f"Invalid level-clear terminal: {clear_terminal!r}")

        report["level_clear"] = {
            "lives": clear_state["metrics"]["lives"],
            "terminal": clear_terminal,
            "events": clear_events,
            "score": clear_state["game_state"]["score"],
            "bricks_remaining": clear_state["metrics"]["bricks_remaining"],
            "completion_progress": clear_state["game_state"][
                "completion_progress"
            ],
        }
        report["ok"] = True
    finally:
        await environment.close()

    print(json.dumps(report, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-port", type=int, default=8292)
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()

"""Pure Breakout transition semantics shared by the browser RPC service tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar


ACTION_DURATION_S = 0.2
POST_ACTION_IDLE_S = 0.05
NOMINAL_OBSERVATION_INTERVAL_S = ACTION_DURATION_S + POST_ACTION_IDLE_S
ACTION_MAP: dict[int, dict[str, Any]] = {
    0: {"action": "wait", "duration": ACTION_DURATION_S},
    1: {"action": "press_key", "key": "ArrowLeft", "duration": ACTION_DURATION_S},
    2: {"action": "press_key", "key": "ArrowRight", "duration": ACTION_DURATION_S},
}

EvaluationStateT = TypeVar("EvaluationStateT")
ObservationStateT = TypeVar("ObservationStateT")
FrameT = TypeVar("FrameT")


@dataclass(frozen=True, slots=True)
class TransitionEvents:
    """Policy-neutral events derived from two browser game states.

    The browser service reports facts.  The DIAMOND wrapper decides whether a
    life loss is a logical training boundary and when a physical reset is
    required, because those choices intentionally differ between train/test.
    """

    score_delta: float
    positive_score_delta: float
    bricks_destroyed: int
    task_success: bool
    task_time_limit: bool
    brick_hit: bool
    life_lost: bool
    last_life_reset: bool
    terminal_failure: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def action_payload(action: int) -> dict[str, Any]:
    """Translate the three-action DIAMOND space to a legal GameWorld action."""
    if isinstance(action, bool) or not isinstance(action, int) or action not in ACTION_MAP:
        raise ValueError(f"Breakout action must be one of {sorted(ACTION_MAP)}, got {action!r}")
    return dict(ACTION_MAP[action])


def evaluation_timing() -> dict[str, Any]:
    """Return the machine-checkable GameWorld observation cadence contract."""
    return {
        "action_hold_s": ACTION_DURATION_S,
        "post_action_idle_s": POST_ACTION_IDLE_S,
        "nominal_observation_interval_s": NOMINAL_OBSERVATION_INTERVAL_S,
        "evaluation_state_after_action": True,
        "screenshot_before_pause": True,
        "observation_state_after_pause": True,
    }


async def execute_step_with_evaluation_cadence(
    *,
    resume_game: Callable[[], Awaitable[None]],
    execute_action: Callable[[], Awaitable[None]],
    capture_evaluation_state: Callable[[], Awaitable[EvaluationStateT]],
    capture_frame: Callable[[], Awaitable[FrameT]],
    pause_game: Callable[[], Awaitable[None]],
    capture_observation_state: Callable[[], Awaitable[ObservationStateT]],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[EvaluationStateT, ObservationStateT, FrameT]:
    """Execute one action using GameWorld's fixed evaluation-loop timing.

    The official Breakout action is held for 0.2 seconds by ``execute_action``.
    It reads evaluator state immediately after the action, leaves the game
    running for its fixed 0.05-second loop delay, captures the next screenshot
    while the game is still running, and only then pauses for inference.  We
    reproduce that observable ordering exactly.  A second state read after the
    pause supplies an unambiguous transition boundary for training rewards.
    """
    await resume_game()
    try:
        await execute_action()
        evaluation_state = await capture_evaluation_state()
        await sleep(POST_ACTION_IDLE_S)
        frame = await capture_frame()
    finally:
        await pause_game()
    observation_state = await capture_observation_state()
    return evaluation_state, observation_state, frame


def _nested(state: dict[str, Any] | None, *path: str) -> Any:
    value: Any = state
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _required_number(state: dict[str, Any], *path: str) -> float:
    value = _number(_nested(state, *path))
    if value is None:
        raise ValueError(f"Breakout state is missing numeric field {'.'.join(path)!r}")
    return value


def transition_events(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    step_count: int,
    max_steps: int = 100,
) -> TransitionEvents:
    """Extract native score/life/task events without choosing RL boundaries."""
    if step_count < 1:
        raise ValueError("step_count must be at least 1 after an executed action")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    previous_level = _required_number(previous, "game_state", "level")
    current_level = _required_number(current, "game_state", "level")
    previous_bricks = _required_number(previous, "metrics", "bricks_remaining")
    current_bricks = _required_number(current, "metrics", "bricks_remaining")
    previous_lives = _required_number(previous, "metrics", "lives")
    current_lives = _required_number(current, "metrics", "lives")
    previous_score = _required_number(previous, "game_state", "score")
    current_score = _required_number(current, "game_state", "score")

    terminal = _nested(current, "terminal")
    terminal_success = bool(
        isinstance(terminal, dict)
        and terminal.get("isTerminal") is True
        and terminal.get("outcome") == "success"
    )
    terminal_failure = bool(
        isinstance(terminal, dict)
        and terminal.get("isTerminal") is True
        and terminal.get("outcome") == "fail"
    )
    level_changed = current_level != previous_level
    task_success = terminal_success or level_changed

    bricks_destroyed = max(0, int(round(previous_bricks - current_bricks)))
    brick_hit = bricks_destroyed > 0
    ordinary_life_loss = current_lives < previous_lives
    last_life_reset = bool(
        previous_lives == 1
        and current_lives >= 3
        and current_level == previous_level
    )
    life_lost = ordinary_life_loss or last_life_reset
    score_delta = current_score - previous_score

    return TransitionEvents(
        score_delta=score_delta,
        positive_score_delta=max(0.0, score_delta),
        bricks_destroyed=bricks_destroyed,
        task_success=task_success,
        task_time_limit=not task_success and step_count >= max_steps,
        brick_hit=brick_hit,
        life_lost=life_lost,
        last_life_reset=last_life_reset,
        terminal_failure=terminal_failure,
    )


__all__ = [
    "ACTION_DURATION_S",
    "ACTION_MAP",
    "NOMINAL_OBSERVATION_INTERVAL_S",
    "POST_ACTION_IDLE_S",
    "TransitionEvents",
    "action_payload",
    "evaluation_timing",
    "execute_step_with_evaluation_cadence",
    "transition_events",
]

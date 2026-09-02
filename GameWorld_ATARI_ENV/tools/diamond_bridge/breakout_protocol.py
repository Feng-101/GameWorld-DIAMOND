"""GameWorld Breakout actions with deterministic Atari-like step semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FRAME_RATE_HZ = 60
FRAME_SKIP = 4
FRAME_DURATION_S = 1.0 / FRAME_RATE_HZ
GAME_TIME_PER_STEP_S = FRAME_SKIP * FRAME_DURATION_S
# Match ALE Breakout's reduced four-action surface. The browser keeps its
# native left/right physics; Space calls the existing launchNow() path.
ACTION_MEANINGS = ("NOOP", "FIRE", "RIGHT", "LEFT")
ACTION_KEYS: dict[int, str | None] = {
    0: None,
    1: "Space",
    2: "ArrowRight",
    3: "ArrowLeft",
}


@dataclass(frozen=True, slots=True)
class TransitionEvents:
    """Facts produced by one deterministic four-frame emulator step."""

    score_delta: float
    positive_score_delta: float
    bricks_destroyed: int
    brick_hit: bool
    life_lost: bool
    game_over: bool
    level_cleared: bool
    terminal_failure: bool
    terminal_success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def action_key(action: int) -> str | None:
    """Map an ALE-compatible Breakout action to its browser key."""
    if isinstance(action, bool) or not isinstance(action, int) or action not in ACTION_KEYS:
        raise ValueError(
            f"Breakout action must be one of {sorted(ACTION_KEYS)}, got {action!r}"
        )
    return ACTION_KEYS[action]


def atari_timing() -> dict[str, Any]:
    """Machine-checkable deterministic frame-step contract."""
    return {
        # Atari-like fixed-step logic, applied to the native GameWorld game.
        "mode": "deterministic_gameworld_frames",
        "frame_rate_hz": FRAME_RATE_HZ,
        "frame_skip": FRAME_SKIP,
        "game_time_per_step_s": GAME_TIME_PER_STEP_S,
        "action_repeat_frames": FRAME_SKIP,
        "screenshots_while_paused": True,
        # The GameWorld ball is black on a light background. Pixelwise max
        # pooling would select the lighter old background and erase the ball.
        "observation_frame": "last_executed_frame",
        "max_pool_last_two_frames": False,
        "wall_clock_sleep_s": 0.0,
    }


def _nested(state: dict[str, Any] | None, *path: str) -> Any:
    value: Any = state
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(state: dict[str, Any], *path: str) -> float:
    value = _nested(state, *path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Breakout state is missing numeric field {'.'.join(path)!r}"
        )
    return float(value)


def transition_events(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> TransitionEvents:
    """Extract Atari-like reward, life loss, game-over, and clear events."""
    previous_level = _number(previous, "game_state", "level")
    current_level = _number(current, "game_state", "level")
    if current_level != previous_level:
        raise ValueError(
            "Atari-style single-level Breakout must never auto-advance levels: "
            f"previous={previous_level}, current={current_level}"
        )

    previous_bricks = _number(previous, "metrics", "bricks_remaining")
    current_bricks = _number(current, "metrics", "bricks_remaining")
    previous_lives = _number(previous, "metrics", "lives")
    current_lives = _number(current, "metrics", "lives")
    previous_score = _number(previous, "game_state", "score")
    current_score = _number(current, "game_state", "score")

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
    level_cleared = terminal_success or current_bricks == 0
    game_over = terminal_failure or current_lives <= 0
    if level_cleared and game_over:
        raise ValueError("Breakout cannot clear the level and lose its last life together")

    score_delta = current_score - previous_score
    if score_delta < 0:
        raise ValueError(
            "Atari-style score must not reset inside a physical game: "
            f"previous={previous_score}, current={current_score}"
        )

    bricks_destroyed = max(0, int(round(previous_bricks - current_bricks)))
    life_lost = current_lives < previous_lives
    if game_over and current_lives != 0:
        raise ValueError(
            f"Game-over must preserve the zero-life terminal state, got {current_lives}"
        )
    if game_over and not life_lost:
        raise ValueError("Game-over must also report the fifth life loss")

    return TransitionEvents(
        score_delta=score_delta,
        positive_score_delta=score_delta,
        bricks_destroyed=bricks_destroyed,
        brick_hit=bricks_destroyed > 0,
        life_lost=life_lost,
        game_over=game_over,
        level_cleared=level_cleared,
        terminal_failure=terminal_failure,
        terminal_success=terminal_success,
    )


__all__ = [
    "ACTION_KEYS",
    "ACTION_MEANINGS",
    "FRAME_DURATION_S",
    "FRAME_RATE_HZ",
    "FRAME_SKIP",
    "GAME_TIME_PER_STEP_S",
    "TransitionEvents",
    "action_key",
    "atari_timing",
    "transition_events",
]

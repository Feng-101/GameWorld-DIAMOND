"""Atari-semantics DIAMOND environment backed by deterministic browser Breakout.

This environment is intentionally independent from ``GameWorldBreakoutEnv``.
The browser service is an emulator-like synchronous process: one action
advances at most four 60 Hz game frames, screen capture advances zero game
time, a game starts with five lives, and only game-over or level-clear causes a
physical reset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import gymnasium
import numpy as np
import torch
from torch import Tensor

from integrations.gameworld import (
    AtariBreakoutRPCClient,
    AtariRPCObservation,
    EXPECTED_ACTION_MEANINGS,
    EXPECTED_ATARI_TIMING,
    frame_to_tensor,
)


class AtariBreakoutClient(Protocol):
    def health(self) -> dict[str, Any]: ...

    def reset(
        self,
        *,
        level: int,
        seed: int,
        initial_lives: int,
    ) -> AtariRPCObservation: ...

    def step(self, action: int) -> AtariRPCObservation: ...

    def close(self) -> None: ...


class GameWorldAtariBreakoutEnv:
    """Single-browser Torch environment with DIAMOND Atari episode semantics."""

    # Match ALE Breakout: NOOP, FIRE, RIGHT, LEFT.
    num_actions = 4

    def __init__(
        self,
        *,
        device: torch.device,
        endpoint: str = "tcp://127.0.0.1:5661",
        num_envs: int = 1,
        size: int = 64,
        max_episode_steps: None = None,
        levels: Sequence[int] = (5,),
        game_seeds: Sequence[int] | None = None,
        initial_lives: int = 5,
        max_task_life_losses: int = 5,
        done_on_life_loss: bool = True,
        timeout_ms: int = 30_000,
        client: AtariBreakoutClient | None = None,
        check_health: bool = True,
        validate_service_config: bool = True,
    ) -> None:
        if num_envs != 1:
            raise ValueError(
                "Atari-style browser Breakout owns one Chromium session and "
                "requires num_envs=1"
            )
        if size < 1:
            raise ValueError("size must be positive")
        if max_episode_steps is not None:
            raise ValueError(
                "Atari-style browser Breakout has no step limit; "
                "max_episode_steps must be null"
            )
        if initial_lives != 5 or max_task_life_losses != 5:
            raise ValueError(
                "Atari-style browser Breakout must start with five lives and "
                "reach game-over on the fifth life loss"
            )
        if not isinstance(done_on_life_loss, bool):
            raise ValueError("done_on_life_loss must be a boolean")

        normalized_levels: list[int] = []
        for level in levels:
            if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
                raise ValueError(f"Breakout level must be an integer, got {level!r}")
            normalized_level = int(level)
            if normalized_level not in range(1, 6):
                raise ValueError(
                    f"Breakout level must be between 1 and 5, got {level!r}"
                )
            if normalized_level not in normalized_levels:
                normalized_levels.append(normalized_level)
        if len(normalized_levels) != 1:
            raise ValueError(
                "Atari-style specialist training requires exactly one level; "
                f"received={normalized_levels}"
            )

        normalized_game_seeds: tuple[int, ...] | None = None
        if game_seeds is not None:
            seeds: list[int] = []
            for game_seed in game_seeds:
                if isinstance(game_seed, bool) or not isinstance(
                    game_seed, (int, np.integer)
                ):
                    raise ValueError(
                        f"Browser game seeds must be integers, got {game_seed!r}"
                    )
                seeds.append(int(game_seed) & 0xFFFFFFFF)
            if not seeds:
                raise ValueError("game_seeds must be non-empty when configured")
            normalized_game_seeds = tuple(seeds)

        self.device = device
        self.num_envs = 1
        self.size = int(size)
        self.max_episode_steps = None
        self.level = normalized_levels[0]
        self.levels = (self.level,)
        self.game_seeds = normalized_game_seeds
        self.initial_lives = 5
        self.max_task_life_losses = 5
        self.done_on_life_loss = done_on_life_loss
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 3, self.size, self.size),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(self.num_actions)

        self._client = client or AtariBreakoutRPCClient(
            endpoint,
            timeout_ms=timeout_ms,
        )
        self._rng = np.random.default_rng()
        self._game_seed_cycle_index = 0
        self._task_step = 0
        self._task_seed: int | None = None
        self._task_best_progress = 0.0
        self._task_native_return = 0.0
        self._task_lives_lost = 0
        self._closed = False

        if check_health:
            try:
                self._validate_health(
                    self._client.health(),
                    strict=validate_service_config,
                )
            except Exception:
                self.close()
                raise

    @staticmethod
    def _single_int(value: Any, *, name: str) -> int:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain exactly one value")
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(f"{name} must contain exactly one value")
            value = value.item()
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if len(value) != 1:
                raise ValueError(f"{name} must contain exactly one value")
            value = value[0]

        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)

    @staticmethod
    def _validate_health(health: dict[str, Any], *, strict: bool) -> None:
        if health.get("environment") != "gameworld_deterministic_breakout":
            raise RuntimeError(
                "Connected service is not the independent deterministic "
                "GameWorld Breakout "
                f"environment: {health.get('environment')!r}"
            )
        if not strict:
            return

        expected = {
            "max_steps": None,
            "initial_lives": 5,
            "game_over_lives": 0,
            "reset_noop_max": 30,
            "viewport": [1280, 720],
            "action_meanings": EXPECTED_ACTION_MEANINGS,
            "timing": EXPECTED_ATARI_TIMING,
        }
        mismatches = {
            key: {"received": health.get(key), "expected": expected_value}
            for key, expected_value in expected.items()
            if health.get(key) != expected_value
        }
        if mismatches:
            raise RuntimeError(
                "Atari-style browser service contract mismatch: "
                f"{mismatches}"
            )

    def _next_game_seed(self) -> int:
        if self.game_seeds is not None:
            seed = self.game_seeds[
                self._game_seed_cycle_index % len(self.game_seeds)
            ]
            self._game_seed_cycle_index += 1
            return seed
        return int(self._rng.integers(0, 2**32, dtype=np.uint64))

    @staticmethod
    def _completion_progress(metadata: dict[str, Any]) -> float:
        state = metadata.get("state")
        game_state = state.get("game_state") if isinstance(state, dict) else None
        progress = (
            game_state.get("completion_progress")
            if isinstance(game_state, dict)
            else None
        )
        if (
            isinstance(progress, bool)
            or not isinstance(progress, (int, float))
            or not np.isfinite(progress)
            or not 0.0 <= float(progress) <= 1.0
        ):
            raise RuntimeError(
                f"Atari-style browser returned invalid completion_progress: {progress!r}"
            )
        return float(progress)

    @staticmethod
    def _state_lives(metadata: dict[str, Any]) -> int:
        state = metadata.get("state")
        metrics = state.get("metrics") if isinstance(state, dict) else None
        lives = metrics.get("lives") if isinstance(metrics, dict) else None
        if (
            isinstance(lives, bool)
            or not isinstance(lives, (int, float))
            or int(lives) != lives
            or int(lives) not in range(0, 6)
        ):
            raise RuntimeError(
                f"Atari-style browser returned invalid lives: {lives!r}"
            )
        return int(lives)

    @staticmethod
    def _transition_events(metadata: dict[str, Any]) -> dict[str, Any]:
        events = metadata.get("transition_events")
        if not isinstance(events, dict):
            raise RuntimeError(
                "Atari-style browser response is missing transition_events"
            )

        for name in (
            "brick_hit",
            "life_lost",
            "game_over",
            "level_cleared",
            "terminal_failure",
            "terminal_success",
        ):
            if not isinstance(events.get(name), bool):
                raise RuntimeError(
                    f"Atari-style transition event {name!r} must be boolean"
                )

        for name in ("score_delta", "positive_score_delta"):
            value = events.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or float(value) < 0
            ):
                raise RuntimeError(
                    f"Atari-style transition event {name!r} is invalid: {value!r}"
                )

        if not np.isclose(
            float(events["score_delta"]),
            float(events["positive_score_delta"]),
        ):
            raise RuntimeError("Atari-style positive reward must equal score delta")

        bricks_destroyed = events.get("bricks_destroyed")
        if (
            isinstance(bricks_destroyed, bool)
            or not isinstance(bricks_destroyed, int)
            or bricks_destroyed < 0
        ):
            raise RuntimeError(
                "Atari-style bricks_destroyed must be a non-negative integer"
            )
        if events["brick_hit"] != (bricks_destroyed > 0):
            raise RuntimeError("Atari-style brick event fields are inconsistent")
        if events["game_over"] and not events["life_lost"]:
            raise RuntimeError("Game-over must include the fifth life-loss event")
        if events["game_over"] != events["terminal_failure"]:
            raise RuntimeError("Game-over and terminal failure must be identical")
        if events["level_cleared"] != events["terminal_success"]:
            raise RuntimeError("Level-clear and terminal success must be identical")
        if events["game_over"] and events["level_cleared"]:
            raise RuntimeError(
                "Level-clear and game-over are mutually exclusive terminal states"
            )
        return events

    def _to_observation(self, response: AtariRPCObservation) -> Tensor:
        return frame_to_tensor(
            response.png,
            device=self.device,
            size=self.size,
        )

    def _episode_boundary_payload(
        self,
        *,
        boundary_reason: str,
        physical_reset: bool,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, float]]]:
        if self._task_seed is None:
            raise RuntimeError("Browser task seed is unavailable at episode boundary")

        success = boundary_reason == "level_cleared"
        game_over = boundary_reason == "game_over"
        episode_info = {
            "level": torch.tensor([self.level], dtype=torch.int64),
            "game_seed": torch.tensor([self._task_seed], dtype=torch.int64),
            "best_completion_progress": torch.tensor(
                [self._task_best_progress],
                dtype=torch.float32,
            ),
            "task_success": torch.tensor([success], dtype=torch.uint8),
            "game_over": torch.tensor([game_over], dtype=torch.uint8),
            "physical_reset": torch.tensor([physical_reset], dtype=torch.uint8),
            "initial_lives": torch.tensor([5], dtype=torch.int64),
            "task_steps": torch.tensor([self._task_step], dtype=torch.int64),
            "task_native_return": torch.tensor(
                [self._task_native_return],
                dtype=torch.float32,
            ),
            "lives_lost": torch.tensor(
                [self._task_lives_lost],
                dtype=torch.int64,
            ),
        }
        episode_metrics = {
            # Reuse the trainer's stable validation schema so existing
            # validation_metrics.jsonl tooling works for both browser backends.
            "gameworld/level": float(self.level),
            "gameworld/game_seed": float(self._task_seed),
            "gameworld/best_completion_progress": self._task_best_progress,
            "gameworld/success": float(success),
            "gameworld/game_over": float(game_over),
            "gameworld/physical_reset": float(physical_reset),
            "gameworld/initial_lives": 5.0,
            "gameworld/task_steps": float(self._task_step),
            "gameworld/task_native_return": self._task_native_return,
            "gameworld/lives_lost": float(self._task_lives_lost),
        }
        return [episode_info], [episode_metrics]

    def _reset_task(self, *, game_seed: int | None = None) -> tuple[Tensor, dict[str, Any]]:
        seed = self._next_game_seed() if game_seed is None else game_seed
        response = self._client.reset(
            level=self.level,
            seed=seed,
            initial_lives=5,
        )
        metadata = response.metadata
        if (
            metadata.get("level") != self.level
            or metadata.get("seed") != seed
            or metadata.get("initial_lives") != 5
            or metadata.get("step_count") != 0
            or self._state_lives(metadata) != 5
        ):
            raise RuntimeError(
                "Atari-style browser reset does not match the requested game: "
                f"requested=(level={self.level}, seed={seed}, lives=5), "
                f"received=(level={metadata.get('level')}, "
                f"seed={metadata.get('seed')}, "
                f"lives={self._state_lives(metadata)}, "
                f"step={metadata.get('step_count')})"
            )
        if metadata.get("timing") != EXPECTED_ATARI_TIMING:
            raise RuntimeError("Atari reset violated the deterministic timing contract")
        if metadata.get("action_meanings") != EXPECTED_ACTION_MEANINGS:
            raise RuntimeError("Atari reset returned the wrong action meanings")
        reset_noop_frames = metadata.get("reset_noop_frames")
        if (
            isinstance(reset_noop_frames, bool)
            or not isinstance(reset_noop_frames, int)
            or reset_noop_frames not in range(1, 31)
        ):
            raise RuntimeError(
                "Atari reset must execute 1..30 hidden raw NOOP frames, "
                f"got {reset_noop_frames!r}"
            )
        if abs(float(metadata.get("screenshot_game_time_advance_ms", np.inf))) > 1e-3:
            raise RuntimeError("Atari reset screenshot advanced game time")

        self._task_step = 0
        self._task_seed = seed
        self._task_best_progress = self._completion_progress(metadata)
        self._task_native_return = 0.0
        self._task_lives_lost = 0
        return self._to_observation(response), {
            "gameworld_atari": metadata,
            "level": self.level,
            "seed": seed,
            "initial_lives": 5,
        }

    def reset(
        self,
        *,
        seed: int | Sequence[int] | np.ndarray | Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Atari-style browser Breakout environment is closed")

        gym_seed: int | None = None
        if seed is not None:
            gym_seed = self._single_int(seed, name="seed") & 0xFFFFFFFF
            self._rng = np.random.default_rng(gym_seed)
            self._game_seed_cycle_index = 0
        if options and options.get("level") is not None:
            requested_level = self._single_int(options["level"], name="level")
            if requested_level != self.level:
                raise ValueError(
                    f"This specialist is fixed to level {self.level}, "
                    f"not level {requested_level}"
                )
        if options and options.get("game_seed") is not None:
            gym_seed = (
                self._single_int(options["game_seed"], name="game_seed")
                & 0xFFFFFFFF
            )

        # A configured validation seed schedule takes precedence over Gym's
        # collector seed, exactly as in the earlier deterministic evaluation
        # grid. Without a schedule, the first reset uses Gym's seed directly.
        game_seed = None if self.game_seeds is not None else gym_seed
        return self._reset_task(game_seed=game_seed)

    def step(
        self,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Atari-style browser Breakout environment is closed")

        action = self._single_int(actions, name="action")
        if action not in range(self.num_actions):
            raise ValueError(f"action must be 0, 1, 2 or 3, got {action}")

        response = self._client.step(action)
        transition_observation = self._to_observation(response)
        metadata = response.metadata
        self._task_step += 1

        if metadata.get("step_count") != self._task_step:
            raise RuntimeError(
                "Atari browser and DIAMOND step counters diverged: "
                f"server={metadata.get('step_count')}, "
                f"diamond={self._task_step}"
            )
        if metadata.get("timing") != EXPECTED_ATARI_TIMING:
            raise RuntimeError("Atari step violated the deterministic timing contract")
        if metadata.get("action_meanings") != EXPECTED_ACTION_MEANINGS:
            raise RuntimeError("Atari step returned the wrong action meanings")
        if metadata.get("action") != action:
            raise RuntimeError(
                f"Atari service executed action {metadata.get('action')}, "
                f"requested {action}"
            )
        if abs(float(metadata.get("screenshot_game_time_advance_ms", np.inf))) > 1e-3:
            raise RuntimeError("Atari step screenshot advanced game time")

        frames_executed = metadata.get("frames_executed")
        if (
            isinstance(frames_executed, bool)
            or not isinstance(frames_executed, int)
            or frames_executed not in range(1, 5)
        ):
            raise RuntimeError(
                f"Atari service returned invalid frame count: {frames_executed!r}"
            )
        expected_advance_ms = frames_executed * (1000.0 / 60.0)
        game_time_advance_ms = metadata.get("game_time_advance_ms")
        if (
            isinstance(game_time_advance_ms, bool)
            or not isinstance(game_time_advance_ms, (int, float))
            or not np.isclose(
                float(game_time_advance_ms),
                expected_advance_ms,
                atol=1e-3,
                rtol=0.0,
            )
        ):
            raise RuntimeError(
                "Atari service advanced the wrong amount of game time: "
                f"frames={frames_executed}, "
                f"advance={game_time_advance_ms!r}"
            )

        events = self._transition_events(metadata)
        lives = self._state_lives(metadata)
        if events["game_over"] and lives != 0:
            raise RuntimeError("Game-over must preserve a zero-life terminal frame")
        if events["level_cleared"] and lives <= 0:
            raise RuntimeError("Level-clear must not be encoded as life exhaustion")

        reward_value = float(events["positive_score_delta"])
        self._task_native_return += reward_value
        self._task_best_progress = max(
            self._task_best_progress,
            self._completion_progress(metadata),
        )
        if events["life_lost"]:
            self._task_lives_lost += 1
        if lives != 5 - self._task_lives_lost:
            raise RuntimeError(
                "Browser lives and observed life-loss events diverged: "
                f"lives={lives}, life_losses={self._task_lives_lost}"
            )

        level_cleared = bool(events["level_cleared"])
        game_over = bool(events["game_over"])
        life_boundary = (
            self.done_on_life_loss
            and bool(events["life_lost"])
            and not level_cleared
            and not game_over
        )
        physical_reset = level_cleared or game_over
        terminated = physical_reset or life_boundary

        reward = torch.tensor(
            [reward_value],
            dtype=torch.float32,
            device=self.device,
        )
        end = torch.tensor(
            [terminated],
            dtype=torch.uint8,
            device=self.device,
        )
        # There is deliberately no time-limit truncation in this environment.
        trunc = torch.zeros(1, dtype=torch.uint8, device=self.device)
        info: dict[str, Any] = {
            "gameworld_atari": metadata,
            "level": self.level,
            "game_seed": self._task_seed,
            "task_step": self._task_step,
            "task_best_completion_progress": self._task_best_progress,
        }

        if physical_reset:
            boundary_reason = "level_cleared" if level_cleared else "game_over"
            final_observation = transition_observation
            episode_info, episode_metrics = self._episode_boundary_payload(
                boundary_reason=boundary_reason,
                physical_reset=True,
            )
            next_observation, reset_info = self._reset_task()
            info.update(
                {
                    "final_observation": final_observation,
                    "final_gameworld_atari": metadata,
                    "reset_info": reset_info,
                    "boundary_reason": boundary_reason,
                    "physical_reset": True,
                    "episode_info": episode_info,
                    "episode_metrics": episode_metrics,
                }
            )
        elif life_boundary:
            # DIAMOND's Atari DoneOnLifeLoss wrapper exposes a logical end after
            # a lost life, but it neither resets ALE nor fabricates a new frame.
            episode_info, episode_metrics = self._episode_boundary_payload(
                boundary_reason="life_loss",
                physical_reset=False,
            )
            next_observation = transition_observation
            info.update(
                {
                    "final_observation": transition_observation,
                    "final_gameworld_atari": metadata,
                    "boundary_reason": "life_loss",
                    "physical_reset": False,
                    "episode_info": episode_info,
                    "episode_metrics": episode_metrics,
                }
            )
        else:
            next_observation = transition_observation

        return next_observation, reward, end, trunc, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()


__all__ = ["GameWorldAtariBreakoutEnv"]

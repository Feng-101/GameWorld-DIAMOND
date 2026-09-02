"""DIAMOND-compatible real environment backed by GameWorld Breakout."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import gymnasium
import numpy as np
import torch
from torch import Tensor

from integrations.gameworld import (
    EXPECTED_EVALUATION_TIMING,
    BreakoutRPCClient,
    RPCObservation,
    frame_to_tensor,
)


class BreakoutClient(Protocol):
    def health(self) -> dict[str, Any]: ...
    def reset(
        self,
        *,
        level: int,
        seed: int,
        initial_lives: int,
    ) -> RPCObservation: ...
    def step(self, action: int) -> RPCObservation: ...
    def close(self) -> None: ...


class GameWorldBreakoutEnv:
    """A single-vector Torch environment matching DIAMOND's collector contract."""

    num_actions = 3

    def __init__(
        self,
        *,
        device: torch.device,
        endpoint: str = "tcp://127.0.0.1:5561",
        num_envs: int = 1,
        size: int = 64,
        max_episode_steps: int = 100,
        levels: Sequence[int] = (1, 2, 3, 4, 5),
        level_strategy: str = "random",
        game_seeds: Sequence[int] | None = None,
        initial_lives: int = 3,
        max_task_life_losses: int | None = None,
        done_on_life_loss: bool = True,
        timeout_ms: int = 30_000,
        client: BreakoutClient | None = None,
        check_health: bool = True,
        validate_service_config: bool = True,
    ) -> None:
        if num_envs != 1:
            raise ValueError(
                "GameWorld Breakout currently owns one Chromium session and requires num_envs=1"
            )
        if size < 1:
            raise ValueError("size must be positive")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if level_strategy not in {"random", "cycle", "shuffled_cycle"}:
            raise ValueError(
                "level_strategy must be 'random', 'cycle' or 'shuffled_cycle'"
            )
        if not isinstance(done_on_life_loss, bool):
            raise ValueError("done_on_life_loss must be a boolean")
        if (
            isinstance(initial_lives, bool)
            or not isinstance(initial_lives, (int, np.integer))
            or int(initial_lives) not in range(1, 6)
        ):
            raise ValueError("initial_lives must be an integer from 1 to 5")
        if max_task_life_losses is not None:
            if (
                isinstance(max_task_life_losses, bool)
                or not isinstance(max_task_life_losses, (int, np.integer))
                or int(max_task_life_losses) < 1
            ):
                raise ValueError("max_task_life_losses must be null or a positive integer")
            if int(max_task_life_losses) != int(initial_lives):
                raise ValueError(
                    "max_task_life_losses must equal initial_lives so the physical "
                    "task ends exactly when its configured lives are exhausted"
                )

        normalized_levels = []
        for level in levels:
            if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
                raise ValueError(f"Breakout levels must be integers, got {level!r}")
            normalized_level = int(level)
            if normalized_level not in range(1, 6):
                raise ValueError(f"Breakout levels must be between 1 and 5, got {level!r}")
            if normalized_level not in normalized_levels:
                normalized_levels.append(normalized_level)
        if not normalized_levels:
            raise ValueError("At least one Breakout level is required")

        normalized_game_seeds: list[int] | None = None
        if game_seeds is not None:
            normalized_game_seeds = []
            for game_seed in game_seeds:
                if isinstance(game_seed, bool) or not isinstance(
                    game_seed, (int, np.integer)
                ):
                    raise ValueError(
                        f"GameWorld game seeds must be integers, got {game_seed!r}"
                    )
                normalized_game_seeds.append(int(game_seed) & 0xFFFFFFFF)
            if not normalized_game_seeds:
                raise ValueError("game_seeds must be non-empty when configured")

        self.device = device
        self.num_envs = 1
        self.size = size
        self.max_episode_steps = max_episode_steps
        self.levels = tuple(normalized_levels)
        self.level_strategy = level_strategy
        self.game_seeds = (
            tuple(normalized_game_seeds) if normalized_game_seeds is not None else None
        )
        self.initial_lives = int(initial_lives)
        self.max_task_life_losses = (
            int(max_task_life_losses)
            if max_task_life_losses is not None
            else None
        )
        self.done_on_life_loss = done_on_life_loss
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 3, size, size),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(self.num_actions)

        self._client = client or BreakoutRPCClient(endpoint, timeout_ms=timeout_ms)
        self._rng = np.random.default_rng()
        self._cycle_index = 0
        self._shuffled_levels: list[int] = []
        self._game_seed_cycle_index = 0
        # This is the physical GameWorld task counter.  A training-only life
        # boundary resets DIAMOND's recurrent state but deliberately does not
        # reset this counter or the browser game.
        self._task_step = 0
        self._task_level: int | None = None
        self._task_seed: int | None = None
        self._task_best_progress = 0.0
        self._task_native_return = 0.0
        self._task_lives_lost = 0
        self._closed = False
        if check_health:
            health = self._client.health()
            server_max_steps = health.get("max_steps")
            if (
                validate_service_config
                and server_max_steps is not None
                and server_max_steps != self.max_episode_steps
            ):
                self.close()
                raise RuntimeError(
                    "GameWorld client/server episode limits differ: "
                    f"client={self.max_episode_steps}, server={server_max_steps}"
                )
            viewport = health.get("viewport")
            if (
                validate_service_config
                and viewport is not None
                and viewport != [1280, 720]
            ):
                self.close()
                raise RuntimeError(
                    "GameWorld browser viewport violates the observation crop contract: "
                    f"received={viewport}, expected={[1280, 720]}"
                )
            timing = health.get("timing")
            if validate_service_config and timing != EXPECTED_EVALUATION_TIMING:
                self.close()
                raise RuntimeError(
                    "GameWorld browser timing violates the evaluation cadence contract: "
                    f"received={timing!r}, expected={EXPECTED_EVALUATION_TIMING!r}"
                )
            if validate_service_config and health.get("supports_initial_lives") is not True:
                self.close()
                raise RuntimeError(
                    "GameWorld browser service does not support per-task initial_lives"
                )
            if validate_service_config and health.get("managed_task_boundaries") is not True:
                self.close()
                raise RuntimeError(
                    "GameWorld browser service can leak native next-level frames at "
                    "success boundaries"
                )

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
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) != 1:
                raise ValueError(f"{name} must contain exactly one value")
            value = value[0]

        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)

    def _choose_level(self, forced_level: int | None = None) -> int:
        if forced_level is not None:
            if forced_level not in self.levels:
                raise ValueError(
                    f"Requested level {forced_level} is outside configured levels {self.levels}"
                )
            return forced_level
        if self.level_strategy == "random":
            return int(self._rng.choice(self.levels))
        if self.level_strategy == "cycle":
            level = self.levels[self._cycle_index % len(self.levels)]
            self._cycle_index += 1
            return level

        # Exact balance without a predictable 1->3->4 ordering: every block
        # contains each training level once, in a newly shuffled order.
        if not self._shuffled_levels:
            self._shuffled_levels = list(self.levels)
            self._rng.shuffle(self._shuffled_levels)
        return self._shuffled_levels.pop()

    def _next_game_seed(self) -> int:
        if self.game_seeds is not None:
            seed = self.game_seeds[self._game_seed_cycle_index % len(self.game_seeds)]
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
                f"GameWorld returned invalid completion_progress: {progress!r}"
            )
        return float(progress)

    def _episode_boundary_payload(
        self,
        *,
        task_success: bool,
        life_budget_exhausted: bool,
        physical_reset: bool,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, float]]]:
        if self._task_level is None or self._task_seed is None:
            raise RuntimeError("GameWorld task metadata is unavailable at episode boundary")

        episode_info = {
            "level": torch.tensor([self._task_level], dtype=torch.int64),
            "game_seed": torch.tensor([self._task_seed], dtype=torch.int64),
            "best_completion_progress": torch.tensor(
                [self._task_best_progress], dtype=torch.float32
            ),
            "task_success": torch.tensor([task_success], dtype=torch.uint8),
            "life_budget_exhausted": torch.tensor(
                [life_budget_exhausted], dtype=torch.uint8
            ),
            "physical_reset": torch.tensor([physical_reset], dtype=torch.uint8),
            "initial_lives": torch.tensor([self.initial_lives], dtype=torch.int64),
            "task_steps": torch.tensor([self._task_step], dtype=torch.int64),
            "task_native_return": torch.tensor(
                [self._task_native_return], dtype=torch.float32
            ),
            "lives_lost": torch.tensor([self._task_lives_lost], dtype=torch.int64),
        }
        episode_metrics = {
            "gameworld/level": float(self._task_level),
            "gameworld/game_seed": float(self._task_seed),
            "gameworld/best_completion_progress": self._task_best_progress,
            "gameworld/success": float(task_success),
            "gameworld/life_budget_exhausted": float(life_budget_exhausted),
            "gameworld/physical_reset": float(physical_reset),
            "gameworld/initial_lives": float(self.initial_lives),
            "gameworld/task_steps": float(self._task_step),
            "gameworld/task_native_return": self._task_native_return,
            "gameworld/lives_lost": float(self._task_lives_lost),
        }
        return [episode_info], [episode_metrics]

    def _to_observation(self, response: RPCObservation) -> Tensor:
        return frame_to_tensor(response.png, device=self.device, size=self.size)

    def _reset_task(
        self,
        *,
        game_seed: int | None = None,
        forced_level: int | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        level = self._choose_level(forced_level)
        seed = self._next_game_seed() if game_seed is None else game_seed
        response = self._client.reset(
            level=level,
            seed=seed,
            initial_lives=self.initial_lives,
        )
        observation = self._to_observation(response)
        metadata = response.metadata
        state = metadata.get("state")
        metrics = state.get("metrics") if isinstance(state, dict) else None
        received_lives = metrics.get("lives") if isinstance(metrics, dict) else None
        if (
            metadata.get("level") != level
            or metadata.get("seed") != seed
            or metadata.get("initial_lives") != self.initial_lives
            or received_lives != self.initial_lives
        ):
            raise RuntimeError(
                "GameWorld reset response does not match the requested episode: "
                f"requested=(level={level}, seed={seed}, lives={self.initial_lives}), "
                f"received=(level={metadata.get('level')}, "
                f"seed={metadata.get('seed')}, lives={received_lives})"
            )
        self._task_step = 0
        self._task_level = level
        self._task_seed = seed
        self._task_best_progress = self._completion_progress(metadata)
        self._task_native_return = 0.0
        self._task_lives_lost = 0
        return observation, {
            "gameworld": metadata,
            "level": level,
            "seed": seed,
            "initial_lives": self.initial_lives,
        }

    def reset(
        self,
        *,
        seed: int | Sequence[int] | np.ndarray | Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("GameWorld Breakout environment is closed")

        game_seed = None
        if seed is not None:
            game_seed = self._single_int(seed, name="seed") & 0xFFFFFFFF
            self._rng = np.random.default_rng(game_seed)
            # A seeded Gym reset starts a new reproducible scheduling run. In
            # particular, each fixed validation schedule must replay the same
            # (level, game_seed) grid even though the vector-style env auto-reset
            # has already prepared one unused task at the previous boundary.
            self._cycle_index = 0
            self._shuffled_levels.clear()
            self._game_seed_cycle_index = 0

        # With an explicit evaluation seed schedule, Gym's reset seed controls
        # only environment-side randomness; browser task seeds come from the
        # configured cycle.  Without a schedule, preserve the original
        # convention that the first browser reset uses Gym's seed directly.
        if self.game_seeds is not None:
            game_seed = None

        forced_level = None
        if options and options.get("level") is not None:
            forced_level = self._single_int(options["level"], name="level")
        if options and options.get("game_seed") is not None:
            game_seed = self._single_int(options["game_seed"], name="game_seed") & 0xFFFFFFFF
        return self._reset_task(game_seed=game_seed, forced_level=forced_level)

    @staticmethod
    def _transition_events(metadata: dict[str, Any]) -> dict[str, Any]:
        events = metadata.get("transition_events")
        if not isinstance(events, dict):
            raise RuntimeError("GameWorld response is missing transition_events")

        for name in (
            "task_success",
            "task_time_limit",
            "life_lost",
            "last_life_reset",
            "brick_hit",
            "terminal_failure",
        ):
            if not isinstance(events.get(name), bool):
                raise RuntimeError(f"GameWorld transition event {name!r} must be boolean")

        reward = events.get("positive_score_delta")
        score_delta = events.get("score_delta")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not np.isfinite(reward)
            or float(reward) < 0
        ):
            raise RuntimeError(f"GameWorld returned invalid positive_score_delta: {reward!r}")
        if (
            isinstance(score_delta, bool)
            or not isinstance(score_delta, (int, float))
            or not np.isfinite(score_delta)
            or not np.isclose(float(reward), max(0.0, float(score_delta)))
        ):
            raise RuntimeError(
                "GameWorld score deltas are inconsistent: "
                f"score_delta={score_delta!r}, positive_score_delta={reward!r}"
            )

        bricks_destroyed = events.get("bricks_destroyed")
        if (
            isinstance(bricks_destroyed, bool)
            or not isinstance(bricks_destroyed, int)
            or bricks_destroyed < 0
        ):
            raise RuntimeError(
                f"GameWorld returned invalid bricks_destroyed: {bricks_destroyed!r}"
            )
        if events["brick_hit"] != (bricks_destroyed > 0):
            raise RuntimeError(
                "GameWorld brick events are inconsistent: "
                f"brick_hit={events['brick_hit']!r}, bricks_destroyed={bricks_destroyed!r}"
            )
        if events["last_life_reset"] and not events["life_lost"]:
            raise RuntimeError("GameWorld last_life_reset must also report life_lost")
        if events["task_success"] and events["task_time_limit"]:
            raise RuntimeError("GameWorld task cannot be both successful and time-limited")
        return events

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("GameWorld Breakout environment is closed")

        action = self._single_int(actions, name="action")
        if action not in range(self.num_actions):
            raise ValueError(f"action must be 0, 1 or 2, got {action}")

        response = self._client.step(action)
        transition_observation = self._to_observation(response)
        metadata = response.metadata
        self._task_step += 1

        server_step = metadata.get("step_count")
        if isinstance(server_step, bool) or not isinstance(server_step, int):
            raise RuntimeError(f"GameWorld returned invalid step_count: {server_step!r}")
        if server_step != self._task_step:
            raise RuntimeError(
                "GameWorld and DIAMOND step counters diverged: "
                f"server={server_step}, diamond={self._task_step}"
            )

        events = self._transition_events(metadata)
        reward_value = float(events["positive_score_delta"])
        self._task_native_return += reward_value
        self._task_best_progress = max(
            self._task_best_progress,
            self._completion_progress(metadata),
        )
        if events["life_lost"]:
            self._task_lives_lost += 1
        task_success = events["task_success"]
        task_time_limit = (
            events["task_time_limit"] or self._task_step >= self.max_episode_steps
        ) and not task_success
        life_budget_exhausted = (
            self.max_task_life_losses is not None
            and self._task_lives_lost >= self.max_task_life_losses
            and not task_success
            and not task_time_limit
        )
        if (
            events["terminal_failure"]
            and self.max_task_life_losses is not None
            and not life_budget_exhausted
            and not task_time_limit
        ):
            raise RuntimeError(
                "Breakout exhausted its engine lives before DIAMOND's configured "
                f"life budget: observed={self._task_lives_lost}, "
                f"configured={self.max_task_life_losses}"
            )
        life_boundary = (
            self.done_on_life_loss
            and events["life_lost"]
            and not task_success
            and not task_time_limit
            and not life_budget_exhausted
        )
        terminated = task_success or life_boundary or life_budget_exhausted
        truncated = task_time_limit
        physical_reset = task_success or task_time_limit or life_budget_exhausted

        reward = torch.tensor([float(reward_value)], dtype=torch.float32, device=self.device)
        end = torch.tensor([terminated], dtype=torch.uint8, device=self.device)
        trunc = torch.tensor([truncated], dtype=torch.uint8, device=self.device)
        info: dict[str, Any] = {
            "gameworld": metadata,
            "level": self._task_level,
            "game_seed": self._task_seed,
            "task_step": self._task_step,
            "task_best_completion_progress": self._task_best_progress,
        }

        if physical_reset:
            final_observation = transition_observation
            final_metadata = metadata
            episode_info, episode_metrics = self._episode_boundary_payload(
                task_success=task_success,
                life_budget_exhausted=life_budget_exhausted,
                physical_reset=True,
            )
            next_observation, reset_info = self._reset_task()
            info.update(
                {
                    "final_observation": final_observation,
                    "final_gameworld": final_metadata,
                    "reset_info": reset_info,
                    "boundary_reason": (
                        "task_success"
                        if task_success
                        else (
                            "task_time_limit"
                            if task_time_limit
                            else "life_budget_exhausted"
                        )
                    ),
                    "physical_reset": True,
                    "episode_info": episode_info,
                    "episode_metrics": episode_metrics,
                }
            )
        elif life_boundary:
            # Match DIAMOND's Atari DoneOnLifeLoss wrapper: expose a logical end
            # and reset the agent LSTM, but continue from the same browser frame.
            episode_info, episode_metrics = self._episode_boundary_payload(
                task_success=False,
                life_budget_exhausted=False,
                physical_reset=False,
            )
            next_observation = transition_observation
            info.update(
                {
                    "final_observation": transition_observation,
                    "final_gameworld": metadata,
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


__all__ = ["GameWorldBreakoutEnv"]

from __future__ import annotations

import sys
import unittest
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from envs.factory import (  # noqa: E402
    GAMEWORLD_ATARI_BREAKOUT_KIND,
    validate_env_setup,
)
from envs.gameworld_atari_breakout import GameWorldAtariBreakoutEnv  # noqa: E402
from integrations.gameworld import (  # noqa: E402
    AtariRPCObservation,
    EXPECTED_ACTION_MEANINGS,
    EXPECTED_ATARI_TIMING,
)


def make_frame(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (1280, 720), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((240, 17, 1039, 616), fill=color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeAtariClient:
    def __init__(self, transitions: list[dict] | None = None) -> None:
        self.transitions = transitions or []
        self.reset_calls: list[tuple[int, int, int]] = []
        self.actions: list[int] = []
        self.level = 5
        self.seed = 0
        self.lives = 5
        self.step_count = 0
        self.score = 0.0
        self.progress = 0.0
        self.closed = False

    def health(self) -> dict:
        return {
            "ok": True,
            "event": "health",
            "environment": "gameworld_deterministic_breakout",
            "max_steps": None,
            "initial_lives": 5,
            "game_over_lives": 0,
            "reset_noop_max": 30,
            "viewport": [1280, 720],
            "action_meanings": list(EXPECTED_ACTION_MEANINGS),
            "timing": dict(EXPECTED_ATARI_TIMING),
        }

    def _metadata(self, event: str, **extra) -> dict:
        return {
            "ok": True,
            "event": event,
            "level": self.level,
            "seed": self.seed,
            "initial_lives": 5,
            "step_count": self.step_count,
            "action_meanings": list(EXPECTED_ACTION_MEANINGS),
            "timing": dict(EXPECTED_ATARI_TIMING),
            "screenshot_game_time_advance_ms": 0.0,
            "state": {
                "game_state": {
                    "level": self.level,
                    "score": self.score,
                    "completion_progress": self.progress,
                },
                "metrics": {
                    "lives": self.lives,
                    "bricks_remaining": int(round(100 * (1 - self.progress))),
                    "bricks_total": 100,
                },
            },
            **extra,
        }

    def reset(
        self,
        *,
        level: int,
        seed: int,
        initial_lives: int,
    ) -> AtariRPCObservation:
        self.reset_calls.append((level, seed, initial_lives))
        self.level = level
        self.seed = seed
        self.lives = 5
        self.step_count = 0
        self.score = 0.0
        self.progress = 0.0
        return AtariRPCObservation(
            metadata=self._metadata("reset", reset_noop_frames=7),
            png=make_frame((level * 20, 0, 0)),
        )

    def step(self, action: int) -> AtariRPCObservation:
        self.actions.append(action)
        transition = (
            self.transitions.pop(0)
            if self.transitions
            else {}
        )
        self.step_count += 1
        score_delta = float(transition.get("score_delta", 0.0))
        life_lost = bool(transition.get("life_lost", False))
        game_over = bool(transition.get("game_over", False))
        level_cleared = bool(transition.get("level_cleared", False))
        if life_lost:
            self.lives -= 1
        if game_over:
            self.lives = 0
        self.score += score_delta
        if level_cleared:
            self.progress = 1.0
        elif score_delta > 0:
            self.progress = min(0.99, self.progress + 0.01)
        bricks_destroyed = int(transition.get("bricks_destroyed", score_delta > 0))
        events = {
            "score_delta": score_delta,
            "positive_score_delta": score_delta,
            "bricks_destroyed": bricks_destroyed,
            "brick_hit": bricks_destroyed > 0,
            "life_lost": life_lost,
            "game_over": game_over,
            "level_cleared": level_cleared,
            "terminal_failure": game_over,
            "terminal_success": level_cleared,
        }
        frames_executed = int(transition.get("frames_executed", 4))
        return AtariRPCObservation(
            metadata=self._metadata(
                "step",
                action=action,
                action_meaning=EXPECTED_ACTION_MEANINGS[action],
                frames_executed=frames_executed,
                game_time_advance_ms=frames_executed * 1000 / 60,
                transition_events=events,
            ),
            png=make_frame((0, min(self.step_count * 20, 255), 0)),
        )

    def close(self) -> None:
        self.closed = True


class AtariBrowserEnvTests(unittest.TestCase):
    def make_env(
        self,
        client: FakeAtariClient,
        **overrides,
    ) -> GameWorldAtariBreakoutEnv:
        kwargs = {
            "device": torch.device("cpu"),
            "client": client,
            "levels": (5,),
            "max_episode_steps": None,
            "initial_lives": 5,
            "max_task_life_losses": 5,
        }
        kwargs.update(overrides)
        return GameWorldAtariBreakoutEnv(**kwargs)

    def test_reset_and_ale_four_action_contract(self) -> None:
        client = FakeAtariClient()
        env = self.make_env(client)
        observation, info = env.reset(seed=[42])
        self.assertEqual(tuple(observation.shape), (1, 3, 64, 64))
        self.assertEqual(info["level"], 5)
        self.assertEqual(info["seed"], 42)
        self.assertEqual(env.num_actions, 4)
        self.assertIsNone(env.max_episode_steps)
        for action in range(4):
            _, _, end, trunc, _ = env.step(torch.tensor([action]))
            self.assertEqual(end.tolist(), [0])
            self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(client.actions, [0, 1, 2, 3])

    def test_training_life_loss_is_only_a_logical_boundary(self) -> None:
        client = FakeAtariClient([{"life_lost": True}])
        env = self.make_env(client, done_on_life_loss=True)
        env.reset(seed=[11])
        observation, reward, end, trunc, info = env.step(torch.tensor([0]))
        self.assertEqual(reward.tolist(), [0.0])
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(info["boundary_reason"], "life_loss")
        self.assertFalse(info["physical_reset"])
        self.assertTrue(torch.equal(observation, info["final_observation"]))
        self.assertEqual(client.reset_calls, [(5, 11, 5)])

    def test_evaluation_life_loss_continues_same_game(self) -> None:
        client = FakeAtariClient([{"life_lost": True}])
        env = self.make_env(client, done_on_life_loss=False)
        env.reset(seed=[12])
        _, _, end, trunc, info = env.step(torch.tensor([0]))
        self.assertEqual(end.tolist(), [0])
        self.assertEqual(trunc.tolist(), [0])
        self.assertNotIn("final_observation", info)
        self.assertEqual(client.reset_calls, [(5, 12, 5)])

    def test_fifth_life_loss_is_game_over_and_physically_resets(self) -> None:
        client = FakeAtariClient(
            [{"life_lost": True} for _ in range(4)]
            + [{"life_lost": True, "game_over": True}]
        )
        env = self.make_env(client, done_on_life_loss=True)
        env.reset(seed=[13])
        for _ in range(4):
            _, _, end, trunc, info = env.step(torch.tensor([1]))
            self.assertEqual(end.tolist(), [1])
            self.assertEqual(trunc.tolist(), [0])
            self.assertEqual(info["boundary_reason"], "life_loss")
            self.assertFalse(info["physical_reset"])

        _, _, end, trunc, info = env.step(torch.tensor([1]))
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(info["boundary_reason"], "game_over")
        self.assertTrue(info["physical_reset"])
        self.assertEqual(info["episode_info"][0]["game_over"].tolist(), [1])
        self.assertEqual(info["episode_info"][0]["task_success"].tolist(), [0])
        self.assertEqual(info["episode_metrics"][0]["gameworld/game_over"], 1.0)
        self.assertEqual(len(client.reset_calls), 2)
        self.assertEqual(client.reset_calls[1][0], 5)
        self.assertEqual(client.reset_calls[1][2], 5)

    def test_level_clear_is_distinct_success_terminal(self) -> None:
        client = FakeAtariClient(
            [
                {
                    "score_delta": 40,
                    "bricks_destroyed": 1,
                    "level_cleared": True,
                }
            ]
        )
        env = self.make_env(client, done_on_life_loss=False)
        env.reset(seed=[14])
        _, reward, end, trunc, info = env.step(torch.tensor([2]))
        self.assertEqual(reward.tolist(), [40.0])
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(info["boundary_reason"], "level_cleared")
        self.assertEqual(info["episode_info"][0]["task_success"].tolist(), [1])
        self.assertEqual(info["episode_info"][0]["game_over"].tolist(), [0])

    def test_no_local_step_limit_or_truncation(self) -> None:
        client = FakeAtariClient()
        env = self.make_env(client, done_on_life_loss=False)
        env.reset(seed=[15])
        for _ in range(25):
            _, _, end, trunc, _ = env.step(torch.tensor([0]))
            self.assertEqual(end.tolist(), [0])
            self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(len(client.reset_calls), 1)

    def test_rejects_cap_multiple_levels_and_invalid_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "no step limit"):
            self.make_env(FakeAtariClient(), max_episode_steps=500)
        with self.assertRaisesRegex(ValueError, "exactly one level"):
            self.make_env(FakeAtariClient(), levels=(1, 5))
        env = self.make_env(FakeAtariClient())
        env.reset(seed=[1])
        with self.assertRaisesRegex(ValueError, "0, 1, 2 or 3"):
            env.step(torch.tensor([4]))


class AtariConfigTests(unittest.TestCase):
    @staticmethod
    def split(endpoint: str, *, training: bool) -> dict:
        return {
            "endpoint": endpoint,
            "size": 64,
            "max_episode_steps": None,
            "initial_lives": 5,
            "max_task_life_losses": 5,
            "levels": [5],
            "done_on_life_loss": training,
        }

    def test_valid_single_level_no_cap_setup(self) -> None:
        kind = validate_env_setup(
            GAMEWORLD_ATARI_BREAKOUT_KIND,
            train=self.split("tcp://127.0.0.1:5661", training=True),
            test=self.split("tcp://127.0.0.1:5662", training=False),
            train_num_envs=1,
            test_num_envs=1,
            model_free=False,
            heldout_test=None,
            protocol="atari_single_level",
        )
        self.assertEqual(kind, GAMEWORLD_ATARI_BREAKOUT_KIND)

    def test_rejects_step_cap_and_different_test_level(self) -> None:
        train = self.split("tcp://127.0.0.1:5661", training=True)
        test = self.split("tcp://127.0.0.1:5662", training=False)
        train["max_episode_steps"] = 500
        with self.assertRaisesRegex(ValueError, "no train or test step limit"):
            validate_env_setup(
                GAMEWORLD_ATARI_BREAKOUT_KIND,
                train=train,
                test=test,
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                protocol="atari_single_level",
            )


if __name__ == "__main__":
    unittest.main()

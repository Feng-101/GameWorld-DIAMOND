from __future__ import annotations

import json
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from integrations.gameworld.preprocess import (
    DEFAULT_CANVAS_CROP,
    frame_to_tensor,
    preprocess_gameworld_frame,
)
from integrations.gameworld.rpc_client import (
    EXPECTED_EVALUATION_TIMING,
    PROTOCOL_VERSION,
    BreakoutRPCClient,
    BreakoutRPCError,
    RPCObservation,
)

_ENV_SPEC = spec_from_file_location(
    "_gameworld_breakout_env_under_test",
    REPO_ROOT / "src" / "envs" / "gameworld_breakout.py",
)
if _ENV_SPEC is None or _ENV_SPEC.loader is None:
    raise RuntimeError("Unable to load GameWorldBreakoutEnv for tests")
_ENV_MODULE = module_from_spec(_ENV_SPEC)
_ENV_SPEC.loader.exec_module(_ENV_MODULE)
GameWorldBreakoutEnv = _ENV_MODULE.GameWorldBreakoutEnv


def make_frame(canvas_color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (1280, 720), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((240, 17, 1039, 616), fill=canvas_color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeBreakoutClient:
    def __init__(
        self,
        *,
        terminate_at: int | None = None,
        life_loss_at: int | None = None,
        life_loss_steps: set[int] | None = None,
        score_deltas: dict[int, float] | None = None,
    ) -> None:
        self.terminate_at = terminate_at
        self.life_loss_at = life_loss_at
        self.life_loss_steps = (
            set(life_loss_steps)
            if life_loss_steps is not None
            else ({life_loss_at} if life_loss_at is not None else set())
        )
        self.score_deltas = score_deltas or {}
        self.reset_calls: list[tuple[int, int, int]] = []
        self.actions: list[int] = []
        self.step_count = 0
        self.current_level = 1
        self.current_lives = 3
        self.closed = False

    def health(self) -> dict:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "event": "health",
            "timing": dict(EXPECTED_EVALUATION_TIMING),
            "supports_initial_lives": True,
            "managed_task_boundaries": True,
        }

    def reset(self, *, level: int, seed: int, initial_lives: int) -> RPCObservation:
        self.reset_calls.append((level, seed, initial_lives))
        self.step_count = 0
        self.current_level = level
        self.current_lives = initial_lives
        return RPCObservation(
            metadata={
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "event": "reset",
                "level": level,
                "seed": seed,
                "initial_lives": initial_lives,
                "step_count": 0,
                "state": {
                    "game_state": {
                        "level": level,
                        "completion_progress": 0.0,
                    },
                    "metrics": {"lives": self.current_lives},
                },
            },
            png=make_frame((level * 20, 0, 0)),
        )

    def step(self, action: int) -> RPCObservation:
        self.actions.append(action)
        self.step_count += 1
        task_success = self.terminate_at == self.step_count
        life_lost = self.step_count in self.life_loss_steps
        if life_lost:
            self.current_lives -= 1
        last_life_reset = life_lost and self.current_lives == 0
        if last_life_reset:
            # The real Breakout engine resets lives synchronously on the last
            # miss. The DIAMOND env must count the event, then perform its own
            # physical task reset before another action is issued.
            self.current_lives = 3
        score_delta = float(
            self.score_deltas.get(self.step_count, 35.0 if task_success else 0.0)
        )
        transition_events = {
            "score_delta": score_delta,
            "positive_score_delta": max(0.0, score_delta),
            "bricks_destroyed": 1 if score_delta > 0 else 0,
            "task_success": task_success,
            "task_time_limit": False,
            "brick_hit": score_delta > 0,
            "life_lost": life_lost,
            "last_life_reset": last_life_reset,
            "terminal_failure": last_life_reset,
        }
        completion_progress = 1.0 if task_success else (0.25 if score_delta > 0 else 0.0)
        return RPCObservation(
            metadata={
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "event": "step",
                "action": action,
                "step_count": self.step_count,
                "transition_events": transition_events,
                "state": {
                    "game_state": {
                        "level": self.current_level,
                        "completion_progress": completion_progress,
                    },
                    "metrics": {"lives": self.current_lives},
                },
            },
            png=make_frame((0, self.step_count * 20, 0)),
        )

    def close(self) -> None:
        self.closed = True


class IncompatibleHealthClient(FakeBreakoutClient):
    def health(self) -> dict:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "event": "health",
            "max_steps": 99,
            "viewport": [1280, 720],
            "timing": dict(EXPECTED_EVALUATION_TIMING),
            "supports_initial_lives": True,
            "managed_task_boundaries": True,
        }


class IncompatibleTimingClient(FakeBreakoutClient):
    def health(self) -> dict:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "event": "health",
            "timing": {
                "action_hold_s": 0.2,
                "post_action_idle_s": 0.0,
                "nominal_observation_interval_s": 0.2,
            },
            "supports_initial_lives": True,
            "managed_task_boundaries": True,
        }


class NoInitialLivesHealthClient(FakeBreakoutClient):
    def health(self) -> dict:
        health = super().health()
        health["supports_initial_lives"] = False
        return health


class NativeNextLevelHealthClient(FakeBreakoutClient):
    def health(self) -> dict:
        health = super().health()
        health["managed_task_boundaries"] = False
        return health


class PreprocessTests(unittest.TestCase):
    def test_default_crop_tracks_black_frame_with_safety_margin(self) -> None:
        self.assertEqual(
            (
                DEFAULT_CANVAS_CROP.x,
                DEFAULT_CANVAS_CROP.y,
                DEFAULT_CANVAS_CROP.width,
                DEFAULT_CANVAS_CROP.height,
            ),
            (352, 54, 577, 492),
        )

    def test_preprocess_crops_canvas_and_preserves_rgb_order(self) -> None:
        rgb = preprocess_gameworld_frame(make_frame((255, 0, 0)), size=64)
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb[32, 32], np.array([255, 0, 0], dtype=np.uint8))

        tensor = frame_to_tensor(
            make_frame((255, 0, 0)),
            device=torch.device("cpu"),
            size=64,
        )
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(tensor[0, 0, 32, 32].item(), 1.0)
        self.assertAlmostEqual(tensor[0, 1, 32, 32].item(), -1.0)
        self.assertAlmostEqual(tensor[0, 2, 32, 32].item(), -1.0)

    def test_preprocess_rejects_wrong_viewport(self) -> None:
        image = Image.new("RGB", (800, 600))
        output = BytesIO()
        image.save(output, format="PNG")
        with self.assertRaisesRegex(ValueError, "Unexpected GameWorld viewport"):
            preprocess_gameworld_frame(output.getvalue())


class RPCProtocolTests(unittest.TestCase):
    @staticmethod
    def metadata(**overrides) -> bytes:
        payload = {"protocol_version": PROTOCOL_VERSION, "ok": True, "event": "health"}
        payload.update(overrides)
        return json.dumps(payload).encode("utf-8")

    def test_decode_response_validates_protocol_and_frame_count(self) -> None:
        metadata, png = BreakoutRPCClient.decode_response(
            [self.metadata(), b"\x89PNG\r\n\x1a\nrest"],
            expect_png=True,
        )
        self.assertEqual(metadata["protocol_version"], PROTOCOL_VERSION)
        self.assertIsNotNone(png)

        with self.assertRaisesRegex(BreakoutRPCError, "expected 2"):
            BreakoutRPCClient.decode_response([self.metadata()], expect_png=True)
        with self.assertRaisesRegex(BreakoutRPCError, "Incompatible"):
            BreakoutRPCClient.decode_response(
                [self.metadata(protocol_version=999)],
                expect_png=False,
            )
        with self.assertRaisesRegex(BreakoutRPCError, "Browser service error"):
            BreakoutRPCClient.decode_response(
                [self.metadata(ok=False, error="boom")],
                expect_png=False,
            )


class GameWorldBreakoutEnvTests(unittest.TestCase):
    def make_env(self, client: FakeBreakoutClient, **overrides) -> GameWorldBreakoutEnv:
        kwargs = {
            "device": torch.device("cpu"),
            "client": client,
            "size": 64,
            "max_episode_steps": 2,
            "levels": (1, 2, 3, 4, 5),
            "level_strategy": "cycle",
        }
        kwargs.update(overrides)
        return GameWorldBreakoutEnv(**kwargs)

    def test_reset_matches_diamond_tensor_contract(self) -> None:
        client = FakeBreakoutClient()
        env = self.make_env(client)
        observation, info = env.reset(seed=[42])

        self.assertEqual(tuple(observation.shape), (1, 3, 64, 64))
        self.assertEqual(observation.device.type, "cpu")
        self.assertGreaterEqual(observation.min().item(), -1.0)
        self.assertLessEqual(observation.max().item(), 1.0)
        self.assertEqual(info["level"], 1)
        self.assertEqual(info["seed"], 42)
        self.assertEqual(client.reset_calls, [(1, 42, 3)])

    def test_local_time_limit_auto_resets_and_preserves_final_observation(self) -> None:
        client = FakeBreakoutClient()
        env = self.make_env(client)
        env.reset(seed=[42])

        _, reward, end, trunc, info = env.step(torch.tensor([0]))
        self.assertEqual(reward.tolist(), [0.0])
        self.assertEqual(end.tolist(), [0])
        self.assertEqual(trunc.tolist(), [0])
        self.assertNotIn("final_observation", info)

        next_observation, reward, end, trunc, info = env.step(torch.tensor([2]))
        self.assertEqual(end.tolist(), [0])
        self.assertEqual(trunc.tolist(), [1])
        self.assertEqual(end.dtype, torch.uint8)
        self.assertEqual(trunc.dtype, torch.uint8)
        self.assertEqual(tuple(info["final_observation"].shape), (1, 3, 64, 64))
        self.assertEqual(tuple(next_observation.shape), (1, 3, 64, 64))
        self.assertFalse(torch.equal(info["final_observation"], next_observation))
        self.assertEqual(info["reset_info"]["level"], 2)
        self.assertEqual(client.actions, [0, 2])
        self.assertEqual(len(client.reset_calls), 2)

    def test_success_terminates_and_auto_resets(self) -> None:
        client = FakeBreakoutClient(terminate_at=1)
        env = self.make_env(client, max_episode_steps=100)
        env.reset(seed=[7])

        _, reward, end, trunc, info = env.step(torch.tensor([1]))
        self.assertEqual(reward.tolist(), [35.0])
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertIn("final_observation", info)
        self.assertEqual(info["boundary_reason"], "task_success")
        self.assertTrue(info["physical_reset"])
        self.assertEqual(info["episode_info"][0]["best_completion_progress"].tolist(), [1.0])
        self.assertEqual(info["episode_info"][0]["task_success"].tolist(), [1])
        self.assertEqual(info["episode_metrics"][0]["gameworld/success"], 1.0)
        self.assertEqual(
            info["episode_metrics"][0]["gameworld/best_completion_progress"],
            1.0,
        )
        self.assertEqual(info["reset_info"]["level"], 2)

    def test_training_life_loss_is_logical_boundary_without_browser_reset(self) -> None:
        client = FakeBreakoutClient(life_loss_at=1)
        env = self.make_env(
            client,
            max_episode_steps=100,
            done_on_life_loss=True,
        )
        env.reset(seed=[11])

        life_observation, reward, end, trunc, info = env.step(torch.tensor([0]))
        self.assertEqual(reward.tolist(), [0.0])
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(info["boundary_reason"], "life_loss")
        self.assertFalse(info["physical_reset"])
        self.assertTrue(torch.equal(life_observation, info["final_observation"]))
        self.assertEqual(info["episode_info"][0]["level"].tolist(), [1])
        self.assertEqual(info["episode_info"][0]["game_seed"].tolist(), [11])
        self.assertEqual(info["episode_metrics"][0]["gameworld/lives_lost"], 1.0)
        self.assertEqual(client.reset_calls, [(1, 11, 3)])

        _, _, end, trunc, info = env.step(torch.tensor([2]))
        self.assertEqual(end.tolist(), [0])
        self.assertEqual(trunc.tolist(), [0])
        self.assertNotIn("final_observation", info)
        self.assertEqual(client.step_count, 2)
        self.assertEqual(client.reset_calls, [(1, 11, 3)])

    def test_evaluation_life_loss_continues_without_logical_boundary(self) -> None:
        client = FakeBreakoutClient(life_loss_at=1)
        env = self.make_env(
            client,
            max_episode_steps=100,
            done_on_life_loss=False,
        )
        env.reset(seed=[13])

        _, reward, end, trunc, info = env.step(torch.tensor([1]))
        self.assertEqual(reward.tolist(), [0.0])
        self.assertEqual(end.tolist(), [0])
        self.assertEqual(trunc.tolist(), [0])
        self.assertNotIn("final_observation", info)
        self.assertEqual(client.reset_calls, [(1, 13, 3)])

    def test_fifth_life_loss_physically_resets_to_next_training_level(self) -> None:
        client = FakeBreakoutClient(life_loss_steps={1, 2, 3, 4, 5})
        env = self.make_env(
            client,
            max_episode_steps=500,
            levels=(1, 3, 4),
            initial_lives=5,
            max_task_life_losses=5,
            done_on_life_loss=True,
        )
        env.reset(seed=[17])

        for expected_lives_lost in range(1, 5):
            observation, _, end, trunc, info = env.step(torch.tensor([0]))
            self.assertEqual(end.tolist(), [1])
            self.assertEqual(trunc.tolist(), [0])
            self.assertEqual(info["boundary_reason"], "life_loss")
            self.assertFalse(info["physical_reset"])
            self.assertTrue(torch.equal(observation, info["final_observation"]))
            self.assertEqual(
                info["episode_info"][0]["lives_lost"].tolist(),
                [expected_lives_lost],
            )
            self.assertEqual(client.reset_calls, [(1, 17, 5)])

        _, _, end, trunc, info = env.step(torch.tensor([0]))
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(info["boundary_reason"], "life_budget_exhausted")
        self.assertTrue(info["physical_reset"])
        self.assertEqual(info["episode_info"][0]["lives_lost"].tolist(), [5])
        self.assertEqual(info["episode_info"][0]["task_steps"].tolist(), [5])
        self.assertEqual(info["reset_info"]["level"], 3)
        self.assertEqual(client.reset_calls[1][0], 3)
        self.assertEqual(client.reset_calls[1][2], 5)

    def test_level5_specialist_resets_only_to_level5_after_game_over(self) -> None:
        client = FakeBreakoutClient(life_loss_steps={1, 2, 3, 4, 5})
        env = self.make_env(
            client,
            max_episode_steps=500,
            levels=(5,),
            level_strategy="cycle",
            initial_lives=5,
            max_task_life_losses=5,
            done_on_life_loss=False,
        )
        env.reset(seed=[23])

        for _ in range(4):
            _, _, end, trunc, info = env.step(torch.tensor([0]))
            self.assertEqual(end.tolist(), [0])
            self.assertEqual(trunc.tolist(), [0])
            self.assertFalse(info.get("physical_reset", False))

        _, _, end, trunc, info = env.step(torch.tensor([0]))
        self.assertEqual(end.tolist(), [1])
        self.assertEqual(trunc.tolist(), [0])
        self.assertEqual(info["boundary_reason"], "life_budget_exhausted")
        self.assertTrue(info["physical_reset"])
        self.assertEqual(info["reset_info"]["level"], 5)
        self.assertEqual(
            [level for level, _, _ in client.reset_calls],
            [5, 5],
        )
        self.assertEqual(
            [lives for _, _, lives in client.reset_calls],
            [5, 5],
        )

    def test_random_level_sequence_is_reproducible_from_reset_seed(self) -> None:
        client_a = FakeBreakoutClient()
        client_b = FakeBreakoutClient()
        env_a = self.make_env(client_a, levels=(1, 2, 3, 4, 5), level_strategy="random")
        env_b = self.make_env(client_b, levels=(1, 2, 3, 4, 5), level_strategy="random")

        env_a.reset(seed=[12345])
        env_b.reset(seed=[12345])
        self.assertEqual(client_a.reset_calls[0], client_b.reset_calls[0])

    def test_shuffled_cycle_balances_every_three_training_tasks(self) -> None:
        client = FakeBreakoutClient()
        env = self.make_env(
            client,
            max_episode_steps=1,
            levels=(1, 3, 4),
            level_strategy="shuffled_cycle",
        )
        env.reset(seed=[42])
        for _ in range(5):
            env.step(torch.tensor([0]))

        levels = [level for level, _, _ in client.reset_calls]
        self.assertEqual(set(levels[:3]), {1, 3, 4})
        self.assertEqual(set(levels[3:6]), {1, 3, 4})

    def test_validation_level_and_seed_cycles_cover_cartesian_product(self) -> None:
        client = FakeBreakoutClient()
        fixed_seeds = (4242, 9001)
        env = self.make_env(
            client,
            max_episode_steps=1,
            levels=(1, 3, 4),
            level_strategy="cycle",
            game_seeds=fixed_seeds,
            done_on_life_loss=False,
        )
        env.reset(seed=[999])
        for _ in range(5):
            env.step(torch.tensor([0]))

        self.assertEqual(len(client.reset_calls), 6)
        for level in (1, 3, 4):
            observed = {
                game_seed
                for observed_level, game_seed, _ in client.reset_calls
                if observed_level == level
            }
            self.assertEqual(observed, set(fixed_seeds))

        env.reset(seed=[1234])
        self.assertEqual(client.reset_calls[-1], (1, fixed_seeds[0], 3))

    def test_rejects_multiple_browser_envs_and_invalid_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_envs=1"):
            self.make_env(FakeBreakoutClient(), num_envs=2)

        env = self.make_env(FakeBreakoutClient())
        env.reset(seed=[1])
        with self.assertRaisesRegex(ValueError, "action must be 0, 1 or 2"):
            env.step(torch.tensor([3]))

    def test_health_handshake_rejects_mismatched_episode_limit(self) -> None:
        client = IncompatibleHealthClient()
        with self.assertRaisesRegex(RuntimeError, "episode limits differ"):
            self.make_env(client, max_episode_steps=100)
        self.assertTrue(client.closed)

    def test_health_handshake_rejects_mismatched_timing(self) -> None:
        client = IncompatibleTimingClient()
        with self.assertRaisesRegex(RuntimeError, "timing violates"):
            self.make_env(client)
        self.assertTrue(client.closed)

    def test_health_handshake_requires_initial_lives_support(self) -> None:
        client = NoInitialLivesHealthClient()
        with self.assertRaisesRegex(RuntimeError, "initial_lives"):
            self.make_env(client)
        self.assertTrue(client.closed)

    def test_health_handshake_rejects_native_next_level_frame_leakage(self) -> None:
        client = NativeNextLevelHealthClient()
        with self.assertRaisesRegex(RuntimeError, "next-level"):
            self.make_env(client)
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()

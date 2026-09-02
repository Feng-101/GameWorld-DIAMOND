from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from visualize_gameworld_world_model import (  # noqa: E402
    RepeatingInitialConditionLoader,
    _agent_state_dict,
    collect_initial_condition,
    frame_metrics,
    tensor_to_image,
)


class AgentCheckpointTests(unittest.TestCase):
    def test_accepts_nested_trainer_state_and_normalizes_prefix(self) -> None:
        payload = {
            "agent": {
                "module.denoiser.weight": torch.tensor([1.0]),
                "module.rew_end_model.weight": torch.tensor([2.0]),
                "module.actor_critic.weight": torch.tensor([3.0]),
            }
        }
        state = _agent_state_dict(payload)
        self.assertEqual(
            set(state),
            {
                "denoiser.weight",
                "rew_end_model.weight",
                "actor_critic.weight",
            },
        )

    def test_rejects_incomplete_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing prefixes"):
            _agent_state_dict({"denoiser.weight": torch.tensor([1.0])})


class InitialConditionLoaderTests(unittest.TestCase):
    def test_repeats_current_condition_and_accepts_update(self) -> None:
        obs = torch.zeros((1, 4, 3, 64, 64))
        act = torch.tensor([[1, 2, 0, 0]])
        loader = RepeatingInitialConditionLoader(obs, act)
        iterator = iter(loader)
        first = next(iterator)
        self.assertEqual(tuple(first.obs.shape), (1, 4, 3, 64, 64))
        self.assertTrue(torch.equal(first.act, act))

        new_obs = torch.ones_like(obs)
        new_act = torch.tensor([[2, 2, 1, 0]])
        loader.set_initial_condition(new_obs, new_act)
        second = next(iterator)
        self.assertTrue(torch.equal(second.obs, new_obs))
        self.assertTrue(torch.equal(second.act, new_act))

    def test_collection_never_builds_history_across_episode_boundary(self) -> None:
        class FakeActor:
            def predict_act_value(self, obs, hx_cx):
                batch_size = obs.size(0)
                logits = torch.tensor([[5.0, 0.0, 0.0]]).repeat(batch_size, 1)
                value = torch.zeros(batch_size)
                return logits, value, hx_cx

        class FakeAgent:
            actor_critic = FakeActor()

        class FakeRealEnv:
            max_episode_steps = 10

            def __init__(self) -> None:
                self.step_id = 0

            def reset(self, **kwargs):
                self.step_id = 0
                return torch.zeros((1, 3, 64, 64)), {}

            def step(self, action):
                self.step_id += 1
                obs = torch.full((1, 3, 64, 64), float(self.step_id) / 10)
                # Force one boundary during the requested five-step warm-up.
                end = torch.tensor([self.step_id == 3], dtype=torch.uint8)
                trunc = torch.tensor([False], dtype=torch.uint8)
                return obs, torch.zeros(1), end, trunc, {}

        initial = collect_initial_condition(
            FakeRealEnv(),
            FakeAgent(),
            level=1,
            game_seed=42,
            num_conditioning_steps=4,
            minimum_warmup_steps=5,
            deterministic=True,
        )
        # Three further steps are needed after the boundary to obtain four
        # contiguous observations; stopping at step five would cross it.
        self.assertEqual(initial.warmup_steps, 6)
        self.assertEqual(initial.warmup_boundaries, 1)
        self.assertEqual(tuple(initial.obs.shape), (1, 4, 3, 64, 64))
        self.assertEqual(tuple(initial.act.shape), (1, 4))


class FrameRenderingTests(unittest.TestCase):
    def test_tensor_image_and_identical_metrics(self) -> None:
        obs = torch.zeros((1, 3, 64, 64))
        image = tensor_to_image(obs)
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(image.getpixel((0, 0)), (128, 128, 128))
        mae, psnr = frame_metrics(obs, obs)
        self.assertEqual(mae, 0.0)
        self.assertEqual(psnr, 99.0)


if __name__ == "__main__":
    unittest.main()

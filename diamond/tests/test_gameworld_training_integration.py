from __future__ import annotations

import sys
import tempfile
import unittest
import random
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from coroutines.collector import NumToCollect, make_collector
from data import Dataset
from envs.factory import make_env, validate_env_setup
from utils import isolated_rng, set_seed

from tests.test_gameworld_breakout_env import FakeBreakoutClient


TRAIN_CONFIG = {
    "endpoint": "tcp://127.0.0.1:5561",
    "size": 64,
    "max_episode_steps": 500,
    "initial_lives": 5,
    "max_task_life_losses": 5,
    "done_on_life_loss": True,
    "levels": [1, 3, 4],
    "level_strategy": "shuffled_cycle",
}
TEST_CONFIG = {
    "endpoint": "tcp://127.0.0.1:5562",
    "size": 64,
    "max_episode_steps": 100,
    "initial_lives": 3,
    "max_task_life_losses": None,
    "done_on_life_loss": False,
    "levels": [1, 3, 4],
    "level_strategy": "cycle",
    "game_seeds": [4242, 9001],
}
HELDOUT_CONFIG = {
    "levels": [2, 5],
    "game_seeds": [42, 123, 2025, 31415, 271828],
    "max_episode_steps": 100,
    "initial_lives": 3,
}


class FormalExperimentConfigTests(unittest.TestCase):
    def test_formal_config_composes_the_generalization_protocol(self) -> None:
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(REPO_ROOT / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=[
                    "env=gameworld_breakout",
                    "+experiment=gameworld_breakout_formal",
                ],
            )
        OmegaConf.resolve(cfg)

        self.assertEqual(list(cfg.env.train.levels), [1, 3, 4])
        self.assertEqual(cfg.env.train.level_strategy, "shuffled_cycle")
        self.assertNotIn("game_seeds", cfg.env.train)
        self.assertEqual(cfg.env.train.max_episode_steps, 500)
        self.assertEqual(cfg.env.train.initial_lives, 5)
        self.assertEqual(cfg.env.train.max_task_life_losses, 5)
        self.assertEqual(list(cfg.env.test.levels), [1, 3, 4])
        self.assertEqual(cfg.env.test.level_strategy, "cycle")
        self.assertEqual(
            list(cfg.env.test.game_seeds),
            [4242, 9001],
        )
        self.assertEqual(list(cfg.env.heldout_test.levels), [2, 5])
        self.assertEqual(
            list(cfg.env.heldout_test.game_seeds),
            [42, 123, 2025, 31415, 271828],
        )
        self.assertEqual(cfg.collection.train.num_steps_total, 160_000)
        self.assertEqual(cfg.collection.train.first_epoch.min, 10_000)
        self.assertEqual(cfg.collection.train.first_epoch.max, 10_000)
        self.assertEqual(cfg.collection.train.first_epoch.threshold_rew, 0)
        self.assertEqual(cfg.denoiser.training.steps_first_epoch, 15_000)
        self.assertEqual(cfg.rew_end_model.training.steps_first_epoch, 12_000)
        self.assertEqual(cfg.actor_critic.training.steps_first_epoch, 5_000)
        self.assertEqual(cfg.training.num_final_epochs, 50)
        expected_collection_epochs = (
            cfg.collection.train.num_steps_total
            - cfg.collection.train.first_epoch.min
        ) // cfg.collection.train.steps_per_epoch
        self.assertEqual(expected_collection_epochs, 1_500)
        self.assertEqual(
            expected_collection_epochs + cfg.training.num_final_epochs,
            1_550,
        )
        self.assertEqual(cfg.collection.test.num_episodes, 6)
        self.assertEqual(cfg.collection.test.num_final_episodes, 6)
        self.assertEqual(cfg.world_model_env.horizon, 10)
        self.assertEqual(cfg.actor_critic.actor_critic_loss.backup_every, 10)
        self.assertEqual(cfg.rew_end_model.training.seq_length, 14)
        self.assertEqual(cfg.evaluation.every, 50)
        self.assertEqual(cfg.checkpointing.save_agent_every, 100)
        self.assertEqual(cfg.checkpointing.num_to_keep, 4)
        self.assertFalse(cfg.training.compile_wm)

    def test_formal_horizon_override_updates_all_dependent_lengths(self) -> None:
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(REPO_ROOT / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=[
                    "env=gameworld_breakout",
                    "+experiment=gameworld_breakout_formal",
                    "gameworld.imagination_horizon=15",
                ],
            )
        OmegaConf.resolve(cfg)
        self.assertEqual(cfg.world_model_env.horizon, 15)
        self.assertEqual(cfg.actor_critic.actor_critic_loss.backup_every, 15)
        self.assertEqual(cfg.rew_end_model.training.seq_length, 19)


class Level5SpecialistConfigTests(unittest.TestCase):
    @staticmethod
    def compose(*overrides: str):
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(REPO_ROOT / "config"),
        ):
            cfg = compose(
                config_name="trainer",
                overrides=list(overrides),
            )
        OmegaConf.resolve(cfg)
        return cfg

    def test_level5_specialist_is_single_layout_and_five_life(self) -> None:
        cfg = self.compose(
            "env=gameworld_breakout_level5",
            "+experiment=gameworld_breakout_level5_atari",
        )

        self.assertEqual(cfg.env.protocol, "level5_specialist")
        self.assertEqual(list(cfg.env.train.levels), [5])
        self.assertEqual(list(cfg.env.test.levels), [5])
        self.assertIsNone(cfg.env.heldout_test)
        self.assertNotEqual(cfg.env.train.endpoint, cfg.env.test.endpoint)
        for split in ("train", "test"):
            split_cfg = cfg.env[split]
            self.assertEqual(split_cfg.max_episode_steps, 500)
            self.assertEqual(split_cfg.initial_lives, 5)
            self.assertEqual(split_cfg.max_task_life_losses, 5)
        self.assertTrue(cfg.env.train.done_on_life_loss)
        self.assertFalse(cfg.env.test.done_on_life_loss)
        self.assertEqual(
            list(cfg.env.test.game_seeds),
            [4242, 9001, 2025, 31415],
        )
        self.assertEqual(cfg.checkpointing.save_agent_every, 100)
        self.assertEqual(cfg.checkpointing.num_to_keep, 4)

        kind = validate_env_setup(
            cfg.env.kind,
            train=cfg.env.train,
            test=cfg.env.test,
            train_num_envs=cfg.collection.train.num_envs,
            test_num_envs=cfg.collection.test.num_envs,
            model_free=cfg.training.model_free,
            heldout_test=cfg.env.heldout_test,
            protocol=cfg.env.protocol,
        )
        self.assertEqual(kind, "gameworld_breakout")

    def test_level5_and_mixed_collectors_use_four_distinct_services(self) -> None:
        mixed = self.compose(
            "env=gameworld_breakout",
            "+experiment=gameworld_breakout_formal",
        )
        specialist = self.compose(
            "env=gameworld_breakout_level5",
            "+experiment=gameworld_breakout_level5_atari",
        )
        endpoints = {
            str(mixed.env.train.endpoint),
            str(mixed.env.test.endpoint),
            str(specialist.env.train.endpoint),
            str(specialist.env.test.endpoint),
        }
        self.assertEqual(
            endpoints,
            {
                "tcp://127.0.0.1:5561",
                "tcp://127.0.0.1:5562",
                "tcp://127.0.0.1:5571",
                "tcp://127.0.0.1:5572",
            },
        )

    def test_level5_endpoints_can_be_overridden_for_node_local_slurm_ports(self) -> None:
        specialist = self.compose(
            "env=gameworld_breakout_level5",
            "+experiment=gameworld_breakout_level5_atari",
            "env.train.endpoint=tcp://127.0.0.1:23001",
            "env.test.endpoint=tcp://127.0.0.1:23002",
        )
        self.assertEqual(
            str(specialist.env.train.endpoint),
            "tcp://127.0.0.1:23001",
        )
        self.assertEqual(
            str(specialist.env.test.endpoint),
            "tcp://127.0.0.1:23002",
        )

    def test_level5_algorithm_budget_matches_default_atari_diamond(self) -> None:
        atari = self.compose("env=atari")
        specialist = self.compose(
            "env=gameworld_breakout_level5",
            "+experiment=gameworld_breakout_level5_atari",
        )

        paths = (
            "collection.train.num_steps_total",
            "collection.train.first_epoch.min",
            "collection.train.first_epoch.max",
            "collection.train.first_epoch.threshold_rew",
            "collection.train.steps_per_epoch",
            "collection.train.epsilon",
            "collection.test.num_episodes",
            "collection.test.num_final_episodes",
            "collection.test.epsilon",
            "world_model_env.horizon",
            "denoiser.training.steps_first_epoch",
            "denoiser.training.steps_per_epoch",
            "denoiser.training.batch_size",
            "rew_end_model.training.steps_first_epoch",
            "rew_end_model.training.steps_per_epoch",
            "rew_end_model.training.batch_size",
            "actor_critic.training.steps_first_epoch",
            "actor_critic.training.steps_per_epoch",
            "actor_critic.training.batch_size",
            "actor_critic.actor_critic_loss.backup_every",
            "actor_critic.actor_critic_loss.gamma",
            "actor_critic.actor_critic_loss.lambda_",
            "training.num_final_epochs",
            "training.compile_wm",
            "evaluation.every",
        )
        for path in paths:
            self.assertEqual(
                OmegaConf.select(specialist, path),
                OmegaConf.select(atari, path),
                path,
            )
        self.assertEqual(specialist.rew_end_model.training.seq_length, 19)


class EvaluationIsolationTests(unittest.TestCase):
    def test_validation_rng_stream_does_not_advance_training_rngs(self) -> None:
        set_seed(123)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )

        set_seed(123)
        with isolated_rng(999):
            _ = random.random()
            _ = np.random.random(32)
            _ = torch.rand(32)

        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        self.assertEqual(actual, expected)


class DeterministicActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm_dim = 4
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def predict_act_value(self, obs, hx_cx):
        batch_size = obs.size(0)
        logits = torch.full((batch_size, 3), -100.0, device=obs.device)
        logits[:, 0] = 100.0
        value = torch.zeros(batch_size, device=obs.device)
        return logits, value, hx_cx


class EnvironmentFactoryTests(unittest.TestCase):
    def test_gameworld_configuration_requires_isolated_train_and_test_services(self) -> None:
        kind = validate_env_setup(
            "gameworld_breakout",
            train=TRAIN_CONFIG,
            test=TEST_CONFIG,
            train_num_envs=1,
            test_num_envs=1,
            model_free=False,
            heldout_test=HELDOUT_CONFIG,
        )
        self.assertEqual(kind, "gameworld_breakout")

        with self.assertRaisesRegex(ValueError, "must be different"):
            validate_env_setup(
                "gameworld_breakout",
                train=TRAIN_CONFIG,
                test={**TEST_CONFIG, "endpoint": TRAIN_CONFIG["endpoint"]},
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                heldout_test=HELDOUT_CONFIG,
            )

    def test_mixed_training_requires_500_step_safety_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_episode_steps=500"):
            validate_env_setup(
                "gameworld_breakout",
                train={**TRAIN_CONFIG, "max_episode_steps": 200},
                test=TEST_CONFIG,
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                heldout_test=HELDOUT_CONFIG,
                protocol="mixed_generalization",
            )

    def test_gameworld_configuration_rejects_model_free_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_free"):
            validate_env_setup(
                "gameworld_breakout",
                train=TRAIN_CONFIG,
                test=TEST_CONFIG,
                train_num_envs=1,
                test_num_envs=1,
                model_free=True,
                heldout_test=HELDOUT_CONFIG,
            )

    def test_gameworld_configuration_enforces_train_test_life_semantics(self) -> None:
        with self.assertRaisesRegex(ValueError, "training must set"):
            validate_env_setup(
                "gameworld_breakout",
                train={**TRAIN_CONFIG, "done_on_life_loss": False},
                test=TEST_CONFIG,
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                heldout_test=HELDOUT_CONFIG,
            )
        with self.assertRaisesRegex(ValueError, "evaluation must set"):
            validate_env_setup(
                "gameworld_breakout",
                train=TRAIN_CONFIG,
                test={**TEST_CONFIG, "done_on_life_loss": True},
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                heldout_test=HELDOUT_CONFIG,
            )

    def test_gameworld_configuration_rejects_validation_or_heldout_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly the training layouts"):
            validate_env_setup(
                "gameworld_breakout",
                train=TRAIN_CONFIG,
                test={**TEST_CONFIG, "levels": [1, 3]},
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                heldout_test=HELDOUT_CONFIG,
            )

        with self.assertRaisesRegex(ValueError, "must not appear"):
            validate_env_setup(
                "gameworld_breakout",
                train=TRAIN_CONFIG,
                test=TEST_CONFIG,
                train_num_envs=1,
                test_num_envs=1,
                model_free=False,
                heldout_test={**HELDOUT_CONFIG, "levels": [2, 4, 5]},
            )

    def test_factory_builds_three_action_torch_environment(self) -> None:
        client = FakeBreakoutClient()
        env = make_env(
            "gameworld_breakout",
            num_envs=1,
            device=torch.device("cpu"),
            client=client,
            check_health=False,
            size=64,
            max_episode_steps=2,
            levels=(1,),
            level_strategy="cycle",
        )
        try:
            self.assertEqual(env.num_actions, 3)
            self.assertEqual(tuple(env.observation_space.shape), (1, 3, 64, 64))
        finally:
            env.close()


class CollectorContractTests(unittest.TestCase):
    def test_real_environment_transitions_form_diamond_episodes(self) -> None:
        client = FakeBreakoutClient()
        env = make_env(
            "gameworld_breakout",
            num_envs=1,
            device=torch.device("cpu"),
            client=client,
            check_health=False,
            size=64,
            max_episode_steps=2,
            levels=(1,),
            level_strategy="cycle",
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset(
                Path(directory),
                name="collector_contract",
                cache_in_ram=True,
                save_on_disk=False,
            )
            collector = make_collector(
                env,
                DeterministicActor(),
                dataset,
                epsilon=0.0,
                verbose=False,
            )
            logs = collector.send(NumToCollect(steps=3))

            self.assertEqual(dataset.num_steps, 3)
            self.assertEqual(dataset.num_episodes, 2)
            completed = dataset.load_episode(0)
            self.assertEqual(len(completed), 2)
            self.assertEqual(tuple(completed.info["final_observation"].shape), (3, 64, 64))
            self.assertEqual(completed.trunc.tolist(), [0, 1])
        env.close()

    def test_life_loss_splits_training_data_without_resetting_browser_task(self) -> None:
        client = FakeBreakoutClient(life_loss_at=1)
        env = make_env(
            "gameworld_breakout",
            num_envs=1,
            device=torch.device("cpu"),
            client=client,
            check_health=False,
            size=64,
            max_episode_steps=4,
            levels=(1,),
            level_strategy="cycle",
            done_on_life_loss=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Dataset(
                Path(directory),
                name="life_boundary_contract",
                cache_in_ram=True,
                save_on_disk=False,
            )
            collector = make_collector(
                env,
                DeterministicActor(),
                dataset,
                epsilon=0.0,
                verbose=False,
            )
            logs = collector.send(NumToCollect(steps=3))

            self.assertEqual(dataset.num_steps, 3)
            self.assertEqual(dataset.num_episodes, 2)
            life_episode = dataset.load_episode(0)
            self.assertEqual(len(life_episode), 1)
            self.assertEqual(life_episode.end.tolist(), [1])
            self.assertEqual(
                tuple(life_episode.info["final_observation"].shape),
                (3, 64, 64),
            )
            self.assertEqual(life_episode.info["level"].tolist(), [1])
            self.assertEqual(life_episode.info["game_seed"].tolist(), [client.reset_calls[0][1]])
            self.assertEqual(life_episode.info["lives_lost"].tolist(), [1])
            life_log = next(log for log in logs if "gameworld/level" in log)
            self.assertEqual(life_log["gameworld/level"], 1.0)
            self.assertEqual(life_log["gameworld/lives_lost"], 1.0)
            self.assertEqual(life_log["gameworld/physical_reset"], 0.0)
            self.assertEqual(client.step_count, 3)
            self.assertEqual(len(client.reset_calls), 1)
        env.close()


if __name__ == "__main__":
    unittest.main()

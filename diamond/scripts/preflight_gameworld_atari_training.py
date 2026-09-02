"""Non-mutating preflight for deterministic Atari-style browser Breakout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent import Agent  # noqa: E402
from envs import validate_env_setup  # noqa: E402
from integrations.gameworld import (  # noqa: E402
    AtariBreakoutRPCClient,
    EXPECTED_ACTION_MEANINGS,
    EXPECTED_ATARI_TIMING,
)


def _health(endpoint: str, timeout_ms: int) -> dict:
    client = AtariBreakoutRPCClient(endpoint, timeout_ms=timeout_ms)
    try:
        health = client.health()
    finally:
        client.close()

    expected = {
        "environment": "gameworld_deterministic_breakout",
        "max_steps": None,
        "initial_lives": 5,
        "game_over_lives": 0,
        "viewport": [1280, 720],
        "reset_noop_max": 30,
        "action_meanings": EXPECTED_ACTION_MEANINGS,
        "timing": EXPECTED_ATARI_TIMING,
    }
    mismatches = {
        key: {"received": health.get(key), "expected": value}
        for key, value in expected.items()
        if health.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Service {endpoint} violates the Atari contract: {mismatches}"
        )
    return health


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument(
        "--train-endpoint",
        default="tcp://127.0.0.1:5661",
    )
    parser.add_argument(
        "--test-endpoint",
        default="tcp://127.0.0.1:5662",
    )
    args = parser.parse_args()

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    overrides = [
        "env=gameworld_atari_breakout",
        "+experiment=gameworld_atari_breakout_formal",
        f"env.level={args.level}",
        f"env.train.endpoint={args.train_endpoint}",
        f"env.test.endpoint={args.test_endpoint}",
    ]
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(REPO_ROOT / "config"),
    ):
        cfg = compose(config_name="trainer", overrides=overrides)
    OmegaConf.resolve(cfg)

    kind = validate_env_setup(
        cfg.env.get("kind"),
        train=cfg.env.train,
        test=cfg.env.test,
        train_num_envs=cfg.collection.train.num_envs,
        test_num_envs=cfg.collection.test.num_envs,
        model_free=cfg.training.model_free,
        heldout_test=cfg.env.get("heldout_test"),
        protocol=cfg.env.get("protocol"),
    )

    if list(map(int, cfg.env.train.levels)) != [args.level]:
        raise RuntimeError("Training level did not resolve to the requested level")
    if list(map(int, cfg.env.test.levels)) != [args.level]:
        raise RuntimeError("Test level did not resolve to the requested level")
    if cfg.env.train.max_episode_steps is not None:
        raise RuntimeError("Training unexpectedly has a step limit")
    if cfg.env.test.max_episode_steps is not None:
        raise RuntimeError("Test unexpectedly has a step limit")

    expected_budget = {
        "real_steps_total": 100_000,
        "initial_min": 5_000,
        "initial_max": 10_000,
        "reward_threshold": 10,
        "steps_per_epoch": 100,
        "denoiser_first_epoch": 10_000,
        "rew_end_first_epoch": 10_000,
        "actor_first_epoch": 5_000,
        "denoiser_per_epoch": 400,
        "rew_end_per_epoch": 400,
        "actor_per_epoch": 400,
        "horizon": 15,
        "backup_every": 15,
        "batch_size_denoiser": 32,
        "batch_size_rew_end": 32,
        "batch_size_actor": 32,
        "final_epochs": 50,
    }
    actual_budget = {
        "real_steps_total": int(cfg.collection.train.num_steps_total),
        "initial_min": int(cfg.collection.train.first_epoch.min),
        "initial_max": int(cfg.collection.train.first_epoch.max),
        "reward_threshold": int(cfg.collection.train.first_epoch.threshold_rew),
        "steps_per_epoch": int(cfg.collection.train.steps_per_epoch),
        "denoiser_first_epoch": int(cfg.denoiser.training.steps_first_epoch),
        "rew_end_first_epoch": int(cfg.rew_end_model.training.steps_first_epoch),
        "actor_first_epoch": int(cfg.actor_critic.training.steps_first_epoch),
        "denoiser_per_epoch": int(cfg.denoiser.training.steps_per_epoch),
        "rew_end_per_epoch": int(cfg.rew_end_model.training.steps_per_epoch),
        "actor_per_epoch": int(cfg.actor_critic.training.steps_per_epoch),
        "horizon": int(cfg.world_model_env.horizon),
        "backup_every": int(cfg.actor_critic.actor_critic_loss.backup_every),
        "batch_size_denoiser": int(cfg.denoiser.training.batch_size),
        "batch_size_rew_end": int(cfg.rew_end_model.training.batch_size),
        "batch_size_actor": int(cfg.actor_critic.training.batch_size),
        "final_epochs": int(cfg.training.num_final_epochs),
    }
    if actual_budget != expected_budget:
        raise RuntimeError(
            "Formal configuration no longer matches the DIAMOND Atari schedule: "
            f"received={actual_budget}, expected={expected_budget}"
        )

    services = {
        split: _health(
            str(cfg.env[split].endpoint),
            int(cfg.env[split].timeout_ms),
        )
        for split in ("train", "test")
    }

    agent = Agent(instantiate(cfg.agent, num_actions=4))
    if agent.actor_critic.actor_linear.out_features != 4:
        raise RuntimeError("Actor head does not expose four Atari actions")
    if agent.denoiser.inner_model.act_emb[0].num_embeddings != 4:
        raise RuntimeError("Denoiser action embedding does not expose four actions")
    if agent.rew_end_model.act_emb.num_embeddings != 4:
        raise RuntimeError("Reward/end model does not expose four actions")

    report = {
        "ok": True,
        "environment": kind,
        "protocol": str(cfg.env.protocol),
        "level": args.level,
        "num_actions": 4,
        "action_meanings": EXPECTED_ACTION_MEANINGS,
        "observation_shape": [1, 3, 64, 64],
        "initial_lives": 5,
        "game_over_lives": 0,
        "max_episode_steps": None,
        "train_done_on_life_loss": bool(cfg.env.train.done_on_life_loss),
        "test_done_on_life_loss": bool(cfg.env.test.done_on_life_loss),
        "training_budget": actual_budget,
        "compile_wm": bool(cfg.training.compile_wm),
        "num_agent_parameters": sum(
            parameter.numel() for parameter in agent.parameters()
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "services": services,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

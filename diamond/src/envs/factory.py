"""Environment construction and cross-environment configuration checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


ATARI_KIND = "atari"
GAMEWORLD_BREAKOUT_KIND = "gameworld_breakout"
GAMEWORLD_ATARI_BREAKOUT_KIND = "gameworld_atari_breakout"
SUPPORTED_ENV_KINDS = {
    ATARI_KIND,
    GAMEWORLD_BREAKOUT_KIND,
    GAMEWORLD_ATARI_BREAKOUT_KIND,
}
MIXED_GENERALIZATION_PROTOCOL = "mixed_generalization"
LEVEL5_SPECIALIST_PROTOCOL = "level5_specialist"
ATARI_SINGLE_LEVEL_PROTOCOL = "atari_single_level"
SUPPORTED_GAMEWORLD_PROTOCOLS = {
    MIXED_GENERALIZATION_PROTOCOL,
    LEVEL5_SPECIALIST_PROTOCOL,
}


def _normalized_kind(kind: str | None) -> str:
    # Older DIAMOND run configs predate ``env.kind`` and are Atari configs.
    normalized = ATARI_KIND if kind is None else str(kind)
    if normalized not in SUPPORTED_ENV_KINDS:
        raise ValueError(
            f"Unsupported environment kind {normalized!r}; "
            f"expected one of {sorted(SUPPORTED_ENV_KINDS)}"
        )
    return normalized


def make_env(
    kind: str | None,
    *,
    num_envs: int,
    device: torch.device,
    **kwargs: Any,
):
    """Build one DIAMOND real environment from a Hydra environment kind."""
    normalized = _normalized_kind(kind)
    if normalized == ATARI_KIND:
        from .env import make_atari_env

        return make_atari_env(num_envs=num_envs, device=device, **kwargs)
    if normalized == GAMEWORLD_ATARI_BREAKOUT_KIND:
        from .gameworld_atari_breakout import GameWorldAtariBreakoutEnv

        return GameWorldAtariBreakoutEnv(
            num_envs=num_envs,
            device=device,
            **kwargs,
        )

    from .gameworld_breakout import GameWorldBreakoutEnv

    return GameWorldBreakoutEnv(num_envs=num_envs, device=device, **kwargs)


def validate_env_setup(
    kind: str | None,
    *,
    train: Mapping[str, Any],
    test: Mapping[str, Any],
    train_num_envs: int,
    test_num_envs: int,
    model_free: bool,
    heldout_test: Mapping[str, Any] | None = None,
    protocol: str | None = None,
) -> str:
    """Reject configurations that would silently corrupt a training run."""
    normalized = _normalized_kind(kind)

    train_size = int(train["size"])
    test_size = int(test["size"])
    if train_size != test_size:
        raise ValueError(
            "Train and test observation sizes must match the shared DIAMOND model: "
            f"train={train_size}, test={test_size}"
        )

    if normalized == ATARI_KIND:
        return normalized

    if train_num_envs != 1 or test_num_envs != 1:
        raise ValueError(
            "Browser Breakout requires collection.train.num_envs=1 "
            "and collection.test.num_envs=1"
        )
    if model_free:
        raise ValueError(
            "Browser Breakout integration targets DIAMOND world-model training; "
            "training.model_free must remain false"
        )

    train_endpoint = str(train.get("endpoint", "")).strip()
    test_endpoint = str(test.get("endpoint", "")).strip()
    if not train_endpoint or not test_endpoint:
        raise ValueError("Browser Breakout train and test endpoints must be configured")
    if train_endpoint == test_endpoint:
        raise ValueError(
            "Browser Breakout train and test endpoints must be different. "
            "Each collector requires an independent physical game."
        )

    if train.get("done_on_life_loss") is not True:
        raise ValueError(
            "Browser Breakout training must set done_on_life_loss=true to match "
            "DIAMOND's Atari training semantics"
        )
    if test.get("done_on_life_loss") is not False:
        raise ValueError(
            "Browser Breakout evaluation must set done_on_life_loss=false to "
            "evaluate a complete five-life game"
        )

    if normalized == GAMEWORLD_ATARI_BREAKOUT_KIND:
        normalized_protocol = (
            ATARI_SINGLE_LEVEL_PROTOCOL if protocol is None else str(protocol)
        )
        if normalized_protocol != ATARI_SINGLE_LEVEL_PROTOCOL:
            raise ValueError(
                "Atari-style browser Breakout requires "
                f"protocol={ATARI_SINGLE_LEVEL_PROTOCOL!r}"
            )

        train_levels = tuple(int(level) for level in train.get("levels", ()))
        test_levels = tuple(int(level) for level in test.get("levels", ()))
        if (
            len(train_levels) != 1
            or len(test_levels) != 1
            or train_levels != test_levels
            or train_levels[0] not in range(1, 6)
        ):
            raise ValueError(
                "Atari-style browser Breakout train/test must use the same one "
                f"level from 1..5; train={train_levels}, test={test_levels}"
            )
        if (
            train.get("max_episode_steps") is not None
            or test.get("max_episode_steps") is not None
        ):
            raise ValueError(
                "Atari-style browser Breakout has no train or test step limit"
            )
        for split_name, split in (("train", train), ("test", test)):
            if int(split.get("initial_lives", 0)) != 5 or int(
                split.get("max_task_life_losses", 0)
            ) != 5:
                raise ValueError(
                    f"Atari-style {split_name} must use exactly five lives"
                )
        if heldout_test is not None:
            raise ValueError(
                "The Atari-style single-level specialist does not define a "
                "held-out level split"
            )
        return normalized

    normalized_protocol = (
        MIXED_GENERALIZATION_PROTOCOL if protocol is None else str(protocol)
    )
    if normalized_protocol not in SUPPORTED_GAMEWORLD_PROTOCOLS:
        raise ValueError(
            f"Unsupported GameWorld Breakout protocol {normalized_protocol!r}; "
            f"expected one of {sorted(SUPPORTED_GAMEWORLD_PROTOCOLS)}"
        )

    train_levels = {int(level) for level in train.get("levels", ())}
    test_levels = {int(level) for level in test.get("levels", ())}
    if not train_levels or not test_levels:
        raise ValueError("GameWorld Breakout train and validation levels must be configured")
    if test_levels != train_levels:
        raise ValueError(
            "Training-time validation must use exactly the training layouts; "
            f"train={sorted(train_levels)}, validation={sorted(test_levels)}"
        )

    if int(train.get("initial_lives", 0)) != 5 or int(
        train.get("max_task_life_losses", 0)
    ) != 5:
        raise ValueError(
            "GameWorld Breakout training must start with five lives and switch "
            "physical tasks after five life losses"
        )
    if normalized_protocol == MIXED_GENERALIZATION_PROTOCOL:
        if int(train.get("max_episode_steps", 0)) != 500:
            raise ValueError(
                "Mixed-level training must use max_episode_steps=500 as a "
                "five-life game safety cap"
            )
        if int(test.get("max_episode_steps", 0)) != 100:
            raise ValueError(
                "Mixed-level validation must retain the official 100-step limit"
            )
        if int(test.get("initial_lives", 0)) != 3:
            raise ValueError(
                "Mixed-level validation must retain the official three lives"
            )
        if test.get("max_task_life_losses") is not None:
            raise ValueError(
                "Mixed-level validation must continue through engine game-over "
                "resets until its official 100-step limit"
            )

        if heldout_test is None:
            raise ValueError("Mixed-level heldout_test protocol must be configured")
        heldout_levels = {int(level) for level in heldout_test.get("levels", ())}
        if not heldout_levels:
            raise ValueError("Mixed-level heldout_test levels must be configured")
        overlap = train_levels & heldout_levels
        if overlap:
            raise ValueError(
                "Final held-out layouts must not appear in training or validation: "
                f"overlap={sorted(overlap)}"
            )
    else:
        if train_levels != {5}:
            raise ValueError(
                "Level 5 specialist training and validation must use only Level 5"
            )
        if int(train.get("max_episode_steps", 0)) != 500 or int(
            test.get("max_episode_steps", 0)
        ) != 500:
            raise ValueError(
                "Level 5 specialist train/test tasks must use the 500-step safety cap"
            )
        if int(test.get("initial_lives", 0)) != 5 or int(
            test.get("max_task_life_losses", 0)
        ) != 5:
            raise ValueError(
                "Level 5 specialist validation must run complete five-life games"
            )
        if heldout_test is not None:
            raise ValueError(
                "Level 5 specialist is a trained-layout control and must not define "
                "a heldout_test split"
            )

    return normalized


__all__ = [
    "ATARI_KIND",
    "ATARI_SINGLE_LEVEL_PROTOCOL",
    "GAMEWORLD_BREAKOUT_KIND",
    "GAMEWORLD_ATARI_BREAKOUT_KIND",
    "LEVEL5_SPECIALIST_PROTOCOL",
    "MIXED_GENERALIZATION_PROTOCOL",
    "SUPPORTED_ENV_KINDS",
    "SUPPORTED_GAMEWORLD_PROTOCOLS",
    "make_env",
    "validate_env_setup",
]

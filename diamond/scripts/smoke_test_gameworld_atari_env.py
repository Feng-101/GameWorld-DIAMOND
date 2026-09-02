"""Exercise the real DIAMOND adapter against one Atari-style browser service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from envs.gameworld_atari_breakout import GameWorldAtariBreakoutEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5661")
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    environment = GameWorldAtariBreakoutEnv(
        device=torch.device("cpu"),
        endpoint=args.endpoint,
        levels=(args.level,),
        max_episode_steps=None,
        initial_lives=5,
        max_task_life_losses=5,
        done_on_life_loss=False,
    )
    records = []
    try:
        observation, reset_info = environment.reset(seed=[args.seed])
        for index in range(args.steps):
            action = (1, 0, 2, 3)[index % 4]
            _, reward, end, trunc, info = environment.step(
                torch.tensor([action])
            )
            metadata = info["gameworld_atari"]
            records.append(
                {
                    "step": index + 1,
                    "action": action,
                    "action_meaning": metadata["action_meaning"],
                    "reward": float(reward.item()),
                    "end": int(end.item()),
                    "trunc": int(trunc.item()),
                    "frames_executed": metadata["frames_executed"],
                    "game_time_advance_ms": metadata["game_time_advance_ms"],
                    "screenshot_game_time_advance_ms": metadata[
                        "screenshot_game_time_advance_ms"
                    ],
                    "transition_events": metadata["transition_events"],
                }
            )

        print(
            json.dumps(
                {
                    "ok": True,
                    "endpoint": args.endpoint,
                    "level": reset_info["level"],
                    "seed": reset_info["seed"],
                    "observation_shape": list(observation.shape),
                    "num_actions": environment.num_actions,
                    "max_episode_steps": environment.max_episode_steps,
                    "records": records,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()

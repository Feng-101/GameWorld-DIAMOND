"""Evaluate a DIAMOND checkpoint only in the real GameWorld Breakout browser.

The actor always receives the real, cropped 64x64 observation returned after
each GameWorld macro-step.  No diffusion/world-model prediction is used.  Each
task is recorded as an MP4 whose default 5 FPS gives a simple 0.2-second
display interval per discrete observation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import cv2
from hydra.utils import instantiate
import numpy as np
import torch
from torch import Tensor

from agent import Agent
from envs import GameWorldBreakoutEnv
from visualize_gameworld_world_model import (
    PROFILE_DEFAULT_LEVEL,
    _agent_state_dict,
    _torch_load,
    load_config,
    tensor_to_image,
    validate_profile_config,
)


ACTION_NAMES = ("wait", "left", "right")


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


class AgentObservationMp4:
    """Incrementally encode enlarged copies of the exact agent observation."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        output_size: int,
        codec: str = "mp4v",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if output_size < 2 or output_size % 2:
            raise ValueError("output_size must be a positive even integer")
        if len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = float(fps)
        self.output_size = output_size
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.fps,
            (output_size, output_size),
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(
                f"OpenCV could not open MP4 writer for {path} with codec {codec!r}"
            )
        self.num_frames = 0

    def write(self, observation: Tensor) -> None:
        rgb = np.asarray(tensor_to_image(observation), dtype=np.uint8)
        enlarged = cv2.resize(
            rgb,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_NEAREST,
        )
        self.writer.write(cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR))
        self.num_frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> AgentObservationMp4:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _episode_name(index: int, level: int, game_seed: int) -> str:
    return f"episode_{index:03d}_level{level}_seed{game_seed}"


def _float_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


@torch.no_grad()
def evaluate_one_task(
    *,
    env: GameWorldBreakoutEnv,
    agent: Agent,
    level: int,
    game_seed: int,
    task_index: int,
    deterministic: bool,
    policy_seed: int,
    output_dir: Path,
    video_fps: float,
    video_size: int,
    video_codec: str,
) -> dict[str, Any]:
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)

    observation, reset_info = env.reset(
        seed=[game_seed],
        options={"level": level, "game_seed": game_seed},
    )
    hx = torch.zeros(
        (1, agent.actor_critic.lstm_dim),
        device=agent.device,
    )
    cx = torch.zeros_like(hx)
    native_return = 0.0
    best_progress = 0.0
    lives_lost = 0
    step_records: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    entropies: list[float] = []
    values: list[float] = []
    task_success = False
    boundary_reason: str | None = None

    stem = _episode_name(task_index, level, game_seed)
    video_path = output_dir / f"{stem}.mp4"
    started = time.perf_counter()
    with AgentObservationMp4(
        video_path,
        fps=video_fps,
        output_size=video_size,
        codec=video_codec,
    ) as video:
        # Include the reset observation, followed by exactly one frame for each
        # real GameWorld macro-step.
        video.write(observation)

        for step in range(1, env.max_episode_steps + 1):
            logits, value, (hx, cx) = agent.actor_critic.predict_act_value(
                observation,
                (hx, cx),
            )
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
            entropy_bits = float((distribution.entropy() / math.log(2)).item())
            value_float = float(value.item())

            next_observation, reward, end, trunc, info = env.step(action)
            dead = bool(torch.logical_or(end, trunc).item())
            frame_observation = (
                info["final_observation"] if dead else next_observation
            )
            video.write(frame_observation)

            action_index = int(action.item())
            reward_float = float(reward.item())
            native_return += reward_float
            actions[ACTION_NAMES[action_index]] += 1
            entropies.append(entropy_bits)
            values.append(value_float)

            gameworld = info.get("gameworld")
            events = (
                gameworld.get("transition_events")
                if isinstance(gameworld, Mapping)
                else None
            )
            if isinstance(events, Mapping) and events.get("life_lost"):
                lives_lost += 1
            task_success = task_success or bool(
                isinstance(events, Mapping) and events.get("task_success")
            )
            best_progress = max(
                best_progress,
                float(info.get("task_best_completion_progress", 0.0)),
            )
            boundary_reason = info.get("boundary_reason", boundary_reason)
            step_records.append(
                {
                    "step": step,
                    "action": action_index,
                    "action_name": ACTION_NAMES[action_index],
                    "reward": reward_float,
                    "policy_entropy_bits": entropy_bits,
                    "policy_value": value_float,
                    "end": bool(end.item()),
                    "trunc": bool(trunc.item()),
                    "life_lost": bool(
                        isinstance(events, Mapping) and events.get("life_lost")
                    ),
                    "task_success": bool(
                        isinstance(events, Mapping) and events.get("task_success")
                    ),
                    "completion_progress_best": best_progress,
                    "boundary_reason": info.get("boundary_reason"),
                }
            )

            if dead:
                break
            observation = next_observation

        num_video_frames = video.num_frames

    elapsed_s = time.perf_counter() - started
    trace_path = output_dir / f"{stem}.json"
    result = {
        "task_index": task_index,
        "level": level,
        "game_seed": game_seed,
        "initial_lives": env.initial_lives,
        "max_episode_steps": env.max_episode_steps,
        "deterministic": deterministic,
        "policy_seed": policy_seed,
        "num_steps": len(step_records),
        "native_return": native_return,
        "best_completion_progress": best_progress,
        "lives_lost": lives_lost,
        "task_success": task_success,
        "boundary_reason": boundary_reason,
        "action_counts": dict(actions),
        "mean_policy_entropy_bits": _float_mean(entropies),
        "mean_policy_value": _float_mean(values),
        "wall_elapsed_s": elapsed_s,
        "video": str(video_path.resolve()),
        "video_fps": video_fps,
        "video_frame_interval_s": 1.0 / video_fps,
        "num_video_frames": num_video_frames,
        "video_playback_duration_s": num_video_frames / video_fps,
        "reset_info": {
            "level": reset_info.get("level"),
            "seed": reset_info.get("seed"),
            "initial_lives": reset_info.get("initial_lives"),
        },
        "steps": step_records,
    }
    trace_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["trace"] = str(trace_path.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("mixed", "level5"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5581")
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--game-seeds", type=int, nargs="+", default=[4242])
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--initial-lives", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--policy-seed", type=int, default=20260715)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.levels is not None and any(level not in range(1, 6) for level in args.levels):
        raise ValueError("--levels values must be between 1 and 5")
    if not args.game_seeds:
        raise ValueError("--game-seeds must contain at least one seed")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive")
    if args.initial_lives is not None and args.initial_lives not in range(1, 6):
        raise ValueError("--initial-lives must be between 1 and 5")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.video_size < 2 or args.video_size % 2:
        raise ValueError("--video-size must be a positive even integer")
    if len(args.video_codec) != 4:
        raise ValueError("--video-codec must contain exactly four characters")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    cfg, config_path = load_config(args.profile, args.checkpoint, args.config)
    validate_profile_config(args.profile, cfg)
    device = _choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    agent = Agent(instantiate(cfg.agent, num_actions=3)).to(device).eval()
    checkpoint_payload = _torch_load(args.checkpoint, map_location=device)
    agent.load_state_dict(_agent_state_dict(checkpoint_payload), strict=True)

    levels = (
        list(args.levels)
        if args.levels is not None
        else [PROFILE_DEFAULT_LEVEL[args.profile]]
    )
    test_cfg = cfg.env.test
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else int(test_cfg.max_episode_steps)
    )
    initial_lives = (
        args.initial_lives
        if args.initial_lives is not None
        else int(test_cfg.initial_lives)
    )
    configured_life_budget = test_cfg.get("max_task_life_losses")
    max_task_life_losses = (
        None
        if configured_life_budget is None
        else initial_lives
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path("real_agent_evaluations")
        / f"{args.profile}_{args.checkpoint.stem}_{timestamp}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = GameWorldBreakoutEnv(
        device=device,
        endpoint=args.endpoint,
        size=int(test_cfg.size),
        max_episode_steps=max_episode_steps,
        levels=levels,
        level_strategy="cycle",
        initial_lives=initial_lives,
        max_task_life_losses=max_task_life_losses,
        done_on_life_loss=False,
        timeout_ms=int(test_cfg.get("timeout_ms", 30_000)),
    )
    results: list[dict[str, Any]] = []
    try:
        schedule = [
            (level, game_seed)
            for level in levels
            for game_seed in args.game_seeds
        ]
        for task_index, (level, game_seed) in enumerate(schedule):
            print(
                f"Evaluating real GameWorld task {task_index + 1}/{len(schedule)}: "
                f"level={level} seed={game_seed}",
                flush=True,
            )
            result = evaluate_one_task(
                env=env,
                agent=agent,
                level=level,
                game_seed=game_seed,
                task_index=task_index,
                deterministic=args.deterministic,
                policy_seed=args.policy_seed + task_index,
                output_dir=output_dir,
                video_fps=args.video_fps,
                video_size=args.video_size,
                video_codec=args.video_codec,
            )
            results.append(result)
            print(
                f"  return={result['native_return']:.1f} "
                f"progress={result['best_completion_progress']:.3f} "
                f"steps={result['num_steps']} video={result['video']}",
                flush=True,
            )
    finally:
        env.close()

    summary = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
        "profile": args.profile,
        "configuration": (
            str(config_path.resolve()) if config_path is not None else None
        ),
        "endpoint": args.endpoint,
        "device": str(device),
        "observation_source": "real_gameworld_only",
        "world_model_used": False,
        "levels": levels,
        "game_seeds": args.game_seeds,
        "max_episode_steps": max_episode_steps,
        "initial_lives": initial_lives,
        "deterministic": args.deterministic,
        "video_fps": args.video_fps,
        "video_frame_interval_s": 1.0 / args.video_fps,
        "num_tasks": len(results),
        "native_return_mean": _float_mean(
            [float(result["native_return"]) for result in results]
        ),
        "progress_mean": _float_mean(
            [float(result["best_completion_progress"]) for result in results]
        ),
        "success_rate": _float_mean(
            [float(result["task_success"]) for result in results]
        ),
        "tasks": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved evaluation summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

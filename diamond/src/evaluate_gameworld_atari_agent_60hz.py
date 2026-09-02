"""Evaluate the final DIAMOND policy in deterministic fixed-4-frame Breakout.

This mirrors DIAMOND's training-time test policy, not the official GameWorld
evaluator. The actor sees only real 64x64 observations, samples from its
categorical policy with epsilon zero, preserves recurrent state across the
first four life losses, and stops only at game-over or level clear.

The diagnostic browser command returns every executed native 60 Hz canvas
frame. The MP4 therefore shows true simulated game frames plus the selected
action and policy statistics; video capture advances zero game time.
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
import sys
import time
from typing import Any, Mapping

import cv2
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "config"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent  # noqa: E402
from integrations.gameworld import (  # noqa: E402
    AtariBreakoutRPCClient,
    EXPECTED_ACTION_MEANINGS,
    EXPECTED_ATARI_TIMING,
    frame_to_tensor,
)
from visualize_gameworld_world_model import _agent_state_dict, _torch_load  # noqa: E402


ACTION_NAMES = tuple(EXPECTED_ACTION_MEANINGS)
FRAMES_PER_ACTION = 4
CANVAS_SIZE = (800, 600)
# Same region consumed by training before resizing to 64x64. Full viewport crop
# (352, 54, 577, 492) minus canvas origin (240, 17).
VIDEO_CROP = (112, 37, 577, 492)
VIDEO_PAD_RIGHT = 1
STATUS_HEIGHT = 64
VIDEO_SIZE = (VIDEO_CROP[2] + VIDEO_PAD_RIGHT, VIDEO_CROP[3] + STATUS_HEIGHT)
VIDEO_FPS = 60.0
ACTION_COLORS_BGR = {
    0: (180, 180, 180),
    1: (40, 145, 255),
    2: (255, 180, 50),
    3: (70, 210, 90),
}


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _load_agent(checkpoint: Path, device: torch.device) -> Agent:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        cfg = compose(
            config_name="trainer",
            overrides=[
                "env=gameworld_atari_breakout",
                "+experiment=gameworld_atari_breakout_formal",
            ],
        )
    OmegaConf.resolve(cfg)
    agent = Agent(instantiate(cfg.agent, num_actions=4)).to(device).eval()
    payload = _torch_load(checkpoint, map_location=device)
    agent.load_state_dict(_agent_state_dict(payload), strict=True)
    return agent


def _decode_png(png: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Browser returned an undecodable PNG")
    return image


def _state_value(frame_state: Mapping[str, Any], key: str, default: Any = "?") -> Any:
    value = frame_state.get(key, default)
    return default if value is None else value


class ActionVideo:
    """Native-pixel game crop plus action/status panel."""

    def __init__(self, path: Path, *, codec: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            VIDEO_FPS,
            VIDEO_SIZE,
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(
                f"OpenCV could not open video writer codec={codec!r} path={path}"
            )
        self.num_frames = 0
        self.preview: np.ndarray | None = None

    def write(
        self,
        canvas_png: bytes,
        *,
        step: int,
        action: int,
        subframe: int,
        frames_in_step: int,
        probabilities: list[float],
        value: float,
        frame_state: Mapping[str, Any],
    ) -> None:
        canvas = _decode_png(canvas_png)
        if (canvas.shape[1], canvas.shape[0]) != CANVAS_SIZE:
            raise RuntimeError(
                "Recording canvas has the wrong size: "
                f"actual={(canvas.shape[1], canvas.shape[0])}, expected={CANVAS_SIZE}"
            )
        x, y, width, height = VIDEO_CROP
        game = canvas[y : y + height, x : x + width]
        game = cv2.copyMakeBorder(
            game,
            0,
            0,
            0,
            VIDEO_PAD_RIGHT,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        panel = np.full((STATUS_HEIGHT, VIDEO_SIZE[0], 3), 20, dtype=np.uint8)
        color = ACTION_COLORS_BGR[action]
        line1 = (
            f"STEP {step:04d}  FRAME {subframe:02d}/{frames_in_step:02d}  "
            f"ACTION {action}: {ACTION_NAMES[action]}"
        )
        probs = "/".join(f"{probability:.2f}" for probability in probabilities)
        progress = _state_value(frame_state, "completion_progress", "?")
        progress_text = (
            f"{float(progress):.3f}" if isinstance(progress, (int, float)) else "?"
        )
        line2 = (
            f"P[0/1/2/3]={probs}  V={value:+.2f}  "
            f"LIVES={_state_value(frame_state, 'lives')}  "
            f"SCORE={_state_value(frame_state, 'score')}  "
            f"PROGRESS={progress_text}"
        )
        cv2.putText(
            panel,
            line1,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            line2,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        output = np.vstack((panel, game))
        self.writer.write(output)
        if self.preview is None:
            self.preview = output.copy()
        self.num_frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "ActionVideo":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _events(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = metadata.get("transition_events")
    if not isinstance(value, dict):
        raise RuntimeError("Browser step omitted transition_events")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5675")
    parser.add_argument("--level", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--game-seed", type=int, default=4242)
    parser.add_argument("--policy-seed", type=int, default=20260720)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--safety-max-steps",
        type=int,
        default=0,
        help="0 means no cap: stop only on five-life game-over or level clear.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--video-codec",
        choices=("avc1", "mp4v", "FFV1"),
        default="mp4v",
        help="mp4v is the most portable local default; avc1 needs H.264 support.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.safety_max_steps < 0:
        raise ValueError("--safety-max-steps must be non-negative")

    random.seed(args.policy_seed)
    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.policy_seed)

    device = _choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    agent = _load_agent(checkpoint, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT.parent
        / "real_agent_evaluations"
        / f"atari_f4_{checkpoint.stem}_level{args.level}_{timestamp}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = ".avi" if args.video_codec == "FFV1" else ".mp4"
    video_path = output_dir / f"agent_real_60hz{extension}"
    preview_path = output_dir / "preview_action_overlay.png"
    initial_path = output_dir / "initial_real_observation_1280x720.png"
    trace_path = output_dir / "trace.json"

    client = AtariBreakoutRPCClient(args.endpoint, timeout_ms=args.timeout_ms)
    records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    started = time.perf_counter()
    terminal_reason: str | None = None
    truncated = False
    native_return = 0.0
    best_progress = 0.0
    lives_lost = 0
    try:
        health = client.health()
        expected_health = {
            "environment": "gameworld_deterministic_breakout",
            "action_meanings": list(ACTION_NAMES),
            "timing": EXPECTED_ATARI_TIMING,
            "initial_lives": 5,
            "max_steps": None,
        }
        mismatches = {
            key: {"received": health.get(key), "expected": value}
            for key, value in expected_health.items()
            if health.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Browser service contract mismatch: {mismatches}")

        reset = client.reset(level=args.level, seed=args.game_seed, initial_lives=5)
        initial_path.write_bytes(reset.png)
        observation = frame_to_tensor(reset.png, device=device, size=64)
        hx = torch.zeros((1, agent.actor_critic.lstm_dim), device=device)
        cx = torch.zeros_like(hx)

        with ActionVideo(video_path, codec=args.video_codec) as video:
            step = 0
            while True:
                step += 1
                logits, value, (hx, cx) = agent.actor_critic.predict_act_value(
                    observation, (hx, cx)
                )
                distribution = torch.distributions.Categorical(logits=logits)
                action_tensor = (
                    logits.argmax(dim=-1)
                    if args.deterministic
                    else distribution.sample()
                )
                action = int(action_tensor.item())
                probabilities = distribution.probs[0].detach().cpu().tolist()
                value_float = float(value.item())
                entropy_bits = float(distribution.entropy().item() / math.log(2))

                rpc_started = time.perf_counter()
                result = client.step_record(action)
                rpc_elapsed_s = time.perf_counter() - rpc_started
                metadata = result.metadata
                frame_states = metadata["recording_frames"]
                for subframe, (png, frame_state) in enumerate(
                    zip(result.recorded_pngs, frame_states, strict=True), start=1
                ):
                    video.write(
                        png,
                        step=step,
                        action=action,
                        subframe=subframe,
                        frames_in_step=len(result.recorded_pngs),
                        probabilities=probabilities,
                        value=value_float,
                        frame_state=frame_state,
                    )

                events = _events(metadata)
                reward = float(events.get("positive_score_delta", 0.0))
                native_return += reward
                progress = float(
                    metadata.get("state", {})
                    .get("game_state", {})
                    .get("completion_progress", 0.0)
                )
                best_progress = max(best_progress, progress)
                if events.get("life_lost"):
                    lives_lost += 1
                action_counts[ACTION_NAMES[action]] += 1
                terminal_reason = (
                    "level_cleared"
                    if events.get("level_cleared")
                    else "game_over"
                    if events.get("game_over")
                    else None
                )
                records.append(
                    {
                        "step": step,
                        "action": action,
                        "action_name": ACTION_NAMES[action],
                        "probabilities": probabilities,
                        "policy_value": value_float,
                        "policy_entropy_bits": entropy_bits,
                        "reward": reward,
                        "native_return": native_return,
                        "completion_progress": progress,
                        "lives_lost_total": lives_lost,
                        "frames_executed": metadata.get("frames_executed"),
                        "game_time_advance_ms": metadata.get("game_time_advance_ms"),
                        "recorded_frame_count": len(result.recorded_pngs),
                        "rpc_wall_elapsed_s": rpc_elapsed_s,
                        "transition_events": events,
                    }
                )
                print(
                    f"step={step:04d} action={action}:{ACTION_NAMES[action]:5s} "
                    f"reward={reward:+.0f} progress={progress:.3f} "
                    f"lives_lost={lives_lost} frames={len(result.recorded_pngs)}",
                    flush=True,
                )
                if terminal_reason is not None:
                    break
                if args.safety_max_steps and step >= args.safety_max_steps:
                    truncated = True
                    terminal_reason = "diagnostic_safety_cap"
                    break
                observation = frame_to_tensor(result.png, device=device, size=64)

            num_video_frames = video.num_frames
            if video.preview is not None:
                cv2.imwrite(str(preview_path), video.preview)
    finally:
        client.close()

    summary = {
        "schema_version": 1,
        "mode": "diamond_training_time_real_environment_test",
        "official_gameworld_evaluation": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "device": str(device),
        "level": args.level,
        "game_seed": args.game_seed,
        "policy_seed": args.policy_seed,
        "policy": "argmax" if args.deterministic else "categorical_sample_epsilon_0",
        "frames_per_action": FRAMES_PER_ACTION,
        "num_steps": len(records),
        "num_video_frames": num_video_frames,
        "video_fps": VIDEO_FPS,
        "video_game_duration_s": num_video_frames / VIDEO_FPS,
        "video_codec": {
            "avc1": "H.264_AVC",
            "mp4v": "MPEG-4_Part_2",
            "FFV1": "FFV1_lossless",
        }[args.video_codec],
        "video_frame_source": "every_executed_60hz_browser_canvas_frame",
        "video_game_pixels": {
            "source_canvas": list(CANVAS_SIZE),
            "native_crop_xywh": list(VIDEO_CROP),
            "resized": False,
            "status_panel_height": STATUS_HEIGHT,
            "encoded_size": list(VIDEO_SIZE),
        },
        "initial_lives": 5,
        "done_on_life_loss": False,
        "native_return": native_return,
        "best_completion_progress": best_progress,
        "lives_lost": lives_lost,
        "success": terminal_reason == "level_cleared",
        "terminal_reason": terminal_reason,
        "truncated_by_safety_cap": truncated,
        "action_counts": dict(action_counts),
        "wall_elapsed_s": time.perf_counter() - started,
        "video": str(video_path),
        "preview": str(preview_path),
        "initial_observation": str(initial_path),
        "steps": records,
    }
    trace_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "num_steps",
                    "num_video_frames",
                    "video_game_duration_s",
                    "native_return",
                    "best_completion_progress",
                    "lives_lost",
                    "terminal_reason",
                    "video",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved trace: {trace_path}", flush=True)


if __name__ == "__main__":
    main()

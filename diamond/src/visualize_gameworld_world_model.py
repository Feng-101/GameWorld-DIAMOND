"""Inspect a GameWorld Breakout checkpoint inside DIAMOND's world model.

The viewer first uses the checkpoint policy in the real browser environment to
build the four-frame history expected by DIAMOND.  It then forks that state:
the real GameWorld environment and the learned world model receive exactly the
same actions.  A browser UI supports human or checkpoint-policy control, while
batch mode exports a GIF and a machine-readable JSON trace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import random
import threading
from typing import Any, Iterator, Mapping

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
import torch
from torch import Tensor

from agent import Agent
from data import Batch
from envs import GameWorldBreakoutEnv, WorldModelEnv


if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


ACTION_NAMES = ("wait", "left", "right")
PROFILE_OVERRIDES = {
    "mixed": ("env=gameworld_breakout", "+experiment=gameworld_breakout_formal"),
    "level5": (
        "env=gameworld_breakout_level5",
        "+experiment=gameworld_breakout_level5_atari",
    ),
}
PROFILE_DEFAULT_LEVEL = {"mixed": 1, "level5": 5}


def _torch_load(path: Path, *, map_location: torch.device) -> Any:
    """Load old and new PyTorch checkpoints without assuming a torch version."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _agent_state_dict(payload: Any) -> Mapping[str, Tensor]:
    """Accept an agent snapshot or Trainer's nested ``state.pt``."""

    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    state = payload.get("agent", payload)
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint field 'agent' must contain a state dictionary")

    normalized: dict[str, Tensor] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value

    required = ("denoiser.", "rew_end_model.", "actor_critic.")
    missing = [prefix for prefix in required if not any(k.startswith(prefix) for k in normalized)]
    if missing:
        raise ValueError(
            "Checkpoint is not a complete DIAMOND agent snapshot; missing prefixes "
            + ", ".join(missing)
        )
    return normalized


def _find_checkpoint_config(checkpoint: Path) -> Path | None:
    candidates = [checkpoint.parent / "trainer.yaml"]
    for parent in checkpoint.parents:
        candidates.extend((parent / "trainer.yaml", parent / "config" / "trainer.yaml"))
        if len(candidates) >= 10:
            break
    return next((path for path in candidates if path.is_file()), None)


def load_config(profile: str, checkpoint: Path, config_path: Path | None) -> tuple[DictConfig, Path | None]:
    discovered = config_path or _find_checkpoint_config(checkpoint)
    if discovered is not None:
        cfg = OmegaConf.load(discovered)
        OmegaConf.resolve(cfg)
        return cfg, discovered

    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="trainer", overrides=list(PROFILE_OVERRIDES[profile]))
    OmegaConf.resolve(cfg)
    return cfg, None


def validate_profile_config(profile: str, cfg: DictConfig) -> None:
    expected_protocol = {
        "mixed": "mixed_generalization",
        "level5": "level5_specialist",
    }[profile]
    actual_protocol = str(cfg.env.get("protocol", ""))
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"Checkpoint configuration/profile mismatch: --profile={profile!r} "
            f"expects protocol={expected_protocol!r}, found {actual_protocol!r}"
        )


class _OneBatchSampler:
    batch_size = 1


class RepeatingInitialConditionLoader:
    """The minimal DataLoader contract consumed by ``WorldModelEnv``."""

    def __init__(self, obs: Tensor, act: Tensor) -> None:
        self.batch_sampler = _OneBatchSampler()
        self._lock = threading.Lock()
        self.set_initial_condition(obs, act)

    def set_initial_condition(self, obs: Tensor, act: Tensor) -> None:
        if obs.ndim != 5 or obs.size(0) != 1:
            raise ValueError(f"Initial observations must have shape [1,L,C,H,W], got {tuple(obs.shape)}")
        if act.shape != obs.shape[:2]:
            raise ValueError(
                f"Initial actions must have shape {tuple(obs.shape[:2])}, got {tuple(act.shape)}"
            )
        with self._lock:
            self._obs = obs.detach().cpu().clone()
            self._act = act.detach().cpu().long().clone()

    def __iter__(self) -> Iterator[Batch]:
        while True:
            with self._lock:
                obs = self._obs.clone()
                act = self._act.clone()
            length = obs.size(1)
            yield Batch(
                obs=obs,
                act=act,
                rew=torch.zeros((1, length), dtype=torch.float32),
                end=torch.zeros((1, length), dtype=torch.long),
                trunc=torch.zeros((1, length), dtype=torch.long),
                mask_padding=torch.ones((1, length), dtype=torch.bool),
                info=[{}],
                segment_ids=[],
            )


@dataclass
class InitialCondition:
    obs: Tensor
    act: Tensor
    real_obs: Tensor
    warmup_steps: int
    warmup_boundaries: int


def _sample_action(logits: Tensor, *, deterministic: bool) -> Tensor:
    if deterministic:
        return logits.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logits).sample()


@torch.no_grad()
def collect_initial_condition(
    real_env: GameWorldBreakoutEnv,
    agent: Agent,
    *,
    level: int,
    game_seed: int,
    num_conditioning_steps: int,
    minimum_warmup_steps: int,
    deterministic: bool,
) -> InitialCondition:
    """Collect a contiguous policy-driven history while preserving real state."""

    obs, _ = real_env.reset(
        seed=[game_seed],
        options={"level": level, "game_seed": game_seed},
    )
    observations = [obs.detach().clone()]
    actions: list[Tensor] = []
    hx_cx = None
    total_steps = 0
    boundaries = 0
    maximum_steps = minimum_warmup_steps + 2 * real_env.max_episode_steps

    while total_steps < maximum_steps:
        logits, _, hx_cx = agent.actor_critic.predict_act_value(obs, hx_cx)
        action = _sample_action(logits, deterministic=deterministic)
        next_obs, _, end, trunc, _ = real_env.step(action)

        actions.append(action.detach().clone())
        observations.append(next_obs.detach().clone())
        total_steps += 1

        dead = bool(torch.logical_or(end, trunc).item())
        if dead:
            boundaries += 1
            observations = [next_obs.detach().clone()]
            actions = []
            hx_cx = None

        obs = next_obs
        if total_steps >= minimum_warmup_steps and len(observations) >= num_conditioning_steps:
            break
    else:
        raise RuntimeError(
            "Unable to obtain a contiguous world-model initialization history after "
            f"{maximum_steps} real GameWorld steps"
        )

    history_obs = torch.cat(observations[-num_conditioning_steps:], dim=0).unsqueeze(0)
    history_actions = actions[-(num_conditioning_steps - 1) :]
    action_tensor = torch.cat(history_actions, dim=0)
    action_tensor = torch.cat(
        (action_tensor, torch.zeros(1, dtype=torch.long, device=action_tensor.device))
    ).unsqueeze(0)
    return InitialCondition(
        obs=history_obs,
        act=action_tensor,
        real_obs=obs.detach().clone(),
        warmup_steps=total_steps,
        warmup_boundaries=boundaries,
    )


def tensor_to_image(obs: Tensor) -> Image.Image:
    if obs.ndim == 4:
        if obs.size(0) != 1:
            raise ValueError("Viewer supports exactly one observation")
        obs = obs[0]
    if obs.ndim != 3 or obs.size(0) != 3:
        raise ValueError(f"Expected CHW RGB observation, got {tuple(obs.shape)}")
    array = (
        obs.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def frame_metrics(real_obs: Tensor, model_obs: Tensor) -> tuple[float, float]:
    real = real_obs.detach().float().cpu().add(1).div(2)
    model = model_obs.detach().float().cpu().add(1).div(2)
    mse = float(torch.mean((real - model) ** 2).item())
    mae = float(torch.mean(torch.abs(real - model)).item())
    psnr = 99.0 if mse == 0 else min(99.0, 10.0 * math.log10(1.0 / mse))
    return mae, psnr


def render_comparison(
    model_obs: Tensor,
    real_obs: Tensor | None,
    *,
    panel_size: int,
    title: str,
    subtitle: str,
) -> Image.Image:
    nearest = getattr(Image, "Resampling", Image).NEAREST
    model_image = tensor_to_image(model_obs).resize((panel_size, panel_size), nearest)
    images = [model_image] if real_obs is None else [
        tensor_to_image(real_obs).resize((panel_size, panel_size), nearest),
        model_image,
    ]
    labels = ["WORLD MODEL"] if real_obs is None else ["REAL GAMEWORLD", "WORLD MODEL"]
    header_height = 58
    canvas = Image.new("RGB", (panel_size * len(images), panel_size + header_height), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * panel_size
        canvas.paste(image, (x, header_height))
        draw.text((x + 6, 5), label, fill="white")
    draw.text((6, 23), title[:120], fill=(255, 210, 80))
    draw.text((6, 40), subtitle[:180], fill=(190, 220, 255))
    return canvas


class WorldModelViewerSession:
    def __init__(
        self,
        *,
        agent: Agent,
        cfg: DictConfig,
        real_env: GameWorldBreakoutEnv,
        profile: str,
        checkpoint: Path,
        level: int,
        game_seed: int,
        controller: str,
        deterministic: bool,
        compare_real: bool,
        warmup_steps: int,
        rollout_horizon: int,
        panel_size: int,
        fps: int,
        output_dir: Path,
    ) -> None:
        self.agent = agent
        self.cfg = cfg
        self.real_env = real_env
        self.profile = profile
        self.checkpoint = checkpoint
        self.level = level
        self.game_seed = game_seed
        self.controller = controller
        self.deterministic = deterministic
        self.compare_real = compare_real
        self.warmup_steps = warmup_steps
        self.rollout_horizon = rollout_horizon
        self.panel_size = panel_size
        self.fps = fps
        self.output_dir = output_dir
        self.lock = threading.RLock()
        self.num_conditioning_steps = int(cfg.agent.denoiser.inner_model.num_steps_conditioning)
        self.initial_loader: RepeatingInitialConditionLoader | None = None
        self.world_env: WorldModelEnv | None = None
        self.model_obs: Tensor | None = None
        self.real_obs: Tensor | None = None
        self.policy_hx_cx = None
        self.episode_return_model = 0.0
        self.episode_return_real = 0.0
        self.episode_step = 0
        self.episode_id = -1
        self.terminal = False
        self.frames: list[Image.Image] = []
        self.records: list[dict[str, Any]] = []
        self.current_panel: Image.Image | None = None
        self._closed = False
        self.reset()

    def _new_initial_condition(self) -> InitialCondition:
        return collect_initial_condition(
            self.real_env,
            self.agent,
            level=self.level,
            game_seed=self.game_seed + self.episode_id,
            num_conditioning_steps=self.num_conditioning_steps,
            minimum_warmup_steps=self.warmup_steps,
            deterministic=self.deterministic,
        )

    @torch.no_grad()
    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.episode_id += 1
            initial = self._new_initial_condition()
            if self.initial_loader is None:
                self.initial_loader = RepeatingInitialConditionLoader(initial.obs, initial.act)
                wm_cfg = instantiate(
                    self.cfg.world_model_env,
                    horizon=self.rollout_horizon,
                    num_batches_to_preload=1,
                )
                self.world_env = WorldModelEnv(
                    self.agent.denoiser,
                    self.agent.rew_end_model,
                    self.initial_loader,
                    wm_cfg,
                )
            else:
                self.initial_loader.set_initial_condition(initial.obs, initial.act)

            assert self.world_env is not None
            self.model_obs, _ = self.world_env.reset()
            self.real_obs = initial.real_obs
            self.policy_hx_cx = None
            self.episode_return_model = 0.0
            self.episode_return_real = 0.0
            self.episode_step = 0
            self.terminal = False
            self.current_panel = render_comparison(
                self.model_obs,
                self.real_obs if self.compare_real else None,
                panel_size=self.panel_size,
                title=(
                    f"{self.profile} | level {self.level} | checkpoint {self.checkpoint.name}"
                ),
                subtitle=(
                    f"reset | real warm-up {initial.warmup_steps} steps "
                    f"({initial.warmup_boundaries} boundaries)"
                ),
            )
            self.frames.append(self.current_panel.copy())
            self.records.append(
                {
                    "event": "reset",
                    "episode_id": self.episode_id,
                    "level": self.level,
                    "game_seed": self.game_seed + self.episode_id,
                    "warmup_steps": initial.warmup_steps,
                    "warmup_boundaries": initial.warmup_boundaries,
                }
            )
            return self.state()

    def set_controller(self, controller: str) -> dict[str, Any]:
        if controller not in {"human", "agent"}:
            raise ValueError("controller must be 'human' or 'agent'")
        self.controller = controller
        return self.reset()

    @torch.no_grad()
    def step(self, requested_action: int | None = None) -> dict[str, Any]:
        with self.lock:
            if self.terminal:
                raise RuntimeError("Rollout is terminal; reset before stepping again")
            assert self.model_obs is not None and self.real_obs is not None
            assert self.world_env is not None

            entropy: float | None = None
            value_float: float | None = None
            if self.controller == "agent":
                logits, value, self.policy_hx_cx = self.agent.actor_critic.predict_act_value(
                    self.model_obs, self.policy_hx_cx
                )
                distribution = torch.distributions.Categorical(logits=logits)
                action = logits.argmax(dim=-1) if self.deterministic else distribution.sample()
                entropy = float((distribution.entropy() / math.log(2)).item())
                value_float = float(value.item())
            else:
                if requested_action not in range(3):
                    raise ValueError("Human action must be 0=wait, 1=left or 2=right")
                action = torch.tensor([requested_action], device=self.agent.device)

            model_next, model_rew, model_end, model_trunc, model_info = self.world_env.step(action)
            model_dead = bool(torch.logical_or(model_end, model_trunc).item())
            model_frame = (
                model_info["final_observation"] if model_dead else model_next
            )

            if self.compare_real:
                real_next, real_rew, real_end, real_trunc, real_info = self.real_env.step(action)
                real_dead = bool(torch.logical_or(real_end, real_trunc).item())
                real_frame = real_info.get("final_observation", real_next) if real_dead else real_next
            else:
                real_next = self.real_obs
                real_rew = torch.zeros_like(model_rew)
                real_end = torch.zeros_like(model_end)
                real_trunc = torch.zeros_like(model_trunc)
                real_dead = False
                real_frame = None

            mae, psnr = (
                frame_metrics(real_frame, model_frame)
                if real_frame is not None
                else (None, None)
            )
            self.episode_step += 1
            self.episode_return_model += float(model_rew.item())
            self.episode_return_real += float(real_rew.item())
            self.terminal = model_dead or real_dead
            self.model_obs = model_next
            self.real_obs = real_next

            action_index = int(action.item())
            record = {
                "event": "step",
                "episode_id": self.episode_id,
                "step": self.episode_step,
                "controller": self.controller,
                "action": action_index,
                "action_name": ACTION_NAMES[action_index],
                "policy_entropy_bits": entropy,
                "policy_value": value_float,
                "model_reward": float(model_rew.item()),
                "real_reward": float(real_rew.item()) if self.compare_real else None,
                "model_end": bool(model_end.item()),
                "model_trunc": bool(model_trunc.item()),
                "real_end": bool(real_end.item()) if self.compare_real else None,
                "real_trunc": bool(real_trunc.item()) if self.compare_real else None,
                "pixel_mae_0_1": mae,
                "psnr_db": psnr,
                "terminal": self.terminal,
            }
            self.records.append(record)
            self.current_panel = render_comparison(
                model_frame,
                real_frame,
                panel_size=self.panel_size,
                title=(
                    f"{self.profile} L{self.level} | {self.controller} | "
                    f"step {self.episode_step} | action {ACTION_NAMES[action_index]}"
                ),
                subtitle=(
                    f"reward real/model={record['real_reward']}/{record['model_reward']} | "
                    f"PSNR={psnr:.2f} dB MAE={mae:.4f}"
                    if psnr is not None and mae is not None
                    else f"model reward={record['model_reward']}"
                ),
            )
            self.frames.append(self.current_panel.copy())
            return self.state()

    def state(self) -> dict[str, Any]:
        latest = self.records[-1] if self.records else {}
        return {
            "profile": self.profile,
            "checkpoint": str(self.checkpoint),
            "level": self.level,
            "controller": self.controller,
            "deterministic": self.deterministic,
            "compare_real": self.compare_real,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "model_return": self.episode_return_model,
            "real_return": self.episode_return_real if self.compare_real else None,
            "terminal": self.terminal,
            "latest": latest,
        }

    def panel_png(self) -> bytes:
        with self.lock:
            if self.current_panel is None:
                raise RuntimeError("Viewer has no current frame")
            output = BytesIO()
            self.current_panel.save(output, format="PNG")
            return output.getvalue()

    def save(self) -> tuple[Path, Path]:
        with self.lock:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            gif_path = self.output_dir / "rollout.gif"
            json_path = self.output_dir / "trace.json"
            if self.frames:
                self.frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=self.frames[1:],
                    duration=max(1, round(1000 / self.fps)),
                    loop=0,
                    optimize=False,
                )

            step_records = [record for record in self.records if record["event"] == "step"]
            paired = [record for record in step_records if record["psnr_db"] is not None]
            summary = {
                "num_steps": len(step_records),
                "num_episodes_started": self.episode_id + 1,
                "mean_psnr_db": (
                    float(np.mean([record["psnr_db"] for record in paired]))
                    if paired
                    else None
                ),
                "mean_pixel_mae_0_1": (
                    float(np.mean([record["pixel_mae_0_1"] for record in paired]))
                    if paired
                    else None
                ),
                "reward_sign_accuracy": (
                    float(
                        np.mean(
                            [
                                np.sign(record["model_reward"])
                                == np.sign(record["real_reward"])
                                for record in paired
                            ]
                        )
                    )
                    if paired
                    else None
                ),
                "termination_accuracy": (
                    float(
                        np.mean(
                            [
                                (record["model_end"] or record["model_trunc"])
                                == (record["real_end"] or record["real_trunc"])
                                for record in paired
                            ]
                        )
                    )
                    if paired
                    else None
                ),
            }
            payload = {
                "schema_version": 1,
                "checkpoint": str(self.checkpoint.resolve()),
                "checkpoint_sha256": sha256(self.checkpoint.read_bytes()).hexdigest(),
                "profile": self.profile,
                "level": self.level,
                "base_game_seed": self.game_seed,
                "controller": self.controller,
                "deterministic": self.deterministic,
                "compare_real": self.compare_real,
                "rollout_horizon": self.rollout_horizon,
                "summary": summary,
                "records": self.records,
            }
            json_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return gif_path, json_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.real_env.close()


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GameWorld DIAMOND World Model</title>
  <style>
    body { background:#111; color:#eee; font:16px system-ui,sans-serif; margin:24px; }
    main { max-width:1100px; margin:auto; }
    img { width:100%; image-rendering:pixelated; border:1px solid #555; background:#000; }
    button { font-size:16px; margin:6px 3px; padding:9px 16px; }
    code,pre { background:#1c1c1c; padding:8px; white-space:pre-wrap; }
    .hint { color:#b9d9ff; }
    .terminal { color:#ffcf6b; }
  </style>
</head>
<body><main>
  <h2>GameWorld Breakout — DIAMOND world model</h2>
  <img id="frame" alt="real and imagined Breakout frames">
  <div>
    <button onclick="step(1)">← / A: left</button>
    <button onclick="step(0)">Space: wait</button>
    <button onclick="step(2)">→ / D: right</button>
    <button onclick="step(null)">Agent step</button>
    <button onclick="resetRollout()">R: reset</button>
    <button onclick="toggleController()">M: human/agent</button>
    <label><input id="auto" type="checkbox" onchange="autoChanged()"> agent autoplay</label>
  </div>
  <p class="hint">The policy always observes the WORLD MODEL panel. Both panels receive the same action.</p>
  <p id="terminal" class="terminal"></p>
  <pre id="state"></pre>
</main>
<script>
let state = null;
let running = false;
const frame = document.getElementById("frame");
const stateBox = document.getElementById("state");
const terminalBox = document.getElementById("terminal");

async function request(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type":"application/json"}, body:JSON.stringify(body)
  };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
async function refresh() {
  state = await request("/api/state");
  frame.src = "/frame.png?t=" + Date.now();
  stateBox.textContent = JSON.stringify(state, null, 2);
  terminalBox.textContent = state.terminal ? "Rollout ended. Press R to collect a fresh real history." : "";
}
async function step(action) {
  if (running || (state && state.terminal)) return;
  running = true;
  try {
    await request("/api/step", {action:action});
    await refresh();
  } catch (error) {
    terminalBox.textContent = error.message;
  } finally {
    running = false;
  }
}
async function resetRollout() {
  if (running) return;
  running = true;
  terminalBox.textContent = "Collecting a new real initialization history...";
  try { await request("/api/reset", {}); await refresh(); }
  finally { running = false; }
}
async function toggleController() {
  if (running) return;
  const next = state.controller === "agent" ? "human" : "agent";
  running = true;
  terminalBox.textContent = "Switching controller and resetting...";
  try { await request("/api/controller", {controller:next}); await refresh(); }
  finally { running = false; }
}
async function autoLoop() {
  if (!document.getElementById("auto").checked) return;
  if (state && state.controller === "agent" && !state.terminal) await step(null);
  setTimeout(autoLoop, 250);
}
function autoChanged() { if (document.getElementById("auto").checked) autoLoop(); }
document.addEventListener("keydown", event => {
  if (event.repeat) return;
  if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") step(1);
  else if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") step(2);
  else if (event.key === " " || event.key.toLowerCase() === "s") step(0);
  else if (event.key.toLowerCase() === "r") resetRollout();
  else if (event.key.toLowerCase() === "m") toggleController();
});
refresh();
</script></body></html>"""


class ViewerRequestHandler(BaseHTTPRequestHandler):
    session: WorldModelViewerSession

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[viewer] {self.address_string()} - {fmt % args}", flush=True)

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/" or self.path.startswith("/index.html"):
                self._send_bytes(VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/frame.png"):
                self._send_bytes(self.session.panel_png(), "image/png")
            elif self.path.startswith("/api/state"):
                self._send_json(self.session.state())
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/step":
                raw_action = payload.get("action")
                action = None if raw_action is None else int(raw_action)
                result = self.session.step(action)
            elif self.path == "/api/reset":
                result = self.session.reset()
            elif self.path == "/api/controller":
                result = self.session.set_controller(str(payload.get("controller")))
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
        except (ValueError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_OVERRIDES), required=True)
    parser.add_argument("--config", type=Path, help="Resolved trainer.yaml; auto-discovered when omitted")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5581")
    parser.add_argument("--level", type=int)
    parser.add_argument("--game-seed", type=int, default=4242)
    parser.add_argument("--initial-lives", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int)
    parser.add_argument("--controller", choices=("human", "agent"), default="agent")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of policy sampling")
    parser.add_argument(
        "--no-compare-real",
        dest="compare_real",
        action="store_false",
        help="Stop stepping the real environment after initialization",
    )
    parser.set_defaults(compare_real=True)
    parser.add_argument("--ui", choices=("web", "batch"), default="web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--steps", type=int, default=100, help="Total generated steps in batch mode")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--seed", type=int, default=20260715, help="Model/policy sampling seed")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.level is not None and args.level not in range(1, 6):
        raise ValueError("--level must be between 1 and 5")
    if args.initial_lives not in range(1, 6):
        raise ValueError("--initial-lives must be between 1 and 5")
    for name in ("max_episode_steps", "warmup_steps", "steps", "fps", "panel_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.rollout_horizon is not None and args.rollout_horizon < 1:
        raise ValueError("--rollout-horizon must be positive")
    if args.ui == "batch" and args.controller != "agent":
        raise ValueError("Batch mode requires --controller agent")


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def build_session(args: argparse.Namespace) -> tuple[WorldModelViewerSession, Path | None]:
    cfg, config_path = load_config(args.profile, args.checkpoint, args.config)
    validate_profile_config(args.profile, cfg)
    device = _choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    agent = Agent(instantiate(cfg.agent, num_actions=3)).to(device).eval()
    payload = _torch_load(args.checkpoint, map_location=device)
    agent.load_state_dict(_agent_state_dict(payload), strict=True)

    level = args.level if args.level is not None else PROFILE_DEFAULT_LEVEL[args.profile]
    rollout_horizon = (
        args.rollout_horizon
        if args.rollout_horizon is not None
        else int(cfg.world_model_env.horizon)
    )
    real_env = GameWorldBreakoutEnv(
        device=device,
        endpoint=args.endpoint,
        num_envs=1,
        size=int(cfg.env.train.size),
        max_episode_steps=args.max_episode_steps,
        levels=[level],
        level_strategy="cycle",
        game_seeds=[args.game_seed],
        initial_lives=args.initial_lives,
        max_task_life_losses=args.initial_lives,
        done_on_life_loss=True,
        timeout_ms=int(cfg.env.train.get("timeout_ms", 30_000)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path("world_model_visualizations")
        / f"{args.profile}_{args.checkpoint.stem}_level{level}_{timestamp}"
    )
    try:
        session = WorldModelViewerSession(
            agent=agent,
            cfg=cfg,
            real_env=real_env,
            profile=args.profile,
            checkpoint=args.checkpoint,
            level=level,
            game_seed=args.game_seed,
            controller=args.controller,
            deterministic=args.deterministic,
            compare_real=args.compare_real,
            warmup_steps=args.warmup_steps,
            rollout_horizon=rollout_horizon,
            panel_size=args.panel_size,
            fps=args.fps,
            output_dir=output_dir,
        )
    except Exception:
        real_env.close()
        raise
    return session, config_path


def run_batch(session: WorldModelViewerSession, steps: int) -> None:
    generated = 0
    while generated < steps:
        if session.terminal:
            session.reset()
        session.step()
        generated += 1
        if generated % 10 == 0 or generated == steps:
            print(f"Generated {generated}/{steps} imagined steps", flush=True)


def run_web(session: WorldModelViewerSession, host: str, port: int) -> None:
    handler = type("BoundViewerRequestHandler", (ViewerRequestHandler,), {"session": session})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"World-model viewer: http://{host}:{port}", flush=True)
    print("Forward this port in VSCode when the server is remote.", flush=True)
    print("Press Ctrl+C to save the current GIF/trace and stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("Viewer interrupted.", flush=True)
    finally:
        server.server_close()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    session: WorldModelViewerSession | None = None
    try:
        session, config_path = build_session(args)
        print(f"Loaded checkpoint: {args.checkpoint.resolve()}")
        print(f"Configuration: {config_path.resolve() if config_path else 'repository profile defaults'}")
        print(
            f"Profile={args.profile} level={session.level} controller={args.controller} "
            f"horizon={session.rollout_horizon} compare_real={session.compare_real}"
        )
        if args.ui == "batch":
            run_batch(session, args.steps)
        else:
            run_web(session, args.host, args.port)
    finally:
        if session is not None:
            try:
                gif_path, trace_path = session.save()
                print(f"Saved GIF: {gif_path.resolve()}")
                print(f"Saved trace: {trace_path.resolve()}")
            finally:
                session.close()


if __name__ == "__main__":
    main()

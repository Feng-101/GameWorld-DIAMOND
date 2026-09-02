from __future__ import annotations

from io import BytesIO
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_gameworld_real_agent import evaluate_one_task  # noqa: E402
from integrations.gameworld import (  # noqa: E402
    EXPECTED_EVALUATION_TIMING,
    RPCObservation,
)
from view_gameworld_real_steps import RealStepSession  # noqa: E402


def _raw_png(value: int = 80) -> bytes:
    array = np.full((720, 1280, 3), value, dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(array, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _metadata(*, event: str, step: int, terminal: bool = False) -> dict:
    payload = {
        "ok": True,
        "event": event,
        "step_count": step,
        "timing": dict(EXPECTED_EVALUATION_TIMING),
        "state": {
            "game_state": {
                "level": 1,
                "score": step * 5,
                "completion_progress": step / 10,
            },
            "metrics": {"lives": 3},
        },
    }
    if event == "step":
        payload["transition_events"] = {
            "task_success": terminal,
            "task_time_limit": False,
            "terminal_failure": False,
        }
    return payload


class FakeRPCClient:
    def __init__(self) -> None:
        self.actions: list[int] = []
        self.resets: list[tuple[int, int, int]] = []
        self.closed = False

    def health(self):
        return {
            "ok": True,
            "viewport": [1280, 720],
            "timing": dict(EXPECTED_EVALUATION_TIMING),
        }

    def reset(self, *, level: int, seed: int, initial_lives: int):
        self.resets.append((level, seed, initial_lives))
        metadata = _metadata(event="reset", step=0)
        metadata["level"] = level
        metadata["seed"] = seed
        metadata["initial_lives"] = initial_lives
        metadata["state"]["game_state"]["level"] = level
        metadata["state"]["metrics"]["lives"] = initial_lives
        return RPCObservation(metadata, _raw_png(70 + level))

    def step(self, action: int):
        self.actions.append(action)
        step = len(self.actions)
        return RPCObservation(
            _metadata(event="step", step=step, terminal=step == 2),
            _raw_png(90 + step),
        )

    def close(self):
        self.closed = True


class RealStepSessionTests(unittest.TestCase):
    def test_displays_preprocessed_agent_frames_and_switches_level(self) -> None:
        client = FakeRPCClient()
        session = RealStepSession(
            client,
            level=1,
            game_seed=42,
            initial_lives=3,
        )
        try:
            image = Image.open(BytesIO(session.get_frame_png()))
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(client.resets, [(1, 42, 3)])

            state = session.step(2)
            self.assertEqual(client.actions, [2])
            self.assertEqual(state["last_action_name"], "right")
            self.assertEqual(state["step_count"], 1)
            self.assertFalse(state["terminal"])

            state = session.step(0)
            self.assertTrue(state["terminal"])
            with self.assertRaisesRegex(RuntimeError, "reset"):
                session.step(1)

            reset_state = session.reset(level=5, game_seed=123, initial_lives=5)
            self.assertEqual(client.resets[-1], (5, 123, 5))
            self.assertEqual(reset_state["level"], 5)
            self.assertFalse(reset_state["terminal"])
        finally:
            session.close()
        self.assertTrue(client.closed)


class RecordingActor:
    lstm_dim = 4

    def __init__(self) -> None:
        self.observed_means: list[float] = []

    def predict_act_value(self, observation, hx_cx):
        self.observed_means.append(float(observation.mean().item()))
        logits = torch.tensor([[5.0, 0.0, 0.0]])
        value = torch.tensor([0.25])
        return logits, value, hx_cx


class FakeAgent:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.actor_critic = RecordingActor()


class FakeRealEnv:
    initial_lives = 3
    max_episode_steps = 3

    def __init__(self) -> None:
        self.step_id = 0

    def reset(self, **kwargs):
        self.step_id = 0
        return torch.zeros((1, 3, 64, 64)), {
            "level": 1,
            "seed": 42,
            "initial_lives": 3,
        }

    def step(self, action):
        self.step_id += 1
        observation = torch.full(
            (1, 3, 64, 64),
            self.step_id / 10,
        )
        end = torch.tensor([False], dtype=torch.uint8)
        trunc = torch.tensor([self.step_id == self.max_episode_steps], dtype=torch.uint8)
        info = {
            "task_best_completion_progress": self.step_id / 10,
            "gameworld": {
                "transition_events": {
                    "life_lost": self.step_id == 2,
                    "task_success": False,
                }
            },
        }
        if bool(trunc.item()):
            info["final_observation"] = observation
            info["boundary_reason"] = "task_time_limit"
        return observation, torch.tensor([5.0]), end, trunc, info


class RealAgentVideoTests(unittest.TestCase):
    def test_actor_observes_only_successive_real_frames_and_writes_mp4(self) -> None:
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = evaluate_one_task(
                env=FakeRealEnv(),
                agent=agent,
                level=1,
                game_seed=42,
                task_index=0,
                deterministic=True,
                policy_seed=7,
                output_dir=output_dir,
                video_fps=5.0,
                video_size=128,
                video_codec="mp4v",
            )
            np.testing.assert_allclose(
                agent.actor_critic.observed_means,
                [0.0, 0.1, 0.2],
                atol=1e-6,
            )
            self.assertEqual(result["num_steps"], 3)
            self.assertEqual(result["num_video_frames"], 4)
            self.assertAlmostEqual(result["video_frame_interval_s"], 0.2)

            capture = cv2.VideoCapture(result["video"])
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 5.0)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 128)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 128)
            finally:
                capture.release()
            self.assertTrue(Path(result["trace"]).is_file())


if __name__ == "__main__":
    unittest.main()

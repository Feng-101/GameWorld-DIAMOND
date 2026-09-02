"""Unit tests for the DIAMOND Breakout browser bridge."""

from __future__ import annotations

import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.diamond_bridge.breakout_protocol import (
    ACTION_DURATION_S,
    NOMINAL_OBSERVATION_INTERVAL_S,
    POST_ACTION_IDLE_S,
    action_payload,
    evaluation_timing,
    execute_step_with_evaluation_cadence,
    transition_events,
)

try:
    from env.browser_manager import (
        BrowserConfig,
        BrowserGameManager,
        CDPScreenshotter,
        ScreenshotConfig,
    )
    from env.action_executor import ActionExecutor
except ModuleNotFoundError as exc:
    if exc.name != "playwright":
        raise
    BrowserConfig = None  # type: ignore[assignment,misc]
    BrowserGameManager = None  # type: ignore[assignment,misc]
    CDPScreenshotter = None  # type: ignore[assignment,misc]
    ScreenshotConfig = None  # type: ignore[assignment,misc]
    ActionExecutor = None  # type: ignore[assignment,misc]

HAS_PLAYWRIGHT = BrowserGameManager is not None


def make_state(
    *,
    level: int = 1,
    score: int = 0,
    lives: int = 3,
    bricks: int = 10,
    terminal: bool = False,
    outcome: str | None = None,
) -> dict:
    return {
        "game_state": {"level": level, "score": score},
        "metrics": {"lives": lives, "bricks_remaining": bricks},
        "terminal": {"isTerminal": terminal, "outcome": outcome},
    }


class BreakoutProtocolTests(unittest.TestCase):
    def test_evaluation_timing_keeps_hold_and_post_action_idle_distinct(self) -> None:
        self.assertEqual(ACTION_DURATION_S, 0.2)
        self.assertEqual(POST_ACTION_IDLE_S, 0.05)
        self.assertEqual(NOMINAL_OBSERVATION_INTERVAL_S, 0.25)
        self.assertEqual(
            evaluation_timing(),
            {
                "action_hold_s": 0.2,
                "post_action_idle_s": 0.05,
                "nominal_observation_interval_s": 0.25,
                "evaluation_state_after_action": True,
                "screenshot_before_pause": True,
                "observation_state_after_pause": True,
            },
        )

    def test_action_space_maps_to_only_legal_breakout_actions(self) -> None:
        self.assertEqual(action_payload(0), {"action": "wait", "duration": 0.2})
        self.assertEqual(
            action_payload(1),
            {"action": "press_key", "key": "ArrowLeft", "duration": 0.2},
        )
        self.assertEqual(
            action_payload(2),
            {"action": "press_key", "key": "ArrowRight", "duration": 0.2},
        )
        for invalid in (-1, 3, True, None, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                action_payload(invalid)  # type: ignore[arg-type]

    def test_native_score_delta_and_brick_count_are_preserved(self) -> None:
        events = transition_events(
            make_state(bricks=10),
            make_state(bricks=8, score=35),
            step_count=1,
        )
        self.assertEqual(events.score_delta, 35.0)
        self.assertEqual(events.positive_score_delta, 35.0)
        self.assertEqual(events.bricks_destroyed, 2)
        self.assertTrue(events.brick_hit)
        self.assertFalse(events.task_success)

    def test_ordinary_life_loss_is_an_event_without_reward_policy(self) -> None:
        events = transition_events(
            make_state(lives=3),
            make_state(lives=2),
            step_count=8,
        )
        self.assertEqual(events.positive_score_delta, 0.0)
        self.assertTrue(events.life_lost)
        self.assertFalse(events.last_life_reset)
        self.assertFalse(events.task_success)

    def test_zero_score_last_life_implicit_engine_reset_is_detected(self) -> None:
        events = transition_events(
            make_state(lives=1, score=0, bricks=10),
            make_state(lives=3, score=0, bricks=10),
            step_count=40,
        )
        self.assertEqual(events.score_delta, 0.0)
        self.assertTrue(events.life_lost)
        self.assertTrue(events.last_life_reset)

    def test_score_reset_is_reported_but_not_turned_into_negative_reward(self) -> None:
        events = transition_events(
            make_state(lives=1, score=120, bricks=3),
            make_state(lives=3, score=0, bricks=10),
            step_count=40,
        )
        self.assertEqual(events.score_delta, -120.0)
        self.assertEqual(events.positive_score_delta, 0.0)
        self.assertTrue(events.life_lost)
        self.assertTrue(events.last_life_reset)

    def test_gameworld_failure_is_reported_without_choosing_episode_policy(self) -> None:
        events = transition_events(
            make_state(lives=1),
            make_state(lives=0, terminal=True, outcome="fail"),
            step_count=50,
        )
        self.assertEqual(events.positive_score_delta, 0.0)
        self.assertTrue(events.life_lost)
        self.assertTrue(events.terminal_failure)
        self.assertFalse(events.task_success)

    def test_success_has_priority_over_time_limit(self) -> None:
        events = transition_events(
            make_state(level=1, bricks=1),
            make_state(level=1, bricks=0, score=50, terminal=True, outcome="success"),
            step_count=100,
        )
        self.assertEqual(events.positive_score_delta, 50.0)
        self.assertTrue(events.task_success)
        self.assertFalse(events.task_time_limit)

    def test_managed_same_level_terminal_success_is_detected(self) -> None:
        events = transition_events(
            make_state(level=4, bricks=1),
            make_state(
                level=4,
                bricks=0,
                score=50,
                terminal=True,
                outcome="success",
            ),
            step_count=27,
            max_steps=200,
        )
        self.assertTrue(events.task_success)
        self.assertEqual(events.bricks_destroyed, 1)
        self.assertEqual(events.positive_score_delta, 50.0)

    def test_step_limit_is_reported_without_live_browser_run(self) -> None:
        events = transition_events(make_state(), make_state(), step_count=100)
        self.assertFalse(events.task_success)
        self.assertTrue(events.task_time_limit)


class EvaluationCadenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_order_matches_evaluator_then_adds_paused_boundary_state(self) -> None:
        events: list[object] = []

        async def resume() -> None:
            events.append("resume")

        async def execute() -> None:
            events.append("execute_0.2s_action")

        async def sleep(duration: float) -> None:
            events.append(("idle", duration))

        async def evaluation_state() -> str:
            events.append("evaluation_state")
            return "evaluation"

        async def frame() -> str:
            events.append("screenshot")
            return "pixels"

        async def pause() -> None:
            events.append("pause")

        async def observation_state() -> str:
            events.append("observation_state")
            return "observation"

        captured = await execute_step_with_evaluation_cadence(
            resume_game=resume,
            execute_action=execute,
            capture_evaluation_state=evaluation_state,
            capture_frame=frame,
            pause_game=pause,
            capture_observation_state=observation_state,
            sleep=sleep,
        )

        self.assertEqual(captured, ("evaluation", "observation", "pixels"))
        self.assertEqual(
            events,
            [
                "resume",
                "execute_0.2s_action",
                "evaluation_state",
                ("idle", 0.05),
                "screenshot",
                "pause",
                "observation_state",
            ],
        )

    async def test_execution_failure_still_pauses_without_snapshot(self) -> None:
        events: list[str] = []

        async def resume() -> None:
            events.append("resume")

        async def execute() -> None:
            events.append("execute")
            raise RuntimeError("synthetic failure")

        async def pause() -> None:
            events.append("pause")

        async def capture() -> None:
            events.append("capture")

        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            await execute_step_with_evaluation_cadence(
                resume_game=resume,
                execute_action=execute,
                capture_evaluation_state=capture,
                capture_frame=capture,
                pause_game=pause,
                capture_observation_state=capture,
            )

        self.assertEqual(events, ["resume", "execute", "pause"])


class FakePage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def evaluate(self, script: str, argument: object = None) -> bool:
        self.calls.append((script, argument))
        return True


class FakeCDPSession:
    def __init__(self, png: bytes) -> None:
        self.png = png
        self.calls: list[tuple[str, dict]] = []

    async def send(self, method: str, options: dict) -> dict[str, str]:
        self.calls.append((method, options))
        return {"data": base64.b64encode(self.png).decode("ascii")}

    async def detach(self) -> None:
        return None


@unittest.skipUnless(HAS_PLAYWRIGHT, "requires the GameWorld Playwright environment")
class BrowserManagerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_executor_does_not_require_custom_logging_bootstrap(self) -> None:
        executor = ActionExecutor(page=object(), controls=None)  # type: ignore[arg-type,misc]
        await executor.execute({"action": "wait", "duration": 0})

    async def test_reset_forwards_level_seed_and_lives_to_game_api(self) -> None:
        manager = BrowserGameManager(BrowserConfig(game_url="http://unused"))
        page = FakePage()
        manager.page = page  # type: ignore[assignment]

        request = {"level": 5, "seed": 1234, "initial_lives": 5}
        self.assertTrue(await manager.reset_game(request))
        self.assertEqual(page.calls[0][1], request)

    async def test_in_memory_screenshot_is_normalized_without_file_output(self) -> None:
        source = BytesIO()
        Image.new("RGB", (2, 2), color=(255, 0, 0)).save(source, format="PNG")
        fake_session = FakeCDPSession(source.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            screenshotter = CDPScreenshotter(
                ScreenshotConfig(width=4, height=3, screenshot_dir=Path(directory))
            )

            async def new_session() -> FakeCDPSession:
                return fake_session

            png = await screenshotter.capture_bytes(
                context=object(),  # type: ignore[arg-type]
                page=object(),  # type: ignore[arg-type]
                new_cdp_session=new_session,  # type: ignore[arg-type]
            )
            with Image.open(BytesIO(png)) as image:
                self.assertEqual(image.size, (4, 3))
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(fake_session.calls[0][0], "Page.captureScreenshot")


if __name__ == "__main__":
    unittest.main()

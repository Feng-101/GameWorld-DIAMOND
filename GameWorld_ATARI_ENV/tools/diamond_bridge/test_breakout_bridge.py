from __future__ import annotations

import json
import unittest

from PIL import Image, ImageChops

from .breakout_protocol import (
    ACTION_MEANINGS,
    FRAME_SKIP,
    GAME_TIME_PER_STEP_S,
    action_key,
    atari_timing,
    transition_events,
)
from .browser_env_server import (
    PROTOCOL_VERSION,
    BrowserEnvironmentRPCServer,
)


def state(
    *,
    score: float = 0,
    lives: int = 5,
    bricks: int = 10,
    total_bricks: int = 10,
    level: int = 5,
    terminal: tuple[bool, str | None, str | None] = (False, None, None),
) -> dict:
    is_terminal, outcome, reason = terminal
    return {
        "game_state": {
            "score": score,
            "level": level,
            "completion_progress": 1 - bricks / total_bricks,
        },
        "metrics": {
            "lives": lives,
            "bricks_remaining": bricks,
            "bricks_total": total_bricks,
        },
        "terminal": {
            "isTerminal": is_terminal,
            "outcome": outcome,
            "reason": reason,
        },
    }


class AtariProtocolTests(unittest.TestCase):
    def test_action_order_matches_ale_breakout(self) -> None:
        self.assertEqual(ACTION_MEANINGS, ("NOOP", "FIRE", "RIGHT", "LEFT"))
        self.assertIsNone(action_key(0))
        self.assertEqual(action_key(1), "Space")
        self.assertEqual(action_key(2), "ArrowRight")
        self.assertEqual(action_key(3), "ArrowLeft")
        with self.assertRaisesRegex(ValueError, "one of"):
            action_key(4)

    def test_timing_is_exactly_four_60hz_frames_without_sleep(self) -> None:
        timing = atari_timing()
        self.assertEqual(FRAME_SKIP, 4)
        self.assertAlmostEqual(GAME_TIME_PER_STEP_S, 4 / 60)
        self.assertEqual(timing["frame_rate_hz"], 60)
        self.assertEqual(timing["frame_skip"], 4)
        self.assertEqual(timing["action_repeat_frames"], 4)
        self.assertEqual(timing["wall_clock_sleep_s"], 0.0)
        self.assertTrue(timing["screenshots_while_paused"])
        self.assertEqual(timing["observation_frame"], "last_executed_frame")
        self.assertFalse(timing["max_pool_last_two_frames"])

    def test_score_and_brick_events(self) -> None:
        events = transition_events(
            state(score=10, bricks=10),
            state(score=50, bricks=9),
        )
        self.assertEqual(events.score_delta, 40)
        self.assertEqual(events.positive_score_delta, 40)
        self.assertEqual(events.bricks_destroyed, 1)
        self.assertTrue(events.brick_hit)
        self.assertFalse(events.life_lost)
        self.assertFalse(events.game_over)
        self.assertFalse(events.level_cleared)

    def test_intermediate_life_loss_is_not_game_over(self) -> None:
        events = transition_events(
            state(lives=5),
            state(lives=4),
        )
        self.assertTrue(events.life_lost)
        self.assertFalse(events.game_over)
        self.assertFalse(events.terminal_failure)
        self.assertFalse(events.level_cleared)

    def test_fifth_life_loss_is_failure_but_not_success(self) -> None:
        events = transition_events(
            state(lives=1),
            state(
                lives=0,
                terminal=(True, "fail", "no_lives_left"),
            ),
        )
        self.assertTrue(events.life_lost)
        self.assertTrue(events.game_over)
        self.assertTrue(events.terminal_failure)
        self.assertFalse(events.level_cleared)
        self.assertFalse(events.terminal_success)

    def test_level_clear_is_success_but_not_life_loss(self) -> None:
        events = transition_events(
            state(score=100, bricks=1),
            state(
                score=140,
                bricks=0,
                terminal=(True, "success", "level_cleared"),
            ),
        )
        self.assertTrue(events.level_cleared)
        self.assertTrue(events.terminal_success)
        self.assertFalse(events.life_lost)
        self.assertFalse(events.game_over)

    def test_rejects_native_level_advance_and_midgame_score_reset(self) -> None:
        with self.assertRaisesRegex(ValueError, "never auto-advance"):
            transition_events(state(level=5), state(level=1))
        with self.assertRaisesRegex(ValueError, "must not reset"):
            transition_events(state(score=100), state(score=0))


class ObservationTests(unittest.TestCase):
    def test_rgb_max_pool_would_erase_black_ball(self) -> None:
        background = Image.new("RGB", (3, 2), color=(210, 210, 210))
        ball_frame = background.copy()
        ball_frame.putpixel((1, 1), (0, 0, 0))
        pooled = ImageChops.lighter(background, ball_frame)
        self.assertEqual(pooled.getpixel((1, 1)), (210, 210, 210))
        self.assertEqual(ball_frame.getpixel((1, 1)), (0, 0, 0))

    def test_rpc_health_advertises_no_step_limit_and_five_lives(self) -> None:
        class Environment:
            width = 1280
            height = 720
            noop_max = 30

        server = object.__new__(BrowserEnvironmentRPCServer)
        server.environment = Environment()
        response, should_close = __import__("asyncio").run(
            server._handle({"cmd": "health"})
        )
        self.assertFalse(should_close)
        payload = json.loads(response[0])
        self.assertEqual(payload["protocol_version"], PROTOCOL_VERSION)
        self.assertIsNone(payload["max_steps"])
        self.assertEqual(payload["initial_lives"], 5)
        self.assertEqual(payload["game_over_lives"], 0)
        self.assertEqual(payload["reset_noop_max"], 30)
        self.assertEqual(payload["action_meanings"], list(ACTION_MEANINGS))
        self.assertEqual(payload["timing"], atari_timing())


if __name__ == "__main__":
    unittest.main()

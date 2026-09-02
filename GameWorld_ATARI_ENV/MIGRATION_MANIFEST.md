# Independent environment manifest

`GameWorld_ATARI_ENV` was extracted as a separate runtime so deterministic
training changes cannot affect the original `GameWorld` evaluator or the
retained legacy `GameWorld_TRAIN_ENV`. It preserves GameWorld Breakout's native
visuals, physics, scoring, and layouts, while deliberately replacing its
automatic countdown with ALE-compatible FIRE launch semantics.

| Path | Purpose |
| --- | --- |
| `games/benchmark/05_breakout/` | Breakout engine and assets; five lives, FIRE launch, stable game-over and clear frames |
| `env/browser_scripts/dynamic_speed_control.js` | Paused virtual clock and exact synchronous 60 Hz frame stepper |
| `env/browser_scripts/deterministic_random.js` | Reproducible browser randomness |
| `env/browser_manager.py` | Chromium lifecycle, state reads, pause, and in-memory screenshots |
| `env/game_launcher.py` | Local HTTP server for the game |
| `tools/diamond_bridge/breakout_protocol.py` | ALE four-action, deterministic four-frame, reward/life/terminal contract |
| `tools/diamond_bridge/browser_env_server.py` | Versioned ZeroMQ service returning the final executed frame without max pooling |
| `tools/diamond_bridge/smoke_test.py` | Live exact-game-time validation |
| `tools/diamond_bridge/verify_terminal_boundaries.py` | Live five-life and level-clear boundary validation |
| `tools/diamond_bridge/test_breakout_bridge.py` | Protocol and boundary regression tests |

The folder intentionally excludes GameWorld agents, tasks, evaluator,
coordinator, runtime, provider integrations, and benchmark catalogs. It is a
training emulator, not a GameWorld evaluation installation.

# Migration manifest

`GameWorld_TRAIN_ENV` is a minimal extraction from the evaluation GameWorld
project. The copied components and their reasons are:

| Path | Required for |
| --- | --- |
| `env/action_executor.py` | Legal 0.2-second wait/left/right Playwright actions |
| `env/browser_manager.py` | Chromium lifecycle, pause/resume, state reads and in-memory CDP PNGs |
| `env/game_launcher.py` | Local HTTP serving of the Breakout assets |
| `env/game_state_tracker.py` | `gameAPI` init/reset/state and pause/resume scripts |
| `env/browser_scripts/*.js` | Deterministic seed and GameWorld time control |
| `games/benchmark/05_breakout/` | The complete Breakout engine and static assets |
| `tools/diamond_bridge/` | Policy-neutral score/life/task events, the evaluation-aligned 0.2-second action plus 0.05-second post-action cadence, versioned ZeroMQ service and smoke tests |

`env/controls.py` and the minimal `env/__init__.py` replace the original catalog
dependency. The following evaluation-only subsystems are intentionally absent:

- `agents/`
- `catalog/`
- `runtime/`
- task YAML files
- evaluators and coordinator
- model/provider integrations

Consequently this directory can provide training transitions but cannot run a
GameWorld benchmark evaluation. Formal evaluation must use the original
`GameWorld` directory.

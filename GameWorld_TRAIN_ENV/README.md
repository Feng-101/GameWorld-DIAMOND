# GameWorld_TRAIN_ENV

Standalone Breakout browser environment used only for DIAMOND data collection.
The original `GameWorld` directory remains the unmodified evaluation project.

Each training step follows the fixed part of the official evaluation cadence:
hold the selected action for 0.2 seconds, release it and let the game run for
0.05 seconds, capture the screenshot while the game is still running, then
pause for agent inference and read the precise transition-boundary state. Thus
0.25 seconds is the nominal observation interval, not a 0.25-second key hold.

The browser bridge reports native score/brick/life/task events but does not
choose rewards or RL episode boundaries. Those train/test policies live in the
DIAMOND adapter.

This directory intentionally contains only:

- Breakout's HTML/JavaScript/CSS/audio/image assets;
- the GameWorld browser manager, game launcher and action executor;
- deterministic random and pause/speed browser hooks;
- the versioned DIAMOND ZeroMQ bridge and its tests.

It does not contain GameWorld agents, catalogs, tasks, coordinator, runtime, or
evaluator code. Training changes must be made here, never in the evaluation
`GameWorld` directory.

## Test

```bash
python -m unittest tools.diamond_bridge.test_breakout_bridge -v
python -m tools.diamond_bridge.smoke_test
```

## Serve DIAMOND

Training and periodic validation require independent browser sessions.
Start both services before DIAMOND:

```bash
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5561 \
  --game-port 8101 \
  --max-steps 500
```

In a second terminal:

```bash
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5562 \
  --game-port 8102 \
  --max-steps 100
```

The training adapter requests five lives per physical task. A cleared board or
the fifth lost life normally ends the task; 500 steps is only a safety cap.
Periodic validation requests the original three lives and remains capped at
100 steps. Level 2/5 are not used by the validation collector; they remain
reserved for final evaluation in the original `GameWorld` project.

# DIAMOND Breakout standalone training environment

This package exposes the copied `05_breakout` browser game to the separate
DIAMOND Python environment. Run all commands from the `GameWorld_TRAIN_ENV`
root inside the GameWorld Conda environment. It has no dependency on the
evaluation-only `GameWorld` directory.

## Fast tests

Pure transition tests plus browser-manager contract tests:

```bash
python -m unittest tools.diamond_bridge.test_breakout_bridge -v
```

Short live-Chromium test (three training-layout resets and nine total actions,
not three full episodes):

```bash
python -m tools.diamond_bridge.smoke_test
```

The live test must print JSON with `"ok": true`, levels 1, 3 and 4, the
canvas bounds, and exactly three action results per level. Each result reports
both the full macro-step game-time delta and the evaluator-state-to-observation
delta; the latter must be at least 30 ms, allowing scheduling tolerance around
the configured 50 ms idle interval. It also reports native score, brick, life,
success, and task-limit events without imposing DIAMOND's train/test episode
policy in the browser process.

## Training service

```bash
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5561 \
  --game-port 8101 \
  --max-steps 500
```

The request protocol is one JSON frame:

- `{"cmd":"health"}`
- `{"cmd":"reset","level":1,"seed":42,"initial_lives":5}`
- `{"cmd":"step","action":0}`
- `{"cmd":"close"}`

Reset and step replies contain a JSON metadata frame followed by one
normalized 1280x720 PNG frame. Actions are `0=wait`, `1=left`, and `2=right`.
Each step reproduces the evaluator's observable order: a 0.2-second action
hold, an immediate evaluator-state read, 0.05 seconds of unpressed-key game
time, a screenshot while the game is still running, and then a pause for agent
inference. A second state read after pausing defines the state boundary used by
the training reward. The nominal interval is 0.25 seconds; variable evaluator
and logging overhead is not treated as action time.

Step metadata contains `transition_events`: signed `score_delta`, non-negative
`positive_score_delta`, `bricks_destroyed`, life-loss/reset facts, and physical
task success/time-limit facts. The service deliberately does not turn a life
loss into a reward or choose whether it terminates an RL episode.

`initial_lives` is applied per reset and accepts values 1 through 5; the copied
game's global defaults are not changed. This lets the training service use five
lives while a separate 100-step validation service keeps the original three.

Every JSON response includes `"protocol_version": 6` and timing metadata so the DIAMOND client
can reject incompatible bridge implementations instead of silently training
on a changed protocol.

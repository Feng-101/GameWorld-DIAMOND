# GameWorld_ATARI_ENV

This is an independent, deterministic GameWorld Breakout environment for
DIAMOND training. It aligns only the key temporal and episode logic with the
Atari setup. It uses ALE Breakout's four-action/FIRE launch semantics while
preserving GameWorld Breakout's rendering, physics, scoring, and level layouts.
It does **not** use GameWorld's evaluation cadence and it does not modify
either `GameWorld` or the retained legacy `GameWorld_TRAIN_ENV`.

## Environment contract

One `step(action)` uses an Atari-like fixed decision interval without copying
Atari-specific observation processing:

1. keep the browser game paused;
2. press one ALE action from `NOOP`, `FIRE`, `RIGHT`, `LEFT`;
3. synchronously advance at most four engine frames at exactly 60 Hz;
4. capture and return the final executed frame while game time is stopped.

There is intentionally no pixelwise max pooling. GameWorld's ball is black on
a light background, so RGB maximum pooling across two frames can replace the
moving ball with the old light background and erase it from the observation.

Therefore a normal transition advances exactly `4 / 60 = 0.0666667` seconds
of **game time**. Screenshot, ZeroMQ, model inference, disk I/O, and wall-clock
latency advance zero game time. A terminal event may stop the four-frame repeat
early, just as Atari preprocessing stops when ALE terminates.

Reset starts exactly five lives and advances 1--30 hidden raw NOOP frames, as
in DIAMOND's Atari preprocessing. The ball remains attached with no countdown
until the agent selects `FIRE`; every non-final life loss returns to the same
FIRE-waiting state. The first four life losses keep the same board and score.
The fifth life loss leaves a stable zero-life game-over frame. Clearing all
bricks leaves a distinct successful terminal frame and never auto-advances to
another level. There is no train or test step cap.

One service owns one browser. Training and validation need two independent
services so validation cannot reset the training game.

## Install

From this directory in the browser Conda environment:

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
```

If Chromium is already in `PLAYWRIGHT_BROWSERS_PATH`, the second command is not
needed.

## Verify the browser implementation

Pure protocol tests:

```bash
python -m unittest tools.diamond_bridge.test_breakout_bridge -v
```

Live Chromium test:

```bash
python -m tools.diamond_bridge.smoke_test \
  --game-port 8291 \
  --level 5 \
  --seed 4242 \
  --noop-max 30 \
  --steps 20
```

Success requires `"ok": true`, four executed frames per non-terminal action,
about `66.667 ms` game-time advancement, and `0 ms` screenshot game-time
advancement. Reset must leave the ball attached with no countdown, and the
first `FIRE` step must launch it immediately. The test also verifies that the
returned frame still contains the black ball.

Rare physical boundaries can be isolated without manually playing a full game:

```bash
python -m tools.diamond_bridge.verify_terminal_boundaries \
  --game-port 8292 \
  --level 5 \
  --seed 4242
```

This must report four non-terminal life losses followed by a distinct
zero-life failure terminal, then a five-life successful clear terminal.

An optional detailed profiler is:

```bash
python -m tools.diamond_bridge.profile_timing \
  --game-port 8291 \
  --level 5 \
  --action 1 \
  --steps 20 \
  --output timing_atari_level5.json
```

Wall-clock screenshot time can be large without changing the simulated game
time; the reported game-time fields are the correctness criterion.

## Start train and test services

Terminal 1:

```bash
cd GameWorld_ATARI_ENV
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5661 \
  --game-port 8201 \
  --noop-max 30
```

Terminal 2:

```bash
cd GameWorld_ATARI_ENV
python -m tools.diamond_bridge.browser_env_server \
  --bind tcp://127.0.0.1:5662 \
  --game-port 8202 \
  --noop-max 30
```

There is intentionally no `--max-steps` option.

## Train DIAMOND

In the DIAMOND environment:

```bash
cd diamond
python scripts/smoke_test_gameworld_atari_env.py \
  --endpoint tcp://127.0.0.1:5661 \
  --level 5
python scripts/preflight_gameworld_atari_training.py --level 5
bash scripts/run_gameworld_atari_breakout.sh my_atari_level5 5 0 true
```

The script arguments are `run_name`, `level`, `CUDA device`, and
`compile_wm`. Level 5 is the default; any one level from 1 through 5 can be
selected. The formal configuration uses DIAMOND's Atari 100k collection and
model-update schedule.

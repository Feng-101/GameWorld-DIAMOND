#!/usr/bin/env bash

# Train one deterministic Atari-style browser Breakout specialist.
# Two independent GameWorld_ATARI_ENV services must already be listening on
# ports 5661 (train) and 5662 (test), unless endpoint overrides are supplied.

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_name="${1:-gameworld_atari_level5_$(date +%Y%m%d_%H%M%S)}"
level="${2:-5}"
device="${3:-0}"
compile_wm="${4:-true}"
train_endpoint="${TRAIN_ENDPOINT:-tcp://127.0.0.1:5661}"
test_endpoint="${TEST_ENDPOINT:-tcp://127.0.0.1:5662}"
run_dir="$repo_root/results/$run_name"
console_log="$repo_root/results/${run_name}.console.log"
manifest="$repo_root/results/${run_name}.manifest.txt"

if [[ ! "$level" =~ ^[1-5]$ ]]; then
  echo "Second argument level must be an integer from 1 to 5" >&2
  exit 2
fi
if [[ "$compile_wm" != "true" && "$compile_wm" != "false" ]]; then
  echo "Fourth argument compile_wm must be true or false" >&2
  exit 2
fi
if [[ -e "$run_dir" || -e "$console_log" || -e "$manifest" ]]; then
  echo "Refusing to reuse an existing run name: $run_name" >&2
  exit 2
fi
mkdir -p "$repo_root/results"

cat > "$manifest" <<EOF
run_name=$run_name
started_at=$(date --iso-8601=seconds)
repository=$repo_root
environment=gameworld_atari_breakout
experiment=gameworld_atari_breakout_formal
level=$level
device=$device
train_endpoint=$train_endpoint
test_endpoint=$test_endpoint
frame_rate_hz=60
frame_skip=4
game_time_per_action_seconds=0.06666666666666667
initial_lives=5
max_episode_steps=none
compile_wm=$compile_wm
EOF

echo "Checking the deterministic browser services and DIAMOND configuration..."
if ! python scripts/preflight_gameworld_atari_training.py \
  --level "$level" \
  --train-endpoint "$train_endpoint" \
  --test-endpoint "$test_endpoint"; then
  echo "status=preflight_failed" >> "$manifest"
  exit 1
fi

started_epoch="$(date +%s)"
set +e
TORCHINDUCTOR_CACHE_DIR="$run_dir/.torchinductor" \
python src/main.py \
  env=gameworld_atari_breakout \
  +experiment=gameworld_atari_breakout_formal \
  "env.level=$level" \
  "env.train.endpoint=$train_endpoint" \
  "env.test.endpoint=$test_endpoint" \
  "common.devices=$device" \
  "training.compile_wm=$compile_wm" \
  wandb.mode=offline \
  wandb.project=gameworld-diamond \
  wandb.group=gameworld-atari-breakout \
  "wandb.name=$run_name" \
  "hydra.run.dir=$run_dir" \
  2>&1 | tee "$console_log"
status="${PIPESTATUS[0]}"
set -e

cat >> "$manifest" <<EOF
finished_at=$(date --iso-8601=seconds)
elapsed_seconds=$(( $(date +%s) - started_epoch ))
exit_code=$status
status=$(if [[ "$status" -eq 0 ]]; then echo complete; else echo failed; fi)
EOF

if [[ "$status" -eq 0 ]]; then
  echo "Training complete: $run_dir"
  echo "Latest validation: $run_dir/validation_latest.json"
else
  echo "Training failed with exit code $status" >&2
fi
exit "$status"

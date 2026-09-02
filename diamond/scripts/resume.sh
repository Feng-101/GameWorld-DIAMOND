#!/usr/bin/env bash
set -Eeuo pipefail

# Run from a Hydra run directory. Additional overrides are forwarded so local
# browser endpoints and CUDA selection can change after migration/requeue.
python src/main.py \
  common.resume=True \
  hydra.output_subdir=null \
  hydra.run.dir=. \
  "$@"

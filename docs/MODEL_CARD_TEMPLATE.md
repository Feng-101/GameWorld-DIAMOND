---
license: other
library_name: pytorch
tags:
  - reinforcement-learning
  - world-model
  - diffusion
  - gameworld
  - breakout
---

# DIAMOND GameWorld Breakout Level-5 Specialist (4 frames/action)

## Model description

This checkpoint adapts DIAMOND (Diffusion As a Model Of eNvironment Dreams) to
GameWorld's browser Breakout. It is a Level-5 specialist trained with a
deterministic browser emulator interface. It preserves GameWorld rendering,
physics, scoring, and layout while using four ALE-style actions and exactly
four 60 Hz engine frames per action.

`agent_epoch_01000.pt` is a full DIAMOND Agent state dict containing the
diffusion denoiser, reward/end model, and LSTM actor-critic.

## Training protocol

- level: 5 only
- actions: NOOP, FIRE, RIGHT, LEFT
- action repeat: 4 × 60 Hz frames (0.066667 s simulated game time)
- observation: last executed frame, cropped/resized to RGB 64×64
- max pooling: disabled to preserve the black ball
- episode: five lives, no artificial step cap; level clear is success
- real interaction budget: 100,000 steps
- imagination horizon: 15
- DIAMOND batch size: 32 for denoiser, reward/end model, and actor-critic
- world-model compilation: enabled
- final epoch: 1000
- run: `atari_v9_level5_20260721_163502`

## Final training-time validation

100 complete five-life Level-5 games:

| Metric | Value |
| --- | ---: |
| progress mean | 0.974583 |
| progress std | 0.066625 |
| success rate | 0.65 |
| native return mean | 10848.65 |
| native return std | 846.51 |
| task steps mean | 1905.01 |
| lives lost mean | 3.42 |

These metrics use the deterministic training/test environment, not the
official GameWorld evaluator.

## Files

- `agent_epoch_01000.pt`: full Agent weights for inference/visualization
- `trainer.yaml`: resolved training configuration
- `validation_metrics.jsonl`: periodic and final validation history
- `validation_latest.json`: final validation record
- `SHA256SUMS`: integrity hashes

## Usage

Use the repository's `docs/INFERENCE_VIDEO.md` and
`src/evaluate_gameworld_atari_agent_60hz.py`. The environment and model code
must match the handed-off GameWorld_DIAMOND repository.

## Limitations

- It was trained only on Level 5.
- Cross-level behavior is uneven, with especially weak Level-3 generalization.
- Results are not directly comparable to ALE Atari Breakout scores.
- The deterministic training environment is intentionally different from the
  original GameWorld wall-clock evaluation loop.

## License

Before publication, verify the redistribution terms of the DIAMOND code,
GameWorld code, Breakout game assets, and trained weights. Do not replace this
section with an upstream code license without checking that it also covers the
model and bundled assets.

## Citation

Please cite the original DIAMOND paper and the GameWorld benchmark in addition
to this adaptation.

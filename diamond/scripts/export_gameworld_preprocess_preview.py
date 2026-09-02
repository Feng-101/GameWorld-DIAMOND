"""Export the exact GameWorld canvas crop and DIAMOND 64x64 observation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from integrations.gameworld.preprocess import (  # noqa: E402
    DEFAULT_CANVAS_CROP,
    EXPECTED_VIEWPORT,
    preprocess_gameworld_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path, help="Full 1280x720 GameWorld PNG")
    parser.add_argument("--output-dir", type=Path, default=Path("preprocess_preview"))
    args = parser.parse_args()

    png = args.frame.expanduser().read_bytes()
    encoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if encoded is None:
        raise ValueError(f"Unable to decode {args.frame}")
    height, width = encoded.shape[:2]
    if (width, height) != EXPECTED_VIEWPORT:
        raise ValueError(
            f"Expected viewport {EXPECTED_VIEWPORT}, received {(width, height)}"
        )

    crop = DEFAULT_CANVAS_CROP
    canvas_bgr = encoded[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
    observation_rgb = preprocess_gameworld_frame(png, size=64)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas_path = output_dir / f"breakout_crop_{crop.width}x{crop.height}.png"
    observation_path = output_dir / "breakout_diamond_64x64.png"
    if not cv2.imwrite(str(canvas_path), canvas_bgr):
        raise RuntimeError(f"Failed to write {canvas_path}")
    if not cv2.imwrite(
        str(observation_path), cv2.cvtColor(observation_rgb, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"Failed to write {observation_path}")

    print(f"canvas={canvas_path} shape={canvas_bgr.shape}")
    print(f"diamond={observation_path} shape={observation_rgb.shape}")


if __name__ == "__main__":
    main()

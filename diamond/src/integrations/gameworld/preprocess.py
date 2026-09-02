"""The single image preprocessing path used by GameWorld training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor


EXPECTED_VIEWPORT = (1280, 720)


@dataclass(frozen=True, slots=True)
class CanvasCrop:
    """Pixel bounds selected from GameWorld's fixed 1280x720 viewport."""

    # The detected outer black frame is (368, 70, 545, 460). Keeping a
    # 16-pixel margin avoids placing the antialiased frame exactly on the
    # observation boundary while removing the canvas UI and excess backdrop.
    x: int = 352
    y: int = 54
    width: int = 577
    height: int = 492


DEFAULT_CANVAS_CROP = CanvasCrop()


def preprocess_gameworld_frame(
    png: bytes,
    *,
    size: int = 64,
    crop: CanvasCrop = DEFAULT_CANVAS_CROP,
) -> np.ndarray:
    """Decode, crop and resize a GameWorld PNG to an RGB uint8 observation."""
    if not isinstance(png, bytes) or not png:
        raise ValueError("GameWorld observation must be non-empty PNG bytes")
    if size < 1:
        raise ValueError(f"Observation size must be positive, got {size}")

    encoded = np.frombuffer(png, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("GameWorld observation is not a decodable PNG image")

    expected_width, expected_height = EXPECTED_VIEWPORT
    actual_height, actual_width = image_bgr.shape[:2]
    if (actual_width, actual_height) != EXPECTED_VIEWPORT:
        raise ValueError(
            "Unexpected GameWorld viewport: "
            f"actual={(actual_width, actual_height)}, "
            f"expected={(expected_width, expected_height)}"
        )

    if crop.x < 0 or crop.y < 0 or crop.width < 1 or crop.height < 1:
        raise ValueError(f"Invalid canvas crop: {crop}")
    right = crop.x + crop.width
    bottom = crop.y + crop.height
    if right > actual_width or bottom > actual_height:
        raise ValueError(
            f"Canvas crop {crop} exceeds viewport {(actual_width, actual_height)}"
        )

    canvas_bgr = image_bgr[crop.y:bottom, crop.x:right]
    resized_bgr = cv2.resize(canvas_bgr, (size, size), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(resized_rgb, dtype=np.uint8)


def frame_to_tensor(
    png: bytes,
    *,
    device: torch.device,
    size: int = 64,
    crop: CanvasCrop = DEFAULT_CANVAS_CROP,
) -> Tensor:
    """Convert one full GameWorld PNG to DIAMOND's normalized BCHW tensor."""
    rgb = preprocess_gameworld_frame(png, size=size, crop=crop)
    observation = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    observation = observation.permute(2, 0, 1).unsqueeze(0).contiguous()
    return observation.div(255).mul(2).sub(1)


__all__ = [
    "CanvasCrop",
    "DEFAULT_CANVAS_CROP",
    "EXPECTED_VIEWPORT",
    "frame_to_tensor",
    "preprocess_gameworld_frame",
]

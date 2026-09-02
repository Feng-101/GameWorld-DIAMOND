"""Measure initial raw-frame ball motion in ALE Breakout.

This is a diagnostic for comparing DIAMOND's four-frame decision interval with
the deterministic GameWorld browser backend. It requires the Atari ROM to be
installed in the current ALE environment.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import cv2
import gymnasium
import numpy as np


def _small_components(frame: np.ndarray) -> list[dict[str, Any]]:
    # Atari Breakout's moving ball is a tiny non-black component. Exclude the
    # score area and reject the much wider paddle and brick rows.
    mask = np.any(frame[40:] != 0, axis=2).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        y += 40
        if width <= 5 and height <= 6 and area <= 24:
            component_mask = labels == label
            colors, frequencies = np.unique(
                frame[40:][component_mask],
                axis=0,
                return_counts=True,
            )
            dominant_color = colors[int(np.argmax(frequencies))]
            components.append(
                {
                    "center_x": float(centroids[label][0]),
                    "center_y": float(centroids[label][1] + 40),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": area,
                    "dominant_rgb": [int(value) for value in dominant_color],
                }
            )
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    # Importing ale_py registers the Atari environments.
    import ale_py  # noqa: F401

    environment = gymnasium.make(
        "BreakoutNoFrameskip-v4",
        frameskip=1,
        render_mode="rgb_array",
    )
    try:
        observation, _ = environment.reset(seed=args.seed)
        action_meanings = list(environment.unwrapped.get_action_meanings())
        if action_meanings[:2] != ["NOOP", "FIRE"]:
            raise RuntimeError(f"Unexpected ALE actions: {action_meanings}")

        # FIRE launches the native Atari ball. Subsequent raw frames use NOOP.
        observation, _, terminated, truncated, _ = environment.step(1)
        if terminated or truncated:
            raise RuntimeError("ALE terminated during initial FIRE")

        frames: list[dict[str, Any]] = []
        for frame_index in range(args.frames):
            observation, reward, terminated, truncated, _ = environment.step(0)
            components = _small_components(observation)
            frames.append(
                {
                    "frame": frame_index + 1,
                    "reward": float(reward),
                    "components": components,
                }
            )
            if terminated or truncated:
                break

        wall_region = observation[50:105]
        colors, frequencies = np.unique(
            wall_region.reshape(-1, 3),
            axis=0,
            return_counts=True,
        )
        wall_color_bounds: list[dict[str, Any]] = []
        for color, frequency in zip(colors, frequencies):
            if np.all(color == 0) or int(frequency) < 50:
                continue
            pixels = np.argwhere(np.all(wall_region == color, axis=2))
            wall_color_bounds.append(
                {
                    "rgb": [int(value) for value in color],
                    "count": int(frequency),
                    "x_min": int(pixels[:, 1].min()),
                    "x_max": int(pixels[:, 1].max()),
                    "y_min": int(pixels[:, 0].min() + 50),
                    "y_max": int(pixels[:, 0].max() + 50),
                }
            )

        # The ball is the small component that persists while changing its
        # center in both axes. Select the longest color/shape-consistent track.
        tracks: list[list[dict[str, Any]]] = []
        for frame in frames:
            candidates = frame["components"]
            next_tracks: list[list[dict[str, Any]]] = []
            used: set[int] = set()
            for track in tracks:
                previous = track[-1]
                matches = [
                    (index, candidate)
                    for index, candidate in enumerate(candidates)
                    if index not in used
                    and candidate["dominant_rgb"] == previous["dominant_rgb"]
                    and abs(candidate["center_x"] - previous["center_x"]) <= 3
                    and abs(candidate["center_y"] - previous["center_y"]) <= 3
                ]
                if matches:
                    index, candidate = min(
                        matches,
                        key=lambda item: (
                            abs(item[1]["center_x"] - previous["center_x"])
                            + abs(item[1]["center_y"] - previous["center_y"])
                        ),
                    )
                    used.add(index)
                    next_tracks.append(track + [candidate])
                else:
                    next_tracks.append(track)
            for index, candidate in enumerate(candidates):
                if index not in used:
                    next_tracks.append([candidate])
            tracks = next_tracks

        moving_tracks = [
            track
            for track in tracks
            if len(track) >= 4
            and (
                abs(track[-1]["center_x"] - track[0]["center_x"]) > 0
                or abs(track[-1]["center_y"] - track[0]["center_y"]) > 0
            )
        ]
        if not moving_tracks:
            raise RuntimeError(
                f"Could not identify the Atari ball track: {frames!r}"
            )
        ball_track = max(moving_tracks, key=len)
        deltas = [
            {
                "dx": ball_track[index]["center_x"]
                - ball_track[index - 1]["center_x"],
                "dy": ball_track[index]["center_y"]
                - ball_track[index - 1]["center_y"],
            }
            for index in range(1, len(ball_track))
        ]

        print(
            json.dumps(
                {
                    "environment": "BreakoutNoFrameskip-v4",
                    "raw_frame_shape": list(observation.shape),
                    "action_meanings": action_meanings,
                    "initial_lives": int(environment.unwrapped.ale.lives()),
                    "wall_region_color_bounds": wall_color_bounds,
                    "ball_track": ball_track,
                    "per_frame_deltas": deltas,
                    "median_abs_dx": float(
                        np.median([abs(delta["dx"]) for delta in deltas])
                    ),
                    "median_abs_dy": float(
                        np.median([abs(delta["dy"]) for delta in deltas])
                    ),
                },
                indent=2,
            )
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()

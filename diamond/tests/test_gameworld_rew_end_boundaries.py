from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.rew_end_model import _replace_boundary_next_observations


class RewardEndBoundaryObservationTests(unittest.TestCase):
    def test_true_final_frame_is_restored_for_end_and_truncation(self) -> None:
        next_obs = torch.zeros((2, 4, 1, 2, 2), dtype=torch.float32)
        end = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.uint8)
        trunc = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.uint8)
        info = [
            {"final_observation": torch.ones((1, 2, 2))},
            {"final_observation": torch.full((1, 2, 2), 2.0)},
        ]

        restored = _replace_boundary_next_observations(next_obs, end, trunc, info)

        self.assertTrue(torch.equal(restored[0, 1], info[0]["final_observation"]))
        self.assertTrue(torch.equal(restored[1, 2], info[1]["final_observation"]))
        self.assertEqual(restored[0, 0].sum().item(), 0.0)
        self.assertEqual(restored[1, 3].sum().item(), 0.0)
        # The sampled batch must stay immutable across model losses.
        self.assertEqual(next_obs.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

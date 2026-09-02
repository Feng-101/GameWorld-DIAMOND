from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.utils import keep_agent_copies_every


class CheckpointScheduleTests(unittest.TestCase):
    def test_only_periodic_and_forced_final_agent_versions_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            state = {"weight": torch.tensor([1.0])}

            for epoch in range(0, 251):
                keep_agent_copies_every(
                    state,
                    epoch,
                    checkpoint_dir,
                    every=100,
                    num_to_keep=4,
                )

            versions = checkpoint_dir / "agent_versions"
            self.assertEqual(
                sorted(path.name for path in versions.iterdir()),
                [
                    "agent_epoch_00100.pt",
                    "agent_epoch_00200.pt",
                ],
            )

            keep_agent_copies_every(
                state,
                250,
                checkpoint_dir,
                every=100,
                num_to_keep=4,
                force=True,
            )
            self.assertEqual(
                sorted(path.name for path in versions.iterdir()),
                [
                    "agent_epoch_00100.pt",
                    "agent_epoch_00200.pt",
                    "agent_epoch_00250.pt",
                ],
            )

    def test_retains_four_periodic_versions_plus_non_aligned_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            state = {"weight": torch.tensor([1.0])}
            for epoch in range(0, 601, 100):
                keep_agent_copies_every(
                    state,
                    epoch,
                    checkpoint_dir,
                    every=100,
                    num_to_keep=4,
                )
            keep_agent_copies_every(
                state,
                650,
                checkpoint_dir,
                every=100,
                num_to_keep=4,
                force=True,
            )

            versions = checkpoint_dir / "agent_versions"
            self.assertEqual(
                sorted(path.name for path in versions.iterdir()),
                [
                    "agent_epoch_00300.pt",
                    "agent_epoch_00400.pt",
                    "agent_epoch_00500.pt",
                    "agent_epoch_00600.pt",
                    "agent_epoch_00650.pt",
                ],
            )


if __name__ == "__main__":
    unittest.main()

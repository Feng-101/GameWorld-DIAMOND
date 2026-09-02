from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_gameworld_horizon_pair import _extract


def metrics(namespace: str, progress: float, native_return: float) -> dict[str, float]:
    return {
        f"{namespace}/progress_mean": progress,
        f"{namespace}/progress_std": 0.1,
        f"{namespace}/success_rate": 0.0,
        f"{namespace}/native_return_mean": native_return,
        f"{namespace}/native_return_std": 2.0,
        f"{namespace}/task_steps_mean": 100.0,
        f"{namespace}/lives_lost_mean": 1.5,
    }


class HorizonPairSummaryTests(unittest.TestCase):
    def test_extracts_periodic_best_and_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "horizon_10.console.log"
            path.write_text(
                "\n".join(
                    [
                        str(metrics("validation", 0.2, 10.0)),
                        str(metrics("validation", 0.4, 15.0)),
                        str(metrics("final_validation", 0.3, 12.0)),
                    ]
                ),
                encoding="utf-8",
            )
            result = _extract(path, 10)

        self.assertEqual(result["num_periodic_validations"], 2)
        self.assertEqual(result["best_periodic_progress_mean"], 0.4)
        self.assertEqual(result["best_periodic_native_return_mean"], 15.0)
        self.assertEqual(result["final_validation"]["progress_mean"], 0.3)


if __name__ == "__main__":
    unittest.main()

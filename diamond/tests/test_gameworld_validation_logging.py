from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validation_logging import append_validation_record
from watch_gameworld_validation import _default_output_paths, extract_records


def metrics(namespace: str, progress: float) -> dict[str, float]:
    return {
        f"{namespace}/progress_mean": progress,
        f"{namespace}/progress_std": 0.1,
        f"{namespace}/success_rate": 0.0,
        f"{namespace}/native_return_mean": 123.0,
        f"{namespace}/native_return_std": 2.0,
        f"{namespace}/task_steps_mean": 100.0,
        f"{namespace}/lives_lost_mean": 3.0,
    }


class ValidationLoggingTests(unittest.TestCase):
    def test_native_writer_appends_history_and_refreshes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "validation_metrics.jsonl"
            latest = root / "validation_latest.json"
            append_validation_record({"epoch": 50, "metrics": {"value": 1}}, history, latest)
            append_validation_record({"epoch": 100, "metrics": {"value": 2}}, history, latest)

            records = [
                json.loads(line)
                for line in history.read_text(encoding="utf-8").splitlines()
            ]
            latest_record = json.loads(latest.read_text(encoding="utf-8"))

        self.assertEqual([record["epoch"] for record in records], [50, 100])
        self.assertEqual(latest_record["epoch"], 100)

    def test_extracts_epoch_and_validation_metrics_from_console_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "horizon_10.console.log"
            path.write_text(
                "\n".join(
                    [
                        "Epoch 50 / 1600",
                        str(metrics("validation", 0.25)),
                        "Epoch 100 / 1600",
                        str(metrics("validation", 0.40)),
                        str(metrics("final_validation", 0.35)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            records = extract_records(path)

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["epoch"], 50)
        self.assertEqual(records[1]["epoch"], 100)
        self.assertTrue(records[2]["final"])
        self.assertEqual(records[2]["namespace"], "final_validation")

    def test_default_extracted_output_names_are_concise(self) -> None:
        source = Path("/tmp/pair/horizon_10.console.log")
        output, latest = _default_output_paths(source)
        self.assertEqual(output.name, "horizon_10.validation.jsonl")
        self.assertEqual(latest.name, "horizon_10.validation.latest.json")


if __name__ == "__main__":
    unittest.main()

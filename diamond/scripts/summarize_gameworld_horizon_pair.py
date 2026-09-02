"""Summarize fixed-grid validation metrics from an H=10/H=15 pair run."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


METRICS = (
    "progress_mean",
    "progress_std",
    "success_rate",
    "native_return_mean",
    "native_return_std",
    "task_steps_mean",
    "lives_lost_mean",
)


def _metric_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pair console log: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "validation/progress_mean" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _extract(path: Path, horizon: int) -> dict[str, Any]:
    records = _metric_dicts(path)
    periodic = [
        record for record in records if "validation/progress_mean" in record
    ]
    final = next(
        (
            record
            for record in reversed(records)
            if "final_validation/progress_mean" in record
        ),
        None,
    )
    if not periodic:
        raise RuntimeError(f"No periodic validation metrics found in {path}")
    if final is None:
        raise RuntimeError(
            f"No final validation metrics found in {path}; the run may be incomplete"
        )

    def selected(record: dict[str, Any], namespace: str) -> dict[str, float]:
        return {
            metric: float(record[f"{namespace}/{metric}"])
            for metric in METRICS
        }

    best_progress = max(
        float(record["validation/progress_mean"]) for record in periodic
    )
    best_return = max(
        float(record["validation/native_return_mean"]) for record in periodic
    )
    return {
        "horizon": horizon,
        "num_periodic_validations": len(periodic),
        "best_periodic_progress_mean": best_progress,
        "best_periodic_native_return_mean": best_return,
        "last_periodic": selected(periodic[-1], "validation"),
        "final_validation": selected(final, "final_validation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pair_dir = args.pair_dir.expanduser().resolve()
    runs = {
        str(horizon): _extract(
            pair_dir / f"horizon_{horizon}.console.log",
            horizon,
        )
        for horizon in (10, 15)
    }
    h10 = runs["10"]["final_validation"]
    h15 = runs["15"]["final_validation"]
    delta = {
        metric: float(h15[metric] - h10[metric])
        for metric in METRICS
    }
    report = {
        "ok": True,
        "pair_dir": str(pair_dir),
        "selection_data_uses_levels": [1, 3, 4],
        "heldout_levels_not_used": [2, 5],
        "runs": runs,
        "final_delta_h15_minus_h10": delta,
        "note": (
            "Prefer the horizon with consistently higher progress/return across "
            "periodic and final validation; treat success rate as decisive when nonzero."
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

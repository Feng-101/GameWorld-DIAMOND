"""Extract concise GameWorld validation records from a running console log.

This is useful for jobs that were started before Trainer gained native
validation_metrics.jsonl and validation_latest.json outputs.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Optional


EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")


class ConsoleValidationParser:
    def __init__(self, source_log: Path) -> None:
        self.source_log = source_log
        self.epoch: Optional[int] = None
        self.total_epochs: Optional[int] = None

    def consume(self, line: str, line_number: int) -> Optional[dict[str, Any]]:
        epoch_match = EPOCH_PATTERN.search(line)
        if epoch_match is not None:
            self.epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))

        if (
            "validation/progress_mean" not in line
            and "final_validation/progress_mean" not in line
        ):
            return None
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            metrics = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            return None
        if not isinstance(metrics, dict):
            return None

        final = "final_validation/progress_mean" in metrics
        namespace = "final_validation" if final else "validation"
        return {
            "schema_version": 1,
            "source_log": str(self.source_log),
            "source_line": line_number,
            "timestamp_unix": time.time(),
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "namespace": namespace,
            "final": final,
            "metrics": metrics,
        }


def extract_records(path: Path) -> list[dict[str, Any]]:
    parser = ConsoleValidationParser(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = parser.consume(line, line_number)
            if record is not None:
                records.append(record)
    return records


def _write_history(
    records: Iterable[dict[str, Any]],
    output: Path,
    latest: Path,
) -> list[dict[str, Any]]:
    values = list(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in values:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    if values:
        latest.write_text(
            json.dumps(values[-1], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return values


def _append_record(record: dict[str, Any], output: Path, latest: Path) -> None:
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary_latest = latest.with_name(f".{latest.name}.tmp")
    temporary_latest.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_latest.replace(latest)


def _print_record(record: dict[str, Any]) -> None:
    metrics = record["metrics"]
    namespace = record["namespace"]
    print(
        f"epoch={record['epoch']} "
        f"type={namespace} "
        f"progress={float(metrics[f'{namespace}/progress_mean']):.4f} "
        f"success={float(metrics[f'{namespace}/success_rate']):.4f} "
        f"return={float(metrics[f'{namespace}/native_return_mean']):.1f} "
        f"lives_lost={float(metrics[f'{namespace}/lives_lost_mean']):.2f}",
        flush=True,
    )


def _wait_for_source(path: Path, poll_seconds: float) -> None:
    announced = False
    while not path.is_file():
        if not announced:
            print(f"Waiting for console log: {path}", flush=True)
            announced = True
        time.sleep(poll_seconds)


def follow(
    source: Path,
    output: Path,
    latest: Path,
    poll_seconds: float,
) -> None:
    _wait_for_source(source, poll_seconds)
    parser = ConsoleValidationParser(source)
    line_number = 0
    existing: list[dict[str, Any]] = []

    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = parser.consume(line, line_number)
            if record is not None:
                existing.append(record)
        _write_history(existing, output, latest)
        if existing:
            _print_record(existing[-1])
            if existing[-1]["final"]:
                print("Final validation already present; follower is complete.", flush=True)
                return
        print(f"Following validation output in {source}", flush=True)

        while True:
            line = stream.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            line_number += 1
            record = parser.consume(line, line_number)
            if record is not None:
                _append_record(record, output, latest)
                _print_record(record)
                if record["final"]:
                    print("Final validation received; follower is complete.", flush=True)
                    return


def _default_output_paths(source: Path) -> tuple[Path, Path]:
    name = source.name
    suffix = ".console.log"
    base = name[: -len(suffix)] if name.endswith(suffix) else source.stem
    return (
        source.parent / f"{base}.validation.jsonl",
        source.parent / f"{base}.validation.latest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("console_log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latest", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    source = args.console_log.expanduser().resolve()
    default_output, default_latest = _default_output_paths(source)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output
    )
    latest = (
        args.latest.expanduser().resolve()
        if args.latest is not None
        else default_latest
    )

    if args.follow:
        follow(source, output, latest, args.poll_seconds)
        return
    if not source.is_file():
        raise FileNotFoundError(f"Missing console log: {source}")
    records = _write_history(extract_records(source), output, latest)
    for record in records:
        _print_record(record)
    print(f"Wrote {len(records)} validation records to {output}")
    if records:
        print(f"Latest validation record: {latest}")


if __name__ == "__main__":
    main()

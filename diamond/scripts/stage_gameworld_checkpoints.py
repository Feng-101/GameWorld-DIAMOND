#!/usr/bin/env python3
"""Copy mixed/Level-5 agent snapshots into a portable visualization folder."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Iterable


CHECKPOINT_PATTERN = re.compile(r"agent_epoch_(\d+)\.pt$")


def _checkpoint_directory(source: Path) -> tuple[Path, Path | None]:
    source = source.expanduser().resolve()
    if source.is_file():
        return source.parent, source
    candidates = (
        source / "checkpoints" / "agent_versions",
        source / "agent_versions",
        source,
    )
    directory = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir() and any(candidate.glob("agent_epoch_*.pt"))
        ),
        None,
    )
    if directory is None:
        raise FileNotFoundError(f"No agent_epoch_*.pt checkpoints found under {source}")
    return directory, None


def _find_config(source: Path, checkpoint_dir: Path) -> Path | None:
    candidates = (
        source / "config" / "trainer.yaml",
        source / "trainer.yaml",
        checkpoint_dir.parent.parent / "config" / "trainer.yaml",
        checkpoint_dir / "trainer.yaml",
    )
    return next((path for path in candidates if path.is_file()), None)


def _parse_epochs(value: str) -> set[int] | str:
    normalized = value.strip().lower()
    if normalized in {"all", "latest"}:
        return normalized
    try:
        epochs = {int(item.strip()) for item in normalized.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--epochs must be all, latest, or comma-separated integers"
        ) from exc
    if not epochs or any(epoch < 0 for epoch in epochs):
        raise argparse.ArgumentTypeError("--epochs must contain non-negative integers")
    return epochs


def _select_checkpoints(files: Iterable[Path], selection: set[int] | str) -> list[Path]:
    parsed = []
    for path in files:
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            parsed.append((int(match.group(1)), path))
    parsed.sort()
    if not parsed:
        raise FileNotFoundError("No valid agent_epoch_NNNNN.pt snapshots found")
    if selection == "all":
        return [path for _, path in parsed]
    if selection == "latest":
        return [parsed[-1][1]]

    found = {epoch: path for epoch, path in parsed}
    missing = sorted(selection - found.keys())
    if missing:
        raise FileNotFoundError(
            f"Requested epochs are unavailable: {missing}; available={sorted(found)}"
        )
    return [found[epoch] for epoch in sorted(selection)]


def _copy_profile(
    profile: str,
    source: Path,
    output_dir: Path,
    selection: set[int] | str,
) -> list[dict]:
    checkpoint_dir, single_file = _checkpoint_directory(source)
    files = [single_file] if single_file is not None else checkpoint_dir.glob("agent_epoch_*.pt")
    selected = _select_checkpoints(files, selection)
    destination = output_dir / profile
    destination.mkdir(parents=True, exist_ok=True)

    config = _find_config(source.resolve(), checkpoint_dir)
    if config is not None:
        shutil.copy2(config, destination / "trainer.yaml")

    records = []
    for checkpoint in selected:
        target = destination / checkpoint.name
        if target.exists():
            old_digest = sha256(target.read_bytes()).hexdigest()
            new_digest = sha256(checkpoint.read_bytes()).hexdigest()
            if old_digest != new_digest:
                raise FileExistsError(
                    f"Refusing to overwrite a different checkpoint: {target}"
                )
        else:
            shutil.copy2(checkpoint, target)
            new_digest = sha256(target.read_bytes()).hexdigest()
        match = CHECKPOINT_PATTERN.match(checkpoint.name)
        assert match is not None
        records.append(
            {
                "profile": profile,
                "epoch": int(match.group(1)),
                "source": str(checkpoint.resolve()),
                "checkpoint": str(target.resolve()),
                "sha256": new_digest,
                "config": str((destination / "trainer.yaml").resolve())
                if config is not None
                else None,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-source", type=Path)
    parser.add_argument("--level5-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=_parse_epochs,
        default="all",
        help="all (default), latest, or comma-separated epoch numbers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mixed_source is None and args.level5_source is None:
        raise ValueError("Specify --mixed-source and/or --level5-source")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if args.mixed_source is not None:
        records.extend(
            _copy_profile("mixed", args.mixed_source, output_dir, args.epochs)
        )
    if args.level5_source is not None:
        records.extend(
            _copy_profile("level5", args.level5_source, output_dir, args.epochs)
        )

    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "checkpoints": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Staged {len(records)} checkpoint(s) in {output_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

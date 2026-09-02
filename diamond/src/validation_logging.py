"""Small, dependency-free helpers for persistent validation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/Torch scalar containers to JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def append_validation_record(
    record: Mapping[str, Any],
    history_path: Path,
    latest_path: Path,
) -> dict[str, Any]:
    """Append one JSONL record and atomically refresh a readable latest JSON."""
    normalized = _jsonable(record)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(normalized, sort_keys=True) + "\n")

    temporary_latest = latest_path.with_name(f".{latest_path.name}.tmp")
    temporary_latest.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_latest.replace(latest_path)
    return normalized

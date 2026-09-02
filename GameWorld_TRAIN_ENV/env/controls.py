"""Control constraints needed by the standalone Breakout training environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RoleControls:
    """Subset of GameWorld role controls consumed by ``ActionExecutor``."""

    allowed_keys: set[str] = field(default_factory=set)
    hold_duration: float = 0.2
    key_durations: dict[str, float] = field(default_factory=dict)
    allow_clicks: bool = True


"""Minimal browser surface required by deterministic Atari-style Breakout."""

from .browser_manager import BrowserConfig, BrowserGameManager
from .game_launcher import GameLauncher

__all__ = [
    "BrowserConfig",
    "BrowserGameManager",
    "GameLauncher",
]

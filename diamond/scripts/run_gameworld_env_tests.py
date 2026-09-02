"""Run GameWorld environment tests from any current working directory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TEST_DIR),
        pattern="test_gameworld*.py",
        top_level_dir=str(REPO_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

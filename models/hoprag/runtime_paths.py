"""Shared HopRAG runtime layout, usable from both pinned Python environments."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def runtime_home() -> Path:
    """Use the existing baseline-home setting without resolving venv symlinks."""
    home = Path(os.environ.get("RAG_OFFICIAL_BASELINE_HOME", ROOT / "data/official_baselines")).expanduser()
    return home.absolute() / "hoprag"

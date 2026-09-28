"""Shared HopRAG runtime layout, usable from both pinned Python environments."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def runtime_home(environment: Mapping[str, str] | None = None) -> Path:
    """Use the existing baseline-home setting without resolving venv symlinks."""
    environment = os.environ if environment is None else environment
    home = Path(environment.get("RAG_OFFICIAL_BASELINE_HOME", ROOT / "data/official_baselines")).expanduser()
    return home.absolute() / "hoprag"

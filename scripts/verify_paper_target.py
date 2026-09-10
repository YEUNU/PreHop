#!/usr/bin/env python3
"""Compatibility entry for already-running launchers; no final validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.record_paper_completion import main, persist_admission

__all__ = ["main", "persist_admission"]

if __name__ == '__main__':
    raise SystemExit(main())

"""Non-secret identity of the prepared project runtime recorded with paper artifacts."""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def runtime_identity(strategy: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the current interpreter, installed-distribution digest and lockfile identity.

    Every retained strategy runs in the main project environment, so the
    identity does not depend on ``strategy`` or ``environment``; both are kept
    so saved-run readers and callers keep one call shape.
    """
    _ = strategy, environment
    freeze_path = ROOT / "uv.lock"

    def identity(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest, "size": path.stat().st_size}

    installed = sorted((str(dist.metadata.get("Name", "")).lower(), dist.version)
                       for dist in metadata.distributions())
    main_runtime = {
        "prefix": str(Path(sys.prefix).absolute()),
        "python": str(Path(sys.executable).absolute()),
        "installed_sha256": hashlib.sha256(json.dumps(installed, separators=(",", ":")).encode()).hexdigest(),
    }
    return {"main_runtime": main_runtime, "runtime_freeze": identity(freeze_path)}

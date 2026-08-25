"""Reproducible code provenance for indexing and benchmark artifacts."""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def code_provenance(root: Path = ROOT) -> dict[str, Any]:
    """Return the commit and exact non-ignored source snapshot in use."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        listed = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": None, "source_tree_sha256": "unavailable", "file_count": None}

    digest = hashlib.sha256()
    count = 0
    for raw_path in sorted(path for path in listed.split(b"\0") if path):
        path = root / os.fsdecode(raw_path)
        if not path.is_file():
            continue
        content = path.read_bytes()
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    return {
        "revision": revision,
        "dirty": dirty,
        "source_tree_sha256": digest.hexdigest(),
        "file_count": count,
    }

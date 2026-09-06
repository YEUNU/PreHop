"""Reproducible code provenance for indexing and benchmark artifacts."""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_GENERATED_PREFIXES = (
    b"artifacts/",
    b"fig/",
    b"logs/",
    b"data/debug/",
    b"data/embedding_cache/",
    b"data/failed_runs/",
    b"data/index_cache/",
    b"data/index_failures/",
    b"data/index_locks/",
    b"data/index_stats/",
    b"data/official_baselines/",
    b"data/results/",
    b"data/tmp/",
)


def _is_generated_path(raw_path: bytes) -> bool:
    if raw_path.startswith(_GENERATED_PREFIXES):
        return True
    return raw_path.startswith(b"data/") and b"_output/" in raw_path


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
        if _is_generated_path(raw_path):
            continue
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

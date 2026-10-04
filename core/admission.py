"""Content digests and artifact identities for execution provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_post_query_inventory(strategy: str, dataset: str) -> dict[str, Any]:
    """Describe retrieval artifacts after a query batch; Neo4j-backed indexes have no files to digest."""
    from core.strategy_registry import get_strategy

    get_strategy(strategy)
    _ = dataset
    return {"kind": "service_managed", "file_count": 0, "total_bytes": 0, "sha256": None}


def current_corpus_identity(dataset: str) -> dict[str, Any]:
    """Read recorded corpus identity without scanning or verifying staged files."""
    from cli.index import _load_corpus_manifest
    corpus = ROOT / "data" / f"{dataset}_corpus"
    return {**(_load_corpus_manifest(corpus) or {}),
            "manifest_sha256": sha256_file(corpus / "corpus_manifest.json")}

"""Content digests and artifact identities for execution provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_inventory(root: Path, *, excluded: set[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            relative = path.relative_to(root)
            if any(part in excluded for part in relative.parts):
                continue
            rows.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "sha256": identity_sha256(rows),
    }


def current_post_query_inventory(strategy: str, dataset: str) -> dict[str, Any]:
    """Digest current retrieval artifacts without reading result/output logs."""
    from core.strategy_registry import get_strategy

    spec = get_strategy(strategy)
    if strategy in {"prehop", "naive"}:
        return {"kind": "service_managed", "file_count": 0, "total_bytes": 0, "sha256": None}
    output = Path(os.environ.get(spec.output_env or "", spec.output_default or "")) / dataset
    if spec.driver:
        inventory = _file_inventory(output / "artifacts", excluded=set())
        return {"kind": "external_artifacts", **inventory}
    if strategy == "ms_graphrag":
        inventory = _file_inventory(output, excluded={"_cache", "_logs", "_input", "index_snapshot_metadata.json"})
        return {"kind": "ms_native_artifacts", **inventory}
    return {"kind": "not_applicable", "file_count": 0, "total_bytes": 0, "sha256": None}


def current_corpus_identity(dataset: str) -> dict[str, Any]:
    """Read recorded corpus identity without scanning or verifying staged files."""
    from cli.index import _load_corpus_manifest
    corpus = ROOT / "data" / f"{dataset}_corpus"
    return {**(_load_corpus_manifest(corpus) or {}),
            "manifest_sha256": sha256_file(corpus / "corpus_manifest.json")}

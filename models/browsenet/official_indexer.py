from __future__ import annotations

import time

from models.official_baseline_runtime import (
    OFFICIAL_REVISIONS,
    configured_embedding_model,
    configured_embedding_revision,
    corpus_records_sha256,
    run_index_worker,
    snapshot_metadata_path,
    source_set_sha256,
    stage_corpus,
)
from utils.io import _write_json


async def run_official_index(dataset_path: str, corpus_tag: str, corpus_manifest: dict | None) -> dict[str, float]:
    started = time.perf_counter()
    records, target = stage_corpus("browsenet", dataset_path, corpus_tag)
    staging_seconds = time.perf_counter() - started
    worker_started = time.perf_counter()
    result = run_index_worker(
        "browsenet",
        corpus_tag,
        {"operation": "index", "generation_model": None},
    )
    official_seconds = time.perf_counter() - worker_started
    source_ids = [row["source_id"] for row in records]
    _write_json(
        snapshot_metadata_path("browsenet", corpus_tag),
        {
            "snapshot_version": 1,
            "status": "complete",
            "strategy": "browsenet",
            "corpus_tag": corpus_tag,
            "official_revision": OFFICIAL_REVISIONS["browsenet"],
            "embedding_model": configured_embedding_model(),
            "embedding_revision": configured_embedding_revision(),
            "source_count": len(source_ids),
            "source_set_sha256": source_set_sha256(source_ids),
            "corpus_records_sha256": corpus_records_sha256(records),
            "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
            "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
            "retrieval_artifact_dir": str(target / "artifacts"),
            "official_stats": result.get("stats", {}),
        },
    )
    return {"staging_seconds": staging_seconds, "official_index_seconds": official_seconds}

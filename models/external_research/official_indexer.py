from __future__ import annotations

import time

from core.semantic_config import semantic_config_sha256, semantic_index_policy
from models.official_baseline_runtime import (
    OFFICIAL_REVISIONS,
    artifact_inventory,
    configured_embedding_model,
    configured_embedding_revision,
    corpus_records_sha256,
    run_index_worker,
    snapshot_metadata_path,
    source_set_sha256,
    stage_corpus,
)
from utils.io import _write_json


async def run_official_index(
    strategy: str,
    dataset_path: str,
    corpus_tag: str,
    corpus_manifest: dict | None,
    index_policy: dict | None = None,
) -> dict[str, float]:
    started = time.perf_counter()
    records, target = stage_corpus(strategy, dataset_path, corpus_tag)
    staging_seconds = time.perf_counter() - started
    result = run_index_worker(strategy, corpus_tag, {"operation": "index"})
    source_ids = [row["source_id"] for row in records]
    stats = result.get("stats")
    inventory = artifact_inventory(target)
    operational_config = dict(index_policy or {}).get("operational_config", {})
    policy = semantic_index_policy(index_policy)
    if policy.get("extraction_validation_profile"):
        from .extraction_contract import validate_audit_evidence

        validate_audit_evidence(stats.get("extraction_audit_evidence"))
    semantic_sha256 = semantic_config_sha256(index_policy)
    _write_json(
        snapshot_metadata_path(strategy, corpus_tag),
        {
            "snapshot_version": 1,
            "status": "complete",
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "official_revision": OFFICIAL_REVISIONS[strategy],
            "embedding_model": policy.get("embedding_model", configured_embedding_model()),
            "embedding_revision": policy.get("embedding_revision", configured_embedding_revision()),
            "backbone_mode": policy.get("backbone_mode", "official_faithful"),
            "semantic_config_id": policy.get("semantic_config_id", f"{strategy}-paper-v1"),
            "semantic_config_sha256": semantic_sha256,
            "semantic_config": policy,
            "operational_config": operational_config,
            "source_count": len(source_ids),
            "source_set_sha256": source_set_sha256(source_ids),
            "corpus_records_sha256": corpus_records_sha256(records),
            "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
            "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
            "retrieval_artifact_dir": str(target / "artifacts"),
            "artifact_inventory": inventory,
            "official_stats": stats,
        },
    )
    return {
        "staging_seconds": staging_seconds,
        "official_index_seconds": time.perf_counter() - started - staging_seconds,
    }

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
    validate_runtime,
)
from utils.io import _write_json


def validate_native_generation_profile(strategy: str, stats: dict, policy: dict) -> None:
    if strategy == "youtu_graphrag":
        for field in ("extraction_generation_profile", "extraction_schema_sha256"):
            if not policy.get(field) or stats.get(field) != policy[field]:
                raise RuntimeError("Youtu effective extraction profile differs from semantic policy")
        return
    if strategy != "hipporag2":
        return
    expected = {
        "openie_response_format": {"type": policy.get("openie_response_format")},
        "openie_ner_max_tokens": policy.get("openie_ner_max_tokens"),
        "openie_triple_max_tokens": policy.get("openie_triple_max_tokens"),
    }
    if expected["openie_response_format"] != {"type": "json_object"} or any(
        stats.get(name) != value for name, value in expected.items()
    ):
        raise RuntimeError("HippoRAG2 effective native OpenIE profile differs from the recorded semantic policy")


async def run_official_index(
    strategy: str,
    dataset_path: str,
    corpus_tag: str,
    corpus_manifest: dict | None,
    index_policy: dict | None = None,
) -> dict[str, float]:
    started = time.perf_counter()
    validate_runtime(strategy)
    records, target = stage_corpus(strategy, dataset_path, corpus_tag)
    staging_seconds = time.perf_counter() - started
    result = run_index_worker(strategy, corpus_tag, {"operation": "index"})
    source_ids = [row["source_id"] for row in records]
    stats = result.get("stats")
    if (
        not isinstance(stats, dict)
        or stats.get("source_count") != len(source_ids)
        or stats.get("coverage_complete") is not True
    ):
        raise RuntimeError(f"{strategy} official worker did not attest exact source coverage")
    inventory = artifact_inventory(target)
    if inventory["file_count"] < 1 or inventory["total_bytes"] < 1:
        raise RuntimeError(f"{strategy} official worker produced no retrieval artifacts")
    operational_config = dict(index_policy or {}).get("operational_config", {})
    policy = semantic_index_policy(index_policy)
    validate_native_generation_profile(strategy, stats, policy)
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

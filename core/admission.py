"""Content digests and artifact identities for execution provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCES = (
    ROOT / "scripts/verify_paper_target.py",
    ROOT / "core/paper_policy.py",
    ROOT / "core/index_reuse.py",
    ROOT / "core/phase_timing.py",
    ROOT / "core/paper_compatibility.py",
    ROOT / "core/generation_profiles.py",
    ROOT / "core/semantic_config.py",
    ROOT / "core/strategy_registry.py",
    ROOT / "core/runtime_requirements.py",
    ROOT / "core/inference_transport.py",
    ROOT / "core/embedding_policy.py",
    ROOT / "core/structured_outputs.py",
    ROOT / "models/external_research/extraction_contract.py",
    ROOT / "scripts/runner_environment.py",
    ROOT / "scripts/paper_gate_ledger.py",
    ROOT / "scripts/paper_cold_canary.py",
    ROOT / "scripts/paper_stage_runner.py",
    ROOT / "scripts/paper_campaign.py",
    ROOT / "scripts/campaign_attempts.py",
    ROOT / "scripts/recovery_checkpoint.py",
    ROOT / "scripts/cold_canary_fixture.py",
    ROOT / "configs/cold_canary/museum_rich_entities_v2.json",
    ROOT / "cli/index.py",
    ROOT / "utils/provenance.py",
    ROOT / "utils/metrics.py",
    ROOT / "utils/reporting.py",
    ROOT / "configs/serving_observation.json",
    ROOT / "configs/paper_runtime_requirements.json",
    ROOT / "configs/paper_gateway.json",
    Path(__file__).resolve(),
)


def verifier_sources() -> tuple[Path, ...]:
    constraints = tuple(sorted((ROOT / "configs/runtime_constraints").glob("*.txt")))
    return (*VERIFIER_SOURCES, *constraints)


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

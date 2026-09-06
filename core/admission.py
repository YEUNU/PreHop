"""Content bindings for immutable paper-target admission records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCES = (
    ROOT / "scripts/verify_paper_target.py",
    ROOT / "scripts/verify_submission_consistency.py",
    ROOT / "core/paper_policy.py",
    ROOT / "core/paper_compatibility.py",
    ROOT / "core/generation_profiles.py",
    ROOT / "core/semantic_config.py",
    ROOT / "core/strategy_registry.py",
    ROOT / "core/runtime_requirements.py",
    ROOT / "core/inference_transport.py",
    ROOT / "core/embedding_policy.py",
    ROOT / "core/structured_outputs.py",
    ROOT / "core/native_structured_profile.py",
    ROOT / "scripts/check_paper_runtime.py",
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


def _index_stats_binding(payload: dict | None) -> tuple[str | None, str | None]:
    raw_path = payload.get("index_manifest_stats_path") if isinstance(payload, dict) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, None
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if ROOT not in resolved.parents or not resolved.is_file():
        return str(resolved), None
    return str(resolved), sha256_file(resolved)


def current_corpus_identity(dataset: str) -> dict[str, Any]:
    """Revalidate v2 manifest and every current prepared source without writes."""
    from cli.index import _load_corpus_manifest, _validate_staged_snapshot

    if dataset not in {"multihoprag", "musique"}:
        raise ValueError("unsupported paper corpus")
    corpus = ROOT / "data" / f"{dataset}_corpus"
    manifest = _load_corpus_manifest(corpus)
    if not manifest or manifest.get("schema_version") != 2:
        raise ValueError("current paper corpus requires a schema_version=2 manifest")
    files = sorted(path.name for path in corpus.iterdir() if path.is_file() and path.suffix in {".txt", ".md"})
    _validate_staged_snapshot(files, manifest, corpus)
    return {**manifest, "manifest_sha256": sha256_file(corpus / "corpus_manifest.json")}


def admission_bindings(result_path: Path, payload: dict | None) -> dict[str, str | None]:
    """Bind an admission decision to result bytes and verifier semantics."""
    details_path = result_path.with_name(result_path.stem + ".details.jsonl")
    stats_path, stats_digest = _index_stats_binding(payload)
    strategy = str(payload.get("strategy") or "") if isinstance(payload, dict) else ""
    dataset = str(payload.get("corpus_tag") or "") if isinstance(payload, dict) else ""
    from core.paper_compatibility import EVIDENCE_VERSION, runtime_compatibility, target_configuration
    from core.runtime_requirements import runtime_identity

    try:
        corpus_identity = current_corpus_identity(dataset) if dataset else None
    except (OSError, TypeError, ValueError, RuntimeError):
        corpus_identity = None
    current_runtime = runtime_identity(strategy) if strategy else None
    current_inventory = current_post_query_inventory(strategy, dataset) if strategy and dataset else None
    configuration = target_configuration(strategy, dataset) if strategy and dataset else None
    return {
        "current_corpus_identity_sha256": identity_sha256(corpus_identity) if corpus_identity else None,
        "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "details_sha256": sha256_file(details_path) if details_path.is_file() else None,
        "evidence_contract": EVIDENCE_VERSION,
        "configuration_sha256": identity_sha256(configuration) if configuration else None,
        "index_policy_sha256": (
            payload.get("index_provenance", {}).get("policy_sha256") if isinstance(payload, dict) else None
        ),
        "index_stats_path": stats_path,
        "index_stats_sha256": stats_digest,
        "runtime_identity_sha256": identity_sha256(runtime_compatibility(current_runtime) if current_runtime else None),
        "post_query_artifact_inventory_sha256": identity_sha256(current_inventory),
        "recorded_post_query_artifact_inventory_sha256": identity_sha256(
            payload.get("post_query_artifact_inventory") if isinstance(payload, dict) else None
        ),
    }


def admission_path_for_result(result_path: Path) -> Path:
    """Return the target-level ledger beside the strategy output subtree."""
    if result_path.parent.name != "seed_42" or len(result_path.parents) < 4:
        raise ValueError(f"result is not in a seed_42 paper target: {result_path}")
    # .../<run>/<strategy>/<dataset>/seed_42/result.json -> .../<run>/admission.json
    return result_path.parents[3] / "admission.json"


def verification_provenance() -> dict:
    """Describe the actual checker without using its Git state as compatibility."""
    from utils.provenance import code_provenance
    return {'code': code_provenance(), 'verifier_sources_sha256': identity_sha256(
        {str(path): sha256_file(path) for path in verifier_sources()})}

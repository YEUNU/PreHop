"""Validate the current clean full-system matrix and its published values.

This is a manually invoked repository check. It does not run model inference
and is not wired to CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.admission import admission_bindings, admission_path_for_result
from core.semantic_config import semantic_config_sha256
from core.strategy_registry import PRIMARY_STRATEGIES, get_strategy

DATASETS = {
    "multihoprag": {
        "count": 2556,
        "metrics": (
            "avg_official_hits@4",
            "avg_official_hits@10",
            "avg_official_mrr@10",
            "avg_official_map@10",
        ),
    },
    "musique": {
        "count": 2417,
        "metrics": (
            "avg_official_answer_em",
            "avg_official_answer_f1",
            "avg_paragraph_support_precision",
            "avg_paragraph_support_recall",
            "avg_paragraph_support_f1",
        ),
    },
}

STRATEGIES = PRIMARY_STRATEGIES

RESULT_DOCUMENTS = (
    Path("docs/RESULTS.md"),
    Path("docs/prehop_paper.md"),
)

CONFIG_DOCUMENTS = (Path("README.md"), *RESULT_DOCUMENTS)

PRESENTATIONS = (
    Path("presentation/prehop-academic.html"),
    Path("presentation/prehop-professor-briefing.html"),
)


def _semantic_model_config(payload: dict[str, Any], strategy: str | None = None) -> dict[str, Any]:
    """Return model behavior settings, excluding code and throughput metadata."""
    policy = dict(payload.get("index_provenance", {}).get("policy", {}))
    policy.pop("index_namespace", None)
    policy.pop("operational_config", None)
    if strategy == "youtu_graphrag":
        for field in ("dataset_alias", "schema_path", "schema_sha256", "schema_expected_sha256"):
            policy.pop(field, None)
    return {
        "models": payload.get("models", {}),
        "index_policy": policy,
        "ablation": payload.get("ablation", {}),
    }


def _semantic_model_config_sha256(payload: dict[str, Any], strategy: str | None = None) -> str:
    encoded = json.dumps(
        _semantic_model_config(payload, strategy),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_path(prefix: str, dataset: str, strategy: str, *, exact_run_id: bool = False) -> Path:
    run_id = prefix if exact_run_id else f"{prefix}-{dataset}-{strategy}"
    if not run_id or run_id in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in run_id):
        raise ValueError("invalid target run ID")
    filename = f"{strategy}_{dataset}.json"
    return Path("data/results") / run_id / strategy / dataset / "seed_42" / filename


def _expected_embedding_config(strategy: str) -> tuple[str, int | None, str | None]:
    spec = get_strategy(strategy)
    revision = spec.local_embedding_revision
    if revision is None and spec.paper_embedding_model == get_strategy("prehop").paper_embedding_model:
        revision = spec.paper_embedding_model
    return spec.paper_embedding_model, spec.paper_embedding_dimensions, revision


def _registered_revision_error(strategy: str, policy: dict[str, Any], path: Path) -> str | None:
    expected = get_strategy(strategy).revision
    if expected is None or policy.get("official_revision") == expected:
        return None
    return f"{path}: index policy official_revision={policy.get('official_revision')!r}, expected {expected!r}"


def _approved_ablation_policy(strategy: str) -> dict[str, Any]:
    """Checked-in method-defining benchmark policy for primary core methods."""
    from core.paper_policy import canonical_query_policy

    return canonical_query_policy(strategy)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _validate_admission(path: Path, payload: dict[str, Any], dataset: str, strategy: str) -> list[str]:
    """Require a fresh target ledger bound to the exact result and verifier."""
    errors: list[str] = []
    result_path = (ROOT / path).resolve()
    ledger_path = admission_path_for_result(result_path)
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"{path}: missing or invalid admission ledger {ledger_path}: {exc}"]
    if not isinstance(ledger, dict):
        return [f"{path}: admission ledger is not an object"]
    for field, expected in {
        "status": "admitted",
        "path": str(result_path),
        "dataset": dataset,
        "strategy": strategy,
    }.items():
        if ledger.get(field) != expected:
            errors.append(f"{path}: admission {field}={ledger.get(field)!r}, expected {expected!r}")
    current = admission_bindings(result_path, payload)
    if ledger.get("bindings") != current:
        errors.append(f"{path}: admission content bindings are stale")
    if ledger.get("errors") not in ([], None):
        errors.append(f"{path}: admission ledger records verification errors")
    return errors


def _current_query_digests(dataset: str) -> tuple[str, str, int]:
    rows = json.loads((ROOT / "data" / f"{dataset}_queries.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"data/{dataset}_queries.json: expected a list")
    ids = [str(row.get("_id") or "") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"data/{dataset}_queries.json: query identities are invalid")
    records = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in sorted(rows, key=lambda item: str(item["_id"]))
    ]
    return (
        hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest(),
        len(rows),
    )


def _current_query_identity(dataset: str) -> dict[str, dict[str, Any]]:
    rows = json.loads((ROOT / "data" / f"{dataset}_queries.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"data/{dataset}_queries.json: expected a list")
    return {str(row["_id"]): row for row in rows}


def _validate_artifact(
    path: Path,
    payload: dict[str, Any],
    *,
    dataset: str,
    strategy: str,
    expected_count: int,
) -> list[str]:
    errors: list[str] = []

    from core.amortized_cost import query_cost, validate_cost
    from core.execution_profile import execution_profile
    from core.strategy_registry import PAPER_TRANSPORT
    if payload.get("execution_profile") != execution_profile():
        errors.append(f"{path}: execution profile differs from the selected profile")
    try:
        batch = payload.get("query_batch_timing", {})
        if batch.get("version") != 1 or batch.get("resumed") != (payload.get("resume") is not None):
            raise ValueError("query batch timing/resume identity mismatch")
        expected_query_cost = query_cost(
            batch.get("wall_seconds"), expected_count,
            complete=payload.get("queries_count") == expected_count and not any(
                row.get("error") for row in payload.get("details", [])),
            resumed=payload.get("resume") is not None)
        validate_cost(payload.get("amortized_query_cost"), expected_query_cost)
        if payload.get("resume") is None and not expected_query_cost["continuous_run_eligible"]:
            raise ValueError("Complete continuous benchmark requires a positive measured wall time")
    except (ValueError, TypeError) as exc:
        errors.append(f"{path}: {exc}")

    expected = {
        "strategy": strategy,
        "corpus_tag": dataset,
        "evaluation_scope": "full_benchmark",
        "evaluated_queries_count": expected_count,
        "queries_count": expected_count,
        "total_queries": expected_count,
        "status": "completed_unadmitted",
        "benchmark_concurrency": PAPER_TRANSPORT.benchmark_concurrency,
        "judge_enabled": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            errors.append(f"{path}: {field}={payload.get(field)!r}, expected {value!r}")

    if payload.get("corpus_index_fingerprint_status") != "matched":
        errors.append(f"{path}: corpus/index fingerprint is not matched")
    if payload.get("index_manifest_status") != "complete":
        errors.append(f"{path}: index manifest is not complete")
    if payload.get("active_index_snapshot", {}).get("status") != "matched":
        errors.append(f"{path}: active index snapshot is not matched")
    try:
        from core.admission import current_corpus_identity

        corpus_identity = current_corpus_identity(dataset)
        for field, current in {
            "corpus_manifest_fingerprint": corpus_identity["fingerprint"],
            "corpus_manifest_paragraph_count": corpus_identity["paragraph_count"],
        }.items():
            if payload.get(field) != current:
                errors.append(f"{path}: current corpus differs from recorded {field}")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"{path}: current corpus validation failed: {exc}")
    expected_run_id = path.parts[2] if len(path.parts) > 2 else ""
    if payload.get("index_reuse") is not None:
        try:
            from core.index_reuse import bound, validate, validate_costs
            link_path, _ = bound(payload["index_reuse"])
            link = validate(link_path, expected_run_id, strategy, dataset)
            validate_costs(link, payload)
            expected_run_id = link["source_run_id"]
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            errors.append(f"{path}: index reuse validation failed: {exc}")
    elif (ROOT / "data/results" / expected_run_id / "index_link.json").exists():
        errors.append(f"{path}: result omits its required index reuse binding")

    models = payload.get("models", {})
    strategy_spec = get_strategy(strategy)
    expected_embedding, expected_dimensions, expected_embedding_revision = _expected_embedding_config(strategy)
    for field, value in {
        "default": strategy_spec.paper_generation_model,
        "generation_revision": strategy_spec.paper_generation_model,
        "embedding": expected_embedding,
        "llm_seed": strategy_spec.paper_generation_seed,
    }.items():
        if models.get(field) != value:
            errors.append(f"{path}: models.{field}={models.get(field)!r}, expected {value!r}")
    if models.get("embedding_revision") != expected_embedding_revision:
        errors.append(
            f"{path}: models.embedding_revision={models.get('embedding_revision')!r}, "
            f"expected {expected_embedding_revision!r}"
        )

    policy = payload.get("index_provenance", {}).get("policy", {})
    if payload.get("index_provenance", {}).get("run_id") != expected_run_id:
        errors.append(
            f"{path}: index provenance run_id={payload.get('index_provenance', {}).get('run_id')!r}, "
            f"expected target run_id {expected_run_id!r}"
        )
    expected_stats_path = (ROOT / "data/index_stats" / f"{strategy}_{dataset}_{expected_run_id}.json").resolve()
    recorded_stats_path = Path(str(payload.get("index_manifest_stats_path") or ""))
    if not recorded_stats_path.is_absolute():
        recorded_stats_path = (ROOT / recorded_stats_path).resolve()
    if recorded_stats_path != expected_stats_path:
        errors.append(f"{path}: index stats path is not the exact target path {expected_stats_path}")
    else:
        try:
            stats_bytes_digest = hashlib.sha256(expected_stats_path.read_bytes()).hexdigest()
            stats_payload = json.loads(expected_stats_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: exact index stats are unreadable: {exc}")
        else:
            from core.amortized_cost import indexing_cost
            try:
                expected_index_cost = indexing_cost(stats_payload)
                validate_cost(stats_payload.get("amortized_indexing_cost"), expected_index_cost)
                if not expected_index_cost["continuous_run_eligible"]:
                    raise ValueError("Complete index requires positive source count and measured wall time")
                if stats_payload.get("execution_profile") != execution_profile():
                    raise ValueError("index execution profile differs from selected profile")
            except (ValueError, TypeError) as exc:
                errors.append(f"{path}: {exc}")
            if payload.get("index_manifest_stats_sha256") != stats_bytes_digest:
                errors.append(f"{path}: exact index stats byte digest is stale")
            stats_expected = {
                "run_id": expected_run_id,
                "strategy": strategy,
                "corpus_tag": dataset,
                "status": "complete",
            }
            if not isinstance(stats_payload, dict) or any(
                stats_payload.get(field) != value for field, value in stats_expected.items()
            ):
                errors.append(f"{path}: exact index stats identity/status mismatch")
    if not isinstance(payload.get("index_provenance", {}).get("policy_sha256"), str):
        errors.append(f"{path}: index provenance policy_sha256 is missing")
    try:
        from core.paper_policy import validate_canonical_index_policy

        validate_canonical_index_policy(
            strategy,
            dataset,
            policy,
            payload.get("index_provenance", {}).get("policy_sha256"),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"{path}: canonical index policy validation failed: {exc}")
    revision_error = _registered_revision_error(strategy, policy, path)
    if revision_error:
        errors.append(revision_error)
    if strategy_spec.driver:
        active_snapshot = payload.get("active_index_snapshot", {})
        expected_semantic_hash = semantic_config_sha256(policy)
        if active_snapshot.get("semantic_config_id") != f"{strategy}-paper-v1":
            errors.append(f"{path}: active index semantic_config_id is missing or unexpected")
        if active_snapshot.get("semantic_config_sha256") != expected_semantic_hash:
            errors.append(f"{path}: active index semantic config hash differs from recorded index policy")
        inventory = active_snapshot.get("artifact_inventory")
        if not isinstance(inventory, dict) or int(inventory.get("file_count", 0)) < 1:
            errors.append(f"{path}: active index artifact inventory is missing")
    for field, value in {
        "strategy": strategy,
        "generation_seed": strategy_spec.paper_generation_seed,
        "generation_revision": strategy_spec.paper_generation_model,
        "embedding_max_input_tokens": 32768,
        "embedding_token_reserve": 0,
        "generation_max_context_tokens": 262144,
        "fulltext_analyzer": "english",
    }.items():
        if policy.get(field) != value:
            errors.append(f"{path}: index policy {field}={policy.get(field)!r}, expected approved value {value!r}")
    operational = policy.get("operational_config")
    if not isinstance(operational, dict):
        errors.append(f"{path}: index operational_config is missing")
    approved_common = dict(strategy_spec.paper_index_policy)
    for field, value in approved_common.items():
        if policy.get(field) != value:
            errors.append(f"{path}: index policy {field}={policy.get(field)!r}, expected approved value {value!r}")
    ablation = payload.get("ablation")
    if not isinstance(ablation, dict):
        errors.append(f"{path}: ablation policy is missing")
    else:
        for field, value in _approved_ablation_policy(strategy).items():
            if ablation.get(field) != value:
                errors.append(f"{path}: ablation {field}={ablation.get(field)!r}, expected {value!r}")
    if strategy == "gfm_rag" and not str(policy.get("gfm_checkpoint") or "").strip():
        errors.append(f"{path}: GFM-RAG checkpoint provenance is missing")
    if strategy == "gfm_rag" and len(str(policy.get("gfm_checkpoint_sha256") or "")) != 64:
        errors.append(f"{path}: GFM-RAG checkpoint SHA-256 provenance is missing")
    if strategy == "gfm_rag" and len(str(policy.get("gfm_config_sha256") or "")) != 64:
        errors.append(f"{path}: GFM-RAG config SHA-256 provenance is missing")
    if strategy == "youtu_graphrag" and not str(policy.get("schema_path") or "").strip():
        errors.append(f"{path}: Youtu schema provenance is missing")
    if strategy == "youtu_graphrag" and len(str(policy.get("schema_sha256") or "")) != 64:
        errors.append(f"{path}: Youtu schema SHA-256 provenance is missing")
    if strategy == "youtu_graphrag" and policy.get("schema_expected_sha256") != policy.get("schema_sha256"):
        errors.append(f"{path}: Youtu schema digest differs from the approved expected digest")
    if policy.get("embedding_model") != expected_embedding:
        errors.append(
            f"{path}: index embedding_model={policy.get('embedding_model')!r}, expected {expected_embedding!r}"
        )
    if policy.get("embedding_dimensions") != expected_dimensions:
        errors.append(
            f"{path}: embedding_dimensions={policy.get('embedding_dimensions')!r}, expected {expected_dimensions}"
        )
    details = payload.get("details")
    if not isinstance(details, list):
        errors.append(f"{path}: details must be a list")
        details = []
    if len(details) != expected_count:
        errors.append(f"{path}: details has {len(details)} rows, expected {expected_count}")
    indices = [row.get("idx") for row in details if isinstance(row, dict)]
    if indices and sorted(indices) != list(range(1, expected_count + 1)):
        errors.append(f"{path}: detail idx values are duplicated, missing, or out of range")
    query_ids = [str(row.get("query_id") or "") for row in details if isinstance(row, dict)]
    if any(not value for value in query_ids) or len(set(query_ids)) != len(query_ids):
        errors.append(f"{path}: detail query_id values are empty or duplicated")
    elif hashlib.sha256("\n".join(sorted(query_ids)).encode()).hexdigest() != payload.get("evaluated_query_ids_sha256"):
        errors.append(f"{path}: detail query identity digest does not match summary")
    try:
        current_queries = _current_query_identity(dataset)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: cannot validate detail query/ground-truth identity: {exc}")
    else:
        current_query_rows = list(_current_query_identity(dataset).values())
        for position, row in enumerate(details, start=1):
            if not isinstance(row, dict):
                errors.append(f"{path}: detail row is not an object")
                continue
            expected_row = current_queries.get(str(row.get("query_id") or ""))
            if expected_row is None:
                continue
            if row.get("idx") != position:
                errors.append(f"{path}: detail row {position} has idx={row.get('idx')!r}")
            if position <= len(current_query_rows) and str(row.get("query_id") or "") != str(
                current_query_rows[position - 1].get("_id") or ""
            ):
                errors.append(f"{path}: detail row {position} is not in current query-manifest order")
            for detail_field, query_field in (("query", "query"), ("ground_truth", "ground_truth")):
                if row.get(detail_field) != expected_row.get(query_field):
                    errors.append(
                        f"{path}: detail {row.get('query_id')} {detail_field} differs from current query manifest"
                    )
    runtime_errors = sum(bool(row.get("error")) for row in details if isinstance(row, dict))
    if runtime_errors:
        errors.append(f"{path}: {runtime_errors} detail row(s) contain runtime errors")
    try:
        current_ids_digest, current_records_digest, current_count = _current_query_digests(dataset)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: cannot validate current full query manifest: {exc}")
    else:
        if current_count != expected_count:
            errors.append(f"{path}: current query manifest has {current_count}, expected {expected_count}")
        if payload.get("evaluated_query_ids_sha256") != current_ids_digest:
            errors.append(f"{path}: evaluated query-id digest differs from current full manifest")
        if payload.get("evaluated_query_records_sha256") != current_records_digest:
            errors.append(f"{path}: evaluated query-record digest differs from current full manifest")
    for field, value in payload.items():
        if not field.startswith("eligible_") or not field.endswith("_count") or not isinstance(value, int):
            continue
        metric = field[len("eligible_") : -len("_count")]
        actual = sum(
            not row.get("error")
            and isinstance(row.get(metric), (int, float))
            and not isinstance(row.get(metric), bool)
            and row[metric] >= 0
            for row in details
            if isinstance(row, dict)
        )
        if actual != value:
            errors.append(f"{path}: {field}={value}, recomputed={actual}")

    # Every published average must be reproducible from its admitted detail
    # rows; checking only eligible counts permits silent summary tampering.
    for field, value in payload.items():
        if not field.startswith("avg_") or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        metric = field[len("avg_") :]
        values = [
            float(row[metric])
            for row in details
            if isinstance(row, dict)
            and not row.get("error")
            and isinstance(row.get(metric), (int, float))
            and not isinstance(row.get(metric), bool)
            and row[metric] >= 0
        ]
        recomputed = sum(values) / len(values) if values else 0.0
        if not math.isclose(float(value), recomputed, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"{path}: {field}={value}, recomputed={recomputed}")

    details_path = (ROOT / path).with_name((ROOT / path).stem + ".details.jsonl")
    try:
        jsonl_rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid details JSONL: {exc}")
    else:
        if len(jsonl_rows) != len(details):
            errors.append(f"{path}: details JSONL row count differs from main artifact")
        elif jsonl_rows != details:
            errors.append(f"{path}: details JSONL row content differs from main artifact")

    try:
        from core.admission import current_post_query_inventory

        current_inventory = current_post_query_inventory(strategy, dataset)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"{path}: cannot inventory post-query retrieval artifacts: {exc}")
    else:
        if payload.get("post_query_artifact_inventory") != current_inventory:
            errors.append(f"{path}: post-query retrieval artifact inventory is missing or stale")

    if strategy == "youtu_graphrag":
        official_stats = payload.get("active_index_snapshot", {}).get("official_stats")
        required_youtu_metrics = {
            "staged_input_evidence_sha256",
            "staged_input_coverage_complete",
            "native_extraction_evidence_sha256",
            "native_extraction_success_complete",
            "native_source_reachability_evidence_sha256",
            "native_source_reachability_observational",
        }
        if not isinstance(official_stats, dict) or not required_youtu_metrics.issubset(official_stats):
            errors.append(f"{path}: Youtu observational provenance metrics are missing")
        elif (
            official_stats.get("staged_input_coverage_complete") is not True
            or official_stats.get("native_extraction_success_complete") is not True
            or official_stats.get("native_source_reachability_observational") is not True
        ):
            errors.append(f"{path}: Youtu observational provenance metrics are invalid")

    for metric in DATASETS[dataset]["metrics"]:
        value = payload.get(metric)
        if not isinstance(value, (int, float)) or value < 0:
            errors.append(f"{path}: missing or invalid metric {metric}")

    return errors


def verify(prefix: str, *, check_documents: bool, check_presentations: bool) -> dict[str, Any]:
    errors: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}

    for dataset, dataset_spec in DATASETS.items():
        for strategy in STRATEGIES:
            key = f"{dataset}:{strategy}"
            path = _artifact_path(prefix, dataset, strategy)
            paths[key] = str(path)
            if not (ROOT / path).is_file():
                errors.append(f"missing artifact: {path}")
                continue
            try:
                payload = _load(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{path}: {error}")
                continue
            artifacts[key] = payload
            from core.paper_policy import configure_target_environment

            previous_environment = os.environ.copy()
            try:
                configure_target_environment(strategy, dataset, path.parts[2])
                errors.extend(
                    _validate_artifact(
                        path, payload, dataset=dataset, strategy=strategy,
                        expected_count=int(dataset_spec["count"]),
                    )
                )
                errors.extend(_validate_admission(path, payload, dataset, strategy))
            finally:
                os.environ.clear()
                os.environ.update(previous_environment)

    for dataset in DATASETS:
        group = [artifacts.get(f"{dataset}:{strategy}") for strategy in STRATEGIES]
        if any(payload is None for payload in group):
            continue
        for field in ("evaluated_query_ids_sha256", "corpus_manifest_fingerprint"):
            values = {payload.get(field) for payload in group if payload is not None}
            if len(values) != 1 or None in values:
                errors.append(f"{dataset}: strategies disagree on {field}: {sorted(map(str, values))}")

    model_config_fingerprints: dict[str, str] = {}
    for strategy in STRATEGIES:
        group = [artifacts.get(f"{dataset}:{strategy}") for dataset in DATASETS]
        if any(payload is None for payload in group):
            continue
        fingerprints = {_semantic_model_config_sha256(payload, strategy) for payload in group if payload is not None}
        if len(fingerprints) != 1:
            errors.append(f"{strategy}: datasets disagree on semantic model config: {sorted(fingerprints)}")
        else:
            model_config_fingerprints[strategy] = next(iter(fingerprints))

    if check_documents or check_presentations:
        current_values = {
            f"{float(payload[field]):.4f}"
            for dataset in DATASETS
            for strategy in STRATEGIES
            if (payload := artifacts.get(f"{dataset}:{strategy}")) is not None
            for field in DATASETS[dataset]["metrics"]
        }
        current_values.update(
            f"{float(payload['avg_latency']):.2f}"
            for payload in artifacts.values()
            if isinstance(payload.get("avg_latency"), (int, float))
        )

        value_targets: tuple[Path, ...] = ()
        config_targets: tuple[Path, ...] = ()
        if check_documents:
            value_targets += RESULT_DOCUMENTS
            config_targets += CONFIG_DOCUMENTS
        if check_presentations:
            value_targets += PRESENTATIONS
            config_targets += PRESENTATIONS
        for path in {*value_targets, *config_targets}:
            if not (ROOT / path).is_file():
                errors.append(f"missing publication file: {path}")
                continue
            text = (ROOT / path).read_text(encoding="utf-8")
            if path in value_targets:
                missing_values = sorted(value for value in current_values if value not in text)
                if missing_values:
                    errors.append(f"{path}: missing current values {missing_values}")
            if path not in config_targets:
                continue
            for required in (
                get_strategy("prehop").paper_generation_model,
                get_strategy("prehop").paper_embedding_model,
                f"{get_strategy('prehop').paper_embedding_dimensions:,}",
            ):
                if required not in text:
                    errors.append(f"{path}: missing current configuration value {required}")

    return {
        "status": "admitted" if not errors else "failed",
        "matrix_prefix": prefix,
        "artifacts_found": len(artifacts),
        "artifacts_expected": len(DATASETS) * len(STRATEGIES),
        "errors": errors,
        "artifact_paths": paths,
        "model_config_fingerprints": model_config_fingerprints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-prefix", default="naacl27-clean-20260905")
    parser.add_argument("--check-documents", action="store_true")
    parser.add_argument("--check-presentations", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from scripts.check_paper_runtime import _load_runner_environment

    _load_runner_environment()

    result = verify(
        args.matrix_prefix,
        check_documents=args.check_documents,
        check_presentations=args.check_presentations,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["status"] == "admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the current clean full-system matrix and its published values.

This is a manually invoked repository check. It does not run model inference
and is not wired to CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

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

STRATEGIES = ("prehop", "naive", "hoprag", "ms_graphrag", "browsenet", "proprag")
GENERATION_MODEL = "gemma-4-31b-it"
EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_DIMENSIONS = 4096

DOCUMENTS = (
    Path("README.md"),
    Path("docs/RESULTS.md"),
    Path("docs/prehop_paper.md"),
)

PRESENTATIONS = (
    Path("presentation/prehop-academic.html"),
    Path("presentation/prehop-professor-briefing.html"),
)


def _artifact_path(prefix: str, dataset: str, strategy: str) -> Path:
    run_id = f"{prefix}-{dataset}-{strategy}"
    filename = f"{strategy}_{dataset}.json"
    return Path("data/results") / run_id / strategy / dataset / "seed_42" / filename


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _validate_artifact(
    path: Path,
    payload: dict[str, Any],
    *,
    dataset: str,
    strategy: str,
    expected_count: int,
) -> list[str]:
    errors: list[str] = []

    expected = {
        "strategy": strategy,
        "corpus_tag": dataset,
        "evaluation_scope": "full_benchmark",
        "evaluated_queries_count": expected_count,
        "queries_count": expected_count,
        "total_queries": expected_count,
        "status": "completed",
        "benchmark_concurrency": 4,
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

    models = payload.get("models", {})
    for field, value in {
        "default": GENERATION_MODEL,
        "generation_revision": GENERATION_MODEL,
        "embedding": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_MODEL,
    }.items():
        if models.get(field) != value:
            errors.append(f"{path}: models.{field}={models.get(field)!r}, expected {value!r}")

    policy = payload.get("index_provenance", {}).get("policy", {})
    if policy.get("embedding_model") != EMBEDDING_MODEL:
        errors.append(
            f"{path}: index embedding_model={policy.get('embedding_model')!r}, "
            f"expected {EMBEDDING_MODEL!r}"
        )
    if policy.get("embedding_dimensions") != EMBEDDING_DIMENSIONS:
        errors.append(
            f"{path}: embedding_dimensions={policy.get('embedding_dimensions')!r}, "
            f"expected {EMBEDDING_DIMENSIONS}"
        )

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
            errors.extend(
                _validate_artifact(
                    path,
                    payload,
                    dataset=dataset,
                    strategy=strategy,
                    expected_count=int(dataset_spec["count"]),
                )
            )

    for dataset in DATASETS:
        group = [artifacts.get(f"{dataset}:{strategy}") for strategy in STRATEGIES]
        if any(payload is None for payload in group):
            continue
        for field in ("evaluated_query_ids_sha256", "corpus_manifest_fingerprint"):
            values = {payload.get(field) for payload in group if payload is not None}
            if len(values) != 1 or None in values:
                errors.append(f"{dataset}: strategies disagree on {field}: {sorted(map(str, values))}")

    source_digests = {
        payload.get("query_provenance", {}).get("source_tree_sha256")
        for payload in artifacts.values()
    }
    if artifacts and (None in source_digests or len(source_digests) != 1):
        errors.append(f"matrix strategies disagree on executable source digest: {sorted(map(str, source_digests))}")

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

        targets: tuple[Path, ...] = ()
        if check_documents:
            targets += DOCUMENTS
        if check_presentations:
            targets += PRESENTATIONS
        for path in targets:
            if not (ROOT / path).is_file():
                errors.append(f"missing publication file: {path}")
                continue
            text = (ROOT / path).read_text(encoding="utf-8")
            missing_values = sorted(value for value in current_values if value not in text)
            if missing_values:
                errors.append(f"{path}: missing current values {missing_values}")
            for required in (GENERATION_MODEL, EMBEDDING_MODEL, "4,096"):
                if required not in text:
                    errors.append(f"{path}: missing current configuration value {required}")

    return {
        "status": "passed" if not errors else "failed",
        "matrix_prefix": prefix,
        "artifacts_found": len(artifacts),
        "artifacts_expected": len(DATASETS) * len(STRATEGIES),
        "errors": errors,
        "artifact_paths": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-prefix", default="naacl27-clean-20260905")
    parser.add_argument("--check-documents", action="store_true")
    parser.add_argument("--check-presentations", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

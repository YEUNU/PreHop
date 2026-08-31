"""Bind a historical full MultiHop-RAG result to the current corpus manifest.

The early full baseline runs predate ``corpus_manifest.json``.  Re-running a
retrieval system is unnecessary when its immutable raw result already contains
the complete query inputs, retrieved passages, and per-query official metrics.
This verifier performs a read-only audit and writes a derived summary only
after all of the following are proven:

* the current query records and the raw result agree exactly;
* the current corpus files agree with their manifest;
* every retrieved passage is present in the current corpus;
* every official retrieval metric recomputes exactly; and
* the historical index artifact records a completed, exact-size active-source
  snapshot for the same prepared corpus directory.

The original benchmark and index artifacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OFFICIAL_FIELDS = (
    "official_hits@4",
    "official_hits@10",
    "official_mrr@10",
    "official_map@10",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _query_records_sha256(rows: list[dict[str, Any]]) -> str:
    records = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in sorted(rows, key=lambda item: str(item.get("_id") or ""))
    ]
    return _sha256_lines(records)


def _resolve_recorded_path(recorded: str, *, relative_to: Path) -> Path:
    path = Path(recorded)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, relative_to / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(recorded)


def _raw_result_path(summary_path: Path) -> Path:
    suffix = ".summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Summary filename must end with {suffix}: {summary_path}")
    return summary_path.with_name(summary_path.name[: -len(suffix)] + ".json")


def _validate_manifest(manifest_path: Path, queries: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _load_object(manifest_path)
    corpus_dir = manifest_path.parent
    files = sorted(path for path in corpus_dir.iterdir() if path.is_file() and path.suffix in {".txt", ".md"})
    source_ids = [path.stem for path in files]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Prepared corpus contains duplicate filename stems")
    if manifest.get("paragraph_count") != len(files):
        raise ValueError("Corpus file count does not match manifest paragraph_count")
    source_digest = _sha256_lines(source_ids)
    if manifest.get("source_ids_sha256") != source_digest:
        raise ValueError("Corpus source identifiers do not match the manifest")
    file_records = [f"{path.name}\0{_sha256_file(path)}" for path in files]
    if manifest.get("corpus_files_sha256") != _sha256_lines(file_records):
        raise ValueError("Corpus contents do not match the manifest")

    query_ids = sorted(str(row.get("_id") or "") for row in queries)
    if any(not query_id for query_id in query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("Current query records have missing or duplicate identifiers")
    if manifest.get("query_ids_sha256") != _sha256_lines(query_ids):
        raise ValueError("Current query identifiers do not match the manifest")
    if manifest.get("query_records_sha256") != _query_records_sha256(queries):
        raise ValueError("Current query records do not match the manifest")
    fingerprint_payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("Corpus manifest fingerprint does not match its identity fields")
    return manifest, {path.stem: path.read_text(encoding="utf-8") for path in files}


def _validate_index_stats(
    summary: dict[str, Any],
    stats: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> int:
    if stats.get("status") != "complete":
        raise ValueError("Historical index artifact is not complete")
    if stats.get("strategy") != summary.get("strategy"):
        raise ValueError("Historical index strategy differs from the benchmark strategy")
    if stats.get("corpus_tag") != summary.get("corpus_tag"):
        raise ValueError("Historical index corpus tag differs from the benchmark corpus tag")
    dataset_path = Path(str(stats.get("dataset_path") or ""))
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    if dataset_path != manifest_path.parent.resolve():
        raise ValueError("Historical index dataset path differs from the current manifest directory")
    timing = stats.get("timing_seconds")
    if not isinstance(timing, dict) or float(timing.get("active_snapshot_verified") or 0.0) != 1.0:
        raise ValueError("Historical index artifact lacks a successful active-snapshot check")
    source_count = int(float(timing.get("active_snapshot_source_count") or -1))
    if source_count != int(manifest["paragraph_count"]):
        raise ValueError("Historical active-source count differs from the current manifest")
    return source_count


def _normalize_source_id(value: Any) -> str:
    source = str(value or "")
    for suffix in (".txt", ".md"):
        if source.endswith(suffix):
            return source[: -len(suffix)]
    return source


def _validate_rows(
    raw: dict[str, Any],
    queries: list[dict[str, Any]],
    corpus: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    from utils.metrics import calculate_retrieval_ranking_metrics

    details = raw.get("details")
    if not isinstance(details, list):
        raise TypeError("Raw benchmark artifact has no details list")
    by_id = {str(row["_id"]): row for row in queries}
    observed_ids = [str(row.get("query_id") or "") for row in details if isinstance(row, dict)]
    if len(observed_ids) != len(details) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Raw benchmark details have missing or duplicate query identifiers")
    if set(observed_ids) != set(by_id):
        raise ValueError("Raw benchmark query identifiers differ from the current full query set")

    sums = {field: 0.0 for field in OFFICIAL_FIELDS}
    eligible = 0
    attributed_passages = 0
    ambiguous_passages = 0
    referenced_source_ids: set[str] = set()
    corpus_bodies = list(corpus.values())

    for result in details:
        query_id = str(result["query_id"])
        query = by_id[query_id]
        for field in ("query", "ground_truth", "category", "question_type"):
            if result.get(field) != query.get(field):
                raise ValueError(f"{query_id}: raw result differs from current query field {field!r}")
        expected_sources = result.get("expected_sources")
        if not isinstance(expected_sources, dict):
            raise TypeError(f"{query_id}: raw result has no expected_sources object")
        if expected_sources.get("docs", []) != query.get("evidence_docs", []):
            raise ValueError(f"{query_id}: evidence documents differ from the current query record")
        if expected_sources.get("facts", []) != query.get("evidence_facts", []):
            raise ValueError(f"{query_id}: evidence facts differ from the current query record")
        if result.get("error"):
            raise ValueError(f"{query_id}: historical full result contains an error row")

        retrieved = result.get("retrieved_sources") or []
        if not isinstance(retrieved, list):
            raise TypeError(f"{query_id}: retrieved_sources must be a list")
        for source in retrieved:
            if not isinstance(source, dict):
                raise TypeError(f"{query_id}: retrieved source must be an object")
            source_id = _normalize_source_id(source.get("source"))
            text = str(source.get("text") or "")
            if source_id:
                body = corpus.get(source_id)
                if body is None:
                    raise ValueError(f"{query_id}: retrieved source is outside the current corpus: {source_id}")
                if text and text not in body:
                    raise ValueError(f"{query_id}: retrieved text is absent from current source {source_id}")
                referenced_source_ids.add(source_id)
                attributed_passages += 1
            else:
                # Early HopRAG results deliberately blanked provenance when an
                # exact sentence appeared in more than one source.  The text
                # itself must still occur in the current corpus.
                if not text or not any(text in body for body in corpus_bodies):
                    raise ValueError(f"{query_id}: unattributed retrieved text is absent from the current corpus")
                ambiguous_passages += 1

        facts = query.get("evidence_facts") or []
        recomputed = calculate_retrieval_ranking_metrics(retrieved, facts)
        for field in OFFICIAL_FIELDS:
            if not math.isclose(float(result[field]), float(recomputed[field]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{query_id}: stored {field} differs from the current metric calculation")
        if facts:
            eligible += 1
            for field in OFFICIAL_FIELDS:
                sums[field] += float(recomputed[field])

    if int(summary.get("evaluated_queries_count") or -1) != len(details):
        raise ValueError("Summary evaluated query count differs from the raw full result")
    if int(summary.get("official_split_expected_queries") or -1) != len(details):
        raise ValueError("Historical result is not the complete declared official split")
    recomputed_averages = {field: sums[field] / eligible for field in OFFICIAL_FIELDS}
    for field, average in recomputed_averages.items():
        summary_field = f"avg_{field}"
        if not math.isclose(float(summary[summary_field]), average, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Summary {summary_field} differs from the recomputed raw result")
        count_field = f"eligible_{field}_count"
        if int(summary[count_field]) != eligible:
            raise ValueError(f"Summary {count_field} differs from the recomputed eligibility count")

    return {
        "query_count": len(details),
        "query_records_matched": len(details),
        "eligible_metric_rows": eligible,
        "retrieved_passages_attributed": attributed_passages,
        "retrieved_passages_with_ambiguous_exact_text_provenance": ambiguous_passages,
        "referenced_source_count": len(referenced_source_ids),
        "recomputed_official_metrics": {f"avg_{field}": value for field, value in recomputed_averages.items()},
    }


def verify_existing_baseline(
    summary_path: Path,
    queries_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    queries_path = queries_path.resolve()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    summary = _load_object(summary_path)
    if summary.get("strategy") == "prehop":
        raise ValueError("This verifier accepts non-Prehop baselines only")
    if summary.get("dataset") != "MultiHop-RAG" or summary.get("evaluation_scope") != "full_benchmark":
        raise ValueError("This verifier accepts complete MultiHop-RAG artifacts only")
    if summary.get("status") != "completed":
        raise ValueError("Historical benchmark artifact is not completed")
    if summary.get("corpus_index_fingerprint_status") != "manifest_absent":
        raise ValueError("Historical artifact must explicitly predate corpus-manifest binding")

    queries_payload = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries_payload, list) or not all(isinstance(row, dict) for row in queries_payload):
        raise TypeError("Current query file must be a JSON array of objects")
    queries: list[dict[str, Any]] = queries_payload
    manifest, corpus = _validate_manifest(manifest_path, queries)
    raw_path = _raw_result_path(summary_path)
    raw = _load_object(raw_path)
    for field in ("strategy", "corpus_tag", "dataset", "evaluation_scope", "status"):
        if raw.get(field) != summary.get(field):
            raise ValueError(f"Raw result and summary differ on {field!r}")
    if summary.get("evaluated_query_ids_sha256") != manifest.get("query_ids_sha256"):
        raise ValueError("Historical summary query digest differs from the current manifest")
    if raw.get("evaluated_query_ids_sha256") != manifest.get("query_ids_sha256"):
        raise ValueError("Historical raw-result query digest differs from the current manifest")
    stats_path = _resolve_recorded_path(
        str(summary.get("index_manifest_stats_path") or ""),
        relative_to=summary_path.parent,
    )
    stats = _load_object(stats_path)
    source_count = _validate_index_stats(summary, stats, manifest, manifest_path)
    row_checks = _validate_rows(raw, queries, corpus, summary)

    original_identity = {
        "corpus_manifest_path": summary.get("corpus_manifest_path"),
        "corpus_manifest_fingerprint": summary.get("corpus_manifest_fingerprint"),
        "corpus_manifest_paragraph_count": summary.get("corpus_manifest_paragraph_count"),
        "index_manifest_fingerprint": summary.get("index_manifest_fingerprint"),
        "corpus_index_fingerprint_status": summary.get("corpus_index_fingerprint_status"),
        "active_index_snapshot": summary.get("active_index_snapshot"),
    }
    verified = dict(summary)
    verified.update(
        {
            "corpus_manifest_path": str(manifest_path.relative_to(Path.cwd())),
            "corpus_manifest_fingerprint": manifest["fingerprint"],
            "corpus_manifest_paragraph_count": manifest["paragraph_count"],
            "index_manifest_fingerprint": manifest["fingerprint"],
            "index_manifest_status": "complete",
            "corpus_index_fingerprint_status": "matched",
            "active_index_snapshot": {
                "status": "matched",
                "source_count": source_count,
                "source_set_sha256": manifest["source_ids_sha256"],
                "verification_mode": "read_only_historical_artifact",
                "source_set_digest_basis": "strict historical active-snapshot check plus current manifest",
            },
            "official_metric_note": (
                "Full-split MultiHop-RAG retrieval metrics were recomputed from the immutable raw result. "
                "The historical exact-size active-source check and every retrieved passage were matched "
                "read-only to the current content-bound corpus manifest."
            ),
            "compatibility_verification": {
                "mode": "read_only_historical_artifact",
                "original_summary_path": str(summary_path.relative_to(Path.cwd())),
                "original_summary_sha256": _sha256_file(summary_path),
                "raw_result_path": str(raw_path.relative_to(Path.cwd())),
                "raw_result_sha256": _sha256_file(raw_path),
                "index_stats_path": str(stats_path.relative_to(Path.cwd())),
                "index_stats_sha256": _sha256_file(stats_path),
                "queries_path": str(queries_path.relative_to(Path.cwd())),
                "queries_sha256": _sha256_file(queries_path),
                "corpus_manifest_path": str(manifest_path.relative_to(Path.cwd())),
                "corpus_manifest_sha256": _sha256_file(manifest_path),
                "original_identity": original_identity,
                "checks": {
                    "current_manifest_content_matched": True,
                    "current_query_records_matched": True,
                    "historical_active_snapshot_verified": True,
                    "historical_active_source_count": source_count,
                    "retrieved_passages_matched_current_corpus": True,
                    "official_metrics_recomputed": True,
                    **row_checks,
                },
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verified, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="historical full summary")
    parser.add_argument("--queries", type=Path, required=True, help="current full prepared query file")
    parser.add_argument("--manifest", type=Path, required=True, help="current corpus manifest")
    parser.add_argument("--output", type=Path, required=True, help="derived verified summary")
    args = parser.parse_args()
    verified = verify_existing_baseline(args.summary, args.queries, args.manifest, args.output)
    checks = verified["compatibility_verification"]["checks"]
    print(
        json.dumps(
            {
                "status": "matched",
                "strategy": verified["strategy"],
                "query_count": checks["query_count"],
                "recomputed_official_metrics": checks["recomputed_official_metrics"],
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

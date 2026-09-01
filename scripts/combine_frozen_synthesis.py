"""Combine frozen MuSiQue retrieval with a later full answer-only run.

The shared answer prompt changed after some retrieval-heavy full baselines had
finished.  Their immutable retrieved passages and paragraph-support metrics do
not need to be recomputed.  This command verifies the complete query identity,
recomputes support metrics from the original raw result, recomputes answer
metrics from a complete answer-only artifact, and writes one derived summary.
Neither source artifact is modified.
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


def _raw_result_path(summary_path: Path) -> Path:
    suffix = ".summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Summary filename must end with {suffix}: {summary_path}")
    return summary_path.with_name(summary_path.name[: -len(suffix)] + ".json")


def _mean(rows: list[dict[str, float]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def combine_frozen_synthesis(
    baseline_summary_path: Path,
    queries_path: Path,
    synthesis_path: Path,
    synthesis_key: str,
    prompt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    from utils.metrics import calculate_answer_metrics, calculate_musique_support_metrics

    baseline_summary_path = baseline_summary_path.resolve()
    queries_path = queries_path.resolve()
    synthesis_path = synthesis_path.resolve()
    prompt_path = prompt_path.resolve()
    output_path = output_path.resolve()
    baseline = _load_object(baseline_summary_path)
    if baseline.get("strategy") == "prehop":
        raise ValueError("This command accepts non-Prehop baselines only")
    if baseline.get("dataset") != "MuSiQue" or baseline.get("evaluation_scope") != "full_benchmark":
        raise ValueError("Baseline must be a full MuSiQue artifact")
    if baseline.get("status") != "completed" or baseline.get("corpus_index_fingerprint_status") != "matched":
        raise ValueError("Baseline must already have completed, matched corpus/index provenance")

    queries_payload = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries_payload, list) or not all(isinstance(row, dict) for row in queries_payload):
        raise TypeError("Current query file must be a JSON array of objects")
    queries: list[dict[str, Any]] = queries_payload
    by_id = {str(row.get("_id") or ""): row for row in queries}
    if "" in by_id or len(by_id) != len(queries):
        raise ValueError("Current query records have missing or duplicate identifiers")
    query_digest = _sha256_lines(sorted(by_id))
    if baseline.get("evaluated_query_ids_sha256") != query_digest:
        raise ValueError("Baseline query digest differs from the current full query set")
    if int(baseline.get("evaluated_queries_count") or -1) != len(queries):
        raise ValueError("Baseline query count differs from the current full query set")

    raw_path = _raw_result_path(baseline_summary_path)
    raw = _load_object(raw_path)
    details = raw.get("details")
    if not isinstance(details, list):
        raise TypeError("Raw baseline artifact has no details list")
    raw_by_id = {str(row.get("query_id") or ""): row for row in details if isinstance(row, dict)}
    if len(raw_by_id) != len(details) or set(raw_by_id) != set(by_id):
        raise ValueError("Raw baseline details differ from the current full query set")

    support_rows: list[dict[str, float]] = []
    support_by_category: dict[str, list[dict[str, float]]] = {}
    for query_id, query in by_id.items():
        result = raw_by_id[query_id]
        for field in ("query", "ground_truth", "category", "question_type"):
            if result.get(field) != query.get(field):
                raise ValueError(f"{query_id}: raw baseline differs from current query field {field!r}")
        if result.get("error"):
            raise ValueError(f"{query_id}: raw baseline contains an error row")
        expected = result.get("expected_sources")
        if not isinstance(expected, dict):
            raise TypeError(f"{query_id}: raw baseline has no expected_sources object")
        if expected.get("paragraph_ids", []) != query.get("evidence_paragraph_ids", []):
            raise ValueError(f"{query_id}: gold paragraph identities differ from the current query record")
        support = calculate_musique_support_metrics(
            result.get("retrieved_sources") or [],
            query.get("evidence_paragraph_ids") or [],
        )
        for field, value in support.items():
            if not math.isclose(float(result[field]), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{query_id}: stored {field} differs from the current calculation")
        support_rows.append(support)
        support_by_category.setdefault(str(query["category"]), []).append(support)

    synthesis_payload = _load_object(synthesis_path)
    synthesis_rows = synthesis_payload.get(synthesis_key)
    if not isinstance(synthesis_rows, list) or not all(isinstance(row, dict) for row in synthesis_rows):
        raise TypeError(f"Synthesis artifact key {synthesis_key!r} must be a list of objects")
    synthesis_by_id = {str(row.get("query_id") or ""): row for row in synthesis_rows}
    if len(synthesis_by_id) != len(synthesis_rows) or set(synthesis_by_id) != set(by_id):
        raise ValueError("Answer-only artifact differs from the current full query set")
    if synthesis_path.stat().st_mtime_ns < prompt_path.stat().st_mtime_ns:
        raise ValueError("Answer-only artifact predates the selected answer prompt")

    answer_rows: list[dict[str, float]] = []
    answer_by_category: dict[str, list[dict[str, float]]] = {}
    answer_fields = (
        "answer_em",
        "answer_f1",
        "answer_precision",
        "answer_recall",
        "official_answer_em",
        "official_answer_f1",
    )
    synthesis_seconds = 0.0
    for query_id, query in by_id.items():
        result = synthesis_by_id[query_id]
        answer = str(result.get("answer") or "")
        if not answer.strip():
            raise ValueError(f"{query_id}: answer-only artifact contains an empty answer")
        metrics = calculate_answer_metrics(
            answer,
            str(query.get("ground_truth") or ""),
            answer_aliases=query.get("answer_aliases"),
            question_type=str(query.get("question_type") or ""),
        )
        selected = {field: float(metrics[field]) for field in answer_fields}
        for field in ("official_answer_em", "official_answer_f1"):
            if not math.isclose(float(result[field]), selected[field], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{query_id}: stored {field} differs from the current calculation")
        seconds = float(result.get("synthesis_seconds") or 0.0)
        if seconds <= 0.0:
            raise ValueError(f"{query_id}: answer-only artifact lacks positive synthesis timing")
        synthesis_seconds += seconds
        answer_rows.append(selected)
        answer_by_category.setdefault(str(query["category"]), []).append(selected)

    support_fields = (
        "paragraph_support_precision",
        "paragraph_support_recall",
        "paragraph_support_f1",
    )
    aggregate_answers = {f"avg_{field}": _mean(answer_rows, field) for field in answer_fields}
    aggregate_support = {f"avg_{field}": _mean(support_rows, field) for field in support_fields}
    avg_synthesis_seconds = synthesis_seconds / len(answer_rows)
    baseline_latency_count = int(baseline.get("eligible_latency_count") or 0)
    baseline_avg_latency = baseline.get("avg_latency")
    has_complete_baseline_timing = (
        baseline_latency_count == len(queries)
        and isinstance(baseline_avg_latency, (int, float))
        and float(baseline_avg_latency) > 0.0
    )
    avg_recorded_phase_sum_seconds = (
        float(baseline_avg_latency) + avg_synthesis_seconds if has_complete_baseline_timing else None
    )
    for field, value in aggregate_support.items():
        if not math.isclose(float(baseline[field]), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Baseline summary {field} differs from the recomputed raw retrieval result")

    category_summaries: dict[str, dict[str, Any]] = {}
    for category in sorted(answer_by_category):
        answers = answer_by_category[category]
        supports = support_by_category[category]
        category_summaries[category] = {
            "count": len(answers),
            **{f"avg_{field}": _mean(answers, field) for field in answer_fields},
            **{f"eligible_{field}_count": len(answers) for field in answer_fields},
            **{f"avg_{field}": _mean(supports, field) for field in support_fields},
            **{f"eligible_{field}_count": len(supports) for field in support_fields},
        }

    identity_fields = (
        "strategy",
        "corpus_tag",
        "dataset",
        "evaluation_scope",
        "official_split_expected_queries",
        "manifest_queries_count",
        "evaluated_queries_count",
        "evaluated_query_ids_sha256",
        "limit",
        "corpus_manifest_path",
        "corpus_manifest_fingerprint",
        "corpus_manifest_paragraph_count",
        "index_manifest_stats_path",
        "index_provenance",
        "index_manifest_fingerprint",
        "index_manifest_status",
        "corpus_index_fingerprint_status",
        "active_index_snapshot",
        "query_provenance",
        "evaluation_provenance",
        "models",
        "ablation",
        "judge_enabled",
        "judge_policy",
        "judge_independent",
        "judge_self_override",
    )
    combined = {field: baseline.get(field) for field in identity_fields}
    combined.update(
        {
            "status": "completed",
            "queries_count": len(queries),
            "total_queries": len(queries),
            "official_metric_note": (
                "MuSiQue support metrics were recomputed from the matched frozen retrieval result; "
                "answer EM/F1 were recomputed from a complete answer-only run created after the selected shared prompt."
            ),
            **aggregate_answers,
            **{f"eligible_{field}_count": len(answer_rows) for field in answer_fields},
            **aggregate_support,
            **{f"eligible_{field}_count": len(support_rows) for field in support_fields},
            "avg_synthesis_seconds": avg_synthesis_seconds,
            "eligible_synthesis_seconds_count": len(answer_rows),
            "avg_recorded_phase_sum_seconds": avg_recorded_phase_sum_seconds,
            "eligible_recorded_phase_sum_seconds_count": len(answer_rows) if has_complete_baseline_timing else 0,
            "recorded_phase_sum_definition": (
                "Mean latency of the original complete HopRAG workflow plus mean latency of the "
                "separate full answer-only run. This is the recorded cost of the two phases used "
                "to produce the combined result, not a single-pass end-to-end latency."
                if has_complete_baseline_timing
                else None
            ),
            "avg_latency": None,
            "eligible_latency_count": 0,
            "category_summaries": category_summaries,
            "combined_result_provenance": {
                "mode": "frozen_retrieval_plus_full_answer_only_run",
                "baseline_summary_path": str(baseline_summary_path.relative_to(Path.cwd())),
                "baseline_summary_sha256": _sha256_file(baseline_summary_path),
                "raw_retrieval_result_path": str(raw_path.relative_to(Path.cwd())),
                "raw_retrieval_result_sha256": _sha256_file(raw_path),
                "answer_only_result_path": str(synthesis_path.relative_to(Path.cwd())),
                "answer_only_result_sha256": _sha256_file(synthesis_path),
                "answer_only_result_key": synthesis_key,
                "answer_prompt_path": str(prompt_path.relative_to(Path.cwd())),
                "answer_prompt_sha256": _sha256_file(prompt_path),
                "answer_prompt_mtime_ns": prompt_path.stat().st_mtime_ns,
                "answer_only_result_mtime_ns": synthesis_path.stat().st_mtime_ns,
                "checks": {
                    "query_records_matched": len(queries),
                    "retrieval_support_metrics_recomputed": True,
                    "answer_metrics_recomputed": True,
                    "answer_only_result_postdates_prompt": True,
                },
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--answer-only-result", type=Path, required=True)
    parser.add_argument("--answer-only-key", required=True)
    parser.add_argument("--answer-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = combine_frozen_synthesis(
        args.baseline_summary,
        args.queries,
        args.answer_only_result,
        args.answer_only_key,
        args.answer_prompt,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": combined["status"],
                "strategy": combined["strategy"],
                "queries": combined["evaluated_queries_count"],
                "answer_em": combined["avg_official_answer_em"],
                "answer_f1": combined["avg_official_answer_f1"],
                "support_f1": combined["avg_paragraph_support_f1"],
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

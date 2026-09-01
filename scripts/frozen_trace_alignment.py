"""Align unordered frozen candidate traces to complete benchmark query IDs."""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def query_key(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _candidate_views(trace: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(path.get("channel") or ""), query_key(path.get("query_view")))
        for candidate in trace.get("candidates") or []
        for path in candidate.get("retrieval_paths") or []
        if path.get("kind") == "direct"
        and path.get("channel") in {"q_minus", "q_plus"}
        and query_key(path.get("query_view"))
    }


def _benchmark_views(trace: dict[str, Any]) -> set[tuple[str, str]]:
    views: set[tuple[str, str]] = set()
    for step in trace.get("interaction_trace") or []:
        if step.get("step") not in {"query_rewrite", "evidence_query_rewrite"}:
            continue
        output = step.get("output") or {}
        for channel in ("q_minus", "q_plus"):
            views.update(
                (channel, query_key(value))
                for value in output.get(channel) or []
                if query_key(value)
            )
    return views


def _view_similarity(candidate: dict[str, Any], benchmark: dict[str, Any]) -> float:
    candidate_views = _candidate_views(candidate)
    benchmark_views = _benchmark_views(benchmark)
    union = candidate_views | benchmark_views
    return len(candidate_views & benchmark_views) / len(union) if union else 1.0


def align_frozen_traces_to_gold(
    candidate_traces: list[dict[str, Any]],
    benchmark_path: Path,
    query_path: Path,
) -> list[dict[str, Any]]:
    """Return gold rows in candidate-trace order with duplicate-text guards."""
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("status") != "completed" or benchmark.get("evaluation_scope") != "full_benchmark":
        raise ValueError("Frozen analysis requires a completed full_benchmark artifact")
    details = benchmark.get("details") or []
    benchmark_trace_path = benchmark_path.with_name(f"{benchmark_path.stem}.traces.jsonl")
    benchmark_traces = _read_jsonl(benchmark_trace_path)
    gold_rows = json.loads(query_path.read_text())
    if not isinstance(gold_rows, list):
        raise TypeError(f"Expected a query list in {query_path}")
    expected = len(gold_rows)
    if len(details) != expected or len(benchmark_traces) != expected or len(candidate_traces) != expected:
        raise ValueError("Benchmark, benchmark trace, gold queries, and frozen trace must have equal counts")
    query_ids = [str(row.get("_id") or "") for row in gold_rows]
    if any(not value for value in query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("Gold query IDs are blank or non-unique")
    gold_by_id = {str(row["_id"]): row for row in gold_rows}

    for position, (detail, benchmark_trace) in enumerate(
        zip(details, benchmark_traces, strict=True),
        1,
    ):
        if int(benchmark_trace.get("idx", -1)) != position:
            raise ValueError(f"Benchmark trace ordering mismatch at position {position}")
        query_id = str(detail.get("query_id") or "")
        gold = gold_by_id.get(query_id)
        if gold is None:
            raise ValueError(f"Benchmark query ID is absent from gold input at position {position}: {query_id!r}")
        expected_key = query_key(gold.get("query"))
        if query_key(detail.get("query")) != expected_key or query_key(benchmark_trace.get("query")) != expected_key:
            raise ValueError(f"Benchmark detail/trace/gold query mismatch at position {position}")

    candidate_groups: dict[str, list[int]] = defaultdict(list)
    benchmark_groups: dict[str, list[int]] = defaultdict(list)
    for index, trace in enumerate(candidate_traces):
        candidate_groups[query_key(trace.get("query"))].append(index)
    for index, trace in enumerate(benchmark_traces):
        benchmark_groups[query_key(trace.get("query"))].append(index)
    if Counter({key: len(value) for key, value in candidate_groups.items()}) != Counter(
        {key: len(value) for key, value in benchmark_groups.items()}
    ):
        raise ValueError("Frozen trace and benchmark query-text multiplicities differ")

    aligned: list[dict[str, Any] | None] = [None] * expected
    for key, candidate_indices in candidate_groups.items():
        benchmark_indices = benchmark_groups[key]
        if len(candidate_indices) == 1:
            aligned[candidate_indices[0]] = gold_by_id[str(details[benchmark_indices[0]]["query_id"])]
            continue
        scored: list[tuple[float, tuple[int, ...]]] = []
        for permutation in itertools.permutations(benchmark_indices):
            score = sum(
                _view_similarity(candidate_traces[candidate_index], benchmark_traces[benchmark_index])
                for candidate_index, benchmark_index in zip(candidate_indices, permutation, strict=True)
            )
            scored.append((score, permutation))
        best_score = max(score for score, _ in scored)
        best = [permutation for score, permutation in scored if abs(score - best_score) < 1e-12]
        if len(best) > 1:
            label_assignments = {
                tuple(
                    tuple(
                        sorted(
                            gold_by_id[str(details[index]["query_id"])].get("evidence_paragraph_ids")
                            or []
                        )
                    )
                    for index in permutation
                )
                for permutation in best
            }
            if len(label_assignments) > 1:
                raise ValueError(f"Ambiguous duplicate query has different support labels: {key!r}")
        for candidate_index, benchmark_index in zip(candidate_indices, best[0], strict=True):
            aligned[candidate_index] = gold_by_id[str(details[benchmark_index]["query_id"])]

    if any(row is None for row in aligned):
        raise ValueError("One or more frozen traces were not aligned to a gold query")
    return [row for row in aligned if row is not None]

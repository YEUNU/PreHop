#!/usr/bin/env python3
"""Summarize non-overlapping stage timers from one complete benchmark.

Use this only for a benchmark run executed at a declared fixed concurrency.
The script rejects incomplete query/trace pairs and deliberately
excludes the historical ``traversal_ms`` aggregate from the stage sum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROW_METRICS = {
    "answer_em": ("answer_em", 1.0),
    "answer_f1": ("answer_f1", 1.0),
    "latency_seconds": ("latency", 1.0),
    "retrieve_seconds": ("retrieve_ms", 0.001),
    "graph_expand_seconds": ("graph_expand_ms", 0.001),
    "deterministic_score_seconds": ("deterministic_score_ms", 0.001),
    "candidate_order_seconds": ("candidate_order_ms", 0.001),
    "synthesis_seconds": ("synthesis_ms", 0.001),
}


ZERO_IF_MISSING = {field for field, scale in ROW_METRICS.values() if scale == 0.001}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _load(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    artifact = json.loads(path.read_text())
    details = artifact.get("details") or []
    traces_path = path.with_name(f"{path.stem}.traces.jsonl")
    traces = _read_jsonl(traces_path)
    if len(details) != len(traces):
        raise ValueError(f"Detail/trace count mismatch for {path}: {len(details)} != {len(traces)}")
    rows: dict[str, dict[str, Any]] = {}
    trace_by_id: dict[str, dict[str, Any]] = {}
    for position, (row, trace) in enumerate(zip(details, traces, strict=True), 1):
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in rows:
            raise ValueError(f"Missing or duplicate query ID at {path}:{position}")
        if int(trace.get("idx", -1)) != position or trace.get("query") != row.get("query"):
            raise ValueError(f"Detail/trace ordering mismatch for {query_id} in {path}")
        rows[query_id] = row
        trace_by_id[query_id] = trace
    return artifact, rows, trace_by_id


def _row_values(rows: dict[str, dict[str, Any]], query_ids: list[str], label: str) -> np.ndarray:
    field, scale = ROW_METRICS[label]
    return np.array(
        [
            float(rows[query_id].get(field, 0.0) if field in ZERO_IF_MISSING else rows[query_id][field]) * scale
            for query_id in query_ids
        ]
    )


def _point_estimates(rows: dict[str, dict[str, Any]], query_ids: list[str]) -> dict[str, dict[str, float]]:
    estimates: dict[str, dict[str, float]] = {}
    for label in ROW_METRICS:
        values = _row_values(rows, query_ids, label)
        estimates[label] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    accounted = sum(
        (
            _row_values(rows, query_ids, label)
            for label in (
                "retrieve_seconds",
                "graph_expand_seconds",
                "deterministic_score_seconds",
                "candidate_order_seconds",
                "synthesis_seconds",
            )
        ),
        start=np.zeros(len(query_ids)),
    )
    generation = sum(
        (
            _row_values(rows, query_ids, label)
            for label in (
                "candidate_order_seconds",
                "synthesis_seconds",
            )
        ),
        start=np.zeros(len(query_ids)),
    )
    estimates["accounted_stage_seconds"] = {
        "mean": float(accounted.mean()),
        "p50": float(np.percentile(accounted, 50)),
        "p95": float(np.percentile(accounted, 95)),
    }
    estimates["generation_stage_seconds"] = {
        "mean": float(generation.mean()),
        "p50": float(np.percentile(generation, 50)),
        "p95": float(np.percentile(generation, 95)),
    }
    return estimates


STAGE_TIMER_FIELDS = (
    "retrieve_ms",
    "graph_expand_ms",
    "deterministic_score_ms",
    "candidate_order_ms",
    "synthesis_ms",
)


def _rows_with_trace_timers(
    rows: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Copy non-overlapping timers from interaction steps into metric rows."""
    profiled: dict[str, dict[str, Any]] = {}
    for query_id, row in rows.items():
        interaction = traces[query_id].get("interaction_trace") or []
        timer_values = {
            field: [float(step[field]) for step in interaction if field in step]
            for field in STAGE_TIMER_FIELDS
        }
        missing = [field for field, values in timer_values.items() if not values]
        if missing:
            raise ValueError(f"Missing separated stage timers for {query_id}: {missing}")
        profiled[query_id] = {
            **row,
            **{field: sum(values) for field, values in timer_values.items()},
        }
    return profiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--declared-concurrency", type=int, required=True)
    args = parser.parse_args()
    if args.expected_queries < 1 or args.declared_concurrency < 1:
        parser.error("--expected-queries and --declared-concurrency must be positive")

    artifact, rows, traces = _load(args.artifact)
    if artifact.get("status") != "completed_unadmitted" or artifact.get("evaluation_scope") != "full_benchmark":
        raise ValueError("Stage profile requires a completed full_benchmark artifact")
    query_ids = sorted(rows)
    if len(query_ids) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} queries, found {len(query_ids)}")
    failures = [query_id for query_id, row in rows.items() if row.get("error")]
    if failures:
        raise ValueError(f"Complete stage profile contains {len(failures)} failed rows")

    profile_rows = _rows_with_trace_timers(rows, traces)
    point_estimates = _point_estimates(profile_rows, query_ids)
    generation_seconds = float(point_estimates["generation_stage_seconds"]["mean"])
    accounted_seconds = float(point_estimates["accounted_stage_seconds"]["mean"])
    categories = sorted({str(rows[query_id].get("category") or "unspecified") for query_id in query_ids})
    by_category: dict[str, Any] = {}
    for category in categories:
        category_ids = [
            query_id
            for query_id in query_ids
            if str(rows[query_id].get("category") or "unspecified") == category
        ]
        by_category[category] = {
            "queries": len(category_ids),
            "point_estimates": _point_estimates(profile_rows, category_ids),
        }
    profile: dict[str, Any] = {
        "scope": "complete_split_fixed_concurrency_stage_profile",
        "artifact_path": str(args.artifact),
        "dataset": artifact.get("dataset"),
        "queries": len(query_ids),
        "declared_concurrency": args.declared_concurrency,
        "timing_eligible": True,
        "timing_eligibility_scope": "within_run_stage_decomposition_only",
        "cross_run_absolute_timing_eligible": False,
        "timing_note": (
            "Stage means are non-overlapping. traversal_ms is excluded because it combines graph expansion, "
            "deterministic scoring, and candidate ordering. Absolute values are specific to the declared "
            "concurrency and service load; only the within-run decomposition is eligible."
        ),
        "point_estimates": point_estimates,
        "mean_generation_share_of_accounted_stages": (
            generation_seconds / accounted_seconds if accounted_seconds else 0.0
        ),
        "by_category": by_category,
        "ablation": artifact.get("ablation"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

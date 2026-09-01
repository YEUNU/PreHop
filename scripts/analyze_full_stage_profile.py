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

from scripts.analyze_refinement_caps import (
    _load,
    _point_estimates,
    _trace_summary,
)

STAGE_TIMER_FIELDS = (
    "rewrite_ms",
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
        has_rewrite_step = any(step.get("step") == "query_rewrite" for step in interaction)
        if not timer_values["rewrite_ms"] and not has_rewrite_step:
            timer_values["rewrite_ms"] = [0.0]
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
    if artifact.get("status") != "completed" or artifact.get("evaluation_scope") != "full_benchmark":
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
            "trace_summary": _trace_summary(traces, category_ids),
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
        "trace_summary": _trace_summary(traces, query_ids),
        "by_category": by_category,
        "ablation": artifact.get("ablation"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

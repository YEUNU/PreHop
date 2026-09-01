#!/usr/bin/env python3
"""Compare refinement caps with paired metrics, stage timing, and trace counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROW_METRICS = {
    "answer_em": ("answer_em", 1.0),
    "answer_f1": ("answer_f1", 1.0),
    "support_f1": ("paragraph_support_f1", 1.0),
    "support_recall": ("paragraph_support_recall", 1.0),
    "latency_seconds": ("latency", 1.0),
    "rewrite_seconds": ("rewrite_ms", 0.001),
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


def _bootstrap(values: np.ndarray, iterations: int, rng: np.random.Generator) -> dict[str, float]:
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


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
                "rewrite_seconds",
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
                "rewrite_seconds",
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


def _trace_summary(traces: dict[str, dict[str, Any]], query_ids: list[str]) -> dict[str, Any]:
    attempted: list[int] = []
    nonempty: list[int] = []
    stop_reasons: Counter[str] = Counter()
    cap_stops = 0
    for query_id in query_ids:
        steps = traces[query_id].get("interaction_trace") or []
        rewrite = next((step for step in steps if step.get("step") == "query_rewrite"), None)
        if rewrite is None:
            attempted.append(0)
            nonempty.append(0)
            stop_reasons["not_applicable"] += 1
            continue
        rounds = int(rewrite.get("refinement_rounds", 0))
        reason = str(rewrite.get("refinement_stop_reason") or "unspecified")
        attempted.append(rounds)
        stop_reasons[reason] += 1
        cap_stops += reason == "configured_round_cap"
        evidence_steps = [step for step in steps if step.get("step") == "evidence_query_rewrite"]
        nonempty.append(
            sum(
                bool((step.get("output") or {}).get("q_minus") or (step.get("output") or {}).get("q_plus"))
                for step in evidence_steps
            )
        )
    attempted_values = np.array(attempted)
    nonempty_values = np.array(nonempty)
    return {
        "mean_refinement_attempts": float(attempted_values.mean()),
        "p50_refinement_attempts": float(np.percentile(attempted_values, 50)),
        "p95_refinement_attempts": float(np.percentile(attempted_values, 95)),
        "max_refinement_attempts": int(attempted_values.max()),
        "mean_nonempty_refinements": float(nonempty_values.mean()),
        "cap_stop_rate": cap_stops / len(query_ids),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--condition", action="append", required=True, help="NAME=artifact.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-queries", type=int)
    args = parser.parse_args()

    baseline_artifact, baseline_rows, baseline_traces = _load(args.baseline)
    query_ids = sorted(baseline_rows)
    if args.expected_queries is not None and len(query_ids) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} queries, found {len(query_ids)}")

    report: dict[str, Any] = {
        "queries": len(query_ids),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "timing_note": (
            "The displayed stage sum excludes traversal_ms because that field aggregates graph, scoring, and "
            "candidate ordering. Per-call timers are summed across refinement rounds and may differ slightly "
            "from workflow latency because of scheduling and aggregation boundaries."
        ),
        "baseline": {
            "path": str(args.baseline),
            "ablation": baseline_artifact.get("ablation"),
            "point_estimates": _point_estimates(baseline_rows, query_ids),
            "trace_summary": _trace_summary(baseline_traces, query_ids),
        },
        "conditions": {},
    }
    for condition_index, value in enumerate(args.condition):
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --condition {value!r}; expected NAME=PATH")
        path = Path(raw_path.strip())
        artifact, rows, traces = _load(path)
        if set(rows) != set(query_ids):
            raise ValueError(f"Query IDs differ for {name}")
        paired: dict[str, dict[str, float]] = {}
        for metric_index, label in enumerate(ROW_METRICS):
            difference = _row_values(rows, query_ids, label) - _row_values(baseline_rows, query_ids, label)
            paired[label] = _bootstrap(
                difference,
                args.bootstrap_iterations,
                np.random.default_rng(args.seed + condition_index * 100 + metric_index),
            )
        report["conditions"][name.strip()] = {
            "path": str(path),
            "ablation": artifact.get("ablation"),
            "point_estimates": _point_estimates(rows, query_ids),
            "paired_differences": paired,
            "trace_summary": _trace_summary(traces, query_ids),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

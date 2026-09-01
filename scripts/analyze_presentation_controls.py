#!/usr/bin/env python3
"""Summarize query-aligned MuSiQue presentation controls.

The script compares query-aligned artifacts, bootstraps paired differences,
and measures how much the final evidence list changes between conditions.
The caller must declare the expected complete-split query count when the
result is intended for the professor briefing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

METRICS = {
    "answer_em": "answer_em",
    "answer_f1": "answer_f1",
    "support_f1": "paragraph_support_f1",
    "support_recall": "paragraph_support_recall",
    "latency_seconds": "latency",
}


def _load(path: Path) -> tuple[dict, dict[str, dict]]:
    artifact = json.loads(path.read_text())
    rows = {str(row["query_id"]): row for row in artifact["details"]}
    if len(rows) != len(artifact["details"]):
        raise ValueError(f"Duplicate query IDs in {path}")
    return artifact, rows


def _bootstrap(values: np.ndarray, *, iterations: int, rng: np.random.Generator) -> dict[str, float]:
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _candidate_stability(baseline: dict[str, dict], treatment: dict[str, dict]) -> dict[str, float]:
    jaccards: list[float] = []
    overlaps: list[int] = []
    taus: list[float] = []
    exact_sets = 0
    exact_orders = 0
    for query_id, base_row in baseline.items():
        treatment_row = treatment[query_id]
        base_ids = [str(source["chunk_id"]) for source in base_row.get("retrieved_sources") or []]
        treatment_ids = [str(source["chunk_id"]) for source in treatment_row.get("retrieved_sources") or []]
        base_set = set(base_ids)
        treatment_set = set(treatment_ids)
        intersection = base_set & treatment_set
        union = base_set | treatment_set
        jaccards.append(len(intersection) / len(union) if union else 1.0)
        overlaps.append(len(intersection))
        exact_sets += base_set == treatment_set
        exact_orders += base_ids == treatment_ids
        if len(intersection) >= 2:
            common_in_base_order = [chunk_id for chunk_id in base_ids if chunk_id in intersection]
            tau = kendalltau(
                range(len(common_in_base_order)),
                [treatment_ids.index(chunk_id) for chunk_id in common_in_base_order],
            ).statistic
            if tau is not None and not math.isnan(float(tau)):
                taus.append(float(tau))
    n = len(baseline)
    return {
        "mean_top12_jaccard": float(np.mean(jaccards)),
        "median_top12_jaccard": float(np.median(jaccards)),
        "mean_common_candidates": float(np.mean(overlaps)),
        "exact_candidate_set_rate": exact_sets / n,
        "exact_candidate_order_rate": exact_orders / n,
        "mean_kendall_tau_on_common_candidates": float(np.mean(taus)) if taus else 0.0,
        "median_kendall_tau_on_common_candidates": float(np.median(taus)) if taus else 0.0,
    }


def _point_estimates(
    rows: dict[str, dict],
    query_ids: list[str],
    metrics: dict[str, str],
) -> dict[str, dict[str, float]]:
    estimates: dict[str, dict[str, float]] = {}
    for label, field in metrics.items():
        values = np.array([float(rows[query_id][field]) for query_id in query_ids])
        estimates[label] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return estimates


def _category_ids(rows: dict[str, dict]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for query_id, row in rows.items():
        category = str(row.get("category") or row.get("question_type") or "unspecified")
        categories.setdefault(category, []).append(query_id)
    return {category: sorted(query_ids) for category, query_ids in sorted(categories.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="NAME=artifact.json; repeat for each treatment",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", default="query_aligned_component_diagnostic")
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument(
        "--exclude-latency",
        action="store_true",
        help="Omit latency when compared runs did not share a controlled load window.",
    )
    args = parser.parse_args()

    metrics = {
        label: field
        for label, field in METRICS.items()
        if not (args.exclude_latency and label == "latency_seconds")
    }

    baseline_artifact, baseline_rows = _load(args.baseline)
    conditions: dict[str, Path] = {}
    for value in args.condition:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --condition {value!r}; expected NAME=PATH")
        conditions[name.strip()] = Path(raw_path.strip())

    if args.expected_queries is not None and len(baseline_rows) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} baseline rows, found {len(baseline_rows)}")
    sorted_baseline_ids = sorted(baseline_rows)
    categories = _category_ids(baseline_rows)
    report: dict[str, object] = {
        "scope": args.scope,
        "queries": len(baseline_rows),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "latency_included": not args.exclude_latency,
        "baseline": {
            "path": str(args.baseline),
            "ablation": baseline_artifact.get("ablation"),
            "point_estimates": _point_estimates(baseline_rows, sorted_baseline_ids, metrics),
            "category_point_estimates": {
                category: {
                    "queries": len(query_ids),
                    "metrics": _point_estimates(baseline_rows, query_ids, metrics),
                }
                for category, query_ids in categories.items()
            },
        },
        "conditions": {},
    }

    baseline_ids = set(baseline_rows)
    for condition_index, (name, path) in enumerate(conditions.items()):
        artifact, rows = _load(path)
        if set(rows) != baseline_ids:
            raise ValueError(f"Query IDs differ for {name}")
        condition_report: dict[str, object] = {
            "path": str(path),
            "ablation": artifact.get("ablation"),
            "point_estimates": {},
            "paired_differences": {},
            "candidate_stability_vs_baseline": _candidate_stability(baseline_rows, rows),
            "category_results": {},
        }
        for metric_index, (label, field) in enumerate(metrics.items()):
            baseline_values = np.array([float(baseline_rows[qid][field]) for qid in sorted(baseline_ids)])
            values = np.array([float(rows[qid][field]) for qid in sorted(baseline_ids)])
            condition_report["point_estimates"][label] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            rng = np.random.default_rng(args.seed + condition_index * 100 + metric_index)
            condition_report["paired_differences"][label] = _bootstrap(
                values - baseline_values,
                iterations=args.bootstrap_iterations,
                rng=rng,
            )
        for category_index, (category, query_ids) in enumerate(categories.items()):
            category_report: dict[str, object] = {
                "queries": len(query_ids),
                "point_estimates": _point_estimates(rows, query_ids, metrics),
                "paired_differences": {},
            }
            for metric_index, (label, field) in enumerate(metrics.items()):
                baseline_values = np.array([float(baseline_rows[qid][field]) for qid in query_ids])
                values = np.array([float(rows[qid][field]) for qid in query_ids])
                rng = np.random.default_rng(
                    args.seed + 10_000 + condition_index * 1_000 + category_index * 100 + metric_index
                )
                category_report["paired_differences"][label] = _bootstrap(
                    values - baseline_values,
                    iterations=args.bootstrap_iterations,
                    rng=rng,
                )
            condition_report["category_results"][category] = category_report
        report["conditions"][name] = condition_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

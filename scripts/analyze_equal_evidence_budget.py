#!/usr/bin/env python3
"""Summarize paired results from frozen equal-evidence re-synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "answer_em",
    "answer_f1",
    "paragraph_support_precision",
    "paragraph_support_recall",
    "paragraph_support_f1",
    "retained_evidence_count",
    "retained_evidence_characters",
)


def _load_rows(target: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    rows = target.get("details") or []
    failed = [row for row in rows if row.get("error")]
    if failed:
        raise ValueError(f"Target {name!r} has {len(failed)} failed rows")
    by_id: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        row["retained_evidence_characters"] = sum(
            len(str(source.get("text") or "")) for source in row.get("retained_sources") or []
        )
        by_id[str(row["query_id"])] = row
    if len(by_id) != len(rows):
        raise ValueError(f"Target {name!r} has duplicate query IDs")
    return by_id


def _bootstrap(values: np.ndarray, iterations: int, rng: np.random.Generator) -> dict[str, float]:
    sampled = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(sampled, 2.5)),
        "ci95_high": float(np.percentile(sampled, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", default="prehop")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    targets = artifact.get("targets") or {}
    if args.reference not in targets:
        raise ValueError(f"Reference target {args.reference!r} is missing")
    rows_by_target = {name: _load_rows(target, name) for name, target in targets.items()}
    reference = rows_by_target[args.reference]
    query_ids = sorted(reference)
    if len(query_ids) != int(artifact.get("query_count") or 0):
        raise ValueError("Reference row count does not match query_count")
    if args.expected_queries is not None and len(query_ids) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} queries, found {len(query_ids)}")

    report: dict[str, Any] = {
        "scope": artifact.get("scope"),
        "limitations": artifact.get("limitations"),
        "input_path": str(args.input),
        "reference": args.reference,
        "queries": len(query_ids),
        "evidence_top_k": artifact.get("evidence_top_k"),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "targets": {},
    }
    for target_index, (name, rows) in enumerate(rows_by_target.items()):
        if sorted(rows) != query_ids:
            raise ValueError(f"Target {name!r} does not have the reference query IDs")
        target_report: dict[str, Any] = {"point_estimates": {}, "paired_difference_vs_reference": {}}
        for metric_index, metric in enumerate(METRICS):
            values = np.array([float(rows[query_id][metric]) for query_id in query_ids])
            reference_values = np.array([float(reference[query_id][metric]) for query_id in query_ids])
            target_report["point_estimates"][metric] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            rng = np.random.default_rng(args.seed + target_index * 100 + metric_index)
            target_report["paired_difference_vs_reference"][metric] = _bootstrap(
                values - reference_values,
                args.bootstrap_iterations,
                rng,
            )
        report["targets"][name] = target_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

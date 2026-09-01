#!/usr/bin/env python3
"""Estimate graph-on minus graph-off effects in structural MuSiQue subgroups.

The analysis is paired by query ID. Structural subgroup membership comes from
``analyze_gold_hop_coverage.py`` and is fixed independently of either answer
run. The output is exploratory because graph connectivity to gold evidence is
not available in a real query and the compared generation calls are separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROW_METRICS = {
    "answer_em": "answer_em",
    "answer_f1": "answer_f1",
    "support_f1": "paragraph_support_f1",
    "support_recall": "paragraph_support_recall",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_run(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    artifact = json.loads(path.read_text())
    details = artifact.get("details") or []
    traces = _read_jsonl(path.with_name(f"{path.stem}.traces.jsonl"))
    if len(details) != len(traces):
        raise ValueError(f"Detail/trace count mismatch for {path}: {len(details)} != {len(traces)}")
    rows: dict[str, dict[str, Any]] = {}
    execution: dict[str, dict[str, float]] = {}
    for position, (row, trace) in enumerate(zip(details, traces, strict=True), 1):
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in rows:
            raise ValueError(f"Missing or duplicate query ID at {path}:{position}")
        if int(trace.get("idx", -1)) != position or trace.get("query") != row.get("query"):
            raise ValueError(f"Detail/trace ordering mismatch for {query_id} in {path}")
        steps = trace.get("interaction_trace") or []
        refinement_steps = [step for step in steps if step.get("step") == "evidence_query_rewrite"]
        nonempty = sum(
            bool((step.get("output") or {}).get("q_minus") or (step.get("output") or {}).get("q_plus"))
            for step in refinement_steps
        )
        has_rewrite = any(step.get("step") == "query_rewrite" for step in steps)
        retrieve_step = next((step for step in steps if step.get("step") == "retrieve"), {})
        path_counts = (retrieve_step.get("output") or {}).get("retrieval_path_counts") or {}
        rows[query_id] = row
        execution[query_id] = {
            "refinement_attempts": float(len(refinement_steps)),
            "nonempty_refinements": float(nonempty),
            "inferred_retrieval_passes": float(2 + nonempty if has_rewrite else 1),
            "selected_hop_paths": float(path_counts.get("hop", 0)),
        }
    return rows, execution


def _bootstrap(values: np.ndarray, iterations: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _point(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-on", type=Path, required=True)
    parser.add_argument("--graph-off", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--scope", default="paired_exploratory_structural_subgroups")
    args = parser.parse_args()

    on_rows, on_execution = _load_run(args.graph_on)
    off_rows, off_execution = _load_run(args.graph_off)
    if set(on_rows) != set(off_rows):
        raise ValueError("Graph-on and graph-off query IDs differ")
    if args.expected_queries is not None and len(on_rows) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} queries, found {len(on_rows)}")

    coverage = json.loads(args.coverage.read_text())
    coverage_rows = {str(row["query_id"]): row for row in coverage.get("details") or []}
    missing_coverage = set(on_rows) - set(coverage_rows)
    if missing_coverage:
        raise ValueError(f"Coverage is missing {len(missing_coverage)} benchmark query IDs")

    groups = {
        "all": sorted(on_rows),
        "2hop": sorted(query_id for query_id in on_rows if coverage_rows[query_id].get("category") == "2hop"),
        "3hop": sorted(query_id for query_id in on_rows if coverage_rows[query_id].get("category") == "3hop"),
        "4hop": sorted(query_id for query_id in on_rows if coverage_rows[query_id].get("category") == "4hop"),
        "gold_edge_present": sorted(query_id for query_id in on_rows if coverage_rows[query_id]["any_gold_edge"]),
        "gold_edge_absent": sorted(query_id for query_id in on_rows if not coverage_rows[query_id]["any_gold_edge"]),
        "gold_subgraph_connected": sorted(
            query_id for query_id in on_rows if coverage_rows[query_id]["gold_subgraph_connected"]
        ),
    }
    report: dict[str, Any] = {
        "scope": args.scope,
        "definition": "Every effect is graph-on minus graph-off on the same query IDs.",
        "limitations": [
            "Hop-depth groups use the fixed benchmark category from the coverage artifact.",
            "Gold-edge subgroup labels are retrospective diagnostics and are unavailable at inference time.",
            "A structural gold edge need not be activated or selected for the corresponding query.",
            "Answer generation was rerun separately, so decoding variability remains in paired differences.",
            "Latency is intentionally excluded because the runs did not share a controlled load window.",
        ],
        "graph_on_path": str(args.graph_on),
        "graph_off_path": str(args.graph_off),
        "coverage_path": str(args.coverage),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "groups": {},
    }
    all_metrics = {
        **ROW_METRICS,
        "nonempty_refinements": "nonempty_refinements",
        "retrieval_passes": "inferred_retrieval_passes",
    }
    for group_index, (group_name, query_ids) in enumerate(groups.items()):
        group: dict[str, Any] = {"queries": len(query_ids), "metrics": {}}
        for metric_index, (label, field) in enumerate(all_metrics.items()):
            source_on = on_execution if field in {"nonempty_refinements", "inferred_retrieval_passes"} else on_rows
            source_off = off_execution if field in {"nonempty_refinements", "inferred_retrieval_passes"} else off_rows
            on_values = np.array([float(source_on[query_id][field]) for query_id in query_ids])
            off_values = np.array([float(source_off[query_id][field]) for query_id in query_ids])
            group["metrics"][label] = {
                "graph_on": _point(on_values),
                "graph_off": _point(off_values),
                "effect_on_minus_off": _bootstrap(
                    on_values - off_values,
                    args.bootstrap_iterations,
                    args.seed + group_index * 100 + metric_index,
                ),
            }
        group["graph_on_selected_hop_path_rate"] = float(
            np.mean([on_execution[query_id]["selected_hop_paths"] > 0 for query_id in query_ids])
        )
        report["groups"][group_name] = group

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

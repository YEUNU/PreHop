#!/usr/bin/env python3
"""Evaluate deterministic rank variants over every frozen MuSiQue pool.

This analysis does not rerun retrieval, rewriting, candidate ordering, or
answer generation.  It reconstructs the deterministic top-k order from the
scores captured by a complete benchmark and evaluates paragraph support over
all query IDs.  The output therefore tests the rank formula directly without
mixing in generation-model ordering variability.

When per-channel representation scores are present, the analysis also
reconstructs graph propagation with decay values zero and one and validates
that decay 0.5 reproduces every captured graph-only score first.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from scripts.frozen_trace_alignment import align_frozen_traces_to_gold
from utils.metrics import calculate_musique_support_metrics

SUPPORT_FIELDS = (
    "paragraph_support_precision",
    "paragraph_support_recall",
    "paragraph_support_f1",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No trace rows found in {path}")
    return rows


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("node_id") or "")


def _sort_by(candidates: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (float(item.get(field, 0.0)), _candidate_id(item)),
        reverse=True,
    )


def _fused_order(
    candidates: list[dict[str, Any]],
    semantic_score: Callable[[dict[str, Any]], float],
    representation_score: Callable[[dict[str, Any]], float] | None = None,
) -> list[dict[str, Any]]:
    representation_score = representation_score or (lambda item: float(item.get("representation_score", 0.0)))
    semantic = sorted(
        candidates,
        key=lambda item: (float(semantic_score(item)), _candidate_id(item)),
        reverse=True,
    )
    representation = sorted(
        candidates,
        key=lambda item: (float(representation_score(item)), _candidate_id(item)),
        reverse=True,
    )
    semantic_ranks = {_candidate_id(item): rank for rank, item in enumerate(semantic)}
    representation_ranks = {
        _candidate_id(item): rank
        for rank, item in enumerate(representation)
        if float(representation_score(item)) > 0.0
    }

    def key(item: dict[str, Any]) -> tuple[float, float, str]:
        node_id = _candidate_id(item)
        score = 1.0 / (semantic_ranks[node_id] + 1)
        if node_id in representation_ranks:
            score += 1.0 / (representation_ranks[node_id] + 1)
        return score, float(semantic_score(item)), node_id

    return sorted(candidates, key=key, reverse=True)


def _graph_representation_score(
    candidate: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    decay: float,
) -> float:
    paths = list(candidate.get("retrieval_paths") or [])
    if any(str(path.get("kind") or "") == "direct" for path in paths):
        return float(candidate.get("representation_score", 0.0))
    inherited: list[float] = []
    for path in paths:
        kind = str(path.get("kind") or "")
        if kind not in {"hop", "continuation", "next"}:
            continue
        source_id = str(path.get("source_chunk_id") or "")
        source = by_id.get(source_id)
        if source is None:
            raise ValueError(f"Missing graph source candidate {source_id!r}")
        if kind == "hop":
            score = float((source.get("representation_scores") or {}).get("q_plus", 0.0))
        elif kind == "continuation":
            score = float((source.get("representation_scores") or {}).get("q_minus", 0.0))
        else:
            score = float(source.get("representation_score", 0.0))
        inherited.append(score * decay)
    return max(inherited, default=float(candidate.get("representation_score", 0.0)))


def _orders(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    candidates = list(row["candidates"])
    by_id = {_candidate_id(candidate): candidate for candidate in candidates}
    if len(by_id) != len(candidates) or "" in by_id:
        raise ValueError(f"Invalid candidate identity for query {row.get('query')!r}")
    fused = [by_id[str(node_id)] for node_id in row["canonical_node_ids"]]
    recomputed = _fused_order(candidates, lambda item: float(item.get("final_score", 0.0)))
    if [_candidate_id(item) for item in recomputed] != [_candidate_id(item) for item in fused]:
        raise ValueError(f"Captured fused order cannot be reproduced for query {row.get('query')!r}")

    def bridge_score(item: dict[str, Any]) -> float:
        value = item.get("bridge_similarity_score")
        return float(item.get("similarity_score", 0.0) if value is None else value)

    variants = {
        "fused": fused,
        "semantic_only": _sort_by(candidates, "final_score"),
        "representation_only": _sort_by(candidates, "representation_score"),
        "body_only_fused": _fused_order(candidates, lambda item: float(item.get("similarity_score", 0.0))),
        "bridge_only_fused": _fused_order(candidates, bridge_score),
    }
    if all("representation_scores" in candidate for candidate in candidates):
        for candidate in candidates:
            paths = list(candidate.get("retrieval_paths") or [])
            graph_only = paths and not any(str(path.get("kind") or "") == "direct" for path in paths)
            if graph_only:
                expected = _graph_representation_score(candidate, by_id, 0.5)
                captured = float(candidate.get("representation_score", 0.0))
                if not np.isclose(expected, captured):
                    raise ValueError(
                        f"Graph score cannot be reproduced for query {row.get('query')!r}: "
                        f"expected={expected} captured={captured}"
                    )
        variants["graph_decay_zero_fused"] = _fused_order(
            candidates,
            lambda item: float(item.get("final_score", 0.0)),
            lambda item: _graph_representation_score(item, by_id, 0.0),
        )
        variants["graph_decay_one_fused"] = _fused_order(
            candidates,
            lambda item: float(item.get("final_score", 0.0)),
            lambda item: _graph_representation_score(item, by_id, 1.0),
        )
    return variants


def _bootstrap(values: np.ndarray, *, iterations: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _stability(reference: list[str], treatment: list[str]) -> dict[str, float]:
    reference_set = set(reference)
    treatment_set = set(treatment)
    union = reference_set | treatment_set
    return {
        "jaccard": len(reference_set & treatment_set) / len(union) if union else 1.0,
        "exact_set": float(reference_set == treatment_set),
        "exact_order": float(reference == treatment),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    traces = _load_jsonl(args.trace)
    if args.expected_queries is not None and len(traces) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} trace rows, found {len(traces)}")
    aligned_gold = align_frozen_traces_to_gold(traces, args.benchmark, args.queries)

    details: list[dict[str, Any]] = []
    for trace, query_row in zip(traces, aligned_gold, strict=True):
        query = str(trace["query"])
        variants = _orders(trace)
        variant_rows: dict[str, Any] = {}
        for name, ordered in variants.items():
            selected = ordered[: args.top_k]
            selected_ids = [_candidate_id(candidate) for candidate in selected]
            sources = [
                {
                    "paragraph_id": candidate.get("paragraph_id"),
                    "source": candidate.get("source"),
                    "title": candidate.get("title"),
                    "text": candidate.get("text"),
                }
                for candidate in selected
            ]
            metrics = calculate_musique_support_metrics(
                sources,
                query_row.get("evidence_paragraph_ids") or [],
            )
            variant_rows[name] = {
                "selected_node_ids": selected_ids,
                **{field: float(metrics[field]) for field in SUPPORT_FIELDS},
            }
        details.append(
            {
                "query_id": str(query_row["_id"]),
                "query": query,
                "category": str(query_row.get("category") or "unspecified"),
                "variants": variant_rows,
            }
        )

    variant_names = list(details[0]["variants"])
    report: dict[str, Any] = {
        "scope": "complete_split_frozen_deterministic_rank_analysis",
        "limitations": [
            "Retrieval and query rewriting are frozen from one complete baseline run.",
            "This analysis evaluates deterministic top-k support, not answer generation.",
            "The generation-model candidate-ordering call is intentionally bypassed.",
            "Duplicate query texts are aligned through completed benchmark query IDs and generated Q-/Q+ views.",
        ],
        "trace_path": str(args.trace),
        "query_path": str(args.queries),
        "benchmark_path": str(args.benchmark),
        "queries": len(details),
        "top_k": args.top_k,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "variants": {},
        "details": details,
    }
    baseline_name = "fused"
    categories = sorted({row["category"] for row in details})
    for variant_index, name in enumerate(variant_names):
        summary: dict[str, Any] = {"point_estimates": {}, "paired_vs_fused": {}, "categories": {}}
        for metric_index, field in enumerate(SUPPORT_FIELDS):
            values = np.array([row["variants"][name][field] for row in details], dtype=float)
            baseline = np.array([row["variants"][baseline_name][field] for row in details], dtype=float)
            summary["point_estimates"][field] = float(values.mean())
            summary["paired_vs_fused"][field] = _bootstrap(
                values - baseline,
                iterations=args.bootstrap_iterations,
                seed=args.seed + variant_index * 100 + metric_index,
            )
        stability_rows = [
            _stability(
                row["variants"][baseline_name]["selected_node_ids"],
                row["variants"][name]["selected_node_ids"],
            )
            for row in details
        ]
        summary["selection_vs_fused"] = {
            key: float(np.mean([item[key] for item in stability_rows])) for key in stability_rows[0]
        }
        for category_index, category in enumerate(categories):
            rows = [row for row in details if row["category"] == category]
            category_summary: dict[str, Any] = {"queries": len(rows), "paired_vs_fused": {}}
            for metric_index, field in enumerate(SUPPORT_FIELDS):
                values = np.array([row["variants"][name][field] for row in rows], dtype=float)
                baseline = np.array([row["variants"][baseline_name][field] for row in rows], dtype=float)
                category_summary["paired_vs_fused"][field] = _bootstrap(
                    values - baseline,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 10_000 + variant_index * 1_000 + category_index * 100 + metric_index,
                )
            summary["categories"][category] = category_summary
        report["variants"][name] = summary

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

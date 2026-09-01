#!/usr/bin/env python3
"""Replay candidate-list ordering on frozen Prehop candidate pools.

The retrieval and query-rewrite stages are not rerun. Each trace record keeps
the exact candidate texts and canonical deterministic fused order from one completed list
ordering call. This script replays declared input orders, measures selection
stability, and joins support labels through the completed benchmark query IDs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from core.vllm_client import VLLMClient
from models.prehop.llm_json import generate_json_or_raise
from scripts.frozen_trace_alignment import align_frozen_traces_to_gold
from utils.metrics import calculate_musique_support_metrics
from utils.prompts.query_rewrite import build_evidence_ranking_prompt

SUPPORT_FIELDS = (
    "paragraph_support_precision",
    "paragraph_support_recall",
    "paragraph_support_f1",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        candidates = row.get("candidates") or []
        node_ids = [str(candidate.get("node_id") or "") for candidate in candidates]
        if not candidates or any(not node_id for node_id in node_ids):
            raise ValueError(f"{path}:{line_number}: candidate pool is empty or has a blank node ID")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"{path}:{line_number}: candidate node IDs are not unique")
        if node_ids != [str(value) for value in row.get("canonical_node_ids") or []]:
            raise ValueError(f"{path}:{line_number}: candidates do not match canonical_node_ids")
        rows.append(row)
    if not rows:
        raise ValueError(f"No trace records found in {path}")
    return rows


def _support_metrics(
    trace: dict[str, Any],
    selected_node_ids: list[str],
    gold_row: dict[str, Any],
) -> dict[str, float]:
    by_id = {str(candidate["node_id"]): candidate for candidate in trace["candidates"]}
    sources = []
    for node_id in selected_node_ids:
        candidate = by_id.get(str(node_id))
        if candidate is None:
            raise ValueError(f"Selected node {node_id!r} is absent from frozen pool")
        sources.append(
            {
                "paragraph_id": candidate.get("paragraph_id"),
                "source": candidate.get("source"),
                "title": candidate.get("title"),
                "text": candidate.get("text"),
            }
        )
    metrics = calculate_musique_support_metrics(
        sources,
        gold_row.get("evidence_paragraph_ids") or [],
    )
    return {field: float(metrics[field]) for field in SUPPORT_FIELDS}


def _ordered_candidates(
    row: dict[str, Any],
    order: str,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = list(row["candidates"])
    if order == "reverse":
        candidates.reverse()
    elif order == "hash_shuffle":
        query = str(row["query"])

        def key(candidate: dict[str, Any]) -> tuple[bytes, str]:
            node_id = str(candidate["node_id"])
            payload = f"{seed}\0{query}\0{node_id}".encode()
            return hashlib.sha256(payload).digest(), node_id

        candidates.sort(key=key)
    elif order != "search":
        raise ValueError(f"Unsupported order: {order}")
    return candidates


def _candidate_metadata(candidate: dict[str, Any]) -> str:
    labels = {
        "publisher": "Publisher",
        "published_at": "Published",
        "author": "Author",
        "category": "Category",
    }
    metadata = candidate.get("metadata") or {}
    return "; ".join(f"{labels[key]}: {metadata[key]}" for key in labels if metadata.get(key))


async def _replay_one(
    client: VLLMClient,
    row: dict[str, Any],
    order: str,
    seed: int,
) -> dict[str, Any]:
    ordered = _ordered_candidates(row, order, seed)
    candidate_ids = [f"C{index:03d}" for index in range(len(ordered))]
    by_candidate_id = dict(zip(candidate_ids, ordered, strict=True))
    prompt = build_evidence_ranking_prompt(
        str(row["query"]),
        [
            (
                candidate_id,
                str(candidate.get("title") or ""),
                _candidate_metadata(candidate),
                str(candidate.get("text") or ""),
            )
            for candidate_id, candidate in by_candidate_id.items()
        ],
        int(row["top_k"]),
    )
    payload = await generate_json_or_raise(
        client,
        [{"role": "user", "content": prompt}],
        "frozen evidence ranking",
        f"query={row['query']!r} order={order}",
        required_fields={"ranking": list},
        temperature=0.0,
        max_tokens=1024,
    )
    returned: list[str] = []
    for value in payload["ranking"]:
        candidate_id = str(value)
        if candidate_id in by_candidate_id and candidate_id not in returned:
            returned.append(candidate_id)
    complete = [*returned, *(candidate_id for candidate_id in candidate_ids if candidate_id not in returned)]
    selected_ids = [
        str(by_candidate_id[candidate_id]["node_id"])
        for candidate_id in complete[: int(row["top_k"])]
    ]
    return {
        "order": order,
        "input_node_ids": [str(candidate["node_id"]) for candidate in ordered],
        "model_returned_node_ids": [str(by_candidate_id[candidate_id]["node_id"]) for candidate_id in returned],
        "selected_node_ids": selected_ids,
    }


def _stability(reference: list[str], treatment: list[str]) -> dict[str, float]:
    reference_set = set(reference)
    treatment_set = set(treatment)
    common = reference_set & treatment_set
    union = reference_set | treatment_set
    tau = 0.0
    if len(common) >= 2:
        ordered_common = [node_id for node_id in reference if node_id in common]
        value = kendalltau(
            range(len(ordered_common)),
            [treatment.index(node_id) for node_id in ordered_common],
        ).statistic
        if value is not None and not math.isnan(float(value)):
            tau = float(value)
    return {
        "jaccard": len(common) / len(union) if union else 1.0,
        "common_candidates": float(len(common)),
        "exact_set": float(reference_set == treatment_set),
        "exact_order": float(reference == treatment),
        "kendall_tau_on_common": tau,
    }


def _input_order_dependence(replay: dict[str, Any]) -> dict[str, float]:
    input_ids = replay["input_node_ids"]
    selected_ids = replay["selected_node_ids"]
    normalized_ranks = [input_ids.index(node_id) / max(1, len(input_ids) - 1) for node_id in selected_ids]
    tau = 0.0
    if len(selected_ids) >= 2:
        value = kendalltau(
            range(len(selected_ids)),
            [input_ids.index(node_id) for node_id in selected_ids],
        ).statistic
        if value is not None and not math.isnan(float(value)):
            tau = float(value)
    return {
        "mean_normalized_input_rank_of_selected": float(np.mean(normalized_ranks)),
        "input_first_selected": float(bool(input_ids) and input_ids[0] in selected_ids),
        "output_input_kendall_tau": tau,
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _uncertainty_dict(
    rows: list[dict[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Summarize paired replay diagnostics over trace records."""
    if not rows:
        return {}
    report: dict[str, dict[str, float]] = {}
    for metric_index, key in enumerate(rows[0]):
        values = np.array([row[key] for row in rows], dtype=float)
        rng = np.random.default_rng(seed + metric_index)
        samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        report[key] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "ci95_low": float(np.percentile(samples, 2.5)),
            "ci95_high": float(np.percentile(samples, 97.5)),
        }
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


async def main_async(args: argparse.Namespace) -> None:
    traces = _load_jsonl(args.trace)
    if args.expected_traces is not None and len(traces) != args.expected_traces:
        raise ValueError(f"Expected {args.expected_traces} trace records, found {len(traces)}")
    aligned_gold = align_frozen_traces_to_gold(traces, args.benchmark, args.queries)
    client = VLLMClient()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    completed: list[dict[str, Any]] = []
    if args.resume and args.out.exists():
        prior = json.loads(args.out.read_text())
        if str(prior.get("trace_path")) != str(args.trace):
            raise ValueError("Resume artifact trace_path does not match --trace")
        if str(prior.get("query_path")) != str(args.queries):
            raise ValueError("Resume artifact query_path does not match --queries")
        if str(prior.get("benchmark_path")) != str(args.benchmark):
            raise ValueError("Resume artifact benchmark_path does not match --benchmark")
        if list(prior.get("orders") or []) != list(args.orders):
            raise ValueError("Resume artifact orders do not match --orders")
        if int(prior.get("shuffle_seed", -1)) != args.seed:
            raise ValueError("Resume artifact shuffle_seed does not match --seed")
        completed = [record for record in (prior.get("details") or []) if not record.get("error")]
    completed_indices = {int(record["trace_index"]) for record in completed}
    write_lock = asyncio.Lock()

    async def process(index: int, trace: dict[str, Any]) -> None:
        async with semaphore:
            gold_row = aligned_gold[index]
            condition_results: dict[str, Any] = {}
            error = ""
            try:
                for order in args.orders:
                    condition_results[order] = await _replay_one(client, trace, order, args.seed)
                    condition_results[order]["support_metrics"] = _support_metrics(
                        trace,
                        condition_results[order]["selected_node_ids"],
                        gold_row,
                    )
            except Exception as exc:  # noqa: BLE001 - keep every failed replay inspectable
                error = f"{type(exc).__name__}: {exc}"
            captured_support = _support_metrics(
                trace,
                [str(value) for value in trace["selected_node_ids"]],
                gold_row,
            )
            record = {
                "trace_index": index,
                "query_id": str(gold_row["_id"]),
                "query": trace["query"],
                "category": str(gold_row.get("category") or "unspecified"),
                "pool_size": len(trace["candidates"]),
                "top_k": int(trace["top_k"]),
                "captured_selected_node_ids": trace["selected_node_ids"],
                "captured_support_metrics": captured_support,
                "replays": condition_results,
                **({"error": error} if error else {}),
            }
            async with write_lock:
                completed.append(record)
                if len(completed) % args.checkpoint_every == 0:
                    _write_report(args.out, _summarize(args, traces, completed, status="in_progress"))
                print(f"[{len(completed)}/{len(traces)}] trace={index} error={bool(error)}")

    await asyncio.gather(
        *(process(index, trace) for index, trace in enumerate(traces) if index not in completed_indices)
    )
    _write_report(args.out, _summarize(args, traces, completed, status="completed"))


def _summarize(
    args: argparse.Namespace,
    traces: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    valid = [record for record in completed if not record.get("error")]
    captured_comparisons: dict[str, dict[str, float]] = {}
    input_dependence: dict[str, dict[str, float]] = {}
    captured_rows: dict[str, list[dict[str, float]]] = {}
    input_rows: dict[str, list[dict[str, float]]] = {}
    for order in args.orders:
        captured_rows[order] = [
            _stability(record["captured_selected_node_ids"], record["replays"][order]["selected_node_ids"])
            for record in valid
        ]
        input_rows[order] = [_input_order_dependence(record["replays"][order]) for record in valid]
        captured_comparisons[order] = _mean_dict(captured_rows[order])
        input_dependence[order] = _mean_dict(input_rows[order])
    pairwise: dict[str, dict[str, float]] = {}
    pairwise_rows: dict[str, list[dict[str, float]]] = {}
    adjusted_order_effect: dict[str, dict[str, float]] = {}
    adjusted_order_effect_rows: dict[str, list[dict[str, float]]] = {}
    if "search" in args.orders:
        for order in args.orders:
            if order == "search":
                continue
            label = f"{order}_vs_search_replay"
            pairwise_rows[label] = [
                _stability(
                    record["replays"]["search"]["selected_node_ids"],
                    record["replays"][order]["selected_node_ids"],
                )
                for record in valid
            ]
            pairwise[label] = _mean_dict(pairwise_rows[label])
            adjusted_order_effect_rows[label] = [
                {
                    key: pairwise_row[key] - captured_row[key]
                    for key in pairwise_row
                }
                for pairwise_row, captured_row in zip(
                    pairwise_rows[label],
                    captured_rows["search"],
                    strict=True,
                )
            ]
            adjusted_order_effect[label] = _mean_dict(adjusted_order_effect_rows[label])
    first_selected_vs_random_rows: dict[str, list[dict[str, float]]] = {}
    first_selected_vs_random: dict[str, dict[str, float]] = {}
    for order in args.orders:
        first_selected_vs_random_rows[order] = [
            {
                "input_first_selected_minus_random": (
                    input_row["input_first_selected"]
                    - min(1.0, float(record["top_k"]) / float(record["pool_size"]))
                )
            }
            for record, input_row in zip(valid, input_rows[order], strict=True)
        ]
        first_selected_vs_random[order] = _mean_dict(first_selected_vs_random_rows[order])
    support_point_estimates: dict[str, dict[str, float]] = {
        "captured": _mean_dict([record["captured_support_metrics"] for record in valid]),
        **{
            order: _mean_dict([record["replays"][order]["support_metrics"] for record in valid])
            for order in args.orders
        },
    }
    support_vs_search_rows: dict[str, list[dict[str, float]]] = {}
    if "search" in args.orders:
        for order in args.orders:
            if order == "search":
                continue
            support_vs_search_rows[order] = [
                {
                    field: record["replays"][order]["support_metrics"][field]
                    - record["replays"]["search"]["support_metrics"][field]
                    for field in SUPPORT_FIELDS
                }
                for record in valid
            ]
    support_vs_search = {
        order: _mean_dict(rows) for order, rows in support_vs_search_rows.items()
    }
    categories = sorted({str(record["category"]) for record in valid})
    support_by_category: dict[str, Any] = {}
    support_vs_search_by_category_rows: dict[str, dict[str, list[dict[str, float]]]] = {}
    for category in categories:
        category_records = [record for record in valid if record["category"] == category]
        support_by_category[category] = {
            "queries": len(category_records),
            "point_estimates": {
                "captured": _mean_dict(
                    [record["captured_support_metrics"] for record in category_records]
                ),
                **{
                    order: _mean_dict(
                        [record["replays"][order]["support_metrics"] for record in category_records]
                    )
                    for order in args.orders
                },
            },
        }
        support_vs_search_by_category_rows[category] = {}
        if "search" in args.orders:
            for order in args.orders:
                if order == "search":
                    continue
                support_vs_search_by_category_rows[category][order] = [
                    {
                        field: record["replays"][order]["support_metrics"][field]
                        - record["replays"]["search"]["support_metrics"][field]
                        for field in SUPPORT_FIELDS
                    }
                    for record in category_records
                ]
    uncertainty: dict[str, Any] = {}
    if status == "completed":
        uncertainty = {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.seed,
            "captured_baseline_comparisons": {
                order: _uncertainty_dict(
                    rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 100 * order_index,
                )
                for order_index, (order, rows) in enumerate(captured_rows.items())
            },
            "pairwise_replay_comparisons": {
                label: _uncertainty_dict(
                    rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 1_000 + 100 * pair_index,
                )
                for pair_index, (label, rows) in enumerate(pairwise_rows.items())
            },
            "order_effect_beyond_same_order_variability": {
                label: _uncertainty_dict(
                    rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 1_500 + 100 * pair_index,
                )
                for pair_index, (label, rows) in enumerate(adjusted_order_effect_rows.items())
            },
            "input_order_dependence": {
                order: _uncertainty_dict(
                    rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 2_000 + 100 * order_index,
                )
                for order_index, (order, rows) in enumerate(input_rows.items())
            },
            "input_first_selected_vs_random": {
                order: _uncertainty_dict(
                    rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 2_500 + 100 * order_index,
                )
                for order_index, (order, rows) in enumerate(first_selected_vs_random_rows.items())
            },
            "support_vs_search_replay": {
                order: _uncertainty_dict(
                    rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 3_000 + 100 * order_index,
                )
                for order_index, (order, rows) in enumerate(support_vs_search_rows.items())
            },
            "support_vs_search_by_category": {
                category: {
                    order: _uncertainty_dict(
                        rows,
                        iterations=args.bootstrap_iterations,
                        seed=args.seed + 4_000 + 1_000 * category_index + 100 * order_index,
                    )
                    for order_index, (order, rows) in enumerate(order_rows.items())
                }
                for category_index, (category, order_rows) in enumerate(
                    support_vs_search_by_category_rows.items()
                )
            },
        }
    return {
        "status": status,
        "trace_path": str(args.trace),
        "query_path": str(args.queries),
        "benchmark_path": str(args.benchmark),
        "alignment": "completed benchmark query IDs plus generated Q-/Q+ view signatures",
        "trace_records": len(traces),
        "expected_trace_records": args.expected_traces,
        "completed_records": len(completed),
        "valid_records": len(valid),
        "failed_records": len(completed) - len(valid),
        "orders": args.orders,
        "shuffle_seed": args.seed,
        "concurrency": args.concurrency,
        "captured_baseline_comparisons": captured_comparisons,
        "pairwise_replay_comparisons": pairwise,
        "order_effect_beyond_same_order_variability": adjusted_order_effect,
        "input_order_dependence": input_dependence,
        "input_first_selected_vs_random": first_selected_vs_random,
        "support_point_estimates": support_point_estimates,
        "support_vs_search_replay": support_vs_search,
        "support_by_category": support_by_category,
        "uncertainty": uncertainty,
        "details": sorted(completed, key=lambda record: int(record["trace_index"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=("search", "reverse", "hash_shuffle"),
        default=("search", "reverse", "hash_shuffle"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--expected-traces", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

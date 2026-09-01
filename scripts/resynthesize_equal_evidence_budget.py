#!/usr/bin/env python3
"""Re-synthesize answers from frozen retrieval with one evidence budget.

This is a partial fairness control, not a full compute-budget comparison.  It
holds the evaluated query set, retained evidence count, context formatting,
answer prompt, generation model, and sampling settings fixed.  It does not
make the upstream retrieval algorithms or their query-time call counts equal.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.config import RAGConfig
from core.vllm_client import VLLMClient
from utils.metrics import calculate_answer_metrics, calculate_musique_support_metrics
from utils.prompts.shared import build_answer_prompt


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context_from_sources(sources: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[[{source.get('doc', 'Unknown')}, Page {source.get('page', 0)}, "
        f"Chunk {source.get('sent_id', 0)}]]\n{source.get('text', '')}"
        for source in sources
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_targets(values: list[str]) -> dict[str, Path]:
    targets: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --target {value!r}; expected NAME=PATH")
        if name.strip() in targets:
            raise ValueError(f"Duplicate target name: {name.strip()}")
        targets[name.strip()] = Path(raw_path.strip())
    return targets


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    valid = [row for row in rows if not row.get("error")]
    summary: dict[str, float | int] = {
        "queries": len(rows),
        "successful_queries": len(valid),
        "failed_queries": len(rows) - len(valid),
    }
    for field in (
        "answer_em",
        "answer_f1",
        "paragraph_support_precision",
        "paragraph_support_recall",
        "paragraph_support_f1",
        "synthesis_seconds",
        "original_retrieved_count",
        "retained_evidence_count",
    ):
        values = [float(row[field]) for row in valid if float(row.get(field, -1)) >= 0]
        if values:
            summary[f"avg_{field}"] = float(np.mean(values))
    return summary


async def _run(args: argparse.Namespace) -> None:
    queries = _load_json(args.queries)
    if not isinstance(queries, list):
        raise TypeError("--queries must contain a JSON list")
    if args.expected_queries is not None and len(queries) != args.expected_queries:
        raise ValueError(f"Expected {args.expected_queries} queries, found {len(queries)}")
    query_ids = [str(row["_id"]) for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs in --queries")

    targets = _parse_targets(args.target)
    output: dict[str, Any]
    if args.out.is_file():
        output = _load_json(args.out)
        if not output.get("synthesis_timer_excludes_local_semaphore_wait"):
            output["synthesis_timing_eligible"] = False
            output["synthesis_timing_note"] = (
                "Existing rows predate queue-excluding timing; synthesis_seconds must not be reported."
            )
    else:
        output = {
            "scope": (
                "complete_split_frozen_retrieval_partial_fairness_control"
                if args.expected_queries is not None and len(queries) == args.expected_queries
                else "subset_frozen_retrieval_partial_fairness_control"
            ),
            "limitations": [
                "Upstream retrieval algorithms and query-time call counts are not equalized.",
                "Only the retained evidence count and answer-synthesis conditions are held fixed.",
                "Frozen retrieval artifacts were produced by separate benchmark runs.",
            ],
            "queries_path": str(args.queries),
            "queries_sha256": _sha256(args.queries),
            "query_count": len(queries),
            "evidence_top_k": args.top_k,
            "generation_model": RAGConfig.DEFAULT_MODEL,
            "generation_temperature": 0.0,
            "generation_seed": RAGConfig.LLM_SEED,
            "synthesis_max_output_tokens": RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS,
            "synthesis_timer_excludes_local_semaphore_wait": True,
            "synthesis_timing_eligible": True,
            "targets": {},
        }
    if args.exclude_timing:
        output["synthesis_timing_eligible"] = False
        output["synthesis_timing_note"] = args.timing_ineligible_reason

    client = VLLMClient()
    semaphore = asyncio.Semaphore(args.concurrency)
    for strategy, artifact_path in targets.items():
        artifact = _load_json(artifact_path)
        artifact_rows = {str(row["query_id"]): row for row in artifact.get("details", [])}
        missing = [query_id for query_id in query_ids if query_id not in artifact_rows]
        if missing:
            raise ValueError(f"{strategy} artifact is missing {len(missing)} evaluated query IDs")

        prior = output.get("targets", {}).get(strategy, {})
        completed = {
            str(row["query_id"]): row
            for row in prior.get("details", [])
            if not row.get("error")
        }

        async def evaluate(
            query_row: dict[str, Any],
            artifact_rows: dict[str, dict[str, Any]] = artifact_rows,
        ) -> dict[str, Any]:
            query_id = str(query_row["_id"])
            frozen = artifact_rows[query_id]
            original_sources = list(frozen.get("retrieved_sources") or [])
            retained_sources = original_sources[: args.top_k]
            context = _context_from_sources(retained_sources)
            prompt = build_answer_prompt(context, str(query_row["query"]))
            started: float | None = None
            try:
                async with semaphore:
                    started = time.perf_counter()
                    answer = await client.generate_response(
                        [{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=RAGConfig.SYNTHESIS_MAX_OUTPUT_TOKENS,
                    )
                elapsed = time.perf_counter() - started
                answer_text = str(answer or "").strip()
                if not answer_text:
                    raise ValueError("Answer synthesis returned an empty response")
                return {
                    "query_id": query_id,
                    "query": query_row["query"],
                    "ground_truth": query_row.get("ground_truth", ""),
                    "category": query_row.get("category", ""),
                    "answer": answer_text,
                    "original_retrieved_count": len(original_sources),
                    "retained_evidence_count": len(retained_sources),
                    "retained_sources": retained_sources,
                    "synthesis_seconds": elapsed,
                    "error": "",
                    **calculate_answer_metrics(
                        answer_text,
                        str(query_row.get("ground_truth", "")),
                        answer_aliases=query_row.get("answer_aliases") or [],
                        question_type=str(query_row.get("question_type", "")),
                    ),
                    **calculate_musique_support_metrics(
                        retained_sources,
                        query_row.get("evidence_paragraph_ids") or [],
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - checkpoint failures for auditable reruns
                return {
                    "query_id": query_id,
                    "query": query_row["query"],
                    "ground_truth": query_row.get("ground_truth", ""),
                    "category": query_row.get("category", ""),
                    "original_retrieved_count": len(original_sources),
                    "retained_evidence_count": len(retained_sources),
                    "synthesis_seconds": time.perf_counter() - started if started is not None else -1.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        pending = [row for row in queries if str(row["_id"]) not in completed]
        tasks = [asyncio.create_task(evaluate(row)) for row in pending]
        for count, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            completed[result["query_id"]] = result
            if count % args.checkpoint_every == 0 or count == len(tasks):
                ordered = [completed[query_id] for query_id in query_ids if query_id in completed]
                output.setdefault("targets", {})[strategy] = {
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": _sha256(artifact_path),
                    "artifact_strategy": artifact.get("strategy"),
                    "summary": _summarize(ordered),
                    "details": ordered,
                }
                _atomic_write(args.out, output)
                print(f"[{strategy}] {len(ordered)}/{len(queries)}", flush=True)

        ordered = [completed[query_id] for query_id in query_ids]
        output.setdefault("targets", {})[strategy] = {
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "artifact_strategy": artifact.get("strategy"),
            "summary": _summarize(ordered),
            "details": ordered,
        }
        _atomic_write(args.out, output)

    print(f"saved {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("data/musique_queries.json"))
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="NAME=benchmark_artifact.json; repeat for each retrieval method",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--exclude-timing", action="store_true")
    parser.add_argument(
        "--timing-ineligible-reason",
        default="The run did not use an isolated load window; synthesis latency is excluded.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("--top-k, --concurrency, and --checkpoint-every must be positive")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

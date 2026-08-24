import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from core.config import RAGConfig
from core.vllm_client import get_llm_client
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG
from utils.io import _safe_float, _write_json
from utils.metrics import evaluate_multihoprag_response
from utils.reporting import _write_model_report_artifacts

logger = logging.getLogger("Prehop")


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_LABEL_RE = re.compile(r"(?is)(?:final\s+answer|@@ANSWER|answer)\s*:?\s*(.+?)(?:\n\n|\Z)")


def _read_json_file(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_final_answer(answer_text: str) -> str:
    """Extract the final answer from a model response that may contain
    step-by-step reasoning. Order: \\boxed{...} > 'Final Answer:' marker >
    last 300 chars. Avoids substring-matching the reasoning body for
    abstain detection.
    """
    if not answer_text:
        return ""
    boxed = _BOXED_RE.findall(answer_text)
    if boxed:
        return boxed[-1].strip()
    matches = _FINAL_LABEL_RE.findall(answer_text)
    if matches:
        return matches[-1].strip()[:400]
    return answer_text[-300:].strip()


def _build_benchmark_query(query: str, item: dict[str, Any]) -> str:
    """Return the user-facing query as-is.

    The previous implementation appended `[Benchmark Output Format]` blocks
    that forced verbose CoT inside `\\boxed{}`. That suffix leaked into
    retrieval embeddings as noise and collided with the citation-first
    answer format. The judge prompt extracts `\\boxed{}` / `Final Answer:`
    internally, so the scaffolding adds no signal upstream.
    """
    _ = item  # kept for signature stability; type detection no longer alters the query.
    return query


def _extract_stage_timing(trace: Any) -> dict[str, float]:
    """Pull retrieve_ms/traversal_ms/synthesis_ms out of a prehop-style
    `interaction_trace` (see models/prehop/graphrag.py's `run_workflow`),
    if present, so they land as top-level numeric fields on `result_item`
    and get auto-averaged into `avg_retrieve_ms`/`avg_traversal_ms`/
    `avg_synthesis_ms` by `_recompute_aggregates` — the paper's headline
    latency-breakdown claim (no per-hop LLM reasoning) needs this split out,
    not just the single aggregate `latency`.

    Other strategies' traces don't carry these keys, so this returns {} for
    them — deliberately not defaulting to 0.0, which would misreport "zero
    latency" instead of "not measured" once averaged.
    """
    if not isinstance(trace, list):
        return {}
    timing: dict[str, float] = {}
    for step in trace:
        if not isinstance(step, dict):
            continue
        if step.get("step") == "retrieve":
            if "retrieve_ms" in step:
                timing["retrieve_ms"] = float(step.get("retrieve_ms") or 0.0)
            if "traversal_ms" in step:
                timing["traversal_ms"] = float(step.get("traversal_ms") or 0.0)
        elif step.get("step") == "synthesis":
            if "synthesis_ms" in step:
                timing["synthesis_ms"] = float(step.get("synthesis_ms") or 0.0)
    return timing


def _apply_judge_label(result_item: dict[str, Any]) -> None:
    """Derive answer_attempted / answer_label from the LLM judge score."""
    from utils.abstain import answer_label, is_abstain

    answer_text = str(result_item.get("answer", "") or "")
    has_error = bool(result_item.get("error"))
    # Detect abstain on the EXTRACTED final answer (\\boxed{} / 'Final Answer:'),
    # not the full CoT body which often uses 'insufficient evidence' mid-reason.
    final_answer = _extract_final_answer(answer_text).lower()
    abstained = is_abstain(final_answer)
    judge_score = _safe_float(result_item.get("llm_judge_score", 0.0), 0.0)
    result_item["final_answer_extracted"] = final_answer[:300]

    # Unjudged row (score < 0, e.g. batch not yet resolved or judge failed):
    # don't fabricate a label/attempt — leave it out of the 3-way tally.
    if judge_score < 0.0:
        result_item["answer_attempted"] = -1.0
        result_item["answer_label"] = "Unjudged"
        return

    # Judge override: a score >= 0.5 means a usable answer regardless of phrasing.
    if has_error:
        answer_attempted = 0.0
    elif judge_score >= 0.5:
        answer_attempted = 1.0
    else:
        answer_attempted = 0.0 if abstained else 1.0
    result_item["answer_attempted"] = answer_attempted
    if not isinstance(result_item.get("hallucination"), (int, float)):
        result_item["hallucination"] = 1.0 if (answer_attempted > 0.0 and judge_score < 1.0) else 0.0
    result_item["answer_label"] = answer_label(judge_score, final_answer)


def _recompute_aggregates(s: dict[str, Any]) -> None:
    """Recompute avg_<metric>, category_summaries and the 3-way label counts
    from ``s['details']`` in place. Shared by the live benchmark pass and the
    async batch reconcile step so both yield identical aggregates. Averages skip
    the UNJUDGED sentinel (-1); every real metric is in [0, 1] (or latency >= 0).
    """
    rows = s.get("details") or []

    def _avg(subset: list[dict], key: str) -> float:
        vals = [
            r[key]
            for r in subset
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool) and r[key] >= 0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in numeric_keys:
        s[f"avg_{key}"] = _avg(rows, key)

    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r.get("category", "Uncategorized"), []).append(r)
    cat_summaries: dict[str, Any] = {}
    for cat, cat_list in cats.items():
        cat_sum: dict[str, Any] = {"count": len(cat_list)}
        for key in numeric_keys:
            cat_sum[f"avg_{key}"] = _avg(cat_list, key)
        cat_summaries[cat] = cat_sum
    s["category_summaries"] = cat_summaries

    label_counts = {"Correct Answer": 0, "Incorrect Answer": 0, "Refusal": 0}
    for r in rows:
        label = r.get("answer_label")
        if label in label_counts:
            label_counts[label] += 1
    total = sum(label_counts.values()) or 1  # judged rows only ("Unjudged" excluded)
    s["correct_count"] = label_counts["Correct Answer"]
    s["incorrect_count"] = label_counts["Incorrect Answer"]
    s["refusal_count"] = label_counts["Refusal"]
    s["correct_rate"] = label_counts["Correct Answer"] / total
    s["incorrect_rate"] = label_counts["Incorrect Answer"] / total
    s["refusal_rate"] = label_counts["Refusal"] / total


def _unjudged_count(rows: list[dict[str, Any]], key: str = "llm_judge_score") -> int:
    return sum(
        1
        for row in rows
        if not isinstance(row.get(key), (int, float))
        or isinstance(row.get(key), bool)
        or float(row[key]) < 0
    )


def _unjudged_groundedness_count(rows: list[dict[str, Any]]) -> int:
    """Count substantive rows without a context-groundedness judgement."""
    from utils.abstain import is_abstain

    count = 0
    for row in rows:
        if row.get("error"):
            continue
        final_answer = _extract_final_answer(str(row.get("answer", "") or ""))
        if is_abstain(final_answer):
            continue
        value = row.get("groundedness")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
            count += 1
    return count


def _update_summary_status(summary: dict[str, Any]) -> None:
    rows = summary.get("details") or []
    if len(rows) < int(summary.get("total_queries", len(rows)) or 0):
        summary["status"] = "in_progress"
    elif any(
        (row.get("_deferred_judge") or row.get("judge_custom_id")) and _safe_float(row.get("llm_judge_score"), -1.0) < 0
        for row in rows
    ):
        summary["status"] = "pending_judge"
    elif any(row.get("error") for row in rows):
        summary["status"] = "completed_with_errors"
    elif _unjudged_count(rows) or _unjudged_count(rows, "hallucination") or _unjudged_groundedness_count(rows):
        summary["status"] = "completed_with_unjudged"
    else:
        summary["status"] = "completed"


def _assert_benchmark_complete(summary: dict[str, Any], result_file: Path) -> None:
    rows = summary.get("details") or []
    runtime_errors = sum(1 for row in rows if row.get("error"))
    unjudged = _unjudged_count(rows)
    unjudged_hallucination = _unjudged_count(rows, "hallucination")
    unjudged_groundedness = _unjudged_groundedness_count(rows)
    failures = []
    if runtime_errors:
        failures.append(f"{runtime_errors} runtime error(s)")
    if unjudged:
        failures.append(f"{unjudged} unjudged row(s)")
    if unjudged_hallucination:
        failures.append(f"{unjudged_hallucination} row(s) without hallucination judgement")
    if unjudged_groundedness:
        failures.append(f"{unjudged_groundedness} substantive row(s) without groundedness judgement")
    if failures:
        raise RuntimeError(f"Benchmark incomplete ({', '.join(failures)}); results saved to {result_file}")


def _validate_benchmark_data(benchmark_data: Any, source: str) -> list[dict[str, Any]]:
    if not isinstance(benchmark_data, list):
        raise TypeError(f"Benchmark file must contain a JSON list: {source}")
    if not benchmark_data:
        raise ValueError(f"Benchmark file contains no queries: {source}")
    supported = {"multihoprag", "2wikimultihopqa", "musique"}
    validated: list[dict[str, Any]] = []
    dataset_markers: set[str] = set()
    for idx, item in enumerate(benchmark_data):
        if not isinstance(item, dict):
            raise TypeError(f"Benchmark row {idx} must be an object, got {type(item).__name__}")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Benchmark row {idx} has no non-empty 'query'")
        marker = item.get("dataset")
        if not isinstance(marker, str) or marker.strip().lower() not in supported:
            raise ValueError(
                f"Benchmark row {idx} has unsupported dataset marker {marker!r}; expected one of {sorted(supported)}"
            )
        dataset_markers.add(marker.strip().lower())
        validated.append(item)
    if len(dataset_markers) != 1:
        raise ValueError(f"Benchmark file mixes dataset markers: {sorted(dataset_markers)}")
    return validated


def _slim_details(details: list | None) -> list:
    """Strip interaction traces and private fields from the main result JSON."""
    return [
        {k: v for k, v in d.items() if k != "interaction_trace" and not k.startswith("_")} if isinstance(d, dict) else d
        for d in (details or [])
    ]


def _write_slim_main(s: dict[str, Any], result_file: Path) -> None:
    _write_json(result_file, {**s, "details": _slim_details(s.get("details"))})


def _result_json_in_seed_dir(seed_dir: Path) -> Path:
    candidates = [
        path
        for path in seed_dir.glob("*.json")
        if not path.name.endswith(
            (
                ".summary.json",
                ".stage_diagnostics.json",
                ".pending_judge.json",
            )
        )
        and path.name != "seeds_aggregate.json"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one benchmark result in {seed_dir}, found {len(candidates)}")
    return candidates[0]


def _refresh_seed_aggregate(seeds_root: Path) -> None:
    """Rebuild a multi-seed aggregate after every seed's judge has resolved."""
    aggregate_file = seeds_root / "seeds_aggregate.json"
    if not aggregate_file.exists():
        return

    prior = _read_json_file(aggregate_file)
    seeds = prior.get("seeds") or []
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        result_file = _result_json_in_seed_dir(seeds_root / f"seed_{int(seed)}")
        summary = _read_json_file(result_file)
        if summary.get("status") != "completed":
            prior["status"] = "pending_judge" if summary.get("status") == "pending_judge" else "completed_with_errors"
            prior["aggregate"] = {}
            _write_json(aggregate_file, prior)
            return
        _assert_benchmark_complete(summary, result_file)
        summaries.append(summary)

    prior.update(
        {
            "n_seeds": len(summaries),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed",
            "aggregate": _aggregate_seed_summaries(summaries),
        }
    )
    _write_json(aggregate_file, prior)


async def reconcile_pending_judges(run_dir: Path) -> int:
    """Resolve and atomically apply all pending OpenAI judge batches.

    All expected custom IDs must be present before a result is changed. A
    failed or partial batch therefore leaves its manifest and unjudged result
    intact for inspection/retry instead of publishing a partial paper metric.
    """
    from utils.batch_judge import resolve_batches
    from utils.metrics import _resolve_judge_fields

    run_dir = Path(run_dir).resolve()
    manifests = sorted(run_dir.rglob("*.pending_judge.json"))
    if not manifests:
        return 0
    if not RAGConfig.OPENAI_API_KEY:
        raise RuntimeError(f"{len(manifests)} judge batch(es) are pending but OPENAI_API_KEY is missing")

    loaded: list[tuple[Path, dict[str, Any], Path]] = []
    errors: list[str] = []
    for manifest_path in manifests:
        try:
            manifest = _read_json_file(manifest_path)
            batch_id = str(manifest.get("batch_id", "")).strip()
            result_file = Path(str(manifest.get("result_file", ""))).resolve()
            if not batch_id:
                raise ValueError("batch_id is empty")
            if not result_file.is_relative_to(run_dir):
                raise ValueError(f"result_file escapes run directory: {result_file}")
            if not result_file.is_file():
                raise FileNotFoundError(f"result file not found: {result_file}")
            loaded.append((manifest_path, manifest, result_file))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest_path}: {exc}")

    if errors:
        raise RuntimeError("Invalid judge manifest(s): " + "; ".join(errors))

    batch_ids = [str(manifest["batch_id"]) for _, manifest, _ in loaded]
    logger.info("Resolving %d OpenAI judge batch(es) in parallel", len(batch_ids))
    resolved = await asyncio.to_thread(
        resolve_batches,
        RAGConfig.OPENAI_API_KEY,
        batch_ids,
        RAGConfig.JUDGE_BATCH_POLL_SECONDS,
    )

    patched_files = 0
    seed_roots: set[Path] = set()
    unresolved: list[str] = []
    for manifest_path, manifest, result_file in loaded:
        batch_id = str(manifest["batch_id"])
        payloads = resolved.get(batch_id) or {}
        summary = _read_json_file(result_file)
        rows = summary.get("details") or []
        expected_ids = {
            str(row["judge_custom_id"])
            for row in rows
            if row.get("judge_custom_id") is not None and _safe_float(row.get("llm_judge_score"), -1.0) < 0
        }
        submitted = int(manifest.get("submitted", 0) or 0)
        if submitted != len(expected_ids):
            unresolved.append(f"{batch_id}: manifest submitted={submitted}, deferred rows={len(expected_ids)}")
            continue
        missing = sorted(custom_id for custom_id in expected_ids if payloads.get(custom_id) is None)
        if not expected_ids:
            unresolved.append(f"{batch_id}: result has no deferred judge rows")
            continue
        if missing:
            unresolved.append(f"{batch_id}: missing {len(missing)}/{len(expected_ids)} result payload(s)")
            continue

        judge_model = str(manifest.get("judge_model", ""))
        invalid_payloads = []
        for row in rows:
            custom_id = row.get("judge_custom_id")
            if custom_id is None or str(custom_id) not in expected_ids:
                continue
            resolved_fields = _resolve_judge_fields(
                payloads[str(custom_id)],
                # Use a substantive sentinel so abstention rules cannot mask
                # missing explicit context-axis fields during validation.
                "substantive answer",
                judge_model,
            )
            if (
                resolved_fields["llm_judge_score"] < 0
                or resolved_fields["groundedness"] < 0
                or resolved_fields["hallucination"] < 0
            ):
                invalid_payloads.append(str(custom_id))
        if invalid_payloads:
            unresolved.append(
                f"{batch_id}: {len(invalid_payloads)}/{len(expected_ids)} payload(s) "
                "missing valid score, groundedness, or hallucination"
            )
            continue

        for row in rows:
            custom_id = row.get("judge_custom_id")
            if custom_id is None or str(custom_id) not in expected_ids:
                continue
            row.update(_resolve_judge_fields(payloads[str(custom_id)], row.get("answer", ""), judge_model))
            row.pop("_deferred_judge", None)
            _apply_judge_label(row)

        _recompute_aggregates(summary)
        _update_summary_status(summary)
        _write_slim_main(summary, result_file)
        _write_model_report_artifacts(summary, result_file, preserve_trace_artifacts=True)
        manifest_path.unlink()
        patched_files += 1
        try:
            _assert_benchmark_complete(summary, result_file)
        except RuntimeError as exc:
            unresolved.append(str(exc))
        if result_file.parent.name.startswith("seed_"):
            seed_roots.add(result_file.parent.parent)

    for seeds_root in seed_roots:
        _refresh_seed_aggregate(seeds_root)
    if unresolved:
        raise RuntimeError("Judge reconciliation incomplete: " + "; ".join(unresolved))
    return patched_files


async def run_benchmark(
    queries_file: str,
    strategy: str,
    model_id: str,
    is_batch: bool = False,
    corpus_tag: str = "default",
    output_dir: Path | None = None,
    limit: int | None = None,
    seed: int | None = None,
):
    """Run benchmark once. When `seed` is provided, RAGConfig.LLM_SEED is set
    so all external chat.completions calls in this run use that seed, and
    the output directory gets a `seed_<S>` subdir to avoid clobbering other
    seeds' results. Multi-seed orchestration lives in run_benchmark_multi_seed.
    """
    if not os.path.isfile(queries_file):
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    if seed is not None:
        RAGConfig.LLM_SEED = int(seed)

    try:
        if strategy == "prehop":
            engine = GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
        elif strategy == "naive":
            engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
        elif strategy == "hoprag":
            from models.hoprag.hoprag_adapter import HopRAGAdapter

            engine = HopRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
        elif strategy == "ms_graphrag":
            from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter

            engine = MSGraphRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        vllm = get_llm_client(model_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize engine for {strategy}: {exc}") from exc

    benchmark_data = _validate_benchmark_data(
        await asyncio.to_thread(_read_json_file, queries_file),
        queries_file,
    )

    if limit is not None:
        benchmark_data = benchmark_data[: max(0, int(limit))]
        logger.info("--limit %d: evaluating %d queries", limit, len(benchmark_data))
    if not benchmark_data:
        raise ValueError("Benchmark query selection is empty after filtering/limit")

    # Dataset dispatch via the per-query `dataset` marker. MultiHop-RAG,
    # 2WikiMultiHopQA, and MuSiQue share one query schema, but their evidence
    # units differ. The evaluator keeps deterministic answer EM/F1 primary,
    # uses title-level evidence P/R/F1 across datasets, and only runs
    # sentence/fact ranking metrics where the gold unit is aligned.
    _MULTIHOP_DATASET_NAMES = {
        "multihoprag": "MultiHop-RAG",
        "2wikimultihopqa": "2WikiMultiHopQA",
        "musique": "MuSiQue",
    }
    dataset_marker = (benchmark_data[0].get("dataset", "") if benchmark_data else "").strip().lower()
    if dataset_marker not in _MULTIHOP_DATASET_NAMES:
        raise ValueError(
            f"Unrecognized dataset marker {dataset_marker!r} — queries must carry "
            f"a 'dataset' field set to one of {sorted(_MULTIHOP_DATASET_NAMES)}."
        )
    dataset_name = _MULTIHOP_DATASET_NAMES[dataset_marker]
    results = []
    category_results = {}

    logger.info(
        "Starting benchmark [%s] on %s | Queries: %d",
        strategy,
        dataset_name,
        len(benchmark_data),
    )

    if output_dir:
        results_dir = output_dir
    else:
        env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
        start_timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("data/results") / start_timestamp

    results_dir.mkdir(parents=True, exist_ok=True)
    model_results_dir = results_dir / strategy
    model_results_dir.mkdir(parents=True, exist_ok=True)
    ablation_results_dir = model_results_dir / corpus_tag
    ablation_results_dir.mkdir(parents=True, exist_ok=True)

    output_results_dir = ablation_results_dir
    if seed is not None:
        output_results_dir = output_results_dir / f"seed_{int(seed)}"
    output_results_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_results_dir / f"{strategy}_{corpus_tag}.json"
    summary: dict[str, Any] = {}

    batch_collector = None
    if RAGConfig.JUDGE_BATCH:
        eval_model = str(RAGConfig.EVAL_MODEL or "").strip()
        if not eval_model:
            raise RuntimeError(
                "Batch judge is enabled by default, but EVAL_MODEL is missing. "
                "Configure an OpenAI Batch-compatible judge model or explicitly set "
                "RAG_JUDGE_BATCH=false for debugging."
            )
        if not RAGConfig.OPENAI_API_KEY:
            raise RuntimeError(
                "Batch judge is enabled by default, but OPENAI_API_KEY is missing. "
                "Set the key or explicitly set RAG_JUDGE_BATCH=false for a synchronous debug run."
            )
        from utils.batch_judge import OpenAIBatchJudge

        batch_collector = OpenAIBatchJudge(
            eval_model,
            RAGConfig.OPENAI_API_KEY,
            poll_seconds=RAGConfig.JUDGE_BATCH_POLL_SECONDS,
        )
        logger.info("Judge: OpenAI Batch API (default), collecting %d requests", len(benchmark_data))

    benchmark_concurrency = max(1, int(os.environ.get("RAG_BENCHMARK_CONCURRENCY", "4")))
    query_sem = asyncio.Semaphore(benchmark_concurrency)
    write_lock = asyncio.Lock()
    total_queries = len(benchmark_data)
    if benchmark_concurrency > 1:
        logger.info("Benchmark concurrency: %d queries in flight", benchmark_concurrency)

    def _recompute_and_persist() -> dict[str, Any]:
        """Rebuild the summary from `results` and write the result file +
        report artifacts after each completed query."""
        s: dict[str, Any] = {
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "dataset": dataset_name,
            "queries_count": len(results),
            "total_queries": total_queries,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "in_progress",
            "models": {
                "default": RAGConfig.DEFAULT_MODEL,
                "embedding": RAGConfig.EMBEDDING_MODEL,
                "eval": RAGConfig.EVAL_MODEL,
            },
            "ablation": {
                "q_minus": RAGConfig.ABLATION_Q_MINUS,
                "q_plus": RAGConfig.ABLATION_Q_PLUS,
                "chunk_sentences": RAGConfig.CHUNK_SENTENCES,
                "hop_link_limit": RAGConfig.HOP_LINK_LIMIT,
                "hop_candidate_limit": RAGConfig.HOP_CANDIDATE_LIMIT,
                "hop_ann_pool": RAGConfig.HOP_ANN_POOL,
                "hop_same_need_weight": RAGConfig.HOP_SAME_NEED_WEIGHT,
                "hypo_channel_variant": RAGConfig.HYPO_CHANNEL_VARIANT,
            },
        }
        s["details"] = results
        _recompute_aggregates(s)
        _update_summary_status(s)
        # Report artifacts first (writes full traces), then a slim main JSON.
        try:
            _write_model_report_artifacts(s, result_file)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to write report artifacts for %s: %s", result_file, exc)
        _write_slim_main(s, result_file)
        return s

    async def _process_query(idx: int, item: dict[str, Any]):
        nonlocal summary
        async with query_sem:
            started = time.time()
            stage_timing: dict[str, float] = {}
            original_query = str(item.get("query", ""))
            query = original_query
            ground_truth = item.get("ground_truth", "")
            category = item.get("category", "Uncategorized")
            try:
                query = _build_benchmark_query(original_query, item)
                response, retrieved_sources, trace = await engine.run_workflow(query, [])
                latency = time.time() - started
                stage_timing = _extract_stage_timing(trace)

                metrics = await evaluate_multihoprag_response(
                    query=original_query,
                    response=response,
                    ground_truth=ground_truth,
                    retrieved_sources=retrieved_sources,
                    evidence_facts=item.get("evidence_facts", []),
                    evidence_docs=item.get("evidence_docs", []),
                    question_type=item.get("question_type", ""),
                    dataset=dataset_marker,
                    answer_aliases=item.get("answer_aliases", []),
                    vllm_client=vllm,
                    batch_collector=batch_collector,
                    custom_id=str(idx),
                )
                expected_sources = {
                    "docs": item.get("evidence_docs", []),
                    "facts": item.get("evidence_facts", []),
                }
                result_item = {
                    "query": original_query,
                    "category": category,
                    "answer": response,
                    "ground_truth": ground_truth,
                    "expected_sources": expected_sources,
                    "retrieved_sources": retrieved_sources,
                    "interaction_trace": trace,
                    "latency": latency,
                    **stage_timing,
                    **metrics,
                }
            except Exception as exc:  # noqa: BLE001 - isolate and persist each query failure
                logger.error("Error processing query '%s': %s", original_query, exc)
                import traceback

                logger.error(traceback.format_exc())
                latency = time.time() - started
                error_text = f"{type(exc).__name__}: {exc}"

                metrics = {
                    "llm_judge_score": 0.0,
                    "llm_judge_reason": "runtime_error",
                    "groundedness": -1.0,
                    "groundedness_source": "runtime_error",
                    "hallucination": 0.0,
                    "hallucination_reason": "runtime_error",
                    "hallucination_source": "runtime_error",
                    "hallucination_model": str(RAGConfig.EVAL_MODEL or ""),
                    "answer_em": -1.0,
                    "answer_f1": -1.0,
                    "answer_precision": -1.0,
                    "answer_recall": -1.0,
                    "null_refusal": -1.0,
                    "doc_match": 0.0,
                    # Keep the same numeric keys the success path emits so the
                    # summary auto-averaging stays consistent across queries.
                    "mrr@10": 0.0,
                    "map@10": 0.0,
                    "hits@4": 0.0,
                    "hits@10": 0.0,
                    "evidence_doc_recall": 0.0,
                    "evidence_doc_precision": 0.0,
                    "evidence_doc_f1": 0.0,
                }
                expected_sources = {
                    "docs": item.get("evidence_docs", []),
                    "facts": item.get("evidence_facts", []),
                }
                result_item = {
                    "query": original_query,
                    "category": category,
                    "answer": f"@@ANSWER: ERROR - {error_text}",
                    "ground_truth": ground_truth,
                    "expected_sources": expected_sources,
                    "retrieved_sources": [],
                    "interaction_trace": [{"step": "error", "output": error_text}],
                    "latency": latency,
                    "error": error_text,
                    **stage_timing,
                    **metrics,
                }

            if query != original_query:
                result_item["benchmark_query"] = query

            # Every dataset is LLM-judge scored, so the 3-way label /
            # answer_attempted / abstain post-processing is shared.
            _apply_judge_label(result_item)

            async with write_lock:
                results.append(result_item)
                if category not in category_results:
                    category_results[category] = []
                category_results[category].append(result_item)

                error_suffix = " [ERROR]" if result_item.get("error") else ""
                print(
                    f"[{strategy}] ({len(results)}/{total_queries}) [{category}]{error_suffix} "
                    f"Judge: {metrics['llm_judge_score']:.1f} | Hallu: {result_item.get('hallucination', 0.0):.0f} "
                    f"| DocMatch: {metrics['doc_match']:.0f} | Latency: {latency:.1f}s"
                )

                summary = _recompute_and_persist()

    await asyncio.gather(
        *[_process_query(i, it) for i, it in enumerate(benchmark_data)],
        return_exceptions=False,
    )

    if not results:
        return None

    if batch_collector is not None and batch_collector.count > 0:
        pending_path = output_results_dir / f"{strategy}_{corpus_tag}.pending_judge.json"

        def _persist_manifest(batch_id: str) -> None:
            _write_json(
                pending_path,
                {
                    "batch_id": batch_id,
                    "judge_model": str(RAGConfig.EVAL_MODEL or ""),
                    "result_file": str(result_file.resolve()),
                    "strategy": strategy,
                    "corpus_tag": corpus_tag,
                    "dataset": dataset_name,
                    "submitted": batch_collector.count,
                    "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

        batch_id = await batch_collector.submit(on_submitted=_persist_manifest)
        if not batch_id:
            raise RuntimeError("Judge batch contained requests but submission returned no batch ID")
        summary = _recompute_and_persist()
        logger.info("Judge batch %s submitted; pending manifest: %s", batch_id, pending_path)

    try:
        _write_model_report_artifacts(summary, result_file)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to write final report artifacts for %s: %s", result_file, exc)
    _write_slim_main(summary, result_file)

    print(f"\n{'=' * 50}")
    completion_label = "Judge Batch Submitted" if summary.get("status") == "pending_judge" else "Benchmark Complete"
    print(f"[{strategy.upper()}] {completion_label} - {dataset_name}")
    print(f"{'=' * 50}")
    for key, value in summary.items():
        if key.startswith("avg_"):
            print(f"  Overall {key}: {value:.4f}")

    print("\nCategory Breakdown:")
    for cat, cat_sum in summary["category_summaries"].items():
        print(f"  - {cat} (n={cat_sum['count']}):")
        for key, value in cat_sum.items():
            if key.startswith("avg_"):
                print(f"    {key}: {value:.4f}")

    print(f"\n  Final results saved to: {result_file}")
    print(f"{'=' * 50}\n")
    if summary.get("status") != "pending_judge":
        _assert_benchmark_complete(summary, result_file)
    return summary


def _parse_seeds_env(raw: str) -> list[int]:
    """Parse comma/space-separated seed list. Empty -> []."""
    out: list[int] = []
    for token in re.split(r"[,\s]+", (raw or "").strip()):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            logger.warning("Ignoring non-integer seed token: %r", token)
    return out


def _aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std / 95% CI per metric across N seeds.

    CI = mean ± 1.96 * std / sqrt(N)  (normal-approx; fine for N>=3 + smooth metrics).
    Per-category aggregates are computed only over keys that appear in every seed.
    """
    import math

    def _agg_keys(values: list[float]) -> dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
        mean = sum(values) / n
        if n == 1:
            return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
        margin = 1.96 * std / math.sqrt(n)
        return {"mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin, "n": n}

    if not summaries:
        return {}

    avg_keys = sorted({k for s in summaries for k in s if k.startswith("avg_")})
    overall: dict[str, Any] = {}
    for key in avg_keys:
        vals = [_safe_float(s.get(key, 0.0), 0.0) for s in summaries if key in s]
        overall[key] = _agg_keys(vals)

    # Category-level aggregation: only categories that all seeds reported
    common_cats: set[str] | None = None
    for s in summaries:
        cats = set((s.get("category_summaries") or {}).keys())
        common_cats = cats if common_cats is None else (common_cats & cats)
    common_cats = common_cats or set()

    categories: dict[str, dict[str, Any]] = {}
    for cat in sorted(common_cats):
        cat_keys = sorted(
            {k for s in summaries for k in (s.get("category_summaries", {}).get(cat, {}) or {}) if k.startswith("avg_")}
        )
        per_cat = {}
        for key in cat_keys:
            vals = [
                _safe_float(s["category_summaries"][cat].get(key, 0.0), 0.0)
                for s in summaries
                if cat in (s.get("category_summaries") or {})
            ]
            per_cat[key] = _agg_keys(vals)
        per_cat["count"] = int(summaries[0].get("category_summaries", {}).get(cat, {}).get("count", 0))
        categories[cat] = per_cat

    return {"overall": overall, "categories": categories}


async def run_benchmark_multi_seed(
    queries_file: str,
    strategy: str,
    model_id: str,
    seeds: list[int] | None = None,
    is_batch: bool = False,
    corpus_tag: str = "default",
    output_dir: Path | None = None,
    limit: int | None = None,
):
    """Run the benchmark once per seed, then write a `seeds_aggregate.json`
    with mean/std/95%-CI per metric. When seeds is empty/None, behaves
    identically to a single run_benchmark() call.
    """
    if seeds is None:
        seeds = _parse_seeds_env(os.environ.get("RAG_BENCHMARK_SEEDS", ""))

    if not seeds:
        return await run_benchmark(
            queries_file=queries_file,
            strategy=strategy,
            model_id=model_id,
            is_batch=is_batch,
            corpus_tag=corpus_tag,
            output_dir=output_dir,
            limit=limit,
        )

    # Pin a single timestamp across all seeds so they share one result root.
    if not os.environ.get("RAG_BENCHMARK_TIMESTAMP"):
        os.environ["RAG_BENCHMARK_TIMESTAMP"] = time.strftime("%Y%m%d_%H%M%S")

    summaries: list[dict[str, Any]] = []
    for s in seeds:
        logger.info("=== Seed %d (%d/%d) ===", s, len(summaries) + 1, len(seeds))
        summary = await run_benchmark(
            queries_file=queries_file,
            strategy=strategy,
            model_id=model_id,
            is_batch=is_batch,
            corpus_tag=corpus_tag,
            output_dir=output_dir,
            limit=limit,
            seed=s,
        )
        if summary is not None:
            summary["_seed"] = s
            summaries.append(summary)

    if not summaries:
        return None

    timestamp = os.environ.get("RAG_BENCHMARK_TIMESTAMP") or time.strftime("%Y%m%d_%H%M%S")
    parent_root = output_dir or (Path("data/results") / timestamp)
    seeds_root = parent_root / strategy / corpus_tag

    has_pending = any(summary.get("status") == "pending_judge" for summary in summaries)
    aggregate = {} if has_pending else _aggregate_seed_summaries(summaries)
    payload = {
        "strategy": strategy,
        "corpus_tag": corpus_tag,
        "seeds": seeds,
        "n_seeds": len(summaries),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending_judge" if has_pending else "completed",
        "aggregate": aggregate,
    }
    seeds_root.mkdir(parents=True, exist_ok=True)
    out_path = seeds_root / "seeds_aggregate.json"
    _write_json(out_path, payload)

    print(f"\n{'=' * 50}")
    print(f"[{strategy.upper()}] Multi-seed Aggregate (N={len(summaries)} seeds={seeds})")
    print(f"{'=' * 50}")
    for key, stats in aggregate.get("overall", {}).items():
        print(
            f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
            f"(95%CI [{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}], n={stats['n']})"
        )
    print(f"\n  Aggregate saved to: {out_path}")
    print(f"{'=' * 50}\n")
    return payload

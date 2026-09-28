import asyncio
import hashlib
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from core.admission import current_post_query_inventory
from core.benchmark_checkpoint import (
    _benchmark_checkpoint_due,
    _order_benchmark_rows,
    _resume_benchmark_rows,
    write_checkpoint,
)
from core.benchmark_evaluation import (
    _aggregate_seed_summaries,
    _apply_judge_label,
    _evaluation_scope,
    _extract_stage_timing,
    _recompute_aggregates,
    _update_summary_status,
)
from core.benchmark_failures import POLICY as FAILURE_POLICY
from core.benchmark_failures import QUALITY_METRICS, BenchmarkIntegrityError
from core.config import RAGConfig
from core.inference_transport import InferenceTransport
from core.paper_compatibility import method_identity
from core.paper_policy import canonical_query_policy, structured_query_identity
from core.prehop_ablation import ablation_identity
from core.semantic_config import parse_strict_bool
from core.strategy_registry import RESEARCH_EXTERNAL_STRATEGIES
from core.vllm_client import get_llm_client
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG
from utils.io import _write_json
from utils.metrics import evaluate_multihoprag_response
from utils.provenance import code_provenance

logger = logging.getLogger("Prehop")

CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"
INDEX_STATS_DIR = Path("data/index_stats")


def _read_json_file(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_benchmark_corpus_manifest(dataset: str, queries_file: str | Path) -> dict | None:
    path = Path(queries_file).parent / f"{dataset}_corpus" / CORPUS_MANIFEST_FILENAME
    return {**_read_json_file(path), "path": str(path)} if path.is_file() else None


def _latest_index_manifest_metadata(strategy: str, corpus_tag: str, stats_dir: Path = INDEX_STATS_DIR) -> dict | None:
    """Resolve one exact index artifact; never infer identity from mtime."""
    requested_run_id = os.environ.get("RAG_RUN_ID", "").strip()
    explicit_path = os.environ.get("RAG_INDEX_STATS_PATH", "").strip()
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidates = [candidate]
        if not candidate.is_file():
            return None
    elif requested_run_id:
        candidates = [stats_dir / f"{strategy}_{corpus_tag}_{requested_run_id}.json"]
        if not candidates[0].is_file():
            return None
    else:
        candidates = sorted(stats_dir.glob(f"{strategy}_{corpus_tag}_*.json"))
    if not candidates:
        return None
    matches: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            candidate_payload = _read_json_file(candidate)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # The filename prefix is ambiguous when one corpus tag prefixes
            # another. Preserve the invalid-artifact guard only when there is
            # no payload available to disambiguate it.
            return {
                "path": str(candidate),
                "status": "invalid",
                "fingerprint": None,
                "paragraph_count": None,
            }
        if not isinstance(candidate_payload, dict):
            return {
                "path": str(candidate),
                "status": "invalid",
                "fingerprint": None,
                "paragraph_count": None,
            }
        payload_strategy = candidate_payload.get("strategy")
        payload_corpus = candidate_payload.get("corpus_tag")
        if payload_strategy is not None and str(payload_strategy) != strategy:
            continue
        if payload_corpus is not None and str(payload_corpus) != corpus_tag:
            continue
        if requested_run_id and candidate_payload.get("run_id") != requested_run_id:
            return {"path": str(candidate), "status": "invalid", "fingerprint": None, "paragraph_count": None}
        matches.append((candidate, candidate_payload))
    if not matches:
        return None
    if len(matches) != 1:
        return {"path": None, "status": "ambiguous", "fingerprint": None, "paragraph_count": None}
    path, payload = matches[0]
    run_id = payload.get("run_id") or path.stem.removeprefix(f"{strategy}_{corpus_tag}_")
    index_code = payload.get("index_code_provenance")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_id": run_id,
        "status": payload.get("status"),
        "fingerprint": payload.get("corpus_manifest_fingerprint"),
        "paragraph_count": payload.get("corpus_manifest_paragraph_count"),
        "code_provenance": index_code if isinstance(index_code, dict) else None,
        "index_policy": payload.get("index_policy") if isinstance(payload.get("index_policy"), dict) else None,
        "index_policy_sha256": payload.get("index_policy_sha256"),
    }


def _query_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(str(row["_id"]) for row in rows)).encode()).hexdigest()


def _query_records_sha256(rows: list[dict[str, Any]]) -> str:
    records = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in sorted(rows, key=lambda item: str(item["_id"]))
    ]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _judge_independence(eval_model: str, model_id: str, default_model: str, allow_self: bool) -> tuple[bool, bool]:
    """Validate that a supplemental judge is independent of generation."""
    evaluator = str(eval_model or "").strip().casefold()
    generation_models = {str(model_id or "").strip().casefold(), str(default_model or "").strip().casefold()}
    is_independent = bool(evaluator) and evaluator not in generation_models
    override_used = not is_independent and bool(allow_self)
    return is_independent, override_used


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


@asynccontextmanager
async def _benchmark_engine(strategy: str, model_id: str, corpus_tag: str, judge_enabled: bool):
    """Close owned adapters on success, initialization failure, cancellation, and I/O errors."""
    engine = None
    failed = False
    try:
        try:
            if strategy == "prehop":
                engine = GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
                if RAGConfig.CONNECTION_TIMING_MODE:
                    from models.prehop.connection_timing import metadata
                    metadata(RAGConfig.CONNECTION_TIMING_STORE, os.environ["RAG_INDEX_NAMESPACE"])
            elif strategy == "naive":
                engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
            elif strategy == "hoprag":
                from models.hoprag.hoprag_adapter import HopRAGAdapter

                engine = HopRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
            elif strategy == "ms_graphrag":
                from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter

                engine = MSGraphRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
            elif strategy in RESEARCH_EXTERNAL_STRATEGIES:
                from models.external_research.adapter import ExternalResearchAdapter

                engine = ExternalResearchAdapter(strategy, model_id=model_id, corpus_tag=corpus_tag)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            vllm = get_llm_client(model_id) if judge_enabled else None
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize engine for {strategy}: {exc}") from exc
        yield engine, vllm
    except BaseException:
        failed = True
        raise
    finally:
        if engine is not None and hasattr(engine, "close"):
            try:
                await asyncio.to_thread(engine.close)
            except Exception:
                if not failed:
                    raise
                logger.warning("Adapter cleanup failed after benchmark failure", exc_info=True)


async def run_benchmark(
    queries_file: str,
    strategy: str,
    model_id: str,
    corpus_tag: str = "default",
    output_dir: Path | None = None,
    limit: int | None = None,
    seed: int | None = None,
):
    """Run one benchmark seed in its own output directory.

    Paper generation seeds follow the method registry; native unseeded APIs
    remain unseeded. Non-paper generation uses the supplied benchmark seed.
    Multi-seed orchestration lives in run_benchmark_multi_seed.
    """
    from core.execution_profile import execution_profile
    from core.phase_timing import BenchmarkTiming
    phase_timing = BenchmarkTiming()

    # Load the selected evaluation manifest before creating engines.
    benchmark_data = await asyncio.to_thread(_read_json_file, queries_file)
    manifest_queries_count = len(benchmark_data)
    reuse_reference = None
    reuse_link = None
    if os.environ.get("RAG_INDEX_REUSE_LINK"):
        from core.index_reuse import load_link, ref
        link_path = Path(os.environ["RAG_INDEX_REUSE_LINK"])
        reuse_link = load_link(link_path)
        reuse_reference = ref(link_path)
    judge_enabled = bool(RAGConfig.JUDGE_ENABLED)
    judge_independent: bool | None = None
    judge_self_override = False
    if judge_enabled:
        judge_independent, judge_self_override = _judge_independence(
            RAGConfig.EVAL_MODEL,
            model_id,
            InferenceTransport.resolve("core").generation_model,
            RAGConfig.JUDGE_ALLOW_SELF,
        )

    if seed is not None:
        from core.strategy_registry import get_strategy

        generation_seed = None if RAGConfig.PREHOP_ABLATION_PROFILE else (get_strategy(strategy).paper_generation_seed if parse_strict_bool(os.environ.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE") else int(seed))
        if generation_seed is None:
            os.environ["RAG_LLM_SEED"] = ""
        else:
            os.environ["RAG_LLM_SEED"] = str(generation_seed)
        os.environ["RAG_SEED"] = str(int(seed))

    transport = InferenceTransport.resolve("core")

    if limit is not None:
        benchmark_data = benchmark_data[: max(0, int(limit))]
        logger.info("--limit %d: evaluating %d queries", limit, len(benchmark_data))
    # This is the evaluated-record identity, not the source-file identity.
    # For exploratory --limit runs it must describe only the admitted subset.
    manifest_query_records_sha256 = _query_records_sha256(benchmark_data)

    # Dataset dispatch via the per-query `dataset` marker. MultiHop-RAG and
    # HotpotQA share one query schema, but their evidence
    # units differ. The evaluator keeps deterministic answer EM/F1 primary,
    # uses title-level evidence P/R/F1 across datasets, and only runs
    # sentence/fact ranking metrics where the gold unit is aligned.
    _MULTIHOP_DATASET_NAMES = {
        "multihoprag": "MultiHop-RAG",
        "hotpotqa": "HotpotQA",
    }
    dataset_marker = (benchmark_data[0].get("dataset", "") if benchmark_data else "").strip().lower()
    dataset_name = _MULTIHOP_DATASET_NAMES[dataset_marker]
    evaluated_query_ids_sha256 = _query_ids_sha256(benchmark_data)
    corpus_manifest = _load_benchmark_corpus_manifest(dataset_marker, queries_file)
    evaluation_scope, official_split_expected_queries = _evaluation_scope(
        dataset_marker,
        len(benchmark_data),
        queries_file, manifest=corpus_manifest)
    if dataset_marker == "hotpotqa":
        sentence_store = Path(queries_file).parent / "hotpotqa_corpus/sentences.sqlite3"
    index_manifest = _latest_index_manifest_metadata(strategy, corpus_tag)
    benchmark_code = code_provenance()
    corpus_index_fingerprint_status = "not_checked"
    async with _benchmark_engine(strategy, model_id, corpus_tag, judge_enabled) as (engine, vllm):
        # Active-index verification is disabled; retain its explicit report status.
        active_index_snapshot = {"status": "not_checked"}
        results: list[dict[str, Any]] = []

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

        if judge_enabled and RAGConfig.JUDGE_BATCH:
            raise RuntimeError("RAG_JUDGE_BATCH is disabled; supplemental judging must use the LiteLLM gateway")
        elif judge_enabled:
            logger.info("Supplemental judge: synchronous mode")
        else:
            logger.info("Supplemental judge: disabled (deterministic/official metrics only)")

        from core.execution_profile import resolved_execution_environment
        execution_env = resolved_execution_environment()
        benchmark_concurrency = int(execution_env["RAG_BENCHMARK_CONCURRENCY"])
        benchmark_checkpoint_every = max(1, int(execution_env["RAG_BENCHMARK_CHECKPOINT_EVERY"]))
        query_sem = asyncio.Semaphore(benchmark_concurrency)
        query_inflight = 0
        observed_query_peak = 0
        write_lock = asyncio.Lock()
        total_queries = len(benchmark_data)
        resume_metadata: dict[str, Any] | None = None
        retained_query_ids: set[str] = set()
        resume_requested_raw = os.environ.get("RAG_BENCHMARK_RESUME", "").strip()
        if resume_requested_raw and parse_strict_bool(resume_requested_raw, name="RAG_BENCHMARK_RESUME"):
            phase_timing.restore(_read_json_file(result_file).get("benchmark_timing"))
            results, resume_metadata = _resume_benchmark_rows(result_file, benchmark_data)
            retained_query_ids = {str(row["query_id"]) for row in results}
            logger.info(
                "Resuming benchmark with %d retained rows; %d queries remain",
                len(results),
                total_queries - len(results),
            )
        if benchmark_concurrency > 1:
            logger.info("Benchmark concurrency: %d queries in flight", benchmark_concurrency)

        query_batch_started = None
        last_answer_finished = None

        def _recompute_and_persist() -> dict[str, Any]:
            """Rebuild the summary from `results` and write the result file +
            report artifacts after each completed query."""
            _order_benchmark_rows(results)
            s: dict[str, Any] = {
                "strategy": strategy,
                "corpus_tag": corpus_tag,
                "dataset": dataset_name,
                "evaluation_scope": evaluation_scope,
                    "dataset_protocol": (corpus_manifest or {}).get("protocol"),
                    "latency_scope": "frozen_prefix_downstream_only" if os.environ.get("RAG_ABLATION_DIRECT_INPUTS") else "end_to_end",
                "official_split_expected_queries": official_split_expected_queries,
                "manifest_queries_count": manifest_queries_count,
                "evaluated_queries_count": total_queries,
                "evaluated_query_ids_sha256": evaluated_query_ids_sha256,
                "evaluated_query_records_sha256": manifest_query_records_sha256,
                "limit": limit,
                "corpus_manifest_path": (corpus_manifest or {}).get("path"),
                "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
                "corpus_manifest_paragraph_count": (corpus_manifest or {}).get("paragraph_count"),
                "index_manifest_stats_path": (index_manifest or {}).get("path"),
                "index_manifest_stats_sha256": (index_manifest or {}).get("sha256"),
                "index_provenance": {
                    "run_id": (index_manifest or {}).get("run_id"),
                    "code": (index_manifest or {}).get("code_provenance"),
                    "policy": (index_manifest or {}).get("index_policy"),
                    "policy_sha256": (index_manifest or {}).get("index_policy_sha256"),
                },
                "query_provenance": dict(benchmark_code),
                "evaluation_provenance": dict(benchmark_code),
                "index_manifest_fingerprint": (index_manifest or {}).get("fingerprint"),
                "index_manifest_status": (index_manifest or {}).get("status"),
                "corpus_index_fingerprint_status": corpus_index_fingerprint_status,
                "active_index_snapshot": active_index_snapshot,
                "official_metric_note": (
                    "Official-compatible fields require the complete official split and a corpus/index rebuilt from this manifest; "
                    "any sample or subset artifact is exploratory only."
                ),
                "judge_enabled": judge_enabled,
                "judge_policy": "supplemental_optional",
                "judge_independent": judge_independent,
                "judge_self_override": judge_self_override,
                "benchmark_concurrency": benchmark_concurrency,
                "observed_query_peak_concurrency": observed_query_peak,
                "benchmark_checkpoint_every": benchmark_checkpoint_every,
                "queries_count": len(results),
                "total_queries": total_queries,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "in_progress",
                "models": {
                    "default": transport.generation_model,
                    "generation_revision": os.environ.get("RAG_GENERATION_REVISION", "").strip() or None,
                    "llm_seed": transport.generation_seed,
                    "embedding": (index_manifest or {})
                    .get("index_policy", {})
                    .get("embedding_model", transport.embedding_model),
                    "embedding_revision": (index_manifest or {}).get("index_policy", {}).get("embedding_revision"),
                    "eval": RAGConfig.EVAL_MODEL,
                },
                "ablation": {
                    "q_minus": RAGConfig.ABLATION_Q_MINUS,
                    "q_plus": RAGConfig.ABLATION_Q_PLUS,
                    "sentence_channel_enabled": RAGConfig.SENTENCE_CHANNEL_ENABLED,
                    **({"chunk_sentences": RAGConfig.CHUNK_SENTENCES} if strategy in {"prehop", "naive"} else {}),
                    "questions_per_direction": RAGConfig.QUESTIONS_PER_DIRECTION,
                    "graph_hop_depth": RAGConfig.GRAPH_HOP_DEPTH,
                    "graph_path_decay": RAGConfig.GRAPH_PATH_DECAY,
                    "graph_edge_variant": RAGConfig.GRAPH_EDGE_VARIANT,
                    "hop_edge_filter": RAGConfig.HOP_EDGE_FILTER,
                    "hop_seed_policy": RAGConfig.HOP_SEED_POLICY,
                    "qplus_hop_activation": RAGConfig.QPLUS_HOP_ACTIVATION,
                    "continuation_edges_enabled": RAGConfig.CONTINUATION_EDGES_ENABLED,
                    "continuation_anchor_policy": RAGConfig.CONTINUATION_ANCHOR_POLICY,
                    "hop_semantic_variant": RAGConfig.HOP_SEMANTIC_VARIANT,
                    "question_schema": RAGConfig.QUESTION_SCHEMA,
                    "precompute_reciprocal_hops": RAGConfig.PRECOMPUTE_RECIPROCAL_HOPS,
                    "default_top_k": RAGConfig.DEFAULT_TOP_K,
                    "candidate_pool_multiplier": RAGConfig.CANDIDATE_POOL_MULTIPLIER,
                    "fulltext_analyzer": RAGConfig.FULLTEXT_ANALYZER,
                    "hypo_channel_variant": RAGConfig.HYPO_CHANNEL_VARIANT,
                    "source_selection_variant": RAGConfig.SOURCE_SELECTION_VARIANT,
                    "candidate_order_input_order": RAGConfig.CANDIDATE_ORDER_INPUT_ORDER,
                    "candidate_order_shuffle_seed": RAGConfig.CANDIDATE_ORDER_SHUFFLE_SEED,
                    "final_rank_variant": RAGConfig.FINAL_RANK_VARIANT,
                        **structured_query_identity(strategy),
                        **method_identity(strategy),
                        **(ablation_identity() if strategy == "prehop" else {}),
                        **(canonical_query_policy(strategy) if strategy in {"prehop", "hoprag", "linear_rag"} else {}),
                },
            }
            if resume_metadata is not None:
                resumed_rows = [row for row in results if str(row.get("query_id")) not in retained_query_ids]
                s["resume"] = {
                    **resume_metadata,
                    "resumed_rows": len(resumed_rows),
                    "remaining_rows": total_queries - len(results),
                }
                s["evaluation_provenance_segments"] = [
                    {
                        "segment": "retained_before_resume",
                        "query_count": len(retained_query_ids),
                        "query_ids_sha256": resume_metadata["retained_query_ids_sha256"],
                        "code": resume_metadata.get("prior_evaluation_provenance"),
                    },
                    {
                        "segment": "executed_after_resume",
                        "query_count": len(resumed_rows),
                        "query_ids_sha256": hashlib.sha256(
                            "\n".join(sorted(str(row["query_id"]) for row in resumed_rows)).encode()
                        ).hexdigest(),
                        "code": dict(benchmark_code),
                    },
                ]
            from core.amortized_cost import query_cost
            s["execution_profile"] = execution_profile()
            query_wall = (last_answer_finished - query_batch_started
                          if last_answer_finished is not None and query_batch_started is not None else None)
            s["query_batch_timing"] = {"version": 1, "wall_seconds": query_wall,
                                      "resumed": resume_metadata is not None}
            s["amortized_query_cost"] = query_cost(
                query_wall, total_queries,
                complete=len(results) == total_queries and not any(row.get("failure_scope") == "target" for row in results),
                resumed=resume_metadata is not None)
            s["benchmark_timing"] = phase_timing.snapshot()
            if reuse_reference is not None:
                s["index_reuse"] = reuse_reference
                s["phase_costs"] = {
                    "index_source_timing_seconds": reuse_link["index_timing_seconds"],
                    "reuse_preparation_seconds": reuse_link["preparation_elapsed_seconds"],
                    "benchmark_wall_seconds": s["benchmark_timing"]["total_wall_seconds"],
                    "index_plus_benchmark_wall_seconds": reuse_link["index_timing_seconds"]["total_elapsed_seconds"] + s["benchmark_timing"]["total_wall_seconds"],
                    "query_latency_sum_seconds": sum(float(row.get("latency", 0)) for row in results),
                    "scope": "successful_index_plus_checkpointed_query_segments_including_terminal_failures; other_run_attempts_separate",
                }
            s["details"] = results
            if len(results) == total_queries:
                s["post_query_artifact_inventory"] = current_post_query_inventory(strategy, corpus_tag)
            _recompute_aggregates(s)
            _update_summary_status(s)
            write_checkpoint(s, result_file)
            return s

        target_failure = False

        async def _process_query(idx: int, item: dict[str, Any]):
            nonlocal summary, query_inflight, observed_query_peak, last_answer_finished, target_failure
            submitted_at = time.perf_counter()
            async with query_sem:
                if target_failure:
                    return
                queue_wait_seconds = time.perf_counter() - submitted_at
                query_inflight += 1
                observed_query_peak = max(observed_query_peak, query_inflight)
                from core.inference_telemetry import begin as begin_inference
                from core.inference_telemetry import finish as finish_inference

                telemetry_token = begin_inference()
                started = time.time()
                stage_timing: dict[str, float] = {}
                original_query = str(item.get("query", ""))
                query = original_query
                ground_truth = item.get("ground_truth", "")
                category = item.get("category", "Uncategorized")
                try:
                    query = _build_benchmark_query(original_query, item)
                    from contextlib import nullcontext

                    from models.prehop.tracing import trace_identity
                    with (trace_identity(query_id=str(item["_id"]), query_index=idx, phase="benchmark")
                          if strategy == "prehop" else nullcontext()):
                        response, retrieved_sources, trace = await engine.run_workflow(query, [])
                    last_answer_finished = time.perf_counter()
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
                        judge_enabled=judge_enabled,
                        supporting_facts=item.get("supporting_facts"),
                        hotpot_sentence_store=str(sentence_store) if dataset_marker == "hotpotqa" else None,
                    )
                    expected_sources = {
                        "docs": item.get("evidence_docs", []),
                        "facts": item.get("evidence_facts", []),
                        "supporting_facts": item.get("supporting_facts", []),
                    }
                    result_item = {
                        "idx": idx + 1,
                        "query_id": str(item.get("_id", "")),
                        "original_query_id": str(item.get("original_query_id", item.get("_id", ""))),
                        "query": original_query,
                        "category": category,
                        "question_type": item.get("question_type", ""),
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
                    last_answer_finished = time.perf_counter()
                    error_text = f"{type(exc).__name__}: {exc}"
                    target_failure = target_failure or isinstance(exc, BenchmarkIntegrityError)

                    metrics = {
                        "llm_judge_score": -1.0,
                        "llm_judge_reason": "runtime_error",
                        "groundedness": -1.0,
                        "groundedness_source": "runtime_error",
                        "hallucination": -1.0,
                        "hallucination_reason": "runtime_error",
                        "hallucination_source": "runtime_error",
                        "hallucination_model": str(RAGConfig.EVAL_MODEL or ""),
                        "answer_em": -1.0,
                        "answer_f1": -1.0,
                        "answer_precision": -1.0,
                        "answer_recall": -1.0,
                        "official_answer_em": -1.0,
                        "official_answer_f1": -1.0,
                        "official_qa_accuracy": -1.0,
                        "null_refusal": -1.0,
                        "doc_match": -1.0,
                        # Keep the same numeric keys the success path emits so the
                        # summary auto-averaging stays consistent across queries.
                        "official_mrr@10": -1.0,
                        "official_map@10": -1.0,
                        "official_hits@4": -1.0,
                        "official_hits@10": -1.0,
                        "evidence_fact_recall@4": -1.0,
                        "evidence_fact_recall@10": -1.0,
                        "evidence_doc_recall": -1.0,
                        "evidence_doc_precision": -1.0,
                        "evidence_doc_f1": -1.0,
                    }
                    metrics.update({key: 0.0 for key in QUALITY_METRICS})
                    expected_sources = {
                        "docs": item.get("evidence_docs", []),
                        "facts": item.get("evidence_facts", []),
                        "supporting_facts": item.get("supporting_facts", []),
                    }
                    result_item = {
                        "idx": idx + 1,
                        "query_id": str(item.get("_id", "")),
                        "original_query_id": str(item.get("original_query_id", item.get("_id", ""))),
                        "query": original_query,
                        "category": category,
                        "question_type": item.get("question_type", ""),
                        "answer": f"@@ANSWER: ERROR - {error_text}",
                        "ground_truth": ground_truth,
                        "expected_sources": expected_sources,
                        "retrieved_sources": [],
                        "interaction_trace": [{"step": "error", "output": error_text}],
                        "latency": latency,
                        "error": error_text,
                        "failure_scope": "target" if isinstance(exc, BenchmarkIntegrityError) else "query",
                        "failure_policy": FAILURE_POLICY,
                        **stage_timing,
                        **metrics,
                    }

                finally:
                    inference_usage = finish_inference(
                        telemetry_token, external_complete=strategy not in RESEARCH_EXTERNAL_STRATEGIES
                    )
                    query_inflight -= 1

                result_item["queue_wait_seconds"] = queue_wait_seconds
                result_item["wall_latency"] = time.perf_counter() - submitted_at
                worker_queue_seconds = float(result_item.get("worker_queue_seconds", 0.0))
                result_item["execution_latency_including_worker_queue"] = float(result_item["latency"])
                result_item["service_latency"] = max(0.0, float(result_item["latency"]) - worker_queue_seconds)
                # Keep the published avg_latency comparable as active service time;
                # queueing is reported independently rather than hidden in it.
                result_item["latency"] = result_item["service_latency"]
                result_item["inference_usage"] = inference_usage
                if query != original_query:
                    result_item["benchmark_query"] = query

                recorder = getattr(engine, "trace_recorder", None) if strategy == "prehop" else None
                if recorder is not None:
                    result_item["prehop_trace"] = {**recorder.reference, "query_id": str(item["_id"])}
                    recorder.emit("benchmark.query", {"input": item, "result": result_item},
                                  identity={"query_id": str(item["_id"]), "query_index": idx})

                # Primary labels/rates are deterministic; judge labels remain
                # separately named supplemental analysis.
                _apply_judge_label(result_item)

                async with write_lock:
                    results.append(result_item)
                    _order_benchmark_rows(results)

                    error_suffix = " [ERROR]" if result_item.get("error") else ""
                    print(
                        f"[{strategy}] ({len(results)}/{total_queries}) [{category}]{error_suffix} "
                        f"Primary: {result_item.get('primary_answer_score', -1.0):.1f} | "
                        f"Judge: {metrics['llm_judge_score']:.1f} | Hallu: {result_item.get('hallucination', -1.0):.1f} "
                        f"| DocMatch: {metrics['doc_match']:.0f} | Latency: {latency:.1f}s"
                    )

                    if _benchmark_checkpoint_due(len(results), total_queries, benchmark_checkpoint_every):
                        summary = _recompute_and_persist()
                        from scripts.recovery_checkpoint import checkpoint_barrier
                        await checkpoint_barrier(result_file, summary)

        pending_items = [(i, item) for i, item in enumerate(benchmark_data) if str(item["_id"]) not in retained_query_ids]
        query_batch_started = time.perf_counter()
        tasks = [asyncio.create_task(_process_query(i, item)) for i, item in pending_items]
        try:
            await asyncio.gather(*tasks)
        finally:
            # A write failure must not leave sibling queries using an adapter that
            # is about to close. Drain cancellation before leaving its scope.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if not results:
            return None

        # The final write is unconditional so an empty pending set after a valid
        # resume, or a non-divisible checkpoint interval, cannot leave stale
        # aggregate metadata behind.
        summary = _recompute_and_persist()

        print(f"\n{'=' * 50}")
        completion_label = (
            "Benchmark Complete (unadmitted)" if summary["status"] == "completed_unadmitted"
            else f"Benchmark {summary['status']}"
        )
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


async def run_benchmark_multi_seed(
    queries_file: str,
    strategy: str,
    model_id: str,
    seeds: list[int] | None = None,
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

    has_failed = any(summary.get("status") != "completed_unadmitted" for summary in summaries)
    aggregate = {} if has_failed else _aggregate_seed_summaries(summaries)
    payload = {
        "strategy": strategy,
        "corpus_tag": corpus_tag,
        "seeds": seeds,
        "n_seeds": len(summaries),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "failed" if has_failed else "completed_unadmitted",
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

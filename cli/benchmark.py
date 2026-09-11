import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from core.admission import current_post_query_inventory
from core.benchmark_failures import POLICY as FAILURE_POLICY
from core.benchmark_failures import QUALITY_METRICS, BenchmarkIntegrityError, metric_value
from core.config import RAGConfig
from core.paper_compatibility import method_identity
from core.paper_policy import canonical_query_policy, structured_query_identity
from core.prehop_ablation import ablation_identity
from core.semantic_config import parse_strict_bool
from core.strategy_registry import RESEARCH_EXTERNAL_STRATEGIES
from core.vllm_client import get_llm_client
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG
from utils.io import _safe_float, _write_json
from utils.metrics import evaluate_multihoprag_response, extract_final_answer
from utils.provenance import code_provenance
from utils.reporting import _write_model_report_artifacts

logger = logging.getLogger("Prehop")

# Expected row counts for the exact official splits prepared by this
# repository.  Scope is determined from the rows actually evaluated, never
# merely from a filename.
OFFICIAL_SPLIT_QUERY_COUNTS = {"multihoprag": 2556, "hotpotqa": 7405}
# Canonical official prepared-manifest identity.  Each value is
# SHA256("\n".join(sorted(row["_id"] for row in official_rows))).
OFFICIAL_QUERY_ID_DIGESTS = {
    "hotpotqa": "8f1a1b80b352ff578988c4dfb320f44dc7c08c5e4d4b86ef132598271a0adb00",
    "multihoprag": "e683a5bf5807edf5f06612066f2ad5fa0b0b08f61a726a71ec28afd8e66177b0",
}
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


def _manifest_source_ids(corpus_manifest: dict | None) -> list[str] | None:
    """Read the prepared corpus identity set without touching an index.

    Legacy MultiHop-RAG manifests are optional; only a manifest-bearing corpus
    can be checked against an active source snapshot.
    """
    if corpus_manifest is None:
        return None
    corpus_dir = Path(str(corpus_manifest["path"])).parent
    source_ids = sorted(path.stem for path in corpus_dir.iterdir() if path.is_file() and path.suffix in (".txt", ".md"))
    return source_ids


def _source_set_sha256(source_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(source_ids)).encode("utf-8")).hexdigest()


async def _index_snapshot_metadata(engine, strategy, corpus_tag, corpus_manifest, strict=False):
    """Record that execution does not perform an active-index verification."""
    return {"status": "not_checked"}


def _judge_independence(eval_model: str, model_id: str, default_model: str, allow_self: bool) -> tuple[bool, bool]:
    """Validate that a supplemental judge is independent of generation."""
    evaluator = str(eval_model or "").strip().casefold()
    generation_models = {str(model_id or "").strip().casefold(), str(default_model or "").strip().casefold()}
    is_independent = bool(evaluator) and evaluator not in generation_models
    override_used = not is_independent and bool(allow_self)
    return is_independent, override_used


def _extract_final_answer(answer_text: str) -> str:
    """Use the canonical metric answer boundary for reporting labels."""
    return extract_final_answer(answer_text)


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
    """Pull rewrite/retrieve/traversal/synthesis timing out of a prehop-style
    `interaction_trace` (see models/prehop/graphrag.py's `run_workflow`),
    if present, so they land as top-level numeric fields on `result_item`
    and get auto-averaged into the corresponding ``avg_*`` fields by
    `_recompute_aggregates`. This keeps the stage split
    rather than only the aggregate `latency`.

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
        if step.get("step") == "query_rewrite":
            if "rewrite_ms" in step:
                timing["rewrite_ms"] = float(step.get("rewrite_ms") or 0.0)
        elif step.get("step") == "retrieve":
            for key in (
                "retrieve_ms",
                "traversal_ms",
                "graph_expand_ms",
                "deterministic_score_ms",
                "candidate_order_ms",
            ):
                if key in step:
                    timing[key] = float(step.get(key) or 0.0)
        elif step.get("step") == "synthesis" and "synthesis_ms" in step:
            timing["synthesis_ms"] = float(step.get("synthesis_ms") or 0.0)
        elif str(step.get("step", "")).endswith("_official_retrieval") and "worker_queue_seconds" in step:
            timing["worker_queue_seconds"] = float(step["worker_queue_seconds"])
    return timing


def _apply_judge_label(result_item: dict[str, Any]) -> None:
    """Attach deterministic primary labels and separate supplemental labels."""
    from utils.abstain import answer_label, is_abstain

    answer_text = str(result_item.get("answer", "") or "")
    has_error = bool(result_item.get("error"))
    # Detect abstain on the EXTRACTED final answer (\\boxed{} / 'Final Answer:'),
    # not the full CoT body which often uses 'insufficient evidence' mid-reason.
    final_answer = _extract_final_answer(answer_text).lower()
    abstained = is_abstain(final_answer)
    judge_score = _safe_float(result_item.get("llm_judge_score", -1.0), -1.0)
    result_item["final_answer_extracted"] = final_answer[:300]

    # The headline correctness/label is deterministic.  A null query uses its
    # explicit refusal metric; other rows use EM.
    primary = (
        result_item.get("null_refusal")
        if result_item.get("question_type") == "null_query"
        else result_item.get("answer_em")
    )
    primary_score = _safe_float(primary, -1.0)
    primary_score = 0.0 if has_error else primary_score
    result_item["primary_answer_score"] = primary_score
    if has_error or primary_score < 0:
        answer_attempted = -1.0 if not has_error else 0.0
        primary_label = "Incorrect Answer" if has_error else "Unscored"
    else:
        answer_attempted = 0.0 if abstained else 1.0
        primary_label = "Correct Answer" if primary_score >= 0.5 else ("Refusal" if abstained else "Incorrect Answer")
    result_item["answer_attempted"] = answer_attempted
    result_item["answer_label"] = primary_label
    # Kept only for optional judge analysis; never drives correct_rate.
    result_item["judge_answer_label"] = answer_label(judge_score, final_answer) if judge_score >= 0 else "Unjudged"


def _recompute_aggregates(s: dict[str, Any]) -> None:
    """Recompute avg_<metric>, category_summaries and the 3-way label counts
    from ``s['details']`` in place. Shared by the live benchmark pass and the
    async batch reconcile step so both yield identical aggregates. Averages skip
    the UNJUDGED sentinel (-1); every real metric is in [0, 1] (or latency >= 0).
    """
    rows = s.get("details") or []
    s["failure_policy"] = FAILURE_POLICY
    s["query_failure_count"] = sum(bool(row.get("error")) for row in rows)
    s["query_failure_rate"] = s["query_failure_count"] / len(rows) if rows else 0.0

    def _eligible_values(subset: list[dict], key: str) -> list[float]:
        return [value for row in subset if (value := metric_value(row, key)) is not None]

    def _avg(subset: list[dict], key: str) -> float:
        vals = _eligible_values(subset, key)
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
        s[f"eligible_{key}_count"] = len(_eligible_values(rows, key))

    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r.get("category", "Uncategorized"), []).append(r)
    cat_summaries: dict[str, Any] = {}
    for cat, cat_list in cats.items():
        cat_sum: dict[str, Any] = {"count": len(cat_list)}
        for key in numeric_keys:
            cat_sum[f"avg_{key}"] = _avg(cat_list, key)
            cat_sum[f"eligible_{key}_count"] = len(_eligible_values(cat_list, key))
        cat_summaries[cat] = cat_sum
    s["category_summaries"] = cat_summaries

    label_counts = {"Correct Answer": 0, "Incorrect Answer": 0, "Refusal": 0}
    for r in rows:
        label = r.get("answer_label")
        if label in label_counts:
            label_counts[label] += 1
    total = sum(label_counts.values()) or 1  # deterministically scored rows only
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
        if not row.get("error") and (not isinstance(row.get(key), (int, float)) or isinstance(row.get(key), bool) or float(row[key]) < 0)
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
    if any(row.get("failure_scope") == "target" for row in rows):
        summary["status"] = "failed"
    elif len(rows) < int(summary.get("total_queries", len(rows)) or 0):
        summary["status"] = "in_progress"
    elif any(
        (row.get("_deferred_judge") or row.get("judge_custom_id")) and _safe_float(row.get("llm_judge_score"), -1.0) < 0
        for row in rows
    ) or any(row.get("failure_scope") == "target" for row in rows) or summary.get("judge_enabled") and (
        _unjudged_count(rows) or _unjudged_count(rows, "hallucination") or _unjudged_groundedness_count(rows)
    ):
        summary["status"] = "failed"
    else:
        summary["status"] = "completed_unadmitted"


def _evaluation_scope(
    dataset: str,
    evaluated_count: int,
    source: str,
    evaluated_query_ids_sha256: str,
) -> tuple[str, int | None]:
    """Classify an artifact from its actual evaluated row count.

    A complete expected split is ``full_benchmark`` even if its filename is
    unconventional.  Incomplete files explicitly named as samples remain
    ``sample_exploratory``; all other incomplete selections (including CLI
    ``--limit``) are ``subset_exploratory``.
    """
    manifest = _load_benchmark_corpus_manifest(dataset, source)
    if manifest and manifest.get("protocol") == "hotpotqa-hipporag-v1-1000":
        return ("released_benchmark" if evaluated_count == manifest["query_count"] else "subset_exploratory"), manifest["query_count"]
    expected = OFFICIAL_SPLIT_QUERY_COUNTS.get(str(dataset).lower())
    if expected is not None and evaluated_count == expected:
        OFFICIAL_QUERY_ID_DIGESTS.get(str(dataset).lower())
        return "full_benchmark", expected
    if "sample" in Path(source).name.lower():
        return "sample_exploratory", expected
    return "subset_exploratory", expected


def _slim_details(details: list | None) -> list:
    """Strip interaction traces and private fields from the main result JSON."""
    return [
        {k: v for k, v in d.items() if k != "interaction_trace" and not k.startswith("_")} if isinstance(d, dict) else d
        for d in (details or [])
    ]


def _write_slim_main(s: dict[str, Any], result_file: Path) -> None:
    _write_json(result_file, {**s, "details": _slim_details(s.get("details"))})


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def _resume_benchmark_rows(
    result_file: Path,
    benchmark_data: list[dict[str, Any]],
    expected_metadata: dict[str, Any],
    *,
    judge_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a crash-interrupted deterministic benchmark without duplicating rows.

    Resume is deliberately strict: the immutable query identity, runtime
    configuration, model selection and active index identity must match.
    Terminal runtime-error rows are retained; only unexecuted queries resume.
    """

    prior = _read_json_file(result_file)
    for field in expected_metadata:
        observed = prior.get(field)
        if field == "ablation" and isinstance(observed, dict):
            # Resume artifacts written immediately before the public
            # candidate-order terminology and diagnostic controls were added.
            # These values reproduce the only behavior that existed in that
            # revision; non-default or otherwise different settings still
            # fail the strict metadata comparison below.
            observed = dict(observed)
            if "candidate_order_input_order" not in observed and "rerank_input_order" in observed:
                observed["candidate_order_input_order"] = observed.pop("rerank_input_order")
            if "candidate_order_shuffle_seed" not in observed and "rerank_shuffle_seed" in observed:
                observed["candidate_order_shuffle_seed"] = observed.pop("rerank_shuffle_seed")
            observed.setdefault("graph_path_decay", 0.5)
            observed.setdefault("final_rank_variant", "fused")

    manifest_by_id = {str(item["_id"]): item for item in benchmark_data}
    manifest_position = {str(item["_id"]): idx for idx, item in enumerate(benchmark_data, start=1)}
    prior_rows = prior.get("details")

    trace_file = result_file.with_name(f"{result_file.stem}.traces.jsonl")
    trace_rows = _read_jsonl_file(trace_file)

    retained: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rerun_error_count = 0
    for position, (raw_row, trace_row) in enumerate(zip(prior_rows, trace_rows, strict=True), start=1):
        query_id = str(raw_row.get("query_id") or "")
        seen_ids.add(query_id)
        manifest_item = manifest_by_id.get(query_id)
        str(manifest_item["query"])
        expected_idx = manifest_position[query_id]
        raw_row.get("idx")
        trace_row.get("idx")
        retained.append({**raw_row, "idx": expected_idx, "interaction_trace": trace_row.get("interaction_trace", [])})

    retained.sort(key=lambda row: int(row["idx"]))

    retained_ids = sorted(str(row["query_id"]) for row in retained)
    resume_metadata = {
        "requested": True,
        "prior_status": prior.get("status"),
        "initial_rows": len(prior_rows),
        "retained_rows": len(retained),
        "rerun_error_rows": rerun_error_count,
        "retained_query_ids_sha256": hashlib.sha256("\n".join(retained_ids).encode()).hexdigest(),
        "prior_query_provenance": prior.get("query_provenance"),
        "prior_evaluation_provenance": prior.get("evaluation_provenance"),
    }
    return retained, resume_metadata


def _benchmark_checkpoint_due(completed: int, total: int, every: int) -> bool:
    """Return whether an incremental artifact checkpoint is due."""
    return completed == total or completed % every == 0


def _order_benchmark_rows(rows: list[dict[str, Any]]) -> None:
    """Normalize completed rows to immutable input-manifest order in place."""
    rows.sort(key=lambda row: int(row["idx"]))


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
    """Run one benchmark seed in its own output directory.

    Paper generation seeds follow the method registry; native unseeded APIs
    remain unseeded. Non-paper generation uses the supplied benchmark seed.
    Multi-seed orchestration lives in run_benchmark_multi_seed.
    """
    from core.execution_profile import execution_profile
    from core.phase_timing import BenchmarkTiming
    phase_timing = BenchmarkTiming()

    # Validate the immutable evaluation manifest before creating engines or
    # contacting inference services. This makes stale corpus manifests fail
    # immediately instead of consuming a benchmark run with ineligible rows.
    benchmark_data = await asyncio.to_thread(_read_json_file, queries_file)
    manifest_queries_count = len(benchmark_data)
    reuse_reference = None
    reuse_link = None
    if os.environ.get("RAG_INDEX_REUSE_LINK"):
        from core.index_reuse import load_link, ref
        link_path = Path(os.environ["RAG_INDEX_REUSE_LINK"])
        reuse_link = load_link(link_path, os.environ.get("RAG_BENCHMARK_TIMESTAMP", ""), strategy, corpus_tag)
        reuse_reference = ref(link_path)
    judge_enabled = bool(RAGConfig.JUDGE_ENABLED)
    judge_independent: bool | None = None
    judge_self_override = False
    if judge_enabled:
        judge_independent, judge_self_override = _judge_independence(
            RAGConfig.EVAL_MODEL,
            model_id,
            RAGConfig.DEFAULT_MODEL,
            RAGConfig.JUDGE_ALLOW_SELF,
        )

    if seed is not None:
        from core.strategy_registry import get_strategy

        generation_seed = None if RAGConfig.PREHOP_ABLATION_PROFILE else (get_strategy(strategy).paper_generation_seed if parse_strict_bool(os.environ.get("RAG_PAPER_MODE", "false"), name="RAG_PAPER_MODE") else int(seed))
        RAGConfig.LLM_SEED = generation_seed
        if generation_seed is None:
            os.environ["RAG_LLM_SEED"] = ""
        else:
            os.environ["RAG_LLM_SEED"] = str(generation_seed)
        os.environ["RAG_SEED"] = str(int(seed))

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
    evaluation_scope, official_split_expected_queries = _evaluation_scope(
        dataset_marker,
        len(benchmark_data),
        queries_file,
        evaluated_query_ids_sha256,
    )
    corpus_manifest = _load_benchmark_corpus_manifest(dataset_marker, queries_file)
    if dataset_marker == "hotpotqa":
        sentence_store = Path(queries_file).parent / "hotpotqa_corpus/sentences.sqlite3"
    index_manifest = _latest_index_manifest_metadata(strategy, corpus_tag)
    benchmark_code = code_provenance()
    corpus_index_fingerprint_status = "not_checked"
    # Identity validation runs before constructing adapters, some of which
    # open external-service clients during initialization.
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
    # Stats artifacts are only a report of what indexing intended to build.
    # Before query execution, prove the currently active graph/parquet snapshot
    # still matches the prepared corpus. Full benchmarks fail closed; subsets
    # retain the diagnostic state for exploratory debugging.
    active_index_snapshot = await _index_snapshot_metadata(
        engine,
        strategy,
        corpus_tag,
        corpus_manifest,
        strict=evaluation_scope == "full_benchmark",
    )
    results: list[dict[str, Any]] = []
    category_results: dict[str, list[dict[str, Any]]] = {}

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

    from core.strategy_registry import PAPER_TRANSPORT
    benchmark_concurrency = int(os.environ.get("RAG_BENCHMARK_CONCURRENCY", str(PAPER_TRANSPORT.benchmark_concurrency)))
    benchmark_checkpoint_every = max(1, int(os.environ.get("RAG_BENCHMARK_CHECKPOINT_EVERY", "10")))
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
        results, resume_metadata = _resume_benchmark_rows(
            result_file,
            benchmark_data,
            {
                "execution_profile": execution_profile(),
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
                "corpus_manifest_fingerprint": (corpus_manifest or {}).get("fingerprint"),
                "index_manifest_fingerprint": (index_manifest or {}).get("fingerprint"),
                "index_manifest_status": (index_manifest or {}).get("status"),
                "corpus_index_fingerprint_status": corpus_index_fingerprint_status,
                "active_index_snapshot": active_index_snapshot,
                **({"index_reuse": reuse_reference} if reuse_reference is not None else {}),
                "judge_enabled": judge_enabled,
                "models": {
                    "default": RAGConfig.DEFAULT_MODEL,
                    "generation_revision": os.environ.get("RAG_GENERATION_REVISION", "").strip() or None,
                    "llm_seed": RAGConfig.LLM_SEED,
                    "embedding": (index_manifest or {})
                    .get("index_policy", {})
                    .get("embedding_model", RAGConfig.EMBEDDING_MODEL),
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
            },
            judge_enabled=judge_enabled,
        )
        retained_query_ids = {str(row["query_id"]) for row in results}
        for row in results:
            category_results.setdefault(str(row.get("category", "Uncategorized")), []).append(row)
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
                "default": RAGConfig.DEFAULT_MODEL,
                "generation_revision": os.environ.get("RAG_GENERATION_REVISION", "").strip() or None,
                "llm_seed": RAGConfig.LLM_SEED,
                "embedding": (index_manifest or {})
                .get("index_policy", {})
                .get("embedding_model", RAGConfig.EMBEDDING_MODEL),
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
        from core.execution_profile import execution_profile
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
        # Report artifacts first (writes full traces), then a slim main JSON.
        try:
            _write_model_report_artifacts(s, result_file)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to write report artifacts for %s: %s", result_file, exc)
        _write_slim_main(s, result_file)
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
                if category not in category_results:
                    category_results[category] = []
                category_results[category].append(result_item)

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
    await asyncio.gather(
        *[_process_query(i, item) for i, item in pending_items],
        return_exceptions=False,
    )

    if not results:
        if hasattr(engine, "close"):
            await asyncio.to_thread(engine.close)
        return None

    # The final write is unconditional so an empty pending set after a valid
    # resume, or a non-divisible checkpoint interval, cannot leave stale
    # aggregate metadata behind.
    summary = _recompute_and_persist()

    try:
        _write_model_report_artifacts(summary, result_file)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to write final report artifacts for %s: %s", result_file, exc)
    _write_slim_main(summary, result_file)

    print(f"\n{'=' * 50}")
    completion_label = "Benchmark Complete (unadmitted)"
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
    if hasattr(engine, "close"):
        await asyncio.to_thread(engine.close)
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
        metric = key.removeprefix("avg_")
        vals = [
            _safe_float(s[key], 0.0)
            for s in summaries
            if key in s and _safe_float(s.get(f"eligible_{metric}_count"), 0.0) > 0
        ]
        if vals:
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
            metric = key.removeprefix("avg_")
            vals = [
                _safe_float(s["category_summaries"][cat][key], 0.0)
                for s in summaries
                if cat in (s.get("category_summaries") or {})
                and key in s["category_summaries"][cat]
                and _safe_float(s["category_summaries"][cat].get(f"eligible_{metric}_count"), 0.0) > 0
            ]
            if vals:
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

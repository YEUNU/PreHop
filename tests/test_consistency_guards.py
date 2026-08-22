from unittest.mock import AsyncMock, MagicMock

import pytest

from cli.benchmark import (
    _assert_benchmark_complete,
    _recompute_aggregates,
    _validate_benchmark_data,
)
from cli.index import run_indexing
from core.vllm_client import VLLMClient
from models.hoprag.hoprag_adapter import HopRAGAdapter
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts.shared import build_answer_prompt


@pytest.mark.asyncio
async def test_missing_dataset_fails_instead_of_returning_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        await run_indexing(str(tmp_path / "missing"), "prehop", "default")


@pytest.mark.asyncio
async def test_empty_dataset_fails_instead_of_reporting_zero_success(tmp_path):
    with pytest.raises(ValueError, match="no supported"):
        await run_indexing(str(tmp_path), "prehop", "default")


def test_benchmark_schema_rejects_missing_query():
    with pytest.raises(ValueError, match="non-empty 'query'"):
        _validate_benchmark_data([{"dataset": "hotpotqa"}], "queries.json")


def test_removed_optional_routes_are_not_configurable():
    from core.config import RAGConfig

    for name in (
        "JUDGE_BATCH_ASYNC",
        "BENCHMARK_GATE_ENABLED",
        "RERANKER_THRESHOLD",
        "HOP_THRESHOLD",
    ):
        assert not hasattr(RAGConfig, name)


def test_config_rejects_unknown_hypothetical_channel_variant(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "typo")
    with pytest.raises(ValueError, match="HYPO_CHANNEL_VARIANT"):
        RAGConfig.validate()


def test_config_rejects_disabled_requested_channel(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "qminus_only")
    monkeypatch.setattr(RAGConfig, "ABLATION_Q_MINUS", False)
    with pytest.raises(ValueError, match="requires"):
        RAGConfig.validate()


def test_config_rejects_embedding_pressure_above_server_capacity(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "EMBEDDING_BATCH_SIZE", 65)
    monkeypatch.setattr(RAGConfig, "MAX_CONCURRENT_EMBEDDING_REQUESTS", 2)
    monkeypatch.setattr(RAGConfig, "VLLM_MAX_NUM_SEQS", 128)
    with pytest.raises(ValueError, match="Embedding client can exceed"):
        RAGConfig.validate()


def test_config_rejects_generation_pressure_above_server_capacity(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "MAX_CONCURRENT_LLM_CALLS", 129)
    monkeypatch.setattr(RAGConfig, "VLLM_MAX_NUM_SEQS", 128)
    with pytest.raises(ValueError, match="Generation client can exceed"):
        RAGConfig.validate()


def test_matrix_capacity_plan_accounts_for_shared_endpoint(monkeypatch):
    from scripts.run_index_matrix import _fit_inference_capacity

    monkeypatch.setenv("VLLM_URL", "http://inference.example/v1")
    monkeypatch.setenv("VLLM_EMBED_URL", "http://inference.example/v1")
    monkeypatch.setenv("RAG_INFERENCE_CAPACITY_MODE", "shared")
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "128")
    monkeypatch.setenv("MAX_CONCURRENT_LLM_CALLS", "30")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2")

    plan = _fit_inference_capacity(max_parallel=2, max_generation_parallel=1)

    assert plan["generation_embedding_share_endpoint"] is True
    assert plan["effective"]["embedding_concurrency_per_target"] == 1
    assert plan["effective"]["aggregate_capacity_upper_bound"] == 94


def test_matrix_capacity_plan_keeps_independent_server_budgets(monkeypatch):
    from scripts.run_index_matrix import _fit_inference_capacity

    monkeypatch.setenv("VLLM_URL", "http://generation.example/v1")
    monkeypatch.setenv("VLLM_EMBED_URL", "http://embedding.example/v1")
    monkeypatch.setenv("RAG_INFERENCE_CAPACITY_MODE", "auto")
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "128")
    monkeypatch.setenv("MAX_CONCURRENT_LLM_CALLS", "30")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2")

    plan = _fit_inference_capacity(max_parallel=2, max_generation_parallel=1)

    assert plan["generation_embedding_share_endpoint"] is False
    assert plan["adjusted"] is False
    assert plan["effective"]["aggregate_capacity_upper_bound"] == 128


def test_matrix_capacity_plan_supports_separate_servers_behind_one_gateway(monkeypatch):
    from scripts.run_index_matrix import _fit_inference_capacity

    monkeypatch.setenv("VLLM_URL", "http://gateway.example/v1")
    monkeypatch.setenv("VLLM_EMBED_URL", "http://gateway.example/v1")
    monkeypatch.setenv("RAG_INFERENCE_CAPACITY_MODE", "separate")
    monkeypatch.setenv("VLLM_GENERATION_MAX_NUM_SEQS", "128")
    monkeypatch.setenv("VLLM_EMBED_MAX_NUM_SEQS", "128")
    monkeypatch.setenv("MAX_CONCURRENT_LLM_CALLS", "120")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS", "2")

    plan = _fit_inference_capacity(max_parallel=2, max_generation_parallel=1)

    assert plan["generation_embedding_share_gateway"] is True
    assert plan["generation_embedding_share_endpoint"] is False
    assert plan["effective"]["generation_concurrency_per_target"] == 120
    assert plan["effective"]["embedding_concurrency_per_target"] == 2
    assert plan["effective"]["generation_capacity_upper_bound"] == 120
    assert plan["effective"]["embedding_capacity_upper_bound"] == 128
    assert plan["effective"]["aggregate_capacity_upper_bound"] == 128


def test_matrix_scheduler_fills_spare_slot_with_embedding_only_target():
    from scripts.run_index_matrix import DATASETS, Target, _next_compatible_target_index

    active = [Target("multihoprag", "prehop", DATASETS["multihoprag"])]
    pending = [
        Target("multihoprag", "hoprag", DATASETS["multihoprag"]),
        Target("multihoprag", "ms_graphrag", DATASETS["multihoprag"]),
        Target("hotpotqa", "naive", DATASETS["hotpotqa"]),
    ]

    assert _next_compatible_target_index(pending, active, max_generation_parallel=1) == 2


def test_matrix_scheduler_waits_when_only_generation_targets_remain():
    from scripts.run_index_matrix import DATASETS, Target, _next_compatible_target_index

    active = [Target("multihoprag", "prehop", DATASETS["multihoprag"])]
    pending = [Target("multihoprag", "hoprag", DATASETS["multihoprag"])]

    assert _next_compatible_target_index(pending, active, max_generation_parallel=1) is None


def test_unjudged_benchmark_cannot_complete(tmp_path):
    summary = {
        "details": [{"llm_judge_score": -1.0}],
        "total_queries": 1,
    }
    with pytest.raises(RuntimeError, match="1 unjudged row"):
        _assert_benchmark_complete(summary, tmp_path / "result.json")


@pytest.mark.asyncio
async def test_json_guard_rejects_truthy_wrong_schema():
    llm = AsyncMock()
    llm.generate_json.return_value = {"unexpected": "shape"}

    with pytest.raises(ValueError, match="missing required field"):
        await generate_json_or_raise(
            llm,
            [{"role": "user", "content": "prompt"}],
            "Q-/Q+ generation",
            required_fields={"q_minus": list},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, message",
    [
        ({"summary": "ok", "q_minus": [123], "q_plus": []}, "non-string/blank"),
        ({"summary": "ok", "q_minus": [""], "q_plus": []}, "non-string/blank"),
        (
            {"summary": "ok", "q_minus": ["q1", "q2", "q3", "q4"], "q_plus": []},
            "more than 3",
        ),
        ({"summary": " ", "q_minus": [], "q_plus": []}, "blank summary"),
    ],
)
async def test_knowledge_mapping_rejects_malformed_inner_schema(payload, message):
    rag = GraphRAG(strategy="prehop")
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json.return_value = payload

    with pytest.raises(ValueError, match=message):
        await rag.extract_hoprag_queries("chunk", "title")


@pytest.mark.asyncio
async def test_generate_json_propagates_transport_error_without_parse_retry():
    llm = VLLMClient()
    llm.generate_response = AsyncMock(side_effect=ConnectionError("offline"))

    with pytest.raises(ConnectionError, match="offline"):
        await llm.generate_json([{"role": "user", "content": "prompt"}])

    llm.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_similarity_scoring_rejects_empty_embedding_instead_of_scoring_zero():
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(side_effect=[[[]], [[1.0, 0.0]]])

    with pytest.raises(ValueError, match="query embedding is missing"):
        await rag._get_query_and_doc_embeddings("query", ["document"])


@pytest.mark.asyncio
async def test_sparse_embedding_rejects_partial_batch():
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0, 0.0]])

    with pytest.raises(ValueError, match="expected 2 non-empty vectors"):
        await rag._embed_sparse_texts(["first", "second"])


@pytest.mark.asyncio
async def test_graph_flush_restores_batch_after_write_failure():
    rag = GraphRAG(strategy="prehop")
    pending = {"data": [{"id": "c1"}], "doc_id": "doc.txt", "doc_title": "Document"}
    rag._pending_batch = [pending]
    rag.retry_query = AsyncMock(side_effect=RuntimeError("neo4j down"))

    with pytest.raises(RuntimeError, match="neo4j down"):
        await rag._flush_graph_batch_unlocked()

    assert rag._pending_batch == [pending]


@pytest.mark.asyncio
async def test_hoprag_official_failure_is_not_replaced_by_vector_search():
    class BrokenRetriever:
        def search_docs(self, _query):
            raise RuntimeError("official traversal failed")

    adapter = object.__new__(HopRAGAdapter)
    adapter._retriever = BrokenRetriever()

    with pytest.raises(RuntimeError, match="official traversal failed"):
        await adapter._run_official_retrieval("query")


def test_hoprag_question_list_validation_rejects_empty_generation():
    from models.hoprag.official_indexer import _validated_question_list

    with pytest.raises(ValueError, match="empty question list"):
        _validated_question_list(([], []))
    with pytest.raises(ValueError, match="blank item"):
        _validated_question_list((["valid?", ""], []))
    assert _validated_question_list((["valid?"], [])) == ["valid?"]


def test_cli_has_no_domain_gate():
    import main

    parser = main._build_parser()
    assert "--domain" not in parser.format_help()


def test_naive_uses_shared_page_scoped_fixed_windows(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "CHUNK_SENTENCES", 2)
    monkeypatch.setattr(RAGConfig, "MIN_CHUNK_SENTENCES", 2)
    title, chunks = NaiveRAG._parse_document(
        "doc.txt",
        "Title: Shared\n--- Page 1 ---\nOne. Two. Three.\n--- Page 2 ---\nFour. Five.",
    )
    assert title == "Shared"
    assert chunks == [
        {"text": "One. Two. Three.", "page": 1, "sent_id": 0},
        {"text": "Four. Five.", "page": 2, "sent_id": 1},
    ]


@pytest.mark.asyncio
async def test_naive_batches_embeddings_across_source_documents(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "EMBEDDING_DIMENSIONS", 2)
    rag = NaiveRAG(strategy="naive", corpus_tag="batch_test")
    rag.vllm.get_embeddings = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    result = AsyncMock()
    session = AsyncMock()
    session.run.return_value = result
    context = AsyncMock()
    context.__aenter__.return_value = session
    context.__aexit__.return_value = False
    rag.neo4j.driver.session = MagicMock(return_value=context)

    indexed = await rag.index_documents(
        [
            ("first.txt", "Title: First\nOne sentence."),
            ("second.txt", "Title: Second\nAnother sentence."),
        ]
    )

    assert indexed == 2
    rag.vllm.get_embeddings.assert_awaited_once()
    assert len(rag.vllm.get_embeddings.await_args.args[0]) == 2
    assert session.run.await_args.kwargs["sources"] == ["first.txt", "second.txt"]


def test_debug_output_is_namespaced_by_run_strategy_and_corpus(monkeypatch):
    monkeypatch.setenv("RAG_RUN_ID", "manual run/one")

    rag = GraphRAG(strategy="prehop", corpus_tag="multi hop")

    assert rag.debug_output_dir.endswith("data/debug/manual_run_one/prehop/multi_hop")


def test_shared_answer_prompt_is_the_prehop_prompt():
    assert GraphRAG._build_answer_prompt("context", "question") == build_answer_prompt("context", "question")


def test_aggregate_numeric_keys_are_union_of_all_rows():
    summary = {
        "details": [
            {"latency": 1.0, "retrieve_ms": 2.0},
            {"latency": 3.0, "error": "boom"},
        ]
    }

    _recompute_aggregates(summary)

    assert summary["avg_retrieve_ms"] == 2.0


def test_message_truncation_does_not_mutate_caller_input():
    client = VLLMClient()
    messages = [{"role": "user", "content": "x" * 10_000}]
    original = messages[0]["content"]

    client._truncate_messages(messages, max_tokens=1200)

    assert messages[0]["content"] == original


@pytest.mark.asyncio
async def test_zero_retry_configuration_fails_clearly(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "LLM_MAX_RETRIES", 0)
    client = VLLMClient()
    with pytest.raises(ValueError, match="at least 1"):
        await client._retry_with_backoff(AsyncMock())

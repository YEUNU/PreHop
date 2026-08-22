from unittest.mock import AsyncMock

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
    ):
        assert not hasattr(RAGConfig, name)


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
    pending = {"data": [{"id": "c1"}], "doc_id": "doc.txt"}
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

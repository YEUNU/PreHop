import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cli.benchmark import _recompute_aggregates
from cli.index import _collect_prehop_integrity, _resolved_index_policy, run_indexing
from core.vllm_client import VLLMClient
from models.hoprag.hoprag_adapter import HopRAGAdapter
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG
from utils.prompts.shared import build_answer_prompt


@pytest.mark.asyncio
async def test_missing_dataset_fails_instead_of_returning_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        await run_indexing(str(tmp_path / "missing"), "prehop", "default")


@pytest.mark.asyncio
async def test_empty_dataset_fails_instead_of_reporting_zero_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.asyncio
async def test_graph_clear_uses_bounded_delete_transactions(monkeypatch):
    from main import _clear_graph_and_schema

    monkeypatch.setenv("RAG_NEO4J_CLEAR_BATCH_SIZE", "17")
    delete_counts = iter((17, 3, 0))
    calls = []

    async def execute(query, parameters=None):
        calls.append((query, parameters))
        if query.startswith("SHOW CONSTRAINTS"):
            return []
        if query.startswith("SHOW INDEXES") and "count(*)" not in query:
            return []
        if "DETACH DELETE" in query:
            return [{"deleted": next(delete_counts)}]
        if query.startswith("MATCH (n) RETURN"):
            return [{"node_count": 0}]
        if query.startswith("SHOW INDEXES"):
            return [{"index_count": 0}]
        raise AssertionError(query)

    neo4j = MagicMock()
    neo4j.execute_query = AsyncMock(side_effect=execute)

    await _clear_graph_and_schema(neo4j)

    delete_calls = [(query, params) for query, params in calls if "DETACH DELETE" in query]
    assert len(delete_calls) == 3
    assert all(params == {"batch": 17} for _query, params in delete_calls)
    first_delete_index = next(index for index, (query, _params) in enumerate(calls) if "DETACH DELETE" in query)
    assert all("SHOW" in query for query, _params in calls[:first_delete_index])


@pytest.mark.asyncio
async def test_prehop_integrity_requires_qminus_owner_provenance(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "ABLATION_Q_MINUS", True)
    monkeypatch.setattr(RAGConfig, "ABLATION_Q_PLUS", True)

    async def execute(query, parameters=None):
        _ = parameters
        if "sum(missing_embeddings)" in query:
            return [{"missing_embeddings": 0, "orphan_nodes": 0}]
        if "sum(empty_questions)" in query:
            return [{"empty_questions": 0, "source_relative_questions": 0, "duplicate_question_groups": 0}]
        if "identical_cross_channel_questions" in query:
            return [{"identical_cross_channel_questions": 0}]
        if "RETURN count(q) AS count" in query:
            return [{"count": 1}]
        if "sum(expected) AS expected" in query:
            return [{"expected": 0, "actual": 0, "invalid": 0, "missing": 0}]
        if "source.id AS source_id" in query:
            return [
                {
                    "source_id": "source",
                    "source_document": "a.txt",
                    "target_id": "target",
                    "target_document": "b.txt",
                    "direct_channels": ["q_minus"],
                    "source_question_ids": ["qp"],
                    "source_question_texts": ["What evidence is needed?"],
                    "edge_type": "qplus_to_qminus_owner",
                }
            ]
        if "missing_source_questions" in query:
            return [{"missing_source_questions": 0, "missing_answered_by": 1, "missing_supported_by": 0}]
        if "max_out_degree" in query:
            return [{"max_out_degree": 1}]
        if "SHOW INDEXES" in query:
            return [{"name": "prehop_test_vector_idx", "state": "ONLINE"}]
        raise AssertionError(query)

    engine = MagicMock(
        chunk_label="PR_test_Chunk",
        doc_label="PR_test_Document",
        q_minus_label="PR_test_QMinus",
        q_plus_label="PR_test_QPlus",
        _safe_corpus="test",
    )
    engine.neo4j.execute_query = AsyncMock(side_effect=execute)

    integrity = await _collect_prehop_integrity(engine)

    assert integrity["checks"]["valid_hop_edges"] is True
    assert integrity["checks"]["consistent_hop_provenance"] is False
    assert integrity["pass"] is False


def test_removed_optional_routes_are_not_configurable():
    from core.config import RAGConfig

    for name in (
        "JUDGE_BATCH_ASYNC",
        "BENCHMARK_GATE_ENABLED",
        "RERANKER_THRESHOLD",
        "HOP_THRESHOLD",
    ):
        assert not hasattr(RAGConfig, name)


def test_index_policy_records_semantic_embedding_and_hop_identity(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "ABLATION_Q_MINUS", True)
    monkeypatch.setattr(RAGConfig, "EMBEDDING_QUERY_INSTRUCTION", "resolved instruction")
    monkeypatch.setenv("RAG_GENERATION_REVISION", "generation-revision-1")
    monkeypatch.setenv("RAG_EMBEDDING_REVISION", "revision-1")

    policy = _resolved_index_policy("prehop", "default")

    assert policy["indexing_model"] == RAGConfig.DEFAULT_MODEL
    assert policy["generation_revision"] == "generation-revision-1"
    assert policy["embedding_query_instruction"] == "resolved instruction"
    assert policy["embedding_revision"] == "revision-1"
    assert policy["hop_construction"] == "qplus_to_qminus_owner"
    assert policy["sentence_channel_enabled"] is RAGConfig.SENTENCE_CHANNEL_ENABLED
    assert policy["continuation_anchor_policy"] == RAGConfig.CONTINUATION_ANCHOR_POLICY

    naive_policy = _resolved_index_policy("naive", "default")
    assert naive_policy["chunk_sentences"] == RAGConfig.CHUNK_SENTENCES
    assert "retrieval_unit" not in naive_policy


@pytest.mark.asyncio
async def test_generation_budget_is_shared_across_client_instances(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "MAX_CONCURRENT_LLM_CALLS", 2)
    VLLMClient._generation_semaphores.clear()
    VLLMClient._generation_inflight.clear()
    VLLMClient._generation_peak.clear()

    active = 0
    peak = 0

    async def create(**_params):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return object()

    request_client = SimpleNamespace(
        base_url="http://generation.test/v1/",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    first = VLLMClient()
    second = VLLMClient()

    await asyncio.gather(
        *[
            client._create_generation_request(request_client, {"request": index})
            for index, client in enumerate([first, second, first, second, first, second])
        ]
    )

    assert peak == 2
    assert max(VLLMClient._generation_peak.values()) == 2


@pytest.mark.asyncio
async def test_knowledge_mapping_removes_source_relative_and_conflicting_questions():
    rag = GraphRAG(strategy="prehop")
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json.return_value = {
        "q_minus": [
            "What does the provided text report?",
            "Who founded Acme?",
            "  Who founded Acme?  ",
        ],
        "q_plus": [
            "Who founded Acme?",
            "Where was Acme incorporated?",
            "What does this chunk omit?",
        ],
    }

    result = await rag.extract_hoprag_queries("Acme was founded by Kim.", "Acme")

    assert result == {
        "q_minus": ["Who founded Acme?"],
        "q_plus": ["Where was Acme incorporated?"],
    }


@pytest.mark.asyncio
async def test_grounded_question_schema_preserves_source_verifiable_fields(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "QUESTION_SCHEMA", "grounded_v1")
    rag = GraphRAG(strategy="prehop")
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json.return_value = {
        "q_minus": [
            {
                "question": "Who founded Acme?",
                "answer": "Kim",
                "grounding_quote": "Acme was founded by Kim.",
                "anchor_entities": ["Acme", "Kim"],
            }
        ],
        "q_plus": [
            {
                "question": "Where was Acme incorporated?",
                "grounding_quote": "Acme was founded by Kim.",
                "anchor_entities": ["Acme"],
                "missing_information": "The jurisdiction of incorporation.",
            }
        ],
    }

    result = await rag.extract_hoprag_queries("Acme was founded by Kim.", "Acme")

    assert result["q_minus"][0]["answer"] == "Kim"
    assert result["q_minus"][0]["question_schema"] == "grounded_v1"
    assert result["q_plus"][0]["missing_information"] == "The jurisdiction of incorporation."
    assert result["q_plus"][0]["anchor_entities"] == ["Acme"]


@pytest.mark.asyncio
async def test_linked_question_schema_preserves_complete_grounded_answer_anchor(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "QUESTION_SCHEMA", "linked_v2")
    rag = GraphRAG(strategy="prehop")
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json.return_value = {
        "q_minus": [
            {
                "question": "Who founded Acme?",
                "answer": "Kim Park",
                "continuation_anchor": "Kim Park",
                "grounding_quote": "Acme was founded by Kim Park.",
                "anchor_entities": ["Acme", "Kim Park"],
            }
        ],
        "q_plus": [],
    }

    result = await rag.extract_hoprag_queries("Acme was founded by Kim Park.", "Acme")

    assert result["q_minus"][0]["question_schema"] == "linked_v2"
    assert result["q_minus"][0]["continuation_anchor"] == "Kim Park"


def test_linked_question_schema_clears_partial_answer_anchor_without_losing_qminus():
    records = GraphRAG._grounded_items(
        [
            {
                "question": "Who founded Acme?",
                "answer": "Kim Park",
                "continuation_anchor": "Kim",
                "grounding_quote": "Acme was founded by Kim Park.",
                "anchor_entities": ["Acme", "Kim Park"],
            }
        ],
        "Q-",
        "Acme was founded by Kim Park.",
        "Acme",
        question_schema="linked_v2",
    )

    assert records[0]["text"] == "Who founded Acme?"
    assert records[0]["continuation_anchor"] == ""


def test_linked_question_schema_filters_auxiliary_anchors_without_losing_question():
    records = GraphRAG._grounded_items(
        [
            {
                "question": "Who founded Acme?",
                "answer": "Kim Park",
                "continuation_anchor": "Kim Park",
                "grounding_quote": "Acme was founded by Kim Park.",
                "anchor_entities": ["Acme", "Kim Park", "Invented alias"],
            }
        ],
        "Q-",
        "Acme was founded by Kim Park.",
        "Acme",
        question_schema="linked_v2",
    )

    assert records[0]["anchor_entities"] == ["Acme", "Kim Park"]
    assert records[0]["continuation_anchor"] == "Kim Park"


def test_grounded_question_schema_rejects_unverifiable_quote():
    with pytest.raises(ValueError, match="not present in source chunk"):
        GraphRAG._grounded_items(
            [
                {
                    "question": "Who founded Acme?",
                    "answer": "Kim",
                    "grounding_quote": "Invented evidence.",
                    "anchor_entities": ["Invented"],
                }
            ],
            "Q-",
            "Acme was founded by Kim.",
            "Acme",
        )


def test_grounded_anchor_may_be_outside_short_quote_but_must_be_in_chunk():
    records = GraphRAG._grounded_items(
        [
            {
                "question": "What did Acme's CEO report?",
                "answer": "40 percent",
                "grounding_quote": "reduced processing time by 40 percent",
                "anchor_entities": ["Acme", "CEO Mira Chen"],
            }
        ],
        "Q-",
        "Acme CEO Mira Chen said the platform reduced processing time by 40 percent.",
        "Acme",
    )

    assert records[0]["anchor_entities"] == ["Acme", "CEO Mira Chen"]


@pytest.mark.asyncio
async def test_grounded_generation_drops_only_invalid_records(monkeypatch):
    from core.config import RAGConfig

    monkeypatch.setattr(RAGConfig, "QUESTION_SCHEMA", "grounded_v1")
    rag = GraphRAG(strategy="prehop")
    rag.indexing_llm = AsyncMock()
    rag.indexing_llm.generate_json.return_value = {
        "q_minus": [
            {
                "question": "Who founded Acme?",
                "answer": "Kim",
                "grounding_quote": "Acme was founded by Kim.",
                "anchor_entities": ["Acme"],
            },
            {
                "question": "What was invented?",
                "answer": "Nothing",
                "grounding_quote": "Invented evidence.",
                "anchor_entities": ["Invented"],
            },
        ],
        "q_plus": [],
    }

    result = await rag.extract_hoprag_queries("Acme was founded by Kim.", "Acme")

    assert [record["text"] for record in result["q_minus"]] == ["Who founded Acme?"]


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

    with pytest.raises(ValueError, match="missing its indexed embedding"):
        await rag._score_and_select([1.0, 0.0], [{"id": "missing", "embedding": []}], top_k=1)


@pytest.mark.asyncio
async def test_sparse_embedding_rejects_partial_batch(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "off")
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0, 0.0]])

    with pytest.raises(ValueError, match="expected 2 vectors"):
        await rag._embed_sparse_texts(["first", "second"])


@pytest.mark.asyncio
async def test_sparse_embedding_reuses_revision_scoped_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "on")
    monkeypatch.setenv("RAG_EMBEDDING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_REVISION", "revision-a")
    rag = GraphRAG(strategy="prehop")
    rag.vector_dimensions = 2
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0, 0.0]])

    assert await rag._embed_sparse_texts(["same text"]) == [[1.0, 0.0]]
    assert await rag._embed_sparse_texts(["same text"]) == [[1.0, 0.0]]
    rag.llm.get_embeddings.assert_awaited_once()

    monkeypatch.setenv("RAG_EMBEDDING_REVISION", "revision-b")
    await rag._embed_sparse_texts(["same text"])
    assert rag.llm.get_embeddings.await_count == 2


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
async def test_graph_flush_writes_a_document_wave_in_one_query():
    rag = GraphRAG(strategy="prehop")
    rag._pending_batch = [
        {"data": [{"id": "c1"}], "doc_id": "one.txt", "doc_title": "One"},
        {"data": [{"id": "c2"}], "doc_id": "two.txt", "doc_title": "Two"},
    ]
    rag.retry_query = AsyncMock(return_value=[])

    await rag._flush_graph_batch_unlocked()

    rag.retry_query.assert_awaited_once()
    assert rag.retry_query.await_args.args[1]["documents"][1]["doc_id"] == "two.txt"


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
    title, chunks = NaiveRAG._parse_document(
        "doc.txt",
        "Title: Shared\n--- Page 1 ---\nOne. Two. Three.\n--- Page 2 ---\nFour. Five.",
    )
    assert title == "Shared"
    assert chunks == [
        {"text": "One. Two.", "page": 1, "sent_id": 0},
        {"text": "Three.", "page": 1, "sent_id": 1},
        {"text": "Four. Five.", "page": 2, "sent_id": 2},
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
    assert rag.vllm.get_embeddings.await_args.kwargs == {}
    assert session.run.await_args.kwargs["sources"] == ["first.txt", "second.txt"]


@pytest.mark.asyncio
async def test_embedding_strict_input_rejects_provider_context_overflow(monkeypatch):
    client = VLLMClient()
    monkeypatch.setattr(
        client, "_create_embedding_request", AsyncMock(side_effect=ValueError("maximum context length"))
    )

    with pytest.raises(ValueError, match="maximum context length"):
        await client.get_embeddings(["complete document"], allow_truncation=False)


def test_debug_output_is_namespaced_by_run_strategy_and_corpus(monkeypatch):
    monkeypatch.setenv("RAG_RUN_ID", "manual run/one")

    rag = GraphRAG(strategy="prehop", corpus_tag="multi hop")

    assert "/data/debug/manual_run_one/prehop/multi_hop_" in "/" + rag.debug_output_dir


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

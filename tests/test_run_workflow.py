"""Prehop query rewriting, one retrieval pass, and single synthesis."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("words", [1, 32, 33, 100])
async def test_original_question_has_one_search_and_no_conversion(monkeypatch, words):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    rag = GraphRAG(strategy="prehop")
    rag.llm.generate_json = AsyncMock(side_effect=AssertionError("Unexpected question conversion"))
    rag.llm.generate_response = AsyncMock(return_value="Answer")
    rag.graph_search = AsyncMock(return_value=("Evidence", [{"id":"x","source":"s","title":"T","text":"Evidence","sent_id":0}], {}))
    original = " ".join(["word"] * words)
    _, _, trace = await rag.run_workflow(original + " [Benchmark Output Format] answer briefly")
    rag.graph_search.assert_awaited_once_with(entities=[original], depth=1, top_k=RAGConfig.DEFAULT_TOP_K)
    rag.llm.generate_response.assert_awaited_once()
    rag.llm.generate_json.assert_not_awaited()
    assert not hasattr(rag, "_rewrite_query_roles") and not hasattr(rag, "_refine_query_roles")
    assert not any("rewrite" in row["step"] for row in trace)


from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG

# ---------------------------------------------------------------------------
# Helpers (no mocking needed — pure functions / classmethods)
# ---------------------------------------------------------------------------


def test_ensure_answer_prefix_adds_marker_when_missing():
    rag = GraphRAG(strategy="prehop")
    out = rag._ensure_answer_prefix("Revenue was $394B in FY2022.")
    assert out.startswith("@@ANSWER:")
    assert "Revenue was $394B" in out


def test_ensure_answer_prefix_is_noop_when_marker_present():
    rag = GraphRAG(strategy="prehop")
    raw = "@@ANSWER: Revenue was $394B in FY2022."
    assert rag._ensure_answer_prefix(raw) == raw


def test_ensure_answer_prefix_handles_empty_and_none():
    rag = GraphRAG(strategy="prehop")
    assert rag._ensure_answer_prefix(None).startswith("@@ANSWER:")
    assert rag._ensure_answer_prefix("").startswith("@@ANSWER:")


def test_strip_format_instruction_drops_benchmark_suffix():
    rag = GraphRAG(strategy="prehop")
    q = "What was Apple's FY2022 revenue? [Benchmark Output Format] respond with..."
    assert rag._strip_format_instruction(q) == "What was Apple's FY2022 revenue?"


def test_strip_format_instruction_passthrough_without_marker():
    rag = GraphRAG(strategy="prehop")
    q = "What was Apple's FY2022 revenue?"
    assert rag._strip_format_instruction(q) == q


def test_build_unique_sources_dedups_by_doc_page_sent():
    rag = GraphRAG(strategy="prehop")
    rows = [
        {"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "..."},
        {"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "..."},  # dup
        {"doc": "AAPL_10K", "page": 41, "sent_id": 4, "text": "..."},  # different sent
        {"title": "AAPL_10K", "page": 42, "sent_id": 3, "text": "..."},  # different page
    ]
    out = rag._build_unique_sources(rows)
    assert len(out) == 3
    keys = {(s["doc"], s["page"], s["sent_id"]) for s in out}
    assert keys == {("AAPL_10K", 41, 3), ("AAPL_10K", 41, 4), ("AAPL_10K", 42, 3)}


def test_build_unique_sources_uses_unknown_when_doc_missing():
    rag = GraphRAG(strategy="prehop")
    out = rag._build_unique_sources([{"page": 1, "sent_id": 0, "text": "x"}])
    assert out[0]["doc"] == "Unknown"


def test_build_unique_sources_uses_source_identity_before_display_title():
    rag = GraphRAG(strategy="prehop")
    rows = [
        {"title": "Repeated", "source": "hotpotqa_first.txt", "page": 1, "sent_id": 0, "text": "first"},
        {"title": "Repeated", "source": "hotpotqa_second.txt", "page": 1, "sent_id": 0, "text": "second"},
        {"title": "Repeated", "source": "hotpotqa_first.txt", "page": 1, "sent_id": 0, "text": "duplicate"},
    ]

    out = rag._build_unique_sources(rows)

    assert [source["source"] for source in out] == ["hotpotqa_first.txt", "hotpotqa_second.txt"]


def test_build_unique_sources_preserves_retrieval_provenance():
    paths = [{"kind": "hop", "source_chunk_id": "seed", "depth": 1, "edge_rank": 0}]
    out = GraphRAG._build_unique_sources(
        [
            {
                "id": "target",
                "title": "Doc",
                "source": "doc.txt",
                "sent_id": 2,
                "text": "evidence",
                "retrieval_paths": paths,
            }
        ]
    )

    assert out[0]["chunk_id"] == "target"
    assert out[0]["retrieval_paths"] == paths


def test_build_answer_prompt_contains_context_and_query():
    prompt = GraphRAG._build_answer_prompt("CTX_BLOCK", "QUESTION_TEXT")
    assert "CTX_BLOCK" in prompt
    assert "QUESTION_TEXT" in prompt
    # Dataset-neutral multi-hop synthesis with conservative abstention.
    assert "only the provided context" in prompt
    assert "connect the intermediate entities and relationships" in prompt
    assert "do not show reasoning" in prompt
    assert "lacks a required link" in prompt
    assert "do not refuse merely because multiple passages must be combined" in prompt


# ---------------------------------------------------------------------------
# run_workflow — full path with mocked retrieve + LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_workflow_returns_answer_sources_trace_tuple():
    nodes = [{"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "Revenue $394B"}]
    rag, p = _make_rag_with_mocks(nodes=nodes)
    try:
        answer, sources, trace = await rag.run_workflow("What was Apple's FY2022 revenue?")
        assert answer.startswith("@@ANSWER:")
        assert "Apple's FY2022 revenue was $394B" in answer
        assert sources == [{"doc": "AAPL_10K", "page": 41, "sent_id": 3, "text": "Revenue $394B"}]
        assert isinstance(trace, list) and len(trace) == 2
        assert trace[0]["step"] == "retrieve"
        assert trace[0]["output"]["retrieved_chunks"] == 1
        assert trace[0]["output"]["retrieved_sources"] == 1
        assert trace[1]["step"] == "synthesis"
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_uses_graph_search_when_depth_positive():
    rag, p = _make_rag_with_mocks(graph_depth=1)
    try:
        await rag.run_workflow("any query")
        rag.graph_search.assert_awaited_once()
        rag.retrieve.assert_not_awaited()
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_falls_back_to_retrieve_when_depth_zero():
    rag, p = _make_rag_with_mocks(graph_depth=0)
    try:
        await rag.run_workflow("any query")
        rag.retrieve.assert_awaited_once()
        rag.graph_search.assert_not_awaited()
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_abstains_on_empty_context():
    rag, p = _make_rag_with_mocks(context="", nodes=[])
    try:
        answer, sources, trace = await rag.run_workflow("query nobody can answer")
        assert "Insufficient evidence" in answer
        assert answer.startswith("@@ANSWER:")
        # Synthesis step should record the empty-context reason and the LLM
        # should NOT have been called.
        rag.llm.generate_response.assert_not_awaited()
        assert trace[1]["output"]["reason"] == "empty_context"
        assert sources == []
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_run_workflow_strips_benchmark_format_marker_before_retrieving():
    nodes = [{"title": "AAPL_10K", "page": 41, "sent_id": 3, "text": "Revenue $394B"}]
    rag, p = _make_rag_with_mocks(nodes=nodes, graph_depth=1)
    try:
        await rag.run_workflow("What was Apple's FY2022 revenue? [Benchmark Output Format] foo")
        # graph_search receives the stripped query, not the suffix-tainted one.
        call_kwargs = rag.graph_search.await_args.kwargs
        assert "[Benchmark Output Format]" not in (call_kwargs.get("user_query") or "")
        assert "[Benchmark Output Format]" not in (call_kwargs.get("entities") or [""])[0]
        synthesis_prompt = rag.llm.generate_response.await_args.args[0][0]["content"]
        assert "[Benchmark Output Format]" not in synthesis_prompt
    finally:
        p.stop()


@pytest.mark.asyncio
async def test_naive_run_workflow_uses_shared_default_top_k():
    from core.config import RAGConfig
    from models.naive.naive_rag import NaiveRAG

    rag = NaiveRAG(strategy="naive")
    node = {"title": "Doc", "text": "context", "source": "doc.txt", "page": 0, "sent_id": 0}
    rag._retrieve_nodes = AsyncMock(return_value=[node])
    rag.vllm = MagicMock()
    rag.vllm._count_tokens.return_value = 10
    rag.vllm.generate_response = AsyncMock(return_value="answer")

    answer, _sources, trace = await rag.run_workflow("question")

    rag._retrieve_nodes.assert_awaited_once_with("question", top_k=RAGConfig.DEFAULT_TOP_K)
    assert answer == "@@ANSWER: answer"
    assert trace[0]["output"] == answer


@pytest.mark.asyncio
async def test_hoprag_run_workflow_marks_answer_boundary():
    from models.hoprag.hoprag_adapter import HopRAGAdapter

    adapter = object.__new__(HopRAGAdapter)
    adapter.top_k = 8
    node = {"title": "Doc", "text": "context", "source": "doc.txt", "page": 0, "sent_id": 0}
    adapter.retrieve = AsyncMock(return_value=("context", [node]))
    adapter._pipeline = MagicMock()
    adapter._pipeline.rag.return_value = ("answer", ["context"], [1.0])
    adapter._lookup_nodes_by_text = AsyncMock(return_value=[node])

    answer, _sources, trace = await adapter.run_workflow("question")

    assert answer == "@@ANSWER: answer"
    assert trace[0]["output"] == "answer"
    adapter._pipeline.rag.assert_called_once_with("question")


def test_naive_context_budget_keeps_complete_chunks_in_rank_order(monkeypatch):
    from core.config import RAGConfig
    from models.naive.naive_rag import NaiveRAG

    monkeypatch.setattr(RAGConfig, "MAX_CONTEXT_LENGTH", 100)
    monkeypatch.setattr(RAGConfig, "SYNTHESIS_MAX_OUTPUT_TOKENS", 10)
    rag = NaiveRAG(strategy="naive")
    rag.vllm = MagicMock()
    rag.vllm._count_tokens.side_effect = [50, 95]
    nodes = [
        {"title": "First", "text": "first body", "sent_id": 0},
        {"title": "Second", "text": "second body", "sent_id": 1},
    ]

    context, accepted = rag._fit_ranked_context(nodes, "question")

    assert accepted == [nodes[0]]
    assert "first body" in context
    assert "second body" not in context


def test_hoprag_adapter_keeps_official_top_k():
    from models.hoprag.hoprag_adapter import OFFICIAL_HOPRAG_TOP_K, HopRAGAdapter

    assert OFFICIAL_HOPRAG_TOP_K == 8
    assert HopRAGAdapter.__init__.__defaults__[2] == OFFICIAL_HOPRAG_TOP_K


def _make_rag_with_mocks(
    *,
    nodes=None,
    context="Some retrieved context.",
    llm_answer="Apple's FY2022 revenue was $394B.",
    graph_depth=1,
):
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value=llm_answer)
    rag.graph_search = AsyncMock(return_value=(context, nodes or [], {"retrieve_ms": 0.0, "traversal_ms": 0.0}))
    rag.retrieve = AsyncMock(return_value=(context, nodes or []))
    # Pin unrelated policy choices so these tests isolate the workflow branch.
    # Role rewriting has dedicated tests above.
    rag_patch = patch.multiple(
        "core.config.RAGConfig",
        GRAPH_HOP_DEPTH=graph_depth,
    )
    rag_patch.start()
    return rag, rag_patch

"""Tests for the retrieval path on GraphRAG.run_workflow().

The default Prehop query path is retrieve -> single synthesis call. A discrete
evidence-conditioned rewrite ablation adds one preview stage, while an
iterative ablation continues only while new questions select new chunks.
These tests pin the public contract of run_workflow() and the small helpers it
composes (_ensure_answer_prefix, _strip_format_instruction,
_build_unique_sources, _build_answer_prompt).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        {"title": "Repeated", "source": "musique_first.txt", "page": 1, "sent_id": 0, "text": "first"},
        {"title": "Repeated", "source": "musique_second.txt", "page": 1, "sent_id": 0, "text": "second"},
        {"title": "Repeated", "source": "musique_first.txt", "page": 1, "sent_id": 0, "text": "duplicate"},
    ]

    out = rag._build_unique_sources(rows)

    assert [source["source"] for source in out] == ["musique_first.txt", "musique_second.txt"]


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


def test_role_query_validation_deduplicates_without_silently_truncating():
    payload = {
        "q_minus": ["What did A report?", "  What did A report?  ", "What did B report?"],
        "q_plus": ["Were the reports consistent?"],
    }

    assert GraphRAG._validate_role_queries(payload) == {
        "q_minus": ["What did A report?", "What did B report?"],
        "q_plus": ["Were the reports consistent?"],
    }


def test_role_query_validation_allows_empty_roles_and_bounds_excessive_roles():
    assert GraphRAG._validate_role_queries({"q_minus": ["direct?"], "q_plus": []}) == {
        "q_minus": ["direct?"],
        "q_plus": [],
    }
    assert GraphRAG._validate_role_queries({"q_minus": [], "q_plus": ["bridge?"]}) == {
        "q_minus": [],
        "q_plus": ["bridge?"],
    }
    assert GraphRAG._validate_role_queries({"q_minus": ["one?"], "q_plus": ["one?", "two?", "three?", "four?"]}) == {
        "q_minus": ["one?"],
        "q_plus": ["one?", "two?", "three?"],
    }


@pytest.mark.asyncio
async def test_role_aligned_rewrite_is_schema_constrained(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr("core.config.RAGConfig.QUERY_REWRITE_VARIANT", "role_aligned")
    rag.llm.generate_json = AsyncMock(
        return_value={"q_minus": ["What did A report?", "What did B report?"], "q_plus": ["Were they consistent?"]}
    )

    rewritten = await rag._rewrite_query_roles("Was report A consistent with report B?")

    assert rewritten == {
        "q_minus": ["What did A report?", "What did B report?"],
        "q_plus": ["Were they consistent?"],
    }
    assert rag.llm.generate_json.await_args.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_role_aligned_rewrite_skips_queries_above_word_limit(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr("core.config.RAGConfig.QUERY_REWRITE_VARIANT", "role_aligned")
    monkeypatch.setattr("core.config.RAGConfig.QUERY_REWRITE_MAX_WORDS", 3)
    rag.llm.generate_json = AsyncMock()

    rewritten = await rag._rewrite_query_roles("one two three four")

    assert rewritten is None
    rag.llm.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_evidence_conditioned_rewrite_is_grounded_in_preview(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    rag.llm.generate_json = AsyncMock(return_value={"q_minus": ["Where is British Aerospace based?"], "q_plus": []})

    rewritten = await rag._refine_query_roles(
        "Which company follows the maker of ALARM?",
        "British Aerospace offered ALARM.",
        ["Which company makes ALARM?"],
    )

    assert rewritten == {
        "q_minus": ["Where is British Aerospace based?"],
        "q_plus": [],
    }
    prompt = rag.llm.generate_json.await_args.args[0][0]["content"]
    assert "British Aerospace offered ALARM." in prompt
    assert "Which company makes ALARM?" in prompt
    assert rag.llm.generate_json.await_args.kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# run_workflow — full path with mocked retrieve + LLM
# ---------------------------------------------------------------------------


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
        QUERY_REWRITE_VARIANT="none",
    )
    rag_patch.start()
    return rag, rag_patch


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
async def test_evidence_conditioned_variant_previews_then_searches_new_role_queries(monkeypatch):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    monkeypatch.setattr(RAGConfig, "QUERY_REWRITE_VARIANT", "role_aligned_evidence")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value="final answer")
    rag._rewrite_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["Which company makes ALARM?"], "q_plus": []}
    )
    rag._refine_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["Which company does British Aerospace follow?"], "q_plus": []}
    )
    preview_node = {
        "id": "preview",
        "title": "ALARM",
        "sent_id": 0,
        "text": "British Aerospace offered ALARM.",
        "embedding": [1.0],
    }
    final_node = {
        "id": "final",
        "title": "Answer",
        "sent_id": 0,
        "text": "Final evidence.",
        "embedding": [1.0],
    }
    rag.graph_search = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("preview context", [preview_node], {"retrieve_ms": 1.0, "traversal_ms": 2.0}),
            ("final context", [final_node], {"retrieve_ms": 3.0, "traversal_ms": 4.0}),
        ]
    )

    _answer, _sources, trace = await rag.run_workflow("Which company follows the maker of ALARM?")

    assert rag.graph_search.await_count == 2
    final_queries = rag.graph_search.await_args_list[1].kwargs["channel_queries"]
    assert final_queries["q_minus"] == [
        "Which company makes ALARM?",
        "Which company does British Aerospace follow?",
    ]
    retrieve_trace = next(item for item in trace if item["step"] == "retrieve")
    assert retrieve_trace["retrieve_ms"] == 4.0
    assert retrieve_trace["traversal_ms"] == 6.0
    assert any(item["step"] == "evidence_query_rewrite" for item in trace)


@pytest.mark.asyncio
async def test_evidence_conditioned_variant_reuses_preview_when_no_new_question(monkeypatch):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    monkeypatch.setattr(RAGConfig, "QUERY_REWRITE_VARIANT", "role_aligned_evidence")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value="answer")
    rag._rewrite_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["direct?"], "q_plus": []}
    )
    rag._refine_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["direct?"], "q_plus": []}
    )
    node = {
        "id": "one",
        "title": "Doc",
        "sent_id": 0,
        "text": "Evidence.",
        "embedding": [1.0],
    }
    rag.graph_search = AsyncMock(  # type: ignore[method-assign]
        return_value=("context", [node], {"retrieve_ms": 1.0, "traversal_ms": 2.0})
    )

    await rag.run_workflow("question")

    rag.graph_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_ranking_runs_once_after_evidence_refinement(monkeypatch):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    monkeypatch.setattr(RAGConfig, "QUERY_REWRITE_VARIANT", "role_aligned_evidence_iterative")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "role_body_list_ranking")
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value="answer")
    rag._rewrite_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["direct?"], "q_plus": []}
    )
    rag._refine_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["direct?"], "q_plus": []}
    )
    preview = {
        "id": "preview",
        "title": "Preview",
        "sent_id": 0,
        "text": "preview evidence",
    }
    final = {"id": "final", "title": "Final", "sent_id": 0, "text": "final evidence"}
    rag.graph_search = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("preview context", [preview], {"retrieve_ms": 1.0, "traversal_ms": 2.0}),
            ("final context", [final], {"retrieve_ms": 3.0, "traversal_ms": 4.0}),
        ]
    )

    _answer, sources, trace = await rag.run_workflow("question")

    assert rag.graph_search.await_count == 2
    assert rag.graph_search.await_args_list[0].kwargs["selection_variant"] == "role_body_rounds"
    assert "selection_variant" not in rag.graph_search.await_args_list[1].kwargs
    assert sources[0]["doc"] == "Final"
    retrieve_trace = next(item for item in trace if item["step"] == "retrieve")
    assert retrieve_trace["retrieve_ms"] == 4.0
    assert retrieve_trace["traversal_ms"] == 6.0


@pytest.mark.asyncio
async def test_iterative_evidence_variant_continues_until_no_new_question(monkeypatch):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    monkeypatch.setattr(RAGConfig, "QUERY_REWRITE_VARIANT", "role_aligned_evidence_iterative")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value="final answer")
    rag._rewrite_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["first dependency?"], "q_plus": []}
    )
    rag._refine_query_roles = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {"q_minus": ["second dependency?"], "q_plus": []},
            {"q_minus": [], "q_plus": ["third dependency?"]},
            {"q_minus": [], "q_plus": ["third dependency?"]},
        ]
    )
    nodes = [
        {"id": identity, "title": identity, "sent_id": 0, "text": f"evidence {identity}", "embedding": [1.0]}
        for identity in ("first", "second", "third")
    ]
    rag.graph_search = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("first context", [nodes[0]], {"retrieve_ms": 1.0, "traversal_ms": 2.0}),
            ("second context", [nodes[1]], {"retrieve_ms": 3.0, "traversal_ms": 4.0}),
            ("third context", [nodes[2]], {"retrieve_ms": 5.0, "traversal_ms": 6.0}),
        ]
    )

    _answer, _sources, trace = await rag.run_workflow("multi-step question")

    assert rag.graph_search.await_count == 3
    assert rag._refine_query_roles.await_count == 3
    final_queries = rag.graph_search.await_args_list[-1].kwargs["channel_queries"]
    assert final_queries["q_minus"] == ["first dependency?", "second dependency?"]
    assert final_queries["q_plus"] == ["third dependency?"]
    refinements = [item for item in trace if item["step"] == "evidence_query_rewrite"]
    assert len(refinements) == 3
    retrieve_trace = next(item for item in trace if item["step"] == "retrieve")
    assert retrieve_trace["retrieve_ms"] == 9.0
    assert retrieve_trace["traversal_ms"] == 12.0


@pytest.mark.asyncio
async def test_iterative_evidence_variant_stops_when_selected_chunks_do_not_change(monkeypatch):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    monkeypatch.setattr(RAGConfig, "QUERY_REWRITE_VARIANT", "role_aligned_evidence_iterative")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value="final answer")
    rag._rewrite_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["first dependency?"], "q_plus": []}
    )
    rag._refine_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["second dependency?"], "q_plus": []}
    )
    node = {"id": "same", "title": "Doc", "sent_id": 0, "text": "evidence", "embedding": [1.0]}
    rag.graph_search = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("first context", [node], {"retrieve_ms": 1.0, "traversal_ms": 2.0}),
            ("same context", [node], {"retrieve_ms": 3.0, "traversal_ms": 4.0}),
        ]
    )

    await rag.run_workflow("multi-step question")

    assert rag.graph_search.await_count == 2
    rag._refine_query_roles.assert_awaited_once()


@pytest.mark.asyncio
async def test_iterative_evidence_variant_honors_configured_round_cap(monkeypatch):
    monkeypatch.setattr(RAGConfig, "GRAPH_HOP_DEPTH", 1)
    monkeypatch.setattr(RAGConfig, "QUERY_REWRITE_VARIANT", "role_aligned_evidence_iterative")
    monkeypatch.setattr(RAGConfig, "QUERY_REFINEMENT_MAX_ROUNDS", 1)
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm = MagicMock()
    rag.llm.generate_response = AsyncMock(return_value="final answer")
    rag._rewrite_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["first dependency?"], "q_plus": []}
    )
    rag._refine_query_roles = AsyncMock(  # type: ignore[method-assign]
        return_value={"q_minus": ["second dependency?"], "q_plus": []}
    )
    first = {"id": "first", "title": "First", "sent_id": 0, "text": "first", "embedding": [1.0]}
    second = {"id": "second", "title": "Second", "sent_id": 0, "text": "second", "embedding": [1.0]}
    rag.graph_search = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("first context", [first], {"retrieve_ms": 1.0, "traversal_ms": 2.0}),
            ("second context", [second], {"retrieve_ms": 3.0, "traversal_ms": 4.0}),
        ]
    )

    _answer, _sources, trace = await rag.run_workflow("multi-step question")

    assert rag.graph_search.await_count == 2
    rag._refine_query_roles.assert_awaited_once()
    rewrite_trace = next(item for item in trace if item["step"] == "query_rewrite")
    assert rewrite_trace["refinement_rounds"] == 1
    assert rewrite_trace["refinement_stop_reason"] == "configured_round_cap"


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
    adapter.top_k = 20
    node = {"title": "Doc", "text": "context", "source": "doc.txt", "page": 0, "sent_id": 0}
    adapter.retrieve = AsyncMock(return_value=("context", [node]))
    adapter.llm = MagicMock()
    adapter.llm.generate_response = AsyncMock(return_value="answer")

    answer, _sources, trace = await adapter.run_workflow("question")

    assert answer == "@@ANSWER: answer"
    assert trace[0]["output"] == answer


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

    assert OFFICIAL_HOPRAG_TOP_K == 20
    assert HopRAGAdapter.__init__.__defaults__[2] == OFFICIAL_HOPRAG_TOP_K

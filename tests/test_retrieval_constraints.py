from unittest.mock import AsyncMock

import pytest

from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG


def test_retrieval_has_no_dataset_specific_metadata_gate():
    rag = GraphRAG(strategy="prehop")
    assert not hasattr(rag, "_extract_query_metadata")
    assert not hasattr(rag, "_apply_retrieval_calibration")
    assert not hasattr(RAGConfig, "COMPANY_ANCHORING")
    assert not hasattr(RAGConfig, "RERANKER_THRESHOLD")


@pytest.mark.asyncio
async def test_similarity_ordering_has_no_score_gate():
    rag = GraphRAG(strategy="prehop")
    rag._embedding_similarity_scores = AsyncMock(return_value=[-0.8, -0.2])  # type: ignore[method-assign]

    selected, ordered = await rag._score_and_select(
        "query",
        [{"id": "low", "text": "a"}, {"id": "high", "text": "b"}],
        top_k=2,
    )

    assert [node["id"] for node in selected] == ["high", "low"]
    assert selected == ordered


@pytest.mark.asyncio
async def test_retrieve_always_runs_q_plus_stage_in_full_mode():
    rag = GraphRAG(strategy="prehop")

    stage1_node = {
        "id": "n1",
        "title": "AMD_2022_10K",
        "sent_id": 1,
        "page": 10,
        "text": "Operating cash flow was 3.6 billion.",
        "rrf_score": 1.0,
    }
    stage2_node = {
        "id": "n2",
        "title": "AMD_2022_10K",
        "sent_id": 2,
        "page": 11,
        "text": "Capital expenditures were 1.1 billion in FY2022.",
        "rrf_score": 0.9,
    }

    async def fake_candidates(_q_text: str, limit: int, channel: str = "body"):
        if channel == "q_minus":
            return [dict(stage1_node)]
        if channel == "q_plus":
            return [dict(stage2_node)]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    # Uniform embeddings give both candidates the same cosine score; Q+ is
    # still included because the full method always executes both stages.
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts, encoding_type="document": [[1.0, 0.0] for _ in texts])

    _, nodes = await rag.retrieve(
        "What was AMD FY2022 free cash flow?",
        top_k=2,
    )

    ids = {n.get("id") for n in nodes}
    assert "n1" in ids
    assert "n2" in ids


@pytest.mark.asyncio
async def test_build_graph_stores_generated_q_plus_without_heuristic_filter():
    rag = GraphRAG(strategy="prehop")
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts: [[0.1] for _ in texts])

    # Storage keeps generated non-empty Q+ values without applying a
    # domain/keyword heuristic after the model's schema-validated response.
    knowledge = {
        "chunks": [
            {
                "text": "Consolidated Statements of Cash Flows for fiscal year 2022.",
                "title": "AMD_2022_10K",
                "sent_id": 0,
                "page": 1,
                "q_minus": ["For AMD FY2022, what was operating cash flow?"],
                "q_plus": [
                    "For FY2022, what happened?",
                    "For AMD FY2022 cash flow statement, what was operating cash flow?",
                    "For FY2022, what happened?",
                    "",
                ],
                "summary": "Cash flow statement summary.",
            }
        ]
    }

    await rag.build_graph(knowledge, source="unit_test", document_filename="AMD_2022_10K.txt")

    assert rag._pending_batch, "Expected build_graph to enqueue at least one batch item."
    payload = rag._pending_batch[-1]["data"][0]
    assert payload["q_plus_text"] == (
        "For FY2022, what happened? For AMD FY2022 cash flow statement, what was operating cash flow?"
    )

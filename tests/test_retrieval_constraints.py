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


def test_hop_bridge_provenance_is_normalized_for_query_scoring():
    rag = GraphRAG(strategy="prehop")

    assert rag._normalize_bridge_text(["  Which manual?  ", "Which manual?", "What year?\n"]) == (
        "Which manual?\nWhat year?"
    )
    assert rag._normalize_bridge_text(None) == ""


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
async def test_hop_candidate_scoring_preserves_bridge_question_semantics():
    rag = GraphRAG(strategy="prehop")
    rag._embedding_similarity_scores = AsyncMock(  # type: ignore[method-assign]
        side_effect=[[0.6], [0.8]]
    )
    candidate = {
        "id": "target",
        "title": "2023 manual",
        "text": "The target body.",
        "bridge_text": "Bridge questions: Which 2023 manual superseded the 2022 port?",
    }

    selected, _ = await rag._score_and_select("Which manual superseded it?", [candidate], top_k=1)

    assert rag._embedding_similarity_scores.await_args_list[1].args[1] == [candidate["bridge_text"]]
    assert selected[0]["similarity_score"] == 0.6
    assert selected[0]["bridge_similarity_score"] == 0.8
    assert selected[0]["final_score"] == pytest.approx(0.7)


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
    rag.vector_dimensions = 1
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts, encoding_type="document": [[0.1] for _ in texts])

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
    assert [question["text"] for question in payload["q_plus"]] == [
        "For FY2022, what happened?",
        "For AMD FY2022 cash flow statement, what was operating cash flow?",
    ]
    assert len({question["id"] for question in payload["q_plus"]}) == 2
    assert all(question["query_embedding"] == [0.1] for question in payload["q_plus"])


@pytest.mark.asyncio
async def test_qplus_only_similarity_cannot_create_hop_answer(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr(RAGConfig, "HOP_LINK_LIMIT", 5)

    async def fake_candidates(wave, channel):
        if channel == "q_plus":
            return [
                {
                    "source_chunk_id": wave[0]["id"],
                    "source_question_id": wave[0]["questions"][0]["id"],
                    "target_id": "same-need-only",
                    "target_question_id": "qp-target",
                    "score": 0.99,
                }
            ]
        return []

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "manual-2022.txt",
                "questions": [{"id": "qp-source", "query_embedding": [1.0]}],
            }
        ],
    )

    assert edges == []


@pytest.mark.asyncio
async def test_hop_ann_pool_is_sized_per_source_not_by_corpus_max(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr(RAGConfig, "HOP_ANN_POOL", 50)
    monkeypatch.setattr(RAGConfig, "HOP_CANDIDATE_LIMIT", 15)
    rag.retry_query = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._find_hop_candidates_batch(
        [
            {
                "id": "short-source-chunk",
                "source": "short.txt",
                "ann_pools": {"body": 22},
                "questions": [{"id": "q-short", "query_embedding": [1.0]}],
            },
            {
                "id": "long-source-chunk",
                "source": "long.txt",
                "ann_pools": {"body": 180},
                "questions": [{"id": "q-long", "query_embedding": [1.0]}],
            },
        ],
        "body",
    )

    parameters = rag.retry_query.await_args.args[1]
    assert [item["ann_pool"] for item in parameters["source_questions"]] == [50, 180]


@pytest.mark.asyncio
async def test_qplus_to_qminus_preserves_same_need_as_support(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr(RAGConfig, "HOP_LINK_LIMIT", 5)

    async def fake_candidates(wave, channel):
        source = {
            "source_chunk_id": wave[0]["id"],
            "source_question_id": wave[0]["questions"][0]["id"],
        }
        if channel == "q_minus":
            return [{**source, "target_id": "manual-2023", "target_question_id": "qm-2023", "score": 0.8}]
        if channel == "q_plus":
            return [{**source, "target_id": "manual-2023", "target_question_id": "qp-2023", "score": 0.9}]
        return []

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "manual-2022.txt",
                "questions": [{"id": "qp-source", "query_embedding": [1.0]}],
            }
        ],
    )

    assert len(edges) == 1
    assert edges[0]["tgt_id"] == "manual-2023"
    assert edges[0]["direct_channels"] == ["q_minus"]
    assert edges[0]["same_need_match"]["target_question_id"] == "qp-2023"


@pytest.mark.asyncio
async def test_each_individual_qplus_keeps_one_direction_before_global_fill(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr(RAGConfig, "HOP_LINK_LIMIT", 2)

    async def fake_candidates(wave, channel):
        if channel != "body":
            return []
        return [
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-first",
                "target_id": "first-best",
                "target_question_id": None,
                "score": 0.99,
            },
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-first",
                "target_id": "first-second",
                "target_question_id": None,
                "score": 0.98,
            },
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-second",
                "target_id": "second-best",
                "target_question_id": None,
                "score": 0.70,
            },
        ]

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "source.txt",
                "questions": [
                    {"id": "q-first", "query_embedding": [1.0]},
                    {"id": "q-second", "query_embedding": [1.0]},
                ],
            }
        ],
    )

    assert {edge["tgt_id"] for edge in edges} == {"first-best", "second-best"}

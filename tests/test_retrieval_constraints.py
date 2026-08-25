from pathlib import Path
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


def test_primary_source_selection_is_global_cosine_order():
    assert RAGConfig.SOURCE_SELECTION_VARIANT == "global"


@pytest.mark.asyncio
async def test_similarity_ordering_has_no_score_gate():
    rag = GraphRAG(strategy="prehop")

    selected, ordered = await rag._score_and_select(
        [1.0, 0.0],
        [
            {"id": "low", "text": "a", "embedding": [-0.8, 0.6]},
            {"id": "high", "text": "b", "embedding": [-0.2, 0.98]},
        ],
        top_k=2,
    )

    assert [node["id"] for node in selected] == ["high", "low"]
    assert selected == ordered


@pytest.mark.asyncio
async def test_final_order_preserves_representation_rank_without_fitted_weight():
    rag = GraphRAG(strategy="prehop")

    selected, _ = await rag._score_and_select(
        [1.0, 0.0],
        [
            {"id": "semantic-first", "embedding": [1.0, 0.0], "representation_score": 0.5},
            {"id": "representation-first", "embedding": [0.8, 0.6], "representation_score": 1.0},
            {"id": "middle", "embedding": [0.9, 0.43589], "representation_score": 0.75},
        ],
        top_k=3,
    )

    assert [node["id"] for node in selected] == ["semantic-first", "representation-first", "middle"]
    assert selected[0]["rank_fusion_score"] == pytest.approx(1.0 + 1.0 / 3.0)
    assert selected[1]["rank_fusion_score"] == pytest.approx(1.0 / 3.0 + 1.0)


@pytest.mark.asyncio
async def test_hop_candidate_scoring_preserves_bridge_question_semantics():
    rag = GraphRAG(strategy="prehop")
    candidate = {
        "id": "target",
        "title": "2023 manual",
        "text": "The target body.",
        "embedding": [0.6, 0.8],
        "bridge_embeddings": [[0.8, 0.6], [0.2, 0.98]],
    }

    selected, _ = await rag._score_and_select([1.0, 0.0], [candidate], top_k=1)

    assert selected[0]["similarity_score"] == 0.6
    assert selected[0]["bridge_similarity_score"] == 0.8
    assert selected[0]["final_score"] == pytest.approx(0.6)


def test_final_selection_round_robins_sources_without_fraction_parameter():
    rag = GraphRAG(strategy="prehop")
    ordered = [
        {"id": "a1", "source": "a"},
        {"id": "a2", "source": "a"},
        {"id": "b1", "source": "b"},
        {"id": "a3", "source": "a"},
        {"id": "b2", "source": "b"},
    ]

    selected = rag._source_round_robin(ordered, top_k=4)

    assert [node["id"] for node in selected] == ["a1", "b1", "a2", "b2"]
    assert not hasattr(RAGConfig, "MAX_CHUNKS_PER_SOURCE_FRACTION")


@pytest.mark.asyncio
async def test_global_source_selection_is_an_explicit_ablation(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    candidates = [
        {"id": "a1", "source": "a", "embedding": [1.0, 0.0]},
        {"id": "a2", "source": "a", "embedding": [0.9, 0.1]},
        {"id": "b1", "source": "b", "embedding": [0.8, 0.2]},
    ]

    selected, _ = await rag._score_and_select([1.0, 0.0], candidates, top_k=2)

    assert [node["id"] for node in selected] == ["a1", "a2"]


def test_transient_index_embeddings_are_not_returned_publicly():
    rag = GraphRAG(strategy="prehop")
    output = rag._without_transient_retrieval_scores(
        {"id": "chunk", "text": "evidence", "embedding": [1.0], "bridge_embeddings": [[1.0]]}
    )

    assert output == {"id": "chunk", "text": "evidence"}


@pytest.mark.asyncio
async def test_retrieve_searches_each_role_channel_once_in_full_mode():
    rag = GraphRAG(strategy="prehop")

    stage1_node = {
        "id": "n1",
        "title": "AMD_2022_10K",
        "sent_id": 1,
        "page": 10,
        "text": "Operating cash flow was 3.6 billion.",
        "rrf_score": 1.0,
        "embedding": [1.0, 0.0],
    }
    stage2_node = {
        "id": "n2",
        "title": "AMD_2022_10K",
        "sent_id": 2,
        "page": 11,
        "text": "Capital expenditures were 1.1 billion in FY2022.",
        "rrf_score": 0.9,
        "embedding": [1.0, 0.0],
    }

    async def fake_candidates(_q_text: str, query_embedding, limit: int, channel: str = "body"):
        assert query_embedding == [1.0, 0.0]
        if channel == "q_minus":
            return [dict(stage1_node)]
        if channel == "q_plus":
            return [dict(stage2_node)]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])

    _, nodes = await rag.retrieve(
        "What was AMD FY2022 free cash flow?",
        top_k=2,
    )

    ids = {n.get("id") for n in nodes}
    assert "n1" in ids
    assert "n2" in ids


@pytest.mark.asyncio
async def test_retrieval_embeds_query_once_before_parallel_channels():
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._retrieve_with_candidate_pool("one query", top_k=12)

    rag.llm.get_embedding.assert_awaited_once_with("one query")
    assert rag._hybrid_rrf_candidates.await_count == 3
    assert [call.kwargs["channel"] for call in rag._hybrid_rrf_candidates.await_args_list] == [
        "q_minus",
        "body",
        "q_plus",
    ]
    assert all(call.kwargs["query_embedding"] == [1.0, 0.0] for call in rag._hybrid_rrf_candidates.await_args_list)


@pytest.mark.asyncio
async def test_retrieval_records_every_direct_representation_path():
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])

    async def candidates(_query, *, query_embedding, limit, channel):
        _ = query_embedding, limit
        if channel in {"q_minus", "q_plus"}:
            return [{"id": "shared", "text": "evidence", "embedding": [1.0]}]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=candidates)  # type: ignore[method-assign]

    _, pool = await rag._retrieve_with_candidate_pool("query", top_k=1)

    assert pool[0]["retrieval_paths"] == [
        {"kind": "direct", "channel": "q_minus", "query_view": "query", "depth": 0},
        {"kind": "direct", "channel": "q_plus", "query_view": "query", "depth": 0},
    ]
    assert pool[0]["representation_scores"] == {"q_minus": 1.0, "q_plus": 1.0}
    assert pool[0]["representation_score"] == 2.0


@pytest.mark.asyncio
async def test_hop_target_inherits_only_the_qplus_seed_rank():
    rag = GraphRAG(strategy="prehop")
    seed = {
        "id": "seed",
        "title": "Seed",
        "sent_id": 0,
        "text": "dependency",
        "embedding": [1.0],
        "dependency_seed": True,
        "representation_score": 1.5,
        "representation_scores": {"q_minus": 1.0, "q_plus": 0.5},
    }
    rag._retrieve_with_candidate_pool = AsyncMock(return_value=([seed], [seed]))  # type: ignore[method-assign]
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._expand_frontier = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {
                "source_id": "seed",
                "id": "target",
                "title": "Target",
                "sent_id": 0,
                "text": "answer",
                "embedding": [1.0],
                "path_type": "hop",
                "bridge_embeddings": [[1.0]],
            }
        ]
    )
    captured: list[dict] = []

    async def capture_scores(_query_embedding, candidates, _top_k):
        captured.extend(candidates)
        return candidates, candidates

    rag._score_and_select = AsyncMock(side_effect=capture_scores)  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=2)

    target = next(candidate for candidate in captured if candidate["id"] == "target")
    assert target["representation_score"] == 0.25


@pytest.mark.asyncio
async def test_role_aligned_views_search_only_their_matching_channels(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "single_combined")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0], [0.9], [0.8], [0.7]])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._retrieve_with_candidate_pool(
        "original",
        top_k=12,
        channel_queries={"q_minus": ["minus one", "minus two"], "q_plus": ["plus one"]},
    )

    assert rag.llm.get_embeddings.await_args.args[0] == ["original", "minus one", "minus two", "plus one"]
    assert [
        (call.args[0], call.kwargs["channel"])
        for call in rag._hybrid_rrf_candidates.await_args_list
    ] == [("minus one", "q_minus"), ("minus two", "q_minus"), ("plus one", "q_plus")]


@pytest.mark.asyncio
async def test_hybrid_search_derives_modality_width_from_owner_budget():
    rag = GraphRAG(strategy="prehop")
    rag._run_channel_query = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._hybrid_rrf_candidates("query", [1.0], limit=7, channel="q_minus")
    qminus_limits = [call.args[1]["limit"] for call in rag._run_channel_query.await_args_list]
    assert qminus_limits == [21]
    assert "CALL () {" in rag._run_channel_query.await_args.args[0]

    rag._run_channel_query.reset_mock()
    await rag._hybrid_rrf_candidates("query", [1.0], limit=7, channel="body")
    body_limits = [call.args[1]["limit"] for call in rag._run_channel_query.await_args_list]
    assert body_limits == [7]
    assert "CALL () {" in rag._run_channel_query.await_args.args[0]


@pytest.mark.asyncio
async def test_only_query_matched_dependency_seeds_can_initiate_hop_traversal():
    rag = GraphRAG(strategy="prehop")
    direct = {
        "id": "direct",
        "title": "Direct",
        "sent_id": 0,
        "text": "direct",
        "dependency_seed": False,
    }
    dependency = {
        "id": "dependency",
        "title": "Dependency",
        "sent_id": 0,
        "text": "dependency",
        "dependency_seed": True,
    }
    rag._retrieve_with_candidate_pool = AsyncMock(  # type: ignore[method-assign]
        return_value=([direct], [direct, dependency])
    )
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._expand_frontier = AsyncMock(return_value=[])  # type: ignore[method-assign]
    rag._score_and_select = AsyncMock(return_value=([direct], [direct]))  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=1)

    rag._expand_frontier.assert_awaited_once()
    assert set(rag._expand_frontier.await_args.args[0]) == {"direct", "dependency"}
    assert rag._expand_frontier.await_args.args[2] == {"dependency"}


@pytest.mark.asyncio
async def test_build_graph_stores_generated_q_plus_without_heuristic_filter(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "off")
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
async def test_qplus_similarity_is_not_an_indexing_candidate_channel():
    rag = GraphRAG(strategy="prehop")

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
    assert [item["ann_pool"] for item in parameters["source_questions"]] == [22, 180]
    assert "candidate_limit" not in parameters


@pytest.mark.asyncio
async def test_full_hop_policy_follows_qminus_owner_without_second_body_search():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        source = {
            "source_chunk_id": wave[0]["id"],
            "source_question_id": wave[0]["questions"][0]["id"],
        }
        if channel == "q_minus":
            return [{**source, "target_id": "manual-2023", "target_question_id": "qm-2023", "score": 0.8}]
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

    assert [edge["tgt_id"] for edge in edges] == ["manual-2023"]
    assert [call.args[1] for call in rag._find_hop_candidates_batch.await_args_list] == ["q_minus"]


@pytest.mark.asyncio
async def test_each_individual_qplus_keeps_one_direct_evidence_target():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        candidates = [
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-first",
                "target_id": "first-best",
                "target_question_id": "qm-first" if channel == "q_minus" else None,
                "target_text": "First target answers the first question.",
                "target_title": "First",
                "score": 0.99,
            },
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-second",
                "target_id": "second-best",
                "target_question_id": "qm-second" if channel == "q_minus" else None,
                "target_text": "Second target answers the second question.",
                "target_title": "Second",
                "score": 0.70,
            },
        ]
        return candidates

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
    assert all(edge["direct_channels"] == ["q_minus"] for edge in edges)


@pytest.mark.asyncio
async def test_questions_sharing_one_target_keep_all_provenance():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        return [
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": question_id,
                "target_id": "shared-target",
                "target_question_id": f"qm-{question_id}" if channel == "q_minus" else None,
                "target_text": "One passage answers both questions.",
                "target_title": "Shared",
                "score": 0.9,
            }
            for question_id in ("q-first", "q-second")
        ]

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]

    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "source.txt",
                "questions": [
                    {"id": "q-first", "text": "First question?", "query_embedding": [1.0]},
                    {"id": "q-second", "text": "Second question?", "query_embedding": [1.0]},
                ],
            }
        ]
    )

    assert len(edges) == 1
    assert edges[0]["source_question_ids"] == ["q-first", "q-second"]


@pytest.mark.asyncio
async def test_hop_and_provenance_edges_are_written_in_one_query():
    rag = GraphRAG(strategy="prehop")
    rag.retry_query = AsyncMock(return_value=[])
    edge = {
        "src_id": "source",
        "tgt_id": "target",
        "direct_channels": ["q_minus"],
        "q_minus_match": {"raw_score": 0.8},
        "body_match": None,
        "q_minus_matches": [
            {"source_question_id": "q-plus", "target_question_id": "q-minus", "raw_score": 0.8}
        ],
        "body_matches": [],
        "source_question_ids": ["q-plus"],
        "source_question_texts": ["Which evidence is missing?"],
        "construction_mode": "qplus_to_qminus_owner",
    }

    await rag._flush_hop_edges([edge])

    rag.retry_query.assert_awaited_once()
    assert rag.retry_query.await_args.args[1] == {"edges": [edge]}


def test_indexing_has_no_tuned_hop_fusion_or_width_knobs():
    for name in (
        "RRF_K_CONSTANT",
        "HOP_LINK_LIMIT",
        "HOP_CANDIDATE_LIMIT",
        "HOP_ANN_POOL",
        "HOP_SAME_NEED_WEIGHT",
    ):
        assert not hasattr(RAGConfig, name)


def test_benchmark_does_not_reference_removed_hop_tuning_knobs():
    benchmark_source = Path("cli/benchmark.py").read_text(encoding="utf-8")
    for name in (
        "HOP_LINK_LIMIT",
        "HOP_CANDIDATE_LIMIT",
        "HOP_ANN_POOL",
        "HOP_SAME_NEED_WEIGHT",
    ):
        assert name not in benchmark_source
